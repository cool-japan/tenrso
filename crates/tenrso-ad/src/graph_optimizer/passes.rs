//! In-place graph rewrite passes: constant folding, CSE, dead code elimination.
//!
//! These passes mutate a [`ComputationGraph`] directly. They cannot fuse
//! operations — see [`crate::graph_optimizer::fusion`] for that.

use super::topological_order_from_snapshot;
use crate::graph::{ComputationGraph, NodeId, Operation};
use scirs2_core::ndarray_ext::ScalarOperand;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, HashSet};

/// Default maximum element count for a node eligible for constant folding.
///
/// Nodes whose cached value has more than this many elements are left alone
/// so the compiler does not materialize huge constant tensors that might be
/// dead weight after further optimization. The threshold is configurable
/// per call via [`constant_folding_with_threshold`].
pub const DEFAULT_CONST_FOLD_ELEMENT_LIMIT: usize = 1024;

// ============================================================================
// Constant folding
// ============================================================================

/// Statistics produced by the constant-folding pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstFoldStats {
    /// Number of nodes reclassified as constants (i.e. folded).
    pub nodes_folded: usize,
    /// Number of nodes skipped because their output would exceed the element
    /// threshold.
    pub skipped_too_large: usize,
    /// Number of nodes skipped because their op is non-deterministic or has
    /// side effects.
    pub skipped_non_deterministic: usize,
}

/// Fold nodes whose inputs are all compile-time constants into constants.
///
/// A node is considered a constant iff its op is [`Operation::Input`] *and*
/// its `requires_grad` flag is `false` — i.e. produced via
/// [`ComputationGraph::constant`].
///
/// The graph evaluates operations eagerly during construction, so each
/// candidate node already has a cached value. Folding therefore consists of
/// rewriting the node's operation to `Operation::Input`, clearing its
/// `requires_grad` flag, and detaching it from its former parents; the
/// cached value is preserved.
///
/// # Size threshold
///
/// Nodes whose output has more than [`DEFAULT_CONST_FOLD_ELEMENT_LIMIT`]
/// elements are skipped to avoid bloating the graph with large compile-time
/// constants. Use [`constant_folding_with_threshold`] to supply a custom
/// threshold.
///
/// # Supported ops
///
/// Deterministic, side-effect-free ops are folded: `Add`, `Sub`, `Mul`,
/// `Div`, `MatMul`, `Neg`, `Exp`, `Log`, `Pow`, `Sum`, `Mean`, `Reshape`,
/// `Transpose`, `Broadcast`, `ReLU`, `Sigmoid`, `Tanh`, `Slice`. All ops
/// currently in [`Operation`] are deterministic, but the pass is explicit
/// about this so future additions (e.g. `Dropout`, `RandomNormal`) do not
/// silently get folded.
pub fn constant_folding<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
) -> ConstFoldStats {
    constant_folding_with_threshold(graph, DEFAULT_CONST_FOLD_ELEMENT_LIMIT)
}

/// Variant of [`constant_folding`] with a configurable element-count
/// threshold.
///
/// Setting `max_elements = usize::MAX` disables the size guard.
pub fn constant_folding_with_threshold<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
    max_elements: usize,
) -> ConstFoldStats {
    let mut stats = ConstFoldStats::default();
    let snapshot = graph.snapshot_ops();
    let order = topological_order_from_snapshot(&snapshot);

    for id in order {
        let op = match graph.node_operation(id) {
            Some(o) => o,
            None => continue,
        };
        // Already a leaf / constant — nothing to do.
        if matches!(op, Operation::Input) {
            continue;
        }
        if !is_foldable_op(&op) {
            stats.skipped_non_deterministic += 1;
            continue;
        }
        let parents = match graph.node_parents(id) {
            Some(p) => p,
            None => continue,
        };
        if parents.is_empty() {
            continue;
        }
        // Every parent must be a constant (or already folded to one during
        // this pass, since we iterate in topological order).
        if !parents.iter().all(|&p| graph.is_constant(p)) {
            continue;
        }
        // Do not fold nodes that the user still wants gradients for.
        if graph.node_requires_grad(id) {
            continue;
        }
        let size = graph.node_element_count(id).unwrap_or(usize::MAX);
        if size > max_elements {
            stats.skipped_too_large += 1;
            continue;
        }
        if graph.reclassify_as_constant(id).is_ok() {
            stats.nodes_folded += 1;
        }
    }

    stats
}

/// Return `true` for side-effect-free, deterministic ops that are safe to
/// evaluate at graph-construction time.
fn is_foldable_op(op: &Operation) -> bool {
    match op {
        Operation::Input => false,
        Operation::Add { .. }
        | Operation::Sub { .. }
        | Operation::Mul { .. }
        | Operation::Div { .. }
        | Operation::MatMul { .. }
        | Operation::Neg { .. }
        | Operation::Exp { .. }
        | Operation::Log { .. }
        | Operation::Pow { .. }
        | Operation::Sum { .. }
        | Operation::Mean { .. }
        | Operation::Reshape { .. }
        | Operation::Transpose { .. }
        | Operation::Broadcast { .. }
        | Operation::ReLU { .. }
        | Operation::Sigmoid { .. }
        | Operation::Tanh { .. }
        | Operation::Slice { .. } => true,
    }
}

// ============================================================================
// Common subexpression elimination (CSE)
// ============================================================================

/// Statistics produced by the CSE pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CseStats {
    /// Number of nodes removed as duplicates.
    pub nodes_eliminated: usize,
}

/// Canonical key used to identify structurally-identical operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CanonicalKey {
    tag: u8,
    /// Sorted for commutative ops, ordered otherwise.
    inputs: Vec<usize>,
    /// Op-specific extra parameters encoded to strings (so `Hash + Eq` holds
    /// without touching float identity for `Pow`).
    params: Vec<String>,
}

/// Identify two nodes as duplicates iff their canonical keys match, then
/// redirect every consumer of the duplicate onto the canonical survivor.
/// Follow-up dead-code elimination is performed inline so that orphaned
/// nodes are removed and the stats are accurate.
///
/// # Commutativity
///
/// `Add` and `Mul` are commutative: `Add(x, y)` and `Add(y, x)` are
/// deduplicated. All other binary ops (`Sub`, `Div`, `MatMul`) are treated
/// as order-sensitive.
///
/// # Skipped ops
///
/// `Operation::Input` is never deduplicated: two input nodes may share the
/// same cached value but represent two distinct user-owned variables whose
/// identity must be preserved for gradient reporting. This also keeps the
/// pass safe with respect to side effects should any future op gain them.
pub fn common_subexpression_elimination<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
) -> CseStats {
    let mut stats = CseStats::default();
    let snapshot = graph.snapshot_ops();
    let order = topological_order_from_snapshot(&snapshot);

    let mut table: HashMap<CanonicalKey, NodeId> = HashMap::new();
    let mut to_remove: HashSet<NodeId> = HashSet::new();

    for id in order {
        // Read the op from the *live* graph so that any previous
        // `redirect_consumers` calls in this same pass are reflected in the
        // canonical key — that's what makes chained deduplication work in a
        // single traversal.
        let op = match graph.node_operation(id) {
            Some(o) => o,
            None => continue,
        };
        // Inputs are never deduplicated. Two input nodes may carry equal
        // cached values but represent distinct user variables whose identity
        // must be preserved for gradient reporting.
        if matches!(op, Operation::Input) {
            continue;
        }
        let key = canonical_key(&op);
        if let Some(&canonical) = table.get(&key) {
            if canonical != id && graph.redirect_consumers(id, canonical).is_ok() {
                to_remove.insert(id);
                stats.nodes_eliminated += 1;
            }
        } else {
            table.insert(key, id);
        }
    }

    if !to_remove.is_empty() {
        graph.remove_nodes(&to_remove);
    }

    stats
}

/// Build a canonical key for a non-Input operation.
fn canonical_key(op: &Operation) -> CanonicalKey {
    let r = |id: NodeId| -> usize { id.0 };
    match op {
        Operation::Input => CanonicalKey {
            tag: 0,
            inputs: vec![],
            params: vec![],
        },
        Operation::Add { lhs, rhs } => {
            let mut inputs = vec![r(*lhs), r(*rhs)];
            inputs.sort_unstable();
            CanonicalKey {
                tag: 1,
                inputs,
                params: vec![],
            }
        }
        Operation::Mul { lhs, rhs } => {
            let mut inputs = vec![r(*lhs), r(*rhs)];
            inputs.sort_unstable();
            CanonicalKey {
                tag: 2,
                inputs,
                params: vec![],
            }
        }
        Operation::Sub { lhs, rhs } => CanonicalKey {
            tag: 3,
            inputs: vec![r(*lhs), r(*rhs)],
            params: vec![],
        },
        Operation::Div { lhs, rhs } => CanonicalKey {
            tag: 4,
            inputs: vec![r(*lhs), r(*rhs)],
            params: vec![],
        },
        Operation::MatMul { lhs, rhs } => CanonicalKey {
            tag: 5,
            inputs: vec![r(*lhs), r(*rhs)],
            params: vec![],
        },
        Operation::Neg { input } => CanonicalKey {
            tag: 6,
            inputs: vec![r(*input)],
            params: vec![],
        },
        Operation::Exp { input } => CanonicalKey {
            tag: 7,
            inputs: vec![r(*input)],
            params: vec![],
        },
        Operation::Log { input } => CanonicalKey {
            tag: 8,
            inputs: vec![r(*input)],
            params: vec![],
        },
        Operation::Pow { input, exponent } => CanonicalKey {
            tag: 9,
            inputs: vec![r(*input)],
            params: vec![format!("{:?}", exponent.to_bits())],
        },
        Operation::Sum { input, axis } => CanonicalKey {
            tag: 10,
            inputs: vec![r(*input)],
            params: vec![format!("{:?}", axis)],
        },
        Operation::Mean { input, axis } => CanonicalKey {
            tag: 11,
            inputs: vec![r(*input)],
            params: vec![format!("{:?}", axis)],
        },
        Operation::Reshape { input, old_shape } => CanonicalKey {
            tag: 12,
            inputs: vec![r(*input)],
            params: vec![format!("{:?}", old_shape)],
        },
        Operation::Transpose { input, axes } => CanonicalKey {
            tag: 13,
            inputs: vec![r(*input)],
            params: vec![format!("{:?}", axes)],
        },
        Operation::Broadcast {
            input,
            original_shape,
        } => CanonicalKey {
            tag: 14,
            inputs: vec![r(*input)],
            params: vec![format!("{:?}", original_shape)],
        },
        Operation::ReLU { input } => CanonicalKey {
            tag: 15,
            inputs: vec![r(*input)],
            params: vec![],
        },
        Operation::Sigmoid { input } => CanonicalKey {
            tag: 16,
            inputs: vec![r(*input)],
            params: vec![],
        },
        Operation::Tanh { input } => CanonicalKey {
            tag: 17,
            inputs: vec![r(*input)],
            params: vec![],
        },
        Operation::Slice { input, ranges } => CanonicalKey {
            tag: 18,
            inputs: vec![r(*input)],
            params: vec![format!("{:?}", ranges)],
        },
    }
}

// ============================================================================
// Dead code elimination (graph-mutating)
// ============================================================================

/// Statistics produced by the DCE pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DceStats {
    /// Number of nodes removed.
    pub nodes_removed: usize,
}

/// Remove nodes unreachable from `outputs` through the `parents` edges.
///
/// Used by the combined optimization pipeline after constant folding and
/// CSE to collect orphaned intermediates.
pub fn dead_code_elimination_on_graph<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
    outputs: &[NodeId],
) -> DceStats {
    if outputs.is_empty() {
        return DceStats::default();
    }
    let live = graph.reachable_from(outputs);
    let all_ids = graph.all_node_ids();
    let dead: HashSet<NodeId> = all_ids
        .into_iter()
        .filter(|id| !live.contains(id))
        .collect();
    let removed = graph.remove_nodes(&dead);
    DceStats {
        nodes_removed: removed,
    }
}

// ============================================================================
// Combined pipeline
// ============================================================================

/// Statistics from the combined [`optimize_graph`] pipeline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineStats {
    /// Nodes in the graph before optimization.
    pub nodes_before: usize,
    /// Nodes in the graph after optimization.
    pub nodes_after: usize,
    /// Stats from the first CSE pass (pre-fold structural dedup).
    pub cse_pre: CseStats,
    /// Stats from the constant-folding pass.
    pub const_fold: ConstFoldStats,
    /// Stats from the second CSE pass (post-fold; dedups the newly minted
    /// constants against previously-distinct-but-equal expressions).
    pub cse_post: CseStats,
    /// Stats from the final DCE pass.
    pub dce: DceStats,
}

/// Run the graph-mutating optimization pipeline in the canonical order:
///
/// 1. `common_subexpression_elimination` (pre-fold) — deduplicates
///    structurally-identical ops, crucially collapsing commutatively-equal
///    expressions (e.g. `a + b` vs `b + a`) *before* folding opacifies them.
/// 2. `constant_folding` — reclassifies nodes whose inputs are all constants
///    as constants themselves (preserving the already-cached value), so long
///    as the result fits within the element-size threshold.
/// 3. `common_subexpression_elimination` (post-fold) — deduplicates any
///    expressions that only became structurally identical after folding
///    (e.g. two distinct `Mul` nodes whose operands both folded to the
///    same numeric constant path).
/// 4. `dead_code_elimination_on_graph` — collects any orphans produced by
///    the previous passes given the supplied set of outputs.
///
/// This pipeline performs no operation fusion; see
/// [`compile_fused_plan`](crate::graph_optimizer::compile_fused_plan).
pub fn optimize_graph<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
    outputs: &[NodeId],
) -> PipelineStats {
    let nodes_before = graph.num_nodes();
    let cse_pre = common_subexpression_elimination(graph);
    let const_fold = constant_folding(graph);
    let cse_post = common_subexpression_elimination(graph);
    let dce = dead_code_elimination_on_graph(graph, outputs);
    let nodes_after = graph.num_nodes();
    PipelineStats {
        nodes_before,
        nodes_after,
        cse_pre,
        const_fold,
        cse_post,
        dce,
    }
}
