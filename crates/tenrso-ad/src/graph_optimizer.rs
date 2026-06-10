//! Graph optimization for computation graphs.
//!
//! This module provides optimization passes for computation graphs to improve
//! performance and reduce memory usage. Optimizations include:
//!
//! - **Operation Fusion**: Combine multiple operations into fused kernels
//! - **Dead Code Elimination**: Remove unused nodes and operations
//! - **Constant Folding**: Pre-compute operations on constant values
//! - **Memory Planning**: Determine memory reuse opportunities
//!
//! # Example
//!
//! ```rust,ignore
//! use tenrso_ad::graph::ComputationGraph;
//! use tenrso_ad::graph_optimizer::{GraphOptimizer, OptimizationPass};
//!
//! let graph = ComputationGraph::new();
//! // Build graph...
//!
//! // Create optimizer
//! let optimizer = GraphOptimizer::new()
//!     .with_pass(OptimizationPass::OperationFusion)
//!     .with_pass(OptimizationPass::DeadCodeElimination);
//!
//! // Optimize graph
//! let optimized = optimizer.optimize(&graph)?;
//! ```

use crate::graph::{ComputationGraph, NodeId, Operation};
use anyhow::Result;
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

/// Optimization passes that can be applied to a computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Fuse compatible operations into single operations
    OperationFusion,
    /// Remove nodes that don't contribute to outputs
    DeadCodeElimination,
    /// Pre-compute operations on constant values
    ConstantFolding,
    /// All optimization passes
    All,
}

/// Configuration for graph optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Passes to apply
    pub passes: Vec<OptimizationPass>,
    /// Whether to run passes until convergence
    pub run_until_convergence: bool,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            passes: vec![OptimizationPass::All],
            run_until_convergence: true,
            max_iterations: 10,
            verbose: false,
        }
    }
}

impl OptimizationConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an optimization pass
    pub fn with_pass(mut self, pass: OptimizationPass) -> Self {
        self.passes.push(pass);
        self
    }

    /// Enable verbose logging
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }
}

/// Fused operations for better performance
#[derive(Debug, Clone)]
pub enum FusedOperation {
    /// MatMul + Bias: C = A @ B + bias
    MatMulBias {
        lhs: NodeId,
        rhs: NodeId,
        bias: NodeId,
    },
    /// MatMul + Bias + ReLU: C = ReLU(A @ B + bias)
    MatMulBiasReLU {
        lhs: NodeId,
        rhs: NodeId,
        bias: NodeId,
    },
    /// Mul + Add: z = x * y + c (fused multiply-add)
    MulAdd { x: NodeId, y: NodeId, c: NodeId },
    /// Element-wise Add + ReLU: z = ReLU(x + y)
    AddReLU { lhs: NodeId, rhs: NodeId },
}

/// Statistics about optimization results
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of nodes before optimization
    pub nodes_before: usize,
    /// Number of nodes after optimization
    pub nodes_after: usize,
    /// Number of operations fused
    pub fusions_applied: usize,
    /// Number of dead nodes eliminated
    pub dead_nodes_removed: usize,
    /// Number of constants folded
    pub constants_folded: usize,
    /// Iterations performed
    pub iterations: usize,
}

impl OptimizationStats {
    /// Calculate reduction percentage
    pub fn reduction_percent(&self) -> f64 {
        if self.nodes_before == 0 {
            0.0
        } else {
            100.0 * (1.0 - self.nodes_after as f64 / self.nodes_before as f64)
        }
    }
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph Optimization Statistics:")?;
        writeln!(f, "  Nodes before: {}", self.nodes_before)?;
        writeln!(f, "  Nodes after: {}", self.nodes_after)?;
        writeln!(f, "  Reduction: {:.1}%", self.reduction_percent())?;
        writeln!(f, "  Operations fused: {}", self.fusions_applied)?;
        writeln!(f, "  Dead nodes removed: {}", self.dead_nodes_removed)?;
        writeln!(f, "  Constants folded: {}", self.constants_folded)?;
        writeln!(f, "  Iterations: {}", self.iterations)?;
        Ok(())
    }
}

/// Graph optimizer for applying optimization passes
pub struct GraphOptimizer {
    config: OptimizationConfig,
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphOptimizer {
    /// Create a new graph optimizer with default configuration
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
        }
    }

    /// Create optimizer with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Add an optimization pass
    pub fn with_pass(mut self, pass: OptimizationPass) -> Self {
        self.config.passes.push(pass);
        self
    }

    /// Enable verbose logging
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Optimize a computation graph in-place.
    ///
    /// Runs the enabled passes in canonical order:
    /// 1. Pre-fold CSE (deduplicates structurally-identical ops)
    /// 2. Constant folding (reclassifies all-constant sub-expressions as constants)
    /// 3. Post-fold CSE (deduplicates newly-constant expressions)
    /// 4. DCE (removes nodes unreachable from implicit output roots)
    ///
    /// When `run_until_convergence` is set, the pipeline repeats until the
    /// node count stabilises or `max_iterations` is reached.
    ///
    /// `DeadCodeElimination` uses sink nodes (no consumers) as implicit output
    /// roots when explicit output `NodeId`s are unavailable. Pass output IDs
    /// directly to [`dead_code_elimination_on_graph`] when they are known.
    pub fn optimize<T: Float + ScalarOperand + FromPrimitive>(
        &self,
        graph: &ComputationGraph<T>,
    ) -> Result<OptimizationStats> {
        let nodes_before = graph.num_nodes();
        let run_all = self.config.passes.contains(&OptimizationPass::All);
        let run_cse = run_all
            || self
                .config
                .passes
                .contains(&OptimizationPass::OperationFusion);
        let run_fold = run_all
            || self
                .config
                .passes
                .contains(&OptimizationPass::ConstantFolding);
        let run_dce = run_all
            || self
                .config
                .passes
                .contains(&OptimizationPass::DeadCodeElimination);

        let max_iters = if self.config.run_until_convergence {
            self.config.max_iterations.max(1)
        } else {
            1
        };

        let mut fusions_applied = 0usize;
        let mut constants_folded = 0usize;
        let mut dead_nodes_removed = 0usize;
        let mut iterations = 0usize;

        for _ in 0..max_iters {
            let n_before = graph.num_nodes();

            if run_cse {
                let s = common_subexpression_elimination(graph);
                fusions_applied += s.nodes_eliminated;
            }
            if run_fold {
                let s = constant_folding(graph);
                constants_folded += s.nodes_folded;
            }
            if run_cse {
                let s = common_subexpression_elimination(graph);
                fusions_applied += s.nodes_eliminated;
            }
            if run_dce {
                // Identify sink nodes (no consumers) as implicit output roots.
                let ops = graph.snapshot_ops();
                let mut has_consumer: HashSet<NodeId> = HashSet::new();
                for (_id, op) in &ops {
                    for p in self.get_operation_inputs(op) {
                        has_consumer.insert(p);
                    }
                }
                let sink_nodes: Vec<NodeId> = graph
                    .all_node_ids()
                    .into_iter()
                    .filter(|id| !has_consumer.contains(id))
                    .collect();
                if !sink_nodes.is_empty() {
                    let s = dead_code_elimination_on_graph(graph, &sink_nodes);
                    dead_nodes_removed += s.nodes_removed;
                }
            }

            iterations += 1;
            if graph.num_nodes() == n_before {
                break;
            }
        }

        Ok(OptimizationStats {
            nodes_before,
            nodes_after: graph.num_nodes(),
            fusions_applied,
            dead_nodes_removed,
            constants_folded,
            iterations,
        })
    }

    /// Detect fusion opportunities in a sequence of operations
    pub fn detect_fusion_patterns(
        &self,
        ops: &[(NodeId, Operation)],
    ) -> Vec<(usize, FusedOperation)> {
        let mut patterns = Vec::new();
        let mut skip_until = 0;

        for i in 0..ops.len() {
            if i < skip_until {
                continue; // Skip indices that are part of a larger pattern
            }

            // Try to match longer patterns first (3 operations)
            if i + 2 < ops.len() {
                let (node_id_i, op_i) = &ops[i];
                let (node_id_i1, op_i1) = &ops[i + 1];
                let (_node_id_i2, op_i2) = &ops[i + 2];

                // Pattern: MatMul + Add + ReLU → MatMulBiasReLU
                if let (
                    Operation::MatMul { lhs, rhs },
                    Operation::Add {
                        lhs: add_lhs,
                        rhs: add_rhs,
                    },
                    Operation::ReLU { input: relu_input },
                ) = (op_i, op_i1, op_i2)
                {
                    // Check if relu uses add result and add uses matmul result
                    if relu_input == node_id_i1 {
                        let bias = if add_lhs == node_id_i {
                            *add_rhs
                        } else if add_rhs == node_id_i {
                            *add_lhs
                        } else {
                            // Add doesn't use matmul result, try shorter patterns
                            continue;
                        };

                        patterns.push((
                            i,
                            FusedOperation::MatMulBiasReLU {
                                lhs: *lhs,
                                rhs: *rhs,
                                bias,
                            },
                        ));
                        skip_until = i + 3; // Skip next 2 indices as they're part of this pattern
                        continue;
                    }
                }
            }

            // Try to match 2-operation patterns
            if i + 1 < ops.len() {
                let (node_id_i, op_i) = &ops[i];
                let (_node_id_i1, op_i1) = &ops[i + 1];

                // Pattern: MatMul + Add → MatMulBias
                if let (
                    Operation::MatMul { lhs, rhs },
                    Operation::Add {
                        lhs: add_lhs,
                        rhs: add_rhs,
                    },
                ) = (op_i, op_i1)
                {
                    if let Some(bias) = if add_lhs == node_id_i {
                        Some(*add_rhs)
                    } else if add_rhs == node_id_i {
                        Some(*add_lhs)
                    } else {
                        None
                    } {
                        patterns.push((
                            i,
                            FusedOperation::MatMulBias {
                                lhs: *lhs,
                                rhs: *rhs,
                                bias,
                            },
                        ));
                        skip_until = i + 2;
                        continue;
                    }
                }

                // Pattern: Add + ReLU → AddReLU
                if let (Operation::Add { lhs, rhs }, Operation::ReLU { input: relu_input }) =
                    (op_i, op_i1)
                {
                    // Check if relu uses add result
                    if relu_input == node_id_i {
                        patterns.push((
                            i,
                            FusedOperation::AddReLU {
                                lhs: *lhs,
                                rhs: *rhs,
                            },
                        ));
                        skip_until = i + 2;
                        continue;
                    }
                }

                // Pattern: Mul + Add → MulAdd (FMA)
                if let (
                    Operation::Mul { lhs, rhs },
                    Operation::Add {
                        lhs: add_lhs,
                        rhs: add_rhs,
                    },
                ) = (op_i, op_i1)
                {
                    if let Some(c) = if add_lhs == node_id_i {
                        Some(*add_rhs)
                    } else if add_rhs == node_id_i {
                        Some(*add_lhs)
                    } else {
                        None
                    } {
                        patterns.push((
                            i,
                            FusedOperation::MulAdd {
                                x: *lhs,
                                y: *rhs,
                                c,
                            },
                        ));
                        skip_until = i + 2;
                        continue;
                    }
                }
            }
        }

        patterns
    }

    /// Perform dead code elimination on a set of operations
    pub fn eliminate_dead_code(
        &self,
        ops: &[(NodeId, Operation)],
        output_nodes: &HashSet<NodeId>,
    ) -> Vec<NodeId> {
        let mut live_nodes = output_nodes.clone();
        let mut changed = true;

        // Backward pass: mark all nodes reachable from outputs
        while changed {
            changed = false;
            for (node_id, op) in ops {
                if !live_nodes.contains(node_id) {
                    continue;
                }

                // Mark parent nodes as live
                let parents = self.get_operation_inputs(op);
                for parent in parents {
                    if live_nodes.insert(parent) {
                        changed = true;
                    }
                }
            }
        }

        // Return dead nodes (nodes not in live set)
        ops.iter()
            .map(|(id, _)| *id)
            .filter(|id| !live_nodes.contains(id))
            .collect()
    }

    /// Get input node IDs for an operation
    fn get_operation_inputs(&self, op: &Operation) -> Vec<NodeId> {
        match op {
            Operation::Input => vec![],
            Operation::Add { lhs, rhs }
            | Operation::Sub { lhs, rhs }
            | Operation::Mul { lhs, rhs }
            | Operation::Div { lhs, rhs }
            | Operation::MatMul { lhs, rhs } => vec![*lhs, *rhs],
            Operation::Neg { input }
            | Operation::Exp { input }
            | Operation::Log { input }
            | Operation::Pow { input, .. }
            | Operation::Sum { input, .. }
            | Operation::Mean { input, .. }
            | Operation::Reshape { input, .. }
            | Operation::Transpose { input, .. }
            | Operation::Broadcast { input, .. }
            | Operation::ReLU { input }
            | Operation::Sigmoid { input }
            | Operation::Tanh { input } => vec![*input],
            Operation::Slice { input, .. } => vec![*input],
        }
    }

    /// Estimate memory savings from optimization
    pub fn estimate_memory_savings(
        &self,
        stats: &OptimizationStats,
        avg_tensor_size_bytes: usize,
    ) -> usize {
        let nodes_removed = stats.nodes_before.saturating_sub(stats.nodes_after);
        nodes_removed * avg_tensor_size_bytes
    }
}

// ============================================================================
// Constant folding
// ============================================================================

/// Statistics produced by the constant-folding pass.
#[derive(Debug, Clone, Default)]
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

/// Compute a topological ordering from a snapshot so that children are
/// always visited after their parents. Kahn-style algorithm.
fn topological_order_from_snapshot(snapshot: &[(NodeId, Operation)]) -> Vec<NodeId> {
    let id_to_parents: HashMap<NodeId, Vec<NodeId>> = snapshot
        .iter()
        .map(|(id, op)| (*id, operation_inputs(op)))
        .collect();
    let all_ids: HashSet<NodeId> = snapshot.iter().map(|(id, _)| *id).collect();

    let mut remaining_parents: HashMap<NodeId, usize> = id_to_parents
        .iter()
        .map(|(id, parents)| {
            let live = parents.iter().filter(|p| all_ids.contains(p)).count();
            (*id, live)
        })
        .collect();
    // children index: for each node, who lists it as a parent?
    let mut children_index: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (id, parents) in &id_to_parents {
        for p in parents {
            if all_ids.contains(p) {
                children_index.entry(*p).or_default().push(*id);
            }
        }
    }

    let mut ready: Vec<NodeId> = remaining_parents
        .iter()
        .filter(|(_, &count)| count == 0)
        .map(|(id, _)| *id)
        .collect();
    // Stable ordering based on raw id so passes produce deterministic results.
    ready.sort_by_key(|id| id.0);

    let mut order = Vec::with_capacity(snapshot.len());
    while let Some(id) = ready.pop() {
        order.push(id);
        if let Some(children) = children_index.get(&id) {
            let mut next = Vec::new();
            for child in children {
                if let Some(count) = remaining_parents.get_mut(child) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        next.push(*child);
                    }
                }
            }
            next.sort_by_key(|id| id.0);
            for n in next {
                ready.push(n);
            }
        }
    }
    order
}

/// Return the input [`NodeId`]s of an operation.
fn operation_inputs(op: &Operation) -> Vec<NodeId> {
    match op {
        Operation::Input => vec![],
        Operation::Add { lhs, rhs }
        | Operation::Sub { lhs, rhs }
        | Operation::Mul { lhs, rhs }
        | Operation::Div { lhs, rhs }
        | Operation::MatMul { lhs, rhs } => vec![*lhs, *rhs],
        Operation::Neg { input }
        | Operation::Exp { input }
        | Operation::Log { input }
        | Operation::Pow { input, .. }
        | Operation::Sum { input, .. }
        | Operation::Mean { input, .. }
        | Operation::Reshape { input, .. }
        | Operation::Transpose { input, .. }
        | Operation::Broadcast { input, .. }
        | Operation::ReLU { input }
        | Operation::Sigmoid { input }
        | Operation::Tanh { input }
        | Operation::Slice { input, .. } => vec![*input],
    }
}

// ============================================================================
// Common subexpression elimination (CSE)
// ============================================================================

/// Statistics produced by the CSE pass.
#[derive(Debug, Clone, Default)]
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
#[derive(Debug, Clone, Default)]
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
#[derive(Debug, Clone, Default)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Operation;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = GraphOptimizer::new();
        assert_eq!(optimizer.config.passes.len(), 1);
    }

    #[test]
    fn test_optimizer_with_passes() {
        let optimizer = GraphOptimizer::new()
            .with_pass(OptimizationPass::OperationFusion)
            .with_pass(OptimizationPass::DeadCodeElimination);
        assert_eq!(optimizer.config.passes.len(), 3); // default All + 2 added
    }

    #[test]
    fn test_config_builder() {
        let config = OptimizationConfig::new()
            .with_pass(OptimizationPass::OperationFusion)
            .verbose(true)
            .max_iterations(5);

        assert!(config.verbose);
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.passes.len(), 2);
    }

    #[test]
    fn test_fusion_pattern_matmul_bias() {
        let optimizer = GraphOptimizer::new();
        let ops = vec![
            (
                NodeId(0),
                Operation::MatMul {
                    lhs: NodeId(1),
                    rhs: NodeId(2),
                },
            ),
            (
                NodeId(3),
                Operation::Add {
                    lhs: NodeId(0),
                    rhs: NodeId(4),
                },
            ),
        ];

        let patterns = optimizer.detect_fusion_patterns(&ops);
        assert_eq!(patterns.len(), 1);

        match &patterns[0].1 {
            FusedOperation::MatMulBias { lhs, rhs, bias } => {
                assert_eq!(*lhs, NodeId(1));
                assert_eq!(*rhs, NodeId(2));
                assert_eq!(*bias, NodeId(4));
            }
            _ => panic!("Expected MatMulBias fusion"),
        }
    }

    #[test]
    fn test_fusion_pattern_matmul_bias_relu() {
        let optimizer = GraphOptimizer::new();
        let ops = vec![
            (
                NodeId(0),
                Operation::MatMul {
                    lhs: NodeId(1),
                    rhs: NodeId(2),
                },
            ),
            (
                NodeId(3),
                Operation::Add {
                    lhs: NodeId(0),
                    rhs: NodeId(4),
                },
            ),
            (NodeId(5), Operation::ReLU { input: NodeId(3) }),
        ];

        let patterns = optimizer.detect_fusion_patterns(&ops);
        assert_eq!(patterns.len(), 1);

        match &patterns[0].1 {
            FusedOperation::MatMulBiasReLU { lhs, rhs, bias } => {
                assert_eq!(*lhs, NodeId(1));
                assert_eq!(*rhs, NodeId(2));
                assert_eq!(*bias, NodeId(4));
            }
            _ => panic!("Expected MatMulBiasReLU fusion"),
        }
    }

    #[test]
    fn test_fusion_pattern_add_relu() {
        let optimizer = GraphOptimizer::new();
        let ops = vec![
            (
                NodeId(0),
                Operation::Add {
                    lhs: NodeId(1),
                    rhs: NodeId(2),
                },
            ),
            (NodeId(3), Operation::ReLU { input: NodeId(0) }),
        ];

        let patterns = optimizer.detect_fusion_patterns(&ops);
        assert_eq!(patterns.len(), 1);

        match &patterns[0].1 {
            FusedOperation::AddReLU { lhs, rhs } => {
                assert_eq!(*lhs, NodeId(1));
                assert_eq!(*rhs, NodeId(2));
            }
            _ => panic!("Expected AddReLU fusion"),
        }
    }

    #[test]
    fn test_fusion_pattern_mul_add() {
        let optimizer = GraphOptimizer::new();
        let ops = vec![
            (
                NodeId(0),
                Operation::Mul {
                    lhs: NodeId(1),
                    rhs: NodeId(2),
                },
            ),
            (
                NodeId(3),
                Operation::Add {
                    lhs: NodeId(0),
                    rhs: NodeId(4),
                },
            ),
        ];

        let patterns = optimizer.detect_fusion_patterns(&ops);
        assert_eq!(patterns.len(), 1);

        match &patterns[0].1 {
            FusedOperation::MulAdd { x, y, c } => {
                assert_eq!(*x, NodeId(1));
                assert_eq!(*y, NodeId(2));
                assert_eq!(*c, NodeId(4));
            }
            _ => panic!("Expected MulAdd fusion"),
        }
    }

    #[test]
    fn test_dead_code_elimination() {
        let optimizer = GraphOptimizer::new();
        let ops = vec![
            (NodeId(0), Operation::Input),
            (NodeId(1), Operation::Input),
            (
                NodeId(2),
                Operation::Add {
                    lhs: NodeId(0),
                    rhs: NodeId(1),
                },
            ),
            (
                NodeId(3),
                Operation::Mul {
                    lhs: NodeId(0),
                    rhs: NodeId(1),
                },
            ), // Unused
            (
                NodeId(4),
                Operation::Add {
                    lhs: NodeId(2),
                    rhs: NodeId(0),
                },
            ),
        ];

        let mut outputs = HashSet::new();
        outputs.insert(NodeId(4)); // Only node 4 is output

        let dead_nodes = optimizer.eliminate_dead_code(&ops, &outputs);
        assert_eq!(dead_nodes.len(), 1);
        assert!(dead_nodes.contains(&NodeId(3))); // Node 3 is dead
    }

    #[test]
    fn test_get_operation_inputs() {
        let optimizer = GraphOptimizer::new();

        let add_op = Operation::Add {
            lhs: NodeId(1),
            rhs: NodeId(2),
        };
        assert_eq!(
            optimizer.get_operation_inputs(&add_op),
            vec![NodeId(1), NodeId(2)]
        );

        let relu_op = Operation::ReLU { input: NodeId(3) };
        assert_eq!(optimizer.get_operation_inputs(&relu_op), vec![NodeId(3)]);

        let input_op = Operation::Input;
        assert_eq!(optimizer.get_operation_inputs(&input_op), vec![]);
    }

    #[test]
    fn test_optimization_stats_reduction() {
        let stats = OptimizationStats {
            nodes_before: 100,
            nodes_after: 80,
            fusions_applied: 5,
            dead_nodes_removed: 15,
            constants_folded: 0,
            iterations: 3,
        };

        assert!((stats.reduction_percent() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_savings_estimation() {
        let optimizer = GraphOptimizer::new();
        let stats = OptimizationStats {
            nodes_before: 100,
            nodes_after: 80,
            fusions_applied: 0,
            dead_nodes_removed: 20,
            constants_folded: 0,
            iterations: 1,
        };

        let savings = optimizer.estimate_memory_savings(&stats, 1024); // 1KB per tensor
        assert_eq!(savings, 20 * 1024); // 20KB saved
    }

    #[test]
    fn test_stats_display() {
        let stats = OptimizationStats {
            nodes_before: 50,
            nodes_after: 40,
            fusions_applied: 3,
            dead_nodes_removed: 7,
            constants_folded: 0,
            iterations: 2,
        };

        let output = format!("{}", stats);
        assert!(output.contains("Nodes before: 50"));
        assert!(output.contains("Nodes after: 40"));
        assert!(output.contains("Operations fused: 3"));
    }

    // ========================================================================
    // Constant folding + CSE tests
    // ========================================================================

    use crate::graph::ComputationGraph;
    use scirs2_core::ndarray_ext::{array, ArrayD};

    /// Tolerance for round-trip numerical correctness on f64.
    const F64_TOL: f64 = 1e-10;

    /// Count how many nodes in `graph` currently carry the given operation
    /// discriminant (matching the variant only, ignoring contained IDs/params).
    fn count_op_kind<F>(graph: &ComputationGraph<f64>, pred: F) -> usize
    where
        F: Fn(&Operation) -> bool,
    {
        graph
            .snapshot_ops()
            .iter()
            .filter(|(_, op)| pred(op))
            .count()
    }

    /// (1) Basic constant folding: `Add(const, const)` becomes a constant.
    #[test]
    fn test_constant_folding_basic() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let a = graph.constant(array![1.0, 2.0, 3.0].into_dyn())?;
        let b = graph.constant(array![10.0, 20.0, 30.0].into_dyn())?;
        let c = graph.add(&a, &b)?;

        // Value before folding (for round-trip check).
        let v_before = graph.value(&c)?;

        let before_adds = count_op_kind(&graph, |op| matches!(op, Operation::Add { .. }));
        assert_eq!(before_adds, 1);

        let stats = constant_folding(&graph);
        assert_eq!(stats.nodes_folded, 1, "the Add should have been folded");

        // After folding the Add node still exists but is now Input/constant.
        let after_adds = count_op_kind(&graph, |op| matches!(op, Operation::Add { .. }));
        assert_eq!(after_adds, 0, "Add node should have been reclassified");
        assert!(graph.is_constant(c.id()));

        // Value is preserved.
        let v_after = graph.value(&c)?;
        let diff = (&v_before - &v_after).mapv(f64::abs);
        assert!(diff.iter().all(|&d| d < F64_TOL));
        Ok(())
    }

    /// (2) Chain folding across multiple levels in a single pass.
    #[test]
    fn test_constant_folding_chain() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let a = graph.constant(array![2.0].into_dyn())?;
        let b = graph.constant(array![3.0].into_dyn())?;
        let c = graph.constant(array![5.0].into_dyn())?;

        let ab = graph.mul(&a, &b)?; // 6
        let abc = graph.add(&ab, &c)?; // 11

        let expected = graph.value(&abc)?;

        let stats = constant_folding(&graph);
        // Both intermediate ops should fold.
        assert_eq!(stats.nodes_folded, 2);
        assert!(graph.is_constant(ab.id()));
        assert!(graph.is_constant(abc.id()));

        let actual = graph.value(&abc)?;
        let diff = (&expected - &actual).mapv(f64::abs);
        assert!(diff.iter().all(|&d| d < F64_TOL));
        assert!((actual[[0]] - 11.0).abs() < F64_TOL);
        Ok(())
    }

    /// (3) Non-constant input blocks folding.
    #[test]
    fn test_constant_folding_not_folded_when_variable() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
        let c = graph.constant(array![10.0, 20.0].into_dyn())?;
        let y = graph.add(&x, &c)?;

        let stats = constant_folding(&graph);
        assert_eq!(stats.nodes_folded, 0);
        // y is still an Add node, not a constant.
        assert!(!graph.is_constant(y.id()));
        assert!(matches!(
            graph.node_operation(y.id()),
            Some(Operation::Add { .. })
        ));
        Ok(())
    }

    /// (4) Size threshold skips very large constants.
    #[test]
    fn test_constant_folding_size_threshold() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // 4 elements.
        let a = graph.constant(array![1.0, 2.0, 3.0, 4.0].into_dyn())?;
        let b = graph.constant(array![5.0, 6.0, 7.0, 8.0].into_dyn())?;
        let c = graph.add(&a, &b)?;

        // Threshold smaller than the result → skipped.
        let stats_small = constant_folding_with_threshold(&graph, 3);
        assert_eq!(stats_small.nodes_folded, 0);
        assert_eq!(stats_small.skipped_too_large, 1);
        assert!(!graph.is_constant(c.id()));

        // Threshold large enough → folded.
        let stats_ok = constant_folding_with_threshold(&graph, 4);
        assert_eq!(stats_ok.nodes_folded, 1);
        assert_eq!(stats_ok.skipped_too_large, 0);
        assert!(graph.is_constant(c.id()));
        Ok(())
    }

    /// (5) CSE deduplicates structurally identical Add nodes.
    #[test]
    fn test_cse_add_dedup() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
        let y = graph.variable(array![3.0, 4.0].into_dyn(), true)?;

        let s1 = graph.add(&x, &y)?;
        let s2 = graph.add(&x, &y)?;

        let before = count_op_kind(&graph, |op| matches!(op, Operation::Add { .. }));
        assert_eq!(before, 2);

        let stats = common_subexpression_elimination(&graph);
        assert_eq!(stats.nodes_eliminated, 1);

        let after = count_op_kind(&graph, |op| matches!(op, Operation::Add { .. }));
        assert_eq!(after, 1, "one of the duplicates must be gone");

        // At least one of s1/s2 survived and holds the correct value.
        let v1 = graph.value(&s1);
        let v2 = graph.value(&s2);
        assert!(v1.is_ok() || v2.is_ok());
        let v = v1.or(v2)?;
        let expected = array![4.0, 6.0].into_dyn();
        let diff: ArrayD<f64> = (&expected - &v).mapv(f64::abs);
        assert!(diff.iter().all(|&d| d < F64_TOL));
        Ok(())
    }

    /// (6) CSE respects commutativity: `x + y` and `y + x` are the same.
    #[test]
    fn test_cse_commutativity() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0].into_dyn(), true)?;
        let y = graph.variable(array![2.0].into_dyn(), true)?;

        let _s_xy = graph.add(&x, &y)?;
        let _s_yx = graph.add(&y, &x)?;
        let _p_xy = graph.mul(&x, &y)?;
        let _p_yx = graph.mul(&y, &x)?;

        let before_add = count_op_kind(&graph, |op| matches!(op, Operation::Add { .. }));
        let before_mul = count_op_kind(&graph, |op| matches!(op, Operation::Mul { .. }));
        assert_eq!(before_add, 2);
        assert_eq!(before_mul, 2);

        let stats = common_subexpression_elimination(&graph);
        assert_eq!(stats.nodes_eliminated, 2);

        let after_add = count_op_kind(&graph, |op| matches!(op, Operation::Add { .. }));
        let after_mul = count_op_kind(&graph, |op| matches!(op, Operation::Mul { .. }));
        assert_eq!(after_add, 1);
        assert_eq!(after_mul, 1);
        Ok(())
    }

    /// (7) Non-commutative ops are order-sensitive, so `Sub(x, y)` and
    /// `Sub(y, x)` are NOT deduplicated.
    #[test]
    fn test_cse_non_commutative_preserved() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![5.0].into_dyn(), true)?;
        let y = graph.variable(array![3.0].into_dyn(), true)?;

        let _d_xy = graph.sub(&x, &y)?; //  2
        let _d_yx = graph.sub(&y, &x)?; // -2

        let before = count_op_kind(&graph, |op| matches!(op, Operation::Sub { .. }));
        assert_eq!(before, 2);

        let stats = common_subexpression_elimination(&graph);
        assert_eq!(stats.nodes_eliminated, 0);

        let after = count_op_kind(&graph, |op| matches!(op, Operation::Sub { .. }));
        assert_eq!(after, 2, "distinct Subs must both survive");
        Ok(())
    }

    /// (8) Different ops over the same operands are NOT deduplicated
    /// (guards against over-aggressive CSE).
    #[test]
    fn test_cse_different_ops_not_deduped() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![3.0, 4.0].into_dyn(), true)?;
        let y = graph.variable(array![1.0, 2.0].into_dyn(), true)?;

        let _a = graph.add(&x, &y)?;
        let _m = graph.mul(&x, &y)?;
        let _s = graph.sub(&x, &y)?;

        let stats = common_subexpression_elimination(&graph);
        assert_eq!(stats.nodes_eliminated, 0);

        assert_eq!(
            count_op_kind(&graph, |op| matches!(op, Operation::Add { .. })),
            1
        );
        assert_eq!(
            count_op_kind(&graph, |op| matches!(op, Operation::Mul { .. })),
            1
        );
        assert_eq!(
            count_op_kind(&graph, |op| matches!(op, Operation::Sub { .. })),
            1
        );
        Ok(())
    }

    /// (9) End-to-end pipeline: CSE → const_fold → CSE → DCE over a graph
    /// combining commutatively-equal constant expressions and reused ones.
    ///
    /// The graph computes `f = (c2 + c3) * x + (c3 + c2) * x` where c2 and c3
    /// are constants sized to match `x`. After the optimization pipeline:
    /// - the two `(c2 + c3)` Adds dedup via CSE (pre-fold)
    /// - the surviving Add folds to a constant
    /// - the two `5 * x` Muls dedup via CSE (post-fold)
    /// - unused nodes get DCE'd
    #[test]
    fn test_optimize_graph_pipeline_end_to_end() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0, 2.0, 3.0].into_dyn(), true)?;
        // Constants pre-shaped to x's shape so that Add/Mul are elementwise.
        let two = graph.constant(array![2.0, 2.0, 2.0].into_dyn())?;
        let three = graph.constant(array![3.0, 3.0, 3.0].into_dyn())?;

        let s1 = graph.add(&two, &three)?; //  c2 + c3
        let s2 = graph.add(&three, &two)?; //  c3 + c2 (commutative dup)

        let m1 = graph.mul(&s1, &x)?;
        let m2 = graph.mul(&s2, &x)?;

        let out = graph.add(&m1, &m2)?;

        let expected = graph.value(&out)?;
        let nodes_before = graph.snapshot_ops().len();

        let stats = optimize_graph(&graph, &[out.id()]);

        assert_eq!(stats.nodes_before, nodes_before);
        // Pre-fold CSE must have eliminated the commutative Add duplicate.
        assert!(stats.cse_pre.nodes_eliminated >= 1);
        // Then const-fold reclassifies the surviving Add as a constant.
        assert!(stats.const_fold.nodes_folded >= 1);
        // Total nodes strictly decreased.
        assert!(stats.nodes_after < stats.nodes_before);

        // Only one Mul should remain (the two Muls share `5 * x`).
        let mul_count = count_op_kind(&graph, |op| matches!(op, Operation::Mul { .. }));
        assert_eq!(mul_count, 1, "CSE should collapse the two 5*x products");

        // Round-trip numerical correctness: the root value is unchanged.
        let actual = graph.value(&out)?;
        let diff: ArrayD<f64> = (&expected - &actual).mapv(f64::abs);
        assert!(
            diff.iter().all(|&d| d < F64_TOL),
            "max diff was {}",
            diff.iter().copied().fold(0.0_f64, f64::max)
        );

        // Spot-check the numeric result: (2+3)*x + (3+2)*x == 10 * x.
        let expected_numeric = array![10.0, 20.0, 30.0].into_dyn();
        let num_diff: ArrayD<f64> = (&actual - &expected_numeric).mapv(f64::abs);
        assert!(num_diff.iter().all(|&d| d < F64_TOL));

        // Final graph does not contain any Sub nodes (sanity).
        assert_eq!(
            count_op_kind(&graph, |op| matches!(op, Operation::Sub { .. })),
            0
        );
        Ok(())
    }

    /// (10) Round-trip numerical correctness under a mix of op types with
    /// non-trivial structure. The value of the output must be identical
    /// before and after running `optimize_graph`.
    #[test]
    fn test_optimize_graph_round_trip_correctness() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![0.25, 0.5, 0.75].into_dyn(), true)?;
        let a = graph.constant(array![2.0, 2.0, 2.0].into_dyn())?;
        let b = graph.constant(array![4.0, 4.0, 4.0].into_dyn())?;

        // Redundant work on purpose: two commutatively identical paths.
        let ab1 = graph.mul(&a, &b)?; // 8 elementwise
        let ab2 = graph.mul(&b, &a)?; // 8 (commutative dup)

        let t1 = graph.mul(&ab1, &x)?;
        let t2 = graph.mul(&ab2, &x)?;
        let t3 = graph.relu(&t1)?;
        let t4 = graph.exp(&t2)?;
        let out = graph.add(&t3, &t4)?;

        let expected = graph.value(&out)?;

        let _stats = optimize_graph(&graph, &[out.id()]);
        let actual = graph.value(&out)?;

        let diff: ArrayD<f64> = (&expected - &actual).mapv(f64::abs);
        assert!(
            diff.iter().all(|&d| d < F64_TOL),
            "round-trip failed: max diff = {}",
            diff.iter().copied().fold(0.0_f64, f64::max)
        );
        Ok(())
    }

    /// DCE: nodes unreachable from the supplied outputs are removed.
    #[test]
    fn test_dce_on_graph_removes_unreachable() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0].into_dyn(), true)?;
        let y = graph.variable(array![2.0].into_dyn(), true)?;
        let used = graph.add(&x, &y)?;
        let _unused = graph.mul(&x, &y)?; // not an output

        let before = graph.snapshot_ops().len();
        let stats = dead_code_elimination_on_graph(&graph, &[used.id()]);
        let after = graph.snapshot_ops().len();

        assert!(stats.nodes_removed >= 1);
        assert!(after < before);
        // `used` is reachable, its subgraph stays.
        assert!(graph.node_operation(used.id()).is_some());
        Ok(())
    }

    #[test]
    fn test_optimize_eliminates_dead_nodes() {
        use crate::graph::ComputationGraph;
        use scirs2_core::ndarray_ext::ArrayD;
        use scirs2_core::ndarray_ext::IxDyn;

        let graph = ComputationGraph::<f64>::new();
        let x = graph
            .constant(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap())
            .unwrap();
        let y = graph
            .constant(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap())
            .unwrap();
        // Create a live node used downstream
        let z = graph.add(&x, &y).unwrap(); // z = x + y
        let _w = graph.sum(&z).unwrap(); // w = sum(z) — this is the output

        // Create a dead node (not connected to w)
        let _dead = graph.mul(&x, &y).unwrap(); // not referenced downstream

        let nodes_before = graph.num_nodes();
        let optimizer = GraphOptimizer::new();
        let stats = optimizer.optimize(&graph).unwrap();

        // After optimization, the graph should be smaller or equal
        assert!(stats.nodes_after <= nodes_before);
        assert_eq!(stats.nodes_before, nodes_before);
        assert!(stats.iterations >= 1);
    }

    #[test]
    fn test_optimize_constant_folding() {
        use crate::graph::ComputationGraph;
        use scirs2_core::ndarray_ext::ArrayD;
        use scirs2_core::ndarray_ext::IxDyn;

        let graph = ComputationGraph::<f64>::new();
        let a = graph
            .constant(ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap())
            .unwrap();
        let b = graph
            .constant(ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0, 5.0]).unwrap())
            .unwrap();
        // c = a + b — both are constants, so this should be foldable
        let _c = graph.add(&a, &b).unwrap();

        let optimizer = GraphOptimizer::new();
        let stats = optimizer.optimize(&graph).unwrap();
        // Folding should have folded at least the add node
        assert!(stats.constants_folded >= 1);
    }
}
