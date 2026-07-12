//! Operation fusion: pattern detection and plan compilation.
//!
//! # What "fusion" means here
//!
//! A fused operation is a *single* kernel that computes what two or three
//! graph nodes used to compute, materializing **one** output buffer instead of
//! two or three, and differentiating with a single VJP that never needs the
//! interior values. Concretely:
//!
//! | Pattern                    | Fused kernel                     | Buffers before → after |
//! |----------------------------|----------------------------------|------------------------|
//! | `MatMul → Add`             | [`FusedOperation::MatMulBias`]     | 2 → 1 |
//! | `MatMul → Add → ReLU`      | [`FusedOperation::MatMulBiasReLU`] | 3 → 1 |
//! | `Mul → Add`                | [`FusedOperation::MulAdd`]         | 2 → 1 |
//! | `Add → ReLU`               | [`FusedOperation::AddReLU`]        | 2 → 1 |
//!
//! # Soundness conditions
//!
//! A chain may only be fused when every **interior** node (the `MatMul` in
//! `MatMul → Add`, the `Add` in `Add → ReLU`, …) satisfies both:
//!
//! 1. it has exactly **one** consumer in the live graph — otherwise some other
//!    node still needs the buffer we are about to stop materializing; and
//! 2. it is **not** one of the requested plan outputs — the caller asked for
//!    that value by name.
//!
//! Both conditions are enforced in [`compile_plan`]; a chain that violates
//! either is left unfused (and is still executed correctly, just unfused).
//!
//! # Why a plan and not an in-place rewrite
//!
//! See the [module docs](crate::graph_optimizer): `ComputationGraph` is an
//! eager tape whose `Operation` enum has no fused variants, so a fused node
//! can neither save forward work (already done) nor be executed by
//! `ComputationGraph::backward` (an exhaustive match over `Operation`). The
//! plan compiled here owns its own executor, in [`super::exec`].

use super::{operation_inputs, topological_order_from_snapshot};
use crate::graph::{ComputationGraph, NodeId, Operation};
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::ScalarOperand;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, HashSet};

/// A fused operation: one kernel standing in for a chain of graph nodes.
///
/// The `NodeId`s stored here always refer to the *live* operands of the chain
/// (never to an interior node that the fusion eliminates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOperation {
    /// `C = A @ B + bias`
    MatMulBias {
        /// Left matmul operand.
        lhs: NodeId,
        /// Right matmul operand.
        rhs: NodeId,
        /// Bias, broadcast against `A @ B`.
        bias: NodeId,
    },
    /// `C = ReLU(A @ B + bias)`
    MatMulBiasReLU {
        /// Left matmul operand.
        lhs: NodeId,
        /// Right matmul operand.
        rhs: NodeId,
        /// Bias, broadcast against `A @ B`.
        bias: NodeId,
    },
    /// `z = x * y + c` (fused multiply-add)
    MulAdd {
        /// First multiplicand.
        x: NodeId,
        /// Second multiplicand.
        y: NodeId,
        /// Addend, broadcast against `x * y`.
        c: NodeId,
    },
    /// `z = ReLU(x + y)`
    AddReLU {
        /// Left addend.
        lhs: NodeId,
        /// Right addend.
        rhs: NodeId,
    },
}

impl FusedOperation {
    /// Operand node IDs consumed by the fused kernel.
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            FusedOperation::MatMulBias { lhs, rhs, bias }
            | FusedOperation::MatMulBiasReLU { lhs, rhs, bias } => vec![*lhs, *rhs, *bias],
            FusedOperation::MulAdd { x, y, c } => vec![*x, *y, *c],
            FusedOperation::AddReLU { lhs, rhs } => vec![*lhs, *rhs],
        }
    }

    /// Human-readable kernel name.
    pub fn name(&self) -> &'static str {
        match self {
            FusedOperation::MatMulBias { .. } => "MatMulBias",
            FusedOperation::MatMulBiasReLU { .. } => "MatMulBiasReLU",
            FusedOperation::MulAdd { .. } => "MulAdd",
            FusedOperation::AddReLU { .. } => "AddReLU",
        }
    }
}

/// One matched fusion site in a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionMatch {
    /// The node whose value the fused kernel produces: the *last* node of the
    /// chain. Consumers of the chain reference this ID, so it is preserved.
    pub output: NodeId,
    /// The fused kernel replacing the chain.
    pub fused: FusedOperation,
    /// Interior nodes eliminated by the fusion. Their buffers are never
    /// materialized, in the forward pass or the backward pass.
    pub eliminated: Vec<NodeId>,
}

/// A single executable step of a [`FusionPlan`].
#[derive(Debug, Clone, PartialEq)]
pub enum PlanStep {
    /// An unfused graph operation.
    Base {
        /// Node this step produces.
        id: NodeId,
        /// The operation.
        op: Operation,
    },
    /// A fused kernel standing in for a chain of graph operations.
    Fused {
        /// Node this step produces (the last node of the fused chain).
        id: NodeId,
        /// The fused kernel.
        op: FusedOperation,
    },
}

impl PlanStep {
    /// The node ID this step produces.
    pub fn id(&self) -> NodeId {
        match self {
            PlanStep::Base { id, .. } | PlanStep::Fused { id, .. } => *id,
        }
    }

    /// The node IDs this step reads.
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            PlanStep::Base { op, .. } => operation_inputs(op),
            PlanStep::Fused { op, .. } => op.inputs(),
        }
    }

    /// Whether this step is a fused kernel.
    pub fn is_fused(&self) -> bool {
        matches!(self, PlanStep::Fused { .. })
    }
}

/// A leaf of the plan: an `Operation::Input` node whose value is supplied at
/// execution time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanLeaf {
    /// The graph node this leaf corresponds to.
    pub id: NodeId,
    /// Whether the graph tracks gradients for it.
    pub requires_grad: bool,
}

/// Knobs for [`compile_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionConfig {
    /// Fuse matched chains. Setting this to `false` compiles a *reference*
    /// plan: identical semantics, one step per graph node, no fused kernels.
    /// The unfused plan is what the fused plan is measured against (node
    /// counts, buffers, elements, gradients).
    pub enable_fusion: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
        }
    }
}

/// Compile-time statistics for a [`FusionPlan`].
///
/// Everything here is *counted*, never estimated. `interior_elements_eliminated`
/// is summed from the actual cached element counts of the eliminated nodes in
/// the source graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionStats {
    /// Live graph nodes reachable from the requested outputs.
    pub graph_nodes: usize,
    /// Leaves + steps in the compiled plan.
    pub plan_nodes: usize,
    /// Executable steps in the plan. This is exactly the number of tensors the
    /// forward pass materializes.
    pub plan_steps: usize,
    /// Number of fused kernels in the plan.
    pub fusions_applied: usize,
    /// `MatMul + Add` fusions.
    pub matmul_bias: usize,
    /// `MatMul + Add + ReLU` fusions.
    pub matmul_bias_relu: usize,
    /// `Mul + Add` fusions.
    pub mul_add: usize,
    /// `Add + ReLU` fusions.
    pub add_relu: usize,
    /// Interior nodes eliminated by fusion (buffers never materialized).
    pub interior_nodes_eliminated: usize,
    /// Total element count of those eliminated interior buffers, taken from the
    /// source graph's cached values. The backward pass saves the same again,
    /// because a fused VJP never allocates a gradient for an interior node.
    pub interior_elements_eliminated: usize,
}

/// An executable, fused compilation of a [`ComputationGraph`].
///
/// The plan is a pure IR: it holds no tensors, so a plan compiled from an
/// `f64` graph can be executed at any element type. Feed leaf values to
/// [`FusionPlan::forward`] and differentiate with [`FusionPlan::backward`].
#[derive(Debug, Clone, PartialEq)]
pub struct FusionPlan {
    pub(crate) leaves: Vec<PlanLeaf>,
    pub(crate) steps: Vec<PlanStep>,
    pub(crate) outputs: Vec<NodeId>,
    /// Nodes that transitively depend on a `requires_grad` leaf. The backward
    /// pass skips everything else, exactly like `ComputationGraph::backward`.
    pub(crate) needs_grad: HashSet<NodeId>,
    pub(crate) stats: FusionStats,
    pub(crate) matches: Vec<FusionMatch>,
}

impl FusionPlan {
    /// Compile-time statistics.
    pub fn stats(&self) -> &FusionStats {
        &self.stats
    }

    /// The plan's leaves, in compilation order.
    pub fn leaves(&self) -> &[PlanLeaf] {
        &self.leaves
    }

    /// Leaf node IDs. Every one of these must be present in the `feeds` map
    /// passed to [`FusionPlan::forward`].
    pub fn leaf_ids(&self) -> Vec<NodeId> {
        self.leaves.iter().map(|l| l.id).collect()
    }

    /// Leaf node IDs that require gradients.
    pub fn requires_grad_leaves(&self) -> Vec<NodeId> {
        self.leaves
            .iter()
            .filter(|l| l.requires_grad)
            .map(|l| l.id)
            .collect()
    }

    /// The executable steps, in topological order.
    pub fn steps(&self) -> &[PlanStep] {
        &self.steps
    }

    /// The requested output node IDs.
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// The fused kernels in the plan, paired with the node each produces.
    pub fn fused_steps(&self) -> Vec<(NodeId, FusedOperation)> {
        self.steps
            .iter()
            .filter_map(|s| match s {
                PlanStep::Fused { id, op } => Some((*id, *op)),
                PlanStep::Base { .. } => None,
            })
            .collect()
    }
}

/// Compile `graph` into an executable plan with fusion enabled.
///
/// Equivalent to [`compile_plan`] with [`FusionConfig::default`].
pub fn compile_fused_plan<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
    outputs: &[NodeId],
) -> Result<FusionPlan> {
    compile_plan(graph, outputs, &FusionConfig::default())
}

/// Compile the sub-graph reachable from `outputs` into an executable plan.
///
/// # Fusion
///
/// With `config.enable_fusion` (the default), chains matching the patterns in
/// the [module docs](self) are collapsed into single kernels, subject to the
/// soundness conditions listed there. With fusion disabled the plan has one
/// step per graph node and is the reference implementation.
///
/// # Unsupported operations
///
/// [`Operation::Reshape`] and [`Operation::Broadcast`] record only their
/// **input** shape (they exist to serve the backward pass), so their output
/// shape cannot be recovered from the graph IR and the plan cannot replay
/// them. Compiling a graph containing either returns an error rather than
/// guessing. Every other operation is supported.
///
/// # Complexity
///
/// O(V + E) over the live sub-graph.
pub fn compile_plan<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
    outputs: &[NodeId],
    config: &FusionConfig,
) -> Result<FusionPlan> {
    if outputs.is_empty() {
        return Err(anyhow!("compile_plan: no outputs requested"));
    }
    for &out in outputs {
        if graph.node_operation(out).is_none() {
            return Err(anyhow!("compile_plan: output {} is not in the graph", out));
        }
    }

    let live = graph.reachable_from(outputs);
    let snapshot: Vec<(NodeId, Operation)> = graph
        .snapshot_ops()
        .into_iter()
        .filter(|(id, _)| live.contains(id))
        .collect();

    for (id, op) in &snapshot {
        if let Some(name) = unreplayable_op_name(op) {
            return Err(anyhow!(
                "compile_plan: node {} is an `Operation::{}`, which records only its input shape; \
                 its output shape cannot be recovered from the graph IR, so the plan cannot replay it",
                id,
                name
            ));
        }
    }

    let ops: HashMap<NodeId, Operation> = snapshot.iter().cloned().collect();
    let order = topological_order_from_snapshot(&snapshot);
    if order.len() != snapshot.len() {
        return Err(anyhow!(
            "compile_plan: the live sub-graph is cyclic ({} of {} nodes ordered)",
            order.len(),
            snapshot.len()
        ));
    }

    // Consumer counts within the live sub-graph. An output counts as an extra
    // consumer: the caller asked for that value, so it must be materialized.
    let output_set: HashSet<NodeId> = outputs.iter().copied().collect();
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for (_, op) in &snapshot {
        for p in operation_inputs(op) {
            *consumers.entry(p).or_insert(0) += 1;
        }
    }
    for &out in &output_set {
        *consumers.entry(out).or_insert(0) += 1;
    }

    let mut builder = PlanBuilder {
        graph_nodes: live.len(),
        leaves: Vec::new(),
        slots: Vec::new(),
        slot_of: HashMap::new(),
        matches: Vec::new(),
        stats: FusionStats {
            graph_nodes: live.len(),
            ..FusionStats::default()
        },
    };

    for id in order {
        let op = match ops.get(&id) {
            Some(o) => o.clone(),
            None => continue,
        };

        if matches!(op, Operation::Input) {
            builder.leaves.push(PlanLeaf {
                id,
                requires_grad: graph.node_requires_grad(id),
            });
            continue;
        }

        if config.enable_fusion && builder.try_fuse(graph, id, &op, &consumers, &output_set) {
            continue;
        }

        builder.push_base(id, op);
    }

    builder.finish(outputs)
}

/// Detect every fusion site in the sub-graph reachable from `outputs`.
///
/// This is the same matcher [`compile_plan`] drives, exposed for inspection.
/// The matches respect the soundness conditions (single-consumer interior
/// nodes that are not themselves outputs), so each one is safe to apply.
pub fn detect_fusion_patterns<T: Float + ScalarOperand + FromPrimitive>(
    graph: &ComputationGraph<T>,
    outputs: &[NodeId],
) -> Result<Vec<FusionMatch>> {
    let plan = compile_plan(graph, outputs, &FusionConfig::default())?;
    Ok(plan.matches_snapshot())
}

impl FusionPlan {
    /// Reconstruct the [`FusionMatch`] list from the compiled steps.
    fn matches_snapshot(&self) -> Vec<FusionMatch> {
        self.matches.clone()
    }
}

/// Name of an operation the plan cannot replay, if any.
fn unreplayable_op_name(op: &Operation) -> Option<&'static str> {
    match op {
        Operation::Reshape { .. } => Some("Reshape"),
        Operation::Broadcast { .. } => Some("Broadcast"),
        _ => None,
    }
}

/// Incremental plan construction with peephole fusion.
struct PlanBuilder {
    graph_nodes: usize,
    leaves: Vec<PlanLeaf>,
    /// `None` marks a slot whose step was absorbed into a later fusion.
    slots: Vec<Option<PlanStep>>,
    /// Node → slot index, for nodes still materialized as a step.
    slot_of: HashMap<NodeId, usize>,
    matches: Vec<FusionMatch>,
    stats: FusionStats,
}

impl PlanBuilder {
    fn push_base(&mut self, id: NodeId, op: Operation) {
        self.slot_of.insert(id, self.slots.len());
        self.slots.push(Some(PlanStep::Base { id, op }));
    }

    fn push_fused(&mut self, id: NodeId, op: FusedOperation) {
        self.slot_of.insert(id, self.slots.len());
        self.slots.push(Some(PlanStep::Fused { id, op }));
    }

    /// Absorb the step producing `id`: its buffer is never materialized.
    fn absorb(&mut self, id: NodeId) {
        if let Some(slot) = self.slot_of.remove(&id) {
            self.slots[slot] = None;
        }
    }

    /// Is `id` an interior node we may legally absorb?
    ///
    /// Requires: currently materialized as a step (so not a leaf, not already
    /// absorbed), exactly one consumer, and not a requested output.
    fn absorbable(
        &self,
        id: NodeId,
        consumers: &HashMap<NodeId, usize>,
        outputs: &HashSet<NodeId>,
    ) -> bool {
        self.slot_of.contains_key(&id)
            && consumers.get(&id).copied() == Some(1)
            && !outputs.contains(&id)
    }

    fn step_of(&self, id: NodeId) -> Option<&PlanStep> {
        self.slot_of
            .get(&id)
            .and_then(|&slot| self.slots[slot].as_ref())
    }

    /// Try to fuse node `id` with one of its already-emitted parents.
    ///
    /// Returns `true` when a fused step was emitted for `id`.
    fn try_fuse<T: Float + ScalarOperand + FromPrimitive>(
        &mut self,
        graph: &ComputationGraph<T>,
        id: NodeId,
        op: &Operation,
        consumers: &HashMap<NodeId, usize>,
        outputs: &HashSet<NodeId>,
    ) -> bool {
        match *op {
            // ReLU(interior) — either upgrades a MatMulBias to MatMulBiasReLU,
            // or fuses a plain Add into AddReLU.
            Operation::ReLU { input } => {
                if !self.absorbable(input, consumers, outputs) {
                    return false;
                }
                match self.step_of(input) {
                    Some(PlanStep::Fused {
                        op: FusedOperation::MatMulBias { lhs, rhs, bias },
                        ..
                    }) => {
                        let fused = FusedOperation::MatMulBiasReLU {
                            lhs: *lhs,
                            rhs: *rhs,
                            bias: *bias,
                        };
                        // The MatMulBias match we are upgrading already recorded
                        // the MatMul it absorbed; carry those eliminations over.
                        let mut eliminated = self.take_match_eliminations(input);
                        eliminated.push(input);
                        self.absorb(input);
                        self.push_fused(id, fused);
                        self.stats.matmul_bias -= 1;
                        self.stats.matmul_bias_relu += 1;
                        self.record_elimination_elements(graph, &[input]);
                        self.matches.push(FusionMatch {
                            output: id,
                            fused,
                            eliminated,
                        });
                        true
                    }
                    Some(PlanStep::Base {
                        op: Operation::Add { lhs, rhs },
                        ..
                    }) => {
                        let fused = FusedOperation::AddReLU {
                            lhs: *lhs,
                            rhs: *rhs,
                        };
                        self.absorb(input);
                        self.push_fused(id, fused);
                        self.stats.add_relu += 1;
                        self.stats.fusions_applied += 1;
                        self.record_elimination_elements(graph, &[input]);
                        self.matches.push(FusionMatch {
                            output: id,
                            fused,
                            eliminated: vec![input],
                        });
                        true
                    }
                    _ => false,
                }
            }

            // Add(interior, other) — absorbs a MatMul (→ MatMulBias) or a Mul
            // (→ MulAdd). Either operand may be the interior one.
            Operation::Add { lhs, rhs } => {
                for (cand, other) in [(lhs, rhs), (rhs, lhs)] {
                    if !self.absorbable(cand, consumers, outputs) {
                        continue;
                    }
                    let fused = match self.step_of(cand) {
                        Some(PlanStep::Base {
                            op: Operation::MatMul { lhs: ml, rhs: mr },
                            ..
                        }) => FusedOperation::MatMulBias {
                            lhs: *ml,
                            rhs: *mr,
                            bias: other,
                        },
                        Some(PlanStep::Base {
                            op: Operation::Mul { lhs: xl, rhs: xr },
                            ..
                        }) => FusedOperation::MulAdd {
                            x: *xl,
                            y: *xr,
                            c: other,
                        },
                        _ => continue,
                    };
                    self.absorb(cand);
                    self.push_fused(id, fused);
                    match fused {
                        FusedOperation::MatMulBias { .. } => self.stats.matmul_bias += 1,
                        FusedOperation::MulAdd { .. } => self.stats.mul_add += 1,
                        _ => unreachable!("only MatMulBias / MulAdd are produced here"),
                    }
                    self.stats.fusions_applied += 1;
                    self.record_elimination_elements(graph, &[cand]);
                    self.matches.push(FusionMatch {
                        output: id,
                        fused,
                        eliminated: vec![cand],
                    });
                    return true;
                }
                false
            }

            _ => false,
        }
    }

    /// Remove the recorded match that produced `id` (it is being upgraded into
    /// a longer fusion) and return the interior nodes it had eliminated.
    fn take_match_eliminations(&mut self, id: NodeId) -> Vec<NodeId> {
        if let Some(pos) = self.matches.iter().position(|m| m.output == id) {
            self.matches.remove(pos).eliminated
        } else {
            Vec::new()
        }
    }

    fn record_elimination_elements<T: Float + ScalarOperand + FromPrimitive>(
        &mut self,
        graph: &ComputationGraph<T>,
        eliminated: &[NodeId],
    ) {
        for &id in eliminated {
            self.stats.interior_nodes_eliminated += 1;
            self.stats.interior_elements_eliminated += graph.node_element_count(id).unwrap_or(0);
        }
    }

    /// Compact the slots, validate the plan, and compute gradient reachability.
    fn finish(mut self, outputs: &[NodeId]) -> Result<FusionPlan> {
        let steps: Vec<PlanStep> = self.slots.into_iter().flatten().collect();

        // Validate: every step reads only leaves or earlier steps, and every
        // requested output is produced. A violation is a compiler bug, so it is
        // reported loudly rather than papered over.
        let mut available: HashSet<NodeId> = self.leaves.iter().map(|l| l.id).collect();
        for step in &steps {
            for input in step.inputs() {
                if !available.contains(&input) {
                    return Err(anyhow!(
                        "compile_plan: step {} reads {} before it is produced",
                        step.id(),
                        input
                    ));
                }
            }
            available.insert(step.id());
        }
        for &out in outputs {
            if !available.contains(&out) {
                return Err(anyhow!(
                    "compile_plan: output {} is not produced by the plan (it was absorbed by a fusion)",
                    out
                ));
            }
        }

        // Gradient reachability: a step needs a gradient iff any input does.
        let mut needs_grad: HashSet<NodeId> = self
            .leaves
            .iter()
            .filter(|l| l.requires_grad)
            .map(|l| l.id)
            .collect();
        for step in &steps {
            if step.inputs().iter().any(|i| needs_grad.contains(i)) {
                needs_grad.insert(step.id());
            }
        }

        self.stats.plan_steps = steps.len();
        self.stats.plan_nodes = steps.len() + self.leaves.len();
        self.stats.graph_nodes = self.graph_nodes;

        Ok(FusionPlan {
            leaves: self.leaves,
            steps,
            outputs: outputs.to_vec(),
            needs_grad,
            stats: self.stats,
            matches: self.matches,
        })
    }
}
