//! Graph optimization for computation graphs.
//!
//! This module provides two distinct, honestly-separated capabilities:
//!
//! 1. **In-place rewrite passes** ([`GraphOptimizer::optimize`]) that mutate a
//!    [`ComputationGraph`] in place:
//!    - **Common subexpression elimination** — deduplicate structurally
//!      identical operations (commutativity-aware).
//!    - **Constant folding** — reclassify all-constant sub-expressions as
//!      constants.
//!    - **Dead code elimination** — drop nodes unreachable from the outputs.
//!
//! 2. **Operation fusion** ([`compile_fused_plan`]) which compiles the graph
//!    into an executable [`FusionPlan`] whose fused kernels
//!    (`MatMul+Bias`, `MatMul+Bias+ReLU`, `Mul+Add`, `Add+ReLU`) materialize
//!    strictly fewer buffers than the unfused graph, and which computes
//!    gradients through fused VJPs.
//!
//! # Why fusion is a *plan* and not an in-place pass
//!
//! [`ComputationGraph`] is an **eager tape**: every operation is evaluated the
//! moment it is recorded. Two consequences follow, and they completely
//! determine the design here:
//!
//! - By the time any optimizer runs, every intermediate buffer has *already*
//!   been allocated and filled. An in-place rewrite therefore cannot save a
//!   single forward FLOP or a single forward allocation — the work is done.
//!   Fusion only pays off when the graph is **re-executed**, which is exactly
//!   what a compiled plan enables (and what a training loop needs).
//! - [`Operation`] has no fused variants, and `ComputationGraph::backward`
//!   dispatches exhaustively over `Operation`, so a fused node cannot be
//!   *executed* by the graph itself even if it could be spliced in.
//!
//! So [`FusionPlan`] owns its own executor: it replays the graph with fused
//! kernels and differentiates it with fused VJPs. See
//! [`fusion`] for the pattern matcher and [`exec`] for the kernels.
//!
//! # Example
//!
//! ```rust,ignore
//! use tenrso_ad::graph::ComputationGraph;
//! use tenrso_ad::graph_optimizer::{compile_fused_plan, GraphOptimizer, OptimizationPass};
//! use std::collections::HashMap;
//!
//! let graph = ComputationGraph::<f64>::new();
//! // ... build graph: out = relu(x @ w + b) ...
//!
//! // (1) In-place rewrites (CSE / constant folding / DCE):
//! let optimizer = GraphOptimizer::new().with_pass(OptimizationPass::DeadCodeElimination);
//! let stats = optimizer.optimize(&graph)?;
//!
//! // (2) Real fusion, with an executable plan:
//! let plan = compile_fused_plan(&graph, &[out.id()])?;
//! assert_eq!(plan.stats().fusions_applied, 1); // MatMulBiasReLU
//!
//! let mut feeds = HashMap::new();
//! feeds.insert(x.id(), x_value);
//! feeds.insert(w.id(), w_value);
//! feeds.insert(b.id(), b_value);
//!
//! let exec = plan.forward(&feeds)?;
//! let grads = plan.backward(&exec, out.id(), &seed)?;
//! ```

use crate::graph::{ComputationGraph, NodeId, Operation};
use anyhow::Result;
use scirs2_core::ndarray_ext::ScalarOperand;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, HashSet};

pub mod exec;
pub mod fusion;
mod passes;

#[cfg(test)]
mod tests;

pub use exec::PlanExecution;
pub use fusion::{
    compile_fused_plan, compile_plan, detect_fusion_patterns, FusedOperation, FusionConfig,
    FusionMatch, FusionPlan, FusionStats, PlanStep,
};
pub use passes::{
    common_subexpression_elimination, constant_folding, constant_folding_with_threshold,
    dead_code_elimination_on_graph, optimize_graph, ConstFoldStats, CseStats, DceStats,
    PipelineStats, DEFAULT_CONST_FOLD_ELEMENT_LIMIT,
};

/// Optimization passes that can be applied *in place* to a computation graph.
///
/// Operation fusion is deliberately **not** a member of this enum: it cannot be
/// performed in place (see the module docs). Use [`compile_fused_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Deduplicate structurally identical sub-expressions.
    CommonSubexpressionElimination,
    /// Remove nodes that don't contribute to outputs.
    DeadCodeElimination,
    /// Pre-compute operations on constant values.
    ConstantFolding,
    /// All in-place passes.
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

/// Statistics about the in-place optimization passes.
///
/// Every field is produced by the pass that names it. Fusion counts live on
/// [`FusionStats`], because fusion is a separate (plan-compiling) pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OptimizationStats {
    /// Number of nodes before optimization
    pub nodes_before: usize,
    /// Number of nodes after optimization
    pub nodes_after: usize,
    /// Number of duplicate nodes removed by common subexpression elimination
    pub cse_nodes_eliminated: usize,
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
        writeln!(f, "  CSE nodes eliminated: {}", self.cse_nodes_eliminated)?;
        writeln!(f, "  Dead nodes removed: {}", self.dead_nodes_removed)?;
        writeln!(f, "  Constants folded: {}", self.constants_folded)?;
        writeln!(f, "  Iterations: {}", self.iterations)?;
        Ok(())
    }
}

/// Graph optimizer for applying in-place optimization passes.
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
    ///
    /// This method performs **no operation fusion** — fusion is not
    /// representable in [`Operation`], see [`compile_fused_plan`].
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
                .contains(&OptimizationPass::CommonSubexpressionElimination);
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

        let mut cse_nodes_eliminated = 0usize;
        let mut constants_folded = 0usize;
        let mut dead_nodes_removed = 0usize;
        let mut iterations = 0usize;

        for _ in 0..max_iters {
            let n_before = graph.num_nodes();

            if run_cse {
                let s = common_subexpression_elimination(graph);
                cse_nodes_eliminated += s.nodes_eliminated;
            }
            if run_fold {
                let s = constant_folding(graph);
                constants_folded += s.nodes_folded;
            }
            if run_cse {
                let s = common_subexpression_elimination(graph);
                cse_nodes_eliminated += s.nodes_eliminated;
            }
            if run_dce {
                // Identify sink nodes (no consumers) as implicit output roots.
                let ops = graph.snapshot_ops();
                let mut has_consumer: HashSet<NodeId> = HashSet::new();
                for (_id, op) in &ops {
                    for p in operation_inputs(op) {
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

        if self.config.verbose {
            eprintln!(
                "graph_optimizer: {} iterations, {} -> {} nodes",
                iterations,
                nodes_before,
                graph.num_nodes()
            );
        }

        Ok(OptimizationStats {
            nodes_before,
            nodes_after: graph.num_nodes(),
            cse_nodes_eliminated,
            dead_nodes_removed,
            constants_folded,
            iterations,
        })
    }

    /// Perform dead code elimination on a set of operations.
    ///
    /// Pure analysis: returns the node IDs that are not reachable from
    /// `output_nodes`. Does not mutate a graph; see
    /// [`dead_code_elimination_on_graph`] for the mutating variant.
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
                for parent in operation_inputs(op) {
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

    /// Estimate the memory freed by the in-place passes, given an average
    /// tensor size.
    ///
    /// This is an *estimate*: it multiplies the number of removed nodes by the
    /// caller-supplied average tensor size. For an exact, measured figure use
    /// [`PlanExecution::elements_allocated`] on a fused versus unfused plan.
    pub fn estimate_memory_savings(
        &self,
        stats: &OptimizationStats,
        avg_tensor_size_bytes: usize,
    ) -> usize {
        let nodes_removed = stats.nodes_before.saturating_sub(stats.nodes_after);
        nodes_removed * avg_tensor_size_bytes
    }
}

/// Return the input [`NodeId`]s of an operation.
pub(crate) fn operation_inputs(op: &Operation) -> Vec<NodeId> {
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

/// Compute a topological ordering from a snapshot so that children are
/// always visited after their parents. Kahn-style algorithm.
pub(crate) fn topological_order_from_snapshot(snapshot: &[(NodeId, Operation)]) -> Vec<NodeId> {
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
