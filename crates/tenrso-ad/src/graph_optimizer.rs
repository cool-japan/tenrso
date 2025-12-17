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
use std::collections::HashSet;

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

    /// Optimize a computation graph (placeholder - returns original graph)
    ///
    /// Note: Full optimization requires deep integration with graph internals.
    /// This is a foundation for future optimization work.
    pub fn optimize<T: Float + ScalarOperand + FromPrimitive>(
        &self,
        _graph: &ComputationGraph<T>,
    ) -> Result<OptimizationStats> {
        // For now, return empty stats as optimization requires graph refactoring
        // to expose internal structure for modification
        Ok(OptimizationStats {
            nodes_before: 0,
            nodes_after: 0,
            fusions_applied: 0,
            dead_nodes_removed: 0,
            constants_folded: 0,
            iterations: 0,
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
}
