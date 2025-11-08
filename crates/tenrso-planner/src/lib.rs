//! # tenrso-planner
//!
//! Contraction order planning and optimization for TenRSo.
//!
//! This crate provides algorithms and tools for finding efficient execution plans
//! for tensor network contractions, particularly those expressed in Einstein summation notation.
//!
//! ## Features
//!
//! - **Contraction Order Search**: Greedy and dynamic programming algorithms for finding
//!   optimal or near-optimal contraction sequences
//! - **Cost Estimation**: FLOPs and memory prediction for dense and sparse tensors
//! - **Representation Selection**: Automatic choice between dense, sparse, and low-rank formats
//! - **Tiling and Blocking**: Cache-aware strategies for efficient memory access
//! - **Planning Hints**: User control over memory budgets, sparsity, and device targeting
//!
//! ## Quick Start
//!
//! ```
//! use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};
//!
//! // Parse Einstein summation notation
//! let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
//!
//! // Define tensor shapes
//! let shapes = vec![vec![100, 200], vec![200, 300]];
//!
//! // Create a plan with default hints
//! let hints = PlanHints::default();
//! let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
//!
//! // Inspect plan details
//! println!("Estimated FLOPs: {:.2e}", plan.estimated_flops);
//! println!("Peak memory: {} bytes", plan.estimated_memory);
//! ```
//!
//! ## Planners
//!
//! ### Greedy Planner
//!
//! The [`greedy_planner`] repeatedly contracts the pair of tensors with minimum cost
//! at each step. This is fast (O(n³) where n is the number of inputs) and produces
//! good results for most tensor networks.
//!
//! **Use when:**
//! - You have many tensors (> 10)
//! - Planning time is critical
//! - You need predictable performance
//!
//! ### Dynamic Programming Planner
//!
//! The [`dp_planner`] uses bitmask DP to find the globally optimal contraction order.
//! This is more expensive (O(3ⁿ) time, O(2ⁿ) space) but guarantees optimality for
//! small tensor networks.
//!
//! **Use when:**
//! - You have few tensors (≤ 20)
//! - Execution time dominates planning time
//! - You need provably optimal plans
//!
//! ## Cost Model
//!
//! The planner estimates costs using:
//! - **FLOPs**: Multiply-add operations (2 FLOPs per element)
//! - **Memory**: Bytes for intermediate results and outputs
//! - **Sparsity**: Density-based predictions for sparse operations
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive usage demonstrations:
//! - `basic_matmul.rs`: Simple matrix multiplication planning
//! - `matrix_chain.rs`: Comparing greedy vs DP for chain contractions
//! - `planning_hints.rs`: Using hints to control optimization
//!
//! ## Performance
//!
//! Planning overhead is typically negligible compared to execution:
//! - Greedy: ~1ms for 10 tensors, ~10ms for 100 tensors
//! - DP: ~1ms for 5 tensors, ~100ms for 10 tensors, ~10s for 15 tensors
//!
//! ## References
//!
//! - Matrix chain multiplication: Cormen et al., "Introduction to Algorithms"
//! - Tensor network contraction: Pfeifer et al., "Faster identification of optimal contraction sequences" (2014)
//! - Einsum optimization: opt_einsum library (Python)

#![deny(warnings)]

pub mod api;
pub mod cost;
pub mod order;
pub mod parser;
pub mod repr;
pub mod tiling;

// Re-exports
pub use api::*;
pub use cost::*;
pub use order::*;
pub use parser::*;
pub use repr::*;
pub use tiling::*;
