//! # tenrso-planner
//!
//! Contraction order planning and optimization for TenRSo.
//!
//! **Version:** 0.1.0-alpha.2
//! **Tests:** 271 passing (264 + 7 ignored) - 100%
//! **Status:** M4 Complete with advanced enhancements and production features
//!
//! This crate provides algorithms and tools for finding efficient execution plans
//! for tensor network contractions, particularly those expressed in Einstein summation notation.
//!
//! ## Features
//!
//! - **6 Planning Algorithms**: Greedy, DP, Beam Search, Simulated Annealing, Genetic Algorithm, Adaptive
//! - **Parallel Planning**: Ensemble planner runs multiple algorithms concurrently for best quality
//! - **ML-Based Calibration**: Learn from execution history to improve cost predictions over time
//! - **Cost Estimation**: FLOPs and memory prediction for dense and sparse tensors
//! - **Plan Caching**: LRU/LFU/ARC eviction policies for memoization
//! - **Execution Simulation**: Hardware-specific cost modeling (CPU + 6 GPU models with Tensor Cores)
//! - **Quality Tracking**: Real-time execution feedback and adaptive tuning
//! - **Representation Selection**: Automatic choice between dense, sparse, and low-rank formats
//! - **Tiling and Blocking**: Multi-level cache-aware strategies for efficient memory access
//! - **Production Features**: Profiling, validation, visualization, serialization
//! - **SciRS2-Core Integration**: Professional-grade RNG for reproducible stochastic algorithms
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
//! TenRSo-Planner provides **6 production-grade planning algorithms**:
//!
//! ### 1. Adaptive Planner (⭐ Recommended)
//!
//! The [`AdaptivePlanner`] automatically selects the best algorithm based on problem size.
//! - Small networks (n ≤ 5): Uses DP (optimal)
//! - Medium networks (5 < n ≤ 20): Uses Beam Search or DP
//! - Large networks (n > 20): Uses Greedy, SA, or GA based on time budget
//!
//! **Use when:** You want optimal results without manual algorithm selection
//!
//! ### 2. Greedy Planner
//!
//! The [`greedy_planner`] repeatedly contracts the pair of tensors with minimum cost
//! at each step. Fast (O(n³)) and produces good results for most tensor networks.
//!
//! **Use when:** Planning time is critical, many tensors (> 10)
//!
//! ### 3. Beam Search Planner
//!
//! The [`BeamSearchPlanner`] explores multiple paths simultaneously. Better quality
//! than greedy while remaining tractable.
//!
//! **Use when:** Medium networks (8-20 tensors), want better than greedy quality
//!
//! ### 4. Dynamic Programming Planner
//!
//! The [`dp_planner`] uses bitmask DP to find globally optimal contraction order.
//! More expensive (O(3ⁿ) time, O(2ⁿ) space) but guarantees optimality.
//!
//! **Use when:** Few tensors (≤ 20), need provably optimal plans
//!
//! ### 5. Simulated Annealing Planner
//!
//! The [`SimulatedAnnealingPlanner`] uses stochastic search to escape local minima.
//!
//! **Use when:** Large networks (> 20), quality more important than planning speed
//!
//! ### 6. Genetic Algorithm Planner
//!
//! The [`GeneticAlgorithmPlanner`] uses evolutionary search with population-based
//! exploration, crossover, and mutation to find high-quality contraction orders.
//!
//! **Use when:** Very large networks (> 20), need best quality with enough time budget
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
//! - **Adaptive**: Auto-selects best algorithm (varies)
//! - **Greedy**: ~1ms for 10 tensors, ~10ms for 100 tensors
//! - **Beam Search (k=5)**: ~5ms for 10 tensors, ~50ms for 100 tensors
//! - **DP**: ~1ms for 5 tensors, ~100ms for 10 tensors, ~10s for 15 tensors
//! - **Simulated Annealing**: ~10-100ms (configurable iterations)
//! - **Genetic Algorithm**: ~50-500ms (depends on population and generations)
//!
//! ## References
//!
//! - Matrix chain multiplication: Cormen et al., "Introduction to Algorithms"
//! - Tensor network contraction: Pfeifer et al., "Faster identification of optimal contraction sequences" (2014)
//! - Einsum optimization: opt_einsum library (Python)

#![deny(warnings)]

pub mod api;
pub mod cache;
pub mod cost;
pub mod ml_cost;
pub mod order;
pub mod parallel;
pub mod parser;
pub mod profiling;
#[cfg(test)]
mod property_tests;
pub mod quality;
pub mod repr;
pub mod simulator;
pub mod tiling;

// Re-exports
pub use api::*;
pub use cache::*;
pub use cost::*;
pub use ml_cost::*;
pub use order::{
    beam_search_planner, dp_planner, genetic_algorithm_planner, greedy_planner, refine_plan,
    simulated_annealing_planner, AdaptivePlanner, BeamSearchPlanner, DPPlanner,
    GeneticAlgorithmPlanner, GreedyPlanner, SimulatedAnnealingPlanner,
};
pub use parallel::*;
pub use parser::*;
pub use profiling::*;
pub use quality::*;
pub use repr::*;
pub use simulator::*;
pub use tiling::*;
