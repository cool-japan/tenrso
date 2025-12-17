//! Auto-generated module structure

pub mod advanced_indexing;
pub mod cpuexecutor_traits;
pub mod custom_ops;
pub mod functions;
#[cfg(test)]
mod functions_tests;
pub mod optimized_ops;
pub mod parallel;
pub mod pool_heuristics;
pub mod pooled_ops;
pub mod simd_ops;
pub mod thread_local_pool;
pub mod tiled_reductions;
pub mod types;
pub mod vectorized_broadcast;

// Re-export all types
pub use custom_ops::{apply_custom_unary, custom_binary_op, custom_reduce, custom_unary_op};
pub use functions::*;
pub use pool_heuristics::{AccessPatternTracker, PoolingPolicy, PoolingRecommender, PoolingReport};
pub use thread_local_pool::{AggregatedPoolStats, ThreadLocalPoolManager, ThreadLocalPoolStats};
pub use types::*;
