//! # StreamConfig - Trait Implementations
//!
//! This module contains trait implementations for `StreamConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StreamConfig;
use crate::prefetch::PrefetchStrategy;

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024,
            default_chunk_size: vec![256],
            temp_dir: std::env::temp_dir(),
            enable_spill: true,
            enable_profiling: false,
            enable_prefetching: false,
            prefetch_strategy: PrefetchStrategy::Sequential,
            prefetch_queue_size: 4,
            enable_parallel: true,
            num_threads: 0,
            min_chunks_for_parallel: 4,
        }
    }
}
