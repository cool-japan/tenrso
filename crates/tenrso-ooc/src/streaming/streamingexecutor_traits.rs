//! # StreamingExecutor - Trait Implementations
//!
//! This module contains trait implementations for `StreamingExecutor`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StreamingExecutor;

impl Clone for StreamingExecutor {
    fn clone(&self) -> Self {
        Self::new(self.config.clone())
    }
}
