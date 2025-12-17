//! # DenseND - Trait Implementations
//!
//! This module contains trait implementations for `DenseND`.
//!
//! ## Implemented Traits
//!
//! - `Index`
//! - `IndexMut`
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DenseND;
use scirs2_core::ndarray_ext::IxDyn;
use scirs2_core::numeric::Num;
use std::fmt;

impl<T> std::ops::Index<&[usize]> for DenseND<T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.data[IxDyn(index)]
    }
}

impl<T> std::ops::IndexMut<&[usize]> for DenseND<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        &mut self.data[IxDyn(index)]
    }
}

impl<T: fmt::Debug + Clone + Num> fmt::Debug for DenseND<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseND")
            .field("shape", &self.shape())
            .field("rank", &self.rank())
            .field("data", &self.data)
            .finish()
    }
}
