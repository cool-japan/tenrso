//! Dense tensor implementation and operations
//!
//! This module provides a comprehensive dense tensor implementation organized into
//! functional sub-modules for better maintainability.

// Core type definition
pub mod types;

// Operation modules (organized by functionality)
mod algebra;
mod combining;
mod comparison;
pub mod convolution;
mod creation;
mod elementwise;
mod fft;
mod indexing;
mod linalg_advanced;
mod manipulation;
mod shape_ops;
mod statistics;

// Supporting modules
pub mod densend_traits;
pub(crate) mod functions;

// Re-export the main type
pub use types::DenseND;

// Re-export convolution types
pub use convolution::ConvPadding;
