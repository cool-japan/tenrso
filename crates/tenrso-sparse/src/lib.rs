//! # tenrso-sparse
//!
//! Sparse tensor formats and operations for TenRSo.
//!
//! This crate provides:
//! - COO (Coordinate) format
//! - CSR/CSC (Compressed Sparse Row/Column) format
//! - BCSR (Block Compressed Sparse Row) format
//! - CSF/HiCOO (Compressed Sparse Fiber/Hierarchical COO) - feature-gated
//! - SpMM/SpSpMM operations
//! - Masked einsum

#![deny(warnings)]

pub mod bcsr;
pub mod coo;
pub mod csc;
#[cfg(feature = "csf")]
pub mod csf;
pub mod csr;
pub mod error;
#[cfg(feature = "csf")]
pub mod hicoo;
pub mod mask;
pub mod ops;
pub mod reductions;
pub mod utils;

// Re-exports
pub use bcsr::*;
pub use coo::*;
pub use csc::*;
#[cfg(feature = "csf")]
pub use csf::*;
pub use csr::*;
pub use error::*;
#[cfg(feature = "csf")]
pub use hicoo::*;
pub use mask::*;
