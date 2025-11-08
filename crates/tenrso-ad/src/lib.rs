//! # tenrso-ad
//!
//! Automatic differentiation support for TenRSo.
//!
//! This crate provides:
//! - Custom VJP (Vector-Jacobian Product) rules for tensor contractions
//! - Gradient rules for decompositions (CP-ALS, Tucker-HOOI, TT-SVD)
//! - Integration hooks for external AD frameworks
//! - Efficient backward passes avoiding AD tape blow-up

#![deny(warnings)]
#![recursion_limit = "4096"]

pub mod grad;
pub mod gradcheck;
pub mod hooks;
pub mod vjp;

// Re-exports
pub use vjp::*;
