//! # tenrso-kernels
//!
//! High-performance tensor kernel operations for TenRSo.
//!
//! **Version:** 0.1.0-alpha.2
//! **Tests:** 138 passing (100%)
//! **Status:** Production-ready with comprehensive statistical toolkit
//!
//! ## Overview
//!
//! This crate provides optimized implementations of fundamental tensor operations
//! used in tensor decompositions (CP-ALS, Tucker, TT) and tensor computations.
//!
//! **Key Features:**
//! - ✅ **Khatri-Rao product** - Column-wise Kronecker product (serial & parallel)
//! - ✅ **Kronecker product** - Tensor product of matrices (serial & parallel)
//! - ✅ **Hadamard product** - Element-wise multiplication (allocating & in-place)
//! - ✅ **N-mode products** - Tensor-matrix multiplication along any mode
//! - ✅ **Tensor-Tensor Product (TTT)** - General tensor contraction operation
//! - ✅ **MTTKRP** - Core CP-ALS kernel (standard, blocked, fused, parallel variants)
//! - ✅ **Outer products** - Tensor construction from vectors
//! - ✅ **Tucker operator** - Multi-mode products with automatic optimization
//! - ✅ **Tensor Train (TT) operations** - TT orthogonalization, norm, dot product
//! - ✅ **Blocked/tiled operations** - Cache-efficient implementations
//! - ✅ **Tensor contractions** - Generalized tensor contraction primitives
//! - ✅ **Tensor reductions** - Sum, mean, variance, std, norms, percentiles, median, skewness, kurtosis, covariance, correlation
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_core::ndarray_ext::{Array, Array2};
//! use tenrso_core::DenseND;
//! use tenrso_kernels::{khatri_rao, mttkrp, nmode_product};
//!
//! // Khatri-Rao product (for CP decomposition)
//! let a = Array2::<f64>::ones((10, 5));
//! let b = Array2::<f64>::ones((8, 5));
//! let kr = khatri_rao(&a.view(), &b.view());
//! assert_eq!(kr.shape(), &[80, 5]);
//!
//! // N-mode product (tensor-matrix multiplication)
//! let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
//! let matrix = Array2::<f64>::ones((2, 3));
//! let result = nmode_product(&tensor.view(), &matrix.view(), 0).unwrap();
//! assert_eq!(result.shape(), &[2, 4, 5]); // mode-0 changed from 3 to 2
//!
//! // MTTKRP (core of CP-ALS)
//! let factors = vec![
//!     Array2::<f64>::ones((3, 2)),
//!     Array2::<f64>::ones((4, 2)),
//!     Array2::<f64>::ones((5, 2)),
//! ];
//! let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
//! let mttkrp_result = mttkrp(&tensor.view(), &factor_views, 1).unwrap();
//! assert_eq!(mttkrp_result.shape(), &[4, 2]);
//! ```
//!
//! ## Performance
//!
//! All operations are highly optimized with:
//! - **SIMD acceleration** via scirs2_core
//! - **Parallel execution** for large problems (feature-gated)
//! - **Cache-efficient tiling** for MTTKRP
//! - **Zero-copy views** to minimize allocations
//!
//! Typical performance (see `PERFORMANCE.md` for details):
//! - Khatri-Rao: **1.5 Gelem/s** (serial), **3× speedup** (parallel)
//! - MTTKRP: **13.3 Gelem/s** (blocked parallel)
//! - N-mode: **>5 Gelem/s** sustained
//! - Hadamard in-place: **11 Gelem/s**, **2.7× faster** than allocating
//!
//! ## Usage Recommendations
//!
//! | Operation | When to Use Parallel | Notes |
//! |-----------|---------------------|-------|
//! | `khatri_rao_parallel` | Matrices ≥200 rows | 2-3× speedup |
//! | `kronecker_parallel` | Rarely beneficial | Use serial version |
//! | `hadamard_inplace` | Always | 2-3× faster than allocating |
//! | `mttkrp_blocked` | Tensors ≥20³ | Cache-efficient |
//! | `mttkrp_blocked_parallel` | Tensors ≥30³ | 4-5× speedup |
//!
//! ## Examples
//!
//! The `examples/` directory contains comprehensive demonstrations:
//! - `khatri_rao.rs` - Khatri-Rao product with parallel speedup measurements
//! - `mttkrp_cp.rs` - CP-ALS iteration and MTTKRP variants
//! - `nmode_tucker.rs` - Tucker decomposition and compression
//!
//! Run with:
//! ```bash
//! cargo run --example khatri_rao --features parallel
//! cargo run --example mttkrp_cp --features parallel
//! cargo run --example nmode_tucker
//! ```
//!
//! ## Features
//!
//! - `parallel` (default) - Enable parallel implementations using rayon
//!
//! ## SciRS2 Integration
//!
//! This crate uses `scirs2-core` for all array operations and numerical computations.
//! Direct use of `ndarray`, `rand`, or `num-traits` is not permitted.
//! See `SCIRS2_INTEGRATION_POLICY.md` for details.

#![deny(warnings)]

pub mod contractions;
pub mod error;
pub mod hadamard;
pub mod khatri_rao;
pub mod kronecker;
pub mod mttkrp;
pub mod nmode;
pub mod outer;
pub mod randomized;
pub mod reductions;
pub mod tt_ops;
pub mod utils;

#[cfg(test)]
mod property_tests;

// Re-exports
pub use contractions::*;
pub use error::{KernelError, KernelResult};
pub use hadamard::*;
pub use khatri_rao::*;
pub use kronecker::*;
pub use mttkrp::*;
pub use nmode::*;
pub use outer::*;
pub use randomized::*;
pub use reductions::*;
pub use tt_ops::*;
pub use utils::*;
