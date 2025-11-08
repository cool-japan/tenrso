//! # tenrso-core
//!
//! Core tensor types, axis metadata, views, and basic operations for TenRSo.
//!
//! This crate provides the foundational building blocks for tensor computing in the TenRSo stack:
//!
//! - **Dense tensor representation** ([`DenseND`]) with views and strides
//! - **Axis metadata** ([`AxisMeta`]) with symbolic names for better debugging
//! - **Shape operations** (reshape, permute) for tensor manipulation
//! - **Matricization** (unfold/fold) for decomposition algorithms
//! - **Unified tensor handle** ([`TensorHandle`]) supporting multiple representations
//!
//! ## Core Principles
//!
//! ### SciRS2 Integration
//!
//! **CRITICAL:** This crate uses `scirs2-core` for all scientific computing operations.
//! Direct use of `ndarray`, `rand`, or `num-traits` is forbidden. See `SCIRS2_INTEGRATION_POLICY.md`.
//!
//! ### Memory Layout
//!
//! Tensors default to C-contiguous (row-major) layout for cache efficiency.
//! Operations like reshape and permute are zero-copy when possible.
//!
//! ### Safety
//!
//! All indexing is bounds-checked. No unsafe code or undefined behavior.
//!
//! ## Quick Start
//!
//! ```
//! use tenrso_core::{DenseND, TensorHandle, AxisMeta};
//!
//! // Create a 3D tensor of zeros
//! let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
//! assert_eq!(tensor.shape(), &[2, 3, 4]);
//! assert_eq!(tensor.rank(), 3);
//!
//! // Create with axis metadata
//! let axes = vec![
//!     AxisMeta::new("batch", 2),
//!     AxisMeta::new("height", 3),
//!     AxisMeta::new("width", 4),
//! ];
//! let handle = TensorHandle::from_dense(tensor, axes);
//! ```
//!
//! ## Creating Tensors
//!
//! Various initialization methods are provided:
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! // Zeros
//! let zeros = DenseND::<f64>::zeros(&[2, 3]);
//!
//! // Ones
//! let ones = DenseND::<f64>::ones(&[2, 3]);
//!
//! // Fill with value
//! let fives = DenseND::from_elem(&[2, 3], 5.0);
//!
//! // From vec (row-major order)
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let tensor = DenseND::from_vec(data, &[2, 2]).unwrap();
//!
//! // Random initialization
//! let uniform = DenseND::<f64>::random_uniform(&[2, 3], 0.0, 1.0);
//! let normal = DenseND::<f64>::random_normal(&[2, 3], 0.0, 1.0);
//! ```
//!
//! ## Shape Operations
//!
//! Tensors support various shape manipulation operations:
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
//!
//! // Reshape (zero-copy when contiguous)
//! let reshaped = tensor.reshape(&[6, 4]).unwrap();
//! assert_eq!(reshaped.shape(), &[6, 4]);
//!
//! // Permute axes (transpose generalization)
//! let permuted = tensor.permute(&[2, 0, 1]).unwrap();
//! assert_eq!(permuted.shape(), &[4, 2, 3]);
//! ```
//!
//! ## Matricization (Unfold/Fold)
//!
//! Critical operations for tensor decompositions:
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
//!
//! // Unfold along mode 1 (matricization)
//! let unfolded = tensor.unfold(1).unwrap();
//! assert_eq!(unfolded.shape(), &[3, 8]); // 8 = 2 * 4
//!
//! // Fold back to original shape
//! let folded = DenseND::fold(&unfolded, &[2, 3, 4], 1).unwrap();
//! assert_eq!(folded.shape(), &[2, 3, 4]);
//! ```
//!
//! ## Views and Zero-Copy Operations
//!
//! Efficient memory access without copying:
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! let mut tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
//!
//! // Immutable view
//! let view = tensor.view();
//!
//! // Mutable view for in-place operations
//! let mut view_mut = tensor.view_mut();
//! ```
//!
//! ## Indexing
//!
//! Direct element access with bounds checking:
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! let mut tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
//!
//! // Read element
//! let value = tensor[&[0, 1, 2]];
//!
//! // Write element
//! tensor[&[0, 1, 2]] = 42.0;
//! assert_eq!(tensor[&[0, 1, 2]], 42.0);
//! ```
//!
//! ## Performance Considerations
//!
//! - **Contiguous memory:** Operations prefer C-contiguous layout for cache efficiency
//! - **Zero-copy:** Reshape and permute avoid copying when possible
//! - **SIMD:** Uses scirs2-core's SIMD-accelerated operations
//! - **Parallel:** Enable `parallel` feature for multi-threaded operations
//!
//! ## Error Handling
//!
//! Operations return `Result<T, anyhow::Error>` for proper error propagation:
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! let tensor = DenseND::<f64>::zeros(&[2, 3]);
//!
//! // This will fail - incompatible size
//! let result = tensor.reshape(&[7]);
//! assert!(result.is_err());
//!
//! // This will fail - invalid mode
//! let result = tensor.unfold(10);
//! assert!(result.is_err());
//! ```
//!
//! ## Integration with Other Crates
//!
//! - **tenrso-kernels:** Uses `DenseND` for Khatri-Rao, MTTKRP, etc.
//! - **tenrso-decomp:** Uses unfold/fold for CP-ALS, Tucker-HOOI, TT-SVD
//! - **tenrso-sparse:** Interoperates via `TensorHandle` unified API
//! - **tenrso-exec:** Orchestrates operations across different representations
//!
//! ## Features
//!
//! - `parallel`: Enable parallel operations via Rayon (through scirs2-core)
//! - `serde`: Enable serialization/deserialization support

#![deny(warnings)]

pub mod dense;
pub mod ops;
pub mod types;

#[cfg(test)]
mod property_tests;

pub use types::{Axis, AxisMeta, DenseND, Rank, Shape, TensorHandle, TensorRepr};
