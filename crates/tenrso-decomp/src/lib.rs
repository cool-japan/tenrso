//! # tenrso-decomp - Tensor Decomposition Methods
//!
//! Production-grade tensor decomposition algorithms for scientific computing,
//! machine learning, and data analysis.
//!
//! **Version:** 0.1.0-alpha.2
//! **Tests:** 158 passing (100%)
//! **Status:** M2 Complete - All algorithms implemented and tested
//!
//! ## Overview
//!
//! This crate provides high-performance implementations of three major tensor
//! decomposition families:
//!
//! ### CP Decomposition (Canonical Polyadic / CANDECOMP/PARAFAC)
//!
//! Factorizes a tensor into a sum of rank-1 components:
//!
//! ```text
//! X ≈ Σᵣ λᵣ (a₁ᵣ ⊗ a₂ᵣ ⊗ ... ⊗ aₙᵣ)
//! ```
//!
//! **Use cases:**
//! - Factor analysis and dimensionality reduction
//! - Signal separation and blind source separation
//! - Chemometrics and spectroscopy
//! - Neuroscience (EEG/fMRI analysis)
//!
//! **Algorithms:**
//! - `cp_als`: Alternating least squares with convergence detection
//! - `cp_als_constrained`: With non-negativity, L2 regularization, orthogonality
//! - `cp_als_accelerated`: Line search optimization for faster convergence
//! - `cp_randomized`: Randomized sketching for large-scale tensors (NEW!)
//! - `cp_completion`: Tensor completion with missing data (CP-WOPT)
//!
//! ### Tucker Decomposition (Higher-Order SVD)
//!
//! Factorizes a tensor into a core tensor and orthogonal factor matrices:
//!
//! ```text
//! X ≈ G ×₁ U₁ ×₂ U₂ ×₃ ... ×ₙ Uₙ
//! ```
//!
//! **Use cases:**
//! - Image/video compression
//! - Feature extraction from multi-way data
//! - Gait recognition and motion analysis
//! - Hyperspectral imaging
//!
//! **Algorithms:**
//! - `tucker_hosvd`: Fast one-pass SVD-based decomposition
//! - `tucker_hooi`: Iterative refinement for better approximation
//! - `tucker_hosvd_auto`: Automatic rank selection (NEW!)
//!   - Energy-based: Preserve X% of singular value energy
//!   - Threshold-based: Keep σ > threshold × σ_max
//!
//! ### Tensor Train (TT) Decomposition
//!
//! Represents a tensor as a sequence of 3-way cores:
//!
//! ```text
//! X(i₁,...,iₙ) = G₁[i₁] × G₂[i₂] × ... × Gₙ[iₙ]
//! ```
//!
//! **Use cases:**
//! - High-order tensor compression (6D+)
//! - Quantum many-body systems
//! - Stochastic PDEs
//! - Tensor networks in machine learning
//!
//! **Algorithms:**
//! - `tt_svd`: Sequential SVD with rank truncation
//! - `tt_round`: Post-decomposition rank reduction
//!
//! **TT Operations (NEW!):**
//! - `tt_add`: Addition of two TT decompositions
//! - `tt_dot`: Inner product without reconstruction
//! - `tt_hadamard`: Element-wise (Hadamard) product
//!
//! ## Quick Start
//!
//! ### CP Decomposition
//!
//! ```
//! use tenrso_core::DenseND;
//! use tenrso_decomp::{cp_als, InitStrategy};
//!
//! // Create a 50×50×50 tensor
//! let tensor = DenseND::<f64>::random_uniform(&[50, 50, 50], 0.0, 1.0);
//!
//! // Decompose into rank-10 CP
//! let cp = cp_als(&tensor, 10, 100, 1e-4, InitStrategy::Random, None)?;
//!
//! println!("Converged in {} iterations", cp.iters);
//! println!("Final fit: {:.4}", cp.fit);
//!
//! // Reconstruct approximation
//! let approx = cp.reconstruct(tensor.shape())?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ### Tucker Decomposition
//!
//! ```
//! use tenrso_core::DenseND;
//! use tenrso_decomp::tucker_hosvd;
//!
//! let tensor = DenseND::<f64>::random_uniform(&[30, 30, 30], 0.0, 1.0);
//! let ranks = vec![15, 15, 15];
//!
//! // Tucker-HOSVD decomposition
//! let tucker = tucker_hosvd(&tensor, &ranks)?;
//!
//! println!("Core shape: {:?}", tucker.core.shape());
//! println!("Compression: {:.2}x", tucker.compression_ratio());
//!
//! let approx = tucker.reconstruct()?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ### Tensor Completion (NEW!)
//!
//! ```
//! use scirs2_core::ndarray_ext::Array;
//! use tenrso_core::DenseND;
//! use tenrso_decomp::{cp_completion, InitStrategy};
//!
//! // Create tensor with missing entries
//! let mut data = Array::<f64, _>::zeros(vec![20, 20, 20]);
//! let mut mask = Array::<f64, _>::zeros(vec![20, 20, 20]);
//!
//! // Mark some entries as observed (1 = observed, 0 = missing)
//! for i in 0..10 {
//!     for j in 0..10 {
//!         for k in 0..10 {
//!             data[[i, j, k]] = (i + j + k) as f64 * 0.1;
//!             mask[[i, j, k]] = 1.0;
//!         }
//!     }
//! }
//!
//! let tensor = DenseND::from_array(data.into_dyn());
//! let mask_tensor = DenseND::from_array(mask.into_dyn());
//!
//! // Complete the tensor (predict missing values)
//! let cp = cp_completion(&tensor, &mask_tensor, 5, 100, 1e-4, InitStrategy::Random)?;
//!
//! // Get predictions for all entries (including missing ones)
//! let completed = cp.reconstruct(tensor.shape())?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ### Tensor Train Decomposition
//!
//! ```
//! use tenrso_core::DenseND;
//! use tenrso_decomp::tt::{tt_svd, tt_round};
//!
//! let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6, 6, 6], 0.0, 1.0);
//!
//! // TT-SVD with max ranks [5, 5, 5, 5]
//! let tt = tt_svd(&tensor, &[5, 5, 5, 5], 1e-6)?;
//! println!("TT-ranks: {:?}", tt.ranks);
//!
//! // Round to smaller ranks
//! let tt_small = tt_round(&tt, &[3, 3, 3, 3], 1e-4)?;
//! println!("Reduced ranks: {:?}", tt_small.ranks);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Feature Flags
//!
//! Currently all features are enabled by default. Future versions may add:
//! - `parallel`: Parallel tensor operations via Rayon
//! - `gpu`: GPU acceleration via cuBLAS/ROCm
//!
//! ## SciRS2 Integration
//!
//! All linear algebra operations use `scirs2_linalg` for SVD, QR, and least-squares.
//! Random number generation uses `scirs2_core::random`.
//! Direct use of `ndarray` or `rand` is forbidden per project policy.
//!
//! ## Performance
//!
//! Typical performance on modern CPUs (single-threaded):
//! - **CP-ALS**: 256³ tensor, rank 64, 10 iters → ~2s
//! - **Tucker-HOOI**: 512×512×128, ranks \[64,64,32\], 10 iters → ~3s
//! - **TT-SVD**: 32⁶ tensor, ε=1e-6 → ~2s
//!
//! ## References
//!
//! - Kolda & Bader (2009), "Tensor Decompositions and Applications"
//! - De Lathauwer et al. (2000), "Multilinear Singular Value Decomposition"
//! - Oseledets (2011), "Tensor-Train Decomposition"
//! - Vervliet et al. (2016), "Tensorlab 3.0"

#![deny(warnings)]

pub mod cp;
pub mod rank_selection;
pub mod tt;
pub mod tucker;
pub mod utils;

#[cfg(test)]
mod property_tests;

// Re-exports
pub use cp::*;
pub use rank_selection::*;
pub use tt::*;
pub use tucker::*;
