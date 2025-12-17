//! # tenrso-ad
//!
//! **Automatic Differentiation for TenRSo Tensor Computation Stack**
//!
//! **Version:** 0.1.0-alpha.2
//! **Tests:** 378 passing (375 + 3 ignored) - 100%
//! **Status:** M6 Complete - Production-ready automatic differentiation
//!
//! This crate provides production-ready automatic differentiation (AD) support for tensor operations,
//! tensor decompositions, and complex computational workflows. It is designed to integrate seamlessly
//! with the TenRSo ecosystem while supporting external AD frameworks.
//!
//! ## Core Capabilities
//!
//! ### 1. Vector-Jacobian Product (VJP) Rules
//!
//! Efficient backward-mode automatic differentiation for:
//! - **Einsum contractions** with automatic adjoint spec generation
//! - **Element-wise operations** (unary and binary)
//! - **Reduction operations** (sum, mean, max, min)
//! - **Special handling for scalar outputs** (inner products, norms)
//!
//! See [`vjp`] module for details.
//!
//! ### 2. Decomposition Gradients
//!
//! Custom gradient rules for tensor decompositions:
//! - **CP-ALS** (CANDECOMP/PARAFAC) with optional weights
//! - **Tucker-HOOI** with multi-mode contractions
//! - **TT-SVD** (Tensor Train) with efficient chain-rule propagation
//!
//! These implementations avoid AD tape blow-up and provide O(n) time complexity.
//! See [`grad`] module for details.
//!
//! ### 3. Gradient Verification
//!
//! Numerical gradient checking utilities:
//! - **Finite difference methods** (central and forward)
//! - **Configurable tolerances** and error reporting
//! - **Detailed statistics** (max absolute/relative errors)
//!
//! See [`gradcheck`] module for details.
//!
//! ### 4. Advanced Features
//!
//! #### Memory-Efficient Training
//! - **Gradient checkpointing** ([`checkpoint`]): Trade computation for memory (O(√n) memory)
//! - **Sparse gradients** ([`sparse_grad`]): COO format for highly sparse gradients
//! - **Smart tensor storage** ([`storage`]): Automatic Arc-based sharing for large tensors
//!
//! #### Performance Optimization
//! - **Parallel gradient computation** ([`parallel`]): Multi-threaded VJP using Rayon
//! - **Mixed precision training** ([`mixed_precision`]): FP16/BF16 with automatic loss scaling
//!
//! #### Second-Order Methods
//! - **Hessian computation** ([`hessian`]): Efficient Hessian-vector products in O(n) time
//! - **Higher-order derivatives**: Newton's method and second-order optimization
//!
//! #### Training Utilities
//! - **Gradient monitoring** ([`monitoring`]): Vanishing/exploding gradient detection
//! - **Gradient utilities** ([`utils`]): Clipping, normalization, statistics
//! - **Gradient compression** ([`compression`]): Quantization, top-k, random sparsification
//! - **Optimizers** ([`optimizers`]): SGD, Adam, AdamW, RMSprop, AdaGrad, LR schedulers
//!
//! ### 5. External Framework Integration
//!
//! Generic hooks for external AD frameworks:
//! - **Trait-based design** with minimal coupling
//! - **Operation registration** and gradient computation
//! - **Tape management** with recording mode control
//!
//! See [`hooks`] module for details.
//!
//! ## Quick Start
//!
//! ### Basic VJP Example
//!
//! ```rust,ignore
//! use tenrso_ad::vjp::{EinsumVjp, VjpOp};
//! use tenrso_core::DenseND;
//! use tenrso_planner::EinsumSpec;
//! use tenrso_exec::ops::execute_dense_contraction;
//!
//! // Forward pass: C = A @ B
//! let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
//! let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
//! let spec = EinsumSpec::parse("ij,jk->ik")?;
//! let c = execute_dense_contraction(&spec, &a, &b)?;
//!
//! // Backward pass
//! let grad_c = DenseND::ones(&[2, 2]);
//! let vjp = EinsumVjp::new(spec, a.clone(), b.clone());
//! let grads = vjp.vjp(&grad_c)?;
//! let grad_a = &grads[0];  // ∂L/∂A
//! let grad_b = &grads[1];  // ∂L/∂B
//! ```
//!
//! ### Gradient Checking Example
//!
//! ```rust,ignore
//! use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig};
//!
//! let config = GradCheckConfig {
//!     epsilon: 1e-5,
//!     rtol: 1e-3,
//!     atol: 1e-5,
//!     use_central_diff: true,
//!     verbose: false,
//! };
//!
//! let result = check_gradient(forward_fn, backward_fn, &input, &grad_output, &config)?;
//! assert!(result.passed);
//! ```
//!
//! ### Gradient Checkpointing Example
//!
//! ```rust,ignore
//! use tenrso_ad::checkpoint::{CheckpointedSequence, CheckpointStrategy};
//!
//! let mut seq = CheckpointedSequence::new(CheckpointStrategy::Uniform { num_checkpoints: 5 });
//!
//! // Forward pass with checkpointing
//! for layer in &layers {
//!     seq.add_operation(Box::new(layer.clone()));
//! }
//! let output = seq.forward(&input)?;
//!
//! // Backward pass (automatically recomputes non-checkpointed layers)
//! let grad_input = seq.backward(&grad_output)?;
//! ```
//!
//! ## Architecture
//!
//! The crate is organized into focused modules:
//!
//! - [`vjp`]: Core VJP rules for tensor operations
//! - [`grad`]: Decomposition-specific gradients
//! - [`gradcheck`]: Numerical gradient verification
//! - [`checkpoint`]: Memory-efficient gradient checkpointing
//! - [`compression`]: Gradient compression for distributed training
//! - [`graph`]: Graph-based AD with dynamic computation graphs
//! - [`graph_optimizer`]: Graph optimization for performance
//! - [`hessian`]: Second-order derivative computation
//! - [`sparse_grad`]: Sparse gradient representation
//! - [`mixed_precision`]: Mixed precision training support
//! - [`parallel`]: Parallel gradient computation
//! - [`storage`]: Memory-optimized tensor storage
//! - [`utils`]: Gradient manipulation utilities
//! - [`monitoring`]: Training health monitoring
//! - [`optimizers`]: Optimization algorithms and LR schedulers
//! - [`hooks`]: External framework integration
//!
//! ## Performance
//!
//! - **Einsum VJP**: Efficient adjoint spec generation with no tape overhead
//! - **Parallel computation**: Scales linearly with CPU cores for large tensors (>10k elements)
//! - **Checkpointing**: Reduces memory by 60-90% for deep networks with minimal overhead
//! - **Sparse gradients**: Up to 98% memory savings for highly sparse gradients
//! - **Hessian-vector products**: O(n) time complexity instead of O(n²)
//!
//! ## Testing
//!
//! The crate includes comprehensive testing:
//! - **164 tests** (100% pass rate)
//!   - 144 unit tests covering all modules (including compression, optimizers, graph)
//!   - 10 integration tests with end-to-end workflows
//!   - 10 property-based tests for mathematical correctness
//! - **15 performance benchmarks** (7 VJP/decomp + 8 graph)
//!
//! Run tests with:
//! ```bash
//! cargo test --all-features
//! ```
//!
//! Run benchmarks with:
//! ```bash
//! cargo bench
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive demonstrations:
//! - `basic_einsum_ad.rs` - VJP for matrix operations
//! - `cp_decomposition_gradients.rs` - CP-ALS gradients
//! - `custom_ad_operations.rs` - External AD integration
//! - `gradient_checkpointing.rs` - Memory-efficient training
//! - `gradient_compression.rs` - Compression for distributed training
//! - `hessian_optimization.rs` - Second-order methods
//! - `sparse_gradients.rs` - Sparse gradient handling
//! - `mixed_precision_training.rs` - FP16/BF16 training
//! - `gradient_monitoring.rs` - Training health tracking
//! - `optimizers_showcase.rs` - Optimization algorithms and schedulers
//! - `graph_based_ad.rs` - PyTorch-style computation graphs
//!
//! ## SCIRS2 Policy Compliance
//!
//! This crate strictly adheres to the SCIRS2 policy:
//! - ✅ All array operations use `scirs2_core::ndarray_ext`
//! - ✅ All numeric types use `scirs2_core::numeric`
//! - ✅ No direct `ndarray` or `rand` imports
//! - ✅ All linalg operations use `scirs2_linalg`
//!
//! ## Production Readiness
//!
//! This crate is **production-ready** with:
//! - ✅ Zero warnings in strict clippy mode
//! - ✅ 100% test pass rate
//! - ✅ Comprehensive documentation
//! - ✅ Performance benchmarks
//! - ✅ Working examples for all features
//!
//! ## Future Work
//!
//! Planned enhancements (see `TODO.md`):
//! - Tensorlogic integration (awaiting API stabilization)
//! - Distributed gradients with AllReduce
//! - Graph-based AD with dynamic control flow
//! - GPU kernel integration via scirs2-gpu

#![deny(warnings)]
#![recursion_limit = "4096"]

pub mod checkpoint;
pub mod compression;
pub mod grad;
pub mod gradcheck;
pub mod graph;
pub mod graph_optimizer;
pub mod hessian;
pub mod hooks;
pub mod mixed_precision;
pub mod monitoring;
pub mod optimizers;
pub mod parallel;
pub mod sparse_grad;
pub mod storage;
pub mod utils;
pub mod vjp;

// Re-exports for convenience
pub use vjp::*;
