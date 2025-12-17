//! # TenRSo - Tensor Computing Stack for COOLJAPAN
//!
//! **Production-grade tensor operations** with generalized contraction, decompositions,
//! sparse formats, and out-of-core processing.
//!
//! **Version:** 0.1.0-alpha.2
//! **Status:** Production-ready alpha release with 1,820+ tests passing
//!
//! This is the **meta crate** that re-exports all TenRSo components for convenient access.
//!
//! ## Quick Start
//!
//! ```
//! use tenrso::prelude::*;
//!
//! // Create a 3D tensor
//! let tensor = DenseND::<f64>::zeros(&[10, 20, 30]);
//! assert_eq!(tensor.shape(), &[10, 20, 30]);
//!
//! // Perform CP decomposition
//! // let cp = cp_als(&tensor, 64, 50, 1e-4, InitStrategy::Random)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Components
//!
//! ### Core Tensor Operations ([`core`])
//!
//! Dense tensor types, axis metadata, views, reshape/permute, unfold/fold.
//!
//! ```
//! use tenrso::core::DenseND;
//!
//! let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
//! let reshaped = tensor.reshape(&[6, 4]).unwrap();
//! assert_eq!(reshaped.shape(), &[6, 4]);
//! ```
//!
//! ### Tensor Kernels ([`kernels`])
//!
//! Khatri-Rao, Kronecker, Hadamard, n-mode products, MTTKRP.
//!
//! ```ignore
//! use tenrso::kernels::khatri_rao;
//! use scirs2_core::ndarray_ext::Array2;
//!
//! let a = Array2::<f64>::ones((10, 5));
//! let b = Array2::<f64>::ones((8, 5));
//! let kr = khatri_rao(&a.view(), &b.view());
//! assert_eq!(kr.shape(), &[80, 5]);
//! ```
//!
//! ### Tensor Decompositions ([`decomp`])
//!
//! CP-ALS, Tucker-HOOI/HOSVD, TT-SVD decomposition algorithms.
//!
//! ```
//! use tenrso::decomp::{cp_als, InitStrategy};
//! use tenrso::core::DenseND;
//!
//! let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
//! let cp = cp_als(&tensor, 5, 50, 1e-4, InitStrategy::Random, None).unwrap();
//! println!("CP decomposition modes: {}", cp.factors.len());
//! ```
//!
//! ### Sparse Tensors ([`sparse`])
//!
//! COO, CSR, BCSR, CSC formats and sparse operations.
//!
//! ```
//! use tenrso::sparse::CooTensor;
//!
//! let indices = vec![vec![0, 0], vec![1, 1]];
//! let values = vec![1.0, 2.0];
//! let shape = vec![2, 2];
//! let coo = CooTensor::new(indices, values, shape).unwrap();
//! assert_eq!(coo.nnz(), 2);
//! ```
//!
//! ### Contraction Planning ([`planner`])
//!
//! Contraction order optimization and representation selection.
//!
//! ```
//! use tenrso::planner::{greedy_planner, EinsumSpec, PlanHints};
//!
//! let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
//! let shapes = vec![vec![100, 200], vec![200, 300]];
//! let hints = PlanHints::default();
//! let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
//! println!("Estimated FLOPs: {:.2e}", plan.estimated_flops);
//! ```
//!
//! ### Out-of-Core Processing ([`ooc`])
//!
//! Arrow/Parquet I/O, memory-mapped arrays, chunked execution.
//!
//! Available when the `ooc` feature is enabled (default).
//!
//! ### Unified Execution ([`exec`])
//!
//! High-level einsum API with automatic optimization.
//!
//! ```
//! use tenrso::exec::{einsum_ex, ExecHints};
//! use tenrso::core::{DenseND, TensorHandle};
//!
//! let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
//! let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
//!
//! let handle_a = TensorHandle::from_dense_auto(a);
//! let handle_b = TensorHandle::from_dense_auto(b);
//!
//! let result = einsum_ex::<f64>("ij,jk->ik")
//!     .inputs(&[handle_a, handle_b])
//!     .run()
//!     .unwrap();
//! # assert_eq!(result.as_dense().unwrap().shape(), &[2, 2]);
//! ```
//!
//! ### Automatic Differentiation
//!
//! Custom VJP/grad rules for tensor operations (available via the `ad` module).
//!
//! Available when the `ad` feature is enabled.
//!
//! ## Features
//!
//! - `ooc` (default): Enable out-of-core processing (Arrow, Parquet, mmap)
//! - `ad`: Enable automatic differentiation support
//! - `full`: Enable all features
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive usage demonstrations.
//!
//! ## Performance
//!
//! TenRSo is optimized for production use:
//! - **SIMD acceleration** via SciRS2-core
//! - **Parallel execution** for large problems
//! - **Cache-efficient** blocking and tiling
//! - **Zero-copy** views and operations
//!
//! ## Documentation
//!
//! - [GitHub Repository](https://github.com/cool-japan/tenrso)
//! - [CHANGELOG](https://github.com/cool-japan/tenrso/blob/main/CHANGELOG.md)
//! - [Contributing Guide](https://github.com/cool-japan/tenrso/blob/main/CONTRIBUTING.md)

#![deny(warnings)]

// Re-export all components
pub use tenrso_core as core;
pub use tenrso_decomp as decomp;
pub use tenrso_exec as exec;
pub use tenrso_kernels as kernels;
#[cfg(feature = "ooc")]
pub use tenrso_ooc as ooc;
pub use tenrso_planner as planner;
pub use tenrso_sparse as sparse;

#[cfg(feature = "ad")]
pub use tenrso_ad as ad;

pub mod prelude {
    //! Prelude module for convenient imports
    //!
    //! # Example
    //!
    //! ```
    //! use tenrso::prelude::*;
    //!
    //! let tensor = DenseND::<f64>::zeros(&[10, 20, 30]);
    //! ```

    // Core types
    pub use crate::core::{AxisMeta, DenseND, TensorHandle, TensorRepr};

    // Decomposition algorithms
    pub use crate::decomp::{cp_als, tucker_hooi, tucker_hosvd, InitStrategy};

    // Common kernels
    pub use crate::kernels::{
        hadamard, hadamard_inplace, khatri_rao, kronecker, mttkrp, nmode_product,
    };

    // Sparse types
    pub use crate::sparse::{CooTensor, CscMatrix, CsrMatrix};

    // Execution
    pub use crate::exec::{einsum_ex, ExecHints};

    // Planner
    pub use crate::planner::{greedy_planner, EinsumSpec, PlanHints};
}
