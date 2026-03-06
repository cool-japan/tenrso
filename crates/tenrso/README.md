# tenrso

**TenRSo — Tensor Computing Stack for Rust**

This is the **meta crate** that provides a unified API by re-exporting all TenRSo sub-crates. Add this single dependency to your project for access to the complete TenRSo ecosystem.

## Overview

TenRSo is a production-grade, Rust-native tensor computing stack providing:

- **Generalized contraction with cost-based planning** — optimal contraction order, tiling, streaming
- **Sparse/low-rank mixed execution** — Dense, Sparse (COO/CSR/BCSR/CSF/HiCOO), and Low-rank (CP/Tucker/TT) representations chosen automatically by the planner
- **Tensor decompositions** — CP-ALS, Tucker-HOOI/HOSVD, TT-SVD
- **Out-of-core processing** — Arrow/Parquet I/O, memory-mapped arrays, chunked execution
- **Automatic differentiation** — custom VJP/grad rules for contractions and decompositions
- **Production discipline** — strong CI, perf budgets, no-panic kernels, semver stability

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso = "0.1.0-alpha.2"
```

To enable all features:

```toml
[dependencies]
tenrso = { version = "0.1.0-alpha.2", features = ["full"] }
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ooc` | yes | Out-of-core processing via Arrow, Parquet, and mmap |
| `ad` | no | Automatic differentiation support |
| `full` | no | Enables all features (`ooc` + `ad`) |

## Quick Start

### Using the Prelude

```rust
use tenrso::prelude::*;

// Create a 3D tensor
let tensor = DenseND::<f64>::zeros(&[10, 20, 30]);
assert_eq!(tensor.shape(), &[10, 20, 30]);
```

### Core Tensor Operations

```rust
use tenrso::core::DenseND;

let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
let reshaped = tensor.reshape(&[6, 4]).unwrap();
assert_eq!(reshaped.shape(), &[6, 4]);
```

### Einsum Contractions

```rust
use tenrso::exec::{einsum_ex, ExecHints};
use tenrso::core::{DenseND, TensorHandle};

let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

let handle_a = TensorHandle::from_dense_auto(a);
let handle_b = TensorHandle::from_dense_auto(b);

let result = einsum_ex::<f64>("ij,jk->ik")
    .inputs(&[handle_a, handle_b])
    .run()
    .unwrap();
```

### Tensor Decompositions

```rust
use tenrso::decomp::{cp_als, tucker_hooi, InitStrategy};
use tenrso::core::DenseND;

let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);

// CP decomposition (rank-5, 50 iterations)
let cp = cp_als(&tensor, 5, 50, 1e-4, InitStrategy::Random, None).unwrap();

// Tucker decomposition
let tucker = tucker_hooi(&tensor, &[4, 4, 4], 30, 1e-4).unwrap();
```

### Sparse Tensors

```rust
use tenrso::sparse::CooTensor;

let indices = vec![vec![0, 0], vec![1, 1]];
let values = vec![1.0f64, 2.0];
let shape = vec![2, 2];
let coo = CooTensor::new(indices, values, shape).unwrap();
assert_eq!(coo.nnz(), 2);
```

### Contraction Planning

```rust
use tenrso::planner::{greedy_planner, EinsumSpec, PlanHints};

let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
let shapes = vec![vec![100, 200], vec![200, 300]];
let hints = PlanHints::default();
let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
println!("Estimated FLOPs: {:.2e}", plan.estimated_flops);
```

### Automatic Differentiation (requires `ad` feature)

```rust
// [dependencies]
// tenrso = { version = "0.1.0-alpha.2", features = ["ad"] }

use tenrso::ad::graph::{ComputationGraph, Variable};

let mut graph = ComputationGraph::<f64>::new();
// ... build computation graph and run backward pass
```

## Re-exported Modules

| Module | Crate | Description |
|--------|-------|-------------|
| `tenrso::core` | `tenrso-core` | Dense tensors, axis metadata, views, reshape/permute, unfold/fold |
| `tenrso::kernels` | `tenrso-kernels` | Khatri-Rao, Kronecker, Hadamard, n-mode products, MTTKRP |
| `tenrso::decomp` | `tenrso-decomp` | CP-ALS, Tucker-HOOI/HOSVD, TT-SVD decompositions |
| `tenrso::sparse` | `tenrso-sparse` | COO, CSR, BCSR, CSC sparse tensor formats |
| `tenrso::planner` | `tenrso-planner` | Contraction order optimization and representation selection |
| `tenrso::ooc` | `tenrso-ooc` | Out-of-core Arrow/Parquet I/O and mmap (requires `ooc` feature) |
| `tenrso::exec` | `tenrso-exec` | Unified execution API with automatic optimization |
| `tenrso::ad` | `tenrso-ad` | Automatic differentiation VJP/grad rules (requires `ad` feature) |

## Prelude Exports

`use tenrso::prelude::*` imports the most commonly used types and functions:

- **Core types:** `DenseND`, `TensorHandle`, `TensorRepr`, `AxisMeta`
- **Decompositions:** `cp_als`, `tucker_hooi`, `tucker_hosvd`, `InitStrategy`
- **Kernels:** `khatri_rao`, `kronecker`, `hadamard`, `hadamard_inplace`, `mttkrp`, `nmode_product`
- **Sparse:** `CooTensor`, `CsrMatrix`, `CscMatrix`
- **Execution:** `einsum_ex`, `ExecHints`
- **Planning:** `greedy_planner`, `EinsumSpec`, `PlanHints`

## Performance Targets

- **Einsum (CPU, float32):** >= 80% of OpenBLAS baseline
- **Masked operations:** >= 5x speedup vs dense naive
- **CP-ALS:** < 2s / 10 iters (256^3, rank-64)
- **Tucker-HOOI:** < 3s / 10 iters (512x512x128)
- **TT-SVD:** < 2s build time (32^6)

## Links

- [GitHub Repository](https://github.com/cool-japan/tenrso)
- [API Documentation](https://docs.rs/tenrso)
- [CHANGELOG](https://github.com/cool-japan/tenrso/blob/main/CHANGELOG.md)
- [Contributing Guide](https://github.com/cool-japan/tenrso/blob/main/CONTRIBUTING.md)

## License

Apache-2.0

Copyright (c) COOLJAPAN OU (Team Kitasan)
