# tenrso-kernels

> **Version:** 0.1.0-rc.1 | **Status:** [Alpha] — 264 tests passing, 7 ignored (100%) | **Updated:** 2026-03-06

Tensor kernel operations: Khatri-Rao, Kronecker, Hadamard, N-mode products, MTTKRP, outer products, Tucker, statistical reductions, and randomized algorithms.

## Overview

`tenrso-kernels` provides high-performance tensor operation kernels that are fundamental building blocks for tensor decompositions, contractions, and scientific computing:

- **Khatri-Rao product** — Column-wise Kronecker product (basic, parallel, blocked, blocked+parallel)
- **Kronecker product** — Tensor product of matrices (basic, parallel)
- **Hadamard product** — Element-wise multiplication (2D, ND, in-place)
- **N-mode products** — Tensor-matrix (TTM), tensor-tensor (TTT), sequential multi-mode
- **Outer products** — 2D, ND, weighted, CP reconstruction, parallel CP reconstruction
- **Tucker operator** — Auto-optimized and ordered multi-mode products, Tucker reconstruction
- **MTTKRP** — Matricized Tensor Times Khatri-Rao Product (basic, blocked, fused, all with parallel variants)
- **Statistical reductions** — 14+ operations: sum, mean, min, max, variance, std, norms, skewness, kurtosis
- **Multivariate analysis** — Covariance, correlation matrices
- **Randomized algorithms** — Randomized SVD, randomized range finder
- **TT operations** — 10 tensor-train functions
- **Tensor contractions** — General contraction primitives

Source: 14 files, ~10K lines of code. 135+ benchmarks via criterion.

## Features

- Cache-friendly implementations with blocked/tiled execution
- SIMD-accelerated operations (via scirs2_core)
- Parallel execution with Rayon (`parallel` feature, enabled by default)
- Generic over scalar types (f32, f64)
- Minimal allocations; fused kernels avoid intermediate materialization
- 40+ property-based tests (proptest)
- Full SCIRS2 policy compliance

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-kernels = "0.1"

# With parallel execution (enabled by default)
tenrso-kernels = { version = "0.1", features = ["parallel"] }
```

### Khatri-Rao Product

```rust
use tenrso_kernels::khatri_rao;
use scirs2_core::ndarray_ext::Array2;

let a = Array2::from_shape_fn((100, 10), |(i, j)| (i + j) as f64);
let b = Array2::from_shape_fn((50, 10), |(i, j)| (i * j) as f64);

// Column-wise Kronecker product: (100*50) x 10
let kr = khatri_rao(&a.view(), &b.view());
assert_eq!(kr.shape(), &[5000, 10]);

// Parallel variant
use tenrso_kernels::khatri_rao_parallel;
let kr_par = khatri_rao_parallel(&a.view(), &b.view());

// Blocked variant for large matrices
use tenrso_kernels::khatri_rao_blocked;
let kr_blocked = khatri_rao_blocked(&a.view(), &b.view(), 64);
```

### Kronecker Product

```rust
use tenrso_kernels::kronecker;
use scirs2_core::ndarray_ext::Array2;

let a = Array2::from_elem((2, 3), 2.0f64);
let b = Array2::from_elem((4, 5), 3.0f64);

// Tensor product: (2*4) x (3*5)
let kron = kronecker(&a.view(), &b.view());
assert_eq!(kron.shape(), &[8, 15]);

// Parallel variant
use tenrso_kernels::kronecker_parallel;
let kron_par = kronecker_parallel(&a.view(), &b.view());
```

### Hadamard Product

```rust
use tenrso_kernels::hadamard;
use scirs2_core::ndarray_ext::Array;

let a = Array::from_elem(vec![10, 20, 30], 2.0f64);
let b = Array::from_elem(vec![10, 20, 30], 3.0f64);

// Element-wise multiplication
let result = hadamard(&a, &b)?;
assert_eq!(result[[5, 10, 15]], 6.0);

// In-place variant
use tenrso_kernels::hadamard_inplace;
let mut c = a.clone();
hadamard_inplace(&mut c, &b)?;
```

### N-Mode Product

```rust
use tenrso_kernels::nmode_product;
use scirs2_core::ndarray_ext::{Array, Array2};

// Tensor: 10 x 20 x 30
let tensor = Array::zeros(vec![10, 20, 30]);

// Matrix: 15 x 20 (contracts along mode 1)
let matrix = Array2::zeros((15, 20));

// Result: 10 x 15 x 30
let result = nmode_product(&tensor, &matrix, 1)?;
assert_eq!(result.shape(), &[10, 15, 30]);

// Sequential multi-mode products
use tenrso_kernels::nmode_products_seq;
let factors = vec![Array2::zeros((5, 10)), Array2::zeros((8, 20))];
let result = nmode_products_seq(&tensor, &factors, &[0, 1])?;
```

### Tensor-Tensor Product (TTT)

```rust
use tenrso_kernels::tensor_tensor_product;
use scirs2_core::ndarray_ext::Array;

let a = Array::zeros(vec![3, 4, 5]);
let b = Array::zeros(vec![4, 5, 6]);

// Contract along modes [1, 2] of a and modes [0, 1] of b
// Result: 3 x 6
let result = tensor_tensor_product(&a, &b, &[1, 2], &[0, 1])?;
assert_eq!(result.shape(), &[3, 6]);
```

### Outer Products and CP Reconstruction

```rust
use tenrso_kernels::{outer_product, cp_reconstruct};
use scirs2_core::ndarray_ext::{Array, Array1};

// ND outer product
let vecs = vec![
    Array1::from_vec(vec![1.0f64, 2.0, 3.0]),
    Array1::from_vec(vec![4.0f64, 5.0]),
    Array1::from_vec(vec![6.0f64, 7.0, 8.0]),
];
let outer = outer_product(&vecs);
assert_eq!(outer.shape(), &[3, 2, 3]);

// CP decomposition reconstruction: sum of weighted outer products
use scirs2_core::ndarray_ext::Array2;
let factors = vec![
    Array2::zeros((10, 4)),
    Array2::zeros((20, 4)),
    Array2::zeros((30, 4)),
];
let weights = vec![1.0f64; 4];
let tensor = cp_reconstruct(&factors, Some(&weights))?;
assert_eq!(tensor.shape(), &[10, 20, 30]);
```

### Tucker Operator

```rust
use tenrso_kernels::{tucker_operator, tucker_reconstruct};
use scirs2_core::ndarray_ext::{Array, Array2};
use std::collections::HashMap;

// Core tensor: 4 x 5 x 6
let core = Array::zeros(vec![4, 5, 6]);

// Factor matrices (mode -> matrix)
let mut factors = HashMap::new();
factors.insert(0usize, Array2::zeros((10, 4)));  // 10 x 4
factors.insert(1usize, Array2::zeros((20, 5)));  // 20 x 5
factors.insert(2usize, Array2::zeros((30, 6)));  // 30 x 6

// Apply Tucker operator: result is 10 x 20 x 30
let result = tucker_operator(&core, &factors)?;
assert_eq!(result.shape(), &[10, 20, 30]);

// Tucker reconstruction (explicit ordering)
let result = tucker_reconstruct(&core, &factors)?;
```

### MTTKRP

```rust
use tenrso_kernels::mttkrp;
use scirs2_core::ndarray_ext::{Array, Array2};

// Tensor: 100 x 200 x 300
let tensor = Array::zeros(vec![100, 200, 300]);

// Factor matrices for CP decomposition
let u = Array2::zeros((200, 64));  // Mode 1 factors
let v = Array2::zeros((300, 64));  // Mode 2 factors

// MTTKRP along mode 0: (100, 64)
let result = mttkrp(&tensor, &[&u, &v], 0)?;
assert_eq!(result.shape(), &[100, 64]);

// Blocked variant (cache-friendly for large tensors)
use tenrso_kernels::mttkrp_blocked;
let result = mttkrp_blocked(&tensor, &[&u, &v], 0, 32)?;

// Fused variant (avoids materializing full KR product)
use tenrso_kernels::mttkrp_fused;
let result = mttkrp_fused(&tensor, &[&u, &v], 0)?;

// Parallel fused variant
use tenrso_kernels::mttkrp_fused_parallel;
let result = mttkrp_fused_parallel(&tensor, &[&u, &v], 0)?;
```

### Statistical Reductions

```rust
use tenrso_kernels::stats;
use scirs2_core::ndarray_ext::Array;

let tensor = Array::from_shape_fn(vec![10, 20, 30], |idx| {
    (idx[0] + idx[1] + idx[2]) as f64
});

let total = stats::sum(&tensor);
let mean = stats::mean(&tensor);
let variance = stats::variance(&tensor);
let std_dev = stats::std(&tensor);
let l2_norm = stats::norm_l2(&tensor);
let frobenius = stats::norm_frobenius(&tensor);
```

### Randomized Algorithms

```rust
use tenrso_kernels::random::{randomized_svd, randomized_range_finder};
use scirs2_core::ndarray_ext::Array2;

let matrix = Array2::<f64>::zeros((500, 300));
let rank = 50;

// Randomized SVD (O(mnk) instead of O(mn²))
let (u, s, vt) = randomized_svd(&matrix.view(), rank, 5, 1)?;
assert_eq!(u.shape(), &[500, rank]);
assert_eq!(s.len(), rank);
assert_eq!(vt.shape(), &[rank, 300]);

// Randomized range finder
let q = randomized_range_finder(&matrix.view(), rank, 5)?;
assert_eq!(q.shape(), &[500, rank]);
```

## API Reference

### Khatri-Rao

| Function | Description |
|----------|-------------|
| `khatri_rao(a, b)` | Serial column-wise Kronecker product |
| `khatri_rao_parallel(a, b)` | Parallel column processing via Rayon |
| `khatri_rao_blocked(a, b, block_size)` | Cache-blocked execution |
| `khatri_rao_blocked_parallel(a, b, block_size)` | Parallel + blocked |

Output shape: `(a.nrows() * b.nrows(), a.ncols())`. Inputs must have same `ncols`.

**Complexity:** O(I × J × K) where a is I×K and b is J×K.

### Kronecker

| Function | Description |
|----------|-------------|
| `kronecker(a, b)` | Serial tensor product |
| `kronecker_parallel(a, b)` | Parallel block processing |

Output shape: `(a.nrows() * b.nrows(), a.ncols() * b.ncols())`.

**Complexity:** O(I × J × K × L) where a is I×K and b is J×L.

### Hadamard

| Function | Description |
|----------|-------------|
| `hadamard(a, b)` | Element-wise multiplication (any D) |
| `hadamard_inplace(a, b)` | In-place variant |

**Complexity:** O(N) where N is total elements.

### N-Mode Products

| Function | Description |
|----------|-------------|
| `nmode_product(tensor, matrix, mode)` | Single mode TTM |
| `nmode_products_seq(tensor, matrices, modes)` | Sequential multi-mode TTM |
| `tensor_tensor_product(a, b, modes_a, modes_b)` | TTT contraction |

**Complexity:** O(I₀ × ... × Iₙ × J) for single TTM.

### Outer Products

| Function | Description |
|----------|-------------|
| `outer_product_2(a, b)` | 2D outer product of two vectors |
| `outer_product(vecs)` | ND outer product of vector list |
| `outer_product_weighted(vecs, weight)` | Weighted ND outer product |
| `cp_reconstruct(factors, weights)` | Sum of rank-1 outer products |
| `cp_reconstruct_parallel(factors, weights)` | Parallel CP reconstruction |

### Tucker

| Function | Description |
|----------|-------------|
| `tucker_operator(core, factors)` | Auto-ordered mode products |
| `tucker_operator_ordered(core, factors, order)` | Explicit mode ordering |
| `tucker_reconstruct(core, factors)` | Full Tucker reconstruction |

### MTTKRP

| Function | Description |
|----------|-------------|
| `mttkrp(tensor, factors, mode)` | Standard implementation |
| `mttkrp_blocked(tensor, factors, mode, tile)` | Cache-blocked |
| `mttkrp_blocked_parallel(tensor, factors, mode, tile)` | Parallel blocked |
| `mttkrp_fused(tensor, factors, mode)` | Fused (no KR materialization) |
| `mttkrp_fused_parallel(tensor, factors, mode)` | Parallel fused |

**Complexity:** O(I₀ × ... × Iₙ × R) where R is the rank.

### Statistical Reductions (14+)

| Function | Description |
|----------|-------------|
| `stats::sum(t)` | Total sum |
| `stats::mean(t)` | Global mean |
| `stats::min(t)` / `stats::max(t)` | Global min/max |
| `stats::variance(t)` | Population variance |
| `stats::std(t)` | Standard deviation |
| `stats::norm_l1(t)` | L1 norm |
| `stats::norm_l2(t)` | L2 norm |
| `stats::norm_frobenius(t)` | Frobenius norm |
| `stats::skewness(t)` | Third standardized moment |
| `stats::kurtosis(t)` | Fourth standardized moment |
| `stats::covariance(a, b)` | Covariance matrix |
| `stats::correlation(a, b)` | Correlation matrix |

### Randomized Algorithms

| Function | Description |
|----------|-------------|
| `random::randomized_svd(m, rank, n_oversamples, n_iter)` | Randomized SVD |
| `random::randomized_range_finder(m, rank, n_oversamples)` | Range finder |

## Performance

Measured on a 16-core CPU:

| Operation | Performance |
|-----------|-------------|
| Khatri-Rao (500×32) serial | ~1.5 Gelem/s |
| Khatri-Rao parallel speedup | ~3× at 500×32 |
| MTTKRP blocked parallel (50³) | ~13.3 Gelem/s peak |
| N-mode product | >5 Gelem/s sustained |
| Hadamard in-place | 70-80% memory bandwidth |

### Optimization Guide

**For large Khatri-Rao:** Use `khatri_rao_blocked_parallel` with block_size=64.

**For MTTKRP in CP-ALS:** Use `mttkrp_fused_parallel` to avoid materializing the Khatri-Rao product and reduce memory pressure.

**For Tucker contraction:** Use `tucker_operator` (auto-orders from smallest to largest to minimize intermediate sizes).

**For approximate low-rank:** Use `randomized_svd` — O(mnk) vs O(mn²) for full SVD.

## Performance Optimization Details

### Blocked Operations

Large matrices use blocked algorithms to improve cache locality:
- MTTKRP uses tiled iteration with configurable tile size
- Khatri-Rao batches column operations with cache-blocking
- N-mode product uses chunked unfolding

### Fused MTTKRP

`mttkrp_fused` avoids materializing the full Khatri-Rao product (which is `prod(dims) × rank`). Instead, it directly accumulates contributions from tensor fibers, significantly reducing peak memory usage.

### Parallel Execution

All `_parallel` variants use Rayon's work-stealing thread pool. The default feature set includes parallel execution. Disable with `default-features = false` if single-threaded behavior is required.

## Benchmarks

135+ individual benchmarks across 10 benchmark groups:

```bash
# Run all benchmarks
cargo bench --features parallel

# Specific benchmark groups
cargo bench --bench khatri_rao
cargo bench --bench mttkrp
cargo bench --bench nmode
cargo bench --bench outer_products
cargo bench --bench tucker
cargo bench --bench statistics

# Compare against a baseline
cargo bench --bench khatri_rao -- --baseline main
```

## Testing

```bash
# Unit + property + integration tests (264 passing, 7 ignored)
cargo test

# With all features
cargo test --all-features

# Property tests only
cargo test --test properties

# Run benchmarks
cargo bench --features parallel
```

**Test breakdown:**
- Unit tests: 132 (correctness, edge cases)
- Property tests (proptest): 40+ (mathematical invariants)
- Integration tests: 22+ (real-world CP-ALS, Tucker-HOOI workflows)
- Doc tests: 55+ (all API examples compile and pass)

## Examples

See `examples/` directory:
- `khatri_rao.rs` — Parallel speedup demonstration with timing
- `mttkrp_cp.rs` — CP-ALS iteration using MTTKRP variants
- `nmode_tucker.rs` — Tucker decomposition workflow

```bash
cargo run --example khatri_rao
cargo run --example mttkrp_cp
cargo run --example nmode_tucker
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | Includes `parallel` |
| `parallel` | Use Rayon for multi-threaded execution |

## Dependencies

- **tenrso-core** — Tensor types and DenseND
- **scirs2-core** — Array operations, SIMD, random generation
- **scirs2-linalg** — SVD and QR for randomized algorithms
- **rayon** (optional) — Parallel iteration
- **num-traits** — Generic numeric traits
- **proptest** (dev) — Property-based testing
- **criterion** (dev) — Benchmarking

## Contributing

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Apache-2.0
