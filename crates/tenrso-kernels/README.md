# tenrso-kernels

Tensor kernel operations: Khatri-Rao, Kronecker, Hadamard, n-mode products, and MTTKRP.

## Overview

`tenrso-kernels` provides high-performance tensor operation kernels that are fundamental building blocks for tensor decompositions and contractions:

- **Khatri-Rao product** - Column-wise Kronecker product
- **Kronecker product** - Tensor product of matrices
- **Hadamard product** - Element-wise (pointwise) multiplication
- **N-mode product** - Tensor-matrix/tensor-tensor products (TTM/TTT)
- **MTTKRP** - Matricized Tensor Times Khatri-Rao Product

All kernels are optimized for performance with SIMD acceleration and parallel execution.

## Features

- Cache-friendly implementations
- SIMD-accelerated operations (via scirs2_core)
- Parallel execution with Rayon (optional feature)
- Generic over scalar types (f32, f64)
- Minimal allocations
- Correctness property tests

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-kernels = "0.1"

# With parallel execution
tenrso-kernels = { version = "0.1", features = ["parallel"] }
```

### Khatri-Rao Product (TODO: M1)

```rust
use tenrso_kernels::khatri_rao;
use scirs2_core::ndarray_ext::Array2;

let a = Array2::from_shape_fn((100, 10), |(i, j)| (i + j) as f64);
let b = Array2::from_shape_fn((50, 10), |(i, j)| (i * j) as f64);

// Column-wise Kronecker product: (100*50) × 10
let kr = khatri_rao(&a.view(), &b.view());
assert_eq!(kr.shape(), &[5000, 10]);
```

### Kronecker Product (TODO: M1)

```rust
use tenrso_kernels::kronecker;
use scirs2_core::ndarray_ext::Array2;

let a = Array2::from_elem((2, 3), 2.0);
let b = Array2::from_elem((4, 5), 3.0);

// Tensor product: (2*4) × (3*5)
let kron = kronecker(&a.view(), &b.view());
assert_eq!(kron.shape(), &[8, 15]);
```

### Hadamard Product (TODO: M1)

```rust
use tenrso_kernels::hadamard;
use scirs2_core::ndarray_ext::Array;

let a = Array::from_elem(vec![10, 20, 30], 2.0);
let b = Array::from_elem(vec![10, 20, 30], 3.0);

// Element-wise multiplication
let result = hadamard(&a, &b)?;
assert_eq!(result[[5, 10, 15]], 6.0);
```

### N-Mode Product (TODO: M1)

```rust
use tenrso_kernels::nmode_product;
use scirs2_core::ndarray_ext::{Array, Array2};

// Tensor: 10 × 20 × 30
let tensor = Array::zeros(vec![10, 20, 30]);

// Matrix: 15 × 20 (contracts along mode 1)
let matrix = Array2::zeros((15, 20));

// Result: 10 × 15 × 30
let result = nmode_product(&tensor, &matrix, 1)?;
assert_eq!(result.shape(), &[10, 15, 30]);
```

### MTTKRP (TODO: M1)

```rust
use tenrso_kernels::mttkrp;
use scirs2_core::ndarray_ext::{Array, Array2};

// Tensor: 100 × 200 × 300
let tensor = Array::zeros(vec![100, 200, 300]);

// Factor matrices for CP decomposition
let u = Array2::zeros((200, 64));  // Mode 1 factors
let v = Array2::zeros((300, 64));  // Mode 2 factors

// MTTKRP along mode 0: (100, 64)
let result = mttkrp(&tensor, &[&u, &v], 0)?;
assert_eq!(result.shape(), &[100, 64]);
```

## API Reference

### Khatri-Rao Product

```rust
pub fn khatri_rao<T>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
) -> Array2<T>
where
    T: LinalgScalar
```

Computes the column-wise Kronecker product. Output shape: `(a.nrows() * b.nrows(), a.ncols())`.

**Complexity:** O(I × J × K) where a is I×K and b is J×K.

### Kronecker Product

```rust
pub fn kronecker<T>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
) -> Array2<T>
where
    T: LinalgScalar
```

Computes the tensor product. Output shape: `(a.nrows() * b.nrows(), a.ncols() * b.ncols())`.

**Complexity:** O(I × J × K × L) where a is I×K and b is J×L.

### Hadamard Product

```rust
pub fn hadamard<T, D>(
    a: &ArrayBase<S, D>,
    b: &ArrayBase<S, D>,
) -> Result<Array<T, D>>
where
    T: LinalgScalar,
    D: Dimension
```

Element-wise multiplication. Shapes must match exactly.

**Complexity:** O(N) where N is total elements.

### N-Mode Product

```rust
pub fn nmode_product<T>(
    tensor: &Array<T, IxDyn>,
    matrix: &Array2<T>,
    mode: usize,
) -> Result<Array<T, IxDyn>>
where
    T: LinalgScalar
```

Contracts tensor with matrix along specified mode.

**Complexity:** O(I₀ × ... × Iₙ × J) where tensor is I₀×...×Iₙ, matrix is J×Iₘₒdₑ.

### MTTKRP

```rust
pub fn mttkrp<T>(
    tensor: &Array<T, IxDyn>,
    factors: &[&Array2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: LinalgScalar
```

Matricized Tensor Times Khatri-Rao Product. Core operation for CP-ALS.

**Complexity:** O(I₀ × ... × Iₙ × R) where tensor is I₀×...×Iₙ, rank is R.

## Performance Optimization

### SIMD Acceleration

Kernels use `scirs2_core` SIMD operations when available:
- AVX2 for f32/f64 operations
- Automatic vectorization for element-wise ops
- Cache-friendly memory access patterns

### Parallel Execution

Enable `parallel` feature for multi-threaded execution:

```toml
[dependencies]
tenrso-kernels = { version = "0.1", features = ["parallel"] }
```

### Blocked Operations

Large matrices use blocked algorithms to improve cache locality:
- MTTKRP uses tiled iteration
- Khatri-Rao batches column operations
- N-mode product uses chunked unfolding

## Benchmarks

Run performance benchmarks:

```bash
cargo bench --features parallel

# Compare with baseline
cargo bench --bench khatri_rao -- --baseline main
```

Expected performance (16-core CPU):
- **Khatri-Rao**: > 50 GFLOP/s (1000×1000, rank 64)
- **MTTKRP**: > 80% of optimal GEMM rate
- **N-mode**: > 75% of GEMM baseline

## Testing

```bash
# Unit tests
cargo test

# Property tests
cargo test --test properties

# With all features
cargo test --all-features
```

## Examples

See `examples/` directory:
- `khatri_rao.rs` - Basic usage and benchmarking (TODO)
- `mttkrp_cp.rs` - MTTKRP in CP decomposition context (TODO)
- `nmode_tucker.rs` - N-mode product for Tucker (TODO)

## Feature Flags

- `default` = `["parallel"]` - Parallel execution enabled
- `parallel` - Use Rayon for multi-threading

## Dependencies

- **tenrso-core** - Tensor types
- **scirs2-core** - Array operations, SIMD
- **rayon** (optional) - Parallel iteration
- **num-traits** - Generic numeric traits

## Contributing

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Apache-2.0
