# tenrso-decomp

Tensor decomposition methods: CP-ALS, Tucker-HOOI/HOSVD, and TT-SVD.

## Overview

`tenrso-decomp` provides production-grade implementations of major tensor decomposition algorithms:

- **CP-ALS** - Canonical Polyadic decomposition via Alternating Least Squares
- **Tucker-HOSVD** - Higher-Order SVD (one-pass truncation)
- **Tucker-HOOI** - Higher-Order Orthogonal Iteration (iterative refinement)
- **TT-SVD** - Tensor Train decomposition via sequential SVD

All methods support initialization strategies, convergence criteria, and reconstruction error tracking.

## Features

- Multiple initialization methods (random, SVD, leverage scores)
- Non-negative factorization (optional)
- Regularization support
- Reconstruction error tracking
- Generic over scalar types (f32, f64)
- Parallel execution

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-decomp = "0.1"
```

### CP Decomposition (TODO: M2)

```rust
use tenrso_decomp::cp_als;
use scirs2_core::ndarray_ext::Array;

let tensor = Array::zeros(vec![100, 200, 300]);

// CP-ALS with rank 64
let cp = cp_als(
    &tensor,
    rank = 64,
    iters = 50,
    tol = 1e-4,
    nonneg = false,
)?;

println!("Reconstruction error: {}", cp.error());
println!("Final factors: {:?}", cp.factors());
```

### Tucker Decomposition (TODO: M2)

```rust
use tenrso_decomp::{tucker_hosvd, tucker_hooi};

// HOSVD (one-pass, fast)
let tucker = tucker_hosvd(&tensor, &[64, 64, 32])?;

// HOOI (iterative, more accurate)
let tucker = tucker_hooi(&tensor, &[64, 64, 32], max_iters=30, tol=1e-4)?;

println!("Core tensor: {:?}", tucker.core().shape());
println!("Factor matrices: {:?}", tucker.factors());
```

### Tensor Train (TODO: M2)

```rust
use tenrso_decomp::tt_svd;

// TT-SVD with truncation tolerance
let tt = tt_svd(&tensor, eps=1e-6)?;

println!("TT-ranks: {:?}", tt.ranks());
println!("Compression ratio: {:.2}×", tt.compression_ratio());
```

## API Reference

### CP Decomposition

```rust
pub fn cp_als<T>(
    tensor: &Array<T, IxDyn>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    nonneg: bool,
) -> Result<CpDecomposition<T>>
```

Computes CP decomposition using Alternating Least Squares.

**Parameters:**
- `tensor` - Input tensor
- `rank` - Number of components
- `max_iters` - Maximum iterations
- `tol` - Convergence tolerance
- `nonneg` - Enable non-negative constraints

**Returns:** `CpDecomposition` with factor matrices and metadata

### Tucker Decomposition

```rust
pub fn tucker_hosvd<T>(
    tensor: &Array<T, IxDyn>,
    ranks: &[usize],
) -> Result<TuckerDecomposition<T>>

pub fn tucker_hooi<T>(
    tensor: &Array<T, IxDyn>,
    ranks: &[usize],
    max_iters: usize,
    tol: f64,
) -> Result<TuckerDecomposition<T>>
```

Computes Tucker decomposition via HOSVD or HOOI.

### Tensor Train

```rust
pub fn tt_svd<T>(
    tensor: &Array<T, IxDyn>,
    eps: f64,
) -> Result<TtDecomposition<T>>
```

Computes Tensor Train decomposition with SVD truncation.

## Initialization Strategies

### CP-ALS Initialization

- **Random** - Random uniform/normal initialization
- **SVD-based** - Use leading left singular vectors
- **Leverage scores** - Importance sampling based initialization
- **NNSVD** - Non-negative SVD for non-negative CP

### Tucker Initialization

- **HOSVD** - Direct SVD of unfolded tensors
- **Random subspace** - Random initialization for HOOI

## Performance Targets

- **CP-ALS** (256³, rank 64): < 2s / 10 iterations (16-core CPU)
- **Tucker-HOOI** (512×512×128, ranks [64,64,32]): < 3s / 10 iterations
- **TT-SVD** (32⁶, eps=1e-6): < 2s build time, ≥10× memory reduction

## Examples

See `examples/` directory:
- `cp_als.rs` - CP decomposition example (TODO)
- `tucker.rs` - Tucker decomposition example (TODO)
- `tt.rs` - Tensor Train example (TODO)
- `reconstruction.rs` - Error analysis (TODO)

## Testing

```bash
# Run unit tests
cargo test

# Run benchmarks
cargo bench

# Property tests
cargo test --test properties
```

## Dependencies

- **tenrso-core** - Tensor types
- **tenrso-kernels** - MTTKRP, n-mode products
- **scirs2-core** - Array operations
- **scirs2-linalg** - SVD, QR decompositions
- **rayon** (optional) - Parallel execution

## Contributing

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Apache-2.0
