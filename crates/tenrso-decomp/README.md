# tenrso-decomp

Tensor decomposition methods: CP-ALS, Tucker-HOOI/HOSVD, TT-SVD, and advanced variants.

**Version:** 0.1.0-rc.1 | **Status:** RC.1 — 165 tests passing (2 ignored), 100% pass rate | **Last Updated:** 2026-03-06

## Overview

`tenrso-decomp` provides production-grade implementations of all major tensor decomposition algorithms:

### CP (Canonical Polyadic) Decompositions
- **cp_als** - Canonical Polyadic decomposition via Alternating Least Squares
- **cp_als_constrained** - CP-ALS with non-negativity, L2 regularization, and orthogonality constraints
- **cp_als_accelerated** - Accelerated CP-ALS with momentum/extrapolation
- **cp_completion** - CP-based tensor completion for missing data
- **cp_randomized** - Randomized CP decomposition for large-scale tensors
- **cp_als_incremental** - Incremental/online CP-ALS for streaming tensor data

### Tucker Decompositions
- **tucker_hosvd** - Higher-Order SVD (one-pass truncation)
- **tucker_hosvd_auto** - HOSVD with automatic rank selection (energy/threshold)
- **tucker_hooi** - Higher-Order Orthogonal Iteration (iterative refinement)
- **tucker_randomized** - Randomized Tucker for large-scale tensors
- **tucker_nonnegative** - Non-negative Tucker decomposition
- **tucker_completion** - Tucker-based tensor completion

### Tensor Train (TT) Decompositions
- **tt_svd** - Tensor Train decomposition via sequential SVD
- **tt_round** - Post-decomposition TT-rank reduction
- **tt_add** - TT format addition
- **tt_dot** - TT format inner product
- **tt_hadamard** - TT Hadamard (element-wise) product
- **TTMatrix::matvec** - TT matrix-vector product (MPO x MPS)
- **tt_matrix_from_diagonal** - Construct TT matrix from diagonal

### Rank Selection
- **compute_information_criterion** - AIC/BIC/MDL information criteria for rank selection
- **select_rank_auto** - Automatic rank selection via information criteria
- **cp_rank_cross_validation** - Cross-validation based CP rank selection

All methods support initialization strategies, convergence criteria, and reconstruction error tracking.

## Features

- Multiple initialization methods (random, random-normal, SVD, NNSVD, leverage scores)
- Non-negative factorization support
- L2 regularization and orthogonality constraints
- Convergence diagnostics with oscillation detection
- Automatic rank selection (energy-based, threshold-based, information criteria, cross-validation)
- Incremental/online CP-ALS for streaming data
- Reconstruction error tracking and compression ratio reporting
- Generic over scalar types (f32, f64)
- Parallel execution via Rayon

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-decomp = "0.1"
```

### CP Decomposition

```rust
use tenrso_decomp::cp_als;
use scirs2_core::ndarray_ext::Array;

let tensor = Array::zeros(vec![100, 200, 300]);

// CP-ALS with rank 64
let cp = cp_als(&tensor, 64, 50, 1e-4, Default::default())?;

println!("Reconstruction error: {}", cp.error());
println!("Final factors: {:?}", cp.factors());
```

### CP with Constraints

```rust
use tenrso_decomp::{cp_als_constrained, CpConstraints};

// Non-negative CP
let cp = cp_als_constrained(&tensor, 32, 100, 1e-4, Default::default(), CpConstraints::nonnegative())?;

// L2 regularized CP
let cp = cp_als_constrained(&tensor, 32, 100, 1e-4, Default::default(), CpConstraints::l2_regularized(0.01))?;
```

### Tucker Decomposition

```rust
use tenrso_decomp::{tucker_hosvd, tucker_hooi, tucker_hosvd_auto, RankSelection};

// HOSVD (one-pass, fast)
let tucker = tucker_hosvd(&tensor, &[64, 64, 32])?;

// HOOI (iterative, more accurate)
let tucker = tucker_hooi(&tensor, &[64, 64, 32], 30, 1e-4)?;

// Automatic rank selection (preserve 99% energy)
let tucker = tucker_hosvd_auto(&tensor, RankSelection::Energy(0.99))?;

println!("Core tensor: {:?}", tucker.core().shape());
println!("Factor matrices: {:?}", tucker.factors());
```

### Tensor Train

```rust
use tenrso_decomp::{tt_svd, tt_round, tt_add, tt_dot};

// TT-SVD with truncation tolerance
let tt = tt_svd(&tensor, 1e-6)?;

println!("TT-ranks: {:?}", tt.ranks());
println!("Compression ratio: {:.2}x", tt.compression_ratio());

// Post-decomposition rank reduction
let tt_rounded = tt_round(&tt, 1e-4)?;

// TT arithmetic
let tt_sum = tt_add(&tt_a, &tt_b)?;
let inner = tt_dot(&tt_a, &tt_b)?;
```

### Automatic Rank Selection

```rust
use tenrso_decomp::{select_rank_auto, compute_information_criterion, cp_rank_cross_validation};

// Information-criterion based rank selection
let rank = select_rank_auto(&tensor, 1..=50)?;

// Cross-validation for CP rank
let rank = cp_rank_cross_validation(&tensor, &[4, 8, 16, 32], 5)?;
```

### Incremental CP-ALS

```rust
use tenrso_decomp::{cp_als_incremental, IncrementalMode};

// Append mode: grow tensor along mode 0
let cp = cp_als_incremental(&existing_cp, &new_slice, IncrementalMode::Append { mode: 0 })?;

// Sliding window mode
let cp = cp_als_incremental(&existing_cp, &new_slice, IncrementalMode::SlidingWindow { window: 100, mode: 0 })?;
```

## API Reference

### CP Decomposition

```rust
pub fn cp_als<T>(
    tensor: &Array<T, IxDyn>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
) -> Result<CpDecomposition<T>>

pub fn cp_als_constrained<T>(
    tensor: &Array<T, IxDyn>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    constraints: CpConstraints,
) -> Result<CpDecomposition<T>>

pub fn cp_als_accelerated<T>(
    tensor: &Array<T, IxDyn>,
    rank: usize,
    max_iters: usize,
    tol: f64,
) -> Result<CpDecomposition<T>>

pub fn cp_completion<T>(
    tensor: &Array<T, IxDyn>,
    mask: &Array<bool, IxDyn>,
    rank: usize,
    max_iters: usize,
    tol: f64,
) -> Result<CpDecomposition<T>>

pub fn cp_randomized<T>(
    tensor: &Array<T, IxDyn>,
    rank: usize,
    oversample: usize,
) -> Result<CpDecomposition<T>>

pub fn cp_als_incremental<T>(
    existing: &CpDecomposition<T>,
    new_slice: &Array<T, IxDyn>,
    mode: IncrementalMode,
) -> Result<CpDecomposition<T>>
```

**CpDecomposition** fields:
- `factors: Vec<Array2<T>>` - Factor matrices
- `weights: Array1<T>` - Component weights
- `fit: T` - Final fit value (1 - relative error)
- `iters: usize` - Iterations to convergence
- `convergence: Option<ConvergenceInfo<T>>` - Detailed convergence diagnostics

### Tucker Decomposition

```rust
pub fn tucker_hosvd<T>(
    tensor: &Array<T, IxDyn>,
    ranks: &[usize],
) -> Result<TuckerDecomposition<T>>

pub fn tucker_hosvd_auto<T>(
    tensor: &Array<T, IxDyn>,
    selection: RankSelection,
) -> Result<TuckerDecomposition<T>>

pub fn tucker_hooi<T>(
    tensor: &Array<T, IxDyn>,
    ranks: &[usize],
    max_iters: usize,
    tol: f64,
) -> Result<TuckerDecomposition<T>>

pub fn tucker_randomized<T>(
    tensor: &Array<T, IxDyn>,
    ranks: &[usize],
    oversample: usize,
) -> Result<TuckerDecomposition<T>>

pub fn tucker_nonnegative<T>(
    tensor: &Array<T, IxDyn>,
    ranks: &[usize],
    max_iters: usize,
    tol: f64,
) -> Result<TuckerDecomposition<T>>

pub fn tucker_completion<T>(
    tensor: &Array<T, IxDyn>,
    mask: &Array<bool, IxDyn>,
    ranks: &[usize],
    max_iters: usize,
    tol: f64,
) -> Result<TuckerDecomposition<T>>
```

### Tensor Train

```rust
pub fn tt_svd<T>(
    tensor: &Array<T, IxDyn>,
    eps: f64,
) -> Result<TtDecomposition<T>>

pub fn tt_round<T>(
    tt: &TtDecomposition<T>,
    eps: f64,
) -> Result<TtDecomposition<T>>

pub fn tt_add<T>(
    a: &TtDecomposition<T>,
    b: &TtDecomposition<T>,
) -> Result<TtDecomposition<T>>

pub fn tt_dot<T>(
    a: &TtDecomposition<T>,
    b: &TtDecomposition<T>,
) -> Result<T>

pub fn tt_hadamard<T>(
    a: &TtDecomposition<T>,
    b: &TtDecomposition<T>,
) -> Result<TtDecomposition<T>>
```

### Rank Selection

```rust
pub fn compute_information_criterion<T>(
    tensor: &Array<T, IxDyn>,
    cp: &CpDecomposition<T>,
    criterion: InformationCriterion,
) -> T

pub fn select_rank_auto<T>(
    tensor: &Array<T, IxDyn>,
    rank_range: RangeInclusive<usize>,
) -> Result<usize>

pub fn cp_rank_cross_validation<T>(
    tensor: &Array<T, IxDyn>,
    candidate_ranks: &[usize],
    n_folds: usize,
) -> Result<usize>
```

## Initialization Strategies

### CP-ALS Initialization (`InitStrategy` enum)

- **Random** - Random uniform initialization
- **RandomNormal** - Random normal initialization
- **Svd** - Use leading left singular vectors of mode-n unfoldings
- **Nnsvd** - Non-negative SVD (Boutsidis & Gallopoulos 2008) for non-negative CP
- **LeverageScore** - Importance sampling based on statistical leverage scores

### Tucker Initialization (`RankSelection` enum)

- **Fixed(Vec<usize>)** - User-specified ranks per mode
- **Energy(f64)** - Preserve X fraction of spectral energy (0.0–1.0)
- **Threshold(f64)** - Singular value cutoff

## Performance Targets

- **CP-ALS** (256^3, rank 64): < 2s / 10 iterations (16-core CPU)
- **Tucker-HOOI** (512x512x128, ranks [64,64,32]): < 3s / 10 iterations
- **TT-SVD** (32^6, eps=1e-6): < 2s build time, >= 10x memory reduction

## Examples

See `examples/` directory:
- `cp_als.rs` - CP decomposition: basic, constrained, initialization strategies, convergence
- `tucker.rs` - Tucker HOSVD/HOOI: comparison, compression, orthogonality verification
- `tt.rs` - TT-SVD: 4D/6D tensors, compression analysis, method comparison
- `reconstruction.rs` - Reconstruction quality analysis across all methods

## Testing

```bash
# Run unit tests
cargo test

# Run benchmarks
cargo bench

# Property tests (19 tests)
cargo test --test properties

# Integration tests (14 tests)
cargo test --test '*'
```

**Test coverage (RC.1):** 165 passing (2 ignored), 100% pass rate
- Unit tests: 52+ (CP, Tucker, TT, utilities)
- Property tests: 19 (proptest framework)
- Integration tests: 14
- Zero `todo!()` / `unimplemented!()` macros

## Benchmarks

27 benchmark functions covering all decomposition methods:
- CP-ALS: 8 benchmarks (small/medium/large, initialization methods, constraints)
- Tucker: 6 benchmarks (HOSVD vs HOOI, asymmetric tensors)
- TT-SVD: 5 benchmarks (target benchmark, varying orders)
- TT-rounding: 4 benchmarks
- Reconstruction: 3 benchmarks + 1 compression analysis benchmark

## Dependencies

- **tenrso-core** - Tensor types
- **tenrso-kernels** - MTTKRP, n-mode products, CP/Tucker reconstruction
- **scirs2-core** - Array operations
- **scirs2-linalg** - SVD, QR decompositions, least-squares
- **rayon** (optional) - Parallel execution

## Contributing

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Apache-2.0
