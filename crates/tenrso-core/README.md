# tenrso-core

> **Version:** 0.1.0-rc.1 | **Status:** [Alpha] — 138 tests passing (100%) | **Updated:** 2026-03-06

Core tensor types, axis metadata, views, and comprehensive operations for TenRSo.

## Overview

`tenrso-core` provides the foundational tensor abstraction for the TenRSo ecosystem. It defines:

- **Dense N-dimensional tensors** (`DenseND<T>`) with efficient memory layout
- **Axis metadata** with symbolic names and size tracking
- **Zero-copy views** and slicing operations
- **Unfold/fold** operations for mode-n matricization
- **Reshape, permute, and shape manipulation** transformations
- **Broadcasting** support across incompatible shapes
- **Fancy indexing** (boolean masks, index arrays, predicate filtering)
- **Linear algebra** (matmul, det, inv, solve, 37+ elementwise ops)
- **Activation functions** (relu, sigmoid, tanh, gelu, and more)
- **Advanced statistics** (mean, variance, std, median, quantile, percentile — global and axis-wise)
- **Convolution and FFT** operations
- **Serialization** via serde (feature-gated)
- **Unified tensor handle** supporting dense, sparse, and low-rank representations

## Features

- Dense tensor storage backed by `scirs2_core::ndarray_ext`
- Flexible axis metadata with symbolic naming
- Memory-efficient views without data copying
- Safe indexing with bounds checking; panic-free variants (`get`, `get_mut`)
- Generic over scalar types (f32, f64, complex, etc.)
- 189+ tensor operations implemented
- Optional serde support for serialization
- Full SCIRS2 policy compliance (no direct ndarray/rand usage)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-core = "0.1"

# With serialization support
tenrso-core = { version = "0.1", features = ["serde"] }
```

### Creating Tensors

```rust
use tenrso_core::{TensorHandle, AxisMeta};
use scirs2_core::ndarray_ext::Array;

// Create a 3D tensor
let data = Array::zeros(vec![10, 20, 30]);
let axes = vec![
    AxisMeta::new("batch", 10),
    AxisMeta::new("height", 20),
    AxisMeta::new("width", 30),
];
let tensor = TensorHandle::from_dense(data, axes);

println!("Rank: {}", tensor.rank());
println!("Shape: {:?}", tensor.shape());
```

### Axis Metadata

```rust
use tenrso_core::AxisMeta;

let axis = AxisMeta::new("channels", 128);
assert_eq!(axis.name, "channels");
assert_eq!(axis.size, 128);
```

### Tensor Views and Slicing

```rust
use tenrso_core::DenseND;

let tensor = DenseND::<f64>::zeros(&[100, 200, 300]);
let view = tensor.view();
// Zero-copy view into tensor data
```

### Unfold/Fold Operations

```rust
use tenrso_core::DenseND;

// Unfold along mode 1 (matricize)
let tensor = DenseND::<f64>::zeros(&[10, 20, 30]);
let matrix = tensor.unfold(1)?; // Shape: (20, 10*30) = (20, 300)

// Fold back to original shape
let reconstructed = DenseND::fold(&matrix, &[10, 20, 30], 1)?;
```

### Broadcasting

```rust
use tenrso_core::DenseND;

let a = DenseND::<f64>::ones(&[3, 1, 5]);
let b = DenseND::<f64>::ones(&[3, 4, 5]);
// a broadcasts to [3, 4, 5] for element-wise operations
let result = (a.broadcast_to(&[3, 4, 5])? + &b)?;
```

### Fancy Indexing

```rust
use tenrso_core::DenseND;

let tensor = DenseND::<f64>::from_elem(&[4, 4], 1.0);

// Boolean mask selection
let mask = vec![true, false, true, false];
let selected = tensor.select_mask(&mask, 0)?;

// Index array selection
let indices = vec![0usize, 2];
let selected = tensor.select_indices(&indices, 0)?;

// Filter by predicate
let positive = tensor.filter(|&x| x > 0.0)?;
```

### Statistics

```rust
use tenrso_core::DenseND;

let tensor = DenseND::<f64>::random_normal(&[100, 100], 0.0, 1.0);

let mean = tensor.mean();
let variance = tensor.variance();
let std_dev = tensor.std();
let median = tensor.median();
let q75 = tensor.quantile(0.75);

// Axis-wise
let col_means = tensor.mean_axis(0)?;
let row_stds = tensor.std_axis(1)?;
let row_medians = tensor.median_axis(0)?;
```

### Linear Algebra

```rust
use tenrso_core::DenseND;

let a = DenseND::<f64>::random_normal(&[32, 32], 0.0, 1.0);
let b = DenseND::<f64>::random_normal(&[32, 16], 0.0, 1.0);

let c = a.matmul(&b)?;         // Matrix multiplication
let det = a.det()?;            // Determinant
let inv = a.inv()?;            // Matrix inverse
let x = a.solve(&b)?;         // Linear solve A*x = b
```

### Activation Functions

```rust
use tenrso_core::DenseND;

let logits = DenseND::<f64>::random_normal(&[8, 10], 0.0, 1.0);

let activated_relu = logits.relu();
let activated_sigmoid = logits.sigmoid();
let activated_tanh = logits.tanh();
```

## Architecture

### Type Hierarchy

```
TensorHandle<T>          // Unified handle for all representations
├── TensorRepr<T>        // Enum of representations
│   ├── Dense(DenseND<T>)    // Dense storage
│   ├── Sparse(SparseND<T>)  // Sparse storage (from tenrso-sparse)
│   └── LowRank(LowRank<T>)  // Low-rank (CP/Tucker/TT)
└── Vec<AxisMeta>        // Axis metadata
```

### Dense Tensor Structure

```
DenseND<T>
├── data: Array<T, IxDyn>   // N-D array from scirs2_core
├── strides: Vec<usize>      // Memory strides
└── offset: usize            // Base offset for views
```

### Module Layout

```
src/
├── lib.rs           // Crate root, re-exports
├── types.rs         // Type aliases, AxisMeta, TensorRepr, TensorHandle
├── dense/
│   ├── mod.rs
│   ├── types.rs     // DenseND struct and core methods
│   ├── creation.rs  // Initialization methods
│   ├── shape_ops.rs // Reshape, permute, flatten, swapaxes, moveaxis
│   ├── combining.rs // Concatenate, stack, split, chunk
│   ├── indexing.rs  // Element access, fancy indexing
│   ├── algebra.rs   // Linear algebra operations
│   ├── statistics.rs // Statistical reductions
│   ├── elementwise.rs // Element-wise ops and activations
│   ├── comparison.rs  // Comparison and logical ops
│   └── manipulation.rs // Array manipulation (roll, flip, tile)
└── ops.rs           // High-level operation dispatch
```

## API Reference

### Public Types

- **`Axis`** — Type alias for `usize` (axis index)
- **`Rank`** — Type alias for `usize` (number of dimensions)
- **`Shape`** — SmallVec for tensor dimensions (optimized for ≤6D)
- **`AxisMeta`** — Metadata for a single axis (name + size)
- **`PadMode`** — Padding mode enum for convolution/FFT
- **`TensorRepr<T>`** — Enum of tensor representations (Dense/Sparse/LowRank)
- **`TensorHandle<T>`** — Main tensor handle
- **`SparseND<T>`** — Sparse tensor placeholder (full impl in tenrso-sparse)
- **`LowRank<T>`** — Low-rank tensor placeholder (full impl in tenrso-decomp)
- **`DenseND<T>`** — Dense N-dimensional tensor

### TensorHandle Methods

| Method | Description |
|--------|-------------|
| `rank() -> Rank` | Number of dimensions |
| `shape() -> Shape` | Tensor dimensions |
| `from_dense(data, axes) -> Self` | Create from DenseND with axis metadata |
| `from_dense_auto(data) -> Self` | Create with auto-named axes |
| `as_dense(&self) -> Option<&DenseND<T>>` | Borrow as dense |
| `as_dense_mut(&mut self) -> Option<&mut DenseND<T>>` | Mutably borrow as dense |

### DenseND Creation Methods

| Method | Description |
|--------|-------------|
| `zeros(shape)` | Zero-initialized tensor |
| `ones(shape)` | One-initialized tensor |
| `from_elem(shape, value)` | Constant-filled tensor |
| `from_array(array)` | Wrap existing ndarray |
| `from_vec(vec, shape)` | Create from flat Vec |
| `eye(n)` | Identity matrix |
| `arange(start, stop, step)` | Evenly spaced values |
| `linspace(start, stop, num)` | Linearly spaced values |
| `random_uniform(shape, low, high)` | Uniform random |
| `random_normal(shape, mean, std)` | Normal random |

### DenseND Shape Operations

| Method | Description |
|--------|-------------|
| `reshape(new_shape)` | Change shape (size must match) |
| `permute(axes)` | Permute dimensions |
| `flatten()` | Flatten to 1D |
| `ravel()` | Flatten to 1D (alias) |
| `swapaxes(ax1, ax2)` | Swap two axes |
| `moveaxis(src, dst)` | Move axis to new position |
| `unfold(mode)` | Mode-n matricization |
| `fold(matrix, shape, mode)` | Inverse matricization |
| `broadcast_to(shape)` | Broadcast to new shape |

### DenseND Indexing

| Method | Description |
|--------|-------------|
| `view()` | Immutable view |
| `view_mut()` | Mutable view |
| `get(indices)` | Panic-free element access |
| `get_mut(indices)` | Panic-free mutable element access |
| `select_mask(mask, axis)` | Boolean mask selection |
| `select_indices(indices, axis)` | Index array selection |
| `filter(predicate)` | Filter by predicate |

### DenseND Combining

| Method | Description |
|--------|-------------|
| `concatenate(tensors, axis)` | Concatenate along axis |
| `stack(tensors, axis)` | Stack along new axis |
| `split(axis, n)` | Split into n parts |
| `chunk(axis, size)` | Chunk into fixed-size pieces |

### DenseND Statistics

| Method | Description |
|--------|-------------|
| `sum()` / `sum_axis(ax)` | Total or axis-wise sum |
| `mean()` / `mean_axis(ax)` | Global or axis-wise mean |
| `min()` / `max()` | Global min/max |
| `variance()` / `variance_axis(ax)` | Variance |
| `std()` / `std_axis(ax)` | Standard deviation |
| `median()` / `median_axis(ax)` | Median |
| `quantile(q)` / `quantile_axis(q, ax)` | Quantile |
| `percentile(p)` / `percentile_axis(p, ax)` | Percentile |
| `argmax()` / `argmin()` | Index of max/min |
| `all()` / `any()` | Logical reductions |
| `count_if(pred)` | Count matching elements |

### DenseND Linear Algebra

| Method | Description |
|--------|-------------|
| `matmul(other)` | Matrix multiplication |
| `det()` | Determinant |
| `inv()` | Matrix inverse |
| `solve(b)` | Linear solve A*x = b |
| `diagonal()` | Extract diagonal |
| `trace()` | Sum of diagonal elements |

### DenseND Element-wise & Activations

| Method | Description |
|--------|-------------|
| `relu()` | ReLU activation |
| `sigmoid()` | Sigmoid activation |
| `tanh()` | Hyperbolic tangent |
| `clip(min, max)` | Value clipping |
| `map(f)` | Apply function to all elements |
| `map_inplace(f)` | In-place element mapping |
| `fill(value)` | Fill all elements with value |
| `gt/lt/gte/lte(scalar)` | Element-wise comparisons |
| `mask_gt/mask_lt(threshold)` | Threshold masking |
| `roll(shift, axis)` | Roll elements along axis |
| `flip(axis)` | Reverse along axis |
| `tile(reps)` | Tile/repeat array |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | Standard features |
| `serde` | Enable serialization/deserialization for all core types |
| `parallel` | Parallel execution via Rayon |

## Dependencies

- **scirs2-core** (mandatory) — Array operations, SIMD, random generation
- **scirs2-linalg** — SVD, QR, eigendecomposition, GEMM
- **smallvec** — Stack-optimized shape storage
- **num-traits** — Generic numeric traits
- **anyhow** — Error handling
- **thiserror** — Error type definitions
- **proptest** (dev) — Property-based testing
- **criterion** (dev) — Benchmarking

## Performance Considerations

- **Views are zero-copy**: Slicing creates views, not copies
- **SmallVec optimization**: Shapes up to 6D avoid heap allocation
- **Contiguous memory**: Default layout is C-contiguous (row-major)
- **SIMD operations**: Element-wise ops use scirs2_core SIMD paths
- **Optimized convolution**: im2col + GEMM approach (5-10x faster than naive)
- **Panic-free indexing**: `get`/`get_mut` for safe, unchecked-free element access

## Examples

See `examples/` directory:
- `basic_tensor.rs` — Creating and manipulating tensors
- `views.rs` — Zero-copy views and slicing
- `unfold_fold.rs` — Mode-n matricization

Run examples:
```bash
cargo run --example basic_tensor
cargo run --example views
cargo run --example unfold_fold
```

## Testing

```bash
# Run unit tests (138 passing)
cargo test

# Run with all features
cargo test --all-features

# Run property tests
cargo test --test properties

# Run benchmarks
cargo bench
```

## Contributing

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Apache-2.0
