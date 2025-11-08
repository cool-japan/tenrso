# tenrso-core

Core tensor types, axis metadata, views, and basic operations for TenRSo.

## Overview

`tenrso-core` provides the foundational tensor abstraction for the TenRSo ecosystem. It defines:

- **Dense N-dimensional tensors** with efficient memory layout
- **Axis metadata** with symbolic names and size tracking
- **Zero-copy views** and slicing operations
- **Unfold/fold** operations for mode-n matricization
- **Reshape and permute** transformations
- **Unified tensor handle** supporting multiple representations

## Features

- Dense tensor storage backed by `scirs2_core::ndarray_ext`
- Flexible axis metadata with symbolic naming
- Memory-efficient views without data copying
- Safe indexing with bounds checking
- Generic over scalar types (f32, f64, complex, etc.)
- Optional serde support for serialization

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-core = "0.1"
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

### Tensor Views (TODO: M1)

```rust
use tenrso_core::DenseND;

let tensor = DenseND::zeros(&[100, 200, 300]);
let view = tensor.slice(&[s![0..10], s![..], s![50..100]]);
// Zero-copy view into subset of data
```

### Unfold/Fold Operations (TODO: M1)

```rust
use tenrso_core::DenseND;

// Unfold along mode 1 (matricize)
let tensor = DenseND::zeros(&[10, 20, 30]);
let matrix = tensor.unfold(1)?; // Shape: (20, 10*30)

// Fold back to original shape
let reconstructed = matrix.fold(&[10, 20, 30], 1)?;
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

### Dense Tensor Structure (TODO: M1)

```
DenseND<T>
├── data: Array<T, IxDyn>   // N-D array from scirs2_core
├── strides: Vec<usize>      // Memory strides
└── offset: usize            // Base offset for views
```

## API Reference

### Types

- **`Axis`** - Type alias for `usize` (axis index)
- **`Rank`** - Type alias for `usize` (number of dimensions)
- **`Shape`** - SmallVec for tensor dimensions (optimized for ≤6D)
- **`AxisMeta`** - Metadata for a single axis (name + size)
- **`TensorRepr<T>`** - Enum of tensor representations
- **`TensorHandle<T>`** - Main tensor handle

### Key Methods

**TensorHandle:**
- `rank() -> Rank` - Get number of dimensions
- `shape() -> Shape` - Get tensor shape
- `from_dense(data, axes) -> Self` (TODO)
- `from_sparse(data, axes) -> Self` (TODO)
- `from_lowrank(factors, axes) -> Self` (TODO)

**DenseND** (TODO: M1):
- `zeros(shape) -> Self` - Create zero-initialized tensor
- `ones(shape) -> Self` - Create one-initialized tensor
- `from_array(array) -> Self` - Create from ndarray
- `view() -> DenseView<T>` - Get immutable view
- `view_mut() -> DenseMut<T>` - Get mutable view
- `slice(&[SliceInfo]) -> DenseView<T>` - Slice tensor
- `unfold(mode) -> Array2<T>` - Mode-n matricization
- `reshape(new_shape) -> Result<Self>` - Change shape
- `permute(axes) -> Self` - Permute dimensions

## Feature Flags

- `default` - Standard features
- `serde` - Enable serialization/deserialization

## Dependencies

- **scirs2-core** (mandatory) - Array operations and random generation
- **smallvec** - Stack-optimized shape storage
- **num-traits** - Generic numeric traits
- **anyhow** - Error handling
- **thiserror** - Error type definitions

## Performance Considerations

- **Views are zero-copy**: Slicing creates views, not copies
- **SmallVec optimization**: Shapes up to 6D avoid heap allocation
- **Contiguous memory**: Default layout is C-contiguous (row-major)
- **SIMD operations**: Element-wise ops use scirs2_core SIMD paths

## Examples

See `examples/` directory:
- `basic_tensor.rs` - Creating and manipulating tensors (TODO)
- `views.rs` - Zero-copy views and slicing (TODO)
- `unfold_fold.rs` - Mode-n matricization (TODO)

Run examples:
```bash
cargo run --example basic_tensor
```

## Testing

```bash
# Run unit tests
cargo test

# Run with all features
cargo test --all-features

# Run property tests
cargo test --test properties
```

## Contributing

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Apache-2.0
