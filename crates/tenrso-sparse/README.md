# tenrso-sparse

Sparse tensor formats and operations for TenRSo.

## Overview

`tenrso-sparse` provides efficient sparse tensor storage formats and operations:

- **COO** - Coordinate format (triplets)
- **CSR/CSC** - Compressed Sparse Row/Column
- **BCSR** - Block Compressed Sparse Row
- **CSF** - Compressed Sparse Fiber (true N-D sparsity, feature-gated)
- **HiCOO** - Hierarchical COO (feature-gated)
- **Masked operations** - Einsum with sparse masks
- **Sparse-dense mixing** - Hybrid tensor contractions

## Features

- Multiple sparse formats optimized for different access patterns
- Efficient conversion between formats
- Sparse matrix operations (SpMM, SpSpMM)
- Masked einsum for selective computation
- Sparsity statistics (nnz, density)
- Generic over scalar types

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-sparse = "0.1"

# Enable CSF/HiCOO formats
tenrso-sparse = { version = "0.1", features = ["csf"] }
```

### COO Format (TODO: M3)

```rust
use tenrso_sparse::Coo;

// Create sparse tensor from triplets
let indices = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
let values = vec![1.0, 2.0, 3.0];
let shape = vec![10, 10, 10];

let coo = Coo::from_triplets(indices, values, shape)?;
println!("NNZ: {}", coo.nnz());
println!("Density: {:.2}%", coo.density() * 100.0);
```

### Masked Einsum (TODO: M3)

```rust
use tenrso_sparse::MaskPack;
use tenrso_exec::einsum_ex;

// Create mask for selective computation
let mask = MaskPack::from_indices(sparse_indices);

// Compute only masked elements
let result = einsum_ex::<f32>("ij,jk->ik")
    .inputs(&[A, B])
    .hints(&ExecHints {
        mask: Some(mask),
        prefer_sparse: true,
        ..Default::default()
    })
    .run()?;
```

## Sparse Formats

### COO (Coordinate)
- **Use case:** Construction, format conversion
- **Access:** O(nnz) scan
- **Memory:** 3 × nnz (indices + values)

### CSR/CSC (Compressed Sparse Row/Column)
- **Use case:** Matrix-vector, matrix-matrix products
- **Access:** O(1) row/col access, O(nnz) iteration
- **Memory:** 2 × nnz + nrows/ncols

### BCSR (Block CSR)
- **Use case:** Block-structured sparsity
- **Access:** Block-level operations
- **Memory:** Efficient for dense blocks

### CSF (Compressed Sparse Fiber)
- **Use case:** True N-dimensional sparsity
- **Access:** Fiber-based iteration
- **Memory:** Hierarchical compression
- **Feature:** Requires `csf` feature flag

## API Reference

### COO Format

```rust
pub struct Coo<T> {
    pub indices: Vec<Vec<usize>>,
    pub values: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> Coo<T> {
    pub fn from_triplets(indices, values, shape) -> Result<Self>;
    pub fn nnz(&self) -> usize;
    pub fn density(&self) -> f64;
    pub fn to_csr(&self) -> Result<Csr<T>>;
}
```

### Masked Operations

```rust
pub struct MaskPack {
    // Sparse mask representation
}

impl MaskPack {
    pub fn from_indices(indices: Vec<Vec<usize>>) -> Self;
    pub fn from_dense_bool(mask: &Array<bool, IxDyn>) -> Self;
}
```

## Performance

- **SpMM**: > 70% of dense GEMM (10% density)
- **Masked einsum**: ≥ 5× speedup vs dense (90% zeros)
- **Format conversion**: Optimized with parallel sorting

## Examples

See `examples/` directory (TODO):
- `coo_basics.rs` - COO format usage
- `csr_operations.rs` - CSR matrix operations
- `masked_einsum.rs` - Selective computation
- `format_conversion.rs` - Converting between formats

## Feature Flags

- `default` = `["parallel"]` - Parallel operations
- `parallel` - Multi-threaded format conversion
- `csf` - Enable CSF and HiCOO formats

## Dependencies

- **tenrso-core** - Tensor types
- **scirs2-core** - Array operations
- **scirs2-sparse** - Sparse matrix primitives (optional)
- **indexmap** - Efficient hash maps
- **rayon** (optional) - Parallel operations

## License

Apache-2.0
