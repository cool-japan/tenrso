# Sparse Format Selection Guide

> **TenRSo Sparse Tensor Formats**
> **Purpose:** Help you choose the optimal sparse format for your use case
> **Last Updated:** 2025-11-06

---

## Quick Decision Tree

```
Is your tensor dense (> 10% nonzeros)?
├─ YES → Use Dense (tenrso-core)
└─ NO → Continue

What dimensionality?
├─ 2D Matrix
│   ├─ Row-wise access pattern → CSR
│   ├─ Column-wise access pattern → CSC
│   ├─ Block-structured sparsity → BCSR
│   └─ Construction/flexible → COO
│
├─ 3D+ Tensor (< 0.01% density)
│   ├─ Very sparse, hierarchical access → CSF
│   ├─ Sparse with locality → HiCOO
│   └─ Construction/flexible → COO
│
└─ Boolean mask/subset → SparseMask
```

---

## Format Comparison Table

| Format | Best For | Density | Dims | Operations | Memory | Construction |
|--------|----------|---------|------|------------|--------|--------------|
| **COO** | Construction, flexibility | Any | N-D | Read, convert | O(nnz) | O(nnz) |
| **CSR** | Row operations, SpMV | < 10% | 2D | SpMV, SpMM, row access | O(nnz) | O(nnz log nnz) |
| **CSC** | Column operations | < 10% | 2D | SpMV, SpMM, col access | O(nnz) | O(nnz log nnz) |
| **BCSR** | Block-structured | 1-10% | 2D | Block SpMV/SpMM | O(blocks) | O(nnz) |
| **CSF** | Very sparse, MTTKRP | < 0.1% | 3D+ | Fiber iteration | O(nnz + fibers) | O(nnz log nnz) |
| **HiCOO** | Clustered nonzeros | < 1% | 3D+ | Cache-friendly ops | O(nnz + blocks) | O(nnz log nnz) |
| **Mask** | Boolean selection | Sparse | N-D | Set ops, indexing | O(nnz) | O(nnz) |

---

## Detailed Format Descriptions

### COO (Coordinate Format)

**Description:** Stores (index, value) pairs as lists.

**Structure:**
```rust
struct CooTensor<T> {
    indices: Vec<Vec<usize>>,  // [[i₀, i₁, ...], ...]
    values: Vec<T>,             // [v₀, v₁, ...]
    shape: Vec<usize>,          // [dim₀, dim₁, ...]
}
```

**Strengths:**
- ✅ Simple, flexible format
- ✅ Easy to construct incrementally
- ✅ Supports any dimensionality
- ✅ Efficient for adding elements
- ✅ Good intermediate format

**Weaknesses:**
- ❌ Slow for most operations
- ❌ No efficient random access
- ❌ Requires sorting for conversions
- ❌ Higher memory overhead

**Use Cases:**
- Building sparse tensors incrementally
- Intermediate format for conversions
- Data loading and I/O
- Prototyping and experimentation

**Example:**
```rust
use tenrso_sparse::CooTensor;

let mut coo = CooTensor::zeros(vec![100, 100, 100])?;
coo.push(vec![10, 20, 30], 5.0)?;
coo.push(vec![50, 60, 70], 3.0)?;

// Convert to optimized format
let csr = csr::CsrMatrix::from_coo(&coo)?;
```

---

### CSR (Compressed Sparse Row)

**Description:** Row-major sparse matrix format with compressed row pointers.

**Structure:**
```rust
struct CsrMatrix<T> {
    row_ptr: Vec<usize>,      // [0, nnz₀, nnz₀+nnz₁, ..., nnz]
    col_indices: Vec<usize>,  // Column indices
    values: Vec<T>,           // Nonzero values
    shape: (usize, usize),    // (nrows, ncols)
}
```

**Strengths:**
- ✅ Fast row access (O(1) per row)
- ✅ Excellent for SpMV (y = A*x)
- ✅ Cache-friendly row iteration
- ✅ Industry standard format
- ✅ Minimal memory overhead

**Weaknesses:**
- ❌ Slow column access
- ❌ 2D only
- ❌ Static structure (hard to modify)

**Use Cases:**
- Matrix-vector multiplication (SpMV)
- Row-wise iteration
- Linear solvers (iterative methods)
- Graph algorithms (adjacency matrices)
- PageRank, HITS, etc.

**Performance:**
- SpMV: O(nnz) time, **≥80% of dense BLAS**
- SpMM: O(nnz × k) time
- Row access: O(1) time

**Example:**
```rust
use tenrso_sparse::CsrMatrix;
use scirs2_core::ndarray_ext::array;

let csr = CsrMatrix::new(
    vec![0, 2, 5, 7],          // row_ptr
    vec![0, 1, 0, 1, 2, 1, 2], // col_indices
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], // values
    (3, 3),
)?;

let x = array![1.0, 2.0, 3.0];
let y = csr.spmv(&x.view())?; // Fast SpMV
```

---

### CSC (Compressed Sparse Column)

**Description:** Column-major sparse matrix format (transpose of CSR).

**Structure:**
```rust
struct CscMatrix<T> {
    col_ptr: Vec<usize>,      // [0, nnz₀, nnz₀+nnz₁, ..., nnz]
    row_indices: Vec<usize>,  // Row indices
    values: Vec<T>,           // Nonzero values
    shape: (usize, usize),    // (nrows, ncols)
}
```

**Strengths:**
- ✅ Fast column access (O(1) per column)
- ✅ Excellent for column operations
- ✅ Transpose of CSR (efficient conversion)
- ✅ Cache-friendly column iteration

**Weaknesses:**
- ❌ Slow row access
- ❌ 2D only
- ❌ Static structure

**Use Cases:**
- Column-wise operations
- Transpose operations (A^T)
- QR decomposition
- Least squares problems
- Statistical computations (column stats)

**Example:**
```rust
use tenrso_sparse::CscMatrix;

// Direct CSR ↔ CSC conversion (transpose)
let csr = /* ... */;
let csc = csr.to_csc(); // O(nnz) transpose

// Column-wise operations
let y = csc.matvec(&x.view())?;
```

---

### BCSR (Block Compressed Sparse Row)

**Description:** Block-based sparse matrix for block-structured sparsity patterns.

**Structure:**
```rust
struct BcsrMatrix<T> {
    block_shape: (usize, usize),  // (r, c) block size
    block_row_ptr: Vec<usize>,    // Block row pointers
    block_col_indices: Vec<usize>,// Block column indices
    blocks: Vec<T>,               // Dense r×c blocks
    shape: (usize, usize),        // Matrix shape
}
```

**Strengths:**
- ✅ Excellent for block-structured matrices
- ✅ Better cache locality than CSR
- ✅ SIMD-friendly dense blocks
- ✅ Reduced index overhead

**Weaknesses:**
- ❌ Only efficient for block sparsity
- ❌ More complex to construct
- ❌ Higher memory if not block-structured

**Use Cases:**
- Finite element methods (FEM)
- Computational fluid dynamics (CFD)
- Block-diagonal matrices
- Multi-physics simulations
- Structured grids

**Recommended Block Sizes:**
- Small problems: 2×2, 3×3
- Medium problems: 4×4, 8×8
- Large problems: Auto-detect from structure

**Example:**
```rust
use tenrso_sparse::BcsrMatrix;

// Create from dense with 4×4 blocks
let bcsr = BcsrMatrix::from_dense(
    &dense.view(),
    (4, 4),     // block_shape
    1e-14,      // threshold
)?;

// Block SpMV - cache-friendly
let y = bcsr.spmv(&x.view())?;
```

---

### CSF (Compressed Sparse Fiber)

**Description:** Hierarchical N-D sparse format organized as a tree of fibers.

**Structure:**
```rust
struct CsfTensor<T> {
    shape: Vec<usize>,           // Tensor shape
    mode_order: Vec<usize>,      // Mode permutation
    fptr: Vec<Vec<usize>>,       // Fiber pointers (per level)
    fids: Vec<Vec<usize>>,       // Fiber indices (per level)
    vals: Vec<T>,                // Leaf values
}
```

**Strengths:**
- ✅ Excellent for very sparse tensors (< 0.1%)
- ✅ Hierarchical fiber access
- ✅ Optimal for MTTKRP
- ✅ Customizable mode ordering
- ✅ Memory-efficient for high dimensions

**Weaknesses:**
- ❌ Complex construction
- ❌ Overhead for moderately sparse
- ❌ Harder to understand/debug

**Use Cases:**
- Tensor decompositions (CP, Tucker)
- MTTKRP (key kernel in CP-ALS)
- Very sparse high-dimensional data
- Tensor contractions with specific mode order
- Network/graph tensors

**Mode Ordering:**
- Natural `[0,1,2,...]`: Good default
- Custom: Optimize for access pattern
- Heuristic: Place densest mode first

**Example:**
```rust
use tenrso_sparse::CsfTensor;

// Create with mode order [0, 1, 2]
let csf = CsfTensor::from_coo(&coo, &[0, 1, 2])?;

// Efficient MTTKRP (tenrso-kernels integration)
// for cp_als, tucker_hooi, etc.
```

---

### HiCOO (Hierarchical COO)

**Description:** Block-hierarchical format for cache-efficient sparse tensor operations.

**Structure:**
```rust
struct HiCooTensor<T> {
    shape: Vec<usize>,              // Tensor shape
    block_shape: Vec<usize>,        // Block size per dimension
    block_coords: Vec<Vec<usize>>,  // Block coordinates
    block_ptrs: Vec<usize>,         // Block pointers
    local_coords: Vec<Vec<usize>>,  // Within-block coords
    values: Vec<T>,                 // Values
}
```

**Strengths:**
- ✅ Better cache locality than COO
- ✅ Good for clustered nonzeros
- ✅ Block-major iteration
- ✅ Flexible block sizes

**Weaknesses:**
- ❌ Overhead for uniform sparsity
- ❌ Block size selection critical

**Use Cases:**
- Sparse tensors with spatial locality
- Clustered nonzero patterns
- Scientific simulations
- Image/video processing tensors

**Block Size Selection:**
- Small tensors: 4×4×4
- Medium tensors: 8×8×8
- Large tensors: 16×16×16
- Heuristic: Maximize nonzeros per block

**Example:**
```rust
use tenrso_sparse::HiCooTensor;

// Create with 8×8×8 blocks
let hicoo = HiCooTensor::from_coo(&coo, &[8, 8, 8])?;

// Cache-friendly iteration
for (indices, value) in hicoo.iter() {
    // Process in block-major order
}
```

---

### SparseMask

**Description:** Boolean mask for sparse index sets.

**Structure:**
```rust
struct SparseMask {
    indices: HashSet<Vec<usize>>, // Sparse index set
    shape: Vec<usize>,             // Tensor shape
}
```

**Strengths:**
- ✅ O(1) membership testing
- ✅ Fast set operations
- ✅ Memory-efficient for boolean data
- ✅ Natural for masking operations

**Weaknesses:**
- ❌ Boolean only (no values)
- ❌ Hash overhead for very small masks

**Use Cases:**
- Masked einsum operations
- Sparse output masks
- Index selection
- Boolean tensor operations
- Subset selection

**Example:**
```rust
use tenrso_sparse::SparseMask;

let mut mask = SparseMask::new(vec![100, 100, 100])?;
mask.insert(vec![10, 20, 30])?;
mask.insert(vec![50, 60, 70])?;

// Fast membership test
if mask.contains(&[10, 20, 30]) {
    // Process element
}

// Set operations
let union = mask.union(&other_mask)?;
let intersection = mask.intersection(&other_mask)?;
```

---

## Performance Guidelines

### Memory Comparison

For a 1000×1000 matrix with 1% density (10,000 nonzeros):

| Format | Memory (bytes) | Ratio vs Dense |
|--------|----------------|----------------|
| Dense  | 8,000,000     | 1.0×           |
| COO    | 240,000       | 0.03× (33×)    |
| CSR    | 168,000       | 0.02× (48×)    |
| CSC    | 168,000       | 0.02× (48×)    |
| BCSR(4×4) | ~200,000   | 0.025× (40×)   |

Use `utils::MemoryFootprint` for precise estimates.

### Operation Performance

| Operation | COO | CSR | CSC | BCSR |
|-----------|-----|-----|-----|------|
| SpMV | Slow | **Fast** | Fast | **Fastest** (blocks) |
| SpMM | Slow | **Fast** | **Fast** | **Fast** (cache) |
| Transpose | O(nnz) | O(nnz) | O(1) | O(nnz) |
| Row access | O(nnz) | **O(1)** | O(nnz) | O(blocks) |
| Col access | O(nnz) | O(nnz) | **O(1)** | O(nnz) |
| Construction | **O(nnz)** | O(nnz log nnz) | O(nnz log nnz) | O(nnz) |

---

## Format Selection Algorithm

Use the `utils::recommend_format()` function:

```rust
use tenrso_sparse::utils::{recommend_format, SparsityStats};

let stats = SparsityStats::from_shape_nnz(vec![1000, 1000], 10_000);
let recommendation = recommend_format(&stats);

match recommendation {
    FormatRecommendation::Dense => /* Use tenrso-core */,
    FormatRecommendation::CSR => /* Use CsrMatrix */,
    FormatRecommendation::CSF { mode_order } => /* Use CsfTensor */,
    // ... etc
}
```

**Algorithm:**
1. Density ≥ 10% → **Dense**
2. 2D matrix → **CSR** (default) or **CSC** (column-wise)
3. 2D with blocks → **BCSR**
4. 3D+ very sparse (< 0.01%) → **CSF**
5. 3D+ sparse (< 1%) → **HiCOO**
6. Flexible/construction → **COO**

---

## Conversion Paths

### Efficient Conversions

```
COO ──────→ CSR (O(nnz log nnz))
 │          ↕ (O(nnz))
 ├────────→ CSC (O(nnz log nnz))
 │
 ├────────→ Dense (O(nnz + total))
 ├────────→ CSF (O(nnz log nnz))
 └────────→ HiCOO (O(nnz log nnz))
```

**Recommended Path:**
1. Build in **COO**
2. Convert to target format
3. Perform operations
4. Convert back if needed

### Benchmark Your Use Case

```bash
cargo bench --package tenrso-sparse --all-features
```

Benchmarks include:
- SpMV across formats
- SpMM across formats
- Format conversions
- Memory analysis
- Sparse addition

---

## Common Patterns

### Pattern 1: Iterative Linear Solver

```rust
// 1. Load/construct as COO
let coo = load_matrix()?;

// 2. Convert to CSR for SpMV
let a = CsrMatrix::from_coo(&coo)?;

// 3. Solve Ax = b iteratively
let mut x = initial_guess();
for _ in 0..max_iters {
    let y = a.spmv(&x.view())?;  // Fast SpMV
    x = update(x, y);
}
```

### Pattern 2: Tensor Decomposition

```rust
// 1. Load 3D tensor as COO
let coo = load_tensor()?;

// 2. Convert to CSF for MTTKRP
let csf = CsfTensor::from_coo(&coo, &[0, 1, 2])?;

// 3. Run CP-ALS (uses MTTKRP with CSF)
let cp = cp_als(&csf, rank, iterations)?;
```

### Pattern 3: Block-Structured Problem

```rust
// 1. Detect block structure
let stats = SparsityStats::from_shape_nnz(shape, nnz);
let (has_blocks, block_shape) = stats.has_block_structure(4);

// 2. Use BCSR if block-structured
if has_blocks {
    let bcsr = BcsrMatrix::from_dense(&dense, block_shape, threshold)?;
    let y = bcsr.spmv(&x.view())?;  // Vectorized blocks
}
```

---

## Tips and Best Practices

### Do's ✅

- **Profile first**: Measure before optimizing
- **Use utilities**: `recommend_format()`, `MemoryFootprint`
- **Batch conversions**: Convert once, use many times
- **Consider access pattern**: Row vs column operations
- **Check density**: Dense may be faster > 10%

### Don'ts ❌

- **Don't guess format**: Use heuristics or profile
- **Don't convert repeatedly**: Cache the result
- **Don't ignore structure**: Use BCSR for blocks
- **Don't use COO for operations**: Convert first
- **Don't forget CSF**: Excellent for very sparse

### Debugging Tips

1. **Check sparsity**: Use `density()` method
2. **Validate conversions**: Roundtrip tests
3. **Profile operations**: Use `criterion` benchmarks
4. **Visualize structure**: Convert small tensor to dense
5. **Test incrementally**: Start with COO, then optimize

---

## Summary

**Quick Reference:**

| If you need... | Use... |
|----------------|--------|
| Construction | COO |
| SpMV (row-wise) | CSR |
| SpMV (column-wise) | CSC |
| Block operations | BCSR |
| Very sparse 3D+ | CSF |
| Clustered 3D+ | HiCOO |
| Masked operations | SparseMask |
| Maximum flexibility | COO |

**When in doubt:** Start with COO, measure, then optimize.

For more details, see:
- `src/*/mod.rs` - Implementation details
- `benches/sparse_ops.rs` - Performance benchmarks
- `tests/property_tests.rs` - Correctness guarantees
- `examples/` - Usage examples (coming soon)

---

**Questions? Open an issue:** https://github.com/cool-japan/tenrso/issues
