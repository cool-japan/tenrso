# Sparse Tensor Format Selection Guide

> **Last Updated:** 2025-11-04
> **For:** TenRSo Sparse Tensor Library v0.1.0-alpha.1

This guide helps you choose the optimal sparse tensor format for your use case.

---

## Quick Reference Table

| Format | Dimensions | Best For | Avoid When | Memory | Construction | SpMV/SpMM |
|--------|------------|----------|------------|--------|--------------|-----------|
| **COO** | N-D | Construction, conversion | Repeated operations | O(nnz) | O(1) push | O(nnz) unsorted |
| **CSR** | 2-D | Row-wise access, SpMV | Column access | O(nnz + m) | O(nnz log nnz) | O(nnz) |
| **CSC** | 2-D | Column-wise access | Row access | O(nnz + n) | O(nnz log nnz) | O(nnz) |
| **BCSR** | 2-D | Block-structured | Irregular sparsity | O(num_blocks × bs²) | O(nnz) | O(num_blocks × bs²) |
| **CSF** | N-D | MTTKRP, tensor ops | Simple 2-D | O(nnz + fibers) | O(nnz log nnz) | O(nnz) |
| **HiCOO** | N-D | Clustered nonzeros | Scattered sparsity | O(nnz + blocks) | O(nnz log nnz) | O(nnz) |

---

## Format Descriptions

### COO (Coordinate Format)

**Structure:** List of (indices, value) tuples

**When to Use:**
- **Constructing** sparse tensors incrementally
- **Converting** between formats
- **One-time operations** where format doesn't matter
- **Very sparse** tensors (< 1% density)

**Advantages:**
- ✅ Simple and flexible
- ✅ O(1) element insertion
- ✅ Natural for any sparsity pattern
- ✅ Easy to construct and modify

**Disadvantages:**
- ❌ Slow for repeated operations
- ❌ No spatial locality
- ❌ Requires sorting for some operations

**Example Use Cases:**
```rust
use tenrso_sparse::CooTensor;

// Building a sparse tensor incrementally
let mut tensor = CooTensor::zeros(vec![1000, 1000, 1000]).unwrap();
for (i, j, k, val) in data {
    tensor.push(vec![i, j, k], val).unwrap();
}

// Convert to optimized format for operations
let csr = CsrMatrix::from_coo(&tensor.to_2d());
```

**Performance Profile:**
- Construction: **Excellent** (O(1) per element)
- Element access: **Poor** (O(log n) with sorting, O(n) without)
- SpMV/SpMM: **Poor** (no cache locality)
- Memory: **Excellent** (minimal overhead)

---

### CSR (Compressed Sparse Row)

**Structure:** Row pointers + column indices + values

**When to Use:**
- **Row-wise** operations (SpMV: y = Ax)
- **Sparse matrix-vector** multiplication
- **Sparse matrix-dense matrix** multiplication
- Most **general 2-D** sparse operations

**Advantages:**
- ✅ Excellent SpMV performance
- ✅ Good cache locality for row operations
- ✅ Compact representation
- ✅ Industry standard format

**Disadvantages:**
- ❌ Poor column access
- ❌ Requires deduplication and sorting
- ❌ 2-D only

**Example Use Cases:**
```rust
use tenrso_sparse::CsrMatrix;

// Linear system solving: Ax = b
let y = csr.spmv(&x.view());

// Sparse-dense matrix multiply: C = A @ B
let c = csr.spmm(&b.view()).unwrap();

// Sparse-sparse multiply: C = A @ B
let c = csr.spspmm(&b).unwrap();
```

**Performance Profile:**
- Construction from COO: **Good** (O(nnz log nnz))
- SpMV: **Excellent** (O(nnz))
- SpMM: **Excellent** (O(nnz × k))
- SpSpMM: **Good** (O(nnz_A × nnz_B / m))
- Memory: **Excellent** (O(nnz + m))

---

### CSC (Compressed Sparse Column)

**Structure:** Column pointers + row indices + values

**When to Use:**
- **Column-wise** operations
- **Transpose** operations (CSR^T = CSC)
- **Column** slicing and access
- Algorithms that iterate over columns

**Advantages:**
- ✅ Excellent for column operations
- ✅ Efficient transpose with CSR
- ✅ Good for Cholesky/QR factorizations
- ✅ Complements CSR well

**Disadvantages:**
- ❌ Poor row access
- ❌ 2-D only
- ❌ Less common than CSR

**Example Use Cases:**
```rust
use tenrso_sparse::CscMatrix;

// Column-wise matrix-vector: y = A @ x
let y = csc.matvec(&x.view()).unwrap();

// Access columns efficiently
for col_idx in 0..csc.ncols() {
    let col = csc.column(col_idx);
    // Process column...
}

// Efficient transpose via CSR
let a_csr = /* ... */;
let at_csc = a_csr.to_csc(); // This is A^T in CSC format
```

**Performance Profile:**
- Construction from COO: **Good** (O(nnz log nnz))
- Column access: **Excellent** (O(nnz_col))
- SpMM: **Excellent** (O(nnz × k))
- Memory: **Excellent** (O(nnz + n))

---

### BCSR (Block Compressed Sparse Row)

**Structure:** Block row pointers + dense blocks

**When to Use:**
- **Block-structured** sparsity (e.g., FEM matrices)
- **Dense blocks** within sparse matrix
- **Cache optimization** for structured problems
- Small block sizes (2×2, 4×4, 8×8)

**Advantages:**
- ✅ Exploits block structure
- ✅ Better cache performance
- ✅ SIMD-friendly operations
- ✅ Reduced index storage

**Disadvantages:**
- ❌ Wasted space if blocks aren't dense
- ❌ Requires block-aligned structure
- ❌ More complex construction

**Example Use Cases:**
```rust
use tenrso_sparse::BcsrMatrix;

// FEM stiffness matrices (often 3×3 or 6×6 blocks)
let bcsr = BcsrMatrix::from_dense(&dense.view(), (3, 3), 1e-10).unwrap();

// Block-structured computations
let y = bcsr.spmv(&x.view()).unwrap();
let c = bcsr.spmm(&b.view()).unwrap();
```

**Performance Profile:**
- Construction: **Good** (O(nnz))
- SpMV: **Excellent** for aligned blocks (O(num_blocks × bs²))
- SpMM: **Excellent** for aligned blocks
- Memory: **Variable** (good for dense blocks, poor for sparse blocks)

**Block Size Recommendations:**
- **2×2**: Graphics, rotations
- **3×3**: 3D FEM, geometry
- **4×4**: Quaternions, small matrices
- **8×8**: Image blocks

---

### CSF (Compressed Sparse Fiber)

**Structure:** Hierarchical tree of compressed fibers

**When to Use:**
- **High-dimensional** tensors (≥3 modes)
- **MTTKRP** operations (tensor decomposition)
- **Mode-specific** access patterns
- **Tensor contractions**

**Advantages:**
- ✅ True N-D format
- ✅ Flexible mode ordering
- ✅ Good for MTTKRP
- ✅ Hierarchical iteration

**Disadvantages:**
- ❌ Complex construction
- ❌ Mode order matters
- ❌ Higher memory overhead than COO
- ❌ Requires careful tuning

**Example Use Cases:**
```rust
use tenrso_sparse::{CooTensor, CsfTensor};

// Tensor decomposition (CP-ALS, Tucker)
let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

// MTTKRP: optimized for specific mode ordering
for (indices, value) in csf.iter() {
    // Process in mode-ordered fashion
}

// Different mode orders for different operations
let csf_012 = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap(); // Mode 0 first
let csf_210 = CsfTensor::from_coo(&coo, &[2, 1, 0]).unwrap(); // Mode 2 first
```

**Performance Profile:**
- Construction: **Good** (O(nnz log nnz))
- Fiber iteration: **Excellent** (O(nnz))
- MTTKRP: **Excellent** with right mode order
- Memory: **Good** (O(nnz + fibers))

**Mode Ordering Guidelines:**
- Order by decreasing mode size for memory efficiency
- Order by access pattern for performance
- Experiment with different orders for your workload

---

### HiCOO (Hierarchical COO)

**Structure:** Block coordinates + local coordinates + values

**When to Use:**
- **Clustered** nonzeros (spatial locality)
- **Cache-sensitive** applications
- **Large tensors** with block structure
- **Streaming** operations

**Advantages:**
- ✅ Exploits spatial locality
- ✅ Better cache performance than COO
- ✅ Flexible block sizes
- ✅ N-D support

**Disadvantages:**
- ❌ Wasted space for scattered nonzeros
- ❌ Block size tuning required
- ❌ More complex than COO
- ❌ Construction overhead

**Example Use Cases:**
```rust
use tenrso_sparse::{CooTensor, HiCooTensor};

// Large tensor with clustered nonzeros
let hicoo = HiCooTensor::from_coo(&coo, &[16, 16, 16]).unwrap();

// Cache-friendly iteration
for (indices, value) in hicoo.iter() {
    // Processes blocks sequentially for cache locality
}
```

**Performance Profile:**
- Construction: **Good** (O(nnz log nnz))
- Iteration: **Excellent** (O(nnz) with cache locality)
- Memory: **Variable** (depends on clustering)

**Block Size Recommendations:**
- Start with cache line size (64 bytes)
- Tune based on tensor dimensions
- Larger blocks for clustered data
- Smaller blocks for scattered data

---

## Decision Trees

### 2-D Sparse Matrix

```
Is it block-structured? (e.g., FEM)
├─ YES → BCSR (with appropriate block size)
└─ NO
   └─ What's your primary operation?
      ├─ Construction/conversion → COO
      ├─ Row-wise SpMV → CSR
      ├─ Column-wise access → CSC
      └─ SpSpMM (sparse output) → CSR or CSC
```

### N-D Sparse Tensor (N ≥ 3)

```
What's your use case?
├─ Building/converting → COO
├─ Tensor decomposition (MTTKRP) → CSF (with optimized mode order)
├─ Clustered nonzeros → HiCOO (with tuned block size)
└─ General operations → CSF or COO
```

### Performance-Critical Code

```
Is sparsity < 10%?
├─ YES
│  └─ Is there block structure?
│     ├─ YES → BCSR
│     └─ NO → CSR/CSC based on access pattern
└─ NO (denser) → Consider dense format or hybrid
```

---

## Common Patterns

### Pattern 1: Incremental Construction → Operations

```rust
// Build in COO (flexible)
let mut coo = CooTensor::zeros(vec![n, n]).unwrap();
for (i, j, val) in data {
    coo.push(vec![i, j], val).unwrap();
}

// Convert to CSR for operations (fast)
let csr = CsrMatrix::from_coo(&coo);

// Perform operations
let y = csr.spmv(&x.view());
```

### Pattern 2: Format Switching

```rust
// CSR for row operations
let y = csr.spmv(&x.view());

// CSC for column operations (via transpose)
let csc = csr.to_csc();
let z = csc.matvec(&w.view()).unwrap();
```

### Pattern 3: Tensor Decomposition

```rust
// Build in COO
let coo = /* ... */;

// Convert to CSF with optimal mode order
let csf_mode0 = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
let csf_mode1 = CsfTensor::from_coo(&coo, &[1, 0, 2]).unwrap();
let csf_mode2 = CsfTensor::from_coo(&coo, &[2, 0, 1]).unwrap();

// Use appropriate CSF for each mode's MTTKRP
// (Each mode ordering optimizes different operations)
```

---

## Performance Tips

### General Guidelines

1. **Start with COO** for construction
2. **Convert to optimized format** before heavy operations
3. **Benchmark** your specific use case
4. **Consider density**: < 10% → sparse, > 50% → consider dense
5. **Profile cache behavior** for large problems

### Memory Considerations

- **COO**: Minimal overhead, best for very sparse (< 1%)
- **CSR/CSC**: ~2× overhead, best for 1-10% density
- **BCSR**: Variable, best when blocks are actually dense
- **CSF**: Higher overhead, justified for complex N-D operations
- **HiCOO**: Depends on clustering, tune block size

### Cache Optimization

- Use **BCSR** for block-structured matrices (better cache lines)
- Use **HiCOO** for tensors with spatial locality
- Consider **memory layout** for your access pattern
- **Block sizes** should align with cache line sizes (64 bytes)

---

## Benchmarking Your Use Case

Run benchmarks to find the best format:

```bash
# Run all benchmarks
cargo bench --features csf

# Run specific benchmark group
cargo bench --features csf -- spmv
cargo bench --features csf -- format_conversion

# Compare formats
cargo bench --features csf -- format_comparison
```

---

## Future Formats (Roadmap)

- **Hybrid formats**: Mix of dense and sparse regions
- **GPU-optimized formats**: ELL, COO on GPU
- **Distributed formats**: Partitioned for parallel computation
- **Adaptive formats**: Automatically switch based on sparsity pattern

---

## References

- [TACO: Tensor Algebra Compiler](http://tensor-compiler.org/)
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [Intel MKL Sparse BLAS](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-blas-routines.html)
- [Sparse Tensor Algebra](https://arxiv.org/abs/1802.10574)
