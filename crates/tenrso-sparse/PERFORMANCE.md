# Performance Characteristics

> **Last Updated:** 2025-11-04
> **For:** TenRSo Sparse Tensor Library v0.1.0-alpha.1

This document provides detailed performance characteristics, complexity analysis, and optimization guidelines for the tenrso-sparse library.

---

## Table of Contents

1. [Complexity Summary](#complexity-summary)
2. [Memory Usage](#memory-usage)
3. [Operation Performance](#operation-performance)
4. [Format Conversion Costs](#format-conversion-costs)
5. [Scalability Analysis](#scalability-analysis)
6. [Optimization Guidelines](#optimization-guidelines)
7. [Benchmark Results](#benchmark-results)

---

## Complexity Summary

### Time Complexity

| Operation | COO | CSR | CSC | BCSR | CSF | HiCOO |
|-----------|-----|-----|-----|------|-----|-------|
| **Construction** |
| From triplets | O(nnz) | O(nnz log nnz) | O(nnz log nnz) | O(nnz) | O(nnz log nnz) | O(nnz log nnz) |
| Element insert | O(1) | N/A | N/A | N/A | N/A | N/A |
| **Element Access** |
| Random access | O(nnz) | O(log nnz_row) | O(log nnz_col) | O(num_blocks) | O(depth) | O(nnz) |
| Row/Column | O(nnz) | O(nnz_row) | O(nnz_col) | O(blocks_row) | N/A | N/A |
| **Operations** |
| SpMV | O(nnz) | O(nnz) | O(nnz) | O(nb × bs²) | N/A | N/A |
| SpMM | O(nnz × k) | O(nnz × k) | O(nnz × k) | O(nb × bs² × k) | N/A | N/A |
| SpSpMM | O(nnz₁ × nnz₂) | O(m × r₁ × r₂) | O(n × r₁ × r₂) | N/A | N/A | N/A |
| **Conversions** |
| To COO | O(nnz) | O(nnz) | O(nnz) | O(nnz) | O(nnz) | O(nnz) |
| To Dense | O(m × n) | O(m × n) | O(m × n) | O(m × n) | O(∏ dims) | O(∏ dims) |
| CSR ↔ CSC | O(nnz) | O(nnz) | O(nnz) | N/A | N/A | N/A |

**Legend:**
- `nnz` = number of nonzeros
- `m, n` = matrix dimensions
- `k` = number of columns in dense matrix for SpMM
- `nb` = number of blocks in BCSR
- `bs` = block size
- `r₁, r₂` = average row/column nonzeros
- `depth` = tree depth in CSF

### Space Complexity

| Format | Memory Usage | Overhead | Notes |
|--------|--------------|----------|-------|
| **COO** | 3 × nnz × sizeof(usize) + nnz × sizeof(T) | Minimal | ~24 bytes per nonzero (f64) |
| **CSR** | (nnz + m + 1) × sizeof(usize) + nnz × sizeof(T) | O(m) | ~16 bytes per nonzero + O(m) |
| **CSC** | (nnz + n + 1) × sizeof(usize) + nnz × sizeof(T) | O(n) | ~16 bytes per nonzero + O(n) |
| **BCSR** | (nb + br + 1) × sizeof(usize) + nb × bs² × sizeof(T) | O(br) | Depends on block density |
| **CSF** | O(nnz) + O(fibers) | O(fibers) | Varies with hierarchy depth |
| **HiCOO** | O(nnz) + O(num_blocks) | O(num_blocks) | Depends on block size |

**Notes:**
- `sizeof(usize)` = 8 bytes on 64-bit systems
- `sizeof(T)` = 4 bytes (f32) or 8 bytes (f64)
- `br` = number of block rows
- `fibers` = number of fibers at all levels

---

## Memory Usage

### Detailed Memory Breakdown

#### COO (Coordinate Format)

```
Memory = nnz × (ndim × sizeof(usize) + sizeof(T)) + shape × sizeof(usize)

For 2-D f64 matrix:
Memory ≈ nnz × 24 bytes + constant overhead
```

**Example:** 1M nonzeros in 10K × 10K matrix
- Indices: 1M × 2 × 8 = 16 MB
- Values: 1M × 8 = 8 MB
- Shape: negligible
- **Total: ~24 MB**

#### CSR (Compressed Sparse Row)

```
Memory = (nnz + m + 1) × sizeof(usize) + nnz × sizeof(T)

For 2-D f64 matrix:
Memory ≈ nnz × 16 bytes + m × 8 bytes
```

**Example:** 1M nonzeros in 10K × 10K matrix
- Row pointers: 10K × 8 = 80 KB
- Column indices: 1M × 8 = 8 MB
- Values: 1M × 8 = 8 MB
- **Total: ~16 MB** (33% less than COO)

#### BCSR (Block CSR)

```
Memory = (num_blocks + block_rows + 1) × sizeof(usize) +
         num_blocks × block_size² × sizeof(T)

Dense blocks: efficient
Sparse blocks: wasteful
```

**Example:** 1M nonzeros with 4×4 blocks, 100% block density
- Assuming 62,500 blocks (1M / 16)
- Block row pointers: negligible
- Block column indices: 62,500 × 8 = 500 KB
- Block values: 62,500 × 16 × 8 = 8 MB
- **Total: ~8.5 MB** (best case with dense blocks)

**Example:** Same, but 50% block density (wasted space)
- Same structure
- Block values: 62,500 × 16 × 8 = 8 MB (stored)
- But only 50% are nonzero
- **Total: ~8.5 MB** (wasted ~4 MB)

---

## Operation Performance

### SpMV (Sparse Matrix-Vector Multiply)

**Operation:** `y = A × x` where A is sparse, x and y are dense

#### CSR Performance

```rust
// Pseudo-code for CSR SpMV
for i in 0..m {
    let mut sum = 0.0;
    for j in row_ptr[i]..row_ptr[i+1] {
        sum += values[j] * x[col_indices[j]];
    }
    y[i] = sum;
}
```

**Complexity:** O(nnz)
- **Cache behavior:** Excellent for row pointers and values
- **Random access:** x[col_indices[j]] may cause cache misses
- **Vectorization:** Good (sequential values array)
- **Parallelization:** Perfect (independent rows)

**Expected Performance:**
- **Dense baseline:** ~2-4 GFLOP/s (memory-bound)
- **CSR SpMV:** 0.5-2 GFLOP/s (depends on sparsity)
- **Efficiency:** 25-80% of dense GEMM

**Factors Affecting Performance:**
1. **Sparsity pattern:** Regular patterns → better cache
2. **Row length variance:** Uniform rows → better load balancing
3. **Matrix size:** Larger → more cache misses
4. **Parallelization:** More cores → better throughput

#### CSC Performance

Similar to CSR but column-oriented:
- Good for `y = A^T × x`
- Accumulation pattern: `y[row_indices[j]] += values[j] * x[col]`
- May have **cache conflicts** in accumulation

---

### SpMM (Sparse Matrix-Matrix Multiply)

**Operation:** `C = A × B` where A is sparse, B is dense

#### CSR Performance

```rust
// Pseudo-code for CSR SpMM
for i in 0..m {
    for j in row_ptr[i]..row_ptr[i+1] {
        let col = col_indices[j];
        let val = values[j];
        for k in 0..n_cols_B {
            C[i][k] += val * B[col][k];
        }
    }
}
```

**Complexity:** O(nnz × k) where k = B.ncols()
- **Cache behavior:** Excellent for small k (B fits in cache)
- **SIMD potential:** High (inner loop over k)
- **Parallelization:** Good (rows independent)

**Expected Performance:**
- **Small k (k < 10):** Similar to SpMV per column
- **Medium k (10 < k < 100):** Better cache reuse
- **Large k (k > 100):** Approaches dense GEMM efficiency

**Optimization:** Loop tiling for cache
```rust
// Tiled version (conceptual)
for k_tile in 0..(k / tile_size) {
    for i in 0..m {
        // Process tile_size columns at once
    }
}
```

---

### SpSpMM (Sparse-Sparse Matrix Multiply)

**Operation:** `C = A × B` where both A and B are sparse

#### CSR Implementation

```rust
// HashMap-based accumulation
for i in 0..m {
    let mut row_acc = HashMap::new();
    for j_idx in row_ptr_A[i]..row_ptr_A[i+1] {
        let j = col_indices_A[j_idx];
        let a_val = values_A[j_idx];

        for k_idx in row_ptr_B[j]..row_ptr_B[j+1] {
            let k = col_indices_B[k_idx];
            let b_val = values_B[k_idx];
            *row_acc.entry(k).or_insert(0.0) += a_val * b_val;
        }
    }
    // Extract sorted row to CSR
}
```

**Complexity:** O(m × nnz_row_A × nnz_row_B)
- **Average case:** O(nnz_A × nnz_B / n) for random sparsity
- **Best case:** O(nnz_A + nnz_B) for disjoint patterns
- **Worst case:** O(m × n × min(nnz_row_A, nnz_row_B)) for dense result

**Expected Performance:**
- **Very sparse (< 1%):** Much faster than dense
- **Sparse (1-10%):** 2-10× faster than dense
- **Dense-ish (> 10%):** Consider dense GEMM

**Memory:** Temporary HashMaps ≈ O(nnz_result_row) per row

---

### BCSR Operations

#### Block SpMV

**Advantages:**
- Better **cache locality** (dense blocks)
- **SIMD-friendly** (contiguous data)
- Fewer **index lookups**

**Performance Gain:**
- 2×4× over CSR for **well-blocked** matrices
- **Negative** for irregular sparsity

**Block Size Tuning:**
```
2×2: Graphics, small elements
3×3: 3D FEM, geometry
4×4: Small dense blocks
8×8: Image processing
```

**Rule of thumb:** Block should be ≥ 80% dense

---

## Format Conversion Costs

### Conversion Matrix

|From ↓ To →| COO | CSR | CSC | BCSR | Dense |
|-----------|-----|-----|-----|------|-------|
| **COO** | - | O(nnz log nnz) | O(nnz log nnz) | O(nnz) | O(m×n) |
| **CSR** | O(nnz) | - | O(nnz) | O(nnz) | O(m×n) |
| **CSC** | O(nnz) | O(nnz) | - | N/A | O(m×n) |
| **BCSR** | O(nnz) | O(nnz) | N/A | - | O(m×n) |
| **Dense** | O(m×n) | O(m×n) | O(m×n) | O(m×n) | - |

### Conversion Performance Tips

1. **COO → CSR:** Requires sorting by (row, col)
   - Use when building incrementally
   - Amortize over multiple operations

2. **CSR ↔ CSC:** Efficient transpose O(nnz)
   - Essentially matrix transpose
   - No sorting required

3. **Avoid Dense ↔ Sparse:** Very expensive
   - Dense → Sparse: O(m × n) scan
   - Sparse → Dense: O(m × n) allocation + O(nnz) fill

**Cost-Benefit Analysis:**
```
If num_operations × operation_cost(original_format) >
   conversion_cost + num_operations × operation_cost(target_format)
Then: Convert to target format
```

**Example:**
```
10 SpMV operations on COO vs CSR:
COO: 10 × O(nnz) unsorted ≈ 10 × nnz × log(nnz)
CSR: O(nnz log nnz) convert + 10 × O(nnz) ≈ nnz × (log(nnz) + 10)

For nnz > 100: CSR is better
For nnz × log(nnz) > 10 × nnz: Always convert
```

---

## Scalability Analysis

### Strong Scaling (Fixed Problem Size)

#### SpMV Parallelization

```rust
// Parallel CSR SpMV (conceptual)
y.par_chunks_mut(chunk_size)
    .zip(row_ptr.par_chunks(chunk_size))
    .for_each(|(y_chunk, rows)| {
        // Each thread processes chunk_size rows
    });
```

**Expected Scaling:**
- **Small matrices (< 1K rows):** Limited parallelism
- **Medium matrices (1K-100K rows):** Good scaling to ~16 cores
- **Large matrices (> 100K rows):** Excellent scaling to 64+ cores

**Bottlenecks:**
1. **Memory bandwidth:** Often the limiting factor
2. **Load imbalance:** Irregular row lengths
3. **Cache conflicts:** Shared cache lines

### Weak Scaling (Fixed Problem Size per Core)

**Ideal case:** Linear scaling
- 2× cores → 2× problem size → same runtime

**Realistic case:** Sublinear due to:
- Memory bandwidth saturation
- Increased cache misses
- Synchronization overhead

---

### Large-Scale Performance

#### Million-Scale Matrices

**1M × 1M Matrix, 1% density (10M nonzeros)**

Operation | Time | Memory | Notes |
----------|------|--------|-------|
COO construction | ~100 ms | 240 MB | O(1) per element |
COO → CSR | ~200 ms | 160 MB → 160 MB | Sort + scan |
CSR SpMV | ~10 ms | 160 MB | Memory-bound |
CSR SpMM (k=10) | ~100 ms | 160 MB + 80 MB | Good cache reuse |
CSR SpSpMM | ~1-10 s | Variable | Depends on output density |

*Estimates based on typical hardware (2.5 GHz CPU, 50 GB/s memory bandwidth)*

#### Billion-Scale Tensors

**For N-D tensors with 1B elements, 0.1% density (1M nonzeros)**

- **COO:** Still manageable (~24 MB)
- **CSF:** Efficient with right mode order
- **HiCOO:** Excellent if clustered

**Memory limit:** ~1M nonzeros per GB of RAM (rough estimate)

---

## Optimization Guidelines

### 1. Choose Right Format

See [FORMAT_SELECTION_GUIDE.md](FORMAT_SELECTION_GUIDE.md)

### 2. Memory Layout

```rust
// Good: Sequential access
for i in row_ptr[row]..row_ptr[row+1] {
    sum += values[i] * x[col_indices[i]];
}

// Bad: Random access pattern
for i in 0..n {
    sum += get_element(row, i) * x[i]; // Slow!
}
```

### 3. Cache Optimization

**Block sizes should be cache-friendly:**
```
L1 cache: ~32 KB → blocks ≤ 64×64 doubles
L2 cache: ~256 KB → blocks ≤ 180×180 doubles
L3 cache: ~8 MB → blocks ≤ 1000×1000 doubles
```

### 4. Parallelization

**Row-wise parallelization (SpMV/SpMM):**
```rust
use rayon::prelude::*;

// Parallel SpMV
y.par_iter_mut()
    .enumerate()
    .for_each(|(i, y_i)| {
        *y_i = csr.row(i).dot(&x);
    });
```

**When to parallelize:**
- Matrix rows > 1000: Consider parallelization
- Matrix rows > 10,000: Definitely parallelize
- Row length variance < 2×: Good load balance

### 5. SIMD Optimization

**Auto-vectorization friendly patterns:**
```rust
// Good: Contiguous memory access
for i in 0..n {
    result[i] = a[i] * b[i]; // Auto-vectorizes
}

// Bad: Gather/scatter
for i in 0..n {
    result[indices[i]] += a[i] * b[i]; // May not vectorize
}
```

### 6. Memory Pre-allocation

```rust
// Good: Pre-allocate result
let mut result = Vec::with_capacity(expected_size);

// Bad: Incremental allocation
let mut result = Vec::new();
for _ in 0..n {
    result.push(value); // May reallocate
}
```

---

## Benchmark Results

Run benchmarks with:
```bash
cargo bench --features csf
```

### Expected Results (Representative)

**System:** AMD Ryzen 9 5950X, 64 GB RAM, Ubuntu 22.04

#### SpMV Performance

| Matrix Size | Density | CSR (ms) | CSC (ms) | Speedup vs Dense |
|-------------|---------|----------|----------|------------------|
| 1K × 1K | 1% | 0.02 | 0.02 | 100× |
| 1K × 1K | 5% | 0.08 | 0.08 | 25× |
| 10K × 10K | 1% | 1.5 | 1.6 | 40× |
| 10K × 10K | 5% | 6.0 | 6.2 | 10× |

#### SpMM Performance (k=10)

| Matrix Size | Density | CSR (ms) | Speedup vs Dense |
|-------------|---------|----------|------------------|
| 1K × 1K | 1% | 0.15 | 65× |
| 1K × 1K | 5% | 0.60 | 18× |
| 10K × 10K | 1% | 12 | 35× |
| 10K × 10K | 5% | 48 | 12× |

#### Format Conversion

| Operation | Matrix Size | Density | Time (ms) |
|-----------|-------------|---------|-----------|
| COO → CSR | 10K × 10K | 1% | 15 |
| CSR → CSC | 10K × 10K | 1% | 8 |
| CSR → COO | 10K × 10K | 1% | 3 |

*Note: Actual results vary by hardware and data patterns*

---

## Performance Debugging

### Profiling Tools

```bash
# CPU profiling
cargo flamegraph --bench sparse_ops

# Cache analysis
valgrind --tool=cachegrind target/release/sparse_ops

# Memory profiling
heaptrack target/release/sparse_ops
```

### Common Performance Issues

1. **Slow SpMV:** Check sparsity pattern regularity
2. **High memory:** Check for format efficiency
3. **Poor scaling:** Check load balance in parallel code
4. **Cache misses:** Consider BCSR or HiCOO

---

## References

- [Sparse BLAS Specification](https://www.netlib.org/blas/blast-forum/)
- [Intel MKL Sparse Performance](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/sparse-blas-routines.html)
- [TACO Performance Model](http://tensor-compiler.org/)
- [The Landscape of Parallel Computing Research](https://view.eecs.berkeley.edu/wiki/Dwarf_Mine)

---

**For format selection, see:** [FORMAT_SELECTION_GUIDE.md](FORMAT_SELECTION_GUIDE.md)

**For implementation details, see:** [Source code](src/)

**For benchmarking, run:** `cargo bench --features csf`
