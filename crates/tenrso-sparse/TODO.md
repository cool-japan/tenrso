# tenrso-sparse TODO

> **Milestone:** M3
> **Status:** COO, CSR, CSC, BCSR, CSF, HiCOO, Mask, and Property Tests complete! (105 tests passing)

---

## M3: Sparse Implementations - 🚧 IN PROGRESS

### COO (Coordinate) Format ✅ COMPLETE

- [x] Core structure
  - [x] `CooTensor<T>` with indices + values + shape
  - [x] From triplets constructor with validation
  - [x] Duplicate handling (deduplication with summation)
  - [x] Sorting (lexicographic order)

- [x] Operations
  - [x] Element access (push, indices, values)
  - [x] NNZ and density statistics
  - [x] To/from dense conversion
  - [x] Format conversion (to CSR)

**Implementation:** `src/coo.rs` (443 lines, 10 tests passing)

### CSR/CSC (Compressed Sparse Row/Column)

#### CSR ✅ COMPLETE

- [x] Core structure
  - [x] `CsrMatrix<T>` with row_ptr + col_indices + values
  - [x] Shape and nnz tracking
  - [x] Row access with zero-copy slicing

- [x] Operations
  - [x] SpMV (Sparse Matrix-Vector) with O(nnz) complexity
  - [x] SpMM (Sparse Matrix-Matrix, dense) with O(nnz * k) complexity - ✅ **COMPLETE**
  - [x] SpSpMM (Sparse-Sparse Matrix-Matrix) - ✅ **COMPLETE**

- [x] Conversion
  - [x] From COO
  - [x] To COO
  - [x] From/to dense
  - [x] CSR ↔ CSC transpose - ✅ **COMPLETE**

**Implementation:** `src/csr.rs` (~1388 lines, 31 tests passing)

#### CSC ✅ COMPLETE

- [x] Core structure
  - [x] `CscMatrix<T>` with col_ptr + row_indices + values
  - [x] Efficient column access with zero-copy slicing

- [x] Operations
  - [x] Matrix-Vector product (column-wise accumulation) with O(nnz) complexity
  - [x] SpMM (Sparse Matrix-Matrix, dense) with O(nnz * k) complexity - ✅ **COMPLETE**
  - [x] SpSpMM (Sparse-Sparse Matrix-Matrix) - ✅ **COMPLETE**

- [x] Conversion
  - [x] From COO
  - [x] To COO
  - [x] From/to CSR (transpose)
  - [x] From/to dense

**Implementation:** `src/csc.rs` (~965 lines, 19 tests passing)

### BCSR (Block CSR) ✅ COMPLETE

- [x] Core structure
  - [x] Block size specification (block_shape parameter)
  - [x] Block row pointers
  - [x] Dense blocks storage (contiguous row-major)

- [x] Operations
  - [x] Block SpMV (O(num_blocks * block_size))
  - [x] Block SpMM (O(num_blocks * r * c * k)) - ✅ **COMPLETE**
  - [x] Block access patterns (get_block method)
  - [x] From/to dense conversion
  - [x] To CSR conversion

**Implementation:** `src/bcsr.rs` (~967 lines, 16 tests passing)

### CSF (Compressed Sparse Fiber) ✅ COMPLETE - Feature-gated

- [x] Core structure
  - [x] Hierarchical fiber structure with tree-based organization
  - [x] Mode ordering (arbitrary permutation support)
  - [x] Pointer arrays per level (fptr + fids for each level)
  - [x] Error handling for invalid mode orders

- [x] Operations
  - [x] Fiber iteration (O(nnz) traversal)
  - [x] Conversion from COO (O(nnz × log(nnz)))
  - [x] Conversion to COO/dense
  - [x] Density computation
  - [x] Multi-dimensional support (tested up to 5D)

**Implementation:** `src/csf.rs` (689 lines, 11 unit tests + 3 doc tests passing)

### HiCOO (Hierarchical COO) ✅ COMPLETE - Feature-gated

- [x] Core structure
  - [x] Blocked coordinate format with hierarchical organization
  - [x] Block indices + within-block coords separation
  - [x] Block pointers for efficient access
  - [x] Flexible block shape specification

- [x] Operations
  - [x] Cache-blocked iteration (O(nnz) with better locality)
  - [x] Conversion from COO (O(nnz × log(nnz)))
  - [x] Conversion to COO/dense
  - [x] Density computation
  - [x] Block grouping and statistics

**Implementation:** `src/hicoo.rs` (569 lines, 10 unit tests + 3 doc tests passing)

### Masked Operations ✅ COMPLETE

- [x] Mask structure
  - [x] HashSet-based sparse index representation
  - [x] Boolean mask with efficient lookup (O(1) contains)
  - [x] Full/empty mask constructors

- [x] Set Operations
  - [x] Union (logical OR)
  - [x] Intersection (logical AND)
  - [x] Difference (AND NOT)

- [x] Utilities
  - [x] Density computation
  - [x] Sorted index iteration
  - [x] Shape validation

**Implementation:** `src/mask.rs` (350 lines, 11 tests passing)

- [ ] Masked einsum - PENDING (Future M4/planner integration)
  - [ ] Integration with tenrso-exec
  - [ ] Sparse output computation
  - [ ] Mixed sparse/dense inputs

### Format Conversion

- [ ] COO → CSR (parallel sort + scan)
- [ ] CSR → COO
- [ ] Dense → COO (threshold-based)
- [ ] COO → Dense

---

## Testing

- [x] Unit tests per format
  - [x] COO: 9 unit tests + 1 doc test
  - [x] CSR: 29 unit tests + 4 doc tests (including SpMV, SpMM, SpSpMM, and CSR↔CSC transpose)
  - [x] CSC: 19 unit tests + 3 doc tests (including matvec, SpMM, and SpSpMM)
  - [x] Mask: 11 unit tests + 1 doc test
- [x] Correctness vs dense baseline
  - [x] COO ↔ Dense roundtrips
  - [x] CSR ↔ Dense roundtrips
  - [x] CSC ↔ Dense roundtrips
- [x] Format conversion roundtrips
  - [x] COO ↔ CSR
  - [x] COO ↔ CSC
  - [x] CSR ↔ CSC (transpose)
- [x] Sparse operations
  - [x] SpMV (4 comprehensive tests in CSR)
  - [x] SpMM (6 comprehensive tests + doc test in CSR)
  - [x] SpSpMM (6 comprehensive tests + doc test in CSR) - ✅ **NEW!**
  - [x] CSC matvec (1 test - column-wise accumulation)
  - [x] CSC SpMM (5 comprehensive tests + doc test)
- [x] Masked operations
  - [x] Mask creation and validation (3 tests)
  - [x] Set operations (union, intersection, difference) (4 tests)
  - [x] Utilities and edge cases (4 tests)
- [x] Property tests (proptest-based) - ✅ **COMPLETE**
  - [x] Format roundtrip properties (6 tests)
    - [x] COO ↔ Dense preserves data
    - [x] CSR ↔ Dense preserves data
    - [x] CSC ↔ Dense preserves data
    - [x] COO ↔ CSR ↔ COO preserves data
    - [x] COO ↔ CSC ↔ COO preserves data
    - [x] CSR ↔ CSC ↔ CSR preserves data
  - [x] Sparse operations correctness (5 tests)
    - [x] CSR SpMV matches dense baseline
    - [x] CSR SpMM matches dense baseline
    - [x] CSC SpMM matches dense baseline
    - [x] CSC matvec matches dense baseline
    - [x] Handles duplicate indices via deduplication
  - [x] Mask set operation properties (5 tests)
    - [x] Union commutativity
    - [x] Intersection commutativity
    - [x] Union contains both operands
    - [x] Intersection is subset
    - [x] Difference properties
    - [x] Density correctness

**Current:** ✅ **211 tests passing** (149 unit + 16 property + 46 doc) with CSF feature enabled

- [x] Benchmarks - ✅ **COMPLETE**
  - [x] SpMV performance across formats (CSR, CSC, BCSR)
  - [x] SpMM vs dense GEMM (various densities)
  - [x] SpSpMM sparse-sparse multiplication
  - [x] Format conversion speed (COO→CSR, CSR→CSC, etc.)
  - [x] Sparse matrix addition (sparse_add_csr)
  - [x] Memory footprint analysis
  - [x] Format recommendation benchmarks
  - [x] Dense-to-sparse conversion with thresholding
  - [x] CSF operations (feature-gated)
  - [x] Reduction operations (sum, max, min, mean - global & axis-wise) - ✅ **NEW!**
  - **File:** `benches/sparse_ops.rs` (~679 lines, 13 benchmark groups)

---

## Documentation

- [x] Rustdoc for COO format
- [x] Rustdoc for CSR format
- [x] Rustdoc for CSC format
- [x] Rustdoc for BCSR format
- [x] Rustdoc for CSF/HiCOO formats (feature-gated)
- [x] Rustdoc for Mask format
- [x] Rustdoc for Reductions module - ✅ **NEW!**
- [x] SpMV/SpMM/SpSpMM complexity analysis
- [x] Examples for all formats
- [x] Format selection guide - ✅ **COMPLETE** (`FORMAT_GUIDE.md`, comprehensive 600+ line guide)
- [x] Performance characteristics - ✅ **COMPLETE** (in FORMAT_GUIDE.md)

---

## Dependencies

- tenrso-core - ✅ Available and in use
- scirs2-core - ✅ In use (ndarray_ext, numeric::Float)
- scirs2-linalg - ⏳ May be needed for advanced operations

---

**Last Updated:** 2025-11-06 (M3 COMPLETE + Enhanced with ops, utils, benchmarks!)

---

## Recent Updates (2025-11-04 PM - M3 Progress!)

### Session Accomplishments

1. **COO Format - COMPLETE** ✅
   - Full N-dimensional sparse tensor implementation
   - Validation, sorting, deduplication
   - Dense ↔ COO conversion
   - 10 tests passing

2. **CSR Format - COMPLETE** ✅
   - 2D sparse matrix with row-major storage
   - Zero-copy row access
   - COO ↔ CSR ↔ Dense conversions
   - 20 tests passing (including SpMV and SpMM tests)

3. **SpMV Operation - COMPLETE** ✅
   - Sparse Matrix-Vector multiply: y = A * x
   - O(nnz) complexity
   - Comprehensive tests:
     - Basic functionality
     - Empty rows handling
     - Shape validation
     - Identity matrix test

4. **SpMM Operation - COMPLETE** ✅
   - Sparse Matrix-Matrix multiply: C = A * B
   - O(nnz * k) complexity (k = number of columns in B)
   - Comprehensive tests:
     - Basic functionality
     - Single column (matches SpMV)
     - Identity matrix
     - Empty rows handling
     - Shape validation
     - Wide result matrices

5. **CSC Format - COMPLETE** ✅
   - Column-major sparse matrix storage
   - Efficient zero-copy column access
   - Column-wise matrix-vector product
   - Full conversion support (COO, CSR, Dense)
   - CSR ↔ CSC transpose operations
   - 8 comprehensive tests

6. **Masked Operations - COMPLETE** ✅
   - HashSet-based sparse index representation
   - O(1) membership lookup
   - Set operations (union, intersection, difference)
   - Full/empty mask constructors
   - Density computation and iteration
   - 11 comprehensive tests + doc test

7. **CSC SpMM - COMPLETE** ✅
   - Sparse Matrix-Matrix multiply for CSC format
   - Column-wise accumulation: C[:, k] += A[:, j] * B[j, k]
   - O(nnz * k) complexity (k = number of columns in B)
   - Cache-friendly column-major access pattern
   - 5 comprehensive tests + doc test

8. **Property Tests - COMPLETE** ✅
   - Added proptest dependency (v1.5)
   - 16 comprehensive property-based tests
   - Tests verify algebraic properties and correctness:
     - Format roundtrip conversions preserve data
     - Sparse operations match dense baselines
     - Mask set operations satisfy mathematical properties
   - Automatic handling of duplicate indices via deduplication

9. **BCSR (Block CSR) - COMPLETE** ✅
   - Block-based sparse matrix storage for block-structured matrices
   - 631 lines of implementation with comprehensive validation
   - Features:
     - Flexible block shape specification (block_shape parameter)
     - Dense block storage with contiguous row-major layout
     - Block SpMV operation (O(num_blocks * block_size))
     - Block access via get_block() method
     - Conversions: from/to dense, to CSR
     - Full error handling and validation
   - 9 comprehensive tests + doc test

10. **SpSpMM (Sparse-Sparse Matrix Multiply) - COMPLETE** ✅ **NEW!**
   - Sparse matrix multiplication with sparse output
   - ~90 lines added to CSR implementation
   - Features:
     - HashMap-based row-wise accumulation for efficient sparse result construction
     - O(m × nnz_per_row_A × nnz_per_row_B) complexity
     - Automatic zero filtering
     - Sorted column indices in result
     - Full error handling and validation
   - 6 comprehensive tests + doc test
   - Tests cover: basic multiply, identity, zeros, correctness vs dense, shape mismatch, accumulation
   - Total test count: **90 tests** (65 unit + 16 property + 9 doc)

11. **CSR ↔ CSC Transpose Conversion - COMPLETE** ✅ **NEW!**
   - Direct CSR ↔ CSC conversion without COO intermediate
   - ~160 lines added to CSR implementation
   - Features:
     - `to_csc()` - Converts CSR(A) to CSC(A^T) with O(nnz) complexity
     - `from_csc()` - Converts CSC(A) to CSR(A^T) with O(nnz) complexity
     - Efficient direct transpose without intermediate format
     - Full error handling and validation
   - 6 comprehensive tests + 2 doc tests
   - Tests cover: basic conversion, roundtrip, transpose correctness, empty matrix, identity
   - Total test count: **98 tests** (71 unit + 16 property + 11 doc)

12. **CSC SpSpMM - COMPLETE** ✅
   - Sparse-sparse matrix multiplication for CSC format
   - ~95 lines added to CSC implementation
   - Features:
     - Column-wise accumulation with HashMap for efficient sparse result construction
     - O(m × nnz_per_col_A × nnz_per_col_B) complexity
     - Automatic zero filtering and sorted row indices in result
     - Symmetric with CSR SpSpMM for format flexibility
   - 6 comprehensive tests + doc test
   - Tests cover: basic multiply, identity, zeros, correctness vs dense, shape mismatch, accumulation
   - Total test count: **105 tests** (77 unit + 16 property + 12 doc)

13. **BCSR Block SpMM - COMPLETE** ✅
   - Block-wise sparse matrix-matrix multiplication for BCSR format
   - ~105 lines added to BCSR implementation
   - Features:
     - Block-wise accumulation for efficient block-structured computation
     - O(num_blocks × r × c × k) complexity where (r,c) is block_shape, k is B.ncols()
     - Better cache locality than element-wise operations
     - Full error handling and validation
   - 7 comprehensive tests + doc test
   - Tests cover: basic multiply, identity, zeros, shape mismatch, single column (matches SpMV), wide result, correctness vs dense
   - Total test count: **113 tests** (84 unit + 16 property + 13 doc)

14. **CSF (Compressed Sparse Fiber) - COMPLETE** ✅ **NEW!**
   - Hierarchical N-dimensional sparse tensor format (feature-gated with "csf")
   - 689 lines of implementation with comprehensive documentation
   - Features:
     - Tree-based fiber organization with arbitrary mode ordering
     - Hierarchical pointer arrays (fptr + fids per level)
     - O(nnz × log(nnz)) construction from COO
     - O(nnz) fiber iteration with efficient index reconstruction
     - Full conversion support (COO, dense)
     - Multi-dimensional support (tested up to 5D)
   - 11 unit tests + 3 doc tests
   - Tests cover: basic construction, invalid mode orders, empty tensors, different mode orders, iteration, COO roundtrip, dense conversion, density, single element, fiber access, high-dimensional tensors

15. **HiCOO (Hierarchical COO) - COMPLETE** ✅ **NEW!**
   - Blocked coordinate format for cache-efficient sparse tensors (feature-gated with "csf")
   - 569 lines of implementation with comprehensive documentation
   - Features:
     - Hierarchical block organization (block coords + local coords)
     - Flexible block shape specification
     - O(nnz × log(nnz)) construction from COO with block sorting
     - O(nnz) iteration with better cache locality
     - Block-major ordering for improved memory access patterns
     - Full conversion support (COO, dense)
   - 10 unit tests + 3 doc tests
   - Tests cover: basic construction, invalid block shapes, empty tensors, iteration, COO roundtrip, dense conversion, density, single element, block grouping, high-dimensional tensors
   - Total test count: **105 tests** (with CSF feature enabled)

### Recent Enhancements (2025-11-06)

16. **Unified Error Handling - COMPLETE** ✅ **NEW!**
   - Comprehensive error module (`src/error.rs`, 267 lines)
   - Structured error types: `ValidationError`, `ShapeMismatchError`, `ConversionError`, `OperationError`, `IndexError`
   - `SparseError` top-level enum
   - `SparseResult<T>` type alias
   - 4 unit tests
   - Ready for future format-specific error conversions

17. **Sparse Utilities Module - COMPLETE** ✅ **NEW!**
   - Comprehensive utilities (`src/utils.rs`, 481 lines)
   - `SparsityStats` - sparsity analysis and classification
   - `FormatRecommendation` - intelligent format selection
   - `MemoryFootprint` - memory estimation for all formats
   - Performance estimation (FLOPs for SpMV/SpMM/SpSpMM)
   - Helper functions: `sort_coo_inplace`, `deduplicate_coo`, `is_sorted_lex`
   - `recommend_format()` - automatic format selection algorithm
   - 12 comprehensive unit tests

18. **Enhanced Sparse Operations - COMPLETE** ✅ **NEW!**
   - Unified operations interface (`src/ops.rs`, 718 lines from 5 lines!)
   - Trait system: `SparseMatrixOps<T>`, `SparseSparseOps<T>`, `SparseOps<T>`
   - Implementations for CSR and CSC matrices
   - Advanced operations:
     - `sparse_add_csr()` - sparse matrix addition with α/β scaling
     - `sparse_scale_csr()` - scalar multiplication
     - `sparse_transpose_csr()` - efficient transpose
     - `sparse_abs_csr()` - element-wise absolute value
     - `sparse_square_csr()` - element-wise square
     - `nnz_per_row_csr()` - nonzero counting
   - Helper functions: shape checking, output estimation
   - 13 unit tests + 6 doc tests

19. **Comprehensive Benchmark Suite - COMPLETE** ✅ **NEW!**
   - Enhanced benchmarks (`benches/sparse_ops.rs`, 507 lines from 318 lines)
   - 12 benchmark groups covering:
     - SpMV operations (CSR, CSC, format comparison)
     - SpMM operations (dense result)
     - SpSpMM operations (sparse result)
     - Format conversions (COO→CSR, CSR→CSC, etc.)
     - BCSR operations
     - Sparse matrix addition
     - Memory footprint analysis
     - Format recommendation performance
     - Dense-to-sparse conversion
     - CSF operations (feature-gated)
   - Throughput measurements for all operations
   - Multiple sizes and density levels

20. **Format Selection Guide - COMPLETE** ✅ **NEW!**
   - Comprehensive guide (`FORMAT_GUIDE.md`, ~600 lines)
   - Quick decision tree for format selection
   - Detailed format descriptions with strengths/weaknesses
   - Performance comparison tables
   - Memory usage analysis
   - Common usage patterns
   - Best practices and debugging tips
   - Algorithm complexity reference

21. **Sparse Tensor Reductions - COMPLETE** ✅ **NEW!**
   - Comprehensive reductions module (`src/reductions.rs`, 901 lines)
   - Global reductions:
     - `sum()` - O(nnz) sum of all elements
     - `product()` - O(1) or O(nnz) product with zero detection
     - `max()` - O(nnz) maximum with implicit zero handling
     - `min()` - O(nnz) minimum with implicit zero handling
     - `mean()` - O(nnz) average of all elements
   - Axis-wise reductions:
     - `sum_axis()` - O(nnz) sum along specified axis
     - `max_axis()` - O(nnz + result_size) max along axis
     - `min_axis()` - O(nnz + result_size) min along axis
     - `mean_axis()` - O(nnz) mean along axis
   - Features:
     - Handles implicit zeros correctly
     - Produces sparse output when possible
     - Multi-dimensional tensor support (tested up to 3D)
     - Generic over Float types with proper trait bounds
   - 16 comprehensive unit tests + 10 doc tests
   - Reduction benchmarks added (13th benchmark group)
   - Tests global and axis-wise reductions on 2D (100x100, 1000x1000) and 3D (10³, 20³, 50³) tensors
   - Multiple density levels (0.01, 0.05, 0.1)
   - Total test count: **211 tests** (149 unit + 16 property + 46 doc)

### Next Steps

1. **Masked Einsum Integration** - For planner (M4)
   - Integration with tenrso-exec
   - Sparse output computation
   - Mixed sparse/dense inputs
   - Hook into einsum planner

2. **Additional Enhancements** - Optional improvements
   - Parallel format conversions (rayon)
   - More element-wise operations
   - [x] Sparse tensor reductions (sum, max, min along axes) - ✅ **COMPLETE!**
   - Sparse tensor slicing/indexing

### Technical Notes

- All implementations use `scirs2_core` types (no direct `ndarray` ✅)
- Comprehensive error handling with `thiserror` ✅
- Unified error module (`error.rs`) with structured types ✅
- Full Rustdoc with complexity analysis ✅
- Test coverage includes edge cases and doc tests ✅
- **211 tests passing** (149 unit + 16 property + 46 doc) ✅ **UPDATED!**
- Comprehensive benchmark suite with 13 groups ✅ **UPDATED!**
- Format selection guide (FORMAT_GUIDE.md) ✅
- Advanced sparse operations (transpose, scale, abs, square, add) ✅
- Utilities module for sparsity analysis and format recommendation ✅
- Sparse tensor reductions (sum, max, min, mean - global & axis-wise) ✅ **NEW!**

---

## File Summary (2025-11-06)

| File | Lines | Description | Tests |
|------|-------|-------------|-------|
| `src/coo.rs` | 443 | COO N-D sparse tensor | 10 |
| `src/csr.rs` | 1388 | CSR sparse matrix + operations | 31 |
| `src/csc.rs` | 965 | CSC sparse matrix + operations | 19 |
| `src/bcsr.rs` | 967 | Block CSR sparse matrix | 16 |
| `src/csf.rs` | 689 | CSF N-D hierarchical (feature) | 14 |
| `src/hicoo.rs` | 569 | HiCOO hierarchical COO (feature) | 13 |
| `src/mask.rs` | 350 | Sparse boolean mask | 12 |
| `src/ops.rs` | **718** | **Unified operations interface** | **19** |
| `src/utils.rs` | **481** | **Sparse utilities & analysis** | **12** |
| `src/error.rs` | **267** | **Unified error handling** | **4** |
| `src/reductions.rs` | **901** | **Sparse tensor reductions** ✅ **NEW!** | **26** |
| `src/lib.rs` | 40 | Public API exports | - |
| `benches/sparse_ops.rs` | **679** | **Comprehensive benchmarks** (13 groups) | - |
| `tests/property_tests.rs` | ~300 | Property-based tests | 16 |
| **Total** | **~8,757** | **All M3 features complete!** | **211** |

**New Documentation:**
- `FORMAT_GUIDE.md` (~600 lines) - Comprehensive format selection guide ✅
