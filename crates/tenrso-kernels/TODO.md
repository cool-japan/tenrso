# tenrso-kernels TODO

> **Milestone:** M1 (Complete)
> **Version:** 0.1.0-rc.1
> **Status:** RC.1 — 264 tests passing, 7 ignored (100%)
> **Last Updated:** 2026-03-06

---

## Current Status Summary

**Test Coverage:** 264 tests (100% passing), 7 ignored
- Unit tests: 132 (correctness + edge cases)
- Property tests (proptest): 40+ (mathematical invariants)
- Integration tests: 22+ (real-world CP-ALS, Tucker-HOOI workflows)
- Doc tests: 55+ (all API examples verified)

**Code Metrics:**
- Source code: ~10K lines (14 files)
- Benchmarks: 135+ individual benchmarks across 10 groups
- Examples: 3 comprehensive demonstrations

**Features Implemented:**
- [x] Core tensor kernels (Khatri-Rao, Kronecker, Hadamard, N-mode, MTTKRP)
- [x] Advanced operations (TTT, Outer products, Tucker operator)
- [x] Contractions and reductions
- [x] Complete statistical toolkit (14+ operations)
- [x] Multivariate analysis (covariance, correlation)
- [x] Randomized algorithms (randomized SVD, range finder)
- [x] TT operations (10 functions)
- [x] Parallel implementations
- [x] Cache-optimized variants (blocked/tiled)
- [x] Fused MTTKRP kernel

**Quality Metrics:**
- [x] 0 warnings (clippy strict mode)
- [x] 0 unsafe blocks
- [x] 100% documentation coverage
- [x] SCIRS2 policy compliant
- [x] Property-based testing (40+ property tests)
- [x] Comprehensive benchmarks (135+ benchmarks)
- [x] 0 todo!()/unimplemented!() calls

---

## M1: Kernel Implementations - COMPLETE

### Khatri-Rao Product - COMPLETE

- [x] Basic implementation
  - [x] Column-wise Kronecker product
  - [x] Handle arbitrary matrix sizes
  - [x] Validate input dimensions (same ncols)
  - [x] Efficient memory layout

- [x] Optimizations
  - [x] SIMD acceleration via scirs2_core
  - [x] Parallel column processing (rayon) - `khatri_rao_parallel`
  - [x] Pre-allocation strategies
  - [x] Cache-blocking for large matrices
    - [x] `khatri_rao_blocked` - Tiled/blocked execution
    - [x] `khatri_rao_blocked_parallel` - Parallel + blocked
    - [x] Configurable block size
    - [x] Automatic fallback to standard for small matrices
    - [x] 7 comprehensive tests including various block sizes

- [x] Testing
  - [x] Correctness tests (small matrices) - 5 tests
  - [x] Property tests (parallel matches serial)
  - [x] Edge cases (empty, single column, mismatched)
  - [ ] Benchmark against naive implementation - Optional future work

### Kronecker Product - COMPLETE

- [x] Basic implementation
  - [x] Tensor product of two matrices
  - [x] Block structure (a_ij x B)
  - [x] Efficient indexing

- [x] Optimizations
  - [x] SIMD element-wise operations
  - [x] Parallel row/block processing - `kronecker_parallel`
  - [x] Memory-efficient layout

- [x] Testing
  - [x] Correctness tests - 6 tests
  - [x] Property tests (parallel matches serial)
  - [x] Compare with known results (identity, vectors)
  - [ ] Performance benchmarks - Optional future work

### Hadamard Product - COMPLETE

- [x] Basic implementation
  - [x] Element-wise multiplication
  - [x] Support arbitrary dimensions (2D + ND)
  - [x] Shape validation
  - [x] Generic over array types

- [x] Optimizations
  - [x] Use scirs2_core SIMD ops (via ndarray)
  - [x] In-place variant - `hadamard_inplace`
  - [ ] Parallel iteration - Future (small benefit for this op)

- [x] Testing
  - [x] Correctness tests (various shapes) - 8 tests
  - [x] Property tests (commutativity, identity)
  - [x] Edge cases (zeros, large arrays)
  - [ ] Benchmarks - Optional future work

### N-Mode Product (TTM/TTT) - COMPLETE

- [x] Tensor-Matrix Product (TTM)
  - [x] Unfold tensor along mode
  - [x] Matrix multiplication
  - [x] Fold back to tensor
  - [x] Optimize for contiguous memory

- [x] Sequential multi-mode - `nmode_products_seq`
  - [x] Apply multiple matrices in sequence
  - [x] Efficient chaining

- [x] Tensor-Tensor Product (TTT) - `tensor_tensor_product`
  - [x] Contract two tensors along specified modes
  - [x] Handle multiple contraction modes
  - [x] Efficient index computation
  - [x] Support for multiple simultaneous contractions
  - [x] Outer product as special case (no contractions)
  - [x] Complete contraction (scalar result)
  - [x] 10 comprehensive tests including edge cases

- [x] Testing
  - [x] Correctness tests (all modes) - 6 tests
  - [x] Property tests (identity)
  - [x] Unfold/fold roundtrip tests
  - [ ] Benchmarks vs manual unfold+GEMM - Optional future work

### MTTKRP (Matricized Tensor Times Khatri-Rao Product) - COMPLETE

- [x] Core implementation
  - [x] Unfold tensor along mode
  - [x] Compute Khatri-Rao of factor matrices
  - [x] Matrix multiplication
  - [x] Return factor matrix

- [x] Optimizations
  - [x] Tiled/blocked iteration - `mttkrp_blocked`
  - [x] Parallel blocked version - `mttkrp_blocked_parallel`
  - [x] Cache-aware tile sizing
  - [x] Fused MTTKRP kernel
    - [x] `mttkrp_fused` - Avoids materializing full KR product
    - [x] `mttkrp_fused_parallel` - Parallel fused version
    - [x] Significantly reduced memory usage
    - [x] Column-to-multi-index mapping fixed (all previously ignored tests passing)
  - [ ] Fully fused + cache-blocking - Future optimization
  - [ ] SIMD inner loops - Future (using scirs2_core where possible)

- [x] Variants
  - [x] Dense MTTKRP - Complete
  - [x] Blocked MTTKRP - Complete
  - [x] Fused MTTKRP - Complete
  - [ ] Sparse MTTKRP (future M3)
  - [ ] Out-of-core MTTKRP (future M5)

- [x] Testing
  - [x] Correctness tests (all modes) - 5 tests
  - [x] Blocked matches standard - Verified
  - [x] Parallel matches serial - Verified
  - [x] Various tile sizes - Tested
  - [x] Large tensor tests (100^3+)
  - [ ] Performance benchmarks - Optional future work

### Outer Products - COMPLETE

- [x] Basic implementations
  - [x] 2D outer product - `outer_product_2`
  - [x] N-D outer product - `outer_product`
  - [x] Weighted outer product - `outer_product_weighted`

- [x] CP Reconstruction
  - [x] Sum of outer products - `cp_reconstruct`
  - [x] Optional weights support
  - [x] Efficient accumulation
  - [x] Parallel CP reconstruction - `cp_reconstruct_parallel`

- [x] Testing
  - [x] Correctness tests - 8 tests
  - [x] Various dimensionalities (2D, 3D, 4D+)
  - [x] CP reconstruction validation
  - [x] Weighted reconstruction
  - [x] Edge cases (single vector, empty)

- [ ] SIMD element operations - Future

### Tucker Operator - COMPLETE

- [x] Multi-mode products
  - [x] Auto-optimized execution order - `tucker_operator`
  - [x] Explicit ordering - `tucker_operator_ordered`
  - [x] HashMap-based factor specification

- [x] Tucker reconstruction
  - [x] Core + factors -> tensor - `tucker_reconstruct`
  - [x] Dimension validation
  - [x] Sequential application

- [x] Testing
  - [x] Single and multiple modes - 8 tests
  - [x] Empty factor map handling
  - [x] Dimension reduction validation
  - [x] Identity reconstruction
  - [x] 3D reconstruction tests
  - [x] Error cases (invalid modes, mismatched dims)

- [ ] Parallel mode application - Future
- [ ] Fused multi-mode products - Future
- [ ] Memory-efficient intermediate tensors - Future

---

## Statistical Toolkit - COMPLETE

### Reductions (14+)

- [x] sum, mean, min, max
- [x] variance, std
- [x] norm_l1, norm_l2, norm_frobenius
- [x] skewness, kurtosis

### Multivariate Analysis

- [x] covariance matrix
- [x] correlation matrix

### Testing

- [x] Correctness tests for all operations
- [x] Property tests (range validation, monotonicity)
- [x] Numerical stability tests

---

## Randomized Algorithms - COMPLETE

- [x] Randomized SVD (`randomized_svd`)
  - [x] O(mnk) complexity vs O(mn^2) for full SVD
  - [x] Configurable oversampling and power iterations
  - [x] Correctness verified against full SVD

- [x] Randomized range finder (`randomized_range_finder`)
  - [x] Returns approximate orthonormal basis for column space
  - [x] Configurable rank and oversampling

---

## TT (Tensor-Train) Operations - COMPLETE

- [x] 10 tensor-train functions implemented
- [x] Integration with N-mode products

---

## Cross-Cutting Concerns

### Performance - COMPLETE

- [x] Benchmark suite (criterion)
  - [x] Khatri-Rao (various sizes) - Serial & parallel variants
  - [x] Kronecker (various sizes) - Serial & parallel variants
  - [x] Hadamard (N-D tensors) - Allocating & in-place variants
  - [x] N-mode product (3D, 4D, 5D) - Single & sequential modes
  - [x] MTTKRP (realistic CP scenarios) - Standard, blocked, parallel
  - [x] Outer products (2D, N-D, CP reconstruction)
  - [x] Tucker operator (single, multiple modes, reconstruction)
  - [x] Statistics and randomized algorithms

- [x] Performance results documented (PERFORMANCE.md)
  - [x] Khatri-Rao: 1.5 Gelem/s serial, 3x parallel speedup at 500x32
  - [x] MTTKRP: 13.3 Gelem/s peak (blocked parallel, 50^3)
  - [x] N-mode: >5 Gelem/s sustained
  - [x] Memory bandwidth: 70-80% utilization (Hadamard in-place)
  - [x] All operations scale well from small to large tensors

- [ ] Advanced profiling (Optional future work)
  - [ ] Flamegraphs for hot paths
  - [ ] Cache miss analysis
  - [ ] SIMD instruction coverage
  - [ ] Thread scaling analysis on 16+ cores

### Testing - COMPLETE

- [x] Unit tests - 132 passing
  - [x] Small matrix examples
  - [x] Known result validation
  - [x] Edge cases (1x1, empty)
  - [x] Type variations (f32, f64)

- [x] Property tests (proptest) - 40+ implemented
  - [x] Khatri-Rao: column structure
  - [x] Kronecker: block structure
  - [x] Hadamard: commutativity, associativity, distributivity
  - [x] N-mode: identity, associativity, chaining
  - [x] MTTKRP: mathematical correctness
  - [x] Contractions: bilinearity, symmetry
  - [x] Reductions: sum correctness, variance, norms
  - [x] Numerical stability tests

- [x] Integration tests - 22+ implemented
  - [x] Real-world workflows (CP-ALS, Tucker-HOOI simulations)
  - [x] Multi-operation pipelines
  - [x] Error analysis workflows
  - [x] Validation and convergence tracking patterns
  - [x] Cross-crate compatibility (tenrso-core integration)
  - [x] Large tensor scenarios (up to 100^3)
  - [ ] Use with tenrso-decomp - Pending M2

### Documentation - COMPLETE

- [x] Rustdoc for all public functions
  - [x] Mathematical definitions
  - [x] Complexity analysis (O(n) notation)
  - [x] Usage examples (55+ doc tests passing)
  - [x] Performance notes and recommendations

- [x] Module documentation
  - [x] Algorithm descriptions
  - [x] References to literature (Kolda & Bader, etc.)
  - [x] Optimization strategies (SIMD, parallel, blocking)
  - [x] Quick start guide in crate root

- [x] Examples - 3 comprehensive examples
  - [x] `examples/khatri_rao.rs` - Parallel speedup demonstration
  - [x] `examples/mttkrp_cp.rs` - CP-ALS iteration and MTTKRP variants
  - [x] `examples/nmode_tucker.rs` - Tucker decomposition workflow

### Code Quality - COMPLETE

- [x] No unsafe code - 0 unsafe blocks
- [x] No panics in production paths
- [x] Bounds checking for all indexing
- [x] Error handling with Result types (structured KernelError enum)
- [x] Deterministic results - fixed seeds for random generation
- [x] 0 todo!()/unimplemented!() calls

---

## M2+: Future Enhancements - PLANNED

### Advanced Kernels

- [ ] Tucker-TTM (multiple mode products)
- [ ] TT-matrix-vector product
- [ ] TT-rounding operation
- [ ] Tensor contraction primitives

### Sparse Support (M3)

- [ ] Sparse MTTKRP (COO/CSR input)
- [ ] Sparse n-mode product
- [ ] Mixed sparse/dense operations

### GPU Acceleration (Future)

- [ ] CUDA kernels (feature-gated)
- [ ] ROCm support
- [ ] Unified CPU/GPU API

### Distributed (Future)

- [ ] MPI-based distributed MTTKRP
- [ ] Communication-avoiding algorithms
- [ ] Tensor partitioning strategies

### Additional Optimizations (Future)

- [ ] SIMD inner loops for MTTKRP fused kernel
- [ ] Fully fused + cache-blocking combined variant
- [ ] Parallel Tucker mode application
- [ ] Fused multi-mode products
- [ ] Flamegraph profiling / cache miss analysis

---

## Dependencies

### Current (All In Use)

- tenrso-core - In use
- scirs2-core (ndarray_ext, SIMD) - In use
- scirs2-linalg (SVD, QR) - In use
- num-traits - In use
- rayon (optional, default-enabled) - In use
- proptest (dev) - In use
- criterion (dev) - In use

---

## Recent Updates

### RC.1 Release (2026-03-06)

- All M1 kernels complete and production-ready
- 264 tests passing (7 ignored), 100% pass rate
- 135+ benchmarks across 10 benchmark groups
- 0 warnings, 0 unsafe blocks, full SCIRS2 compliance
- Fused MTTKRP kernel column-to-multi-index mapping fixed
- Parallel CP reconstruction added
- Randomized SVD and range finder implemented
- Statistical toolkit (14+ operations) complete
- TT operations (10 functions) complete

### Alpha.2 Release (2025-12-16)

- Fused MTTKRP: all previously-ignored tests now passing
- Parallel CP reconstruction added
- Performance results documented
- Complete statistical toolkit
- 304 tests at time of release

### Session 13 (2025-11-27)

- Parallel CP reconstruction implemented
- Tucker operator auto-ordering from smallest to largest mode
- Tucker reconstruction validated

### Session 12 (2025-11-27)

- Fixed MTTKRP fused: column-to-multi-index mapping corrected
- All 3 previously ignored fused MTTKRP tests now passing

### Session 11 (2025-11-26)

- Large tensor tests added (100^3+)
- Blocked MTTKRP correctness verified at scale

### Session 1 (2025-11-26)

- Cache-blocking for Khatri-Rao (`khatri_rao_blocked`, `khatri_rao_blocked_parallel`)
- Tensor-Tensor Product (TTT) implemented
- Fused MTTKRP kernel initial implementation

---

## Implementation Notes

### Khatri-Rao

- Column-wise operation: parallelize over columns
- Output size: (nrows_a * nrows_b) x ncols
- Memory order: blocking for cache efficiency

### Kronecker

- Block structure: A x B = [a_ij * B]
- Uses BLAS-style ops for each block
- Parallel over blocks of A

### MTTKRP

- Core bottleneck in CP-ALS
- Fused kernel avoids materializing full Khatri-Rao product
- Use blocked variant for large tensors to improve cache locality

### N-Mode Product

- Essentially GEMM after unfolding
- Minimize unfold/fold overhead
- TTT uses index computation for arbitrary contraction modes

### Randomized SVD

- Based on Halko, Martinsson, Tropp (2011)
- Power iteration for improved accuracy
- Suitable for low-rank approximation when rank << min(m, n)

---

## References

- Kolda & Bader (2009) "Tensor Decompositions and Applications"
- Phan et al. (2013) "Fast and efficient PARAFAC2"
- Smith & Karypis (2015) "Tensor-matrix products with a compressed sparse tensor"
- Halko, Martinsson, Tropp (2011) "Finding structure with randomness"
