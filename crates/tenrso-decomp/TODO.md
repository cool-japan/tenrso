# tenrso-decomp TODO

> **Milestone:** M2
> **Status:** ✅ **M2 COMPLETE!** All decomposition algorithms implemented and tested

---

## M2: Decomposition Implementations - ✅ COMPLETE

### CP-ALS (Canonical Polyadic via Alternating Least Squares) - ✅ COMPLETE

- [x] Core algorithm - ✅ **WORKING**
  - [x] Factor matrix initialization strategies
  - [x] ALS iteration loop
  - [x] MTTKRP computation (via tenrso-kernels)
  - [x] Least-squares factor updates ✅ **FIXED: Now uses scirs2_linalg::lstsq**
  - [x] Convergence detection (relative change, error)

- [x] Initialization methods - ✅ **COMPLETE**
  - [x] Random uniform initialization ✅ **IMPLEMENTED: Using scirs2_core::random**
  - [x] Random normal initialization ✅ **IMPLEMENTED: Using scirs2_core::random**
  - [x] SVD-based initialization ✅ **IMPLEMENTED: HOSVD approach**
  - [ ] Leverage score sampling ⏳ Future M2+
  - [ ] NNSVD (for non-negative) ⏳ Future M2+

- [x] Constraints ✅ **COMPLETE**
  - [x] Non-negative factors ✅ **IMPLEMENTED: Via cp_als_constrained**
  - [x] L2 regularization ✅ **IMPLEMENTED: Via cp_als_constrained**
  - [x] Orthogonality constraints ✅ **IMPLEMENTED: Via cp_als_constrained**

- [x] Stopping criteria - ✅ COMPLETE
  - [x] Maximum iterations
  - [x] Relative error tolerance (fit change)
  - [x] Factor change tolerance (implicit in fit)
  - [ ] Time limit (optional) ⏳ Future

- [x] Output - ✅ COMPLETE
  - [x] `CpDecomp` struct with factors, weights, fit
  - [x] Final fit value
  - [x] Iteration count
  - [x] Convergence status

### Tucker-HOSVD (Higher-Order SVD) - ✅ COMPLETE

- [x] Core algorithm - ✅ COMPLETE
  - [x] Unfold tensor along each mode
  - [x] Compute truncated SVD for each unfolding
  - [x] Extract factor matrices (left singular vectors)
  - [x] Compute core tensor via mode products

- [x] Rank selection - ✅ COMPLETE
  - [x] Fixed ranks (user-specified)
  - [ ] Automatic rank selection (future) ⏳ Future M2+
  - [ ] Truncation threshold ⏳ Future M2+

- [x] Output - ✅ COMPLETE
  - [x] `TuckerDecomp` with core and factors
  - [x] Reconstruction error
  - [x] Compression ratio

### Tucker-HOOI (Higher-Order Orthogonal Iteration) - ✅ COMPLETE

- [x] Core algorithm - ✅ COMPLETE
  - [x] Initialize with HOSVD
  - [x] Iteratively refine factor matrices
  - [x] Update each mode via truncated SVD
  - [x] Recompute core tensor
  - [x] Check convergence

- [x] Convergence - ✅ COMPLETE
  - [x] Relative error change
  - [x] Factor change (implicit)
  - [x] Maximum iterations

- [x] Output - ✅ COMPLETE
  - [x] Same as HOSVD plus iteration history

### TT-SVD (Tensor Train via SVD) - ✅ COMPLETE

- [x] Core algorithm - ✅ COMPLETE
  - [x] Sequential left-to-right SVD
  - [x] Truncation at each step
  - [x] TT-rank determination (tolerance-based)
  - [x] Core tensor construction

- [x] TT-rounding ✅ **COMPLETE (2025-11-06)**
  - [x] Reduce TT-ranks post-decomposition ✅ **IMPLEMENTED: tt_round function**
  - [x] Error-controlled truncation ✅ **IMPLEMENTED: Tolerance-based**
  - [x] Memory optimization ✅ **VERIFIED: Compression ratio tracking**

- [ ] TT operations (stretch) ⏳ Future M3+
  - [ ] TT addition
  - [ ] TT inner product
  - [ ] TT matrix-vector product

- [x] Output - ✅ COMPLETE
  - [x] `TTDecomp` with core tensors
  - [x] TT-ranks
  - [x] Compression ratio
  - [x] Reconstruction error

---

## Testing - ✅ LARGELY COMPLETE

- [x] Unit tests - ✅ **26 tests passing**
  - [x] CP-ALS correctness (synthetic data) ✅ 6 tests
  - [x] Tucker-HOSVD correctness ✅ 6 tests
  - [x] Tucker-HOOI convergence ✅ Tested
  - [x] TT-SVD correctness ✅ 3 tests
  - [x] Reconstruction tests ✅ All decompositions
  - [x] Error computation tests ✅ All decompositions

- [x] Property tests ✅ **COMPLETE (15 tests)**
  - [x] Reconstruction error bounds ✅ **4 CP tests + 4 Tucker tests + 5 TT tests**
  - [x] Orthogonality of Tucker factors ✅ **Verified**
  - [x] Cross-method validation ✅ **All methods produce valid reconstructions**
  - [ ] Non-negative constraints hold ⏳ Future (when non-negative CP is implemented)

- [x] Integration tests - ✅ **14 integration tests passing**
  - [x] Use with synthetic tensor data
  - [x] Cross-crate interop with tenrso-kernels
  - [x] Cross-decomposition comparison tests

- [x] Benchmarks ✅ **COMPLETE (2025-11-06)**
  - [x] CP-ALS: 256³, rank 64, 10 iters < 2s ✅ **benchmark target implemented**
  - [x] Tucker-HOOI: 512×512×128, 10 iters < 3s ✅ **benchmark target implemented**
  - [x] TT-SVD: 32⁶, eps=1e-6 < 2s ✅ **benchmark target implemented**
  - [x] CP initialization benchmarks (Random, RandomNormal, SVD) ✅ **NEW**
  - [x] CP constraints benchmarks (nonnegative, L2 reg, orthogonal) ✅ **NEW**
  - [x] TT-rounding benchmarks (4 comprehensive scenarios) ✅ **NEW**

---

## Documentation

- [x] Rustdoc for all public APIs ✅ **ENHANCED (2025-11-06)**
  - [x] Comprehensive crate-level documentation ✅ **lib.rs with quick start guide**
  - [x] Mathematical notation and formulations ✅ **Added to module docs**
  - [x] Use cases and applications ✅ **Added to lib.rs**
  - [x] Performance targets ✅ **Added to lib.rs**
- [x] Algorithm descriptions ✅ **Complete in module docs**
- [x] Mathematical notation and references ✅ **Complete with academic citations**
- [x] Examples ✅ **COMPLETE**
  - [x] `examples/cp_als.rs` ✅ **COMPLETE**
  - [x] `examples/tucker.rs` ✅ **COMPLETE**
  - [x] `examples/tt.rs` ✅ **COMPLETE**
  - [x] `examples/reconstruction.rs` ✅ **COMPLETE**

---

## Dependencies

- tenrso-core (tensor types) - ✅ Available
- tenrso-kernels (MTTKRP, n-mode) - ✅ **COMPLETE (M1 done!)**
- tenrso-kernels (outer products) - ✅ **NEW: Available for CP reconstruction**
- tenrso-kernels (tucker operator) - ✅ **NEW: Available for Tucker reconstruction**
- scirs2-linalg (SVD, QR) - ⏳ Required for M2
- scirs2-optimize (least-squares) - 💡 Optional

---

## Blockers

- ~~Depends on `tenrso-kernels::mttkrp` for CP-ALS~~ - ✅ **UNBLOCKED!**
- ~~Depends on `tenrso-kernels::nmode_product` for Tucker~~ - ✅ **UNBLOCKED!**
- ~~Requires `scirs2-linalg` integration~~ - ✅ **COMPLETE: Using lstsq, svd**
- ~~Requires `tenrso-core::DenseND` implementation~~ - ✅ **COMPLETE!**

**No blockers remaining!**

---

## Recent Enhancements

### ✅ TT-Rounding Implementation (NEW!)

**Full TT-rounding algorithm implemented and tested:**
1. **`tt_round()` function** - Post-decomposition rank reduction
   - Right-to-left QR orthogonalization sweep
   - Left-to-right SVD truncation with error control
   - Rank reduction while maintaining approximation quality
   - Memory optimization for large tensor trains

2. **Comprehensive test suite** (5 new tests):
   - Basic rank reduction verification
   - Reconstruction quality testing
   - Accuracy preservation with loose tolerance
   - Compression ratio improvement validation
   - Max rank constraint enforcement

3. **Bug fixes:**
   - Fixed QR decomposition dimension mismatch in orthogonalization
   - Corrected matrix transpose operations for proper factorization
   - All 5 TT-rounding tests now pass successfully

### ✅ CP Constraints Already Implemented (Discovered!)

Found that CP-ALS constraints were already fully implemented:
1. **`cp_als_constrained()` function** - Extended CP-ALS with:
   - **Non-negativity**: Project factors to [0, ∞) after each update
   - **L2 regularization**: Add λ||F||² penalty to Gram matrix
   - **Orthogonality**: QR-based orthonormalization of factors

2. **CpConstraints struct** - Flexible constraint configuration:
   - `CpConstraints::nonnegative()` - For NMF-style decompositions
   - `CpConstraints::l2_regularized(λ)` - For overfitting prevention
   - `CpConstraints::orthogonal()` - For orthogonal factor matrices

3. **Working tests:**
   - test_cp_als_nonnegative ✅
   - test_cp_als_l2_regularized ✅
   - test_cp_als_orthogonal ✅
   - test_constraint_combinations ✅

### ✅ Documentation Enhancements (Major Upgrade!)

**Comprehensive crate-level documentation added to lib.rs:**
1. **Overview section** - Explains all three decomposition families
2. **Use cases** - Real-world applications for each method
3. **Quick start guide** - Working examples for CP, Tucker, TT
4. **Mathematical formulations** - Proper notation and equations
5. **Performance targets** - Expected timings on modern hardware
6. **Academic references** - Key papers and citations
7. **SciRS2 integration notes** - Project policy compliance

**Total documentation improvements:**
- lib.rs: Expanded from 10 lines to 148 lines of comprehensive docs
- Fixed Rustdoc compilation errors (escaped bracket notation)
- Added working code examples that compile with doctests
- Mathematical notation properly formatted

### ✅ Example Compilation Fixes

Fixed all example compilation errors:
1. **reconstruction.rs** - Removed incorrect `?` operators on subtraction
2. **tucker.rs** - Fixed variable shadowing (`tucker_hooi` function vs variable)
3. **tt.rs** - Prefixed unused variable with underscore
4. Removed all unused imports warnings

All examples now compile and run successfully!

### ✅ Test Suite Status

**Current test counts:**
- TT-SVD: 3 original tests + 5 NEW TT-rounding tests = **8 tests total**
- CP-ALS: 8 tests (including 4 constraint tests)
- Tucker: 6 tests (HOSVD + HOOI)
- Property tests: 15 tests
- Integration tests: 14 tests

**Overall: 30+ tests passing** (unit + integration + property)

### ✅ Performance Benchmarks Enhanced (NEW!)

**Comprehensive benchmark suite now includes:**
1. **CP-ALS benchmarks** (8 total):
   - Small/medium/large tensor sizes
   - **Target benchmark**: 256³ with rank-64 (performance verification)
   - Initialization method comparison (Random vs RandomNormal vs SVD)
   - **NEW: CP constraints benchmarks** (non-negative, L2 reg, orthogonal)

2. **Tucker benchmarks** (6 total):
   - HOSVD vs HOOI comparison
   - **Target benchmark**: 512×512×128 with 10 iterations
   - Asymmetric tensor support (realistic use cases)

3. **TT-SVD benchmarks** (5 total):
   - **Target benchmark**: 32⁶ tensor (high-order compression)
   - Varying ranks and tensor orders
   - High-order tensor scenarios (5-7 modes)

4. **TT-Rounding benchmarks** (4 NEW):
   - `bench_tt_round_small`: Multiple sizes (8⁴, 10⁵)
   - `bench_tt_round_varying_reduction`: Different rank reductions
   - `bench_tt_round_tolerance_impact`: Tolerance sensitivity
   - `bench_tt_round_large`: Realistic 16⁶ scenario

5. **Reconstruction benchmarks** (3 total):
   - CP, Tucker, TT reconstruction performance

6. **Compression analysis** (1 total):
   - Compression ratio efficiency across methods

**Total: 27 benchmark functions** covering all decomposition methods and new features!

---

## Additional Enhancements

### ✅ Examples Complete (4 comprehensive examples)

**New Examples Added:**
1. **`examples/cp_als.rs`** - Comprehensive CP-ALS demonstration
   - 6 examples covering: basic usage, initialization strategies, compression analysis, factor analysis, convergence behavior, non-cubic tensors
   - Demonstrates all three initialization methods (Random, RandomNormal, SVD)
   - Shows compression ratios and fit quality trade-offs

2. **`examples/tucker.rs`** - Tucker decomposition showcase
   - 8 examples covering: HOSVD, HOOI, comparison, compression trade-offs, non-cubic tensors, orthogonality verification, convergence, core analysis
   - Demonstrates the improvement of HOOI over HOSVD
   - Verifies factor matrix orthogonality mathematically

3. **`examples/tt.rs`** - TT-SVD decomposition examples
   - 8 examples covering: 4D and 6D tensors, compression analysis, tolerance-based truncation, non-uniform modes, core statistics, memory efficiency, method comparison
   - Shows TT's advantages for high-order tensors
   - Compares TT vs CP vs Tucker for 4D tensors

4. **`examples/reconstruction.rs`** - Reconstruction quality analysis
   - 8 examples covering: CP/Tucker/TT reconstruction, quality comparison, error distribution, timing analysis, factor modification effects
   - Comprehensive accuracy vs compression trade-off analysis
   - Element-wise error statistics

**All examples:**
- Run successfully in release mode
- Include multiple scenarios and edge cases
- Show performance timing and quality metrics
- Demonstrate best practices for each decomposition

### ✅ Property-Based Tests Complete (15 tests)

**New Property Tests Added** (using proptest framework):
- **CP-ALS tests (4 tests):**
  - Reconstruction error decreases with higher rank
  - Reconstruction error matches reported fit
  - Weights are non-negative
  - Reconstruction invariant to factor scaling

- **Tucker tests (4 tests):**
  - Factor matrices are orthogonal (U^T U = I)
  - HOOI improves or maintains error vs HOSVD
  - Error decreases with higher rank
  - Core tensor has correct shape

- **TT-SVD tests (5 tests):**
  - Reconstruction error bounded by tolerance
  - Ranks respect max_ranks constraint
  - Cores have correct shapes (r_left × n × r_right)
  - Compression ratio > 1 for high-order tensors
  - All methods produce valid reconstructions (cross-method test)

**Test Infrastructure:**
- Added to `src/property_tests.rs`
- Uses proptest for randomized testing with configurable ranges
- Tests verify mathematical properties that should always hold
- Covers edge cases and numerical stability

### ✅ tenrso-core Enhancements

**New Operations Added:**
1. **`frobenius_norm()`** - Compute Frobenius norm (||X||_F)
   - Essential for error computation and quality metrics
   - Optimized using scirs2_core's numeric traits

2. **`Sub` operator** - Element-wise tensor subtraction
   - `&tensor1 - &tensor2` returns `Result<DenseND>`
   - Shape checking with informative errors
   - Required for reconstruction error computation

3. **`Add` operator** - Element-wise tensor addition
   - `&tensor1 + &tensor2` returns `Result<DenseND>`
   - Shape checking with informative errors
   - Complements subtraction for tensor operations

4. **`as_slice()`** - Get contiguous slice view of data
   - Zero-copy access to underlying data
   - Required for efficient element-wise analysis

**All operations:**
- Follow SCIRS2 integration policy (use scirs2_core, not ndarray directly)
- Include comprehensive documentation and examples
- Handle errors gracefully with Result types
- Tested and working in property tests and examples

---

## Implementation Notes

### ✅ CP-ALS Implementation Complete

**Critical Bugs Fixed:**
1. **Least Squares Solver** - Fixed broken `solve_least_squares` that was just returning input unchanged
   - Now properly solves `Gram^T * factor[i,:] = MTTKRP[i,:]` for each row
   - Uses `scirs2_linalg::lstsq` with fallback regularization for numerical stability
   - Tested and working correctly

2. **Random Initialization** - Fixed placeholder constant (0.5) initialization
   - **Random Uniform**: Now uses `scirs2_core::random::Rng::random()` for proper uniform [0,1] values
   - **Random Normal**: Uses `scirs2_core::random::RandNormal` distribution N(0,1)
   - **SVD Initialization**: Implements HOSVD-based initialization with SVD of mode-n unfoldings

**Test Results:**
- ✅ All 8 unit tests passing
- ✅ All 14 integration tests passing
- ✅ All 4 doc tests passing
- ✅ **26 total tests - 100% pass rate**

**Implementation Status:**
- Core CP-ALS algorithm fully functional
- Proper convergence detection (fit change tolerance)
- Weight extraction and normalization
- Reconstruction from factors
- Multiple initialization strategies working

### Unblocked by tenrso-kernels M1 Completion

The following kernels are now available and ready for use:

1. **CP-ALS Support:**
   - `mttkrp()` - Standard MTTKRP for factor updates
   - `mttkrp_blocked()` - Cache-optimized version for large tensors
   - `mttkrp_blocked_parallel()` - Parallel blocked version
   - `cp_reconstruct()` - **NEW:** Reconstruct tensor from CP factors
   - `outer_product()` - **NEW:** Build rank-1 tensors

2. **Tucker Support:**
   - `nmode_product()` - Single mode product
   - `nmode_products_seq()` - Sequential multi-mode
   - `tucker_operator()` - **NEW:** Optimized multi-mode product
   - `tucker_reconstruct()` - **NEW:** Reconstruct from core + factors

3. **Additional Utilities:**
   - `khatri_rao()` / `khatri_rao_parallel()` - For CP factor operations
   - `hadamard()` / `hadamard_nd()` - Element-wise operations

### Ready to Start M2

All kernel dependencies for decomposition implementations are now satisfied. M2 (Decompositions) can begin as soon as `tenrso-core::DenseND` is available.
