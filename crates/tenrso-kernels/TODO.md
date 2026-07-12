# tenrso-kernels TODO

> **Milestone:** M1 (Complete) + early M3 (sparse MTTKRP)
> **Version:** 0.1.0
> **Status:** 0.1.0 — 324 unit tests passing + 31 integration + 80 doc tests, 7 ignored (100%)
> **Last Updated:** 2026-06-03

---

## Current Status Summary

**Test Coverage:** 292 unit tests + 31 integration + 78 doc tests (100% passing), 7 ignored
- Unit tests: 292 (275 baseline + 17 new sparse-MTTKRP tests: 3D/4D correctness vs dense oracle,
  all-zeros / fully-dense edges, rank-1 / rank-128 stress, duplicate-index aggregation,
  invalid-mode / factor-shape / factor-rank errors, parallel parity vs serial and dense)
- Property tests (proptest): 40+ (mathematical invariants)
- Integration tests: 31 (real-world CP-ALS, Tucker-HOOI workflows)
- Doc tests: 78 (all API examples verified, +2 for sparse MTTKRP fns)

**Code Metrics:**
- Source code: ~11K lines (15 files, +`mttkrp_sparse.rs`)
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
  - [x] Parallel iteration — Done (2026-07-11, `hadamard_parallel`). Measured:
        a genuine ~2.4x speedup at 2000x2000 (8-core Xeon), but ~30x
        *slower* than serial at 100x100 (Rayon overhead dominates below
        roughly 500x500). Confirms the original "small benefit" prediction
        was too pessimistic for large matrices but too optimistic for
        small ones — use it selectively. **Kept.**
  - [x] `SimdUnifiedOps` element ops — Added 2026-07-11
        (`hadamard_simd_f64`/`_f32`, `hadamard_nd_simd_f64`/`_f32`,
        `hadamard_parallel_simd_f64`/`_f32`), **removed 2026-07-11 (pruning
        pass)**. Honest result: measured slower than the existing
        scalar/`ndarray` implementation at every size tested
        (500x500..2000x2000) — Hadamard is memory-bandwidth-bound and
        `ndarray`'s own `Mul` is already close to this hardware's ceiling;
        routing through `scirs2_core::simd_ops` (scirs2-core 0.6.0) added
        ~1.5-2x overhead instead of a speedup (both `SimdUnifiedOps` entry
        points tried). A `*_simd` function that is *slower* than the plain
        version is an API trap (a caller reaching for it expecting a
        speedup silently gets a regression), so instead of keeping a
        known-slower public API around, these were deleted outright along
        with their tests and benchmark entries. Nothing else in the
        workspace referenced them. `hadamard`/`hadamard_inplace` remain the
        recommended entry points; `hadamard_parallel` remains for the
        large-matrix case documented above.

- [x] Testing
  - [x] Correctness tests (various shapes) - 8 tests
  - [x] Property tests (commutativity, identity)
  - [x] Edge cases (zeros, large arrays)
  - [x] Benchmarks — `benches/kernel_benchmarks.rs` `hadamard` group:
        allocating/inplace/parallel (the `simd_f64`/`simd_f32`/
        `parallel_simd_f64` entries were removed alongside the functions
        they benchmarked, see above)

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
  - [x] Fully fused + cache-blocking combined variant — Complete (2026-06-10, `mttkrp_fused_blocked` + `mttkrp_fused_blocked_parallel`)
  - [x] **SIMD inner loops (2026-04-14)** — rank-innermost loop reorder
        with compiler auto-vectorization over contiguous slices
    - [x] `mttkrp_fused_simd_f32` / `mttkrp_fused_simd_f64` (serial)
    - [x] `mttkrp_fused_simd_parallel_f32` / `_f64` (Rayon over mode rows)
    - [x] Supported lane widths: AVX-512 (16x f32 / 8x f64), AVX2 (8x f32 /
          4x f64), NEON (4x f32 / 2x f64); scalar fallback on other ISAs.
    - [x] 11 new tests covering 4 shape regimes + tail + edge cases

- [x] Variants
  - [x] Dense MTTKRP - Complete
  - [x] Blocked MTTKRP - Complete
  - [x] Fused MTTKRP - Complete
  - [x] Sparse MTTKRP - Complete (COO, serial + parallel) — 2026-04-15
        - [x] `mttkrp_sparse_coo` - O(nnz * (N-1) * R) serial variant
        - [x] `mttkrp_sparse_coo_parallel` - row-partitioned Rayon variant
              (gated on `parallel` feature, no atomics, no write races)
        - [x] Duplicate-index entries accumulate additively (matches
              `CooTensor::deduplicate` semantics, verified vs dense oracle)
        - [x] N-D general: 3rd, 4th-order tensors exercised in tests
        - [x] Sparse HiCOO MTTKRP — `mttkrp_hicoo` + `mttkrp_hicoo_parallel` (2026-06-03)
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

- [x] SIMD element operations — Added 2026-07-11 (`outer_product_2_simd_f64`/
      `_f32`, `outer_product_simd_f64`/`_f32`, `outer_product_weighted_simd_f64`/
      `_f32`, via `scirs2_core::simd_ops::SimdUnifiedOps::simd_scalar_mul`
      broadcast-fold). **Nuanced honest result**: for N-D (3+ vectors) and
      weighted outer products, measured **~6-14x faster** than the naive
      `flat_to_multi_index`-based baseline — but this win is mostly
      *algorithmic* (O(1) index math per element vs. the baseline's O(ndim)
      div/mod decode per element), with vectorization a secondary
      contributor. For the plain 2-vector case (`outer_product_2_simd_f64`),
      where the naive baseline is *already* O(1)-per-element with no
      inefficiency to remove, the SIMD variant measured **~5-13x slower**.
  - **2026-07-11 pruning pass — folded, not kept as a separate API.** Since
    the win for the N-D/weighted cases was mostly algorithmic (not the SIMD
    substrate itself), the broadcast-fold algorithm was folded directly into
    `outer_product`/`outer_product_weighted` (generic `T: Clone + Num`, no
    `SimdUnifiedOps` bound — this keeps the trait bounds identical to before,
    so no downstream caller needed to change) and the `_simd`-suffixed
    functions were deleted, along with `flat_to_multi_index` (now dead) and
    the old per-element-decode implementation. All callers of
    `outer_product`/`outer_product_weighted`/`cp_reconstruct` now get the
    faster algorithm automatically. `outer_product_2_simd_f64`/`_f32` were
    deleted outright (no algorithmic win to fold in, and the measured
    slowdown makes a `_simd`-named API a trap) — `outer_product_2` is
    unchanged. See doc comments in `src/outer.rs` (`outer_product`'s
    "Algorithm note") for the full writeup.

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

- [x] Parallel mode application - ✅ DONE (2026-06-10) (`nmode_product_parallel`, `nmode_products_parallel`, `tucker_operator_parallel`, feature `parallel`)
- [x] Fused multi-mode products - ✅ DONE (2026-06-10) (`tucker_reconstruct_fused`, `tucker_reconstruct_fused_parallel`)
- [ ] Memory-efficient intermediate tensors - Future

---

## Statistical Toolkit - COMPLETE

### Reductions (14+)

- [x] sum, mean, min, max
- [x] variance, std
- [x] norm_l1, norm_l2, norm_frobenius
- [x] skewness, kurtosis
- [x] Whole-tensor `SimdUnifiedOps` variants — Added 2026-07-11,
      `src/reductions_simd.rs`: `sum_tensor_simd_f64`/`_f32`,
      `mean_tensor_simd_f64`/`_f32`, `frobenius_norm_tensor_simd_f64`/`_f32`,
      `norm_l1_tensor_simd_f64`/`_f32`, `variance_tensor_simd_f64`/`_f32`,
      `std_tensor_simd_f64`/`_f32`. **Honest result: measured ~2-4x
      *slower*** than the equivalent scalar reduction at every size tested
      (1K-8M elements) — scirs2-core 0.6.0's AVX2 reduction kernels
      accumulate into a single vector register reused across loop
      iterations, which is latency-bound rather than throughput-bound.
  - **2026-07-11 pruning pass — removed entirely.** Unlike the outer-product
    case, there was no algorithmic inefficiency in the scalar reductions to
    fold a fix into — the scalar `*_along_modes` reductions (and a plain
    `.iter().sum()`) were already the right implementation, and the
    `SimdUnifiedOps` path was strictly worse at every measured size with no
    redeeming special case. A `*_simd` name that is 2-4x *slower* than the
    non-`_simd` alternative is exactly the API trap this pruning pass exists
    to remove, so `src/reductions_simd.rs` (module, all 12 functions, and
    their tests/benchmarks) was deleted outright rather than kept or
    renamed. **Use the reductions in `src/reductions.rs`** (`sum_along_modes`
    and friends) for production use.

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
  - [x] Use with tenrso-decomp — Complete (2026-06-10, `crates/tenrso/tests/kernels_decomp_integration.rs`)

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

- [x] Tucker-TTM (multiple mode products) — implemented as `tucker_operator()` / `tucker_operator_ordered()` in `nmode.rs`
- [x] TT-matrix-vector product — implemented as `tt_matvec()` in `tt_ops.rs`
- [x] TT-rounding operation — full Oseledets 2-phase (QR + right-to-left SVD) in `tt_round.rs`;
      `tt_truncate()` uses per-bond max_ranks; `tt_qr_decomposition` fixed to SVD-based thin QR
      for correct wide-matrix handling; 23 tests including rank-reduction and boundary checks
- [x] Tensor contraction primitives — implemented in `contractions.rs` (`contract_tensors`, `sum_over_modes`, `tensor_inner_product`, `tensor_trace`)

### Sparse Support (M3)

- [x] Sparse MTTKRP (COO input) - Complete (2026-04-15, serial + parallel)
- [x] Sparse MTTKRP (CSF input) — Complete (2026-06-03, `feature = "csf"`)
      DFS fiber-tree walk amortizes partial KR products: O(nnz·R + Σ|fibers_L|·R)
      vs O(nnz·(N-1)·R) for COO; serial + parallel variants; 18 tests.
      HiCOO: not yet — deferred (similar complexity, lower priority).
- [x] Sparse n-mode product (`nmode_product_sparse_coo`) — Complete (2026-06-03)
      COO × dense matrix → dense output; scatter algorithm, no unfold/GEMM;
      serial + parallel variants; 14 tests. `feature = "sparse"` (default).
- [x] Mixed sparse/dense operations — satisfied by `nmode_product_sparse_coo` above

### GPU Acceleration (Future)

- [ ] CUDA kernels (feature-gated)
- [ ] ROCm support
- [ ] Unified CPU/GPU API

### Distributed (Future)

- [ ] MPI-based distributed MTTKRP
- [ ] Communication-avoiding algorithms
- [ ] Tensor partitioning strategies

### Additional Optimizations (Future)

- [x] SIMD inner loops for MTTKRP fused kernel (2026-04-14)
- [x] Fully fused + cache-blocking combined variant — Complete (2026-06-10, `mttkrp_fused_blocked` + `mttkrp_fused_blocked_parallel`)
- [x] Parallel Tucker mode application — Complete (2026-06-10, `nmode_product_parallel`, `nmode_products_parallel`, `tucker_operator_parallel`, `feature = "parallel"`)
- [x] Fused multi-mode products — Complete (2026-06-10, `tucker_reconstruct_fused` + `tucker_reconstruct_fused_parallel`)
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

### SIMD variant pruning + rustdoc link fixes (2026-07-11)

Follow-up to the `SimdUnifiedOps` pass below in this same file: once the
honest benchmark numbers were in, a public `*_simd` function that is
*slower* than the plain version is an API trap (a caller reaching for
`hadamard_simd_f64` expecting a speedup silently gets a 1.5-2x slowdown), so
every strictly-slower variant was removed rather than kept as a "for
completeness" curiosity, and the one genuine algorithmic win was folded into
the primary function instead of living behind a separate `_simd` name.

- **Removed outright** (functions, tests, and benchmark entries — nothing
  else in the workspace referenced them): `hadamard_simd_f64`/`_f32`,
  `hadamard_nd_simd_f64`/`_f32`, `hadamard_parallel_simd_f64`/`_f32`
  (`src/hadamard.rs`); the entire `src/reductions_simd.rs` module
  (`sum_tensor_simd_f64`/`_f32`, `mean_tensor_simd_f64`/`_f32`,
  `frobenius_norm_tensor_simd_f64`/`_f32`, `norm_l1_tensor_simd_f64`/`_f32`,
  `variance_tensor_simd_f64`/`_f32`, `std_tensor_simd_f64`/`_f32`);
  `outer_product_2_simd_f64`/`_f32` (`src/outer.rs`). All measured slower
  than their non-`_simd` counterparts (see the dated entries below for the
  original numbers).
- **Folded into the primary function** (dropped the `_simd` suffix entirely
  rather than keeping two APIs): `outer_product`/`outer_product_weighted`
  now use the `broadcast_fold` axis-by-axis algorithm directly (generic over
  `T: Clone + Num`, no `SimdUnifiedOps` bound — trait bounds are unchanged
  from before, so no downstream crate needed to change). The old
  `flat_to_multi_index`-based per-element decode and its test were removed
  as dead code. `outer_product_simd_f64`/`_f32` and
  `outer_product_weighted_simd_f64`/`_f32` no longer exist as separate
  functions; every caller of `outer_product`/`outer_product_weighted`/
  `cp_reconstruct` gets the faster algorithm automatically.
- **Kept unchanged**: `hadamard_parallel` (genuine ~2.4x win at 2000x2000,
  with the crossover documented in its doc comment) and the auto-vectorized
  `mttkrp_fused_simd_f32`/`_f64` family (4-12x genuine win, unrelated to
  `SimdUnifiedOps` — see the 2026-04-14 entry below) — both are out of scope
  for this pruning pass since they were never the slower kind.
- Dropped the `features = ["simd"]` override on the `scirs2-core` dependency
  in `Cargo.toml`: after the removals above, no code in this crate calls
  `scirs2_core::simd_ops::SimdUnifiedOps` anymore.
- Fixed all `cargo doc -p tenrso-kernels --no-deps --all-features` errors
  (some pre-existing, some introduced by the `SimdUnifiedOps` pass):
  a private-item link (`PARALLEL_WORK_THRESHOLD` in `mttkrp_dimtree.rs`,
  reworded out of link form since the item can't be public), several
  unresolved links (`nmode_product`, `nmode_products_seq`, `tucker_operator`
  in `nmode_tucker_ext.rs`; `mttkrp` in `mttkrp_fused_blocked.rs`/`utils.rs`)
  fixed by qualifying with `crate::` and, where the linked item is a
  function that shares its name with a module (`mttkrp`, `mttkrp_sparse_csf`
  are each both a function and a `pub mod`), disambiguating with the
  `[`crate::name()`]` function-call syntax.
- Verified clean: `cargo clippy -p tenrso-kernels --all-features --all-targets
  -- -D warnings`, `cargo nextest run -p tenrso-kernels --all-features` (449
  tests passed), `cargo test --doc -p tenrso-kernels --all-features` (90
  passed, 3 ignored), `cargo doc -p tenrso-kernels --no-deps --all-features`,
  and a full `cargo build --workspace --all-features`.

### `mttkrp_all_modes` deprecation restored + `SimdUnifiedOps` pass (2026-07-11)

- **Semver fix:** restored `utils::mttkrp_all_modes` as a `#[deprecated]`
  alias forwarding to `mttkrp_all_modes_naive` (a prior rename had removed
  this published-crate public path outright). Root-level
  `tenrso_kernels::mttkrp_all_modes` still resolves to the fast
  `mttkrp_dimtree::mttkrp_all_modes` (disambiguated explicitly in `lib.rs`
  to fix the resulting `ambiguous_glob_reexports` error).
- **Hadamard:** added `hadamard_simd_f64`/`_f32`, `hadamard_nd_simd_f64`/
  `_f32` (via `SimdUnifiedOps::simd_mul_into`), `hadamard_parallel`
  (row-parallel Rayon), `hadamard_parallel_simd_f64`/`_f32`. Honest result:
  the `SimdUnifiedOps` paths measured **slower** than the existing
  `ndarray`-based `hadamard` (~1.5-2x); `hadamard_parallel` measured a real
  ~2.4x win at 2000x2000 but ~30x loss at 100x100. See `src/hadamard.rs`
  doc comments for the full writeup (both `SimdUnifiedOps` entry points
  were tried).
- **Outer products:** added `outer_product_2_simd_f64`/`_f32`,
  `outer_product_simd_f64`/`_f32`, `outer_product_weighted_simd_f64`/`_f32`
  via a broadcast-fold algorithm built on `SimdUnifiedOps::simd_scalar_mul`.
  Honest result: **~6-14x faster** for N-D (3+ vectors) and weighted
  products (beats the naive `flat_to_multi_index`-based baseline mostly on
  algorithm, not just SIMD); **~5-13x slower** for the already-efficient
  2-vector case (`outer_product_2_simd_f64`).
- **Reductions:** new `src/reductions_simd.rs` module: `sum_tensor_simd_f64`/
  `_f32`, `mean_tensor_simd_f64`/`_f32`, `frobenius_norm_tensor_simd_f64`/
  `_f32`, `norm_l1_tensor_simd_f64`/`_f32`, `variance_tensor_simd_f64`/`_f32`,
  `std_tensor_simd_f64`/`_f32` (whole-tensor reductions, since per-mode
  reductions can't offer one contiguous slice). Honest result: measured
  **~2-4x slower** than scalar equivalents at every size (1K-8M elements) —
  root cause is a single-accumulator (latency-bound) reduction loop in
  scirs2-core 0.6.0's AVX2 kernels.
- Added `features = ["simd"]` to `tenrso-kernels`'s `scirs2-core` dependency
  (on top of the workspace-inherited version) so `SimdUnifiedOps` actually
  dispatches to real AVX2/NEON code instead of silently falling back to
  scalar.
- 57 new tests (correctness vs. scalar/naive references, including
  non-lane-multiple sizes: 0,1,2,3,5,7,9,13,17,31,33,65,129), all passing;
  13 new doc tests; new `hadamard`/`outer_product`/`reductions_simd`
  criterion benchmark groups in `benches/kernel_benchmarks.rs`.
- 0 warnings, 0 new `unsafe`, clippy-clean with `-D warnings`, `cargo fmt`
  clean.

### SIMD Fused MTTKRP (2026-04-14)

- Added `mttkrp_fused_simd_f32` / `mttkrp_fused_simd_f64` with
  rank-innermost loop reorder → compiler auto-vectorization on AVX2,
  AVX-512, and NEON (no `unsafe`, no raw intrinsics).
- Added parallel variants `mttkrp_fused_simd_parallel_f32/_f64` that
  partition the `I_mode` rows across Rayon threads with per-thread scratch.
- Measured speedups (aarch64 NEON, rank=32, f64):
  4.10x-4.63x over scalar fused serial; 8-12x combined with parallelism.
- 11 new unit tests (shape grid, tail R=lane+1, R=1 below-threshold
  fallback, R=0 empty, parallel f32/f64, invalid-mode propagation).
- Numerical tolerance: 1e-12 f64, 1e-6 f32 vs scalar reference.
- 4 new doc tests.
- 0 warnings, 0 `unsafe`, clippy-clean with `-D warnings`.

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
