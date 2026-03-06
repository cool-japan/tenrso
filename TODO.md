# TenRSo TODO

> **Version:** 0.1.0-rc.1
> **Status:** 🎉 **RC.1 RELEASED** - 2,109 tests passing (100%)
> **Release Date:** 2026-03-06

This document tracks high-level tasks across the entire TenRSo project. For crate-specific tasks, see individual `crates/*/TODO.md` files.

---

## Alpha.2 Release Highlights (2025-12-16)

### Documentation Quality Improvements ✅
- [x] Fixed all intra-doc link issues (10 files)
- [x] Comprehensive lib.rs review (995 doc lines, 68 examples)
- [x] docs.rs compatibility verified
- [x] Zero documentation warnings
- [x] All bracket notation properly escaped in rustdoc

### Code Quality ✅
- [x] Zero compiler warnings (all targets, all features)
- [x] Zero clippy warnings (all targets, all features)
- [x] Consistent code formatting
- [x] 1,820+ library tests passing (100%)

### Testing ✅
- [x] All library tests passing
- [x] Property-based tests passing
- [x] Integration tests passing
- [x] Core functionality verified

---

## RC.1 Release Highlights (2026-03-06)

### New Features
- [x] Executor element-wise operations: ScalarOp enum, parallel_elem_op, parallel_binary_op, full_reduce (tenrso-exec)
- [x] TT-SVD gradient backward pass: TtReconstructionGrad, compute_core_gradients, numerically verified (tenrso-ad)
- [x] Masked einsum operations: masked_einsum, specialized kernels, subset reductions (tenrso-sparse)
- [x] CP decomposition regularization: L1 soft-thresholding, L2/Tikhonov, cross-validation rank selection (tenrso-decomp)
- [x] CP module refactored from monolithic cp.rs (3212 lines) into cp/ submodules (tenrso-decomp)

### Quality
- [x] Version bumped from 0.1.0-alpha.2 to 0.1.0-rc.1
- [x] 2,109 tests passing (was 1,820+ in alpha.2), 14 skipped
- [x] Test suite runtime reduced 4.8x (963s to 198s) — zero tests >30s
- [x] All 8 crates production-ready
- [x] Zero compiler/clippy warnings (all targets, all features)
- [x] Workspace policy: all subcrates use version.workspace = true

---

## Legend

- ✅ **Complete** - Implemented and tested
- 🔄 **In Progress** - Currently being worked on
- ⏳ **Planned** - Scheduled for future milestone
- 🔴 **Blocked** - Waiting on dependencies or decisions
- 💡 **Idea** - Future consideration, not yet planned

---

## M0: Repo Hygiene - ✅ COMPLETE

- [x] Workspace skeleton with 8 crates
- [x] CI/CD (fmt, clippy, test, doc, coverage)
- [x] MSRV 1.82 toolchain
- [x] Apache-2.0 license
- [x] Documentation (README, ROADMAP, CONTRIBUTING, blueprint)
- [x] SciRS2 integration policy
- [x] Claude development guide
- [x] `.gitignore` and project structure
- [x] Initial commit and GitHub push

---

## M1: Kernels - ✅ COMPLETE

### Core Dense Tensor (tenrso-core) - ✅ COMPLETE

- [x] Implement `DenseND<T>` with ndarray backend
- [x] Tensor views (zero-copy slicing)
- [x] Strides and memory layout
- [x] Unfold/fold operations (mode-n matricization)
- [x] Reshape and permute
- [x] Axis metadata tracking
- [x] Property tests for shape operations - ✅ 36 tests passing (19 unit + 17 doc)

### Tensor Kernels (tenrso-kernels) - ✅ COMPLETE + ENHANCED

**Core Kernels:**
- [x] Khatri-Rao product (column-wise Kronecker) - ✅ Complete with parallel version
- [x] Kronecker product (matrix/tensor) - ✅ Complete with parallel version
- [x] Hadamard product (element-wise) - ✅ Complete (2D, ND, in-place variants)
- [x] N-mode product (TTM/TTT) - ✅ Complete with sequential multi-mode
- [x] MTTKRP (Matricized Tensor Times Khatri-Rao Product) - ✅ Complete
- [x] Blocked/Tiled MTTKRP - ✅ Complete (cache-optimized + parallel)
- [x] Outer products - ✅ Complete (2D, ND, weighted, CP reconstruction)
- [x] Tucker operator - ✅ Complete (multi-mode products + reconstruction)

**Advanced Operations (2025-11-21):**
- [x] **NEW:** Tensor contractions - ✅ Complete (contract_tensors, sum_over_modes, inner_product, trace)
- [x] **NEW:** Tensor reductions - ✅ Complete (sum, mean, variance, std, norms, min/max)
- [x] **NEW:** Enhanced property tests - ✅ Complete (40 tests for mathematical correctness)

**Quality & Testing:**
- [x] Correctness property tests - ✅ **192 tests passing** (132 unit + 22 integration + 38 doc)
- [x] Comprehensive integration tests - ✅ Complete (CP-ALS, Tucker-HOOI workflows)
- [x] Performance benchmarks - ✅ Complete (13.3 Gelem/s peak, documented in PERFORMANCE.md)
- [x] Production-ready error handling - ✅ Complete (structured error types)
- [x] Utility functions - ✅ Complete (timing, validation, testing helpers)
- [ ] SIMD optimization passes - ⏳ Future

---

## M2: Decompositions - ✅ **100% COMPLETE**

### CP Decomposition (tenrso-decomp) - ✅ COMPLETE

- [x] CP-ALS baseline (dense)
- [x] Reconstruction norm with cross-terms (alpha.1 fix)
- [x] Random initialization
- [x] SVD-based initialization
- [x] Random normal initialization
- [ ] Leverage score initialization - ⏳ Future M2+
- [ ] Non-negative constraints (optional) - ⏳ Future M2+
- [ ] Regularization support - ⏳ Future M2+
- [x] Stopping criteria (tolerance, max iters)
- [x] Reconstruction error tracking (fit value)

### Tucker Decomposition (tenrso-decomp) - ✅ COMPLETE

- [x] Tucker-HOSVD (SVD-based)
- [x] Tucker-HOOI (iterative refinement)
- [x] Tucker-HOOI mode indexing fix (alpha.1)
- [x] Rank selection heuristics
- [x] Reconstruction error benchmarks

### Tensor Train (tenrso-decomp) - ✅ COMPLETE

- [x] TT-SVD baseline
- [x] TT-rank truncation (tolerance-based)
- [x] TT-SVD error bounds fix (alpha.1)
- [x] Compression ratio computation
- [x] TT-rounding
- [ ] Memory reduction verification (≥ 10×) - ⏳ Future

---

## M3: Sparse & Masked - ✅ COMPLETE

### Sparse Formats (tenrso-sparse)

- [x] COO (Coordinate) format
  - [x] N-dimensional sparse tensor storage
  - [x] Validation, sorting, deduplication
  - [x] Dense ↔ COO conversion

- [x] CSR (Compressed Sparse Row)
  - [x] 2D sparse matrix with row pointers
  - [x] Zero-copy row access
  - [x] COO ↔ CSR ↔ Dense conversions

- [x] CSC (Compressed Sparse Column)
  - [x] 2D sparse matrix with column pointers
  - [x] Zero-copy column access
  - [x] COO ↔ CSC ↔ CSR ↔ Dense conversions
  - [x] SpMM (Sparse Matrix-Matrix) operation

- [x] BCSR (Block Compressed Sparse Row)
  - [x] Block-based sparse matrix storage
  - [x] Flexible block shape specification
  - [x] Block SpMV and SpMM operations
  - [x] Conversions: from/to dense, to CSR

- [ ] CSF (Compressed Sparse Fiber) - ⏳ Planned (feature-gated)
- [ ] HiCOO (Hierarchical COO) - ⏳ Planned (feature-gated)

### Sparse Operations (tenrso-sparse)

- [x] SpMV (Sparse Matrix-Vector)
- [x] SpMM (Sparse Matrix-Matrix Multiply)
- [x] SpSpMM (Sparse-Sparse Matrix Multiply)
- [x] Masked operations (boolean masks)
- [x] Sparsity statistics (nnz, density)
- [ ] Masked einsum (dense + sparse mix) - ⏳ Planned
- [ ] Subset reductions - ⏳ Planned

---

## M4: Planner - ✅ COMPLETE

### Contraction Planning (tenrso-planner)

- [x] Einsum specification parser
- [x] Cost model (flops, memory, nnz)
- [x] Heuristic order search (greedy planner)
- [x] Representation selection (dense/sparse/low-rank)
- [x] Tiling strategy (cache-aware)
- [ ] Dynamic programming planner - ⏳ Future

### Execution Integration (tenrso-exec)

- [x] Basic dense contraction operations
- [x] CpuExecutor with planner integration
- [x] TenrsoExecutor trait
- [x] `einsum_ex` builder API
- [x] Multi-input plan execution
- [x] Device abstraction (CPU)
- [ ] Element-wise operations - ⏳ Planned
- [ ] Reduction operations - ⏳ Planned
- [ ] Memory pooling - ⏳ Planned
- [ ] Parallel execution - ⏳ Planned

---

## M5: Out-of-Core - ✅ **COMPLETE**

### I/O Backends (tenrso-ooc)

- [x] Arrow IPC reader/writer
  - [x] ArrowWriter with shape metadata encoding
  - [x] ArrowReader with shape reconstruction

- [x] Parquet reader/writer
- [x] Memory-mapped tensor access
- [x] Chunking infrastructure
- [x] Streaming execution
- [x] Optimization & auto-tuning
- [x] Profiling integration
- [x] Prefetching integration
- [x] Parallel execution
- [x] Adaptive parallel threshold tuning
- [x] SIMD-optimized elementwise operations

### Future Enhancements (tenrso-ooc)

- [ ] Deterministic chunk graph - ⏳ Planned
- [ ] Back-pressure handling - ⏳ Planned
- [ ] OoC benchmarks - ⏳ Planned
- [ ] BLAS-optimized matmul - ⏳ Planned
- [ ] Performance benchmarks - ⏳ Planned

---

## M6: AD Hooks - ✅ CORE COMPLETE

### Automatic Differentiation (tenrso-ad)

- [x] Custom VJP for einsum contractions
- [x] Gradient rules for CP-ALS
- [x] Gradient rules for Tucker-HOOI
- [x] Integration hooks for external AD frameworks
- [x] Gradient checking utilities
- [x] Integration tests
- [x] Examples
- [ ] Gradient rules for TT-SVD - ⏳ Planned
- [ ] Tensorlogic integration demo - ⏳ Planned

---

## Stretch Goals - 💡 IDEAS

### Advanced Features

- [ ] Low-rank + sparse mixed planning
- [ ] TT operations (sum, inner product, matvec)
- [ ] Robust OoC policies (prefetch, caching)
- [ ] GPU backend (CUDA/ROCm)
- [ ] Distributed execution (cluster)
- [ ] Advanced sparse formats (BSR, DIA, ELL)

### Performance Optimization

- [ ] SIMD intrinsics (AVX-512)
- [ ] Cache-oblivious tiling
- [ ] Work-stealing parallelism
- [ ] Memory-pool tuning
- [ ] Profiling dashboard

### Ecosystem Integration

- [ ] Python bindings (PyO3)
- [ ] C FFI interface
- [ ] Integration with PyTorch/TensorFlow
- [ ] ONNX tensor operations support

---

## Cross-Cutting Concerns

### Documentation

- [x] Top-level README with examples
- [x] Blueprint document
- [x] ROADMAP with milestones
- [x] CONTRIBUTING guidelines
- [x] SciRS2 integration policy
- [x] Claude development guide
- [ ] Per-crate READMEs (in progress)
- [ ] Per-crate TODOs (in progress)
- [ ] API documentation (rustdoc)
- [ ] User guide / book
- [ ] Examples collection
- [ ] Tutorials

### Testing

- [ ] Unit tests (per module)
- [ ] Integration tests (cross-crate)
- [ ] Property tests (mathematical correctness)
- [ ] Benchmarks (performance tracking)
- [ ] Fuzzing harness (unsafe code)
- [ ] Regression test suite
- [ ] CI performance budgets

### Quality Assurance

- [x] CI/CD pipeline (fmt, clippy, test)
- [x] No warnings policy (`#![deny(warnings)]`)
- [ ] Code coverage tracking
- [ ] Benchmark comparison (vs baseline)
- [ ] Memory leak detection (valgrind/ASAN)
- [ ] Performance profiling (flamegraphs)
- [ ] API stability tracking

### Infrastructure

- [x] GitHub repository
- [x] CI/CD workflows
- [ ] Benchmark dashboard
- [ ] Documentation hosting (docs.rs)
- [ ] Release automation
- [ ] Changelog generation
- [ ] Version management

---

## Current Test Status - RC.1 RELEASE

**Total Workspace Tests:** 2,109/2,109 tests passing (100%), 14 skipped

### Breakdown by Crate (RC.1)

- **tenrso-core:** 138 tests
- **tenrso-kernels:** 264 tests
- **tenrso-decomp:** 165 tests
- **tenrso-sparse:** 426 tests
- **tenrso-planner:** 271 tests
- **tenrso-ooc:** 238 tests
- **tenrso-exec:** 244 tests
- **tenrso-ad:** 154 tests

**RC.1 Status:** 2,109/2,109 tests passing (100%) — Zero known issues, all milestones M0-M6 complete!

---

## Dependencies & Blockers

### SciRS2 Integration

- [x] scirs2-core (mandatory) - Policy established
- [x] scirs2-linalg (SVD, QR) - Used in M2 decompositions
- [x] scirs2-optimize (ALS convergence) - Used in M2 CP-ALS
- [x] scirs2-sparse (COO/CSR) - Used in M3 sparse formats
- [x] scirs2-parallel (threading) - Used in M4 planner/exec

### External Crates

- [x] ndarray (via scirs2-core)
- [x] rayon (parallel iteration)
- [x] arrow/parquet (OoC I/O)
- [ ] Benchmark harness (criterion)
- [ ] Property test framework (proptest)

---

## Performance Targets (Validation Checklist)

Once implementations are complete, verify:

- [ ] Einsum: ≥ 80% of OpenBLAS baseline (1024³ matmul)
- [ ] Masked einsum: ≥ 5× speedup vs dense naive (90% zeros)
- [ ] CP-ALS: < 2s / 10 iters (256³, rank-64, 16-core CPU)
- [ ] Tucker-HOOI: < 3s / 10 iters (512×512×128, ranks [64,64,32])
- [ ] TT-SVD: < 2s build (32⁶, eps=1e-6), ≥ 10× memory reduction
- [ ] No panics in production kernels
- [ ] All unsafe code bounded and fuzzed

---

## Decision Log

### 2025-11-03: Initial Roadmap

- Established 6 milestone structure (M0-M6)
- Set MSRV to 1.82 for latest dependency support
- Decided on SciRS2-core mandatory usage
- Approved 8-crate modular architecture

---

## Notes for Contributors

- Check crate-specific `TODO.md` for detailed tasks
- Update this file when completing major milestones
- Link GitHub issues to TODO items using `#issue-number`
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for PR process
- Discuss major changes via RFC process

---

## Questions or Suggestions?

Open a GitHub issue with:
- Label: `roadmap` or `enhancement`
- Reference this TODO.md
- Tag @cool-japan maintainers
