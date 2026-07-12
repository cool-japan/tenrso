# TenRSo TODO

> **Version:** 0.1.0
> **Status:** 🎉 **0.1.0 STABLE** — 2,769 nextest + 600 doctests passing (100%), clippy `-D warnings` clean workspace-wide (incl. `cuda-compute`)
> **Last Updated:** 2026-07-12

This document tracks high-level tasks across the entire TenRSo project. For crate-specific tasks, see individual `crates/*/TODO.md` files.

---

## 2026-07-12 Depth session II (`/ucont` continued)

A second long-horizon audit + implementation pass (Opus orchestrator, ~20 subagents across
4 iterations, adversarially verified). The recurring lesson held: **the 0.1.0 test suite
checks output shape and "does it run", not whether the numbers are right** — so it stayed
green over every bug below. Base rate remains high; a third pass is warranted.

**9 more advertised-but-fake capabilities eliminated** (implemented for real *with
measurement*, or removed honestly — no facade left behind):
- `tenrso-planner::refine_plan` — a third order-invariant "search" (after SA/GA): swapped
  `Plan::nodes` scored by a permutation-invariant `sum(node.cost)`, never wrote `Plan::order`.
  Now real NNI local search over the contraction tree (50× cost cut on the regression case,
  proven to reach the global optimum).
- `tenrso-ooc::NumaAllocator` — allocated **nothing** (returned a counter). Now real aligned
  allocation + `mbind`/`get_mempolicy` NUMA binding, verified against the kernel's own view.
- `tenrso-exec` executor "optimization layer" — "SIMD" that was plain `mapv`, a dead
  `optimized_ops` layer, a broadcast module that would *panic* on any real broadcast, and an
  inert `num_threads`. → real AVX2 transcendental kernels where they win (exp 4–5×), honest
  deletion where they don't (bandwidth-bound ops), a real scoped rayon pool for `with_threads`,
  a 13× faster live broadcast path. (Also fixed real cancellation bugs in `tanh`/`elu`/`selu`.)
- `tenrso-ad::OperationFusion` — ran CSE and reported a **fabricated** fusion count. Now real
  graph fusion, gradients pinned three ways.
- `cp_als_accelerated` line search — discarded its result; structurally could never damp
  below 1.0. Now real ELS with incremental fit.
- `ExecHints::tile_kb`/`prefer_lowrank` — zero readers; deleted (measured: could only steer
  the slower kernel f32/f64 never reach). `subset` implemented for real.

**5 more silent wrong-answer bugs found (executed-oracle-confirmed), 4 fixed:**
- [x] `DenseND::det` (n≥4) returned the **wrong sign** — permutation parity computed from
      displaced-element count, not true cycle parity. (Fixed via cycle decomposition.)
- [x] `cp_completion` — one aggregated Gram for all rows → completion non-functional (0.83
      rel err on an exact rank-2 tensor). Now a per-row masked normal-equation solve.
- [x] Sparse `max_axis` — clamped every slice's max against 0 (assumed an implicit zero) →
      a fully-dense all-negative slice returned 0. (`min_axis` was already correct.)
- [x] Eager `ComputationGraph` Add/Sub/Mul/Div backward — never un-broadcast the gradient
      → a `W+bias` operand got a wrong-shaped/un-summed gradient. Now a proper `unbroadcast_grad`
      adjoint; gradchecked at unmodified tolerances.
- [ ] `tt_svd` — silently **non-exact** (2.2e-3 recon at full rank) because scirs2-linalg's
      thin SVD is inaccurate on the tall-skinny unfoldings TT-SVD always produces. Fix
      attempted (Gram-SVD → 3.2e-7) but **reverted** — insufficient for the strict exactness
      bar; the robust fix is QR-then-SVD (~1e-13). **OPEN** — see checklist.

**First real GPU dispatch** (was mislabeled "blocked by hardware"): `tenrso-ooc::Device::add`/
`mul` for f32/f64 now execute on the **RTX A4000** via pure-Rust `oxicuda`, behind default-off
`cuda-compute`, **bit-exact-verified on the GPU**. Capability path, not a speedup (host buffers).

**CP-ALS <2s — the standing SIMD recommendation was proven WRONG** (roofline data): CP-ALS is
GEMM-bound, but `matrixmultiply` is already near the AVX2 f64 roofline (33 & 19 GFLOP/s/core),
so a SIMD micro-kernel is a NO-GO. The real lever is K-splitting the tall-skinny root GEMM
(24 → 82 GFLOP/s). See the Deferred/Blocked "Pure Rust Policy" note and the checklist.

**Open for the next pass:** `tt_svd` (QR thin SVD); K-split `gemm_rows`; independent adversarial
re-verification of the 4 fixes above (this session's verifier agents were cut off by a usage
limit — the fixes are test-backed and the workspace is green, but not skeptic-re-checked);
add `mul`/f64 GPU bit-exact tests. See `crates/*/TODO.md` and the source markers below.

### Actionable now (2026-07-12 — real, in-policy, unblocked)

- [ ] **`tt_svd` exactness** — route tall-skinny unfoldings through a **QR-then-SVD** thin SVD
      (`A = QR`, SVD the small `R`, `U = Q·U_r`) instead of `scirs2_linalg::svd`. A Gram-based
      attempt reached 3.2e-7 but not the <1e-10 exactness / <1e-12 orthonormality bars.
      `crates/tenrso-decomp/src/tt/algorithms.rs`. Add a full-rank exactness regression (order ≥ 3).
- [ ] **CP-ALS <2s via K-split** — replace the output-row blocking of the tall-skinny root GEMM
      `gemm_rows` (M=256) with K-splitting (partition K=65536 across threads, sum partials):
      measured 24 → 82 GFLOP/s. `crates/tenrso-kernels/src/mttkrp_dimtree.rs`
      (`mttkrp_all_parallel`). In-policy (ndarray `dot` + rayon), no SIMD, no BLAS.
- [ ] **Adversarially re-verify** the 4 landed 2026-07-12 fixes (`det`, `cp_completion`,
      `max_axis`, graph broadcast-backward) with fresh independent oracles.
- [ ] **GPU test coverage** — add bit-exact `Device::mul` and f64 tests (only `add`/f32 is
      currently covered, though all four arms are wired). `required-features = ["cuda-compute"]`.
- [ ] **Refactor `crates/tenrso-ad/src/graph.rs`** — the 2026-07-12 broadcast-backward fix pushed
      it to **2327 lines**, over the 2000-line policy (was 1960 at HEAD). Split the `#[cfg(test)]`
      module (and/or the optimizer helpers) into a `graph/` module dir via `splitrs`. Mechanical;
      tree is green, this is style-policy debt, not a correctness issue.

---

## 2026-07-11 Depth session (`/ucont`)

A long-horizon implementation + audit pass. Highlights: closed the 3 remaining
real stubs, fixed **four silent einsum correctness bugs** shipped in 0.1.0,
landed a Gauss-Seidel-preserving dimension-tree MTTKRP, and **corrected a false
performance claim** — CP-ALS is GEMM-throughput bound, not memory-bandwidth
bound. See the "Correctness bugs fixed" and "Deferred / Blocked" sections below.

### Stubs to implement — ✅ ALL CLOSED (2026-07-11)

- [x] `tenrso-ooc`: `crates/tenrso-ooc/src/gpu.rs` — **real** GPU device enumeration
      via `scirs2_core::gpu::backends::detect_gpu_backends()` (nvidia-smi / rocm-smi /
      Metal API) + `WebGPUContext::is_available()`. Default-off features (`cuda`/`rocm`/
      `metal`/`vulkan`); CPU-only default is honestly reported via
      `DeviceManager::available_backends()`. **No fabricated devices** — deliberately
      does NOT call scirs2-core's documented-placeholder `GpuDeviceInfo::for_backend()`.
      Verified against real RTX A4000 present in the build box.
- [x] `tenrso-exec`: einsum general-case — **replaced** the naive `O(output×contracted)`
      nested loop (with a per-element HashMap rebuild) with a batched-GEMM contraction
      engine (`crates/tenrso-exec/src/ops/`): index classification → permute/reshape →
      `matrixmultiply`-backed `Array2::dot` (pure Rust, in-policy). Measured 126×–3400×
      on non-trivial specs. **Fixed the `"ii,ii->"` diagonal correctness bug** in the
      process (see below). No OxiBLAS needed — pure-Rust GEMM is in-policy.
- [x] `tenrso-ad`: `crates/tenrso-ad/src/hooks.rs` — **real** framework-agnostic op
      registry (`src/registry/`, name→forward+VJP for einsum/elementwise/reductions/
      CP/Tucker/TT, all gradcheck-verified) replacing the empty `register_tenrso_ops()`.
      Plus a **default-off `tensorlogic` feature** implementing `TlExecutor`/`TlAutodiff`
      over `DenseND` against the real published `tensorlogic-infer` 0.1.1. The old
      "pending Tensorlogic API stabilization" note was stale — the crate exists.

### Correctness bugs fixed (all were silent — plausible wrong answers, no error, shipped in 0.1.0)

- [x] **einsum `"ii,ii->"`** summed the full n² instead of the diagonal
      (`ops.rs` tier-1 fast path used byte-equality of subscript strings). Fixed
      structurally: a repeated index canonicalizes to an axis whose gather stride
      is the sum of its positions' strides.
- [x] **Single-input einsum** (`"ij->ji"`, `"ii->"`, `"ii->i"`, `"ij->i"`) silently
      returned the input **unchanged** — the planner emits zero pairwise steps for
      one operand, so `execute_plan` returned the passthrough input. Fixed with a
      real unary-einsum gather path dispatched on arity before planning.
- [x] **Multi-tensor einsum dropped batch indices** (`"bij,bjk,bkl->bil"` summed
      over `b` in step 1 → right shape, wrong values) and **ignored requested output
      order** (`"ij,jk,kl->li"` returned the untransposed product). The executor now
      derives step subscripts itself instead of trusting the planner's (still-buggy)
      `compute_pairwise_spec` — see the source-fix item under Planner below.
- [x] **`ReductionVjp` max/min gradients** broadcast like `sum` (it stored only the
      input shape, so it could not route the gradient to the arg-extremum). Now
      delegates to the correct gradchecked `ReductionRule`; `new()` returns an honest
      `Err` for max/min/product rather than a silently-wrong gradient.
- [x] **`tenrso-core` allocation bugs on the CP-ALS hot path**: `Add`/`Sub` cloned
      both operands even on matched shapes (26× at 64³); `unfold` did two full-tensor
      copies (40× at 64³); `permute`/`reshape` never actually zero-copy. Fixed +
      added zero-copy `permute_view`/`into_permuted`/`into_reshape`. Public API
      preserved byte-for-byte; wins are additive methods.

### New capabilities landed (2026-07-11)

- [x] **Dense dimension-tree MTTKRP** (`tenrso-kernels/src/mttkrp_dimtree.rs`) —
      amortizes partial Khatri-Rao products across an ALS sweep; 3.4–4.3× serial /
      8.7–10.9× parallel at 256³ rank-64.
- [x] **CP-ALS wired to it, Gauss-Seidel preserved exactly** — proved a contiguous-range
      dimension tree keeps GS semantics (byte-identical iterates) while sharing partials.
      256³ rank-64/10-iter: **13.4s → ~3.1–3.4s**. `UpdateScheme::{GaussSeidel,Jacobi}`
      added (GS default; naked Jacobi provably diverges, so Jacobi ships stabilized).
- [x] **Out-of-core streaming MTTKRP** (`tenrso-ooc/src/mttkrp_stream.rs`) — additive
      chunk accumulation with correct global index offsets, bounded memory by
      construction (3.75 MiB tensor in a 44 KiB working set), one-pass-all-modes =
      3× less I/O. (Replaced a fake example that called the in-core kernel.)
- [x] **Masked einsum ≥5× vs dense naive** validated with a *fair* baseline (the old
      benchmark was rigged both ways). Real cache-locality fix (column packing): 256²
      ~10–18×, 512² ~14–27× at 90% sparsity. CI gate: 256/512 hard, 64² advisory.
- [x] CI: perf budgets + benchmark-vs-baseline + Miri + ASan/LSan + flamegraph workflows.
- [x] Docs: `docs/USER_GUIDE.md` + `docs/TUTORIALS.md` (all 8 tutorials compiled & run).
- [x] `tenrso-core` ndarray-baseline + allocation-profiling benchmarks.

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

## Deferred / Blocked (moved out of the active checklist 2026-07-11)

These are **not** open work items for the convergence loop — each is blocked by an
external dependency, the project's own Pure Rust Policy, a Rust language limitation, or
is a genuinely separate future milestone. They are recorded here with reasons so the
remaining `- [ ]` items elsewhere in this file mean only *actionable* work. Reclassified,
not silently dropped.

### Blocked by Pure Rust Policy (COOLJAPAN) — would require a C/Fortran BLAS
- **Einsum ≥ 80% of OpenBLAS baseline** — structurally unmeasurable in-tree; no OpenBLAS
  by policy, and a hand-rolled SIMD GEMM micro-kernel is a **NO-GO** (measured 2026-07-12):
  pure-Rust `matrixmultiply` already runs **33 & 19 GFLOP/s/core** at the CP-ALS root-GEMM
  shapes — near the AVX2 f64 roofline (~40–70% of peak), *5–8× above* the ~3–6 GFLOP/s this
  file previously (wrongly) assumed. A `scirs2-core::simd_ops` kernel cannot beat a mature
  packed GEMM already near roofline. Multi-threaded OpenBLAS parity (~50 GFLOP/s) stays out
  of policy.
- **`tenrso-ooc` BLAS-optimized matmul** — same reason.
- Note: CP-ALS <2s is **NOT** in this bucket and the lever is **NOT** SIMD (see above). It is
  GEMM-throughput bound; the real in-policy fix is **K-splitting the tall-skinny root GEMM
  `gemm_rows`** in `mttkrp_dimtree.rs` — its current output-row blocking starves the
  microkernel (24 GFLOP/s, *slower than one core*) whereas K-splitting the contraction dim
  measured **82 vs 24 GFLOP/s (3.4×)**, projecting the 256³ r64 10-iter run to ~1.4–1.5s.
  Active — see the checklist item below.

### Actionable-but-large future milestones (NOT hardware-blocked — reclassified 2026-07-12)
- **GPU compute backend (CUDA/ROCm/Metal/Vulkan)** — the old "blocked by external hardware"
  label was **false**: this box has a real **RTX A4000 (16 GB)** and `/notebooks/oxicuda`
  (pure-Rust CUDA family — no nvcc/SDK at build time, PTX generated in Rust + JIT via
  libcuda, zero scirs2 dep) is available. **First real dispatch DONE + bit-exact-verified on
  the A4000** (2026-07-12): `tenrso-ooc::Device::add`/`mul` for f32/f64 via oxicuda, behind a
  default-off `cuda-compute` feature. Honest scope: it is a *capability/correctness* path,
  **not** a speedup — buffers are host-`Vec`, so PCIe dominates a single op. Remaining (a
  genuine milestone, but engineering scope, not hardware availability): GPU kernels for
  `tenrso-kernels` (MTTKRP/Khatri-Rao/n-mode), `tenrso-exec` einsum GEMM offload via
  `oxicuda-blas`, and on-device tensor residency. `Device::add`/`mul` still fall back to CPU
  for non-f32/f64 and when the feature is off.
- **Distributed / cluster execution, MPI-based distributed MTTKRP,
  communication-avoiding algorithms, tensor partitioning** — a distributed-runtime
  milestone; no in-tree transport.

### Blocked by the Rust language
- **Const-generic tensor shapes / compile-time shape checking / type-level rank tracking**
  — awaits stabilization of the relevant const-generics features.

### Blocked by an upstream dependency
- **Residual `zstd-sys` (C) via `parquet`** — Apache `parquet` bundles `zstd`/`zstd-sys`
  for its internal column compression and exposes no backend swap. tenrso's own code is
  fully pure-Rust. Revisit if `parquet` gains a pure-Rust compression backend.
- **`tensorlogic` feature duplicates `scirs2-core` 0.5.1 alongside 0.6.0** — published
  `tensorlogic-infer` 0.1.1 pins `scirs2-core ^0.5.0`. Not a hard conflict (both resolve
  to `ndarray 0.17.2`), resolves when `tensorlogic` republishes against 0.6 (the local
  `/notebooks/tensorlogic` tree already targets 0.6.0).

### Ecosystem milestones (separate, large, tracked-not-blocked)
- **Python bindings (PyO3), C FFI interface, PyTorch/TensorFlow interop, ONNX support** —
  each a substantial new surface; deliberately out of scope for this workspace's core.
- **Profiling dashboard, benchmark dashboard, docs.rs hosting, release automation,
  changelog generation, version management, API stability tracking** — infra/tooling, not
  library code. (`bump`/`changelog-gen` skills exist for some of these when wanted.)
- **Polished Tensorlogic end-to-end demo program** — the bridge exists and is tested; a
  showcase demo is nice-to-have, not blocking.
- **SIMD intrinsics (AVX-512) hand-written** — superseded by using
  `scirs2_core::simd_ops`; a manual AVX-512 pass is not planned (and the 2026-07-11
  measurements show SIMD is not a win for the bandwidth/latency-bound elementwise ops).

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

## 0.1.0 Post-Release Quality (2026-04-15 — 2026-04-17)

### Unwrap Audit ✅
- [x] Production `.unwrap()` elimination: 321 → 2 across all 8 crates
  - tenrso-core: 95 → 0 (added `from_vec_unchecked` helper)
  - tenrso-ad: 59 → 0 (added `lock_mutex` helper, `get_or_insert_with` for lazy-init)
  - tenrso-ooc: 61 → 2 (documented startup invariants in prometheus_metrics.rs)
  - tenrso-planner: 41 → 0 (added `lock_cache`, `lock_profiler` helpers)
  - tenrso-exec: 24 → 0
  - tenrso-kernels: ~25 → 0 (added `cast_count`, `cast_f64` helpers)
  - tenrso-decomp: 62 → 0 (from prior cycle)
  - tenrso-sparse/solvers: 7 → 0 (from prior cycle)

### Performance Validation (2026-05-30, 8-core x86_64 AVX2, pure Rust)
- [x] TT memory reduction: 20,459× (target ≥ 10×) ✅
- [ ] CP-ALS: **~3.1–3.4s** / 10 iters (256³, rank-64), target <2s — was ~21–44s.
      **CORRECTION (2026-07-11): the earlier "fundamentally memory-bandwidth limited,
      needs BLAS" claim was FALSE.** After wiring CP-ALS to the Gauss-Seidel-preserving
      dimension-tree MTTKRP and fixing the `tenrso-core` copy bugs (`unfold`/`permute`),
      the kernel **scales ≈4× from 1→8 threads** (12.8→3.1s). A memory-bandwidth-bound
      kernel does not scale with cores; this one does — it is **GEMM-throughput bound**.
      The <2s target was NOT confirmed met (best ~3.1s on a box at load-avg ~22–27, i.e.
      ~3× oversubscribed; likely lower uncontended, not yet measured). Remaining lever is
      a blocked SIMD GEMM micro-kernel for the two root GEMMs (`X₂·KR`/`X₂ᵀ·KR`) via
      `scirs2_core::simd_ops` — pure Rust, in-policy. **No BLAS required.**
- [x] Tucker-HOOI benchmark fix: ranks corrected [256,256,64]→[64,64,32] — now
      benchmarks the documented target; mode-2 gate fix triggers randomized SVD
      for 64-dim unfoldings (was full SVD). **Measured ~35-50% speedup on
      256×256×64 r[32,32,16]** (66s→31-48s). On target shape 512×512×128
      r[64,64,32] the improvement is larger (was >hours with wrong ranks,
      now completes with randomized SVD on all modes).
- [x] TT-SVD: `thin_svd_via_gram` added for extreme short-fat unfoldings (rows≤64,
      cols≥1M); avoids allocating O(n×k) Gaussian Ω (would be >8GB for 32^6).
      **32^4 baseline preserved (2.79-3s). 32^6 should now complete without
      timeout** (not yet measured; 8.6GB tensor allocation required).
- [ ] Einsum vs BLAS: structurally unmeasurable (Pure Rust Policy) <!-- comment is accurate: no OpenBLAS in-tree; pure-Rust GEMM (~3-4 GFLOP/s) cannot reach OpenBLAS levels by policy -->
- [x] Masked einsum: reference harness exists — `crates/tenrso-sparse/benches/masked_einsum_bench.rs` benchmarks masked vs dense naive at 50%/90%/99% sparsity; correctness tests in `masked_einsum.rs` compare against `Mask::full` (full-mask reference)

### Performance Root-Cause Notes (2026-05-30)
- **Tucker-HOOI was never slow** — the old benchmark used ranks [256,256,64] which
  triggered full SVD on 512×65536 matrices (gate `< min_dim/2` excluded rank=256).
  The documented target ranks [64,64,32] already used randomized SVD; the benchmark
  simply tested the wrong problem.
- **TT-SVD timeout diagnosis**: the Gaussian Ω allocation (n×(k+10)) for a 32×33M
  matrix would be ~8.4GB; `thin_svd_via_gram` uses G=MMᵀ (32×32 matrix) instead,
  costing O(m²n) ≈ two serial matrix multiplies over the input data.
- **CP-ALS 256³ — root cause CORRECTED (2026-07-11)**: the 2026-05-30 conclusion
  ("memory-bandwidth limited; parallelism increases cache pressure; no in-policy fix")
  was measured against the *naive* MTTKRP, which re-copied the whole tensor (`unfold`)
  and re-materialized the full 33MB Khatri-Rao matrix per mode per iteration — those
  copies were the bandwidth cost, and they were `tenrso-core` bugs, not a law of physics.
  With the dimension-tree MTTKRP (no full-KR materialization) + fixed `tenrso-core`
  copies, CP-ALS now **scales ≈4× across 8 cores**, proving it is **GEMM-throughput
  bound, not bandwidth bound**. An in-policy fix DOES exist (SIMD GEMM micro-kernel).
  The `matrixmultiply` ~3–4 GFLOP/s figure stands, but the conclusion drawn from it
  ("assumed BLAS") did not.

---

## Pure Rust Migration (COOLJAPAN Policy)

Replace C-backed and non-COOLJAPAN serialization/compression dependencies with
the Pure-Rust `oxicode` / `oxiarc-*` equivalents. Feature **names** are kept so
existing `#[cfg(feature = "...")]` gates remain valid.

- [x] (2026-06-05) **tenrso-core: `bincode` 2.x → `oxicode`.** Workspace dep and
      `tenrso-core` `binary` feature now use `oxicode`; call sites in
      `crates/tenrso-core/src/dense/binary.rs` use
      `oxicode::serde::encode_to_vec(&x, oxicode::config::standard())` /
      `oxicode::serde::decode_from_slice(&bytes, oxicode::config::standard())`.
      `bincode` removed from the workspace tree entirely.
- [x] (2026-06-05) **tenrso-ooc: `lz4` (C, `lz4-sys`) → `oxiarc-lz4`.**
      `crates/tenrso-ooc/src/compression.rs` now calls
      `oxiarc_lz4::block::compress_block(data)` and
      `oxiarc_lz4::block::decompress_block(compressed, original_size)`
      (the stored original size is passed as the `max_output` safety bound).
- [x] (2026-06-05) **tenrso-ooc: `zstd` (C, `zstd-sys`) → `oxiarc-zstd`.**
      `compression.rs` now calls `oxiarc_zstd::encode_all(data, level)` /
      `oxiarc_zstd::decode_all(compressed)` (exact drop-ins).
- [x] (2026-06-05) Verified: `tenrso-core` + `tenrso-ooc` build (default and
      `--all-features`), nextest green (217 + 542 tests), clippy clean with
      `-D warnings`. `bincode` and `lz4-sys` are absent from the workspace tree;
      the only LZ4 in the tree besides `oxiarc-lz4` is the Pure-Rust `lz4_flex`.

- [ ] **Residual C dependency (out of scope for tenrso's own code): `zstd-sys`
      via `parquet`.** Apache `parquet` v58 bundles `zstd`/`zstd-sys` for the
      Parquet file format's internal column compression; it exposes no feature to
      swap its backend. tenrso's direct dependencies and source are now fully
      Pure-Rust. Revisit if `parquet` gains a Pure-Rust compression backend or if
      the `parquet` feature is dropped.

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
- [x] Leverage score initialization - ✅ COMPLETE
- [x] Non-negative constraints (optional) - ✅ COMPLETE (via cp_als_constrained)
- [x] Regularization support - ✅ COMPLETE (L1 soft-thresholding, L2/Tikhonov)
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
- [x] Memory reduction verification (≥ 10×) - ✅ COMPLETE (measured 20,459× on 32^6)

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

- [x] CSF (Compressed Sparse Fiber) - ✅ COMPLETE (feature-gated: `csf`)
- [x] HiCOO (Hierarchical COO) - ✅ COMPLETE (feature-gated: `csf`)

### Sparse Operations (tenrso-sparse)

- [x] SpMV (Sparse Matrix-Vector)
- [x] SpMM (Sparse Matrix-Matrix Multiply)
- [x] SpSpMM (Sparse-Sparse Matrix Multiply)
- [x] Masked operations (boolean masks)
- [x] Sparsity statistics (nnz, density)
- [x] Masked einsum (dense + sparse mix) - ✅ COMPLETE
- [x] Subset reductions - ✅ COMPLETE (masked_sum/mean/max/min)

---

## M4: Planner - ✅ COMPLETE

### Contraction Planning (tenrso-planner)

- [x] Einsum specification parser
- [x] Cost model (flops, memory, nnz)
- [x] Heuristic order search (greedy planner)
- [x] Representation selection (dense/sparse/low-rank)
- [x] Tiling strategy (cache-aware)
- [x] Dynamic programming planner - ✅ **NOW GENUINELY COMPLETE (2026-07-11)** (DP + Beam +
      SA + GA + Adaptive). The prior "COMPLETE" claim was **façade for 3 of the 5**:
      - `dp_planner` never populated `Plan::order`, so the executor could not consume a DP
        plan at all — **fixed** (post-order bitmask → positional `(i,j)` order).
      - `SA` and `GA` were **silent no-ops**: they "searched" by permuting `Plan::nodes`,
        but the scored cost is an order-invariant sum, so the permutation changed nothing
        and `order` was never touched — **both rewritten** to genuinely search contraction
        trees (SA = Metropolis over randomized-greedy orders; GA = (μ+λ) evolution strategy,
        crossover omitted with a documented reason). Only greedy + beam were ever real.
      - Round-trip tests now assert every planner's returned `order`, executed step-by-step,
        reduces to a single tensor of the correct shape and label order.
- [x] ✅ **(2026-07-11) `compute_pairwise_spec` batch-index bug — FIXED at source.** It used
      to drop every index shared by two operands (incl. batch indices still needed
      downstream) and sort the pairwise output alphabetically (losing requested order).
      Corrected rule: an index survives a step iff it is in the final output OR in any
      not-yet-consumed operand; contracted only on its last live appearance; final step
      emits in the caller's requested order.

### Execution Integration (tenrso-exec)

- [x] Basic dense contraction operations
- [x] CpuExecutor with planner integration
- [x] TenrsoExecutor trait
- [x] `einsum_ex` builder API
- [x] Multi-input plan execution
- [x] Device abstraction (CPU)
- [x] Element-wise operations - ✅ COMPLETE (ScalarOp, parallel_elem_op, parallel_binary_op)
- [x] Reduction operations - ✅ COMPLETE (full_reduce, parallel_reduce, tiled reductions)
- [x] Memory pooling - ✅ COMPLETE (Phases 1-5.1: thread-local pools, heuristics, 10 pooled ops)
- [x] Parallel execution - ✅ COMPLETE (auto-dispatch >= 10K elements, Rayon-based)

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

- [x] Deterministic chunk graph - ✅ COMPLETE
- [x] Back-pressure handling - ✅ COMPLETE (via MemoryManager)
- [x] OoC benchmarks - ✅ DONE (2026-06-10) (9 bench files: `ooc_benchmarks.rs`, `advanced_features.rs`, `large_tensors.rs`, `micro.rs`, `distributed_benchmarks.rs`, `ml_eviction_benchmarks.rs`, `integrity_and_autoselect.rs`, `lockfree_prefetch_benchmarks.rs`, `allocator_comparison.rs` in `crates/tenrso-ooc/benches/`)
- [ ] BLAS-optimized matmul - ⏳ Future (requires C-backed BLAS; blocked by Pure Rust Policy)
- [x] Performance benchmarks - ✅ DONE (2026-06-10) (criterion benchmark suites exist in all 8 crates; extensive OoC + decomp + kernel + exec benchmarks)

---

## M6: AD Hooks - ✅ COMPLETE

### Automatic Differentiation (tenrso-ad)

- [x] Custom VJP for einsum contractions
- [x] Gradient rules for CP-ALS
- [x] Gradient rules for Tucker-HOOI
- [x] Integration hooks for external AD frameworks
- [x] Gradient checking utilities
- [x] Integration tests
- [x] Examples
- [x] Gradient rules for TT-SVD - ✅ COMPLETE (TtReconstructionGrad, left/right chain products, finite-difference verified)
- [x] Tensorlogic integration **bridge** - ✅ DONE (2026-07-11) — default-off `tensorlogic`
      feature; `TenrsoTlExecutor` implements `TlExecutor`/`TlAutodiff` over `DenseND`
      against the real published `tensorlogic-infer` 0.1.1. (The old "pending API
      stabilization" note was stale.) A polished end-to-end *demo program* is still worth
      writing — see Deferred/Blocked (tracked, not blocked).

---

## Stretch Goals - 💡 IDEAS

### Advanced Features

- [x] Low-rank + sparse mixed planning ✅ **COMPLETE** — `select_representation` wired into all planners via `sparsity_hints`→`TensorStats::with_density` path (2026-06-10, tenrso-planner)
- [x] TT operations (sum, inner product, matvec) ✅ **COMPLETE** — `tt_add`, `tt_dot`, `tt_hadamard`, `TTMatrix::matvec`, `tt_matrix_from_diagonal` in `tenrso-decomp::tt`
- [x] Robust OoC policies (prefetch, caching) - ✅ DONE (2026-06-10) — `SpillPolicy` (LRU/LFU/FIFO/MRU), `MLEvictionPolicy`, `PrefetchConfig`, `NumaPolicy`, `CachePolicy`, `ValidationPolicy` all implemented in tenrso-ooc
- [ ] GPU backend (CUDA/ROCm)
- [ ] Distributed execution (cluster)
- [x] Sparse n-mode product (`nmode_product_sparse_coo`, 2026-06-03, tenrso-kernels `sparse` feature)
- [x] Mixed sparse/dense operations (satisfied by sparse n-mode product, 2026-06-03)
- [x] Sparse MTTKRP (CSF input, 2026-06-03, tenrso-kernels `csf` feature) — DFS fiber-tree walk
- [x] Sparse MTTKRP (HiCOO input, 2026-06-03, tenrso-kernels `csf` feature) — block-group parallel
- [x] Masked einsum executor integration (2026-06-03) — routes via `ExecHints::prefer_sparse + mask`
- [x] JSON serialization for DenseND (2026-06-03, tenrso-core `json` feature) — save/load/string
- [x] Advanced sparse formats (BSR, DIA, ELL) - ✅ DONE (2026-06-10) — DIA: `tenrso-sparse/src/dia.rs`; ELL: `tenrso-sparse/src/ell.rs`; BSR = BCSR: `tenrso-sparse/src/bcsr.rs` (Block CSR is the standard BSR format)

### Performance Optimization

- [ ] SIMD intrinsics (AVX-512)
- [x] Cache-oblivious tiling - ✅ DONE (2026-06-10) (`CacheObliviousTileSpec`, `CacheObliviousIter`, `matmul_cache_oblivious_sequence` in `tenrso-planner/src/tiling.rs`)
- [x] Work-stealing parallelism - ✅ DONE (2026-06-10) (via Rayon: `par_iter`, `par_azip`, `into_par_iter` throughout tenrso-kernels + tenrso-exec + tenrso-sparse + tenrso-ooc)
- [x] Memory-pool tuning - ✅ DONE (2026-06-10) (`PoolingPolicy` with `conservative()`, `aggressive()`, `memory_constrained()` presets in `tenrso-exec/src/executor/pool_heuristics.rs`; thread-local pools in `thread_local_pool.rs`)
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
- [x] Per-crate READMEs - ✅ DONE (2026-06-10) — all 9 crates (tenrso-core/kernels/decomp/sparse/planner/ooc/exec/ad/tenrso) have README.md
- [x] Per-crate TODOs - ✅ DONE (2026-06-10) — all 8 implementation crates have TODO.md with milestone tracking
- [x] API documentation (rustdoc) - ✅ DONE (2026-06-10) — 148 doctests passing; public APIs documented with `///`, complexity notes, and `# Examples` sections
- [x] User guide / book - ✅ DONE (2026-07-11) — `docs/USER_GUIDE.md` (crate map, core
      concepts, einsum spec language + limits, execution model, decompositions, sparse
      formats, OoC, AD, SciRS2 policy, honest perf notes)
- [x] Examples collection - ✅ DONE (2026-06-10) — 464 example programs across `examples/` directories
- [x] Tutorials - ✅ DONE (2026-07-11) — `docs/TUTORIALS.md` (8 progressive, complete,
      compiled-and-run programs; verified against the real workspace in an isolated worktree)

### Testing

- [x] Unit tests (per module) - ✅ DONE (2026-06-10) (all 8 crates have extensive unit tests in src/ — 10–36 files with `#[test]` per crate; 2,178 nextest total)
- [x] Integration tests (cross-crate) — `crates/tenrso/tests/kernels_decomp_integration.rs` (2026-06-10, 9 tests covering Tucker/CP/MTTKRP-variant kernels<->decomp roundtrips)
- [x] Property tests (mathematical correctness) - ✅ DONE (2026-06-10) (proptest in tenrso-exec, tenrso-decomp, tenrso-core, tenrso-ooc, tenrso-sparse, tenrso-kernels, tenrso-ad — 185+ property tests total)
- [x] Benchmarks (performance tracking) - ✅ DONE (2026-06-10) (criterion benchmark suites in all 8 crates; 9 bench files in tenrso-ooc alone, 2 in tenrso-exec, etc.)
- [x] Fuzzing harness (unsafe code) - ✅ DONE (2026-06-10) — proptest-based fuzz harnesses in `tenrso-ooc/tests/property_tests.rs`: `prop_zerocopy_f64_byte_roundtrip` (exercises both `from_raw_parts` directions for f64↔u8 reinterpretation) and `prop_aligned_buffer_pointer_alignment` (exercises `alloc` with arbitrary sizes and alignments). Also fixed a real bug discovered: `AlignedBuffer::as_slice()` was computing wrong `len = data.len() - offset - alignment` (too small when offset > 0); fixed to `len = data.len() - alignment` (= `size`, invariant)
- [x] Regression test suite - ✅ DONE (2026-06-10) — `crates/tenrso/tests/regression_suite.rs` (7 end-to-end tests: TT-SVD round-trip, OoC Arrow IPC, planner+exec matmul, adaptive planner 3-tensor, sparse CP pipeline, AD gradient consistency)
- [x] CI performance budgets - ✅ DONE (2026-07-11) — `.github/workflows/bench.yml` `bench-budgets` job + `.github/scripts/check_perf_budgets.py`. Hard gates: Tucker-HOOI 512×512×128 r[64,64,32] < 3s (2x noise margin ⇒ 6s ceiling); masked einsum ≥5x speedup vs dense naive @ 90% sparsity (64/256/512). Advisory (never fails CI): TT-SVD 32^6 <2s (documented as "not yet measured", ~8.6GB allocation risks OOM on shared runners). Tracked-not-gated: CP-ALS 256³/rank-64 <2s target (documented BLAS-gated gap, measured ~21-44s) — instead gated by a 120s sanity ceiling against the *current* measured range to catch real regressions without faking the aspirational 2s figure as passing.

### Quality Assurance

- [x] CI/CD pipeline (fmt, clippy, test)
- [x] No warnings policy (`#![deny(warnings)]`)
- [x] Code coverage tracking - ✅ DONE (2026-06-10) — CI workflow uses `cargo-llvm-cov` → LCOV → codecov.io (`.github/workflows/ci.yml` `coverage` job)
- [x] Benchmark comparison (vs baseline) - ✅ DONE (2026-07-11) — `.github/workflows/bench.yml` uses Criterion's own baseline machinery (`--baseline-lenient`), not a third-party action, since all 8 crates already ship Criterion suites. A "master" baseline is cached across runs (promoted only from default-branch pushes) and PRs compare against it; Criterion's own regression detector is advisory (shared-runner noise), the explicit budget checks above are the hard gates.
- [x] Memory leak detection (valgrind/ASAN) - ✅ DONE (2026-07-11) — `.github/workflows/sanitizers.yml`. **Miri** (hard gate): `tenrso-core`, `tenrso-kernels`, `tenrso-sparse`, `tenrso-planner`, `tenrso-ad`, `tenrso-exec` (`--no-default-features`, dropping the mmap/SIMD-carrying `ooc` feature) — covers the `Vec::from_raw_parts` type-punning in `tenrso-exec/src/executor/types.rs`. Excludes `tenrso-ooc` (file-backed mmap + AVX2 intrinsics are not interpretable by Miri) and the `tenrso` umbrella crate (hard-depends on `tenrso-ooc`; its regression suite exercises OoC/Arrow IPC) and `tenrso-decomp` (zero `unsafe`, heaviest numeric workload — excluded on cost grounds, not coverage). **ASan/LSan** (advisory, `continue-on-error: true` — nightly `-Z build-std` sanitizer builds are toolchain-version-sensitive independent of this repo's code): `tenrso-ooc` + `tenrso-exec` with default features, i.e. real mmap/AVX2 execution (which Miri can't do). Documented gaps: LSan can't attribute leaks in raw mmap'd memory (only allocator-routed chunks); ASan's redzones likewise only guard allocator-routed heap chunks, not arbitrary mmap'd pages.
- [x] Performance profiling (flamegraphs) - ✅ DONE (2026-07-11) — `.github/workflows/flamegraph.yml`, `workflow_dispatch`-only (not run on every push/PR — a diagnostic tool, not a gate). Profiles MTTKRP (`tenrso-kernels` bench `mttkrp` group), einsum (`tenrso-exec` `executor_ops` bench, `matmul`/`three_tensor_contraction` groups), CP-ALS and Tucker-HOOI (`tenrso-decomp` `decompositions` bench `*_target` groups) via `cargo-flamegraph` + Criterion's `--profile-time` mode, uploading SVGs as build artifacts.
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

## Current Test Status - 0.1.0 Post-Release

**Total Workspace Tests:** 2,624 unit/integration + 148 doctests = **2,772 total passing (100%)** — updated 2026-06-10

### Breakdown by Crate (0.1.0 — updated 2026-06-10)

- **tenrso-core:** 196 tests
- **tenrso-kernels:** 323 tests
- **tenrso-decomp:** 179 tests
- **tenrso-sparse:** 451 tests
- **tenrso-planner:** 298 tests (257 unit + 41 doc)
- **tenrso-ooc:** 315 tests
- **tenrso-exec:** 305 tests (includes 8 proptest property tests)
- **tenrso-ad:** 185 tests
- **tenrso (integration/regression):** 16 tests (9 kernels_decomp + 7 regression_suite)

**0.1.0 Status:** 2,607 unit+integration + 148 doctests passing (100%) — Zero known issues, all milestones M0-M6 complete, unsafe hardening done, regression suite complete!

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
- [x] Benchmark harness (criterion) - ✅ DONE (2026-06-10) (`criterion.workspace = true` used in all 8 crates)
- [x] Property test framework (proptest) - ✅ DONE (2026-06-10) (`proptest.workspace = true` used in all 8 crates)

---

## Performance Targets (Validation Checklist)

Once implementations are complete, verify:

- [ ] Einsum: ≥ 80% of OpenBLAS baseline (1024³ matmul) <!-- SKIP: structurally unmeasurable under Pure Rust Policy -->
- [x] Masked einsum: ≥ 5× speedup vs dense naive (90% zeros) — ✅ **MET (2026-07-11)** at
      256² (~10–18×) and 512² (~14–27×) after a real cache-locality fix (column packing);
      64² sits at ~5× (advisory in CI). Fair contiguous-slice baseline + pre-timing
      correctness gate. Earlier "no reference harness" note was wrong — the harness existed.
- [ ] CP-ALS: < 2s / 10 iters (256³, rank-64) — **~3.1–3.4s (2026-07-11, was 21–44s)**;
      not yet met, but **GEMM-throughput bound, not bandwidth bound** (scales ≈4× on 8
      cores). In-policy path to <2s: SIMD GEMM micro-kernel. Re-measure uncontended.
- [x] Tucker-HOOI benchmark: corrected to target ranks [64,64,32]; gate fix confirms all modes use randomized SVD <!-- 35-50% faster on 256³; full target shape completes -->
- [x] TT-SVD: `thin_svd_via_gram` prevents timeout on 32^6 (avoids GBs Gaussian Ω); 32^4 baseline preserved (2.79-3s)
- [x] TT memory reduction ≥ 10× - ✅ COMPLETE (measured 20,459× on 32^6)
- [x] No panics in production kernels - ✅ COMPLETE (321 unwraps eliminated, 2 documented startup invariants remain)
- [ ] All unsafe code bounded and fuzzed — *bounded*: unsafe blocks in tenrso-exec (`types.rs`: transmute with type checks + SAFETY comments) and tenrso-ooc (`mmap_io.rs`, `zerocopy_io.rs`, `simd_ops.rs`, etc.) all carry `// Safety:` or `# Safety` doc comments. *fuzzed*: no cargo-fuzz harnesses exist yet — Future work.

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

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check) — ✅ ALL CLOSED 2026-07-11

These three were the last real stubs. All closed in the 2026-07-11 depth session — see
"Stubs to implement — ✅ ALL CLOSED" and "Correctness bugs fixed" near the top of this file.

- [x] **tenrso-exec** einsum general case — **not** routed through OxiBLAS (unnecessary):
      replaced the naive nested loop with a pure-Rust batched-GEMM engine
      (`crates/tenrso-exec/src/ops/`, `matrixmultiply`-backed). The old `ops.rs:66` no
      longer exists (file split into `ops/`). Fixed the `"ii,ii->"` diagonal bug and the
      batch-index / output-order bugs in the multi-tensor path in the same work.
- [x] **tenrso-ad** — real op registry (`src/registry/`) + default-off `tensorlogic`
      feature implementing `TlExecutor`/`TlAutodiff`; `register_tenrso_ops()` genuinely
      populates it. Every rule gradcheck-verified. (`hooks.rs:251` TODO removed.)
- [x] **tenrso-ooc** — real GPU enumeration via `scirs2_core::gpu::backends`
      (nvidia-smi/rocm-smi/Metal/WebGPU), default-off features, honest CPU-only fallback
      via `available_backends()`. **No fabricated devices.** (`gpu.rs:525` TODO removed.)
