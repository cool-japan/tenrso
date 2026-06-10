# tenrso-exec TODO

> **Milestone:** M4
> **Version:** 0.1.0
> **Status:** 0.1.0 — 273 tests passing (100%) — 2026-04-15
> **Last Updated:** 2026-04-15

---

## M4: Unified Execution API - COMPLETE

### Core Execution

- [x] **einsum_ex() unified interface** - COMPLETE
  - [x] Builder pattern with fluent API
  - [x] Input tensor list, execution hints
  - [x] Dense/sparse/low-rank dispatch
  - [x] Planner integration for contraction ordering

- [x] **TenrsoExecutor trait** - COMPLETE
  - [x] einsum, elem_op, reduce, binary_op, matmul
  - [x] conv1d, conv2d, conv3d
  - [x] max_pool_2d, avg_pool_2d
  - [x] gather, scatter, fancy_index_mask
  - [x] advanced_gather, advanced_scatter
  - [x] tile, pad, flip, concatenate
  - [x] squeeze, unsqueeze, expand_dims, stack, repeat, roll
  - [x] argmax, argmin
  - [x] CpuExecutor implementation

- [x] **ExecHints** - COMPLETE
  - [x] prefer_sparse, prefer_lowrank, tile_kb
  - [x] mask (MaskPack), subset (SubsetSpec)

### Performance Optimization

- [x] **SIMD Operations** - COMPLETE
  - [x] Vectorized element-wise operations (neg, abs, exp, log, sin, cos, tanh, sigmoid, relu, gelu)
  - [x] Vectorized binary operations (add, sub, mul, div, pow, max, min, fma)
  - [x] Automatic threshold-based selection (>= 1024 elements)
  - [x] Infrastructure for AVX2/AVX-512 SIMD
  - [x] Typical speedup: 2-4x simple ops, up to 8x expensive ops

- [x] **Tiled Reductions** - COMPLETE
  - [x] Cache-friendly blocked reductions using 4KB tiles (L1 optimized)
  - [x] Tiled sum, mean, max, min reductions
  - [x] Blocked matrix-vector multiplication
  - [x] Threshold-based activation (>= 100K elements)
  - [x] Typical speedup: 1.5-3x for large tensors

- [x] **Vectorized Broadcasting** - COMPLETE
  - [x] Pattern detection (SameShape, Scalar, LastDim, FirstDim, General)
  - [x] Specialized kernels per broadcast pattern
  - [x] SIMD-friendly aligned operations
  - [x] Typical speedup: 1.5-2x for broadcast-heavy workloads

- [x] **Configuration API** - COMPLETE
  - [x] `with_simd()`, `with_tiled_reductions()`, `with_vectorized_broadcast()`
  - [x] `unoptimized()` constructor for debugging
  - [x] Chainable builder pattern

### Memory Pool (Phases 1-5.1)

- [x] **Phase 1**: Statistics & monitoring
  - [x] PoolStats struct with 8 detailed metrics
  - [x] Enable/disable functionality
  - [x] Per-shape signature tracking
- [x] **Phase 2**: Type-safe buffer pooling
  - [x] Generic `MemoryPool<T>` with bytemuck integration
  - [x] Dual-pool CpuExecutor (f32 + f64)
  - [x] Public acquire/release API
- [x] **Phase 3**: Integration patterns
  - [x] RAII-style buffer management helpers
  - [x] `pooled_ops.rs` with example pooled operations
  - [x] Three documented patterns (RAII, manual, temporary)
- [x] **Phase 4**: Advanced features
  - [x] Thread-local memory pools (zero-contention parallel execution)
  - [x] Smart pooling heuristics (Default, Conservative, Aggressive, Memory-Constrained)
  - [x] Access pattern tracking and automatic recommendations
- [x] **Phase 5**: Automatic pooling in operations
  - [x] `acquire_pooled_generic<T>()` / `release_pooled_generic<T>()` helpers
  - [x] Binary ops with broadcasting automatically pooled
  - [x] Conv1d/2d/3d automatically pooled
- [x] **Phase 5.1**: Extended pooling coverage
  - [x] Concatenate, MaxPool2d, AvgPool2d, Tile, Pad, Flip automatically pooled
  - [x] Total: 10 operations using automatic pooling

### Advanced Indexing

- [x] Multi-dimensional gather with negative index support
- [x] Advanced scatter with accumulation modes (Replace, Add, Max, Min)
- [x] Fancy (boolean mask) indexing
- [x] ScatterMode enum exported for public use

### Shape Manipulation

- [x] squeeze, unsqueeze / expand_dims
- [x] stack (join tensors along a new axis)
- [x] repeat (repeat elements along an axis)
- [x] roll (circular shift elements along an axis)
- [x] Extended ReduceOp: Prod, All, Any, ArgMax, ArgMin

### Convolutions & Pooling

- [x] Conv1d with configurable stride and padding
- [x] Conv2d with configurable stride and padding
- [x] Conv3d with configurable stride and padding
- [x] MaxPool2d
- [x] AvgPool2d

---

## Testing & Documentation

- [x] Unit tests for core executor operations
- [x] Unit tests for SIMD operations
- [x] Unit tests for tiled reductions
- [x] Unit tests for vectorized broadcasting
- [x] Unit tests for advanced indexing
- [x] Unit tests for shape manipulation
- [x] Unit tests for convolutions
- [x] Unit tests for memory pool (phases 1-5.1)
- [x] Integration tests for optimization dispatch
- [x] Property tests (mathematical correctness) - ✅ DONE (2026-06-10) (`crates/tenrso-exec/tests/property_tests.rs`)
- [x] Benchmarks: `optimization_benchmarks.rs` (50+ individual benchmarks)
- [x] Benchmarks: memory pool benchmarks (7 benchmark groups)

**Total Tests:** 273 tests passing (100%)

---

## Module Structure

```
src/
├── lib.rs                        - Module exports
├── executor/
│   ├── mod.rs                    - Executor sub-tree declarations
│   ├── types.rs                  - Type definitions, enums, MemoryPool
│   ├── functions.rs              - TenrsoExecutor trait definitions
│   ├── cpuexecutor_traits/       - CpuExecutor trait impls (split by category)
│   │   ├── mod.rs                - Thin trait dispatch (327 lines)
│   │   ├── contraction.rs        - einsum (37 lines)
│   │   ├── conv_pool.rs          - conv/pool ops (641 lines)
│   │   ├── elementwise.rs        - elem_op, binary_op, clip, modulo (201)
│   │   ├── indexing.rs           - where/masked_select/gather/scatter (312)
│   │   ├── linalg.rs             - determinant, inverse, solve (176)
│   │   ├── reduction.rs          - reduce/softmax/layer_norm/argmax (437)
│   │   └── shape.rs              - transpose/reshape/concat/split/... (644)
│   ├── functions_tests/          - Tests for TenrsoExecutor impls (split)
│   │   ├── mod.rs                - Test-module declarations (29 lines)
│   │   ├── einsum_tests.rs       - 4 tests (84 lines)
│   │   ├── elementwise_tests.rs  - 33 tests (522 lines)
│   │   ├── reduction_tests.rs    - 21 tests (317 lines)
│   │   ├── shape_tests.rs        - 37 tests (536 lines)
│   │   ├── indexing_tests.rs     - 15 tests (271 lines)
│   │   ├── conv_pool_tests.rs    - 20 tests (355 lines)
│   │   ├── linalg_tests.rs       - 15 tests (202 lines)
│   │   └── pool_tests.rs         - 17 tests (379 lines)
│   └── ...
├── parallel.rs                   - Parallel execution utilities
├── custom_ops.rs                 - Custom user-defined operations
├── simd_ops.rs                   - SIMD-accelerated element-wise ops
├── tiled_reductions.rs           - Cache-friendly blocked reductions
├── advanced_indexing.rs          - Multi-dim gather/scatter/mask
├── vectorized_broadcast.rs       - Pattern-aware broadcasting
├── optimized_ops.rs              - Optimization integration layer
├── pooled_ops.rs                 - RAII-style buffer management helpers
├── thread_local_pool.rs          - Thread-local memory pools
└── pool_heuristics.rs            - Smart pooling heuristics
```

---

## Refactoring

### 2026-04-15 - Large file splits (compliance with <2000 lines policy)

- [x] **`executor/cpuexecutor_traits.rs`** (2221 lines) split into a directory
  module with 8 files (`mod.rs` + 7 per-category submodules). `mod.rs`
  now contains only thin trait dispatches; all helper logic lives in
  submodules named by operation family (contraction, conv_pool,
  elementwise, indexing, linalg, reduction, shape). Largest child
  is `shape.rs` at 644 lines — all children comfortably under 1500.
- [x] **`executor/functions_tests.rs`** (2503 lines) converted to a
  directory module with `mod.rs` + 8 per-category test submodules
  (einsum, elementwise, reduction, shape, indexing, conv_pool, linalg,
  pool). The parent `executor/mod.rs` already gates the subtree with
  `#[cfg(test)]`, so no additional attributes were needed. Largest
  child is `shape_tests.rs` at 536 lines.
- Line-count verification: every new file is well under the 1500-line
  ceiling; no file-level warnings; clippy `-D warnings` clean.
- Test integrity: 273 lib tests pass — identical count to pre-split.

---

## Future Enhancements (Post-RC.1)

- GPU executor via scirs2-gpu (CPU fallback operational)
- Distributed execution across nodes
- Mixed-precision support (FP16/BF16)
- Advanced sparsity-aware execution paths
- Tensorlogic integration (awaiting API stabilization)

---

**Milestone M4:** COMPLETE
**Last Updated:** 2026-04-15
