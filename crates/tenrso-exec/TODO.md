# tenrso-exec TODO

> **Milestone:** M4
> **Version:** 0.1.0-rc.1
> **Status:** RC.1 — 244 tests passing (100%) — 2026-03-06
> **Last Updated:** 2026-03-06

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
- [x] Benchmarks: `optimization_benchmarks.rs` (50+ individual benchmarks)
- [x] Benchmarks: memory pool benchmarks (7 benchmark groups)

**Total Tests:** 244 tests passing (100%)

---

## Module Structure

```
src/
├── lib.rs                      - Module exports
├── types.rs                    - Type definitions, enums, MemoryPool
├── functions.rs                - Trait definitions
├── cpuexecutor_traits.rs       - CpuExecutor trait implementations
├── functions_tests.rs          - Test suite (extracted for compliance)
├── parallel.rs                 - Parallel execution utilities
├── custom_ops.rs               - Custom user-defined operations
├── simd_ops.rs                 - SIMD-accelerated element-wise ops
├── tiled_reductions.rs         - Cache-friendly blocked reductions
├── advanced_indexing.rs        - Multi-dim gather/scatter/mask
├── vectorized_broadcast.rs     - Pattern-aware broadcasting
├── optimized_ops.rs            - Optimization integration layer
├── pooled_ops.rs               - RAII-style buffer management helpers
├── thread_local_pool.rs        - Thread-local memory pools
└── pool_heuristics.rs          - Smart pooling heuristics
```

---

## Future Enhancements (Post-RC.1)

- GPU executor via scirs2-gpu (CPU fallback operational)
- Distributed execution across nodes
- Mixed-precision support (FP16/BF16)
- Advanced sparsity-aware execution paths
- Tensorlogic integration (awaiting API stabilization)

---

**Milestone M4:** COMPLETE
**Last Updated:** 2026-03-06
