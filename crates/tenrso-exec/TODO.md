# tenrso-exec TODO

> **Milestone:** M4

---

## M4: Execution API - ✅ COMPLETE

### Core API

- [x] `einsum_ex` builder implementation ✅
- [x] Input validation and parsing ✅
- [x] Planner integration ✅
- [x] Backend dispatch (dense/sparse/lowrank) - Dense complete ✅

### CpuExecutor

- [x] Basic einsum execution ✅
- [x] Element-wise operations (neg, abs, exp, log) ✅
- [x] Reduction operations (sum, max, min, mean) ✅
- [x] Memory pooling infrastructure ✅
- [x] Thread pool management (basic configuration) ✅

### Integration

- [x] Connect to tenrso-planner ✅
- [x] Route to appropriate kernels ✅
- [ ] Handle representation mixing (sparse/lowrank) - Future work
- [x] Error propagation ✅

---

## Testing & Documentation

- [x] Unit tests for API (33 tests passing) ✅
- [x] Integration tests across crates ✅
- [x] Examples - basic_operations.rs & advanced_operations.rs ✅
- [x] Benchmarks - executor_ops.rs ✅

---

**Last Updated:** 2025-11-06

---

## Completed Features (2025-11-06)

### Element-wise Operations (9 operations)
- **Neg**: Negation operator (-x)
- **Abs**: Absolute value (|x|)
- **Exp**: Exponential function (e^x)
- **Log**: Natural logarithm (ln(x))
- **Sin**: Sine function (sin(x))
- **Cos**: Cosine function (cos(x))
- **Sqrt**: Square root (√x)
- **Sqr**: Square (x²)
- **Recip**: Reciprocal (1/x)
- All operations support dense tensors with proper Float trait bounds

### Reduction Operations
- **Sum**: Reduction along specified axes
- **Max**: Maximum value along axes
- **Min**: Minimum value along axes
- **Mean**: Average value along axes
- Multi-axis reduction support
- Proper error handling for invalid axes

### Memory Management
- **MemoryPool**: Buffer reuse infrastructure with shape-based caching
- **Statistics tracking**: Hit rate monitoring (hits, misses, hit_rate)
- **Configurable pooling**: MAX_POOL_SIZE limit to prevent unbounded growth
- **Thread configuration**: CpuExecutor::with_threads() for custom thread counts

### Test Coverage
- **33 unit tests** covering all features
- Element-wise operations: 9 tests (Neg, Abs, Exp, Log, Sin, Cos, Sqrt, Sqr, Recip)
- Reduction operations: 8 tests (Sum, Max, Min, Mean with multiple axes)
- Memory pooling: 5 tests (acquire/release, stats, shape signatures)
- Integration tests: 11 tests (einsum, multi-tensor, builder API)

### Examples & Benchmarks
- **basic_operations.rs**: Demonstrates all core functionality
- **advanced_operations.rs**: Shows complex pipelines (softmax, attention, statistics)
- **executor_ops.rs**: Comprehensive benchmark suite (matmul, element-wise, reductions)

---

## Future Enhancements

### Sparse/LowRank Support
- [ ] Sparse tensor backend dispatch
- [ ] LowRank tensor backend dispatch
- [ ] Automatic representation selection based on sparsity
- [ ] Mixed representation contractions

### Performance Optimization
- [ ] Actually use memory pool in tensor allocations
- [ ] Parallel execution using thread pool
- [ ] SIMD-optimized element-wise operations
- [ ] Blocked/tiled reductions for large tensors

### Additional Operations
- [x] More element-wise ops (sin, cos, sqrt, sqr, recip) ✅
- [ ] Broadcasting support for element-wise operations
- [ ] Custom reduction operations
- [ ] Fused operations (reduce+elemwise)
- [ ] Division and modulo operations
- [ ] Clipping and thresholding operations

---

## Notes

- **M1 Status:** tenrso-kernels complete with comprehensive operations
- **M4 Status:** Core execution API complete with 28 tests passing
- **Available Kernels:** All core operations (MTTKRP, n-mode, outer, Tucker) ready
- **Integration:** Planner + Executor working end-to-end for dense tensors
