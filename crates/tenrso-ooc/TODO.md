# tenrso-ooc TODO

> **Milestone:** M5

---

## M5: Out-of-Core Implementation - ✅ COMPLETE

### I/O Backends

- [x] Arrow IPC reader/writer
- [x] Parquet chunked I/O
- [x] Memory-mapped tensor access
- [x] Chunked iterator

### Streaming Execution

- [x] Deterministic chunk graphs
- [x] Streaming contraction executor
- [x] Spill-to-disk policies (with automatic memory management)
- [x] Back-pressure handling (via MemoryManager)
- [x] Prefetch strategies

---

## Testing & Documentation

- [x] Unit tests for I/O operations (Arrow, Parquet, mmap)
- [x] Unit tests for chunking and streaming
- [x] Unit tests for memory management, prefetch, and chunk graphs
- [x] Large tensor benchmarks (> memory)
- [x] Examples and documentation (5 examples added)

---

**Last Updated:** 2025-11-06

---

## Notes (2025-11-06)

- **M5 Status:** COMPLETE - All components implemented and tested
- **New Modules:**
  - `chunk_graph`: Deterministic chunk execution plans with topological sorting
  - `contraction`: Streaming contraction executor for general einsum operations
  - `memory`: Advanced memory management with LRU/LFU/pattern-based spill policies
  - `prefetch`: Prefetch strategies (Sequential, Adaptive, Aggressive)
  - `profiling`: Performance monitoring with RAII-based scope profiling
- **Examples:** 6 comprehensive examples demonstrating OoC usage
- **Benchmarks:** Criterion-based benchmarks for large tensor operations
- **Total Code:** 4,400+ lines across 11 modules
- **Status:** M5 milestone complete + additional enhancements, ready for integration testing
