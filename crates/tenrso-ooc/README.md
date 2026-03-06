# tenrso-ooc

[![Crates.io](https://img.shields.io/crates/v/tenrso-ooc)](https://crates.io/crates/tenrso-ooc)
[![Documentation](https://docs.rs/tenrso-ooc/badge.svg)](https://docs.rs/tenrso-ooc)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Production-grade out-of-core tensor processing for the TenRSo ecosystem.**

Part of the [TenRSo](https://github.com/cool-japan/tenrso) tensor computing stack.

## Overview

`tenrso-ooc` enables tensor operations that exceed available RAM by leveraging Arrow IPC, Parquet, and memory-mapped I/O. It provides transparent data movement across memory tiers, advanced prefetching, ML-based eviction, and comprehensive observability.

## Features

### I/O Backends
- **Arrow IPC** - Efficient in-memory tensor serialization
- **Parquet** - Columnar storage for tensor chunks
- **Memory mapping** - Direct file-backed tensor access with zero-copy reads
- **Batch I/O** - Parallel batch reader/writer with error recovery

### Streaming Execution
- Deterministic chunk graphs
- Lazy loading and prefetching
- Streaming contraction executor
- Spill-to-disk with automatic memory management
- Back-pressure handling

### Compression
- **LZ4** - Fast compression for memory-bandwidth trade-off
- **Zstd** - High compression ratios with configurable levels (1-22)
- Transparent compression/decompression in the I/O layer
- Intelligent auto-selection based on data entropy and patterns

### Memory Management
- **Hierarchical memory tiers**: RAM -> SSD -> Disk with cost-aware migration
- **Working set prediction**: 5 modes (Frequency, Recency, Hybrid, Adaptive, Sequential)
- **ML-based eviction**: Online linear/logistic regression, ensemble models, SGD with L2 regularization
- **NUMA-aware allocation**: Multiple policies (Local, Interleaved, Balanced, Preferred, Bind)
- Lock-free prefetcher using crossbeam and DashMap

### Observability & Telemetry
- **Structured logging** via `tracing` (Pretty, JSON, Compact formats)
- **OpenTelemetry** integration with OTLP exporter and configurable sampling
- **Prometheus metrics** with HTTP `/metrics` endpoint (histograms, counters)
- **Resource dashboards**: Real-time monitoring, anomaly detection, performance recommendations

### Distributed Processing
- Multi-node chunk coordination with consistent hashing registry
- Remote chunk caching with ML-based eviction
- Network-aware prefetching with adaptive batching
- Fault tolerance and recovery via heartbeat-based failure detection

### GPU Integration
- Device-agnostic GPU abstraction layer
- CPU fallback implementation
- Unified DeviceBuffer with statistics
- Host-device memory transfers
- Multi-device support with DeviceManager
- Ready for future CUDA/ROCm/Vulkan/Metal backends

### Performance Optimizations
- **SIMD**: AVX2, AVX-512 (x86_64), ARM NEON — element-wise and reduction ops
- **Zero-copy I/O**: Memory-mapped views, vectored I/O (readv/writev), aligned buffer pools
- **Custom allocators**: jemalloc and mimalloc support with benchmarking framework
- **Lock-free data structures**: MPMC queue, concurrent hash map, atomic statistics

### Data Integrity
- Checksumming: CRC32, XXHash64, Blake3
- Chunk metadata with integrity information
- Configurable validation policies (Strict, Opportunistic, None)
- Corruption detection and reporting

## Installation

```toml
[dependencies]
tenrso-ooc = "0.1.0-rc.1"

# With all I/O backends
tenrso-ooc = { version = "0.1.0-rc.1", features = ["arrow", "parquet", "mmap"] }
```

## Quick Start

### Parquet I/O

```rust
use tenrso_ooc::ParquetWriter;

// Write large tensor to Parquet
let writer = ParquetWriter::new("tensor.parquet")?;
writer.write_chunked(&large_tensor, chunk_size=1000)?;

// Read chunks on demand
let reader = ParquetReader::open("tensor.parquet")?;
for chunk in reader.chunks() {
    process_chunk(chunk?);
}
```

### Memory-Mapped Tensors

```rust
use tenrso_ooc::MmapTensor;

// Memory-map large file
let tensor = MmapTensor::<f32>::open("data.bin", &shape)?;

// Access like regular tensor (lazy loading)
let value = tensor[[100, 200, 300]];
```

### Streaming Execution with Back-Pressure

```rust
use tenrso_ooc::{StreamingExecutor, ChunkGraph};

let mut executor = StreamingExecutor::new()
    .with_memory_limit(4 * 1024 * 1024 * 1024) // 4 GB
    .with_prefetch_depth(4);

let graph = ChunkGraph::from_einsum("ij,jk->ik", &tensor_a, &tensor_b)?;
let result = executor.run(&graph)?;
```

### ML-Based Eviction

```rust
use tenrso_ooc::ml_eviction::{MlEvictionPolicy, EnsembleConfig};

let policy = MlEvictionPolicy::ensemble(EnsembleConfig::default())?;
let manager = MemoryManager::new(budget_bytes).with_eviction_policy(policy);
```

### Working Set Prediction

```rust
use tenrso_ooc::memory_policy::{WorkingSetPredictor, PredictionMode};

let predictor = WorkingSetPredictor::new(PredictionMode::Adaptive);
let next_chunks = predictor.predict_next(&access_history, top_k)?;
```

## Feature Flags

- `arrow` - Arrow IPC support
- `parquet` - Parquet I/O
- `mmap` - Memory-mapped access
- `compression` - Enable compression support
- `lz4-compression` - LZ4 fast compression
- `zstd-compression` - Zstd high-ratio compression
- `tracing` - Structured logging via tracing
- `opentelemetry` - OpenTelemetry distributed tracing
- `prometheus-metrics` - Prometheus metrics export
- `lock-free` - Lock-free prefetcher data structures
- `jemalloc` - jemalloc custom allocator
- `mimalloc-allocator` - mimalloc custom allocator

## Examples

Comprehensive examples demonstrating all features:

- `cp_als_ooc.rs` - CP-ALS decomposition with out-of-core data
- `mttkrp_streaming.rs` - MTTKRP operations with streaming
- `tucker_ooc.rs` - Tucker decomposition on large tensors
- `tensor_network_ooc.rs` - Tensor network contraction workflows
- `ml_eviction_demo.rs` - ML-based eviction policy demonstration
- `numa_demo.rs` - NUMA-aware allocation showcase
- `gpu_demo.rs` - GPU integration demonstration
- `lockfree_prefetch_demo.rs` - Lock-free prefetcher benchmark
- `allocator_selection.rs` - Custom allocator comparison
- `data_integrity_demo.rs` - Checksumming and validation
- `compression_auto_demo.rs` - Automatic compression selection

```bash
cargo run --example cp_als_ooc
cargo run --example ml_eviction_demo
cargo run --example gpu_demo
```

## Benchmarks

```bash
# Run all benchmarks
cargo bench --package tenrso-ooc

# Specific benchmark groups
cargo bench --package tenrso-ooc -- ml_eviction
cargo bench --package tenrso-ooc -- integrity_and_autoselect
cargo bench --package tenrso-ooc -- allocator_comparison
```

## Testing

```bash
cargo test --package tenrso-ooc
```

**Test Coverage:** 238 tests (100% passing) across 30 modules

## Architecture

```
tenrso-ooc/
├── arrow_io.rs           - Arrow IPC reader/writer
├── parquet_io.rs         - Parquet chunked I/O
├── mmap.rs               - Memory-mapped tensor access
├── stream.rs             - Streaming contraction executor
├── chunk_graph.rs        - Deterministic chunk graph
├── memory.rs             - MemoryManager + spill-to-disk
├── prefetch.rs           - Prefetching strategies
├── compression.rs        - LZ4 + Zstd compression
├── compression_auto.rs   - Auto codec selection
├── memory_policy.rs      - Working set prediction (5 modes)
├── ml_eviction.rs        - ML-based eviction policies
├── numa.rs               - NUMA-aware allocation
├── batch_io.rs           - Batch reader/writer
├── simd_ops.rs           - AVX2/AVX-512/NEON SIMD ops
├── zero_copy.rs          - Zero-copy I/O
├── custom_allocator.rs   - jemalloc/mimalloc support
├── lockfree_prefetch.rs  - Lock-free prefetcher
├── telemetry.rs          - Structured logging (tracing)
├── opentelemetry.rs      - OpenTelemetry integration
├── prometheus.rs         - Prometheus metrics
├── dashboard.rs          - Resource usage dashboards
├── distributed.rs        - Multi-node coordination
├── gpu.rs                - GPU abstraction layer
├── data_integrity.rs     - Checksumming and validation
└── lib.rs                - Module exports
```

## Performance Targets

- **Streaming throughput**: Near-memory bandwidth for sequential access
- **Prefetch hit rate**: >80% for sequential/periodic patterns
- **ML eviction**: <5ms prediction latency per eviction decision
- **Compression**: LZ4 >500 MB/s encode, Zstd >100 MB/s (level 3)
- **SIMD speedup**: 2-8x over scalar for element-wise ops

## License

Apache-2.0

## Related Projects

- [`tenrso-core`](../tenrso-core) - Core tensor data structures
- [`tenrso-exec`](../tenrso-exec) - Execution engine
- [`tenrso-planner`](../tenrso-planner) - Contraction planning
- [`tenrso`](../tenrso) - Main TenRSo library

---

**Status:** Alpha (production-ready internals) | **Version:** 0.1.0-rc.1 | **Tests:** 238/238 passing
