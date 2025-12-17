//! # tenrso-ooc
//!
//! Out-of-core tensor processing for TenRSo.
//!
//! **Version:** 0.1.0-alpha.2
//! **Tests:** 238 passing (100%)
//! **Status:** M5 Complete - Production-ready out-of-core processing
//!
//! This crate provides:
//! - Arrow/Parquet chunk readers and writers
//! - Memory-mapped tensor access
//! - Deterministic chunk graphs for streaming
//! - Spill-to-disk policies

#![deny(warnings)]

#[cfg(feature = "arrow")]
pub mod arrow_io;

#[cfg(feature = "parquet")]
pub mod parquet_io;

#[cfg(feature = "mmap")]
pub mod mmap_io;

#[cfg(feature = "compression")]
pub mod compression;

#[cfg(feature = "compression")]
pub mod compression_auto;

#[cfg(feature = "opentelemetry")]
pub mod opentelemetry_support;

#[cfg(feature = "prometheus-metrics")]
pub mod prometheus_metrics;

pub mod allocators;
pub mod batch_io;
pub mod chunk_graph;
pub mod chunking;
pub mod contraction;
pub mod dashboard;
pub mod data_integrity;
pub mod gpu;
pub mod memory;
pub mod memory_tiers;
pub mod ml_eviction;
pub mod numa;
pub mod prefetch;
pub mod profiling;
pub mod simd_ops;
pub mod streaming;
pub mod tracing_support;
pub mod working_set;
pub mod zerocopy_io;

#[cfg(feature = "lock-free")]
pub mod lockfree_prefetch;

#[cfg(feature = "distributed")]
pub mod distributed;

// Re-exports
#[cfg(feature = "arrow")]
pub use arrow_io::{ArrowReader, ArrowWriter};

#[cfg(feature = "parquet")]
pub use parquet_io::{ParquetReader, ParquetWriter};

#[cfg(feature = "mmap")]
pub use mmap_io::{read_tensor_binary, write_tensor_binary, MmapTensor, MmapTensorMut};

#[cfg(feature = "compression")]
pub use compression::{compress_bytes, decompress_bytes, CompressionCodec, CompressionStats};

#[cfg(feature = "compression")]
pub use compression_auto::{
    AutoSelectConfig, CompressionAutoSelector, DataCharacteristics, DataPattern, SelectionPolicy,
    SelectionStats,
};

#[cfg(feature = "opentelemetry")]
pub use opentelemetry_support::{
    init_otel_tracer, record_compression_op, record_io_op, record_memory_op, record_tensor_op,
    shutdown_otel, OtelConfig, OtelStats, OtelStatsCollector, SamplerConfig, TensorOpSpan,
};

#[cfg(feature = "prometheus-metrics")]
pub use prometheus_metrics::{
    init_prometheus, MetricsBatch, PrometheusConfig, PrometheusMetrics, METRICS,
};

pub use allocators::{
    get_allocator_stats, AllocatorBenchmark, AllocatorInfo, AllocatorPool, AllocatorStats,
    AllocatorType,
};
pub use batch_io::{BatchConfig, BatchReadResult, BatchReader, BatchWriteResult, BatchWriter};
pub use chunk_graph::{ChunkGraph, ChunkNode, ChunkOp};
pub use chunking::{ChunkIndex, ChunkIterator, ChunkSpec};
pub use contraction::{ContractionConfig, StreamingContractionExecutor};
pub use dashboard::{
    Anomaly, Dashboard, DashboardConfig, DashboardSnapshot, IoSnapshot, MemorySnapshot,
    OperationStats, Recommendation,
};
pub use data_integrity::{
    compute_checksum, ChecksumAlgorithm, ChunkIntegrityMetadata, IntegrityChecker, IntegrityStats,
    ValidationPolicy,
};
pub use gpu::{
    Device, DeviceBuffer, DeviceBufferStatsSnapshot, DeviceInfo, DeviceManager,
    DeviceStatsSnapshot, DeviceType,
};
pub use memory::{AccessPattern, MemoryManager, MemoryStats, SpillPolicy};
pub use memory_tiers::{
    ChunkTierStats, MemoryTier, OverallStatistics, TierAccessPattern, TierConfig, TierStatistics,
    TieredMemoryManager,
};
pub use ml_eviction::{EvictionScore, MLConfig, MLEvictionPolicy, MLPolicyStats};
pub use numa::{NumaAllocator, NumaNode, NumaPolicy, NumaStats, NumaTopology};
pub use prefetch::{PrefetchStats, PrefetchStrategy, Prefetcher};
pub use profiling::{OperationStats as ProfilingOpStats, ProfileScope, ProfileSummary, Profiler};
pub use simd_ops::{
    simd_add_f64, simd_fma_f64, simd_max_f64, simd_min_f64, simd_mul_f64, simd_sum_f64,
    SimdBenchmark, SimdCapabilities,
};
pub use streaming::{StreamConfig, StreamingExecutor};
pub use tracing_support::{
    init_tracing, record_bytes, record_io, record_memory_op as tracing_record_memory_op,
    record_metric, TracingConfig, TracingFormat,
};
pub use working_set::{
    ChunkStatistics, PredictionMode, PredictorStatistics, WorkingSetPrediction, WorkingSetPredictor,
};
pub use zerocopy_io::{
    AlignedBuffer, BufferPool, ZeroCopyReader, ZeroCopyStats, ZeroCopyWriter, DEFAULT_ALIGNMENT,
};

#[cfg(feature = "lock-free")]
pub use lockfree_prefetch::{
    LockFreePrefetcher, LockFreePrefetcherStats, PrefetchEntry, PrefetchStatsSnapshot,
    PrefetchStrategy as LockFreePrefetchStrategy,
};

#[cfg(feature = "distributed")]
pub use distributed::{
    CachePolicy, ChunkLocation, ChunkMetadata, ChunkPlacement, ChunkProvider, ChunkRequest,
    ChunkResponse, ConsistentHashRing, DistributedRegistry, FaultToleranceConfig,
    FaultToleranceManager, FaultToleranceStats, HeartbeatRequest, HeartbeatResponse, MessageType,
    NetworkClient, NetworkConfig, NetworkPrefetchConfig, NetworkPrefetchStats,
    NetworkPrefetchStrategy, NetworkPrefetcher, NetworkServer, NetworkStats, NodeHealth, NodeId,
    NodeInfo, RegistryConfig, RegistryStats, RemoteCache, RemoteCacheConfig, RemoteCacheStats,
    ReplicationPolicy, WritePolicy,
};
