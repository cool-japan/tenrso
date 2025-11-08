//! # tenrso-ooc
//!
//! Out-of-core tensor processing for TenRSo.
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

pub mod chunk_graph;
pub mod chunking;
pub mod contraction;
pub mod memory;
pub mod prefetch;
pub mod profiling;
pub mod streaming;

// Re-exports
#[cfg(feature = "arrow")]
pub use arrow_io::{ArrowReader, ArrowWriter};

#[cfg(feature = "parquet")]
pub use parquet_io::{ParquetReader, ParquetWriter};

#[cfg(feature = "mmap")]
pub use mmap_io::{read_tensor_binary, write_tensor_binary, MmapTensor, MmapTensorMut};

pub use chunk_graph::{ChunkGraph, ChunkNode, ChunkOp};
pub use chunking::{ChunkIndex, ChunkIterator, ChunkSpec};
pub use contraction::{ContractionConfig, StreamingContractionExecutor};
pub use memory::{AccessPattern, MemoryManager, MemoryStats, SpillPolicy};
pub use prefetch::{PrefetchStats, PrefetchStrategy, Prefetcher};
pub use profiling::{OperationStats, ProfileScope, ProfileSummary, Profiler};
pub use streaming::{StreamConfig, StreamingExecutor};
