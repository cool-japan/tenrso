//! Distributed out-of-core processing for TenRSo.
//!
//! This module provides multi-node chunk coordination, remote caching,
//! network-aware prefetching, and fault tolerance.

pub mod fault_tolerance;
pub mod network;
pub mod network_prefetch;
pub mod protocol;
pub mod registry;
pub mod remote_cache;

pub use fault_tolerance::{
    FaultToleranceConfig, FaultToleranceManager, FaultToleranceStats, NodeHealth, ReplicationPolicy,
};
pub use network::{ChunkProvider, NetworkClient, NetworkConfig, NetworkServer, NetworkStats};
pub use network_prefetch::{
    NetworkPrefetchConfig, NetworkPrefetchStats, NetworkPrefetchStrategy, NetworkPrefetcher,
};
pub use protocol::{
    ChunkLocation, ChunkMetadata, ChunkRequest, ChunkResponse, HeartbeatRequest, HeartbeatResponse,
    MessageType, NodeId, NodeInfo,
};
pub use registry::{
    ChunkPlacement, ConsistentHashRing, DistributedRegistry, RegistryConfig, RegistryStats,
};
pub use remote_cache::{
    CachePolicy, RemoteCache, RemoteCacheConfig, RemoteCacheStats, WritePolicy,
};
