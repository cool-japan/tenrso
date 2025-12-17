//! Integration tests for distributed out-of-core processing.

#![cfg(feature = "distributed")]

use anyhow::Result;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tenrso_ooc::distributed::*;

// Mock chunk provider for testing
#[allow(dead_code)]
struct TestChunkProvider {
    chunks: HashMap<String, Vec<u8>>,
}

#[allow(dead_code)]
impl TestChunkProvider {
    fn new() -> Self {
        let mut chunks = HashMap::new();
        // Add some test chunks
        for i in 0..10 {
            let chunk_id = format!("chunk_{}", i);
            let data = vec![i as u8; 1024]; // 1KB per chunk
            chunks.insert(chunk_id, data);
        }
        Self { chunks }
    }
}

impl tenrso_ooc::distributed::network::ChunkProvider for TestChunkProvider {
    fn load_chunk(&self, chunk_id: &str) -> Result<Vec<u8>> {
        self.chunks
            .get(chunk_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Chunk not found: {}", chunk_id))
    }
}

#[test]
fn test_protocol_serialization() {
    let bincode_config = bincode::config::standard();

    // Test ChunkRequest serialization
    let request = ChunkRequest::new("test_chunk".to_string(), NodeId(42));
    let serialized = bincode::serde::encode_to_vec(&request, bincode_config).unwrap();
    let (deserialized, _): (ChunkRequest, usize) =
        bincode::serde::decode_from_slice(&serialized, bincode_config).unwrap();

    assert_eq!(request.chunk_id, deserialized.chunk_id);
    assert_eq!(request.requester, deserialized.requester);

    // Test ChunkMetadata serialization
    let metadata = ChunkMetadata::new_f64("chunk_1".to_string(), vec![10, 20, 30]);
    let serialized = bincode::serde::encode_to_vec(&metadata, bincode_config).unwrap();
    let (deserialized, _): (ChunkMetadata, usize) =
        bincode::serde::decode_from_slice(&serialized, bincode_config).unwrap();

    assert_eq!(metadata.chunk_id, deserialized.chunk_id);
    assert_eq!(metadata.shape, deserialized.shape);
}

#[test]
fn test_consistent_hashing() {
    let config = RegistryConfig {
        virtual_nodes: 150,
        replication_factor: 3,
    };

    let mut ring = ConsistentHashRing::new(config);

    // Add nodes
    for i in 0..5 {
        let addr: SocketAddr = format!("127.0.0.1:800{}", i).parse().unwrap();
        let node = NodeInfo::new(NodeId(i), addr);
        ring.add_node(node);
    }

    assert_eq!(ring.num_nodes(), 5);

    // Test chunk placement
    let chunk_id = "test_chunk";
    let primary = ring.get_node(chunk_id);
    assert!(primary.is_some());

    // Test replica placement
    let replicas = ring.get_replica_nodes(chunk_id);
    assert_eq!(replicas.len(), 3);

    // All replicas should be on different physical nodes
    let unique: std::collections::HashSet<_> = replicas.iter().collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn test_distributed_registry() {
    let config = RegistryConfig::default();
    let registry = DistributedRegistry::new(config);

    // Register nodes
    for i in 0..3 {
        let addr: SocketAddr = format!("127.0.0.1:900{}", i).parse().unwrap();
        let node = NodeInfo::new(NodeId(i), addr);
        registry.register_node(node);
    }

    let stats = registry.stats();
    assert_eq!(stats.total_nodes, 3);

    // Register a chunk
    let metadata = ChunkMetadata::new_f64("chunk_1".to_string(), vec![100, 100]);
    let locations = vec![ChunkLocation {
        node_id: NodeId(0),
        path: "/tmp/chunk_1".to_string(),
        in_memory: false,
    }];

    registry.register_chunk(metadata, locations);

    let stats = registry.stats();
    assert_eq!(stats.total_chunks, 1);
    assert_eq!(stats.registrations, 1);

    // Lookup chunk
    let placement = registry.get_chunk_placement("chunk_1");
    assert!(placement.is_some());

    let placement = placement.unwrap();
    assert_eq!(placement.metadata.chunk_id, "chunk_1");
    assert_eq!(placement.locations.len(), 1);
}

#[test]
fn test_node_removal_from_registry() {
    let config = RegistryConfig::default();
    let registry = DistributedRegistry::new(config);

    let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let node = NodeInfo::new(NodeId(1), addr);
    registry.register_node(node);

    assert_eq!(registry.stats().total_nodes, 1);

    // Unregister node
    registry.unregister_node(NodeId(1)).unwrap();
    assert_eq!(registry.stats().total_nodes, 0);
}

#[test]
fn test_chunk_eviction() {
    let config = RegistryConfig::default();
    let registry = DistributedRegistry::new(config);

    let metadata = ChunkMetadata::new_f64("evict_me".to_string(), vec![50, 50]);
    let locations = vec![ChunkLocation {
        node_id: NodeId(1),
        path: "/tmp/evict_me".to_string(),
        in_memory: true,
    }];

    registry.register_chunk(metadata, locations);
    assert_eq!(registry.stats().total_chunks, 1);

    let evicted = registry.evict_chunk("evict_me");
    assert!(evicted.is_some());
    assert_eq!(registry.stats().total_chunks, 0);
    assert_eq!(registry.stats().evictions, 1);
}

#[tokio::test]
async fn test_network_client_creation() {
    let config = NetworkConfig::default();
    let client = NetworkClient::new(config);

    let stats = client.stats();
    assert_eq!(stats.bytes_sent, 0);
    assert_eq!(stats.messages_sent, 0);
}

#[tokio::test]
async fn test_remote_cache_creation() {
    let registry_config = RegistryConfig::default();
    let registry = Arc::new(DistributedRegistry::new(registry_config));

    let network_config = NetworkConfig::default();
    let network_client = Arc::new(NetworkClient::new(network_config));

    let cache_config = RemoteCacheConfig::default();
    let cache = RemoteCache::new(cache_config, registry, network_client);

    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.num_chunks, 0);
}

#[tokio::test]
async fn test_remote_cache_clear() {
    let registry_config = RegistryConfig::default();
    let registry = Arc::new(DistributedRegistry::new(registry_config));

    let network_config = NetworkConfig::default();
    let network_client = Arc::new(NetworkClient::new(network_config));

    let cache_config = RemoteCacheConfig {
        max_cache_size: 1024 * 1024, // 1MB
        ..Default::default()
    };

    let cache = RemoteCache::new(cache_config, registry, network_client);
    cache.clear().await;

    let stats = cache.stats();
    assert_eq!(stats.current_size, 0);
}

#[tokio::test]
async fn test_network_prefetcher_creation() {
    let registry = Arc::new(DistributedRegistry::new(RegistryConfig::default()));
    let network_client = Arc::new(NetworkClient::new(NetworkConfig::default()));
    let cache = Arc::new(RemoteCache::new(
        RemoteCacheConfig::default(),
        Arc::clone(&registry),
        Arc::clone(&network_client),
    ));

    let config = NetworkPrefetchConfig::default();
    let prefetcher = NetworkPrefetcher::new(config, registry, cache, network_client);

    let stats = prefetcher.stats();
    assert_eq!(stats.total_prefetched, 0);
    assert_eq!(stats.batches_sent, 0);
}

#[tokio::test]
async fn test_fault_tolerance_manager() {
    let registry = Arc::new(DistributedRegistry::new(RegistryConfig::default()));
    let network_client = Arc::new(NetworkClient::new(NetworkConfig::default()));

    let config = FaultToleranceConfig::default();
    let manager = FaultToleranceManager::new(config, registry, network_client);

    let stats = manager.stats();
    assert_eq!(stats.heartbeats_sent, 0);
    assert_eq!(stats.node_failures, 0);
}

#[test]
fn test_replication_policies() {
    let none = ReplicationPolicy::None;
    let fixed = ReplicationPolicy::Fixed(3);
    let dynamic = ReplicationPolicy::Dynamic { min: 2, max: 5 };

    assert_ne!(none, fixed);
    assert_ne!(fixed, dynamic);
}

#[test]
fn test_cache_policies() {
    assert_eq!(CachePolicy::LRU, CachePolicy::LRU);
    assert_ne!(CachePolicy::LRU, CachePolicy::LFU);
    assert_ne!(CachePolicy::LFU, CachePolicy::ML);
}

#[test]
fn test_write_policies() {
    assert_eq!(WritePolicy::WriteThrough, WritePolicy::WriteThrough);
    assert_ne!(WritePolicy::WriteThrough, WritePolicy::WriteBack);
    assert_ne!(WritePolicy::WriteBack, WritePolicy::NoWrite);
}

#[test]
fn test_node_health_states() {
    assert_eq!(NodeHealth::Healthy, NodeHealth::Healthy);
    assert_ne!(NodeHealth::Healthy, NodeHealth::Suspected);
    assert_ne!(NodeHealth::Suspected, NodeHealth::Failed);
}

#[test]
fn test_prefetch_strategies() {
    assert_eq!(
        NetworkPrefetchStrategy::Sequential,
        NetworkPrefetchStrategy::Sequential
    );
    assert_ne!(
        NetworkPrefetchStrategy::Sequential,
        NetworkPrefetchStrategy::Adaptive
    );
}

#[test]
fn test_chunk_metadata_creation() {
    let metadata = ChunkMetadata::new_f64("test".to_string(), vec![10, 20, 30]);

    assert_eq!(metadata.chunk_id, "test");
    assert_eq!(metadata.shape, vec![10, 20, 30]);
    assert_eq!(metadata.dtype, "f64");
    assert_eq!(metadata.size_bytes, 10 * 20 * 30 * 8); // f64 = 8 bytes
}

#[test]
fn test_chunk_metadata_checksum() {
    let mut metadata = ChunkMetadata::new_f64("test".to_string(), vec![5, 5]);
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    metadata.compute_checksum(&data);
    assert!(metadata.verify_checksum(&data));

    let wrong_data = vec![10u8, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    assert!(!metadata.verify_checksum(&wrong_data));
}

#[test]
fn test_node_id_creation() {
    let addr1: SocketAddr = "127.0.0.1:8000".parse().unwrap();
    let addr2: SocketAddr = "127.0.0.1:8001".parse().unwrap();

    let id1 = NodeId::from_addr(&addr1);
    let id2 = NodeId::from_addr(&addr2);

    assert_ne!(id1, id2);
    assert_eq!(id1, id1);
}
