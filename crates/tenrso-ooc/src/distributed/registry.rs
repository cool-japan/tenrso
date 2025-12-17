//! Distributed chunk registry with consistent hashing.
//!
//! Manages chunk locations across multiple nodes using a consistent hash ring
//! for balanced chunk placement.

use super::protocol::{ChunkLocation, ChunkMetadata, NodeId, NodeInfo};
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Configuration for the distributed registry.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Number of virtual nodes per physical node (for better distribution)
    pub virtual_nodes: usize,
    /// Replication factor (number of copies per chunk)
    pub replication_factor: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            virtual_nodes: 150,    // Standard value for consistent hashing
            replication_factor: 3, // Default 3-way replication
        }
    }
}

/// Statistics for the distributed registry.
#[derive(Debug, Clone, Default)]
pub struct RegistryStats {
    /// Total number of chunks tracked
    pub total_chunks: usize,
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of lookups performed
    pub lookups: usize,
    /// Number of registrations
    pub registrations: usize,
    /// Number of evictions
    pub evictions: usize,
}

/// Consistent hash ring for distributing chunks across nodes.
pub struct ConsistentHashRing {
    /// Ring of virtual nodes sorted by hash
    ring: BTreeMap<u64, NodeId>,
    /// Mapping from virtual node hash to physical node
    nodes: HashMap<NodeId, NodeInfo>,
    /// Configuration
    config: RegistryConfig,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            ring: BTreeMap::new(),
            nodes: HashMap::new(),
            config,
        }
    }

    /// Add a node to the ring.
    pub fn add_node(&mut self, info: NodeInfo) {
        let node_id = info.id;
        self.nodes.insert(node_id, info);

        // Add virtual nodes
        for i in 0..self.config.virtual_nodes {
            let hash = Self::hash_virtual_node(node_id, i);
            self.ring.insert(hash, node_id);
        }
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, node_id: NodeId) -> Result<()> {
        self.nodes
            .remove(&node_id)
            .ok_or_else(|| anyhow!("Node not found: {}", node_id))?;

        // Remove virtual nodes
        for i in 0..self.config.virtual_nodes {
            let hash = Self::hash_virtual_node(node_id, i);
            self.ring.remove(&hash);
        }

        Ok(())
    }

    /// Get the primary node for a chunk.
    pub fn get_node(&self, chunk_id: &str) -> Option<NodeId> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = Self::hash_chunk(chunk_id);

        // Find the first node in the ring >= hash (wrap around)
        self.ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, &node_id)| node_id)
    }

    /// Get replica nodes for a chunk (including primary).
    pub fn get_replica_nodes(&self, chunk_id: &str) -> Vec<NodeId> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let hash = Self::hash_chunk(chunk_id);
        let mut replicas = Vec::new();
        let mut seen_physical_nodes = std::collections::HashSet::new();

        // Start from the hash position and walk the ring
        let iter = self
            .ring
            .range(hash..)
            .chain(self.ring.range(..hash))
            .map(|(_, &node_id)| node_id);

        for node_id in iter {
            if seen_physical_nodes.insert(node_id) {
                replicas.push(node_id);
                if replicas.len() >= self.config.replication_factor {
                    break;
                }
            }
        }

        replicas
    }

    /// Get node information.
    pub fn get_node_info(&self, node_id: NodeId) -> Option<&NodeInfo> {
        self.nodes.get(&node_id)
    }

    /// Get all nodes.
    pub fn all_nodes(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Get number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Hash a chunk ID to a position on the ring.
    fn hash_chunk(chunk_id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        chunk_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash a virtual node to a position on the ring.
    fn hash_virtual_node(node_id: NodeId, virtual_index: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        node_id.as_u64().hash(&mut hasher);
        virtual_index.hash(&mut hasher);
        hasher.finish()
    }
}

/// Chunk placement information.
#[derive(Debug, Clone)]
pub struct ChunkPlacement {
    /// Chunk metadata
    pub metadata: ChunkMetadata,
    /// Locations where chunk is stored (primary + replicas)
    pub locations: Vec<ChunkLocation>,
}

/// Distributed chunk registry.
pub struct DistributedRegistry {
    /// Consistent hash ring for placement
    ring: Arc<RwLock<ConsistentHashRing>>,
    /// Chunk metadata and locations
    chunks: Arc<RwLock<HashMap<String, ChunkPlacement>>>,
    /// Statistics
    stats: Arc<RwLock<RegistryStats>>,
    /// Configuration
    config: RegistryConfig,
}

impl DistributedRegistry {
    /// Create a new distributed registry.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            ring: Arc::new(RwLock::new(ConsistentHashRing::new(config.clone()))),
            chunks: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RegistryStats::default())),
            config,
        }
    }

    /// Register a new node in the cluster.
    pub fn register_node(&self, info: NodeInfo) {
        let mut ring = self.ring.write();
        ring.add_node(info);

        let mut stats = self.stats.write();
        stats.total_nodes = ring.num_nodes();
    }

    /// Unregister a node from the cluster.
    pub fn unregister_node(&self, node_id: NodeId) -> Result<()> {
        let mut ring = self.ring.write();
        ring.remove_node(node_id)?;

        // Remove chunks hosted on this node
        let mut chunks = self.chunks.write();
        chunks.retain(|_, placement| {
            placement.locations.retain(|loc| loc.node_id != node_id);
            !placement.locations.is_empty()
        });

        let mut stats = self.stats.write();
        stats.total_nodes = ring.num_nodes();

        Ok(())
    }

    /// Register a chunk in the registry.
    pub fn register_chunk(&self, metadata: ChunkMetadata, locations: Vec<ChunkLocation>) {
        let mut chunks = self.chunks.write();
        chunks.insert(
            metadata.chunk_id.clone(),
            ChunkPlacement {
                metadata,
                locations,
            },
        );

        let mut stats = self.stats.write();
        stats.total_chunks = chunks.len();
        stats.registrations += 1;
    }

    /// Get chunk placement (metadata + locations).
    pub fn get_chunk_placement(&self, chunk_id: &str) -> Option<ChunkPlacement> {
        let chunks = self.chunks.read();
        let placement = chunks.get(chunk_id).cloned();

        if placement.is_some() {
            let mut stats = self.stats.write();
            stats.lookups += 1;
        }

        placement
    }

    /// Get the primary node for a chunk (from consistent hashing).
    pub fn get_primary_node(&self, chunk_id: &str) -> Option<NodeId> {
        let ring = self.ring.read();
        ring.get_node(chunk_id)
    }

    /// Get all replica nodes for a chunk (from consistent hashing).
    pub fn get_replica_nodes(&self, chunk_id: &str) -> Vec<NodeId> {
        let ring = self.ring.read();
        ring.get_replica_nodes(chunk_id)
    }

    /// Get node information.
    pub fn get_node_info(&self, node_id: NodeId) -> Option<NodeInfo> {
        let ring = self.ring.read();
        ring.get_node_info(node_id).cloned()
    }

    /// Get all nodes in the cluster.
    pub fn all_nodes(&self) -> Vec<NodeId> {
        let ring = self.ring.read();
        ring.all_nodes()
    }

    /// Remove a chunk from the registry.
    pub fn evict_chunk(&self, chunk_id: &str) -> Option<ChunkPlacement> {
        let mut chunks = self.chunks.write();
        let placement = chunks.remove(chunk_id);

        if placement.is_some() {
            let mut stats = self.stats.write();
            stats.total_chunks = chunks.len();
            stats.evictions += 1;
        }

        placement
    }

    /// Get current statistics.
    pub fn stats(&self) -> RegistryStats {
        self.stats.read().clone()
    }

    /// Get configuration.
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn create_test_node(id: u64, addr: &str) -> NodeInfo {
        NodeInfo::new(NodeId(id), addr.parse::<SocketAddr>().unwrap())
    }

    #[test]
    fn test_consistent_hash_ring() {
        let config = RegistryConfig::default();
        let mut ring = ConsistentHashRing::new(config);

        let node1 = create_test_node(1, "127.0.0.1:8001");
        let node2 = create_test_node(2, "127.0.0.1:8002");

        ring.add_node(node1);
        ring.add_node(node2);

        assert_eq!(ring.num_nodes(), 2);

        // Get node for a chunk
        let chunk_id = "test_chunk";
        let node = ring.get_node(chunk_id);
        assert!(node.is_some());
    }

    #[test]
    fn test_replica_placement() {
        let config = RegistryConfig {
            virtual_nodes: 150,
            replication_factor: 3,
        };
        let mut ring = ConsistentHashRing::new(config);

        for i in 0..5 {
            let node = create_test_node(i, &format!("127.0.0.1:800{}", i));
            ring.add_node(node);
        }

        let replicas = ring.get_replica_nodes("test_chunk");
        assert_eq!(replicas.len(), 3);

        // All replicas should be different physical nodes
        let unique_nodes: std::collections::HashSet<_> = replicas.iter().collect();
        assert_eq!(unique_nodes.len(), 3);
    }

    #[test]
    fn test_node_removal() {
        let config = RegistryConfig::default();
        let mut ring = ConsistentHashRing::new(config);

        let node1 = create_test_node(1, "127.0.0.1:8001");
        let node2 = create_test_node(2, "127.0.0.1:8002");

        ring.add_node(node1.clone());
        ring.add_node(node2);

        assert_eq!(ring.num_nodes(), 2);

        ring.remove_node(node1.id).unwrap();
        assert_eq!(ring.num_nodes(), 1);
    }

    #[test]
    fn test_registry_operations() {
        let config = RegistryConfig::default();
        let registry = DistributedRegistry::new(config);

        let node1 = create_test_node(1, "127.0.0.1:8001");
        let node2 = create_test_node(2, "127.0.0.1:8002");

        registry.register_node(node1.clone());
        registry.register_node(node2.clone());

        let stats = registry.stats();
        assert_eq!(stats.total_nodes, 2);

        // Register a chunk
        let metadata = ChunkMetadata::new_f64("chunk_1".to_string(), vec![10, 10]);
        let locations = vec![ChunkLocation {
            node_id: node1.id,
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

        let stats = registry.stats();
        assert_eq!(stats.lookups, 1);
    }

    #[test]
    fn test_chunk_eviction() {
        let config = RegistryConfig::default();
        let registry = DistributedRegistry::new(config);

        let metadata = ChunkMetadata::new_f64("chunk_to_evict".to_string(), vec![5, 5]);
        let locations = vec![ChunkLocation {
            node_id: NodeId(1),
            path: "/tmp/chunk".to_string(),
            in_memory: false,
        }];

        registry.register_chunk(metadata, locations);

        let placement = registry.evict_chunk("chunk_to_evict");
        assert!(placement.is_some());

        let placement2 = registry.get_chunk_placement("chunk_to_evict");
        assert!(placement2.is_none());

        let stats = registry.stats();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.evictions, 1);
    }
}
