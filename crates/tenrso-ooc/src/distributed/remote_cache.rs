//! Remote chunk caching for distributed out-of-core processing.
//!
//! Caches chunks fetched from remote nodes to minimize network traffic.

use super::network::NetworkClient;
use super::protocol::{ChunkMetadata, ChunkRequest, NodeId};
use super::registry::DistributedRegistry;
use crate::ml_eviction::{MLConfig, MLEvictionPolicy};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Cache policy for remote chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Machine Learning-based eviction
    ML,
}

/// Write policy for cached chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePolicy {
    /// Write through to remote node immediately
    WriteThrough,
    /// Write back to remote node on eviction
    WriteBack,
    /// No write (read-only cache)
    NoWrite,
}

/// Configuration for remote cache.
#[derive(Debug, Clone)]
pub struct RemoteCacheConfig {
    /// Maximum cache size in bytes
    pub max_cache_size: u64,
    /// Cache eviction policy
    pub cache_policy: CachePolicy,
    /// Write policy
    pub write_policy: WritePolicy,
    /// Whether to prefetch on miss
    pub prefetch_on_miss: bool,
    /// ML configuration (if using ML policy)
    pub ml_config: Option<MLConfig>,
}

impl Default for RemoteCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 1024 * 1024 * 1024, // 1GB
            cache_policy: CachePolicy::ML,
            write_policy: WritePolicy::NoWrite,
            prefetch_on_miss: true,
            ml_config: Some(MLConfig::default()),
        }
    }
}

/// Statistics for remote cache.
#[derive(Debug, Clone, Default)]
pub struct RemoteCacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Current cache size in bytes
    pub current_size: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Network bytes fetched
    pub network_bytes: u64,
    /// Number of cached chunks
    pub num_chunks: usize,
}

impl RemoteCacheStats {
    /// Calculate hit rate (0.0 - 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Cached chunk entry.
struct CacheEntry {
    /// Chunk ID
    #[allow(dead_code)]
    chunk_id: String,
    /// Metadata
    #[allow(dead_code)]
    metadata: ChunkMetadata,
    /// Chunk data
    data: Vec<u8>,
    /// Last access time
    last_accessed: Instant,
    /// Access count
    access_count: u64,
    /// Whether chunk has been modified
    dirty: bool,
    /// Source node ID
    #[allow(dead_code)]
    source_node: NodeId,
}

impl CacheEntry {
    fn size(&self) -> u64 {
        self.data.len() as u64
    }
}

/// Remote chunk cache.
pub struct RemoteCache {
    config: RemoteCacheConfig,
    registry: Arc<DistributedRegistry>,
    network_client: Arc<NetworkClient>,
    cache: Arc<DashMap<String, Arc<RwLock<CacheEntry>>>>,
    ml_policy: Option<Arc<RwLock<MLEvictionPolicy>>>,
    stats: Arc<RemoteCacheStatsInternal>,
}

struct RemoteCacheStatsInternal {
    hits: AtomicU64,
    misses: AtomicU64,
    current_size: AtomicU64,
    evictions: AtomicU64,
    network_bytes: AtomicU64,
}

impl RemoteCacheStatsInternal {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            current_size: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            network_bytes: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, num_chunks: usize) -> RemoteCacheStats {
        RemoteCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            current_size: self.current_size.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            network_bytes: self.network_bytes.load(Ordering::Relaxed),
            num_chunks,
        }
    }
}

impl RemoteCache {
    /// Create a new remote cache.
    pub fn new(
        config: RemoteCacheConfig,
        registry: Arc<DistributedRegistry>,
        network_client: Arc<NetworkClient>,
    ) -> Self {
        let ml_policy = if config.cache_policy == CachePolicy::ML {
            config
                .ml_config
                .as_ref()
                .map(|ml_config| Arc::new(RwLock::new(MLEvictionPolicy::new(ml_config.clone()))))
        } else {
            None
        };

        Self {
            config,
            registry,
            network_client,
            cache: Arc::new(DashMap::new()),
            ml_policy,
            stats: Arc::new(RemoteCacheStatsInternal::new()),
        }
    }

    /// Convert a string chunk ID to a numeric ID for ML policy.
    fn chunk_id_to_usize(chunk_id: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        chunk_id.hash(&mut hasher);
        (hasher.finish() as usize) % 1_000_000 // Limit to reasonable range
    }

    /// Get current timestamp as f64.
    fn current_timestamp_f64() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    /// Get a chunk from cache or fetch from remote node.
    pub async fn get(&self, chunk_id: &str, requester: NodeId) -> Result<Vec<u8>> {
        // Check cache
        if let Some(entry_ref) = self.cache.get(chunk_id) {
            let mut entry = entry_ref.write();
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Record access for ML policy
            if let Some(ml_policy) = &self.ml_policy {
                let numeric_id = Self::chunk_id_to_usize(chunk_id);
                let timestamp = Self::current_timestamp_f64();
                ml_policy.write().record_access(numeric_id, timestamp);
            }

            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(entry.data.clone());
        }

        // Cache miss - fetch from remote node
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        self.fetch_and_cache(chunk_id, requester).await
    }

    /// Fetch chunk from remote node and add to cache.
    async fn fetch_and_cache(&self, chunk_id: &str, requester: NodeId) -> Result<Vec<u8>> {
        // Get chunk location from registry
        let placement = self
            .registry
            .get_chunk_placement(chunk_id)
            .ok_or_else(|| anyhow!("Chunk not found in registry: {}", chunk_id))?;

        // Select a node to fetch from (prefer first location)
        let location = placement
            .locations
            .first()
            .ok_or_else(|| anyhow!("No locations for chunk: {}", chunk_id))?;

        // Get node address
        let node_info = self
            .registry
            .get_node_info(location.node_id)
            .ok_or_else(|| anyhow!("Node not found: {}", location.node_id))?;

        // Send chunk request
        let request = ChunkRequest::new(chunk_id.to_string(), requester);
        let response = self
            .network_client
            .request_chunk(node_info.address, request)
            .await?;

        if !response.success {
            return Err(anyhow!(
                "Failed to fetch chunk: {}",
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string())
            ));
        }

        self.stats
            .network_bytes
            .fetch_add(response.data.len() as u64, Ordering::Relaxed);

        let data = response.data.clone();

        // Add to cache
        self.add_to_cache(
            chunk_id.to_string(),
            response.metadata,
            response.data,
            location.node_id,
        )
        .await?;

        Ok(data)
    }

    /// Add a chunk to the cache.
    async fn add_to_cache(
        &self,
        chunk_id: String,
        metadata: ChunkMetadata,
        data: Vec<u8>,
        source_node: NodeId,
    ) -> Result<()> {
        let size = data.len() as u64;

        // Evict chunks if necessary
        self.make_space(size).await?;

        // Create cache entry
        let entry = Arc::new(RwLock::new(CacheEntry {
            chunk_id: chunk_id.clone(),
            metadata,
            data,
            last_accessed: Instant::now(),
            access_count: 1,
            dirty: false,
            source_node,
        }));

        self.cache.insert(chunk_id.clone(), entry);
        self.stats.current_size.fetch_add(size, Ordering::Relaxed);

        // Record access for ML policy
        if let Some(ml_policy) = &self.ml_policy {
            let numeric_id = Self::chunk_id_to_usize(&chunk_id);
            let timestamp = Self::current_timestamp_f64();
            ml_policy.write().record_access(numeric_id, timestamp);
        }

        Ok(())
    }

    /// Make space in cache by evicting chunks.
    async fn make_space(&self, required_size: u64) -> Result<()> {
        let current_size = self.stats.current_size.load(Ordering::Relaxed);

        if current_size + required_size <= self.config.max_cache_size {
            return Ok(());
        }

        let target_size = self.config.max_cache_size - required_size;

        // Get chunks to evict based on policy
        let chunks_to_evict = self.select_chunks_to_evict(current_size - target_size)?;

        // Evict chunks
        for chunk_id in chunks_to_evict {
            self.evict_chunk(&chunk_id).await?;
        }

        Ok(())
    }

    /// Select chunks to evict based on policy.
    fn select_chunks_to_evict(&self, bytes_to_free: u64) -> Result<Vec<String>> {
        let mut candidates = Vec::new();
        let mut freed_bytes = 0u64;

        match self.config.cache_policy {
            CachePolicy::LRU => {
                // Sort by last access time
                let mut entries: Vec<_> = self
                    .cache
                    .iter()
                    .map(|entry| {
                        let e = entry.value().read();
                        (entry.key().clone(), e.last_accessed, e.size())
                    })
                    .collect();

                entries.sort_by_key(|(_, accessed, _)| *accessed);

                for (chunk_id, _, size) in entries {
                    candidates.push(chunk_id);
                    freed_bytes += size;
                    if freed_bytes >= bytes_to_free {
                        break;
                    }
                }
            }
            CachePolicy::LFU => {
                // Sort by access count
                let mut entries: Vec<_> = self
                    .cache
                    .iter()
                    .map(|entry| {
                        let e = entry.value().read();
                        (entry.key().clone(), e.access_count, e.size())
                    })
                    .collect();

                entries.sort_by_key(|(_, count, _)| *count);

                for (chunk_id, _, size) in entries {
                    candidates.push(chunk_id);
                    freed_bytes += size;
                    if freed_bytes >= bytes_to_free {
                        break;
                    }
                }
            }
            CachePolicy::ML => {
                // Use ML policy
                if let Some(ml_policy) = &self.ml_policy {
                    // Get all chunk IDs and convert to numeric IDs
                    let chunk_ids: Vec<usize> = self
                        .cache
                        .iter()
                        .map(|entry| Self::chunk_id_to_usize(entry.key()))
                        .collect();

                    let current_time = Self::current_timestamp_f64();
                    let scores = ml_policy
                        .read()
                        .get_eviction_candidates(&chunk_ids, current_time);

                    // Convert scores back to chunk ID strings
                    let mut scored_chunks: Vec<(String, f64)> = self
                        .cache
                        .iter()
                        .map(|entry| {
                            let numeric_id = Self::chunk_id_to_usize(entry.key());
                            let score = scores
                                .iter()
                                .find(|s| s.chunk_id == numeric_id)
                                .map(|s| s.combined_score)
                                .unwrap_or(0.0);
                            (entry.key().clone(), score)
                        })
                        .collect();

                    // Sort by score (higher score = more likely to evict)
                    scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    // Take chunks until we've freed enough bytes
                    for (chunk_id, _) in scored_chunks {
                        if let Some(entry_ref) = self.cache.get(&chunk_id) {
                            let size = entry_ref.read().size();
                            candidates.push(chunk_id);
                            freed_bytes += size;
                            if freed_bytes >= bytes_to_free {
                                break;
                            }
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Evict a chunk from cache.
    async fn evict_chunk(&self, chunk_id: &str) -> Result<()> {
        if let Some((_, entry_ref)) = self.cache.remove(chunk_id) {
            let entry = entry_ref.read();
            let size = entry.size();

            // Handle write-back if needed
            if entry.dirty && self.config.write_policy == WritePolicy::WriteBack {
                // TODO: Implement write-back to remote node
            }

            self.stats.current_size.fetch_sub(size, Ordering::Relaxed);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get current statistics.
    pub fn stats(&self) -> RemoteCacheStats {
        self.stats.snapshot(self.cache.len())
    }

    /// Clear the entire cache.
    pub async fn clear(&self) {
        self.cache.clear();
        self.stats.current_size.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::super::network::{NetworkClient, NetworkConfig};
    use super::super::registry::{DistributedRegistry, RegistryConfig};
    use super::*;

    #[test]
    fn test_cache_config() {
        let config = RemoteCacheConfig::default();
        assert!(config.max_cache_size > 0);
        assert_eq!(config.cache_policy, CachePolicy::ML);
    }

    #[test]
    fn test_cache_stats() {
        let stats = RemoteCacheStats {
            hits: 80,
            misses: 20,
            current_size: 1024,
            evictions: 5,
            network_bytes: 10240,
            num_chunks: 10,
        };

        assert_eq!(stats.hit_rate(), 0.8);
    }

    #[tokio::test]
    async fn test_remote_cache_creation() {
        let config = RemoteCacheConfig::default();
        let registry = Arc::new(DistributedRegistry::new(RegistryConfig::default()));
        let network_client = Arc::new(NetworkClient::new(NetworkConfig::default()));

        let cache = RemoteCache::new(config, registry, network_client);
        let stats = cache.stats();

        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.num_chunks, 0);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let config = RemoteCacheConfig::default();
        let registry = Arc::new(DistributedRegistry::new(RegistryConfig::default()));
        let network_client = Arc::new(NetworkClient::new(NetworkConfig::default()));

        let cache = RemoteCache::new(config, registry, network_client);
        cache.clear().await;

        let stats = cache.stats();
        assert_eq!(stats.current_size, 0);
    }
}
