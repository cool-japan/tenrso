//! Network-aware prefetching for distributed out-of-core processing.
//!
//! Extends prefetching with network latency awareness and batch remote requests.

use super::network::NetworkClient;
use super::protocol::NodeId;
use super::registry::DistributedRegistry;
use super::remote_cache::RemoteCache;
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Prefetch strategy for network-aware prefetching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkPrefetchStrategy {
    /// Sequential prefetching with network batching
    Sequential,
    /// Adaptive based on network conditions
    Adaptive,
    /// Aggressive prefetching with parallel requests
    Aggressive,
}

/// Configuration for network-aware prefetcher.
#[derive(Debug, Clone)]
pub struct NetworkPrefetchConfig {
    /// Prefetch strategy
    pub strategy: NetworkPrefetchStrategy,
    /// Number of chunks to prefetch ahead
    pub lookahead: usize,
    /// Maximum batch size for remote requests
    pub max_batch_size: usize,
    /// Network latency threshold (ms) for adaptive mode
    pub latency_threshold_ms: u64,
    /// Bandwidth threshold (MB/s) for adaptive mode
    pub bandwidth_threshold_mbps: f64,
    /// Maximum concurrent prefetch tasks
    pub max_concurrent: usize,
}

impl Default for NetworkPrefetchConfig {
    fn default() -> Self {
        Self {
            strategy: NetworkPrefetchStrategy::Adaptive,
            lookahead: 4,
            max_batch_size: 8,
            latency_threshold_ms: 50,
            bandwidth_threshold_mbps: 100.0,
            max_concurrent: 4,
        }
    }
}

/// Statistics for network-aware prefetching.
#[derive(Debug, Clone, Default)]
pub struct NetworkPrefetchStats {
    /// Total chunks prefetched
    pub total_prefetched: u64,
    /// Prefetch hits (used before eviction)
    pub prefetch_hits: u64,
    /// Prefetch misses (evicted before use)
    pub prefetch_misses: u64,
    /// Total batches sent
    pub batches_sent: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average network latency (ms)
    pub avg_latency_ms: f64,
    /// Estimated network bandwidth (MB/s)
    pub estimated_bandwidth_mbps: f64,
}

impl NetworkPrefetchStats {
    /// Calculate prefetch hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.prefetch_hits + self.prefetch_misses;
        if total == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / total as f64
        }
    }
}

/// Network measurement for adaptive strategy.
struct NetworkMeasurement {
    /// Latency samples (ms)
    latency_samples: VecDeque<u64>,
    /// Bandwidth samples (MB/s)
    bandwidth_samples: VecDeque<f64>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl NetworkMeasurement {
    fn new(max_samples: usize) -> Self {
        Self {
            latency_samples: VecDeque::new(),
            bandwidth_samples: VecDeque::new(),
            max_samples,
        }
    }

    fn record_latency(&mut self, latency_ms: u64) {
        self.latency_samples.push_back(latency_ms);
        if self.latency_samples.len() > self.max_samples {
            self.latency_samples.pop_front();
        }
    }

    fn record_bandwidth(&mut self, bandwidth_mbps: f64) {
        self.bandwidth_samples.push_back(bandwidth_mbps);
        if self.bandwidth_samples.len() > self.max_samples {
            self.bandwidth_samples.pop_front();
        }
    }

    fn avg_latency(&self) -> f64 {
        if self.latency_samples.is_empty() {
            0.0
        } else {
            self.latency_samples.iter().sum::<u64>() as f64 / self.latency_samples.len() as f64
        }
    }

    fn avg_bandwidth(&self) -> f64 {
        if self.bandwidth_samples.is_empty() {
            0.0
        } else {
            self.bandwidth_samples.iter().sum::<f64>() / self.bandwidth_samples.len() as f64
        }
    }
}

/// Network-aware prefetcher.
pub struct NetworkPrefetcher {
    config: NetworkPrefetchConfig,
    #[allow(dead_code)]
    registry: Arc<DistributedRegistry>,
    remote_cache: Arc<RemoteCache>,
    #[allow(dead_code)]
    network_client: Arc<NetworkClient>,
    stats: Arc<NetworkPrefetchStatsInternal>,
    measurements: Arc<RwLock<NetworkMeasurement>>,
    prefetch_queue: Arc<RwLock<VecDeque<String>>>,
}

struct NetworkPrefetchStatsInternal {
    total_prefetched: AtomicU64,
    prefetch_hits: AtomicU64,
    prefetch_misses: AtomicU64,
    batches_sent: AtomicU64,
    total_batch_size: AtomicU64,
}

impl NetworkPrefetchStatsInternal {
    fn new() -> Self {
        Self {
            total_prefetched: AtomicU64::new(0),
            prefetch_hits: AtomicU64::new(0),
            prefetch_misses: AtomicU64::new(0),
            batches_sent: AtomicU64::new(0),
            total_batch_size: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, measurements: &NetworkMeasurement) -> NetworkPrefetchStats {
        let batches = self.batches_sent.load(Ordering::Relaxed);
        let total_size = self.total_batch_size.load(Ordering::Relaxed);

        NetworkPrefetchStats {
            total_prefetched: self.total_prefetched.load(Ordering::Relaxed),
            prefetch_hits: self.prefetch_hits.load(Ordering::Relaxed),
            prefetch_misses: self.prefetch_misses.load(Ordering::Relaxed),
            batches_sent: batches,
            avg_batch_size: if batches > 0 {
                total_size as f64 / batches as f64
            } else {
                0.0
            },
            avg_latency_ms: measurements.avg_latency(),
            estimated_bandwidth_mbps: measurements.avg_bandwidth(),
        }
    }
}

impl NetworkPrefetcher {
    /// Create a new network-aware prefetcher.
    pub fn new(
        config: NetworkPrefetchConfig,
        registry: Arc<DistributedRegistry>,
        remote_cache: Arc<RemoteCache>,
        network_client: Arc<NetworkClient>,
    ) -> Self {
        Self {
            config,
            registry,
            remote_cache,
            network_client,
            stats: Arc::new(NetworkPrefetchStatsInternal::new()),
            measurements: Arc::new(RwLock::new(NetworkMeasurement::new(100))),
            prefetch_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Schedule chunks for prefetching.
    pub fn schedule_prefetch(&self, chunk_ids: &[String]) {
        let mut queue = self.prefetch_queue.write();
        for chunk_id in chunk_ids {
            if !queue.contains(chunk_id) {
                queue.push_back(chunk_id.clone());
            }
        }
    }

    /// Execute prefetching based on strategy.
    pub async fn execute_prefetch(&self, requester: NodeId) -> Result<usize> {
        let batch_size = self.determine_batch_size();
        let mut batch = Vec::new();

        {
            let mut queue = self.prefetch_queue.write();
            for _ in 0..batch_size {
                if let Some(chunk_id) = queue.pop_front() {
                    batch.push(chunk_id);
                }
            }
        }

        if batch.is_empty() {
            return Ok(0);
        }

        // Prefetch batch
        let start = Instant::now();
        let mut fetched = 0;

        for chunk_id in &batch {
            match self.remote_cache.get(chunk_id, requester).await {
                Ok(_) => {
                    fetched += 1;
                    self.stats.total_prefetched.fetch_add(1, Ordering::Relaxed);
                    self.stats.prefetch_hits.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    self.stats.prefetch_misses.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Record measurements
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_millis() as u64;

        let bytes_transferred = fetched * 1024 * 1024; // Estimate
        let bandwidth_mbps = if elapsed.as_secs_f64() > 0.0 {
            (bytes_transferred as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64()
        } else {
            0.0
        };

        {
            let mut measurements = self.measurements.write();
            measurements.record_latency(latency_ms);
            measurements.record_bandwidth(bandwidth_mbps);
        }

        self.stats.batches_sent.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_batch_size
            .fetch_add(batch.len() as u64, Ordering::Relaxed);

        Ok(fetched)
    }

    /// Determine batch size based on strategy and network conditions.
    fn determine_batch_size(&self) -> usize {
        match self.config.strategy {
            NetworkPrefetchStrategy::Sequential => 1,
            NetworkPrefetchStrategy::Aggressive => self.config.max_batch_size,
            NetworkPrefetchStrategy::Adaptive => {
                let measurements = self.measurements.read();
                let avg_latency = measurements.avg_latency();
                let avg_bandwidth = measurements.avg_bandwidth();

                // Increase batch size for low latency and high bandwidth
                if avg_latency < self.config.latency_threshold_ms as f64
                    && avg_bandwidth > self.config.bandwidth_threshold_mbps
                {
                    self.config.max_batch_size
                } else if avg_latency < self.config.latency_threshold_ms as f64 * 2.0 {
                    self.config.max_batch_size / 2
                } else {
                    1
                }
            }
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> NetworkPrefetchStats {
        let measurements = self.measurements.read();
        self.stats.snapshot(&measurements)
    }

    /// Record a prefetch hit (chunk was used).
    pub fn record_hit(&self) {
        self.stats.prefetch_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a prefetch miss (chunk was evicted before use).
    pub fn record_miss(&self) {
        self.stats.prefetch_misses.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::super::network::{NetworkClient, NetworkConfig};
    use super::super::registry::{DistributedRegistry, RegistryConfig};
    use super::super::remote_cache::{RemoteCache, RemoteCacheConfig};
    use super::*;

    #[test]
    fn test_network_prefetch_config() {
        let config = NetworkPrefetchConfig::default();
        assert!(config.lookahead > 0);
        assert!(config.max_batch_size > 0);
    }

    #[test]
    fn test_network_measurement() {
        let mut measurement = NetworkMeasurement::new(10);

        measurement.record_latency(50);
        measurement.record_latency(60);
        measurement.record_latency(40);

        assert_eq!(measurement.avg_latency(), 50.0);

        measurement.record_bandwidth(100.0);
        measurement.record_bandwidth(200.0);

        assert_eq!(measurement.avg_bandwidth(), 150.0);
    }

    #[tokio::test]
    async fn test_network_prefetcher_creation() {
        let config = NetworkPrefetchConfig::default();
        let registry = Arc::new(DistributedRegistry::new(RegistryConfig::default()));
        let network_client = Arc::new(NetworkClient::new(NetworkConfig::default()));
        let cache_config = RemoteCacheConfig::default();
        let remote_cache = Arc::new(RemoteCache::new(
            cache_config,
            Arc::clone(&registry),
            Arc::clone(&network_client),
        ));

        let prefetcher = NetworkPrefetcher::new(config, registry, remote_cache, network_client);

        let stats = prefetcher.stats();
        assert_eq!(stats.total_prefetched, 0);
        assert_eq!(stats.batches_sent, 0);
    }

    #[test]
    fn test_prefetch_stats() {
        let stats = NetworkPrefetchStats {
            total_prefetched: 100,
            prefetch_hits: 80,
            prefetch_misses: 20,
            batches_sent: 10,
            avg_batch_size: 10.0,
            avg_latency_ms: 45.0,
            estimated_bandwidth_mbps: 150.0,
        };

        assert_eq!(stats.hit_rate(), 0.8);
    }
}
