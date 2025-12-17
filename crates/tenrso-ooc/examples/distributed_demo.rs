//! Distributed out-of-core processing demonstration.
//!
//! This example demonstrates:
//! - Setting up a distributed registry with consistent hashing
//! - Network communication between nodes
//! - Remote chunk caching
//! - Network-aware prefetching
//! - Fault tolerance and recovery

#![cfg(feature = "distributed")]

use anyhow::Result;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tenrso_ooc::distributed::*;
use tenrso_ooc::MLConfig;

// Mock chunk provider
#[allow(dead_code)]
struct DemoChunkProvider {
    chunks: HashMap<String, Vec<u8>>,
}

#[allow(dead_code)]
impl DemoChunkProvider {
    fn new() -> Self {
        let mut chunks = HashMap::new();

        // Create some demo chunks
        println!("Creating demo chunks...");
        for i in 0..20 {
            let chunk_id = format!("chunk_{:03}", i);
            let size = (i + 1) * 1024; // Varying sizes
            let data = vec![(i % 256) as u8; size];
            chunks.insert(chunk_id.clone(), data);
            println!("  - {} ({} bytes)", chunk_id, size);
        }

        Self { chunks }
    }
}

impl ChunkProvider for DemoChunkProvider {
    fn load_chunk(&self, chunk_id: &str) -> Result<Vec<u8>> {
        self.chunks
            .get(chunk_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Chunk not found: {}", chunk_id))
    }
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   TenRSo Distributed Out-of-Core Processing Demo             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // 1. Setup distributed registry
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. Setting up Distributed Registry");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let registry_config = RegistryConfig {
        virtual_nodes: 150,
        replication_factor: 3,
    };

    let registry = Arc::new(DistributedRegistry::new(registry_config));

    // Register nodes
    println!("Registering nodes:");
    for i in 0..5 {
        let addr: SocketAddr = format!("127.0.0.1:900{}", i).parse()?;
        let mut node = NodeInfo::new(NodeId(i), addr);
        node.available_memory = (i + 1) * 1024 * 1024 * 1024; // GB
        node.available_disk = (i + 1) * 10 * 1024 * 1024 * 1024; // 10GB per node
        node.bandwidth_bps = 1_000_000_000; // 1 Gbps
        node.load = 0.1 * i as f64;

        registry.register_node(node.clone());
        println!(
            "  ✓ Node {} @ {} (Mem: {} GB, Disk: {} GB, Load: {:.1}%)",
            node.id,
            node.address,
            node.available_memory / (1024 * 1024 * 1024),
            node.available_disk / (1024 * 1024 * 1024),
            node.load * 100.0
        );
    }

    let stats = registry.stats();
    println!("\nRegistry Stats:");
    println!("  • Total Nodes: {}", stats.total_nodes);

    // 2. Demonstrate consistent hashing
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. Consistent Hashing & Chunk Placement");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Registering chunks with consistent hashing:\n");

    for i in 0..10 {
        let chunk_id = format!("chunk_{:03}", i);

        // Get primary node from consistent hashing
        let primary = registry
            .get_primary_node(&chunk_id)
            .expect("Failed to get primary node");

        // Get replica nodes
        let replicas = registry.get_replica_nodes(&chunk_id);

        // Create metadata
        let metadata = ChunkMetadata::new_f64(chunk_id.clone(), vec![100, 100]);

        // Create locations
        let locations: Vec<ChunkLocation> = replicas
            .iter()
            .enumerate()
            .map(|(idx, &node_id)| ChunkLocation {
                node_id,
                path: format!("/data/{}", chunk_id),
                in_memory: idx == 0, // Primary in memory
            })
            .collect();

        registry.register_chunk(metadata, locations.clone());

        println!(
            "  {} → Primary: {} | Replicas: {:?}",
            chunk_id, primary, replicas
        );
    }

    let stats = registry.stats();
    println!("\nRegistry Stats:");
    println!("  • Total Chunks: {}", stats.total_chunks);
    println!("  • Registrations: {}", stats.registrations);

    // 3. Network configuration
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. Network Communication Setup");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let network_config = NetworkConfig {
        bind_address: "127.0.0.1:9000".parse()?,
        max_message_size: 1024 * 1024 * 1024, // 1GB
        connection_timeout: std::time::Duration::from_secs(10),
        read_timeout: std::time::Duration::from_secs(30),
        write_timeout: std::time::Duration::from_secs(30),
        max_connections: 100,
    };

    println!("Network Configuration:");
    println!("  • Bind Address: {}", network_config.bind_address);
    println!(
        "  • Max Message Size: {} MB",
        network_config.max_message_size / (1024 * 1024)
    );
    println!(
        "  • Connection Timeout: {:?}",
        network_config.connection_timeout
    );
    println!("  • Max Connections: {}", network_config.max_connections);

    let network_client = Arc::new(NetworkClient::new(network_config));

    // 4. Remote cache
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. Remote Chunk Caching");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let cache_config = RemoteCacheConfig {
        max_cache_size: 100 * 1024 * 1024, // 100MB
        cache_policy: CachePolicy::ML,
        write_policy: WritePolicy::NoWrite,
        prefetch_on_miss: true,
        ml_config: Some(MLConfig::default()),
    };

    println!("Cache Configuration:");
    println!(
        "  • Max Size: {} MB",
        cache_config.max_cache_size / (1024 * 1024)
    );
    println!("  • Policy: {:?}", cache_config.cache_policy);
    println!("  • Write Policy: {:?}", cache_config.write_policy);
    println!("  • Prefetch on Miss: {}", cache_config.prefetch_on_miss);

    let remote_cache = Arc::new(RemoteCache::new(
        cache_config,
        Arc::clone(&registry),
        Arc::clone(&network_client),
    ));

    // 5. Network-aware prefetching
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. Network-Aware Prefetching");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let prefetch_config = NetworkPrefetchConfig {
        strategy: NetworkPrefetchStrategy::Adaptive,
        lookahead: 4,
        max_batch_size: 8,
        latency_threshold_ms: 50,
        bandwidth_threshold_mbps: 100.0,
        max_concurrent: 4,
    };

    println!("Prefetch Configuration:");
    println!("  • Strategy: {:?}", prefetch_config.strategy);
    println!("  • Lookahead: {}", prefetch_config.lookahead);
    println!("  • Max Batch Size: {}", prefetch_config.max_batch_size);
    println!(
        "  • Latency Threshold: {} ms",
        prefetch_config.latency_threshold_ms
    );
    println!(
        "  • Bandwidth Threshold: {:.0} Mbps",
        prefetch_config.bandwidth_threshold_mbps
    );

    let network_prefetcher = NetworkPrefetcher::new(
        prefetch_config,
        Arc::clone(&registry),
        Arc::clone(&remote_cache),
        Arc::clone(&network_client),
    );

    // Schedule some chunks for prefetching
    let chunks_to_prefetch: Vec<String> = (0..5).map(|i| format!("chunk_{:03}", i)).collect();
    println!("\nScheduling chunks for prefetch:");
    for chunk_id in &chunks_to_prefetch {
        println!("  • {}", chunk_id);
    }
    network_prefetcher.schedule_prefetch(&chunks_to_prefetch);

    // 6. Fault tolerance
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. Fault Tolerance & Recovery");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let ft_config = FaultToleranceConfig {
        heartbeat_interval: std::time::Duration::from_secs(5),
        failure_timeout: std::time::Duration::from_secs(30),
        replication_policy: ReplicationPolicy::Fixed(3),
        auto_recovery: true,
        max_recovery_attempts: 3,
    };

    println!("Fault Tolerance Configuration:");
    println!("  • Heartbeat Interval: {:?}", ft_config.heartbeat_interval);
    println!("  • Failure Timeout: {:?}", ft_config.failure_timeout);
    println!("  • Replication Policy: {:?}", ft_config.replication_policy);
    println!("  • Auto Recovery: {}", ft_config.auto_recovery);
    println!(
        "  • Max Recovery Attempts: {}",
        ft_config.max_recovery_attempts
    );

    let ft_manager = FaultToleranceManager::new(
        ft_config,
        Arc::clone(&registry),
        Arc::clone(&network_client),
    );

    // 7. Statistics summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("7. System Statistics Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let registry_stats = registry.stats();
    let cache_stats = remote_cache.stats();
    let prefetch_stats = network_prefetcher.stats();
    let network_stats = network_client.stats();
    let ft_stats = ft_manager.stats();

    println!("Registry:");
    println!("  • Nodes: {}", registry_stats.total_nodes);
    println!("  • Chunks: {}", registry_stats.total_chunks);
    println!("  • Lookups: {}", registry_stats.lookups);
    println!("  • Registrations: {}", registry_stats.registrations);

    println!("\nRemote Cache:");
    println!("  • Hits: {}", cache_stats.hits);
    println!("  • Misses: {}", cache_stats.misses);
    println!("  • Hit Rate: {:.2}%", cache_stats.hit_rate() * 100.0);
    println!("  • Current Size: {} KB", cache_stats.current_size / 1024);
    println!("  • Cached Chunks: {}", cache_stats.num_chunks);

    println!("\nNetwork Prefetcher:");
    println!("  • Total Prefetched: {}", prefetch_stats.total_prefetched);
    println!("  • Batches Sent: {}", prefetch_stats.batches_sent);
    println!("  • Avg Batch Size: {:.1}", prefetch_stats.avg_batch_size);
    println!("  • Avg Latency: {:.1} ms", prefetch_stats.avg_latency_ms);
    println!(
        "  • Bandwidth: {:.1} Mbps",
        prefetch_stats.estimated_bandwidth_mbps
    );

    println!("\nNetwork:");
    println!("  • Bytes Sent: {} KB", network_stats.bytes_sent / 1024);
    println!(
        "  • Bytes Received: {} KB",
        network_stats.bytes_received / 1024
    );
    println!("  • Messages Sent: {}", network_stats.messages_sent);
    println!("  • Messages Received: {}", network_stats.messages_received);

    println!("\nFault Tolerance:");
    println!("  • Heartbeats Sent: {}", ft_stats.heartbeats_sent);
    println!("  • Heartbeats Received: {}", ft_stats.heartbeats_received);
    println!("  • Node Failures: {}", ft_stats.node_failures);
    println!("  • Healthy Nodes: {}", ft_stats.healthy_nodes);
    println!("  • Failed Nodes: {}", ft_stats.failed_nodes);

    // 8. Benefits summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("8. Benefits of Distributed Out-of-Core Processing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("✓ Scalability:");
    println!("  • Process tensors larger than single-node memory");
    println!("  • Linear scaling with number of nodes");
    println!("  • Dynamic node addition/removal");

    println!("\n✓ Fault Tolerance:");
    println!("  • Automatic failure detection via heartbeats");
    println!("  • Chunk replication for redundancy");
    println!("  • Automatic recovery on node failure");

    println!("\n✓ Performance:");
    println!("  • Consistent hashing for balanced load");
    println!("  • ML-based cache eviction");
    println!("  • Network-aware batched prefetching");
    println!("  • Adaptive strategy based on network conditions");

    println!("\n✓ Flexibility:");
    println!("  • Configurable replication factors");
    println!("  • Multiple cache policies (LRU/LFU/ML)");
    println!("  • Tunable network parameters");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}
