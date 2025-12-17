//! Benchmarks for distributed out-of-core processing.

#![cfg(feature = "distributed")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::hint::black_box;
use std::net::SocketAddr;
use tenrso_ooc::distributed::*;

// Mock chunk provider for benchmarks
#[allow(dead_code)]
struct BenchChunkProvider {
    chunks: HashMap<String, Vec<u8>>,
}

#[allow(dead_code)]
impl BenchChunkProvider {
    fn new(num_chunks: usize, chunk_size: usize) -> Self {
        let mut chunks = HashMap::new();
        for i in 0..num_chunks {
            let chunk_id = format!("chunk_{}", i);
            let data = vec![i as u8; chunk_size];
            chunks.insert(chunk_id, data);
        }
        Self { chunks }
    }
}

impl tenrso_ooc::distributed::network::ChunkProvider for BenchChunkProvider {
    fn load_chunk(&self, chunk_id: &str) -> anyhow::Result<Vec<u8>> {
        self.chunks
            .get(chunk_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Chunk not found"))
    }
}

fn bench_consistent_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("consistent_hashing");

    for num_nodes in [5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*num_nodes as u64));
        group.bench_with_input(
            BenchmarkId::new("node_lookup", num_nodes),
            num_nodes,
            |b, &num_nodes| {
                let config = RegistryConfig {
                    virtual_nodes: 150,
                    replication_factor: 3,
                };
                let mut ring = ConsistentHashRing::new(config);

                // Add nodes
                for i in 0..num_nodes {
                    let addr: SocketAddr = format!("127.0.0.1:{}", 9000 + i).parse().unwrap();
                    let node = NodeInfo::new(NodeId(i as u64), addr);
                    ring.add_node(node);
                }

                b.iter(|| {
                    let chunk_id = format!("chunk_{}", black_box(42));
                    black_box(ring.get_node(&chunk_id));
                });
            },
        );
    }

    group.finish();
}

fn bench_replica_placement(c: &mut Criterion) {
    let mut group = c.benchmark_group("replica_placement");

    for replication_factor in [3, 5, 7].iter() {
        group.throughput(Throughput::Elements(*replication_factor as u64));
        group.bench_with_input(
            BenchmarkId::new("get_replicas", replication_factor),
            replication_factor,
            |b, &replication_factor| {
                let config = RegistryConfig {
                    virtual_nodes: 150,
                    replication_factor,
                };
                let mut ring = ConsistentHashRing::new(config);

                // Add many nodes
                for i in 0..20 {
                    let addr: SocketAddr = format!("127.0.0.1:{}", 9000 + i).parse().unwrap();
                    let node = NodeInfo::new(NodeId(i as u64), addr);
                    ring.add_node(node);
                }

                b.iter(|| {
                    let chunk_id = format!("chunk_{}", black_box(42));
                    black_box(ring.get_replica_nodes(&chunk_id));
                });
            },
        );
    }

    group.finish();
}

fn bench_registry_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_operations");

    for num_chunks in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*num_chunks as u64));
        group.bench_with_input(
            BenchmarkId::new("chunk_registration", num_chunks),
            num_chunks,
            |b, &num_chunks| {
                let config = RegistryConfig::default();
                let registry = DistributedRegistry::new(config);

                // Register nodes
                for i in 0..5 {
                    let addr: SocketAddr = format!("127.0.0.1:{}", 9000 + i).parse().unwrap();
                    let node = NodeInfo::new(NodeId(i as u64), addr);
                    registry.register_node(node);
                }

                b.iter(|| {
                    for i in 0..num_chunks {
                        let chunk_id = format!("chunk_{}", i);
                        let metadata = ChunkMetadata::new_f64(chunk_id.clone(), vec![10, 10]);
                        let locations = vec![ChunkLocation {
                            node_id: NodeId(0),
                            path: format!("/tmp/{}", chunk_id),
                            in_memory: false,
                        }];
                        registry.register_chunk(metadata, locations);
                    }
                });
            },
        );
    }

    for num_lookups in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*num_lookups as u64));
        group.bench_with_input(
            BenchmarkId::new("chunk_lookup", num_lookups),
            num_lookups,
            |b, &num_lookups| {
                let config = RegistryConfig::default();
                let registry = DistributedRegistry::new(config);

                // Pre-register chunks
                for i in 0..num_lookups {
                    let chunk_id = format!("chunk_{}", i);
                    let metadata = ChunkMetadata::new_f64(chunk_id.clone(), vec![10, 10]);
                    let locations = vec![ChunkLocation {
                        node_id: NodeId(0),
                        path: format!("/tmp/{}", chunk_id),
                        in_memory: false,
                    }];
                    registry.register_chunk(metadata, locations);
                }

                b.iter(|| {
                    for i in 0..num_lookups {
                        let chunk_id = format!("chunk_{}", i);
                        black_box(registry.get_chunk_placement(&chunk_id));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_protocol_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("protocol_serialization");

    group.bench_function("chunk_request", |b| {
        let request = ChunkRequest::new("test_chunk".to_string(), NodeId(42));
        let bincode_config = bincode::config::standard();
        b.iter(|| {
            let serialized = bincode::serde::encode_to_vec(&request, bincode_config).unwrap();
            black_box(serialized);
        });
    });

    group.bench_function("chunk_response", |b| {
        let metadata = ChunkMetadata::new_f64("test".to_string(), vec![100, 100]);
        let data = vec![42u8; 1024 * 1024]; // 1MB
        let response = ChunkResponse::success("test".to_string(), metadata, data);
        let bincode_config = bincode::config::standard();

        b.iter(|| {
            let serialized = bincode::serde::encode_to_vec(&response, bincode_config).unwrap();
            black_box(serialized);
        });
    });

    group.bench_function("heartbeat", |b| {
        let request = HeartbeatRequest::new(NodeId(1));
        let bincode_config = bincode::config::standard();
        b.iter(|| {
            let serialized = bincode::serde::encode_to_vec(&request, bincode_config).unwrap();
            black_box(serialized);
        });
    });

    group.finish();
}

fn bench_node_addition_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_management");

    group.bench_function("add_node", |b| {
        b.iter_with_setup(
            || {
                let config = RegistryConfig {
                    virtual_nodes: 150,
                    replication_factor: 3,
                };
                ConsistentHashRing::new(config)
            },
            |mut ring| {
                let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
                let node = NodeInfo::new(NodeId(1), addr);
                ring.add_node(node);
                black_box(ring);
            },
        );
    });

    group.bench_function("remove_node", |b| {
        b.iter_with_setup(
            || {
                let config = RegistryConfig {
                    virtual_nodes: 150,
                    replication_factor: 3,
                };
                let mut ring = ConsistentHashRing::new(config);
                let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
                let node = NodeInfo::new(NodeId(1), addr);
                ring.add_node(node);
                ring
            },
            |mut ring| {
                ring.remove_node(NodeId(1)).unwrap();
                black_box(ring);
            },
        );
    });

    group.finish();
}

fn bench_chunk_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_distribution");

    for num_nodes in [5, 10, 20].iter() {
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("distribute_chunks", num_nodes),
            num_nodes,
            |b, &num_nodes| {
                let config = RegistryConfig {
                    virtual_nodes: 150,
                    replication_factor: 3,
                };
                let mut ring = ConsistentHashRing::new(config);

                for i in 0..num_nodes {
                    let addr: SocketAddr = format!("127.0.0.1:{}", 9000 + i).parse().unwrap();
                    let node = NodeInfo::new(NodeId(i as u64), addr);
                    ring.add_node(node);
                }

                b.iter(|| {
                    let mut distribution = std::collections::HashMap::new();
                    for i in 0..1000 {
                        let chunk_id = format!("chunk_{}", i);
                        if let Some(node) = ring.get_node(&chunk_id) {
                            *distribution.entry(node).or_insert(0) += 1;
                        }
                    }
                    black_box(distribution);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_consistent_hashing,
    bench_replica_placement,
    bench_registry_operations,
    bench_protocol_serialization,
    bench_node_addition_removal,
    bench_chunk_distribution,
);

criterion_main!(benches);
