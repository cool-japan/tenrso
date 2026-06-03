//! Micro-benchmarks for individual OoC primitive operations.
//!
//! These benchmarks measure the overhead of foundational operations that compose
//! larger out-of-core pipelines. Each benchmark targets a single primitive and
//! uses small, fast-running inputs to measure sub-millisecond latencies.
//!
//! Run with:
//! ```bash
//! cargo bench --bench micro -p tenrso-ooc --all-features
//! ```

#![deny(warnings)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::DenseND;
use tenrso_ooc::{
    chunk_graph::{ChunkGraph, ChunkNode, ChunkOp},
    compute_checksum,
    data_integrity::ChecksumAlgorithm,
    memory::{AccessPattern, MemoryManager},
    prefetch::{PrefetchStrategy, Prefetcher},
    working_set::{PredictionMode, WorkingSetPredictor},
};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build a deterministic flat `f64` payload of `n` bytes (rounded down to
/// the nearest f64 boundary).
fn make_payload_bytes(n: usize) -> Vec<u8> {
    let count = n / std::mem::size_of::<f64>();
    let values: Vec<f64> = (0..count).map(|i| (i as f64) * 0.001).collect();
    let mut bytes = vec![0u8; values.len() * std::mem::size_of::<f64>()];
    for (i, v) in values.iter().enumerate() {
        let b = v.to_le_bytes();
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&b);
    }
    bytes
}

/// Create a small dense f64 tensor with deterministic content.
fn make_tensor(shape: &[usize]) -> DenseND<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
    DenseND::from_vec(data, shape).expect("make_tensor: shape must match element count")
}

// ─── 1. Chunk graph construction ─────────────────────────────────────────────

fn bench_chunk_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_graph_construction");

    // Small graph: 4 inputs + 2 ops + 1 sink
    group.bench_function("small_7_nodes", |b| {
        b.iter(|| {
            let mut graph = ChunkGraph::new();
            let n0 = graph.add_node(ChunkNode::input("A", vec![0, 0]));
            let n1 = graph.add_node(ChunkNode::input("A", vec![0, 1]));
            let n2 = graph.add_node(ChunkNode::input("B", vec![0, 0]));
            let n3 = graph.add_node(ChunkNode::input("B", vec![1, 0]));
            let n4 = graph.add_node(ChunkNode::operation(ChunkOp::MatMul, vec![n0, n2]));
            let n5 = graph.add_node(ChunkNode::operation(ChunkOp::MatMul, vec![n1, n3]));
            let _n6 = graph.add_node(ChunkNode::operation(ChunkOp::Add, vec![n4, n5]));
            black_box(graph.len())
        });
    });

    // Large graph: many input tiles + reduction tree
    group.bench_function("large_64_nodes", |b| {
        b.iter(|| {
            let mut graph = ChunkGraph::new();
            // 32 input nodes
            let inputs: Vec<usize> = (0..32)
                .map(|i| graph.add_node(ChunkNode::input("T", vec![i, 0])))
                .collect();
            // 16 first-level accumulations
            let level1: Vec<usize> = inputs
                .chunks(2)
                .map(|pair| {
                    graph.add_node(ChunkNode::operation(
                        ChunkOp::Accumulate,
                        vec![pair[0], pair[1]],
                    ))
                })
                .collect();
            // 8 second-level
            let level2: Vec<usize> = level1
                .chunks(2)
                .map(|pair| {
                    graph.add_node(ChunkNode::operation(
                        ChunkOp::Accumulate,
                        vec![pair[0], pair[1]],
                    ))
                })
                .collect();
            // 4 third-level
            let level3: Vec<usize> = level2
                .chunks(2)
                .map(|pair| {
                    graph.add_node(ChunkNode::operation(
                        ChunkOp::Accumulate,
                        vec![pair[0], pair[1]],
                    ))
                })
                .collect();
            // 2 fourth-level
            let level4: Vec<usize> = level3
                .chunks(2)
                .map(|pair| {
                    graph.add_node(ChunkNode::operation(
                        ChunkOp::Accumulate,
                        vec![pair[0], pair[1]],
                    ))
                })
                .collect();
            // 1 sink
            let _ = graph.add_node(ChunkNode::operation(
                ChunkOp::Accumulate,
                vec![level4[0], level4[1]],
            ));
            black_box(graph.len())
        });
    });

    group.finish();
}

fn bench_chunk_graph_topo_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_graph_topological_sort");

    for node_count in [7usize, 31, 63] {
        // Build graph once, benchmark only the sort
        let mut graph = ChunkGraph::new();
        let input_count = node_count.div_ceil(2);
        let inputs: Vec<usize> = (0..input_count)
            .map(|i| graph.add_node(ChunkNode::input("X", vec![i])))
            .collect();
        // Chain of accumulations
        let mut prev = inputs[0];
        for &inp in inputs.iter().skip(1) {
            prev = graph.add_node(ChunkNode::operation(ChunkOp::Accumulate, vec![prev, inp]));
        }

        group.bench_with_input(BenchmarkId::from_parameter(graph.len()), &graph, |b, g| {
            b.iter(|| black_box(g.topological_order().expect("topological_order failed")));
        });
    }

    group.finish();
}

// ─── 2. Memory manager alloc/free cycle ─────────────────────────────────────

fn bench_memory_manager_alloc_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_manager_alloc_free");

    // Sizes in elements (f64): 100 = 800 B, 1_000 = 8 KB, 10_000 = 80 KB
    let shapes: &[(&str, &[usize])] = &[
        ("100x1", &[100, 1]),
        ("100x100", &[100, 100]),
        ("10x10x10", &[10, 10, 10]),
    ];

    for (label, shape) in shapes {
        let n: usize = shape.iter().product();
        let size_bytes = n * std::mem::size_of::<f64>();
        group.throughput(Throughput::Bytes(size_bytes as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), label, |b, _| {
            b.iter_with_setup(
                || {
                    // Fresh manager per iteration so we don't saturate it.
                    // 16 MB should be ample for all shapes.
                    MemoryManager::new().max_memory_mb(16).auto_spill(false)
                },
                |mut mgr| {
                    let tensor = make_tensor(shape);
                    mgr.register_chunk("bench_chunk", tensor, AccessPattern::ReadOnce)
                        .expect("register_chunk failed");
                    // Access it once
                    let _ = mgr
                        .access_chunk("bench_chunk")
                        .expect("access_chunk failed");
                    // Decref → reference count to 0
                    mgr.decref("bench_chunk").expect("decref failed");
                    black_box(mgr.current_memory())
                },
            );
        });
    }

    group.finish();
}

// ─── 3. Prefetch queue push/pop throughput ───────────────────────────────────

fn bench_prefetch_queue_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_queue_throughput");

    for &n in &[10usize, 100, 500] {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("schedule_and_clear", n),
            &n,
            |b, &count| {
                let mut prefetcher = Prefetcher::new()
                    .strategy(PrefetchStrategy::Sequential)
                    .queue_size(count + 1);

                // Pre-build the name strings outside the timing loop.
                let names: Vec<String> = (0..count).map(|i| format!("chunk_{}", i)).collect();

                b.iter(|| {
                    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                    let scheduled = prefetcher
                        .schedule_prefetch(refs)
                        .expect("schedule_prefetch failed");
                    prefetcher.clear();
                    black_box(scheduled)
                });
            },
        );

        // Separately benchmark `add_prefetched` + `get` (the pop side).
        group.bench_with_input(BenchmarkId::new("add_and_get", n), &n, |b, &count| {
            let tensor = make_tensor(&[10, 10]);

            b.iter(|| {
                let mut pf = Prefetcher::new().queue_size(count + 1);
                for i in 0..count {
                    pf.add_prefetched(&format!("c_{}", i), tensor.clone());
                }
                let mut found = 0usize;
                for i in 0..count {
                    if pf.get(&format!("c_{}", i)).is_some() {
                        found += 1;
                    }
                }
                black_box(found)
            });
        });
    }

    group.finish();
}

// ─── 4. Working set predictor record_access throughput ───────────────────────

fn bench_working_set_record_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("working_set_record_access");

    for &n in &[50usize, 200, 1000] {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("adaptive", n), &n, |b, &count| {
            // Pre-build chunk IDs to avoid timing string formatting.
            let ids: Vec<String> = (0..count).map(|i| format!("chunk_{}", i % 20)).collect();

            b.iter(|| {
                let mut predictor = WorkingSetPredictor::new()
                    .window_size(count)
                    .prediction_mode(PredictionMode::Adaptive);

                for id in &ids {
                    predictor.record_access(id, 8192);
                }
                black_box(
                    predictor
                        .predict_working_set(5)
                        .expect("predict failed")
                        .len(),
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("frequency", n), &n, |b, &count| {
            let ids: Vec<String> = (0..count).map(|i| format!("chunk_{}", i % 20)).collect();

            b.iter(|| {
                let mut predictor = WorkingSetPredictor::new()
                    .window_size(count)
                    .prediction_mode(PredictionMode::Frequency);

                for id in &ids {
                    predictor.record_access(id, 8192);
                }
                black_box(
                    predictor
                        .predict_working_set(5)
                        .expect("predict failed")
                        .len(),
                )
            });
        });
    }

    group.finish();
}

// ─── 5. Chunk hash computation: CRC32 vs XXHash vs Blake3 ───────────────────

fn bench_chunk_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_hash");

    // Payload sizes: 1 KB, 10 KB, 100 KB
    let sizes_bytes: &[usize] = &[1_024, 10_240, 102_400];

    for &size in sizes_bytes {
        let payload = make_payload_bytes(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("crc32", size), &payload, |b, data| {
            b.iter(|| black_box(compute_checksum(ChecksumAlgorithm::Crc32, black_box(data))));
        });

        group.bench_with_input(BenchmarkId::new("xxhash64", size), &payload, |b, data| {
            b.iter(|| {
                black_box(compute_checksum(
                    ChecksumAlgorithm::XxHash64,
                    black_box(data),
                ))
            });
        });

        group.bench_with_input(BenchmarkId::new("blake3", size), &payload, |b, data| {
            b.iter(|| black_box(compute_checksum(ChecksumAlgorithm::Blake3, black_box(data))));
        });
    }

    group.finish();
}

// ─── 6. Memory tier latency lookup ──────────────────────────────────────────

fn bench_memory_tier_lookup(c: &mut Criterion) {
    use tenrso_ooc::{MemoryTier, TierAccessPattern, TieredMemoryManager};

    let mut group = c.benchmark_group("memory_tier_lookup");

    // Benchmark the cost of registering a tiny chunk and immediately retrieving
    // it from RAM (hot path) versus checking stats (cold metadata path).
    let shape = &[10usize, 10];

    group.bench_function("register_get_ram_hot", |b| {
        b.iter_with_setup(
            || TieredMemoryManager::new().auto_migration(false),
            |mut mgr| {
                let tensor = make_tensor(shape);
                mgr.register_chunk("hot", tensor, TierAccessPattern::Temporal)
                    .expect("register_chunk failed");
                let fetched = mgr.get_chunk("hot").expect("get_chunk failed");
                black_box(fetched.len())
            },
        );
    });

    group.bench_function("tier_latency_us_lookup", |b| {
        // Trivial method — measures pure enum match overhead.
        let tiers = [MemoryTier::Ram, MemoryTier::Ssd, MemoryTier::Disk];
        b.iter(|| {
            let mut total = 0u64;
            for &t in &tiers {
                total += black_box(t.latency_us());
            }
            black_box(total)
        });
    });

    group.finish();
}

// ─── criterion wiring ────────────────────────────────────────────────────────

criterion_group!(
    ooc_micro,
    bench_chunk_graph_construction,
    bench_chunk_graph_topo_sort,
    bench_memory_manager_alloc_free,
    bench_prefetch_queue_throughput,
    bench_working_set_record_access,
    bench_chunk_hash,
    bench_memory_tier_lookup,
);
criterion_main!(ooc_micro);
