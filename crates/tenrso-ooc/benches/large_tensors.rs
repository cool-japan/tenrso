//! Benchmarks for large tensor out-of-core operations
//!
//! These benchmarks test performance of OoC operations on tensors
//! larger than typical memory constraints.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tenrso_core::DenseND;
use tenrso_ooc::{
    ChunkSpec, MemoryManager, PrefetchStrategy, Prefetcher, SpillPolicy, StreamConfig,
    StreamingExecutor,
};

/// Benchmark chunked matrix multiplication
fn bench_chunked_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_matmul");

    for size in [128, 256, 512] {
        let config = StreamConfig::new()
            .max_memory_mb(64) // Constrain memory to force chunking
            .chunk_size(vec![64]);

        let mut executor = StreamingExecutor::new(config);

        // Create test matrices
        let a = DenseND::<f64>::zeros(&[size, size]);
        let b = DenseND::<f64>::zeros(&[size, size]);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let result = executor.matmul_chunked(&a, &b, Some(64));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark chunk iteration
fn bench_chunk_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_iteration");

    for num_chunks in [16, 64, 256] {
        let spec = ChunkSpec::with_num_chunks(&[1000, 1000], &[num_chunks, num_chunks]).unwrap();

        group.throughput(Throughput::Elements((num_chunks * num_chunks) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_chunks),
            &num_chunks,
            |bencher, &_num_chunks| {
                bencher.iter(|| {
                    let chunks: Vec<_> = spec.iter().collect();
                    black_box(chunks)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory management and spill policies
#[cfg(feature = "mmap")]
fn bench_memory_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_management");

    for policy in [SpillPolicy::LRU, SpillPolicy::LargestFirst] {
        let mut manager = MemoryManager::new()
            .max_memory_mb(1) // Very tight limit to trigger spills
            .spill_policy(policy)
            .auto_spill(true);

        group.bench_with_input(
            BenchmarkId::new("spill_policy", format!("{:?}", policy)),
            &policy,
            |bencher, _policy| {
                bencher.iter(|| {
                    // Register multiple chunks
                    for i in 0..10 {
                        let tensor = DenseND::<f64>::zeros(&[100, 100]);
                        let chunk_id = format!("chunk_{}", i);
                        let _ = manager.register_chunk(
                            &chunk_id,
                            tensor,
                            tenrso_ooc::AccessPattern::ReadOnce,
                        );
                    }

                    black_box(manager.stats())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark prefetching strategies
fn bench_prefetch_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_strategies");

    for strategy in [
        PrefetchStrategy::Sequential,
        PrefetchStrategy::Adaptive,
        PrefetchStrategy::None,
    ] {
        let mut prefetcher = Prefetcher::new().strategy(strategy).queue_size(10);

        group.bench_with_input(
            BenchmarkId::new("strategy", format!("{:?}", strategy)),
            &strategy,
            |bencher, _strategy| {
                bencher.iter(|| {
                    // Simulate access pattern
                    for i in 0..20 {
                        let chunk_id = format!("chunk_{}", i);
                        prefetcher.record_access(&chunk_id);
                    }

                    black_box(prefetcher.stats())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark element-wise operations with chunking
fn bench_chunked_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_elementwise");

    for size in [512, 1024, 2048] {
        let config = StreamConfig::new().max_memory_mb(32).chunk_size(vec![256]);

        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::zeros(&[size, size]);
        let b = DenseND::<f64>::zeros(&[size, size]);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let result = executor.add_chunked(&a, &b);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark chunk graph construction and topological sort
fn bench_chunk_graph(c: &mut Criterion) {
    use tenrso_ooc::{ChunkGraph, ChunkNode, ChunkOp};

    let mut group = c.benchmark_group("chunk_graph");

    for num_nodes in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            &num_nodes,
            |bencher, &num_nodes| {
                bencher.iter(|| {
                    let mut graph = ChunkGraph::new();

                    // Build a linear chain of operations
                    let mut prev_node = graph.add_node(ChunkNode::input("A", vec![0]));

                    for i in 1..num_nodes {
                        let input_node = graph.add_node(ChunkNode::input("B", vec![i]));
                        prev_node = graph.add_node(ChunkNode::operation(
                            ChunkOp::Add,
                            vec![prev_node, input_node],
                        ));
                    }

                    let order = graph.topological_order();
                    black_box(order)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_chunked_matmul,
    bench_chunk_iteration,
    bench_prefetch_strategies,
    bench_chunked_elementwise,
    bench_chunk_graph,
);

#[cfg(feature = "mmap")]
criterion_group!(mmap_benches, bench_memory_management,);

#[cfg(feature = "mmap")]
criterion_main!(benches, mmap_benches);

#[cfg(not(feature = "mmap"))]
criterion_main!(benches);
