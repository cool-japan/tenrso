//! Benchmarks for large tensor out-of-core operations
//!
//! These benchmarks test performance of OoC operations on tensors
//! larger than typical memory constraints.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::Array;
use scirs2_core::random::{rngs::StdRng, RngCore, SeedableRng};
use std::hint::black_box;
use tenrso_core::DenseND;
use tenrso_ooc::{ChunkSpec, PrefetchStrategy, Prefetcher, StreamConfig, StreamingExecutor};

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

/// Benchmark streaming execution with memory constraints
fn bench_streaming_with_memory_limits(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_memory_limits");

    for mem_mb in [16, 32, 64] {
        let config = StreamConfig::new()
            .max_memory_mb(mem_mb)
            .chunk_size(vec![32])
            .enable_spill(true);

        let mut executor = StreamingExecutor::new(config);

        group.bench_with_input(
            BenchmarkId::new("memory_mb", mem_mb),
            &mem_mb,
            |bencher, _mem_mb| {
                bencher.iter(|| {
                    let a = DenseND::<f64>::zeros(&[128, 128]);
                    let b = DenseND::<f64>::zeros(&[128, 128]);
                    let result = executor.add_chunked(&a, &b);
                    black_box(result)
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

/// Benchmark FMA operations (real-world ML workload)
fn bench_fma_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fma_operations");

    for size in [256, 512, 1024] {
        let config = StreamConfig::new()
            .max_memory_mb(64)
            .chunk_size(vec![128])
            .enable_profiling(false);

        let mut executor = StreamingExecutor::new(config);

        // Create random tensors for realistic workload
        let mut rng = StdRng::seed_from_u64(42);
        let a_data: Vec<f64> = (0..size * size)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();
        let b_data: Vec<f64> = (0..size * size)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();
        let c_data: Vec<f64> = (0..size * size)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();

        let a = DenseND::from_array(Array::from_shape_vec(vec![size, size], a_data).unwrap());
        let b = DenseND::from_array(Array::from_shape_vec(vec![size, size], b_data).unwrap());
        let c = DenseND::from_array(Array::from_shape_vec(vec![size, size], c_data).unwrap());

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let result = executor.fma_chunked(&a, &b, &c);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark element-wise min/max operations (real-world data processing)
fn bench_min_max_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_max_operations");

    for size in [512, 1024, 2048] {
        let config = StreamConfig::new()
            .max_memory_mb(64)
            .chunk_size(vec![256])
            .enable_profiling(false);

        let mut executor = StreamingExecutor::new(config);

        let mut rng = StdRng::seed_from_u64(123);
        let a_data: Vec<f64> = (0..size * size)
            .map(|_| (rng.next_u64() % 1000) as f64)
            .collect();
        let b_data: Vec<f64> = (0..size * size)
            .map(|_| (rng.next_u64() % 1000) as f64)
            .collect();

        let a = DenseND::from_array(Array::from_shape_vec(vec![size, size], a_data).unwrap());
        let b = DenseND::from_array(Array::from_shape_vec(vec![size, size], b_data).unwrap());

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("min", size), &size, |bencher, &_size| {
            bencher.iter(|| {
                let result = executor.min_chunked(&a, &b);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("max", size), &size, |bencher, &_size| {
            bencher.iter(|| {
                let result = executor.max_chunked(&a, &b);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark adaptive chunking based on memory pressure
fn bench_adaptive_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_chunking");

    let size = 1024;

    for mem_mb in [16, 32, 64, 128] {
        let config = StreamConfig::new()
            .max_memory_mb(mem_mb)
            .enable_profiling(false);

        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::zeros(&[size, size]);
        let b = DenseND::<f64>::zeros(&[size, size]);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(mem_mb),
            &mem_mb,
            |bencher, &_mem_mb| {
                bencher.iter(|| {
                    let result = executor.add_chunked(&a, &b);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel vs sequential execution
#[cfg(feature = "parallel")]
fn bench_parallel_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_execution");

    let size = 1024;

    for enable_parallel in [false, true] {
        let config = StreamConfig::new()
            .max_memory_mb(128)
            .chunk_size(vec![128])
            .enable_parallel(enable_parallel)
            .num_threads(4)
            .enable_profiling(false);

        let mut executor = StreamingExecutor::new(config);

        let mut rng = StdRng::seed_from_u64(456);
        let a_data: Vec<f64> = (0..size * size)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();
        let b_data: Vec<f64> = (0..size * size)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();

        let a = DenseND::from_array(Array::from_shape_vec(vec![size, size], a_data).unwrap());
        let b = DenseND::from_array(Array::from_shape_vec(vec![size, size], b_data).unwrap());

        group.throughput(Throughput::Elements((size * size) as u64));

        let label = if enable_parallel {
            "parallel"
        } else {
            "sequential"
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &label,
            |bencher, &_label| {
                bencher.iter(|| {
                    let result = executor.multiply_chunked(&a, &b);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark realistic deep learning workload (batch processing)
fn bench_deep_learning_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_learning_workload");

    // Simulate mini-batch processing: (batch_size, features)
    for batch_size in [32, 64, 128] {
        let features = 512;

        let config = StreamConfig::new()
            .max_memory_mb(64)
            .chunk_size(vec![16, 256])
            .enable_profiling(false);

        let mut executor = StreamingExecutor::new(config);

        let mut rng = StdRng::seed_from_u64(789);
        let input_data: Vec<f64> = (0..batch_size * features)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();
        let weight_data: Vec<f64> = (0..features * features)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();

        let input = DenseND::from_array(
            Array::from_shape_vec(vec![batch_size, features], input_data).unwrap(),
        );
        let weights = DenseND::from_array(
            Array::from_shape_vec(vec![features, features], weight_data).unwrap(),
        );

        group.throughput(Throughput::Elements((batch_size * features) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |bencher, &_batch_size| {
                bencher.iter(|| {
                    let result = executor.matmul_chunked(&input, &weights, Some(128));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark configuration recommendation system
fn bench_config_recommendation(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_recommendation");

    for size in [512, 1024, 2048] {
        let config = StreamConfig::new().max_memory_mb(128);
        let executor = StreamingExecutor::new(config);

        let a_shape = vec![size, size];
        let b_shape = vec![size, size];

        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let result = executor.recommend_config("matmul", &[&a_shape, &b_shape]);
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("elementwise", size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let result = executor.recommend_config("elementwise", &[&a_shape]);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_chunked_matmul,
    bench_chunk_iteration,
    bench_streaming_with_memory_limits,
    bench_prefetch_strategies,
    bench_chunked_elementwise,
    bench_chunk_graph,
    bench_fma_operations,
    bench_min_max_operations,
    bench_adaptive_chunking,
    bench_parallel_execution,
    bench_deep_learning_workload,
    bench_config_recommendation,
);

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_chunked_matmul,
    bench_chunk_iteration,
    bench_streaming_with_memory_limits,
    bench_prefetch_strategies,
    bench_chunked_elementwise,
    bench_chunk_graph,
    bench_fma_operations,
    bench_min_max_operations,
    bench_adaptive_chunking,
    bench_deep_learning_workload,
    bench_config_recommendation,
);

criterion_main!(benches);
