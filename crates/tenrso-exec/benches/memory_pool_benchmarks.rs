use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_exec::executor::CpuExecutor;

/// Benchmark: Memory pool overhead - acquire and release operations
///
/// Measures the overhead of pool operations vs direct allocation
fn bench_pool_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_overhead");

    let shapes = [
        ("small_10", vec![10]),
        ("medium_100", vec![100]),
        ("large_1000", vec![1000]),
        ("matrix_64x64", vec![64, 64]),
        ("tensor_16x16x16", vec![16, 16, 16]),
    ];

    for (name, shape) in shapes.iter() {
        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        // Benchmark WITH pooling
        group.bench_with_input(BenchmarkId::new("pooled", name), &shape, |b, shape| {
            let mut executor = CpuExecutor::new().with_memory_pool(true);
            b.iter(|| {
                let buffer = executor.acquire_f32(shape);
                executor.release_f32(shape, buffer);
            });
        });

        // Benchmark WITHOUT pooling (direct allocation)
        group.bench_with_input(BenchmarkId::new("direct", name), &shape, |b, shape| {
            b.iter(|| {
                let total_size: usize = shape.iter().product();
                let buffer: Vec<f32> = vec![0.0; total_size];
                black_box(buffer);
            });
        });
    }

    group.finish();
}

/// Benchmark: Pool hit rate with repeated allocations
///
/// Measures performance when reusing buffers from the pool
fn bench_pool_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_hit_rate");

    let shapes = [
        ("small_10", vec![10]),
        ("medium_100", vec![100]),
        ("large_1000", vec![1000]),
        ("matrix_64x64", vec![64, 64]),
    ];

    for (name, shape) in shapes.iter() {
        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        // Cold cache (first allocation - always miss)
        group.bench_with_input(BenchmarkId::new("cold_cache", name), &shape, |b, shape| {
            b.iter(|| {
                let mut executor = CpuExecutor::new().with_memory_pool(true);
                let buffer = executor.acquire_f32(shape);
                black_box(&buffer);
                executor.release_f32(shape, buffer);
            });
        });

        // Warm cache (subsequent allocations - should hit)
        group.bench_with_input(BenchmarkId::new("warm_cache", name), &shape, |b, shape| {
            let mut executor = CpuExecutor::new().with_memory_pool(true);
            // Warm up the pool
            let buffer = executor.acquire_f32(shape);
            executor.release_f32(shape, buffer);

            b.iter(|| {
                let buffer = executor.acquire_f32(shape);
                black_box(&buffer);
                executor.release_f32(shape, buffer);
            });
        });
    }

    group.finish();
}

/// Benchmark: Multiple shapes in pool
///
/// Measures performance when managing multiple shape signatures
fn bench_pool_multiple_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_multiple_shapes");

    let shape_sets = [
        ("3_shapes", vec![vec![10], vec![20], vec![30]]),
        (
            "5_shapes",
            vec![vec![10], vec![20], vec![30], vec![40], vec![50]],
        ),
        (
            "10_shapes",
            vec![
                vec![10],
                vec![20],
                vec![30],
                vec![40],
                vec![50],
                vec![60],
                vec![70],
                vec![80],
                vec![90],
                vec![100],
            ],
        ),
    ];

    for (name, shapes) in shape_sets.iter() {
        group.bench_with_input(
            BenchmarkId::new("acquire_release", name),
            &shapes,
            |b, shapes| {
                let mut executor = CpuExecutor::new().with_memory_pool(true);
                // Warm up the pool with all shapes
                for shape in shapes.iter() {
                    let buffer = executor.acquire_f32(shape);
                    executor.release_f32(shape, buffer);
                }

                b.iter(|| {
                    for shape in shapes.iter() {
                        let buffer = executor.acquire_f32(shape);
                        black_box(&buffer);
                        executor.release_f32(shape, buffer);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Pool contention (simulating parallel workload)
///
/// Measures performance with frequent acquire/release cycles
fn bench_pool_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_contention");

    let iterations = [10, 50, 100, 500];
    let shape = vec![64, 64];

    for &iters in iterations.iter() {
        group.throughput(Throughput::Elements((iters * 64 * 64) as u64));

        // With pooling
        group.bench_with_input(BenchmarkId::new("pooled", iters), &iters, |b, &iters| {
            let mut executor = CpuExecutor::new().with_memory_pool(true);
            b.iter(|| {
                for _ in 0..iters {
                    let buffer = executor.acquire_f32(&shape);
                    black_box(&buffer);
                    executor.release_f32(&shape, buffer);
                }
            });
        });

        // Without pooling
        group.bench_with_input(BenchmarkId::new("direct", iters), &iters, |b, &iters| {
            b.iter(|| {
                for _ in 0..iters {
                    let buffer: Vec<f32> = vec![0.0; 64 * 64];
                    black_box(&buffer);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark: Pool size scaling
///
/// Measures performance impact as pool size grows
fn bench_pool_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_size_scaling");

    let pool_sizes = [1, 4, 8, 16]; // 16 is MAX_POOL_SIZE
    let shape = vec![64, 64];

    for &size in pool_sizes.iter() {
        group.bench_with_input(
            BenchmarkId::new("acquire_from_pool", size),
            &size,
            |b, &size| {
                let mut executor = CpuExecutor::new().with_memory_pool(true);
                // Fill pool to target size
                let mut buffers = Vec::new();
                for _ in 0..size {
                    let buffer = executor.acquire_f32(&shape);
                    buffers.push(buffer);
                }
                for buffer in buffers {
                    executor.release_f32(&shape, buffer);
                }

                b.iter(|| {
                    let buffer = executor.acquire_f32(&shape);
                    black_box(&buffer);
                    executor.release_f32(&shape, buffer);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Type-specific pools (f32 vs f64)
///
/// Compares performance between f32 and f64 pools
fn bench_type_specific_pools(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_specific_pools");

    let shapes = [
        ("small_100", vec![100]),
        ("medium_1000", vec![1000]),
        ("large_10000", vec![10000]),
    ];

    for (name, shape) in shapes.iter() {
        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        // f32 pool
        group.bench_with_input(BenchmarkId::new("f32", name), &shape, |b, shape| {
            let mut executor = CpuExecutor::new().with_memory_pool(true);
            // Warm up
            let buffer = executor.acquire_f32(shape);
            executor.release_f32(shape, buffer);

            b.iter(|| {
                let buffer = executor.acquire_f32(shape);
                black_box(&buffer);
                executor.release_f32(shape, buffer);
            });
        });

        // f64 pool
        group.bench_with_input(BenchmarkId::new("f64", name), &shape, |b, shape| {
            let mut executor = CpuExecutor::new().with_memory_pool(true);
            // Warm up
            let buffer = executor.acquire_f64(shape);
            executor.release_f64(shape, buffer);

            b.iter(|| {
                let buffer = executor.acquire_f64(shape);
                black_box(&buffer);
                executor.release_f64(shape, buffer);
            });
        });
    }

    group.finish();
}

/// Benchmark: Realistic workload simulation
///
/// Simulates a realistic ML training loop with mixed shapes
fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workload");

    // Typical ML shapes: batch size, hidden dims, etc.
    let ml_shapes = [
        vec![32, 128],  // Small batch, hidden layer
        vec![32, 512],  // Small batch, larger hidden
        vec![64, 256],  // Medium batch
        vec![128, 128], // Larger batch, smaller hidden
        vec![32, 1024], // Large hidden layer
    ];

    // With pooling
    group.bench_function("ml_training_loop_pooled", |b| {
        let mut executor = CpuExecutor::new().with_memory_pool(true);
        b.iter(|| {
            // Simulate forward pass
            for shape in ml_shapes.iter() {
                let buffer = executor.acquire_f32(shape);
                black_box(&buffer);
                executor.release_f32(shape, buffer);
            }
            // Simulate backward pass (reverse order)
            for shape in ml_shapes.iter().rev() {
                let buffer = executor.acquire_f32(shape);
                black_box(&buffer);
                executor.release_f32(shape, buffer);
            }
        });
    });

    // Without pooling
    group.bench_function("ml_training_loop_direct", |b| {
        b.iter(|| {
            // Simulate forward pass
            for shape in ml_shapes.iter() {
                let total_size: usize = shape.iter().product();
                let buffer: Vec<f32> = vec![0.0; total_size];
                black_box(&buffer);
            }
            // Simulate backward pass (reverse order)
            for shape in ml_shapes.iter().rev() {
                let total_size: usize = shape.iter().product();
                let buffer: Vec<f32> = vec![0.0; total_size];
                black_box(&buffer);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pool_overhead,
    bench_pool_hit_rate,
    bench_pool_multiple_shapes,
    bench_pool_contention,
    bench_pool_size_scaling,
    bench_type_specific_pools,
    bench_realistic_workload,
);
criterion_main!(benches);
