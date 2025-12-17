use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::array;
use std::hint::black_box;
use tenrso_core::DenseND;
use tenrso_exec::executor::CpuExecutor;

/// Benchmark: Pooled vs non-pooled element-wise operations
///
/// Compares performance of pooled_add_f32 vs standard element-wise add.
/// This shows the benefit of buffer reuse for repeated operations.
fn bench_pooled_vs_nonpooled_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("pooled_vs_nonpooled_add");

    let sizes = [
        ("small_100", vec![10, 10]),
        ("medium_10k", vec![100, 100]),
        ("large_100k", vec![316, 316]),
    ];

    for (name, shape) in sizes.iter() {
        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        let a: Vec<f32> = (0..total_elements).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 2.0).collect();

        let a_tensor = DenseND::from_vec(a.clone(), shape).unwrap();
        let b_tensor = DenseND::from_vec(b.clone(), shape).unwrap();

        // Non-pooled (standard ndarray operations)
        group.bench_with_input(
            BenchmarkId::new("nonpooled", name),
            &(&a_tensor, &b_tensor),
            |bench, (a, b)| {
                bench.iter(|| {
                    // Use mapv for element-wise addition
                    let result = a.view().mapv(|x| x) + b.view().mapv(|y| y);
                    black_box(result);
                });
            },
        );

        // Pooled (using memory pool)
        group.bench_with_input(
            BenchmarkId::new("pooled", name),
            &(&a_tensor, &b_tensor),
            |bench, (a, b)| {
                let mut executor = CpuExecutor::new();
                // Warm up the pool
                let _ = executor.pooled_add_f32(a, b).unwrap();

                bench.iter(|| {
                    let result = executor.pooled_add_f32(a, b).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Pooled matmul vs standard matmul
///
/// Compares pooled_matmul_f32 vs ndarray's dot product.
/// This shows benefits for operations with intermediate buffers.
fn bench_pooled_vs_nonpooled_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("pooled_vs_nonpooled_matmul");

    let sizes = [
        ("small_16x16", (16, 16, 16)),
        ("medium_64x64", (64, 64, 64)),
        ("large_128x128", (128, 128, 128)),
    ];

    for (name, (m, k, n)) in sizes.iter() {
        let throughput = (m * k + k * n + m * n) as u64;
        group.throughput(Throughput::Elements(throughput));

        let a_data: Vec<f32> = (0..(m * k)).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..(k * n)).map(|i| i as f32).collect();

        let a_tensor = DenseND::from_vec(a_data, &[*m, *k]).unwrap();
        let b_tensor = DenseND::from_vec(b_data, &[*k, *n]).unwrap();

        // Non-pooled (manual matmul for fair comparison)
        group.bench_with_input(
            BenchmarkId::new("nonpooled", name),
            &(&a_tensor, &b_tensor),
            |bench, (a, b)| {
                bench.iter(|| {
                    let a_view = a.view();
                    let b_view = b.view();
                    let mut output = vec![0.0f32; m * n];

                    for i in 0..*m {
                        for j in 0..*n {
                            let mut sum = 0.0;
                            for kk in 0..*k {
                                sum += a_view[[i, kk]] * b_view[[kk, j]];
                            }
                            output[i * n + j] = sum;
                        }
                    }
                    black_box(output);
                });
            },
        );

        // Pooled (using memory pool)
        group.bench_with_input(
            BenchmarkId::new("pooled", name),
            &(&a_tensor, &b_tensor),
            |bench, (a, b)| {
                let mut executor = CpuExecutor::new();
                // Warm up the pool
                let _ = executor.pooled_matmul_f32(a, b).unwrap();

                bench.iter(|| {
                    let result = executor.pooled_matmul_f32(a, b).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch processing with pooled buffers
///
/// Measures the benefit of buffer reuse across multiple operations.
/// This is the most common scenario where pooling helps.
fn bench_batch_processing_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing_hit_rate");

    let batch_sizes = [10, 50, 100];

    for &batch_size in batch_sizes.iter() {
        let total_ops = batch_size as u64 * 64 * 64;
        group.throughput(Throughput::Elements(total_ops));

        // Create batch of tensors
        let tensors: Vec<_> = (0..batch_size)
            .map(|i| {
                DenseND::from_array(
                    array![[i as f32, i as f32 + 1.0], [i as f32 + 2.0, i as f32 + 3.0]].into_dyn(),
                )
            })
            .collect();

        // Without pooling (allocate each time)
        group.bench_with_input(
            BenchmarkId::new("nonpooled", batch_size),
            &tensors,
            |bench, tensors| {
                bench.iter(|| {
                    let results: Vec<_> = tensors
                        .iter()
                        .map(|t| {
                            let mut output = vec![0.0f32; 4];
                            for (i, val) in t.view().iter().enumerate() {
                                output[i] = *val * 2.0;
                            }
                            black_box(output);
                        })
                        .collect();
                    black_box(results);
                });
            },
        );

        // With pooling (reuse buffer)
        group.bench_with_input(
            BenchmarkId::new("pooled", batch_size),
            &tensors,
            |bench, tensors| {
                bench.iter(|| {
                    let mut executor = CpuExecutor::new();
                    let results = executor
                        .batch_process_f32(tensors, |buffer, tensor| {
                            for (i, val) in tensor.view().iter().enumerate() {
                                buffer[i] = *val * 2.0;
                            }
                            Ok(tensor.clone())
                        })
                        .unwrap();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: RAII-style buffer management overhead
///
/// Measures the overhead of with_pooled_buffer_* helpers compared to
/// manual acquire/release.
fn bench_raii_buffer_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("raii_buffer_overhead");

    let sizes = [
        ("small_1k", 1024),
        ("medium_10k", 10_240),
        ("large_100k", 102_400),
    ];

    for (name, size) in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Manual acquire/release
        group.bench_with_input(BenchmarkId::new("manual", name), size, |bench, &size| {
            let mut executor = CpuExecutor::new();
            // Warm up
            let buffer = executor.acquire_f32(&[size]);
            executor.release_f32(&[size], buffer);

            bench.iter(|| {
                let mut buffer = executor.acquire_f32(&[size]);
                for val in buffer.iter_mut() {
                    *val = 42.0;
                }
                black_box(&buffer);
                executor.release_f32(&[size], buffer);
            });
        });

        // RAII-style with_pooled_buffer
        group.bench_with_input(BenchmarkId::new("raii", name), size, |bench, &size| {
            let mut executor = CpuExecutor::new();
            // Warm up
            let _ = executor.with_pooled_buffer_f32(&[size], |buffer| Ok(buffer.len()));

            bench.iter(|| {
                let result = executor
                    .with_pooled_buffer_f32(&[size], |mut buffer| {
                        for val in buffer.iter_mut() {
                            *val = 42.0;
                        }
                        Ok(buffer.len())
                    })
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Pool hit rate impact on latency
///
/// Measures the latency difference between pool hits and misses.
fn bench_pool_hit_vs_miss_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_hit_vs_miss_latency");

    let shapes = [
        ("small_1k", vec![1024]),
        ("medium_10k", vec![10_240]),
        ("large_100k", vec![102_400]),
    ];

    for (name, shape) in shapes.iter() {
        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        // Pool miss (first allocation)
        group.bench_with_input(
            BenchmarkId::new("cold_miss", name),
            shape,
            |bench, shape| {
                bench.iter(|| {
                    let mut executor = CpuExecutor::new();
                    let result = executor
                        .with_pooled_buffer_f32(shape, |buffer| {
                            black_box(&buffer);
                            Ok(buffer.len())
                        })
                        .unwrap();
                    black_box(result);
                });
            },
        );

        // Pool hit (subsequent allocations)
        group.bench_with_input(BenchmarkId::new("warm_hit", name), shape, |bench, shape| {
            let mut executor = CpuExecutor::new();
            // Warm up the pool
            let _ = executor.with_pooled_buffer_f32(shape, |buffer| Ok(buffer.len()));

            bench.iter(|| {
                let result = executor
                    .with_pooled_buffer_f32(shape, |buffer| {
                        black_box(&buffer);
                        Ok(buffer.len())
                    })
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Realistic ML training loop simulation
///
/// Simulates a typical ML forward/backward pass with multiple operations
/// to show cumulative pooling benefits.
fn bench_ml_training_loop_pooled(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_training_loop_pooled");

    // Typical ML layer shapes
    let forward_shapes = [
        vec![32, 128], // Batch 32, hidden 128
        vec![32, 256], // Batch 32, hidden 256
        vec![32, 128], // Batch 32, hidden 128
    ];

    // Without pooling
    group.bench_function("nonpooled", |bench| {
        bench.iter(|| {
            for shape in forward_shapes.iter() {
                let buffer: Vec<f32> = vec![0.0; shape.iter().product()];
                black_box(&buffer);
            }
            // Backward pass
            for shape in forward_shapes.iter().rev() {
                let buffer: Vec<f32> = vec![0.0; shape.iter().product()];
                black_box(&buffer);
            }
        });
    });

    // With pooling
    group.bench_function("pooled", |bench| {
        bench.iter(|| {
            let mut executor = CpuExecutor::new();
            // Forward pass
            for shape in forward_shapes.iter() {
                let _ = executor.with_pooled_buffer_f32(shape, |buffer| {
                    black_box(&buffer);
                    Ok(())
                });
            }
            // Backward pass (shapes in reverse, should hit pool)
            for shape in forward_shapes.iter().rev() {
                let _ = executor.with_pooled_buffer_f32(shape, |buffer| {
                    black_box(&buffer);
                    Ok(())
                });
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pooled_vs_nonpooled_add,
    bench_pooled_vs_nonpooled_matmul,
    bench_batch_processing_hit_rate,
    bench_raii_buffer_overhead,
    bench_pool_hit_vs_miss_latency,
    bench_ml_training_loop_pooled,
);
criterion_main!(benches);
