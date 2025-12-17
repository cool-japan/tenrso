//! Comprehensive benchmarks for optimization features
//!
//! This benchmark suite measures the performance impact of:
//! - SIMD-accelerated element-wise operations
//! - Tiled/blocked reductions for large tensors
//! - Vectorized broadcasting optimizations
//!
//! Each benchmark compares optimized vs unoptimized performance to quantify speedups.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{BinaryOp, CpuExecutor, ElemOp, ReduceOp, TenrsoExecutor};

/// Benchmark SIMD element-wise operations vs standard implementation
fn bench_simd_element_wise(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_element_wise");

    // Test sizes: small (no SIMD), medium (SIMD threshold), large (SIMD optimal)
    for size in [512, 1024, 4096, 16384, 65536].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        // Create test tensor
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 1.0).collect();
        let tensor = DenseND::from_vec(data, &[n]).unwrap();
        let handle = TensorHandle::from_dense_auto(tensor);

        // Benchmark with SIMD enabled
        group.bench_with_input(BenchmarkId::new("simd_neg", n), &handle, |b, handle| {
            let mut executor = CpuExecutor::new().with_simd(true);
            b.iter(|| {
                let result = executor.elem_op(ElemOp::Neg, black_box(handle)).unwrap();
                black_box(result);
            });
        });

        // Benchmark without SIMD (baseline)
        group.bench_with_input(BenchmarkId::new("standard_neg", n), &handle, |b, handle| {
            let mut executor = CpuExecutor::new().with_simd(false);
            b.iter(|| {
                let result = executor.elem_op(ElemOp::Neg, black_box(handle)).unwrap();
                black_box(result);
            });
        });

        // Benchmark expensive operations (exp, sin, etc.)
        let data_exp: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let tensor_exp = DenseND::from_vec(data_exp, &[n]).unwrap();
        let handle_exp = TensorHandle::from_dense_auto(tensor_exp);

        group.bench_with_input(BenchmarkId::new("simd_exp", n), &handle_exp, |b, handle| {
            let mut executor = CpuExecutor::new().with_simd(true);
            b.iter(|| {
                let result = executor.elem_op(ElemOp::Exp, black_box(handle)).unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("standard_exp", n),
            &handle_exp,
            |b, handle| {
                let mut executor = CpuExecutor::new().with_simd(false);
                b.iter(|| {
                    let result = executor.elem_op(ElemOp::Exp, black_box(handle)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD binary operations
fn bench_simd_binary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_binary_ops");

    for size in [1024, 4096, 16384, 65536].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let data_a: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 1.0).collect();
        let data_b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.002 + 2.0).collect();
        let tensor_a = DenseND::from_vec(data_a, &[n]).unwrap();
        let tensor_b = DenseND::from_vec(data_b, &[n]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(tensor_a);
        let handle_b = TensorHandle::from_dense_auto(tensor_b);

        // Add operation
        group.bench_with_input(
            BenchmarkId::new("simd_add", n),
            &(handle_a.clone(), handle_b.clone()),
            |b, (ha, hb)| {
                let mut executor = CpuExecutor::new().with_simd(true);
                b.iter(|| {
                    let result = executor
                        .binary_op(BinaryOp::Add, black_box(ha), black_box(hb))
                        .unwrap();
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("standard_add", n),
            &(handle_a.clone(), handle_b.clone()),
            |b, (ha, hb)| {
                let mut executor = CpuExecutor::new().with_simd(false);
                b.iter(|| {
                    let result = executor
                        .binary_op(BinaryOp::Add, black_box(ha), black_box(hb))
                        .unwrap();
                    black_box(result);
                });
            },
        );

        // Multiply operation
        group.bench_with_input(
            BenchmarkId::new("simd_mul", n),
            &(handle_a.clone(), handle_b.clone()),
            |b, (ha, hb)| {
                let mut executor = CpuExecutor::new().with_simd(true);
                b.iter(|| {
                    let result = executor
                        .binary_op(BinaryOp::Mul, black_box(ha), black_box(hb))
                        .unwrap();
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("standard_mul", n),
            &(handle_a.clone(), handle_b.clone()),
            |b, (ha, hb)| {
                let mut executor = CpuExecutor::new().with_simd(false);
                b.iter(|| {
                    let result = executor
                        .binary_op(BinaryOp::Mul, black_box(ha), black_box(hb))
                        .unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tiled reductions vs standard reductions
fn bench_tiled_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiled_reductions");

    // Test sizes: small (no tiling), medium (tiling threshold), large (tiling optimal)
    for size in [50_000, 100_000, 250_000, 500_000, 1_000_000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        // Create test tensor
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let tensor = DenseND::from_vec(data, &[n]).unwrap();
        let handle = TensorHandle::from_dense_auto(tensor);

        // Sum reduction with tiling
        group.bench_with_input(BenchmarkId::new("tiled_sum", n), &handle, |b, handle| {
            let mut executor = CpuExecutor::new().with_tiled_reductions(true);
            b.iter(|| {
                let result = executor
                    .reduce(ReduceOp::Sum, black_box(handle), black_box(&[]))
                    .unwrap();
                black_box(result);
            });
        });

        // Sum reduction without tiling
        group.bench_with_input(BenchmarkId::new("standard_sum", n), &handle, |b, handle| {
            let mut executor = CpuExecutor::new().with_tiled_reductions(false);
            b.iter(|| {
                let result = executor
                    .reduce(ReduceOp::Sum, black_box(handle), black_box(&[]))
                    .unwrap();
                black_box(result);
            });
        });

        // Mean reduction with tiling
        group.bench_with_input(BenchmarkId::new("tiled_mean", n), &handle, |b, handle| {
            let mut executor = CpuExecutor::new().with_tiled_reductions(true);
            b.iter(|| {
                let result = executor
                    .reduce(ReduceOp::Mean, black_box(handle), black_box(&[]))
                    .unwrap();
                black_box(result);
            });
        });

        // Mean reduction without tiling
        group.bench_with_input(
            BenchmarkId::new("standard_mean", n),
            &handle,
            |b, handle| {
                let mut executor = CpuExecutor::new().with_tiled_reductions(false);
                b.iter(|| {
                    let result = executor
                        .reduce(ReduceOp::Mean, black_box(handle), black_box(&[]))
                        .unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark axis-specific tiled reductions
fn bench_tiled_axis_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiled_axis_reductions");

    for size in [256, 512, 1024].iter() {
        let n = *size;
        let total_elements = n * n;
        group.throughput(Throughput::Elements(total_elements as u64));

        let data: Vec<f64> = (0..total_elements).map(|i| (i as f64) * 0.001).collect();
        let tensor = DenseND::from_vec(data, &[n, n]).unwrap();
        let handle = TensorHandle::from_dense_auto(tensor);

        // Sum along axis 0 with tiling
        group.bench_with_input(
            BenchmarkId::new("tiled_sum_axis0", n),
            &handle,
            |b, handle| {
                let mut executor = CpuExecutor::new().with_tiled_reductions(true);
                b.iter(|| {
                    let result = executor
                        .reduce(ReduceOp::Sum, black_box(handle), black_box(&[0]))
                        .unwrap();
                    black_box(result);
                });
            },
        );

        // Sum along axis 0 without tiling
        group.bench_with_input(
            BenchmarkId::new("standard_sum_axis0", n),
            &handle,
            |b, handle| {
                let mut executor = CpuExecutor::new().with_tiled_reductions(false);
                b.iter(|| {
                    let result = executor
                        .reduce(ReduceOp::Sum, black_box(handle), black_box(&[0]))
                        .unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark combined optimizations (all enabled vs all disabled)
fn bench_combined_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_optimizations");

    let size = 100_000;
    group.throughput(Throughput::Elements(size as u64));

    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001 + 1.0).collect();
    let tensor = DenseND::from_vec(data, &[size]).unwrap();
    let handle = TensorHandle::from_dense_auto(tensor.clone());

    // Full optimization pipeline
    group.bench_function("all_optimizations", |b| {
        let mut executor = CpuExecutor::new()
            .with_simd(true)
            .with_tiled_reductions(true)
            .with_vectorized_broadcast(true);

        b.iter(|| {
            // Element-wise operation
            let neg = executor.elem_op(ElemOp::Neg, black_box(&handle)).unwrap();
            // Binary operation
            let mul = executor
                .binary_op(BinaryOp::Mul, black_box(&neg), black_box(&handle))
                .unwrap();
            // Reduction
            let sum = executor
                .reduce(ReduceOp::Sum, black_box(&mul), black_box(&[]))
                .unwrap();
            black_box(sum);
        });
    });

    // No optimizations
    group.bench_function("no_optimizations", |b| {
        let mut executor = CpuExecutor::unoptimized();

        b.iter(|| {
            // Element-wise operation
            let neg = executor.elem_op(ElemOp::Neg, black_box(&handle)).unwrap();
            // Binary operation
            let mul = executor
                .binary_op(BinaryOp::Mul, black_box(&neg), black_box(&handle))
                .unwrap();
            // Reduction
            let sum = executor
                .reduce(ReduceOp::Sum, black_box(&mul), black_box(&[]))
                .unwrap();
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark optimization thresholds (verify smart dispatch)
fn bench_optimization_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_thresholds");

    // Test around SIMD threshold (1024 elements)
    for size in [512, 768, 1024, 1280, 2048].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 1.0).collect();
        let tensor = DenseND::from_vec(data, &[n]).unwrap();
        let handle = TensorHandle::from_dense_auto(tensor);

        group.bench_with_input(
            BenchmarkId::new("auto_dispatch", n),
            &handle,
            |b, handle| {
                let mut executor = CpuExecutor::new(); // All optimizations enabled
                b.iter(|| {
                    let result = executor.elem_op(ElemOp::Abs, black_box(handle)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    // Test around tiling threshold (100K elements)
    for size in [50_000, 75_000, 100_000, 150_000, 200_000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let tensor = DenseND::from_vec(data, &[n]).unwrap();
        let handle = TensorHandle::from_dense_auto(tensor);

        group.bench_with_input(
            BenchmarkId::new("auto_reduction", n),
            &handle,
            |b, handle| {
                let mut executor = CpuExecutor::new();
                b.iter(|| {
                    let result = executor
                        .reduce(ReduceOp::Sum, black_box(handle), black_box(&[]))
                        .unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory bandwidth effects (large tensor operations)
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth");
    group.sample_size(20); // Reduce sample size for very large tensors

    for size_mb in [1, 4, 16, 64].iter() {
        let elements = (size_mb * 1024 * 1024) / 8; // f64 is 8 bytes
        group.throughput(Throughput::Bytes((elements * 8) as u64));

        let data: Vec<f64> = (0..elements).map(|i| (i as f64) * 0.001).collect();
        let tensor = DenseND::from_vec(data, &[elements]).unwrap();
        let handle = TensorHandle::from_dense_auto(tensor);

        group.bench_with_input(
            BenchmarkId::new("tiled_sum_large", size_mb),
            &handle,
            |b, handle| {
                let mut executor = CpuExecutor::new().with_tiled_reductions(true);
                b.iter(|| {
                    let result = executor
                        .reduce(ReduceOp::Sum, black_box(handle), black_box(&[]))
                        .unwrap();
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_neg_large", size_mb),
            &handle,
            |b, handle| {
                let mut executor = CpuExecutor::new().with_simd(true);
                b.iter(|| {
                    let result = executor.elem_op(ElemOp::Neg, black_box(handle)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    optimization_benches,
    bench_simd_element_wise,
    bench_simd_binary_ops,
    bench_tiled_reductions,
    bench_tiled_axis_reductions,
    bench_combined_optimizations,
    bench_optimization_thresholds,
    bench_memory_bandwidth,
);
criterion_main!(optimization_benches);
