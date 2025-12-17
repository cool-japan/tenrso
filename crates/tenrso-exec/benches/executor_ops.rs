//! Benchmarks for tenrso-exec executor operations
//!
//! This benchmark suite measures performance of:
//! - Einsum contractions (matmul, multi-tensor)
//! - Element-wise operations
//! - Reduction operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{einsum_ex, CpuExecutor, ElemOp, ExecHints, ReduceOp, TenrsoExecutor};

/// Benchmark matrix multiplication using einsum
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), size, |b, &size| {
            let a = DenseND::from_vec(vec![1.0; size * size], &[size, size]).unwrap();
            let b_mat = DenseND::from_vec(vec![2.0; size * size], &[size, size]).unwrap();

            let handle_a = TensorHandle::from_dense_auto(a);
            let handle_b = TensorHandle::from_dense_auto(b_mat);

            b.iter(|| {
                let result = einsum_ex::<f64>("ij,jk->ik")
                    .inputs(black_box(&[handle_a.clone(), handle_b.clone()]))
                    .run()
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark three-tensor contraction
fn bench_three_tensor_contraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("three_tensor_contraction");

    for size in [16, 32, 64].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n * n) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), size, |b, &size| {
            let x = DenseND::from_vec(vec![1.0; size * size], &[size, size]).unwrap();
            let y = DenseND::from_vec(vec![2.0; size * size], &[size, size]).unwrap();
            let z = DenseND::from_vec(vec![3.0; size * size], &[size, size]).unwrap();

            let handle_x = TensorHandle::from_dense_auto(x);
            let handle_y = TensorHandle::from_dense_auto(y);
            let handle_z = TensorHandle::from_dense_auto(z);

            b.iter(|| {
                let result = einsum_ex::<f64>("ij,jk,kl->il")
                    .inputs(black_box(&[
                        handle_x.clone(),
                        handle_y.clone(),
                        handle_z.clone(),
                    ]))
                    .run()
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise operations
fn bench_element_wise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise");

    let size = 1024 * 1024; // 1M elements
    group.throughput(Throughput::Elements(size as u64));

    let tensor = DenseND::from_vec(vec![1.5; size], &[1024, 1024]).unwrap();
    let handle = TensorHandle::from_dense_auto(tensor);

    let mut executor = CpuExecutor::new();

    group.bench_function("neg", |b| {
        b.iter(|| {
            let result = executor.elem_op(ElemOp::Neg, black_box(&handle)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("abs", |b| {
        b.iter(|| {
            let result = executor.elem_op(ElemOp::Abs, black_box(&handle)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("exp", |b| {
        b.iter(|| {
            let result = executor.elem_op(ElemOp::Exp, black_box(&handle)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("log", |b| {
        b.iter(|| {
            let result = executor.elem_op(ElemOp::Log, black_box(&handle)).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark reduction operations
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    for size in [128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        group.bench_with_input(BenchmarkId::new("sum_axis0", n), size, |b, &size| {
            let matrix = DenseND::from_vec(vec![1.5; size * size], &[size, size]).unwrap();
            let handle = TensorHandle::from_dense_auto(matrix);
            let mut executor = CpuExecutor::new();

            b.iter(|| {
                let result = executor
                    .reduce(ReduceOp::Sum, black_box(&handle), black_box(&[0]))
                    .unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("mean_axis0", n), size, |b, &size| {
            let matrix = DenseND::from_vec(vec![1.5; size * size], &[size, size]).unwrap();
            let handle = TensorHandle::from_dense_auto(matrix);
            let mut executor = CpuExecutor::new();

            b.iter(|| {
                let result = executor
                    .reduce(ReduceOp::Mean, black_box(&handle), black_box(&[0]))
                    .unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("max_axis0", n), size, |b, &size| {
            let matrix = DenseND::from_vec(vec![1.5; size * size], &[size, size]).unwrap();
            let handle = TensorHandle::from_dense_auto(matrix);
            let mut executor = CpuExecutor::new();

            b.iter(|| {
                let result = executor
                    .reduce(ReduceOp::Max, black_box(&handle), black_box(&[0]))
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark einsum builder overhead
fn bench_builder_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder_overhead");

    let a = DenseND::from_vec(vec![1.0; 64 * 64], &[64, 64]).unwrap();
    let b = DenseND::from_vec(vec![2.0; 64 * 64], &[64, 64]).unwrap();

    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);

    group.bench_function("with_builder", |bench| {
        bench.iter(|| {
            let result = einsum_ex::<f64>("ij,jk->ik")
                .inputs(black_box(&[handle_a.clone(), handle_b.clone()]))
                .run()
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("with_builder_and_hints", |bench| {
        bench.iter(|| {
            let result = einsum_ex::<f64>("ij,jk->ik")
                .inputs(black_box(&[handle_a.clone(), handle_b.clone()]))
                .hints(black_box(&ExecHints::default()))
                .run()
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_three_tensor_contraction,
    bench_element_wise_ops,
    bench_reductions,
    bench_builder_overhead
);
criterion_main!(benches);
