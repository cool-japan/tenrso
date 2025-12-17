//! Benchmarks for tensor creation methods.
//!
//! This benchmark suite measures the performance of various tensor initialization
//! methods to identify bottlenecks and track performance over time.
//!
//! Run with:
//! ```bash
//! cargo bench --bench tensor_creation
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::DenseND;

/// Benchmark zeros creation for various sizes
fn bench_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("zeros");

    let sizes = vec![
        ("small_2d", vec![100, 100]),
        ("medium_2d", vec![1000, 1000]),
        ("small_3d", vec![50, 50, 50]),
        ("medium_3d", vec![100, 100, 100]),
        ("large_3d", vec![200, 200, 200]),
        ("small_4d", vec![10, 20, 30, 40]),
        ("medium_4d", vec![50, 50, 50, 50]),
    ];

    for (name, shape) in sizes {
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &shape, |b, shape| {
            b.iter(|| {
                let tensor = DenseND::<f64>::zeros(black_box(shape));
                black_box(tensor);
            });
        });
    }

    group.finish();
}

/// Benchmark ones creation for various sizes
fn bench_ones(c: &mut Criterion) {
    let mut group = c.benchmark_group("ones");

    let sizes = vec![
        ("small_2d", vec![100, 100]),
        ("medium_2d", vec![1000, 1000]),
        ("small_3d", vec![50, 50, 50]),
        ("medium_3d", vec![100, 100, 100]),
    ];

    for (name, shape) in sizes {
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &shape, |b, shape| {
            b.iter(|| {
                let tensor = DenseND::<f64>::ones(black_box(shape));
                black_box(tensor);
            });
        });
    }

    group.finish();
}

/// Benchmark from_elem creation
fn bench_from_elem(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_elem");

    let sizes = vec![
        ("small_2d", vec![100, 100]),
        ("medium_2d", vec![1000, 1000]),
        ("small_3d", vec![50, 50, 50]),
    ];

    for (name, shape) in sizes {
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &shape, |b, shape| {
            b.iter(|| {
                let tensor = DenseND::from_elem(black_box(shape), black_box(3.5));
                black_box(tensor);
            });
        });
    }

    group.finish();
}

/// Benchmark from_vec creation
fn bench_from_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_vec");

    let sizes = vec![
        ("small_2d", vec![100, 100]),
        ("medium_2d", vec![1000, 1000]),
        ("small_3d", vec![50, 50, 50]),
    ];

    for (name, shape) in sizes {
        let total: usize = shape.iter().product();
        let data: Vec<f64> = (0..total).map(|x| x as f64).collect();

        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(data, shape),
            |b, (data, shape)| {
                b.iter(|| {
                    let tensor =
                        DenseND::from_vec(black_box(data.clone()), black_box(shape)).unwrap();
                    black_box(tensor);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark random uniform initialization
fn bench_random_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_uniform");

    let sizes = vec![
        ("small_2d", vec![100, 100]),
        ("medium_2d", vec![500, 500]),
        ("small_3d", vec![50, 50, 50]),
    ];

    for (name, shape) in sizes {
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &shape, |b, shape| {
            b.iter(|| {
                let tensor = DenseND::<f64>::random_uniform(
                    black_box(shape),
                    black_box(0.0),
                    black_box(1.0),
                );
                black_box(tensor);
            });
        });
    }

    group.finish();
}

/// Benchmark random normal initialization
fn bench_random_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_normal");

    let sizes = vec![
        ("small_2d", vec![100, 100]),
        ("medium_2d", vec![500, 500]),
        ("small_3d", vec![50, 50, 50]),
    ];

    for (name, shape) in sizes {
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &shape, |b, shape| {
            b.iter(|| {
                let tensor =
                    DenseND::<f64>::random_normal(black_box(shape), black_box(0.0), black_box(1.0));
                black_box(tensor);
            });
        });
    }

    group.finish();
}

/// Benchmark element access (indexing)
fn bench_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing");

    let tensor_2d = DenseND::<f64>::ones(&[1000, 1000]);
    let tensor_3d = DenseND::<f64>::ones(&[100, 100, 100]);
    let tensor_4d = DenseND::<f64>::ones(&[50, 50, 50, 50]);

    group.bench_function("2d_read", |b| {
        b.iter(|| {
            let val = tensor_2d[black_box(&[500, 500])];
            black_box(val);
        });
    });

    group.bench_function("3d_read", |b| {
        b.iter(|| {
            let val = tensor_3d[black_box(&[50, 50, 50])];
            black_box(val);
        });
    });

    group.bench_function("4d_read", |b| {
        b.iter(|| {
            let val = tensor_4d[black_box(&[25, 25, 25, 25])];
            black_box(val);
        });
    });

    let mut tensor_2d_mut = DenseND::<f64>::ones(&[1000, 1000]);

    group.bench_function("2d_write", |b| {
        b.iter(|| {
            tensor_2d_mut[black_box(&[500, 500])] = black_box(42.0);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_zeros,
    bench_ones,
    bench_from_elem,
    bench_from_vec,
    bench_random_uniform,
    bench_random_normal,
    bench_indexing
);

criterion_main!(benches);
