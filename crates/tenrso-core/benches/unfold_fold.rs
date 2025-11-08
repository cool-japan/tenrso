//! Benchmarks for unfold/fold (matricization) operations.
//!
//! These operations are critical for tensor decompositions (CP-ALS, Tucker-HOOI, TT-SVD).
//! Performance here directly impacts decomposition speed.
//!
//! Run with:
//! ```bash
//! cargo bench --bench unfold_fold
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::DenseND;

/// Benchmark unfold operations for various tensor sizes and modes
fn bench_unfold(c: &mut Criterion) {
    let mut group = c.benchmark_group("unfold");

    let test_cases = vec![
        ("3d_small", vec![10, 20, 30]),
        ("3d_medium", vec![50, 60, 70]),
        ("3d_large", vec![100, 100, 100]),
        ("3d_rect", vec![200, 50, 25]),
        ("4d_small", vec![10, 10, 10, 10]),
        ("4d_medium", vec![20, 30, 40, 50]),
        ("4d_large", vec![50, 50, 50, 50]),
        ("5d_small", vec![5, 10, 15, 20, 25]),
    ];

    for (name, shape) in test_cases {
        let tensor = DenseND::<f64>::ones(&shape);
        let total: usize = shape.iter().product();

        // Benchmark unfold for each mode
        for mode in 0..shape.len() {
            group.throughput(Throughput::Elements(total as u64));

            group.bench_with_input(
                BenchmarkId::new(name, format!("mode_{}", mode)),
                &(&tensor, mode),
                |b, (tensor, mode)| {
                    b.iter(|| {
                        let unfolded = tensor.unfold(black_box(*mode)).unwrap();
                        black_box(unfolded);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark fold operations for various matrix sizes and target shapes
fn bench_fold(c: &mut Criterion) {
    let mut group = c.benchmark_group("fold");

    let test_cases = vec![
        ("3d_small_mode0", (vec![10, 600], vec![10, 20, 30], 0)),
        ("3d_small_mode1", (vec![20, 300], vec![10, 20, 30], 1)),
        ("3d_small_mode2", (vec![30, 200], vec![10, 20, 30], 2)),
        ("3d_large_mode0", (vec![100, 10000], vec![100, 100, 100], 0)),
        ("3d_large_mode1", (vec![100, 10000], vec![100, 100, 100], 1)),
        ("4d_small_mode0", (vec![10, 1000], vec![10, 10, 10, 10], 0)),
        ("4d_small_mode2", (vec![10, 1000], vec![10, 10, 10, 10], 2)),
    ];

    for (name, (matrix_shape, target_shape, mode)) in test_cases {
        let matrix =
            scirs2_core::ndarray_ext::Array2::<f64>::ones((matrix_shape[0], matrix_shape[1]));
        let total: usize = target_shape.iter().product();

        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(matrix, target_shape.clone(), mode),
            |b, (matrix, target_shape, mode)| {
                b.iter(|| {
                    let folded =
                        DenseND::fold(black_box(matrix), black_box(target_shape), black_box(*mode))
                            .unwrap();
                    black_box(folded);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark roundtrip (unfold → fold) operations
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("unfold_fold_roundtrip");

    let test_cases = vec![
        ("3d_small", vec![10, 20, 30]),
        ("3d_medium", vec![50, 60, 70]),
        ("3d_large", vec![100, 100, 100]),
        ("4d_small", vec![10, 10, 10, 10]),
        ("4d_medium", vec![20, 30, 40, 50]),
    ];

    for (name, shape) in test_cases {
        let tensor = DenseND::<f64>::ones(&shape);
        let total: usize = shape.iter().product();

        // Benchmark roundtrip for first mode
        let mode = 0;
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&tensor, &shape, mode),
            |b, (tensor, shape, mode)| {
                b.iter(|| {
                    let unfolded = tensor.unfold(black_box(*mode)).unwrap();
                    let folded =
                        DenseND::fold(black_box(&unfolded), black_box(*shape), black_box(*mode))
                            .unwrap();
                    black_box(folded);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark unfold performance for CP-ALS typical use case
fn bench_cp_als_unfold(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_als_unfold");

    // Typical tensor sizes for CP decomposition
    let test_cases = vec![
        ("small_256_cubed", vec![256, 256, 256]),
        ("medium_512_cubed", vec![512, 512, 512]),
        ("rect_tall", vec![1000, 100, 100]),
        ("rect_wide", vec![100, 1000, 100]),
    ];

    for (name, shape) in test_cases {
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        // Simulate CP-ALS: unfold along each mode in sequence
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&tensor, &shape),
            |b, (tensor, shape)| {
                b.iter(|| {
                    for mode in 0..shape.len() {
                        let unfolded = tensor.unfold(black_box(mode)).unwrap();
                        black_box(unfolded);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark unfold for Tucker-HOSVD typical use case
fn bench_tucker_unfold(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_unfold");

    // Typical tensor sizes for Tucker decomposition
    let test_cases = vec![
        ("small_3d", vec![128, 128, 128]),
        ("medium_3d", vec![256, 256, 256]),
        ("rect_4d", vec![100, 200, 150, 50]),
    ];

    for (name, shape) in test_cases {
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        // Simulate Tucker-HOSVD: unfold along each mode
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&tensor, &shape),
            |b, (tensor, shape)| {
                b.iter(|| {
                    for mode in 0..shape.len() {
                        let unfolded = tensor.unfold(black_box(mode)).unwrap();
                        black_box(unfolded);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_unfold,
    bench_fold,
    bench_roundtrip,
    bench_cp_als_unfold,
    bench_tucker_unfold
);

criterion_main!(benches);
