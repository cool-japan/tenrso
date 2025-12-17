//! Benchmarks for reshape and permute operations.
//!
//! These operations are fundamental for tensor manipulation and appear frequently
//! in tensor decomposition and neural network operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench reshape_permute
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::DenseND;

/// Benchmark reshape operations for various tensor sizes
fn bench_reshape(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape");

    let test_cases = vec![
        ("2d_to_1d", vec![1000, 1000], vec![1_000_000]),
        ("1d_to_2d", vec![1_000_000], vec![1000, 1000]),
        ("3d_to_2d", vec![100, 100, 100], vec![10000, 100]),
        ("2d_to_3d", vec![10000, 100], vec![100, 100, 100]),
        ("3d_reshape", vec![50, 60, 70], vec![100, 105, 20]),
        ("4d_to_2d", vec![10, 20, 30, 40], vec![200, 1200]),
        ("2d_to_4d", vec![200, 1200], vec![10, 20, 30, 40]),
        ("4d_reshape", vec![32, 64, 64, 3], vec![32, 12288]),
    ];

    for (name, from_shape, to_shape) in test_cases {
        let tensor = DenseND::<f64>::ones(&from_shape);
        let total: usize = from_shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&tensor, &to_shape),
            |b, (tensor, to_shape)| {
                b.iter(|| {
                    let reshaped = tensor.reshape(black_box(to_shape)).unwrap();
                    black_box(reshaped);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark permute operations for various tensor sizes and permutations
fn bench_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("permute");

    let test_cases = vec![
        ("2d_transpose", vec![1000, 1000], vec![1, 0]),
        ("3d_swap_01", vec![100, 100, 100], vec![1, 0, 2]),
        ("3d_swap_02", vec![100, 100, 100], vec![2, 1, 0]),
        ("3d_cycle", vec![100, 100, 100], vec![2, 0, 1]),
        ("3d_reverse", vec![100, 100, 100], vec![2, 1, 0]),
        ("4d_swap_01", vec![50, 50, 50, 50], vec![1, 0, 2, 3]),
        ("4d_cycle", vec![50, 50, 50, 50], vec![3, 0, 1, 2]),
        ("4d_reverse", vec![50, 50, 50, 50], vec![3, 2, 1, 0]),
        (
            "batch_channels_first",
            vec![32, 64, 64, 3],
            vec![0, 3, 1, 2],
        ),
    ];

    for (name, shape, perm) in test_cases {
        let tensor = DenseND::<f64>::ones(&shape);
        let total: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(&tensor, &perm),
            |b, (tensor, perm)| {
                b.iter(|| {
                    let permuted = tensor.permute(black_box(perm)).unwrap();
                    black_box(permuted);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark common reshape patterns (flatten, unflatten)
fn bench_reshape_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape_patterns");

    // Flatten to 1D
    let tensor_3d = DenseND::<f64>::ones(&[100, 100, 100]);
    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("flatten_3d", |b| {
        b.iter(|| {
            let flat = tensor_3d.reshape(black_box(&[1_000_000])).unwrap();
            black_box(flat);
        });
    });

    // Unflatten from 1D
    let tensor_1d = DenseND::<f64>::ones(&[1_000_000]);
    group.bench_function("unflatten_to_3d", |b| {
        b.iter(|| {
            let unflat = tensor_1d.reshape(black_box(&[100, 100, 100])).unwrap();
            black_box(unflat);
        });
    });

    // Batch flatten (keep first dim)
    let batch_tensor = DenseND::<f64>::ones(&[32, 28, 28, 3]);
    group.bench_function("batch_flatten", |b| {
        b.iter(|| {
            let flat = batch_tensor.reshape(black_box(&[32, 28 * 28 * 3])).unwrap();
            black_box(flat);
        });
    });

    // Merge dimensions
    let tensor_4d = DenseND::<f64>::ones(&[10, 20, 30, 40]);
    group.bench_function("merge_dims", |b| {
        b.iter(|| {
            let merged = tensor_4d.reshape(black_box(&[200, 1200])).unwrap();
            black_box(merged);
        });
    });

    group.finish();
}

/// Benchmark common permute patterns (transpose, channels manipulation)
fn bench_permute_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("permute_patterns");

    // Matrix transpose
    let matrix = DenseND::<f64>::ones(&[1000, 2000]);
    group.throughput(Throughput::Elements(2_000_000));
    group.bench_function("matrix_transpose", |b| {
        b.iter(|| {
            let transposed = matrix.permute(black_box(&[1, 0])).unwrap();
            black_box(transposed);
        });
    });

    // Batch operations: channels last → channels first
    let batch_cl = DenseND::<f64>::ones(&[32, 224, 224, 3]);
    group.bench_function("channels_last_to_first", |b| {
        b.iter(|| {
            let cf = batch_cl.permute(black_box(&[0, 3, 1, 2])).unwrap();
            black_box(cf);
        });
    });

    // Batch operations: channels first → channels last
    let batch_cf = DenseND::<f64>::ones(&[32, 3, 224, 224]);
    group.bench_function("channels_first_to_last", |b| {
        b.iter(|| {
            let cl = batch_cf.permute(black_box(&[0, 2, 3, 1])).unwrap();
            black_box(cl);
        });
    });

    // 3D tensor cycle
    let tensor_3d = DenseND::<f64>::ones(&[100, 200, 300]);
    group.bench_function("3d_cycle_axes", |b| {
        b.iter(|| {
            let cycled = tensor_3d.permute(black_box(&[2, 0, 1])).unwrap();
            black_box(cycled);
        });
    });

    group.finish();
}

/// Benchmark combined reshape + permute operations
fn bench_combined_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_reshape_permute");

    let tensor = DenseND::<f64>::ones(&[10, 20, 30, 40]);
    let total: usize = 10 * 20 * 30 * 40;
    group.throughput(Throughput::Elements(total as u64));

    group.bench_function("reshape_then_permute", |b| {
        b.iter(|| {
            let reshaped = tensor.reshape(black_box(&[200, 1200])).unwrap();
            let permuted = reshaped.permute(black_box(&[1, 0])).unwrap();
            black_box(permuted);
        });
    });

    group.bench_function("permute_then_reshape", |b| {
        b.iter(|| {
            let permuted = tensor.permute(black_box(&[3, 2, 1, 0])).unwrap();
            let reshaped = permuted.reshape(black_box(&[40 * 30, 20 * 10])).unwrap();
            black_box(reshaped);
        });
    });

    group.bench_function("multiple_reshapes", |b| {
        b.iter(|| {
            let r1 = tensor.reshape(black_box(&[200, 1200])).unwrap();
            let r2 = r1.reshape(black_box(&[10, 20, 60, 20])).unwrap();
            let r3 = r2.reshape(black_box(&[10 * 20 * 30 * 40])).unwrap();
            black_box(r3);
        });
    });

    group.finish();
}

/// Benchmark view operations (zero-copy access)
fn bench_views(c: &mut Criterion) {
    let mut group = c.benchmark_group("views");

    let tensor = DenseND::<f64>::ones(&[1000, 1000]);

    group.bench_function("create_view", |b| {
        b.iter(|| {
            let view = tensor.view();
            black_box(view);
        });
    });

    let mut tensor_mut = DenseND::<f64>::ones(&[1000, 1000]);

    group.bench_function("create_view_mut", |b| {
        b.iter(|| {
            let view_mut = tensor_mut.view_mut();
            black_box(view_mut);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reshape,
    bench_permute,
    bench_reshape_patterns,
    bench_permute_patterns,
    bench_combined_ops,
    bench_views
);

criterion_main!(benches);
