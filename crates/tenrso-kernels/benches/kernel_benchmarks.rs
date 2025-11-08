//! Performance benchmarks for tenrso-kernels
//!
//! Run with: cargo bench -p tenrso-kernels
//!
//! Benchmarks cover:
//! - Khatri-Rao product (serial & parallel)
//! - Kronecker product (serial & parallel)
//! - Hadamard product (serial & in-place)
//! - N-mode product (single & sequential)
//! - MTTKRP (standard, blocked, parallel)
//! - Outer products (2D & N-D, CP reconstruction)
//! - Tucker operator

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::{Array, Array1, Array2};
use std::collections::HashMap;
use tenrso_core::DenseND;
use tenrso_kernels::*;

fn bench_khatri_rao(c: &mut Criterion) {
    let mut group = c.benchmark_group("khatri_rao");

    for &size in [10, 50, 100, 200, 500].iter() {
        let rank = 32;
        let a = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j) as f64);
        let b = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i * j + 1) as f64);

        // Calculate operations count for throughput
        let ops = size * size * rank * 2; // multiply + store per element
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}x{}", size, rank)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(khatri_rao(&a.view(), &b.view()));
                });
            },
        );

        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", size, rank)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(khatri_rao_parallel(&a.view(), &b.view()));
                });
            },
        );
    }
    group.finish();
}

fn bench_kronecker(c: &mut Criterion) {
    let mut group = c.benchmark_group("kronecker");

    for &size in [5, 10, 20, 30, 40].iter() {
        let a = Array2::<f64>::from_shape_fn((size, size), |(i, j)| (i + j) as f64);
        let b = Array2::<f64>::from_shape_fn((size, size), |(i, j)| (i * j + 1) as f64);

        let ops = size * size * size * size * 2; // multiply + store per element
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}x{}", size, size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(kronecker(&a.view(), &b.view()));
                });
            },
        );

        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", size, size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(kronecker_parallel(&a.view(), &b.view()));
                });
            },
        );
    }
    group.finish();
}

fn bench_hadamard(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard");

    for &size in [100, 500, 1000, 2000].iter() {
        let a = Array2::<f64>::from_shape_fn((size, size), |(i, j)| (i + j) as f64);
        let b = Array2::<f64>::from_shape_fn((size, size), |(i, j)| (i * j + 1) as f64);

        let ops = size * size * 2; // multiply + store per element
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("allocating", format!("{}x{}", size, size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(hadamard(&a.view(), &b.view()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("inplace", format!("{}x{}", size, size)),
            &size,
            |bencher, _| {
                let mut result = a.clone();
                bencher.iter(|| {
                    hadamard_inplace(&mut result.view_mut(), &b.view());
                    black_box(&result);
                });
            },
        );
    }
    group.finish();
}

fn bench_nmode_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("nmode_product");

    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor =
            DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
                (idx[0] + idx[1] * 2 + idx[2] * 3) as f64
            }));
        let matrix = Array2::<f64>::from_shape_fn((size, size), |(i, j)| (i + j) as f64);

        // Approximate FLOP count: unfold + matrix multiply
        let ops = size * size * size * size * 2; // roughly
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("single", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(nmode_product(&tensor.view(), &matrix.view(), 1).unwrap());
                });
            },
        );

        // Benchmark sequential multi-mode products
        group.bench_with_input(
            BenchmarkId::new("sequential_3modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let m_view = matrix.view();
                    let matrix_mode_pairs = [(&m_view, 0), (&m_view, 1), (&m_view, 2)];
                    black_box(nmode_products_seq(&tensor.view(), &matrix_mode_pairs).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn bench_mttkrp(c: &mut Criterion) {
    let mut group = c.benchmark_group("mttkrp");

    for &size in [10, 20, 30, 40, 50].iter() {
        let rank = 32;
        let tensor =
            DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
                (idx[0] + idx[1] + idx[2]) as f64
            }));
        let u1 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j) as f64);
        let u2 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i * 2 + j) as f64);
        let u3 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j * 3) as f64);

        // Approximate FLOP count for MTTKRP
        let ops = size * size * size * rank * 2;
        group.throughput(Throughput::Elements(ops as u64));

        // Standard MTTKRP
        group.bench_with_input(
            BenchmarkId::new("standard", format!("{}^3_r{}", size, rank)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap(),
                    );
                });
            },
        );

        // Blocked MTTKRP
        let tile_size = 16;
        group.bench_with_input(
            BenchmarkId::new("blocked", format!("{}^3_r{}_t{}", size, rank, tile_size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        mttkrp_blocked(
                            &tensor.view(),
                            &[u1.view(), u2.view(), u3.view()],
                            1,
                            tile_size,
                        )
                        .unwrap(),
                    );
                });
            },
        );

        // Blocked parallel MTTKRP
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new(
                "blocked_parallel",
                format!("{}^3_r{}_t{}", size, rank, tile_size),
            ),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        mttkrp_blocked_parallel(
                            &tensor.view(),
                            &[u1.view(), u2.view(), u3.view()],
                            1,
                            tile_size,
                        )
                        .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_outer_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("outer_product");

    // 2D outer products
    for &size in [10, 50, 100, 200, 500].iter() {
        let a = Array1::<f64>::from_shape_fn(size, |i| i as f64);
        let b = Array1::<f64>::from_shape_fn(size, |i| (i * 2) as f64);

        let ops = size * size; // one multiply per element
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(BenchmarkId::new("2d", size), &size, |bencher, _| {
            bencher.iter(|| {
                black_box(outer_product_2(&a.view(), &b.view()));
            });
        });
    }

    // N-D outer products
    for &size in [10, 20, 30, 40].iter() {
        let vectors: Vec<Array1<f64>> = (0..3)
            .map(|i| Array1::<f64>::from_shape_fn(size, |j| (i * size + j) as f64))
            .collect();
        let vector_views: Vec<_> = vectors.iter().map(|v| v.view()).collect();

        let ops = size.pow(3);
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(BenchmarkId::new("3d", size), &size, |bencher, _| {
            bencher.iter(|| {
                black_box(outer_product(&vector_views).unwrap());
            });
        });
    }

    // CP reconstruction
    for &(size, rank) in [(20, 5), (30, 10), (40, 20), (50, 32)].iter() {
        let factors: Vec<Array2<f64>> = (0..3)
            .map(|i| Array2::<f64>::from_shape_fn((size, rank), |(j, r)| (i + j + r) as f64))
            .collect();
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let ops = size.pow(3) * rank;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("cp_reconstruct", format!("{}^3_r{}", size, rank)),
            &(size, rank),
            |bencher, _| {
                bencher.iter(|| {
                    black_box(cp_reconstruct(&factor_views, None).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_tucker_operator(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_operator");

    for &size in [10, 20, 30, 40, 50].iter() {
        let core_size = size / 2;
        let core = DenseND::<f64>::from_array(Array::from_shape_fn(
            vec![core_size, core_size, core_size],
            |idx| (idx[0] + idx[1] + idx[2]) as f64,
        ));

        let factors: Vec<Array2<f64>> = (0..3)
            .map(|i| Array2::<f64>::from_shape_fn((size, core_size), |(j, k)| (i + j + k) as f64))
            .collect();

        let mut factor_map = HashMap::new();
        for (i, factor) in factors.iter().enumerate() {
            factor_map.insert(i, factor.view());
        }

        let ops = size * size * size * core_size * 3; // rough estimate
        group.throughput(Throughput::Elements(ops as u64));

        // Single mode
        group.bench_with_input(
            BenchmarkId::new("single_mode", size),
            &size,
            |bencher, _| {
                let mut single_factor = HashMap::new();
                single_factor.insert(0, factors[0].view());
                bencher.iter(|| {
                    black_box(tucker_operator(&core.view(), &single_factor).unwrap());
                });
            },
        );

        // Multiple modes
        group.bench_with_input(
            BenchmarkId::new("three_modes", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tucker_operator(&core.view(), &factor_map).unwrap());
                });
            },
        );

        // Tucker reconstruction (ordered)
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        group.bench_with_input(
            BenchmarkId::new("reconstruct", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tucker_reconstruct(&core.view(), &factor_views).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_khatri_rao,
    bench_kronecker,
    bench_hadamard,
    bench_nmode_product,
    bench_mttkrp,
    bench_outer_product,
    bench_tucker_operator
);
criterion_main!(benches);
