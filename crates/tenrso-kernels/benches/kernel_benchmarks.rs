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
//! - Tensor Train operations (norm, dot product, orthogonalization)
//! - Utility functions (batch ops, normalization, validation)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::{Array, Array1, Array2};
use std::collections::HashMap;
use std::hint::black_box;
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

fn bench_utilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("utilities");

    // mttkrp_all_modes benchmark
    for &size in [10, 20, 30, 40].iter() {
        let rank = 16;
        let tensor =
            DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
                (idx[0] + idx[1] + idx[2]) as f64
            }));
        let u1 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j) as f64);
        let u2 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i * 2 + j) as f64);
        let u3 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j * 3) as f64);

        // Approximate FLOP count: 3 MTTKRPs
        let ops = size * size * size * rank * 2 * 3;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("mttkrp_all_modes", format!("{}^3_r{}", size, rank)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        mttkrp_all_modes(&tensor.view(), &[u1.view(), u2.view(), u3.view()])
                            .unwrap(),
                    );
                });
            },
        );
    }

    // khatri_rao_batch benchmark
    for &(npairs, size, rank) in [(2, 50, 16), (4, 50, 16), (8, 30, 16)].iter() {
        let pairs: Vec<(Array2<f64>, Array2<f64>)> = (0..npairs)
            .map(|i| {
                let a = Array2::<f64>::from_shape_fn((size, rank), |(j, k)| (i + j + k) as f64);
                let b = Array2::<f64>::from_shape_fn((size, rank), |(j, k)| (i * j + k) as f64);
                (a, b)
            })
            .collect();

        let ops = npairs * size * size * rank * 2;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new(
                "khatri_rao_batch",
                format!("{}_pairs_{}x{}", npairs, size, rank),
            ),
            &npairs,
            |bencher, _| {
                let pair_views: Vec<_> = pairs.iter().map(|(a, b)| (a.view(), b.view())).collect();
                bencher.iter(|| {
                    black_box(khatri_rao_batch(&pair_views).unwrap());
                });
            },
        );
    }

    // normalize_factor benchmark
    for &(rows, cols) in [(100, 10), (200, 20), (500, 50), (1000, 100)].iter() {
        let factor = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64 + 1.0);

        let ops = rows * cols * 2; // compute norm + normalize
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("normalize_factor", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, _| {
                bencher.iter(|| {
                    black_box(normalize_factor(&factor.view()));
                });
            },
        );
    }

    // denormalize_factor benchmark
    for &(rows, cols) in [(100, 10), (200, 20), (500, 50), (1000, 100)].iter() {
        let factor = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64 + 1.0);
        let (normalized, norms) = normalize_factor(&factor.view());

        let ops = rows * cols; // multiply per element
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("denormalize_factor", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, _| {
                bencher.iter(|| {
                    black_box(denormalize_factor(&normalized.view(), &norms));
                });
            },
        );
    }

    // validate_factor_shapes benchmark
    for &(nmodes, size, rank) in [(3, 50, 10), (4, 40, 15), (5, 30, 20)].iter() {
        let factors: Vec<Array2<f64>> = (0..nmodes)
            .map(|i| Array2::<f64>::from_shape_fn((size, rank), |(j, k)| (i + j + k) as f64))
            .collect();
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let shape = vec![size; nmodes];

        // Validation is O(nmodes * (check operations))
        let ops = nmodes * size * 2; // rough estimate
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new(
                "validate_factor_shapes",
                format!("{}_modes_{}x{}", nmodes, size, rank),
            ),
            &nmodes,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(validate_factor_shapes(&shape, &factor_views).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_contractions(c: &mut Criterion) {
    let mut group = c.benchmark_group("contractions");

    // contract_tensors (pairwise contraction) - matrix multiplication pattern
    for &size in [10, 20, 30, 40, 50].iter() {
        let a = Array::from_shape_fn(vec![size, size], |idx| (idx[0] + idx[1]) as f64);
        let b = Array::from_shape_fn(vec![size, size], |idx| (idx[0] * 2 + idx[1]) as f64);

        // Matrix multiplication: O(n³)
        let ops = size * size * size * 2;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("contract_tensors_matmul", format!("{}x{}", size, size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(contract_tensors(&a.view(), &b.view(), &[1], &[0]).unwrap());
                });
            },
        );
    }

    // tensor_inner_product (Frobenius inner product)
    for &size in [10, 20, 30, 40, 50].iter() {
        let a = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });
        let b = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] * 2 + idx[1] + idx[2]) as f64
        });

        // Inner product: O(n³) multiply + sum
        let ops = size * size * size * 2;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("tensor_inner_product", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tensor_inner_product(&a.view(), &b.view()).unwrap());
                });
            },
        );
    }

    // sum_over_modes (mode reduction)
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        // Sum over 1 mode: O(n³)
        let ops = size * size * size;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("sum_over_modes_1mode", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(sum_over_modes(&tensor.view(), &[0]).unwrap());
                });
            },
        );

        // Sum over 2 modes: O(n³)
        group.bench_with_input(
            BenchmarkId::new("sum_over_modes_2modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(sum_over_modes(&tensor.view(), &[0, 1]).unwrap());
                });
            },
        );
    }

    // tensor_trace (diagonal sum)
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        // Trace: O(n) for 2 modes
        let ops = size;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("tensor_trace", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tensor_trace(&tensor.view(), 0, 1).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    // sum_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("sum_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(sum_along_modes(&tensor.view(), &[0]).unwrap());
                });
            },
        );
    }

    // mean_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size * 2; // sum + divide
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("mean_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(mean_along_modes(&tensor.view(), &[0]).unwrap());
                });
            },
        );
    }

    // variance_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size * 4; // mean + squared_diff + mean
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("variance_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(variance_along_modes(&tensor.view(), &[0], 0).unwrap());
                });
            },
        );
    }

    // std_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size * 4; // variance + sqrt
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("std_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(std_along_modes(&tensor.view(), &[0], 0).unwrap());
                });
            },
        );
    }

    // frobenius_norm_tensor
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size * 2; // square + sum + sqrt
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("frobenius_norm_tensor", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(frobenius_norm_tensor(&tensor.view()));
                });
            },
        );
    }

    // pnorm_along_modes (L1, L2, Linf)
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size * 2;
        group.throughput(Throughput::Elements(ops as u64));

        // L1 norm
        group.bench_with_input(
            BenchmarkId::new("pnorm_l1", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(pnorm_along_modes(&tensor.view(), &[0], 1.0).unwrap());
                });
            },
        );

        // L2 norm
        group.bench_with_input(
            BenchmarkId::new("pnorm_l2", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(pnorm_along_modes(&tensor.view(), &[0], 2.0).unwrap());
                });
            },
        );

        // L-infinity norm
        group.bench_with_input(
            BenchmarkId::new("pnorm_linf", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(pnorm_along_modes(&tensor.view(), &[0], f64::INFINITY).unwrap());
                });
            },
        );
    }

    // min_along_modes and max_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("min_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(min_along_modes(&tensor.view(), &[0]).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("max_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(max_along_modes(&tensor.view(), &[0]).unwrap());
                });
            },
        );
    }

    // median_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        // O(n log n) sorting + linear scan
        let ops = size * size * size * ((size as f64).log2() as usize);
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("median_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(median_along_modes(&tensor.view(), &[0]).unwrap());
                });
            },
        );
    }

    // percentile_along_modes (25th percentile)
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        // O(n log n) sorting + linear scan
        let ops = size * size * size * ((size as f64).log2() as usize);
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("percentile_25th", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(percentile_along_modes(&tensor.view(), &[0], 25.0).unwrap());
                });
            },
        );
    }

    // percentile_along_modes (75th percentile)
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let ops = size * size * size * ((size as f64).log2() as usize);
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("percentile_75th", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(percentile_along_modes(&tensor.view(), &[0], 75.0).unwrap());
                });
            },
        );
    }

    // skewness_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        // 3 passes: mean, moments
        let ops = size * size * size * 3;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("skewness_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(skewness_along_modes(&tensor.view(), &[0], true).unwrap());
                });
            },
        );
    }

    // kurtosis_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        // 3 passes: mean, moments
        let ops = size * size * size * 3;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("kurtosis_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(kurtosis_along_modes(&tensor.view(), &[0], true, true).unwrap());
                });
            },
        );
    }

    // covariance_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor_x = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });
        let tensor_y = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] * 2 + idx[1] + idx[2]) as f64
        });

        // 3 passes: mean_x, mean_y, covariance computation
        let ops = size * size * size * 3;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("covariance_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        covariance_along_modes(&tensor_x.view(), &tensor_y.view(), &[0], 1)
                            .unwrap(),
                    );
                });
            },
        );
    }

    // correlation_along_modes
    for &size in [10, 20, 30, 40, 50].iter() {
        let tensor_x = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });
        let tensor_y = Array::from_shape_fn(vec![size, size, size], |idx| {
            (idx[0] * 2 + idx[1] + idx[2]) as f64
        });

        // 5 passes: mean_x, mean_y, cov, std_x, std_y
        let ops = size * size * size * 5;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("correlation_along_modes", format!("{}^3", size)),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        correlation_along_modes(&tensor_x.view(), &tensor_y.view(), &[0]).unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_tt_ops(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array3;

    let mut group = c.benchmark_group("tt_operations");

    // Helper function to create TT cores for benchmarking (as Array3)
    fn create_tt_cores_array3(d: usize, n: usize, r: usize) -> Vec<Array3<f64>> {
        let mut cores = Vec::new();

        // First core: [1, n, r]
        cores.push(Array::from_shape_fn((1, n, r), |(_i0, i1, i2)| {
            (i1 + i2) as f64
        }));

        // Middle cores: [r, n, r]
        for i in 1..d - 1 {
            cores.push(Array::from_shape_fn((r, n, r), |(i0, i1, i2)| {
                (i0 + i1 * 2 + i2 + i) as f64
            }));
        }

        // Last core: [r, n, 1]
        cores.push(Array::from_shape_fn((r, n, 1), |(i0, i1, _i2)| {
            (i0 + i1) as f64
        }));

        cores
    }

    // Benchmark tt_norm - efficient Frobenius norm computation
    for &(d, n, r) in [(3, 10, 5), (4, 20, 8), (5, 30, 10), (6, 40, 12)].iter() {
        let cores = create_tt_cores_array3(d, n, r);
        let core_views: Vec<_> = cores.iter().map(|c| c.view()).collect();

        // Operations count: O(d × r³)
        let ops = d * r * r * r;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("tt_norm", format!("d{}_n{}_r{}", d, n, r)),
            &(d, n, r),
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tt_norm(&core_views).unwrap());
                });
            },
        );
    }

    // Benchmark tt_dot - inner product of two TT tensors
    for &(d, n, r) in [(3, 10, 5), (4, 15, 6), (5, 20, 8)].iter() {
        let cores_x = create_tt_cores_array3(d, n, r);
        let cores_y = create_tt_cores_array3(d, n, r);
        let core_views_x: Vec<_> = cores_x.iter().map(|c| c.view()).collect();
        let core_views_y: Vec<_> = cores_y.iter().map(|c| c.view()).collect();

        // Operations count: O(d × r⁴ × n)
        let ops = d * r * r * r * r * n;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("tt_dot", format!("d{}_n{}_r{}", d, n, r)),
            &(d, n, r),
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tt_dot(&core_views_x, &core_views_y).unwrap());
                });
            },
        );
    }

    // Benchmark tt_left_orthogonalize_core - single core orthogonalization
    for &(r1, n, r2) in [(5, 10, 5), (8, 20, 8), (10, 30, 10), (12, 40, 12)].iter() {
        let core = Array3::from_shape_fn((r1, n, r2), |(i0, i1, i2)| {
            (i0 + i1 * 2 + i2 * 3) as f64 + 0.1
        });

        // QR decomposition: O(r₁ × n × r₂²)
        let ops = r1 * n * r2 * r2;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new(
                "tt_left_orthogonalize_core",
                format!("r1{}_n{}_r2{}", r1, n, r2),
            ),
            &(r1, n, r2),
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tt_left_orthogonalize_core(&core.view()).unwrap());
                });
            },
        );
    }

    // Benchmark tt_right_orthogonalize_core - single core orthogonalization
    for &(r1, n, r2) in [(5, 10, 5), (8, 20, 8), (10, 30, 10), (12, 40, 12)].iter() {
        let core = Array3::from_shape_fn((r1, n, r2), |(i0, i1, i2)| {
            (i0 + i1 * 2 + i2 * 3) as f64 + 0.1
        });

        // QR decomposition: O(r₁ × n × r₂²)
        let ops = r1 * n * r2 * r2;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new(
                "tt_right_orthogonalize_core",
                format!("r1{}_n{}_r2{}", r1, n, r2),
            ),
            &(r1, n, r2),
            |bencher, _| {
                bencher.iter(|| {
                    black_box(tt_right_orthogonalize_core(&core.view()).unwrap());
                });
            },
        );
    }

    // Benchmark tt_left_orthogonalize - full left-orthogonalization
    for &(d, n, r) in [(3, 10, 5), (4, 15, 6), (5, 20, 8)].iter() {
        // Full orthogonalization: O(d × r² × n)
        let ops = d * r * r * n;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("tt_left_orthogonalize", format!("d{}_n{}_r{}", d, n, r)),
            &(d, n, r),
            |bencher, _| {
                bencher.iter(|| {
                    let mut cores = create_tt_cores_array3(d, n, r);
                    tt_left_orthogonalize(&mut cores).unwrap();
                    black_box(&cores);
                });
            },
        );
    }

    // Benchmark tt_right_orthogonalize - full right-orthogonalization
    for &(d, n, r) in [(3, 10, 5), (4, 15, 6), (5, 20, 8)].iter() {
        // Full orthogonalization: O(d × r² × n)
        let ops = d * r * r * n;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("tt_right_orthogonalize", format!("d{}_n{}_r{}", d, n, r)),
            &(d, n, r),
            |bencher, _| {
                bencher.iter(|| {
                    let mut cores = create_tt_cores_array3(d, n, r);
                    tt_right_orthogonalize(&mut cores).unwrap();
                    black_box(&cores);
                });
            },
        );
    }

    // Benchmark tt_round - TT rounding (currently orthogonalization-based)
    for &(d, n, r) in [(3, 10, 5), (4, 15, 6), (5, 20, 8)].iter() {
        let epsilon = 1e-6;

        // Rounding operation: O(d × r² × n)
        let ops = d * r * r * n;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("tt_round", format!("d{}_n{}_r{}_eps{}", d, n, r, epsilon)),
            &(d, n, r),
            |bencher, _| {
                bencher.iter(|| {
                    let mut cores = create_tt_cores_array3(d, n, r);
                    tt_round(&mut cores, None, epsilon).unwrap();
                    black_box(&cores);
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
    bench_tucker_operator,
    bench_utilities,
    bench_contractions,
    bench_reductions,
    bench_tt_ops
);
criterion_main!(benches);
