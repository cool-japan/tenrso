//! Performance benchmarks for tensor decompositions
//!
//! Benchmarks CP-ALS, Tucker-HOSVD/HOOI, and TT-SVD against performance targets:
//! - CP-ALS: < 2s / 10 iters (256³, rank-64)
//! - Tucker-HOOI: < 3s / 10 iters (512×512×128)
//! - TT-SVD: < 2s build time (32⁶)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::DenseND;
use tenrso_decomp::{
    cp_als, cp_als_constrained, cp_randomized, tt_round, tt_svd, tucker_hooi, tucker_hosvd,
    CpConstraints, InitStrategy,
};

// ============================================================================
// CP-ALS Benchmarks
// ============================================================================

fn bench_cp_als_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_als_small");

    for &(size, rank) in &[(32, 10), (32, 20), (64, 10), (64, 20)] {
        let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}_r{}", size, size, size, rank)),
            &(tensor, rank),
            |b, (tensor, rank)| {
                b.iter(|| {
                    black_box(cp_als(
                        black_box(tensor),
                        black_box(*rank),
                        black_box(10), // 10 iterations
                        black_box(1e-6),
                        black_box(InitStrategy::Random),
                        black_box(None),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cp_als_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_als_medium");
    group.sample_size(10); // Fewer samples for longer benchmarks

    for &(size, rank) in &[(128, 32), (128, 64)] {
        let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}_r{}", size, size, size, rank)),
            &(tensor, rank),
            |b, (tensor, rank)| {
                b.iter(|| {
                    black_box(cp_als(
                        black_box(tensor),
                        black_box(*rank),
                        black_box(10),
                        black_box(1e-6),
                        black_box(InitStrategy::Random),
                        black_box(None),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cp_als_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_als_target");
    group.sample_size(10);

    // Target: 256³ with rank-64, 10 iterations, < 2s
    let size = 256;
    let rank = 64;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    group.throughput(Throughput::Elements((size * size * size) as u64));
    group.bench_function("256x256x256_r64_10iters", |b| {
        b.iter(|| {
            black_box(cp_als(
                black_box(&tensor),
                black_box(rank),
                black_box(10),
                black_box(1e-6),
                black_box(InitStrategy::Random),
                black_box(None),
            ))
        })
    });

    group.finish();
}

fn bench_cp_als_normal_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_als_normal_init");

    let size = 64;
    let rank = 20;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    group.throughput(Throughput::Elements((size * size * size) as u64));

    group.bench_function("64x64x64_r20_random_vs_normal", |b| {
        b.iter(|| {
            black_box(cp_als(
                black_box(&tensor),
                black_box(rank),
                black_box(10),
                black_box(1e-6),
                black_box(InitStrategy::RandomNormal),
                black_box(None),
            ))
        })
    });

    group.finish();
}

fn bench_cp_als_svd_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_als_svd_init");

    let size = 64;
    let rank = 20;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    group.throughput(Throughput::Elements((size * size * size) as u64));

    group.bench_function("64x64x64_r20_svd_init", |b| {
        b.iter(|| {
            black_box(cp_als(
                black_box(&tensor),
                black_box(rank),
                black_box(10),
                black_box(1e-6),
                black_box(InitStrategy::Svd),
                black_box(None),
            ))
        })
    });

    group.finish();
}

fn bench_cp_als_constrained(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_als_constrained");
    group.sample_size(10);

    let size = 64;
    let rank = 15;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    group.throughput(Throughput::Elements((size * size * size) as u64));

    // Non-negative constraint
    let nonneg = CpConstraints::nonnegative();
    group.bench_function("64x64x64_r15_nonnegative", |b| {
        b.iter(|| {
            black_box(cp_als_constrained(
                black_box(&tensor),
                black_box(rank),
                black_box(10),
                black_box(1e-6),
                black_box(InitStrategy::Random),
                black_box(nonneg),
                black_box(None),
            ))
        })
    });

    // L2 regularization
    let l2_reg = CpConstraints::l2_regularized(0.01);
    group.bench_function("64x64x64_r15_l2_reg", |b| {
        b.iter(|| {
            black_box(cp_als_constrained(
                black_box(&tensor),
                black_box(rank),
                black_box(10),
                black_box(1e-6),
                black_box(InitStrategy::Random),
                black_box(l2_reg),
                black_box(None),
            ))
        })
    });

    // Orthogonality constraint
    let ortho = CpConstraints::orthogonal();
    group.bench_function("64x64x64_r15_orthogonal", |b| {
        b.iter(|| {
            black_box(cp_als_constrained(
                black_box(&tensor),
                black_box(rank),
                black_box(10),
                black_box(1e-6),
                black_box(InitStrategy::Random),
                black_box(ortho),
                black_box(None),
            ))
        })
    });

    group.finish();
}

// ============================================================================
// Tucker Benchmarks
// ============================================================================

fn bench_tucker_hosvd_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_hosvd_small");

    for &size in &[32, 64] {
        let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);
        let ranks = vec![size / 2, size / 2, size / 2];

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}_r{}", size, size, size, size / 2)),
            &(tensor, ranks),
            |b, (tensor, ranks)| {
                b.iter(|| black_box(tucker_hosvd(black_box(tensor), black_box(ranks))))
            },
        );
    }

    group.finish();
}

fn bench_tucker_hooi_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_hooi_small");

    for &size in &[32, 64] {
        let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);
        let ranks = vec![size / 2, size / 2, size / 2];

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "{}x{}x{}_r{}_10iters",
                size,
                size,
                size,
                size / 2
            )),
            &(tensor, ranks),
            |b, (tensor, ranks)| {
                b.iter(|| {
                    black_box(tucker_hooi(
                        black_box(tensor),
                        black_box(ranks),
                        black_box(10),
                        black_box(1e-6),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tucker_hosvd_vs_hooi(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_hosvd_vs_hooi");

    let size = 64;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);
    let ranks = vec![size / 2, size / 2, size / 2];

    group.throughput(Throughput::Elements((size * size * size) as u64));

    group.bench_function("64x64x64_hosvd", |b| {
        b.iter(|| black_box(tucker_hosvd(black_box(&tensor), black_box(&ranks))))
    });

    group.bench_function("64x64x64_hooi_10iters", |b| {
        b.iter(|| {
            black_box(tucker_hooi(
                black_box(&tensor),
                black_box(&ranks),
                black_box(10),
                black_box(1e-6),
            ))
        })
    });

    group.finish();
}

fn bench_tucker_hooi_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_hooi_target");
    group.sample_size(10);

    // Target: 512×512×128, 10 iterations, < 3s
    let tensor = DenseND::<f64>::random_uniform(&[512, 512, 128], 0.0, 1.0);
    let ranks = vec![256, 256, 64];

    group.throughput(Throughput::Elements((512 * 512 * 128) as u64));
    group.bench_function("512x512x128_r256_256_64_10iters", |b| {
        b.iter(|| {
            black_box(tucker_hooi(
                black_box(&tensor),
                black_box(&ranks),
                black_box(10),
                black_box(1e-6),
            ))
        })
    });

    group.finish();
}

fn bench_tucker_asymmetric(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_asymmetric");

    // Test asymmetric tensors (realistic use case)
    for &(i, j, k) in &[(100, 200, 50), (256, 256, 64)] {
        let tensor = DenseND::<f64>::random_uniform(&[i, j, k], 0.0, 1.0);
        let ranks = vec![i / 2, j / 2, k / 2];

        group.throughput(Throughput::Elements((i * j * k) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", i, j, k)),
            &(tensor, ranks),
            |b, (tensor, ranks)| {
                b.iter(|| black_box(tucker_hosvd(black_box(tensor), black_box(ranks))))
            },
        );
    }

    group.finish();
}

// ============================================================================
// TT-SVD Benchmarks
// ============================================================================

fn bench_tt_svd_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_svd_small");

    for &(size, n_modes) in &[(8, 3), (8, 4), (16, 3), (16, 4)] {
        let shape: Vec<usize> = vec![size; n_modes];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let max_ranks = vec![10; n_modes - 1];

        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}^{}", size, n_modes)),
            &(tensor, max_ranks),
            |b, (tensor, max_ranks)| {
                b.iter(|| {
                    black_box(tt_svd(
                        black_box(tensor),
                        black_box(max_ranks),
                        black_box(1e-10),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tt_svd_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_svd_target");
    group.sample_size(10);

    // Target: 32⁶, < 2s
    let size = 32;
    let n_modes = 6;
    let shape: Vec<usize> = vec![size; n_modes];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let max_ranks = vec![20; n_modes - 1];

    let total_elements: usize = shape.iter().product();
    group.throughput(Throughput::Elements(total_elements as u64));

    group.bench_function("32^6_r20", |b| {
        b.iter(|| {
            black_box(tt_svd(
                black_box(&tensor),
                black_box(&max_ranks),
                black_box(1e-10),
            ))
        })
    });

    group.finish();
}

fn bench_tt_svd_varying_ranks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_svd_varying_ranks");

    let size = 16;
    let n_modes = 4;
    let shape: Vec<usize> = vec![size; n_modes];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

    let total_elements: usize = shape.iter().product();
    group.throughput(Throughput::Elements(total_elements as u64));

    for &rank in &[5, 10, 20, 30] {
        let max_ranks = vec![rank; n_modes - 1];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("16^4_r{}", rank)),
            &max_ranks,
            |b, max_ranks| {
                b.iter(|| {
                    black_box(tt_svd(
                        black_box(&tensor),
                        black_box(max_ranks),
                        black_box(1e-10),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tt_svd_high_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_svd_high_order");
    group.sample_size(10);

    // Test high-order tensors (typical in quantum/tensor networks)
    for &n_modes in &[5, 6, 7] {
        let size = 10;
        let shape: Vec<usize> = vec![size; n_modes];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let max_ranks = vec![15; n_modes - 1];

        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("10^{}", n_modes)),
            &(tensor, max_ranks),
            |b, (tensor, max_ranks)| {
                b.iter(|| {
                    black_box(tt_svd(
                        black_box(tensor),
                        black_box(max_ranks),
                        black_box(1e-10),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// TT-Rounding Benchmarks
// ============================================================================

fn bench_tt_round_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_round_small");

    for &(size, n_modes) in &[(8, 4), (10, 5)] {
        let shape: Vec<usize> = vec![size; n_modes];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let max_ranks = vec![15; n_modes - 1];
        let tt = tt_svd(&tensor, &max_ranks, 1e-10).unwrap();

        // Round to smaller ranks
        let round_ranks = vec![8; n_modes - 1];

        let total_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}^{}_r15->r8", size, n_modes)),
            &(tt, round_ranks),
            |b, (tt, round_ranks)| {
                b.iter(|| {
                    black_box(tt_round(
                        black_box(tt),
                        black_box(round_ranks),
                        black_box(1e-6),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tt_round_varying_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_round_varying_reduction");

    let size = 12;
    let n_modes = 5;
    let shape: Vec<usize> = vec![size; n_modes];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let max_ranks = vec![20; n_modes - 1];
    let tt = tt_svd(&tensor, &max_ranks, 1e-10).unwrap();

    let total_elements: usize = shape.iter().product();
    group.throughput(Throughput::Elements(total_elements as u64));

    // Test different rank reductions
    for &target_rank in &[15, 10, 5] {
        let round_ranks = vec![target_rank; n_modes - 1];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("12^5_r20->r{}", target_rank)),
            &round_ranks,
            |b, round_ranks| {
                b.iter(|| {
                    black_box(tt_round(
                        black_box(&tt),
                        black_box(round_ranks),
                        black_box(1e-6),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tt_round_tolerance_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_round_tolerance_impact");

    let size = 10;
    let n_modes = 4;
    let shape: Vec<usize> = vec![size; n_modes];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let max_ranks = vec![15; n_modes - 1];
    let tt = tt_svd(&tensor, &max_ranks, 1e-10).unwrap();

    let round_ranks = vec![10; n_modes - 1];

    let total_elements: usize = shape.iter().product();
    group.throughput(Throughput::Elements(total_elements as u64));

    // Test different tolerances
    for &tol in &[1e-8, 1e-6, 1e-4, 1e-2] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("10^4_tol_{:.0e}", tol)),
            &tol,
            |b, &tol| {
                b.iter(|| {
                    black_box(tt_round(
                        black_box(&tt),
                        black_box(&round_ranks),
                        black_box(tol),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tt_round_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_round_large");
    group.sample_size(10);

    // Realistic large TT-rounding scenario
    let size = 16;
    let n_modes = 6;
    let shape: Vec<usize> = vec![size; n_modes];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let max_ranks = vec![25; n_modes - 1];
    let tt = tt_svd(&tensor, &max_ranks, 1e-10).unwrap();

    let round_ranks = vec![12; n_modes - 1];

    let total_elements: usize = shape.iter().product();
    group.throughput(Throughput::Elements(total_elements as u64));

    group.bench_function("16^6_r25->r12", |b| {
        b.iter(|| {
            black_box(tt_round(
                black_box(&tt),
                black_box(&round_ranks),
                black_box(1e-6),
            ))
        })
    });

    group.finish();
}

// ============================================================================
// Reconstruction Benchmarks
// ============================================================================

fn bench_cp_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_reconstruction");

    let size = 64;
    let rank = 20;
    let shape = vec![size, size, size];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let cp = cp_als(&tensor, rank, 10, 1e-6, InitStrategy::Random, None).unwrap();

    group.throughput(Throughput::Elements((size * size * size) as u64));
    group.bench_function("cp_64x64x64_r20", |b| {
        b.iter(|| black_box(cp.reconstruct(black_box(&shape))))
    });

    group.finish();
}

fn bench_tucker_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_reconstruction");

    let size = 64;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);
    let ranks = vec![size / 2, size / 2, size / 2];
    let tucker = tucker_hosvd(&tensor, &ranks).unwrap();

    group.throughput(Throughput::Elements((size * size * size) as u64));
    group.bench_function("tucker_64x64x64_r32", |b| {
        b.iter(|| black_box(tucker.reconstruct()))
    });

    group.finish();
}

fn bench_tt_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_reconstruction");

    let size = 16;
    let n_modes = 4;
    let shape: Vec<usize> = vec![size; n_modes];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let max_ranks = vec![10; n_modes - 1];
    let tt = tt_svd(&tensor, &max_ranks, 1e-10).unwrap();

    let total_elements: usize = shape.iter().product();
    group.throughput(Throughput::Elements(total_elements as u64));

    group.bench_function("tt_16^4_r10", |b| b.iter(|| black_box(tt.reconstruct())));

    group.finish();
}

// ============================================================================
// Compression Ratio Analysis
// ============================================================================

fn bench_compression_efficiency(c: &mut Criterion) {
    let group = c.benchmark_group("compression_efficiency");

    let size = 64;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);
    let full_size = size * size * size;

    // CP with varying ranks
    for &rank in &[10, 20, 40, 60] {
        let _cp = cp_als(&tensor, rank, 10, 1e-6, InitStrategy::Random, None).unwrap();
        // CP parameters: 3 factor matrices of size (64 × rank)
        let cp_size = 3 * size * rank;
        let ratio = full_size as f64 / cp_size as f64;
        println!("CP rank-{}: compression {:.2}x", rank, ratio);
    }

    // Tucker with varying ranks
    for &r in &[16, 32, 48] {
        let ranks = vec![r, r, r];
        let _tucker = tucker_hosvd(&tensor, &ranks).unwrap();
        // Tucker parameters: core (r×r×r) + 3 factor matrices (64×r each)
        let tucker_size = r * r * r + 3 * size * r;
        let ratio = full_size as f64 / tucker_size as f64;
        println!("Tucker rank-{}: compression {:.2}x", r, ratio);
    }

    // TT with varying ranks
    let shape = vec![16, 16, 16, 16];
    let tensor_4d = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    for &r in &[5, 10, 15] {
        let max_ranks = vec![r, r, r];
        let tt = tt_svd(&tensor_4d, &max_ranks, 1e-10).unwrap();
        let ratio = tt.compression_ratio();
        println!("TT rank-{}: compression {:.2}x", r, ratio);
    }

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

// ============================================================================
// Tensor Completion Benchmarks (NEW!)
// ============================================================================

fn bench_cp_completion_varying_observation_rates(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::cp_completion;

    let mut group = c.benchmark_group("cp_completion_obs_rates");
    group.sample_size(10);

    let size = 30;
    let rank = 5;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    for &obs_rate in &[0.3, 0.5, 0.7] {
        // Create mask
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < obs_rate {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("obs_rate_{:.0}%", obs_rate * 100.0)),
            &(&tensor, &mask, rank),
            |b, (tensor, mask, rank)| {
                b.iter(|| {
                    black_box(cp_completion(
                        black_box(tensor),
                        black_box(mask),
                        black_box(*rank),
                        black_box(50),
                        black_box(1e-4),
                        black_box(InitStrategy::Random),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cp_completion_varying_sizes(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::cp_completion;

    let mut group = c.benchmark_group("cp_completion_sizes");
    group.sample_size(10);

    let rank = 5;
    let obs_rate = 0.5;

    for &size in &[20, 30, 40] {
        let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

        // Create mask
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < obs_rate {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            &(tensor, mask, rank),
            |b, (tensor, mask, rank)| {
                b.iter(|| {
                    black_box(cp_completion(
                        black_box(tensor),
                        black_box(mask),
                        black_box(*rank),
                        black_box(50),
                        black_box(1e-4),
                        black_box(InitStrategy::Random),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cp_completion_varying_ranks(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::cp_completion;

    let mut group = c.benchmark_group("cp_completion_ranks");
    group.sample_size(10);

    let size = 30;
    let obs_rate = 0.5;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    // Create mask
    let mut mask_data = Array::zeros(vec![size, size, size]);
    let mut rng = thread_rng();

    for idx in mask_data.iter_mut() {
        if rng.random::<f64>() < obs_rate {
            *idx = 1.0;
        }
    }

    let mask = DenseND::from_array(mask_data.into_dyn());

    for &rank in &[3, 5, 8, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rank_{}", rank)),
            &(&tensor, &mask, rank),
            |b, (tensor, mask, rank)| {
                b.iter(|| {
                    black_box(cp_completion(
                        black_box(tensor),
                        black_box(mask),
                        black_box(*rank),
                        black_box(50),
                        black_box(1e-4),
                        black_box(InitStrategy::Random),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cp_completion_vs_full(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::cp_completion;

    let mut group = c.benchmark_group("cp_completion_vs_full");
    group.sample_size(10);

    let size = 35;
    let rank = 6;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    // Full tensor (100% observed)
    let mask_full = DenseND::from_array(Array::<f64, _>::ones(vec![size, size, size]).into_dyn());

    group.bench_with_input(
        BenchmarkId::from_parameter("full_100%"),
        &(&tensor, &mask_full, rank),
        |b, (tensor, _mask, rank)| {
            b.iter(|| {
                black_box(cp_als(
                    black_box(tensor),
                    black_box(*rank),
                    black_box(50),
                    black_box(1e-4),
                    black_box(InitStrategy::Random),
                    black_box(None),
                ))
            })
        },
    );

    // Completion with 60% observed
    let mut mask_data = Array::zeros(vec![size, size, size]);
    let mut rng = thread_rng();

    for idx in mask_data.iter_mut() {
        if rng.random::<f64>() < 0.6 {
            *idx = 1.0;
        }
    }

    let mask_partial = DenseND::from_array(mask_data.into_dyn());

    group.bench_with_input(
        BenchmarkId::from_parameter("completion_60%"),
        &(&tensor, &mask_partial, rank),
        |b, (tensor, mask, rank)| {
            b.iter(|| {
                black_box(cp_completion(
                    black_box(tensor),
                    black_box(mask),
                    black_box(*rank),
                    black_box(50),
                    black_box(1e-4),
                    black_box(InitStrategy::Random),
                ))
            })
        },
    );

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    cp_benches,
    bench_cp_als_small,
    bench_cp_als_medium,
    bench_cp_als_target,
    bench_cp_als_normal_init,
    bench_cp_als_svd_init,
    bench_cp_als_constrained,
);

criterion_group!(
    tucker_benches,
    bench_tucker_hosvd_small,
    bench_tucker_hooi_small,
    bench_tucker_hosvd_vs_hooi,
    bench_tucker_hooi_target,
    bench_tucker_asymmetric,
);

criterion_group!(
    tt_benches,
    bench_tt_svd_small,
    bench_tt_svd_target,
    bench_tt_svd_varying_ranks,
    bench_tt_svd_high_order,
);

criterion_group!(
    tt_round_benches,
    bench_tt_round_small,
    bench_tt_round_varying_reduction,
    bench_tt_round_tolerance_impact,
    bench_tt_round_large,
);

criterion_group!(
    reconstruction_benches,
    bench_cp_reconstruction,
    bench_tucker_reconstruction,
    bench_tt_reconstruction,
);

criterion_group!(compression_benches, bench_compression_efficiency,);

// ========================================================================
// Tucker Completion Benchmarks
// ========================================================================

fn bench_tucker_completion_varying_observation_rates(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::tucker_completion;

    let mut group = c.benchmark_group("tucker_completion_obs_rates");
    group.sample_size(10);

    let size = 20; // Smaller than CP due to Tucker's higher complexity
    let rank = 5;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    for &obs_rate in &[0.3, 0.5, 0.7] {
        // Create mask
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < obs_rate {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("obs_rate_{:.0}%", obs_rate * 100.0)),
            &(&tensor, &mask),
            |b, (tensor, mask)| {
                b.iter(|| {
                    black_box(tucker_completion(
                        black_box(tensor),
                        black_box(mask),
                        black_box(&[rank, rank, rank]),
                        black_box(20),
                        black_box(1e-4),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tucker_completion_varying_sizes(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::tucker_completion;

    let mut group = c.benchmark_group("tucker_completion_sizes");
    group.sample_size(10);

    let rank = 5;
    let obs_rate = 0.5;

    for &size in &[15, 20, 25] {
        let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

        // Create mask
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < obs_rate {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            &(tensor, mask),
            |b, (tensor, mask)| {
                b.iter(|| {
                    black_box(tucker_completion(
                        black_box(tensor),
                        black_box(mask),
                        black_box(&[rank, rank, rank]),
                        black_box(20),
                        black_box(1e-4),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tucker_completion_varying_ranks(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::tucker_completion;

    let mut group = c.benchmark_group("tucker_completion_ranks");
    group.sample_size(10);

    let size = 20;
    let obs_rate = 0.5;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    // Create mask
    let mut mask_data = Array::zeros(vec![size, size, size]);
    let mut rng = thread_rng();

    for idx in mask_data.iter_mut() {
        if rng.random::<f64>() < obs_rate {
            *idx = 1.0;
        }
    }

    let mask = DenseND::from_array(mask_data.into_dyn());

    for &rank in &[3, 5, 7, 9] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rank_{}", rank)),
            &(&tensor, &mask),
            |b, (tensor, mask)| {
                b.iter(|| {
                    black_box(tucker_completion(
                        black_box(tensor),
                        black_box(mask),
                        black_box(&[rank, rank, rank]),
                        black_box(20),
                        black_box(1e-4),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_tucker_completion_vs_full(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{thread_rng, Rng};
    use tenrso_decomp::{tucker_completion, tucker_hooi};

    let mut group = c.benchmark_group("tucker_completion_vs_full");
    group.sample_size(10);

    let size = 25;
    let rank = 6;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    // Full tensor (100% observed)
    let mask_full = DenseND::from_array(Array::<f64, _>::ones(vec![size, size, size]).into_dyn());

    group.bench_with_input(
        BenchmarkId::from_parameter("full_100%"),
        &(&tensor, &mask_full),
        |b, (tensor, _mask)| {
            b.iter(|| {
                black_box(tucker_hooi(
                    black_box(tensor),
                    black_box(&[rank, rank, rank]),
                    black_box(20),
                    black_box(1e-4),
                ))
            })
        },
    );

    // Partial tensor (60% observed)
    let mut mask_partial_data = Array::zeros(vec![size, size, size]);
    let mut rng = thread_rng();

    for idx in mask_partial_data.iter_mut() {
        if rng.random::<f64>() < 0.6 {
            *idx = 1.0;
        }
    }

    let mask_partial = DenseND::from_array(mask_partial_data.into_dyn());

    group.bench_with_input(
        BenchmarkId::from_parameter("partial_60%"),
        &(&tensor, &mask_partial),
        |b, (tensor, mask)| {
            b.iter(|| {
                black_box(tucker_completion(
                    black_box(tensor),
                    black_box(mask),
                    black_box(&[rank, rank, rank]),
                    black_box(20),
                    black_box(1e-4),
                ))
            })
        },
    );

    group.finish();
}

criterion_group!(
    completion_benches,
    bench_cp_completion_varying_observation_rates,
    bench_cp_completion_varying_sizes,
    bench_cp_completion_varying_ranks,
    bench_cp_completion_vs_full,
);

// ============================================================================
// Randomized CP Benchmarks
// ============================================================================

fn bench_cp_randomized_varying_sketch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_randomized_sketch_sizes");
    group.sample_size(10);

    let size = 100;
    let rank = 10;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    for &sketch_mult in &[3, 5, 7] {
        let sketch_size = rank * sketch_mult;
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x_oversampling", sketch_mult)),
            &sketch_size,
            |b, sketch_size| {
                b.iter(|| {
                    black_box(cp_randomized(
                        black_box(&tensor),
                        black_box(rank),
                        black_box(20),
                        black_box(1e-4),
                        black_box(InitStrategy::Random),
                        black_box(*sketch_size),
                        black_box(5),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cp_randomized_varying_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_randomized_sizes");
    group.sample_size(10);

    let rank = 10;
    let sketch_size = rank * 5;

    for &size in &[50, 100, 150] {
        let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    black_box(cp_randomized(
                        black_box(tensor),
                        black_box(rank),
                        black_box(15),
                        black_box(1e-4),
                        black_box(InitStrategy::Random),
                        black_box(sketch_size),
                        black_box(5),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cp_randomized_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_randomized_vs_standard");
    group.sample_size(10);

    let size = 120;
    let rank = 15;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    group.throughput(Throughput::Elements((size * size * size) as u64));

    // Standard CP-ALS
    group.bench_function("standard_cp_als", |b| {
        b.iter(|| {
            black_box(cp_als(
                black_box(&tensor),
                black_box(rank),
                black_box(20),
                black_box(1e-4),
                black_box(InitStrategy::Random),
                black_box(None),
            ))
        })
    });

    // Randomized CP with 5x oversampling
    group.bench_function("randomized_cp_5x", |b| {
        b.iter(|| {
            black_box(cp_randomized(
                black_box(&tensor),
                black_box(rank),
                black_box(20),
                black_box(1e-4),
                black_box(InitStrategy::Random),
                black_box(rank * 5),
                black_box(5),
            ))
        })
    });

    group.finish();
}

fn bench_cp_randomized_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_randomized_large_scale");
    group.sample_size(10);

    // Benchmark for very large tensors where randomized CP shines
    let size = 200;
    let rank = 20;
    let sketch_size = rank * 4;
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    group.throughput(Throughput::Elements((size * size * size) as u64));
    group.bench_function("200x200x200_r20", |b| {
        b.iter(|| {
            black_box(cp_randomized(
                black_box(&tensor),
                black_box(rank),
                black_box(15),
                black_box(1e-4),
                black_box(InitStrategy::Random),
                black_box(sketch_size),
                black_box(5),
            ))
        })
    });

    group.finish();
}

criterion_group!(
    randomized_cp_benches,
    bench_cp_randomized_varying_sketch_sizes,
    bench_cp_randomized_varying_sizes,
    bench_cp_randomized_vs_standard,
    bench_cp_randomized_large_scale,
);

criterion_group!(
    tucker_completion_benches,
    bench_tucker_completion_varying_observation_rates,
    bench_tucker_completion_varying_sizes,
    bench_tucker_completion_varying_ranks,
    bench_tucker_completion_vs_full,
);

// ============================================================================
// TT Matrix-Vector Product Benchmarks
// ============================================================================

fn bench_tt_matvec_varying_sizes(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array1;
    use tenrso_decomp::tt::{tt_matrix_from_diagonal, tt_svd};

    let mut group = c.benchmark_group("tt_matvec_sizes");
    group.sample_size(20);

    for &(size, n_modes) in &[(4, 3), (4, 4), (5, 3), (6, 3)] {
        let shape = vec![size; n_modes];
        let total_size: usize = shape.iter().product();

        // Create diagonal matrix
        let diag = Array1::from_vec(vec![2.0; total_size]);
        let tt_mat = tt_matrix_from_diagonal(&diag, &shape);

        // Create vector
        let vec = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let tt_vec = tt_svd(
            &vec,
            &vec![size.saturating_sub(1); n_modes.saturating_sub(1)],
            1e-10,
        )
        .expect("TT-SVD should succeed");

        group.throughput(Throughput::Elements(total_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}^{}", size, n_modes)),
            &(tt_mat, tt_vec),
            |b, (matrix, vector)| b.iter(|| black_box(matrix.matvec(black_box(vector)))),
        );
    }

    group.finish();
}

fn bench_tt_matvec_varying_ranks(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array1;
    use tenrso_decomp::tt::{tt_matrix_from_diagonal, tt_svd};

    let mut group = c.benchmark_group("tt_matvec_ranks");
    group.sample_size(20);

    let size = 5;
    let n_modes = 4;
    let shape = vec![size; n_modes];
    let total_size: usize = shape.iter().product();

    for &vec_rank in &[2, 3, 4] {
        // Create diagonal matrix (all ranks = 1)
        let diag = Array1::from_vec(vec![2.0; total_size]);
        let tt_mat = tt_matrix_from_diagonal(&diag, &shape);

        // Create vector with specified rank
        let vec = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let max_ranks = vec![vec_rank; n_modes.saturating_sub(1)];
        let tt_vec = tt_svd(&vec, &max_ranks, 1e-10).expect("TT-SVD should succeed");

        group.throughput(Throughput::Elements(total_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rank_{}", vec_rank)),
            &(tt_mat, tt_vec),
            |b, (matrix, vector)| b.iter(|| black_box(matrix.matvec(black_box(vector)))),
        );
    }

    group.finish();
}

fn bench_tt_matvec_identity(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array1;
    use tenrso_decomp::tt::{tt_matrix_from_diagonal, tt_svd};

    let mut group = c.benchmark_group("tt_matvec_identity");
    group.sample_size(30);

    let size = 6;
    let n_modes = 3;
    let shape = vec![size; n_modes];
    let total_size: usize = shape.iter().product();

    // Create identity matrix
    let ones = Array1::from_vec(vec![1.0; total_size]);
    let identity = tt_matrix_from_diagonal(&ones, &shape);

    // Create vector
    let vec = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let tt_vec =
        tt_svd(&vec, &vec![4; n_modes.saturating_sub(1)], 1e-10).expect("TT-SVD should succeed");

    group.throughput(Throughput::Elements(total_size as u64));
    group.bench_function("6x6x6", |b| {
        b.iter(|| black_box(identity.matvec(black_box(&tt_vec))))
    });

    group.finish();
}

fn bench_tt_matvec_vs_reconstruction(c: &mut Criterion) {
    use scirs2_core::ndarray_ext::Array1;
    use tenrso_decomp::tt::{tt_matrix_from_diagonal, tt_svd};

    let mut group = c.benchmark_group("tt_matvec_vs_full_reconstruction");
    group.sample_size(15);

    let size = 5;
    let n_modes = 3;
    let shape = vec![size; n_modes];
    let total_size: usize = shape.iter().product();

    // Create matrix and vector
    let diag = Array1::from_vec(vec![2.0; total_size]);
    let tt_mat = tt_matrix_from_diagonal(&diag, &shape);
    let vec = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let tt_vec =
        tt_svd(&vec, &vec![3; n_modes.saturating_sub(1)], 1e-10).expect("TT-SVD should succeed");

    group.throughput(Throughput::Elements(total_size as u64));

    // Benchmark TT matrix-vector
    group.bench_function("tt_matvec", |b| {
        b.iter(|| black_box(tt_mat.matvec(black_box(&tt_vec))))
    });

    // Benchmark full reconstruction (for comparison)
    group.bench_function("full_reconstruct", |b| {
        b.iter(|| {
            let result = tt_mat.matvec(&tt_vec).unwrap();
            black_box(result.reconstruct())
        })
    });

    group.finish();
}

criterion_group!(
    tt_matvec_benches,
    bench_tt_matvec_varying_sizes,
    bench_tt_matvec_varying_ranks,
    bench_tt_matvec_identity,
    bench_tt_matvec_vs_reconstruction,
);

criterion_main!(
    cp_benches,
    tucker_benches,
    tt_benches,
    tt_round_benches,
    reconstruction_benches,
    compression_benches,
    completion_benches,
    randomized_cp_benches,
    tucker_completion_benches,
    tt_matvec_benches,
);
