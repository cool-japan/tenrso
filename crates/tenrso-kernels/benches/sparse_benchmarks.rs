//! Performance benchmarks for sparse tensor kernels
//!
//! Compares sparse-native paths against dense baselines to document
//! the sparsity speedup at varying nnz fractions.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p tenrso-kernels --features csf --bench sparse_benchmarks
//! ```
//!
//! ## Benchmarks
//!
//! - `mttkrp_coo_vs_dense`: COO MTTKRP vs dense MTTKRP at 1/5/10% fill
//! - `mttkrp_formats`: COO vs CSF vs HiCOO at matched nonzero sets
//! - `nmode_sparse_vs_dense`: sparse n-mode product vs dense at 1/5/10% fill
//!
//! Results are reported with `Throughput::Elements(nnz * cp_rank)` so that
//! the y-axis is effective nnz·R operations per second.

#![allow(clippy::cast_precision_loss)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use scirs2_core::ndarray_ext::Array2;
use std::hint::black_box;
use tenrso_kernels::mttkrp::mttkrp;

#[cfg(feature = "sparse")]
use tenrso_kernels::mttkrp_sparse::{mttkrp_sparse_coo, mttkrp_sparse_coo_parallel};
#[cfg(feature = "sparse")]
use tenrso_kernels::nmode_sparse::{nmode_product_sparse_coo, nmode_product_sparse_coo_parallel};
#[cfg(feature = "sparse")]
use tenrso_sparse::coo::CooTensor;

#[cfg(feature = "csf")]
use tenrso_kernels::mttkrp_hicoo::{mttkrp_hicoo, mttkrp_hicoo_parallel};
#[cfg(feature = "csf")]
use tenrso_kernels::mttkrp_sparse_csf::{mttkrp_sparse_csf, mttkrp_sparse_csf_parallel};
#[cfg(feature = "csf")]
use tenrso_sparse::{csf::CsfTensor, hicoo::HiCooTensor};

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

/// Build a deterministic sparse COO tensor with the given shape and target fill fraction.
///
/// Uses a simple LCG to avoid a `rand` dep: positions selected by
/// `(seed * 6364136223846793005 + 1442695040888963407) % total` repeated.
#[cfg(feature = "sparse")]
fn build_coo(shape: &[usize], fill_frac: f64, seed: u64) -> CooTensor<f64> {
    let total: usize = shape.iter().product();
    let nnz = ((total as f64) * fill_frac).max(1.0) as usize;
    let ndim = shape.len();

    let mut tensor = CooTensor::zeros(shape.to_vec()).unwrap();
    let mut rng = seed;
    let mut inserted = 0usize;

    // Insert `nnz` unique positions (skip duplicates via deduplicate at end).
    while inserted < nnz {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let flat = rng as usize % total;
        let mut idx = vec![0usize; ndim];
        let mut rem = flat;
        for d in (0..ndim).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        let val = (rng >> 32) as f64 / (u32::MAX as f64) + 0.1;
        tensor.push(idx, val).ok(); // ignore duplicate-push errors
        inserted += 1;
    }
    tensor.deduplicate();
    tensor
}

/// Build factor matrices for MTTKRP: `factors[k]` has shape `(shape[k], cp_rank)`.
fn build_factors(shape: &[usize], cp_rank: usize, seed: u64) -> Vec<Array2<f64>> {
    let mut rng = seed;
    shape
        .iter()
        .map(|&rows| {
            let vals: Vec<f64> = (0..rows * cp_rank)
                .map(|_| {
                    rng = rng
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (rng >> 32) as f64 / (u32::MAX as f64) + 0.1
                })
                .collect();
            Array2::from_shape_vec((rows, cp_rank), vals).unwrap()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark: sparse COO MTTKRP vs dense MTTKRP
// ---------------------------------------------------------------------------

#[cfg(feature = "sparse")]
fn bench_mttkrp_coo_vs_dense(c: &mut Criterion) {
    const CP_RANK: usize = 32;
    const MODE: usize = 0;

    let configs: &[([usize; 3], f64, &str)] = &[
        ([50, 50, 50], 0.01, "50³_1pct"),
        ([50, 50, 50], 0.05, "50³_5pct"),
        ([50, 50, 50], 0.10, "50³_10pct"),
        ([100, 50, 50], 0.01, "100x50x50_1pct"),
        ([100, 50, 50], 0.05, "100x50x50_5pct"),
    ];

    for &(shape, fill, label) in configs {
        let coo = build_coo(&shape, fill, 42);
        let factors = build_factors(&shape, CP_RANK, 7);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let nnz = coo.nnz();

        // Dense baseline: densify then MTTKRP
        let dense = coo.to_dense().unwrap();

        let mut group = c.benchmark_group(format!("mttkrp_coo_vs_dense/{}", label));
        group.throughput(Throughput::Elements((nnz * CP_RANK) as u64));

        group.bench_function("dense_baseline", |b| {
            b.iter(|| {
                let result = mttkrp(&dense.view(), &factor_views, MODE).unwrap();
                black_box(result);
            })
        });

        group.bench_function("coo_serial", |b| {
            b.iter(|| {
                let result = mttkrp_sparse_coo(&coo, &factor_views, MODE).unwrap();
                black_box(result);
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("coo_parallel", |b| {
            b.iter(|| {
                let result = mttkrp_sparse_coo_parallel(&coo, &factor_views, MODE).unwrap();
                black_box(result);
            })
        });

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Benchmark: MTTKRP across formats (COO vs CSF vs HiCOO)
// ---------------------------------------------------------------------------

#[cfg(feature = "csf")]
fn bench_mttkrp_formats(c: &mut Criterion) {
    const CP_RANK: usize = 32;
    const MODE: usize = 0;
    const FILL: f64 = 0.05;

    let configs: &[([usize; 3], &[usize; 3], &str)] = &[
        ([40, 40, 40], &[4, 4, 4], "40³_blk4"),
        ([60, 60, 60], &[4, 4, 4], "60³_blk4"),
        ([80, 80, 80], &[8, 8, 8], "80³_blk8"),
    ];

    for &(shape, block_shape, label) in configs {
        let coo = build_coo(&shape, FILL, 99);
        let factors = build_factors(&shape, CP_RANK, 3);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let nnz = coo.nnz();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, block_shape).unwrap();

        let mut group = c.benchmark_group(format!("mttkrp_formats/{}", label));
        group.throughput(Throughput::Elements((nnz * CP_RANK) as u64));

        group.bench_function("coo_serial", |b| {
            b.iter(|| {
                let r = mttkrp_sparse_coo(&coo, &factor_views, MODE).unwrap();
                black_box(r);
            })
        });

        group.bench_function("csf_serial", |b| {
            b.iter(|| {
                let r = mttkrp_sparse_csf(&csf, &factor_views, MODE).unwrap();
                black_box(r);
            })
        });

        group.bench_function("hicoo_serial", |b| {
            b.iter(|| {
                let r = mttkrp_hicoo(&hicoo, &factor_views, MODE).unwrap();
                black_box(r);
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("coo_parallel", |b| {
            b.iter(|| {
                let r = mttkrp_sparse_coo_parallel(&coo, &factor_views, MODE).unwrap();
                black_box(r);
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("csf_parallel", |b| {
            b.iter(|| {
                let r = mttkrp_sparse_csf_parallel(&csf, &factor_views, MODE).unwrap();
                black_box(r);
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("hicoo_parallel", |b| {
            b.iter(|| {
                let r = mttkrp_hicoo_parallel(&hicoo, &factor_views, MODE).unwrap();
                black_box(r);
            })
        });

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Benchmark: sparse n-mode product vs dense
// ---------------------------------------------------------------------------

#[cfg(feature = "sparse")]
fn bench_nmode_sparse_vs_dense(c: &mut Criterion) {
    const MODE: usize = 0;

    let configs: &[([usize; 3], usize, f64, &str)] = &[
        ([50, 40, 30], 12, 0.01, "50x40x30_1pct_rows12"),
        ([50, 40, 30], 12, 0.05, "50x40x30_5pct_rows12"),
        ([80, 60, 50], 16, 0.01, "80x60x50_1pct_rows16"),
        ([80, 60, 50], 16, 0.05, "80x60x50_5pct_rows16"),
    ];

    for &(shape, out_rows, fill, label) in configs {
        let coo = build_coo(&shape, fill, 55);
        let nnz = coo.nnz();

        // Matrix: out_rows × shape[mode]
        let mut rng = 123u64;
        let mat_vals: Vec<f64> = (0..out_rows * shape[MODE])
            .map(|_| {
                rng = rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (rng >> 32) as f64 / (u32::MAX as f64) + 0.1
            })
            .collect();
        let matrix = Array2::from_shape_vec((out_rows, shape[MODE]), mat_vals).unwrap();

        // Dense baseline
        let dense = coo.to_dense().unwrap();

        let mut group = c.benchmark_group(format!("nmode_sparse_vs_dense/{}", label));
        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_function("dense_baseline", |b| {
            b.iter(|| {
                let r = tenrso_kernels::nmode::nmode_product(&dense.view(), &matrix.view(), MODE)
                    .unwrap();
                black_box(r);
            })
        });

        group.bench_function("coo_serial", |b| {
            b.iter(|| {
                let r = nmode_product_sparse_coo(&coo, &matrix.view(), MODE).unwrap();
                black_box(r);
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("coo_parallel", |b| {
            b.iter(|| {
                let r = nmode_product_sparse_coo_parallel(&coo, &matrix.view(), MODE).unwrap();
                black_box(r);
            })
        });

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// criterion_group + criterion_main
// ---------------------------------------------------------------------------

#[cfg(feature = "csf")]
criterion_group!(
    sparse_benches,
    bench_mttkrp_coo_vs_dense,
    bench_mttkrp_formats,
    bench_nmode_sparse_vs_dense,
);

// When compiled without csf but with sparse, exclude the csf-only group member.
#[cfg(all(feature = "sparse", not(feature = "csf")))]
criterion_group!(
    sparse_benches,
    bench_mttkrp_coo_vs_dense,
    bench_nmode_sparse_vs_dense,
);

criterion_main!(sparse_benches);
