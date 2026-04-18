//! Benchmark harness: masked einsum vs dense naive matmul
//!
//! Blueprint target: "Masked einsum: >= 5x speedup vs dense naive at 90% zeros."
//!
//! Tests `masked_einsum("ij,jk->ik", ...)` against a naive triple-loop matmul
//! at various sparsity levels (50%, 90%, 99%) and matrix sizes (64, 256, 512).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tenrso_core::DenseND;
use tenrso_sparse::mask::Mask;
use tenrso_sparse::masked_einsum::masked_einsum;

// ---------------------------------------------------------------------------
// LCG-based reproducible pseudo-random generation (matches sparse_ops.rs)
// ---------------------------------------------------------------------------

/// Simple LCG state for reproducible benchmarks.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance and return the raw state.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    /// Return a value in `[0.0, 1.0)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 1_000_000) as f64 / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

/// Build a random `DenseND<f64>` of the given shape using the provided LCG.
fn make_dense(lcg: &mut Lcg, rows: usize, cols: usize) -> DenseND<f64> {
    let n = rows * cols;
    let data: Vec<f64> = (0..n).map(|_| lcg.next_f64()).collect();
    DenseND::from_vec(data, &[rows, cols]).expect("make_dense: shape matches data length")
}

/// Build a `Mask` for an `m x n` output with approximately `density` fraction
/// of entries active (density = 1 - sparsity, e.g. 0.1 for 90% zeros).
fn make_mask(lcg: &mut Lcg, m: usize, n: usize, density: f64) -> Mask {
    let mut indices: Vec<Vec<usize>> = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if lcg.next_f64() < density {
                indices.push(vec![i, j]);
            }
        }
    }
    // Ensure at least one entry so the masked path does real work.
    if indices.is_empty() {
        indices.push(vec![0, 0]);
    }
    Mask::from_indices(indices, vec![m, n]).expect("make_mask: indices within bounds")
}

// ---------------------------------------------------------------------------
// Dense naive triple-loop matmul (the baseline we want to beat)
// ---------------------------------------------------------------------------

/// Compute C = A * B via a straightforward triple loop.
///
/// Uses `DenseND::view()` indexing, the same per-element work as
/// `masked_matmul` — the only difference is that this computes *all*
/// output positions while the masked path computes a subset.
fn dense_matmul(a: &DenseND<f64>, b: &DenseND<f64>) -> DenseND<f64> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let a_view = a.view();
    let b_view = b.view();

    let mut out = vec![0.0_f64; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a_view[&[i, p][..]] * b_view[&[p, j][..]];
            }
            out[i * n + j] = acc;
        }
    }

    DenseND::from_vec(out, &[m, n]).expect("dense_matmul: output shape matches data length")
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

/// Benchmark parameters: (matrix_size, sparsity_fraction).
/// Sparsity fraction = fraction of *zeros* in the output mask.
const PARAMS: &[(usize, f64)] = &[
    // 64x64
    (64, 0.50),
    (64, 0.90),
    (64, 0.99),
    // 256x256
    (256, 0.50),
    (256, 0.90),
    (256, 0.99),
    // 512x512
    (512, 0.50),
    (512, 0.90),
    (512, 0.99),
];

fn bench_dense_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/dense_naive");

    for &(size, sparsity) in PARAMS {
        let mut lcg = Lcg::new(42);
        let a = make_dense(&mut lcg, size, size);
        let b = make_dense(&mut lcg, size, size);

        let label = format!("{}x{}_sparsity_{:.0}pct", size, size, sparsity * 100.0);

        group.bench_with_input(BenchmarkId::new("dense", &label), &(), |bench, _| {
            bench.iter(|| {
                let result = dense_matmul(black_box(&a), black_box(&b));
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_masked_einsum(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/masked_einsum");

    for &(size, sparsity) in PARAMS {
        let mut lcg = Lcg::new(42);
        let a = make_dense(&mut lcg, size, size);
        let b = make_dense(&mut lcg, size, size);

        // density = 1 - sparsity  (fraction of nonzeros in the mask)
        let density = 1.0 - sparsity;
        let mask = make_mask(&mut lcg, size, size, density);

        let label = format!(
            "{}x{}_sparsity_{:.0}pct_nnz_{}",
            size,
            size,
            sparsity * 100.0,
            mask.nnz()
        );

        group.bench_with_input(BenchmarkId::new("masked", &label), &(), |bench, _| {
            bench.iter(|| {
                let _ = black_box(masked_einsum(
                    black_box("ij,jk->ik"),
                    black_box(&[&a, &b]),
                    black_box(&mask),
                ));
            });
        });
    }

    group.finish();
}

/// Side-by-side comparison at each (size, sparsity) pair.
///
/// This group interleaves "dense" and "masked" benchmarks under the same
/// parameter label so `critcmp` / Criterion HTML reports can directly show
/// the speedup ratio.
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/comparison");

    for &(size, sparsity) in PARAMS {
        let mut lcg = Lcg::new(42);
        let a = make_dense(&mut lcg, size, size);
        let b = make_dense(&mut lcg, size, size);

        let density = 1.0 - sparsity;
        let mask = make_mask(&mut lcg, size, size, density);

        let param_label = format!("{}x{}_sp{:.0}", size, size, sparsity * 100.0);

        // Dense baseline
        group.bench_with_input(
            BenchmarkId::new("dense", &param_label),
            &(),
            |bench, _| {
                bench.iter(|| {
                    black_box(dense_matmul(black_box(&a), black_box(&b)));
                });
            },
        );

        // Masked einsum
        group.bench_with_input(
            BenchmarkId::new("masked", &param_label),
            &(),
            |bench, _| {
                bench.iter(|| {
                    let _ = black_box(masked_einsum(
                        black_box("ij,jk->ik"),
                        black_box(&[&a, &b]),
                        black_box(&mask),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dense_naive, bench_masked_einsum, bench_comparison);
criterion_main!(benches);
