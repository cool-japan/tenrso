//! Benchmark harness: masked einsum vs dense naive matmul
//!
//! Blueprint target: "Masked einsum: >= 5x speedup vs dense naive at 90% zeros."
//!
//! Tests `masked_einsum("ij,jk->ik", ...)` against a naive triple-loop matmul
//! at various sparsity levels (50%, 90%, 99%) and matrix sizes (64, 256, 512).
//!
//! # Fairness of the comparison
//!
//! The point of the target is that *skipping masked-out output positions* buys real
//! time. A ratio is evidence of that only if the two sides differ in the work they
//! do and in nothing else. Concretely:
//!
//! * **Same computation.** Both sides compute `C = A @ B` over `f64`; the masked
//!   side computes the subset of `C` selected by the mask, the dense side computes
//!   all of `C`. `assert_masked_matches_dense` checks, before any timing is taken,
//!   that the masked result equals the dense result at every mask position — a
//!   masked path that "won" by silently skipping arithmetic would fail here rather
//!   than post a fast time.
//! * **Same element-access mechanism.** The baseline indexes raw contiguous
//!   `&[f64]` slices, exactly as `masked_matmul` does. An earlier version of this
//!   harness indexed through `ArrayView<_, IxDyn>` as `view[&[i, j][..]]`, which
//!   costs a dynamic stride dot-product plus bounds checks *per element*. That made
//!   the "dense naive" baseline several times slower than a real naive loop and
//!   inflated the reported speedup by a factor that says nothing about masking.
//! * **Same inner-loop shape.** Both sides accumulate a `K`-long dot product in a
//!   plain scalar loop, in the same order, with no manual unrolling and no multiple
//!   accumulators on either side. Vectorising only the masked side would inflate the
//!   ratio for reasons unrelated to the mask.
//! * **Nothing elided.** Inputs and results pass through `black_box`.
//!
//! The baseline remains *naive* in the sense the target intends: a plain `O(M*N*K)`
//! triple loop — no blocking, no packing, no BLAS.

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

/// Compute `C = A * B` via a straightforward triple loop over contiguous slices.
///
/// This is the "dense naive" reference: `O(M*N*K)`, no blocking, no packing, no
/// BLAS, one scalar accumulator. It deliberately uses the *same* element-access
/// mechanism (direct indexing into a contiguous `&[f64]`) and the *same* scalar
/// inner-loop shape as `masked_matmul`, so that a measured difference reflects the
/// number of output positions computed rather than indexing overhead.
fn dense_matmul(a: &DenseND<f64>, b: &DenseND<f64>) -> DenseND<f64> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let a_slice = a
        .try_as_slice()
        .expect("dense_matmul: A must be contiguous");
    let b_slice = b
        .try_as_slice()
        .expect("dense_matmul: B must be contiguous");

    let mut out = vec![0.0_f64; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a_slice[i * k + p] * b_slice[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }

    DenseND::from_vec(out, &[m, n]).expect("dense_matmul: output shape matches data length")
}

/// Verify the masked path computes the same values as the dense baseline at every
/// mask position, before any timing is taken.
///
/// Guards against the failure mode where the masked kernel "wins" by not doing the
/// arithmetic at all.
fn assert_masked_matches_dense(a: &DenseND<f64>, b: &DenseND<f64>, mask: &Mask) {
    let dense = dense_matmul(a, b);
    let masked = masked_einsum("ij,jk->ik", &[a, b], mask).expect("masked_einsum failed");
    let masked_dense = masked.to_dense().expect("masked result to_dense failed");

    for idx in mask.iter() {
        let expected = dense.as_array()[&idx[..]];
        let actual = masked_dense.as_array()[&idx[..]];
        assert!(
            (expected - actual).abs() <= 1e-9 * expected.abs().max(1.0),
            "masked/dense mismatch at {:?}: dense={}, masked={}",
            idx,
            expected,
            actual
        );
    }
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
/// parameter label so `critcmp` / Criterion HTML reports can directly show the
/// speedup ratio. `.github/scripts/check_perf_budgets.py` reads the
/// `matmul_comparison/{dense,masked}/<size>x<size>_sp90` estimates from this group
/// to gate the ">= 5x at 90% sparsity" budget.
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/comparison");

    for &(size, sparsity) in PARAMS {
        let mut lcg = Lcg::new(42);
        let a = make_dense(&mut lcg, size, size);
        let b = make_dense(&mut lcg, size, size);

        let density = 1.0 - sparsity;
        let mask = make_mask(&mut lcg, size, size, density);

        // Fairness precondition: both sides must agree on the values they produce.
        assert_masked_matches_dense(&a, &b, &mask);

        let param_label = format!("{}x{}_sp{:.0}", size, size, sparsity * 100.0);

        // Dense baseline
        group.bench_with_input(BenchmarkId::new("dense", &param_label), &(), |bench, _| {
            bench.iter(|| {
                black_box(dense_matmul(black_box(&a), black_box(&b)));
            });
        });

        // Masked einsum
        group.bench_with_input(BenchmarkId::new("masked", &param_label), &(), |bench, _| {
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

criterion_group!(
    benches,
    bench_dense_naive,
    bench_masked_einsum,
    bench_comparison
);
criterion_main!(benches);
