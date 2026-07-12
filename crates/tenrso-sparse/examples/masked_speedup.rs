//! Measure masked einsum speedup vs a dense naive matmul.
//!
//! Settles the blueprint target: "Masked operations: >= 5x speedup vs dense naive"
//! (documented at 90% zeros). Run with:
//!
//! ```text
//! cargo run --release -p tenrso-sparse --example masked_speedup
//! ```
//!
//! # Why a direct timing loop rather than Criterion alone
//!
//! The reported figure is a *ratio* between two kernels. On a shared, loaded
//! machine, timing the two sides in separate Criterion runs lets background load
//! drift between them and corrupt the ratio. Here every variant is timed inside one
//! process, round-robin (repetition is the outer loop, variant the inner), so all
//! variants see the same machine conditions and the per-variant median cancels
//! transient load spikes.
//!
//! # Fairness of the comparison
//!
//! * Every variant computes the same `C = A @ B` over `f64`; the masked variants
//!   compute only the mask-selected subset. `check` verifies each variant against
//!   the dense baseline at every mask position *before* timing, so a variant cannot
//!   "win" by skipping arithmetic.
//! * `dense_naive` and the masked kernels use the same element-access mechanism
//!   (indexing a contiguous `&[f64]`) and the same scalar dot-product inner loop —
//!   no unrolling or vectorisation on one side only.
//! * `dense_naive_viewidx` is included solely to quantify how much a benchmark that
//!   indexes the baseline through `ArrayView<_, IxDyn>` overstates the speedup.
//! * Inputs and results pass through `black_box`.

use std::hint::black_box;
use std::time::Instant;
use tenrso_core::DenseND;
use tenrso_sparse::mask::Mask;
use tenrso_sparse::masked_einsum::masked_einsum;

/// Reproducible LCG (same generator as the Criterion harness).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 1_000_000) as f64 / 1_000_000.0
    }
}

fn make_dense(lcg: &mut Lcg, rows: usize, cols: usize) -> DenseND<f64> {
    let data: Vec<f64> = (0..rows * cols).map(|_| lcg.next_f64()).collect();
    DenseND::from_vec(data, &[rows, cols]).expect("make_dense: shape matches data length")
}

fn make_mask(lcg: &mut Lcg, m: usize, n: usize, density: f64) -> Mask {
    let mut indices: Vec<Vec<usize>> = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if lcg.next_f64() < density {
                indices.push(vec![i, j]);
            }
        }
    }
    if indices.is_empty() {
        indices.push(vec![0, 0]);
    }
    Mask::from_indices(indices, vec![m, n]).expect("make_mask: indices within bounds")
}

/// Dense naive triple loop over contiguous slices. The baseline.
///
/// `O(M*N*K)`, no blocking, no packing, no BLAS, one scalar accumulator.
fn dense_naive(a: &DenseND<f64>, b: &DenseND<f64>) -> Vec<f64> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    let a_s = a.try_as_slice().expect("A contiguous");
    let b_s = b.try_as_slice().expect("B contiguous");

    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a_s[i * k + p] * b_s[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// The same naive triple loop, but indexing through `ArrayView<_, IxDyn>`.
///
/// This is what the old benchmark used as its "dense naive" baseline. Included only
/// to quantify how badly that overstates the masked speedup.
fn dense_naive_viewidx(a: &DenseND<f64>, b: &DenseND<f64>) -> Vec<f64> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    let av = a.view();
    let bv = b.view();

    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += av[&[i, p][..]] * bv[&[p, j][..]];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// The *previously shipped* masked matmul: for each masked `(i, k)`, walk `B[j, k]`
/// down a column with stride `N`, indexing through `ArrayView<_, IxDyn>`.
///
/// Reimplemented here (rather than reverting the library) so old and new can be
/// timed side by side in one process.
fn masked_old(a: &DenseND<f64>, b: &DenseND<f64>, mask: &Mask) -> Vec<(Vec<usize>, f64)> {
    let a_view = a.view();
    let b_view = b.view();
    let k_dim = a.shape()[1];

    let sorted = mask.to_sorted_indices();
    let mut out = Vec::with_capacity(sorted.len());
    for idx in &sorted {
        let (i, k) = (idx[0], idx[1]);
        let mut acc = 0.0_f64;
        for j in 0..k_dim {
            acc += a_view[&[i, j][..]] * b_view[&[j, k][..]];
        }
        if acc.abs() > f64::EPSILON {
            out.push((idx.clone(), acc));
        }
    }
    out
}

/// Column-walking masked matmul on raw slices: isolates the *cache* effect of the
/// scattered column gather from the `IxDyn`-indexing effect.
fn masked_columnwalk_slices(
    a: &DenseND<f64>,
    b: &DenseND<f64>,
    mask: &Mask,
) -> Vec<(Vec<usize>, f64)> {
    let k_dim = a.shape()[1];
    let n_cols = b.shape()[1];
    let a_s = a.try_as_slice().expect("A contiguous");
    let b_s = b.try_as_slice().expect("B contiguous");

    let mut positions: Vec<(usize, usize)> = mask.iter().map(|idx| (idx[0], idx[1])).collect();
    positions.sort_unstable();

    let mut out = Vec::with_capacity(positions.len());
    for &(i, k) in &positions {
        let mut acc = 0.0_f64;
        for j in 0..k_dim {
            acc += a_s[i * k_dim + j] * b_s[j * n_cols + k];
        }
        if acc.abs() > f64::EPSILON {
            out.push((vec![i, k], acc));
        }
    }
    out
}

/// Verify a masked result against the dense baseline at every mask position.
fn check(label: &str, dense: &[f64], n_cols: usize, mask: &Mask, got: &[(Vec<usize>, f64)]) {
    assert_eq!(
        got.len(),
        mask.nnz(),
        "{label}: produced {} values for {} mask positions",
        got.len(),
        mask.nnz()
    );
    for (idx, val) in got {
        let expected = dense[idx[0] * n_cols + idx[1]];
        assert!(
            (expected - *val).abs() <= 1e-9 * expected.abs().max(1.0),
            "{label}: mismatch at {idx:?}: dense={expected}, masked={val}"
        );
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn spread(xs: &[f64]) -> (f64, f64) {
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

fn main() {
    let params: &[(usize, f64)] = &[
        (64, 0.50),
        (64, 0.90),
        (64, 0.99),
        (256, 0.50),
        (256, 0.90),
        (256, 0.99),
        (512, 0.50),
        (512, 0.90),
        (512, 0.99),
    ];

    println!("masked einsum vs dense naive matmul  (f64, ij,jk->ik)");
    println!("all variants timed round-robin in one process; medians over N reps\n");
    println!(
        "{:>5} {:>6} {:>8} | {:>10} {:>10} {:>10} {:>10} | {:>9} {:>9} {:>9}",
        "size",
        "sp%",
        "nnz",
        "dense(ms)",
        "old(ms)",
        "colwalk",
        "new(ms)",
        "old/dn",
        "cw/dn",
        "NEW/dn"
    );
    println!("{}", "-".repeat(112));

    for &(size, sparsity) in params {
        let mut lcg = Lcg::new(42);
        let a = make_dense(&mut lcg, size, size);
        let b = make_dense(&mut lcg, size, size);
        let mask = make_mask(&mut lcg, size, size, 1.0 - sparsity);

        // ---- correctness gate: every variant must agree with the baseline ----
        let reference = dense_naive(&a, &b);
        assert_eq!(
            reference,
            dense_naive_viewidx(&a, &b),
            "viewidx baseline differs"
        );
        check("old", &reference, size, &mask, &masked_old(&a, &b, &mask));
        check(
            "colwalk",
            &reference,
            size,
            &mask,
            &masked_columnwalk_slices(&a, &b, &mask),
        );
        let new_coo = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("masked_einsum");
        let new_pairs: Vec<(Vec<usize>, f64)> = new_coo
            .indices()
            .iter()
            .cloned()
            .zip(new_coo.values().iter().copied())
            .collect();
        check("new", &reference, size, &mask, &new_pairs);

        // ---- timing ----
        let reps = if size >= 512 { 3 } else { 5 };
        let (mut t_dense, mut t_old, mut t_cw, mut t_new) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());

        for _ in 0..reps {
            let t = Instant::now();
            black_box(dense_naive(black_box(&a), black_box(&b)));
            t_dense.push(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            black_box(masked_old(black_box(&a), black_box(&b), black_box(&mask)));
            t_old.push(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            black_box(masked_columnwalk_slices(
                black_box(&a),
                black_box(&b),
                black_box(&mask),
            ));
            t_cw.push(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            black_box(
                masked_einsum(
                    black_box("ij,jk->ik"),
                    black_box(&[&a, &b]),
                    black_box(&mask),
                )
                .expect("masked_einsum"),
            );
            t_new.push(t.elapsed().as_secs_f64() * 1e3);
        }

        let (dn, old, cw, new) = (
            median(t_dense.clone()),
            median(t_old.clone()),
            median(t_cw.clone()),
            median(t_new.clone()),
        );
        let (dlo, dhi) = spread(&t_dense);
        let (nlo, nhi) = spread(&t_new);

        println!(
            "{:>5} {:>6.0} {:>8} | {:>10.3} {:>10.3} {:>10.3} {:>10.3} | {:>8.2}x {:>8.2}x {:>8.2}x",
            size,
            sparsity * 100.0,
            mask.nnz(),
            dn,
            old,
            cw,
            new,
            dn / old,
            dn / cw,
            dn / new,
        );
        if (sparsity - 0.90).abs() < 1e-9 {
            println!(
                "        ^ 90% GATE: NEW = {:.2}x  [dense reps {:.2}-{:.2} ms, new reps {:.3}-{:.3} ms]  => {}",
                dn / new,
                dlo,
                dhi,
                nlo,
                nhi,
                if dn / new >= 5.0 { "PASS (>=5x)" } else { "FAIL (<5x)" }
            );
        }
    }
}
