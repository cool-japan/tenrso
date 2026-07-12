//! `cp_als` vs `cp_als_accelerated` — **time to solution**: iterations *and* wall
//! clock to reach the *same* fit target.
//!
//! Comparing the two drivers over a fixed iteration budget is misleading, because an
//! accelerated sweep costs more but each one makes more progress — they end at
//! *different* fits. The honest question is "how long to reach a given quality", so
//! this example, for each tensor family:
//!
//! 1. runs both drivers to a large budget and records their fit histories;
//! 2. picks a common target fit (a hair below the worse of the two final fits);
//! 3. reports, for each driver, the *iterations* to first reach that target and the
//!    *wall clock* of a run truncated to exactly that many iterations (median of
//!    `REPS` reps — the box is contended, so a single number would be noise).
//!
//! Run with:
//!
//! ```text
//! cargo run -p tenrso-decomp --release --example cp_els_bench
//! ```

use std::time::{Duration, Instant};

use scirs2_core::ndarray_ext::{Array, IxDyn};
use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, cp_als_accelerated, CpDecomp, CpError, InitStrategy};

/// Number of timing repetitions; we report the median and the min/max spread.
const REPS: usize = 5;

/// Deterministic linear congruential generator.
///
/// The example must be reproducible run-to-run, and `scirs2_core::random`'s
/// thread RNG is not seedable from here. This is *example* scaffolding for
/// generating input data, not a library RNG.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Build `[[A_0, ..., A_{N-1}]] + noise` where every factor matrix has columns of
/// pairwise correlation `collinearity`.
///
/// `collinearity` near 1 is the classical CP-ALS **swamp**: the Gram matrices become
/// near-singular, the ALS steps become long and nearly parallel across sweeps, and
/// plain ALS crawls. This is precisely the regime line-search acceleration exists for.
fn collinear_cp_tensor(
    shape: &[usize],
    rank: usize,
    collinearity: f64,
    noise: f64,
    seed: u64,
) -> DenseND<f64> {
    let mut rng = Lcg::new(seed);
    let n_modes = shape.len();

    let mut factors: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_modes);
    for &dim in shape {
        // Shared direction + an independent component per column.
        let base: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(rank);
        for _ in 0..rank {
            let indep: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
            let mut col: Vec<f64> = (0..dim)
                .map(|i| {
                    collinearity * base[i] + (1.0 - collinearity * collinearity).sqrt() * indep[i]
                })
                .collect();
            let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            for v in &mut col {
                *v /= norm;
            }
            cols.push(col);
        }
        factors.push(cols);
    }

    let mut data = Array::<f64, IxDyn>::zeros(IxDyn(shape));
    let numel: usize = shape.iter().product();
    let mut index = vec![0usize; n_modes];
    for flat in 0..numel {
        let mut rem = flat;
        for mode in (0..n_modes).rev() {
            index[mode] = rem % shape[mode];
            rem /= shape[mode];
        }
        let mut value = 0.0;
        for r in 0..rank {
            let mut term = 1.0;
            for (mode, &i) in index.iter().enumerate() {
                term *= factors[mode][r][i];
            }
            value += term;
        }
        data[IxDyn(&index)] = value + noise * rng.normal();
    }

    DenseND::from_array(data)
}

fn uniform_tensor(shape: &[usize], seed: u64) -> DenseND<f64> {
    let mut rng = Lcg::new(seed);
    let numel: usize = shape.iter().product();
    let values: Vec<f64> = (0..numel).map(|_| rng.uniform()).collect();
    let data = Array::<f64, IxDyn>::from_shape_vec(IxDyn(shape), values)
        .expect("shape and value count agree by construction");
    DenseND::from_array(data)
}

/// First iteration index (1-based, matching `CpDecomp::iters`) whose fit reaches
/// `target`, or the total sweep count if the target is never met inside the budget.
fn iters_to_target(history: &[f64], target: f64) -> usize {
    history
        .iter()
        .position(|&f| f >= target)
        .map(|i| i + 1)
        .unwrap_or_else(|| history.len().max(1))
}

fn fit_history(decomp: &CpDecomp<f64>) -> Vec<f64> {
    decomp
        .convergence
        .as_ref()
        .map(|c| c.fit_history.clone())
        .unwrap_or_else(|| vec![decomp.fit])
}

/// Median and min/max of `REPS` timed runs of `run`, each truncated to `iters` sweeps.
fn time_to(iters: usize, mut run: impl FnMut(usize) -> f64) -> (Duration, Duration, Duration) {
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let _ = run(iters);
        times.push(start.elapsed());
    }
    times.sort_unstable();
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

fn report(name: &str, tensor: &DenseND<f64>, rank: usize, max_iters: usize) {
    let init = InitStrategy::Svd;
    // Tiny tol so neither driver stops early: we want the full fit trajectory.
    let tiny_tol = 1e-12;

    let run_plain = |budget: usize| -> Result<CpDecomp<f64>, CpError> {
        cp_als(tensor, rank, budget, tiny_tol, init, None)
    };
    let run_accel = |budget: usize| -> Result<CpDecomp<f64>, CpError> {
        cp_als_accelerated(tensor, rank, budget, tiny_tol, init, None)
    };

    let plain_full = run_plain(max_iters).expect("cp_als should succeed");
    let accel_full = run_accel(max_iters).expect("cp_als_accelerated should succeed");
    let plain_hist = fit_history(&plain_full);
    let accel_hist = fit_history(&accel_full);

    // Common target: just under the worse of the two attained fits, so BOTH drivers can
    // actually reach it. This is the fit both are asked to deliver "as fast as possible".
    let target = plain_full.fit.min(accel_full.fit) - 1e-4;

    let plain_iters = iters_to_target(&plain_hist, target);
    let accel_iters = iters_to_target(&accel_hist, target);

    let (p_med, p_lo, p_hi) = time_to(plain_iters, |b| run_plain(b).expect("cp_als").fit);
    let (a_med, a_lo, a_hi) = time_to(accel_iters, |b| {
        run_accel(b).expect("cp_als_accelerated").fit
    });

    let iter_speedup = plain_iters as f64 / accel_iters.max(1) as f64;
    let wall_speedup = p_med.as_secs_f64() / a_med.as_secs_f64().max(1e-12);

    println!("{name}  shape={:?} rank={rank}", tensor.shape());
    println!(
        "  target fit {target:.6}   (final: plain {:.6}, accel {:.6})",
        plain_full.fit, accel_full.fit
    );
    println!(
        "  cp_als             iters->target {plain_iters:>3}   wall={:>9.2?} [{:.2?}..{:.2?}]",
        p_med, p_lo, p_hi
    );
    println!(
        "  cp_als_accelerated iters->target {accel_iters:>3}   wall={:>9.2?} [{:.2?}..{:.2?}]",
        a_med, a_lo, a_hi
    );
    println!(
        "  -> iterations-to-target speedup {iter_speedup:.2}x   \
         wall-clock-to-target speedup {wall_speedup:.2}x"
    );
    println!();
}

fn main() {
    println!(
        "cp_als vs cp_als_accelerated  —  time to a common fit target\n\
         (median of {REPS} reps, contended box)\n"
    );

    report(
        "uniform-random          ",
        &uniform_tensor(&[40, 40, 40], 11),
        10,
        300,
    );
    report(
        "low-rank + noise        ",
        &collinear_cp_tensor(&[40, 40, 40], 5, 0.0, 0.01, 23),
        5,
        300,
    );
    report(
        "swamp c=0.9             ",
        &collinear_cp_tensor(&[40, 40, 40], 5, 0.9, 0.0, 7),
        5,
        300,
    );
    report(
        "swamp c=0.99            ",
        &collinear_cp_tensor(&[40, 40, 40], 5, 0.99, 0.0, 7),
        5,
        300,
    );
    report(
        "swamp c=0.99 + noise    ",
        &collinear_cp_tensor(&[40, 40, 40], 5, 0.99, 1e-3, 13),
        5,
        300,
    );
    report(
        "swamp c=0.95 4-way      ",
        &collinear_cp_tensor(&[16, 16, 16, 16], 4, 0.95, 0.0, 31),
        4,
        300,
    );
}
