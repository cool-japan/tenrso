//! Thread scaling of both batched-GEMM kernels.
//!
//! # Part 1 — the *generic* blocked kernel
//!
//! `f32`/`f64` never reach it: they are routed to `matrixmultiply`.  Every
//! *other* element type does, which for this workspace means integers and
//! `Complex` (the natural amplitude type for the tensor-network / quantum
//! consumers listed in the project overview):
//!
//! * `execute_dense_contraction`          → blocked kernel, serial
//! * `execute_dense_contraction_parallel` → blocked kernel, `(batch, row-block)` tasks
//! * `execute_dense_contraction_accelerated` → what the executor and the AD VJP
//!   rules actually call; for a non-`f32`/`f64` scalar it reaches the parallel
//!   kernel through the concrete `Any` downcast.
//!
//! # Part 2 — the *native* GEMM's batch loop
//!
//! `matrixmultiply` is single-threaded, so a batched `f32`/`f64` einsum used to
//! run on exactly one core however large the batch was.  The batch loop is now a
//! rayon loop, and its speedup is measured here by running the *same* call inside
//! rayon pools of different sizes.  Each batch element writes a disjoint output
//! slice and no dot product is split across tasks, so the result is bit-identical
//! at every thread count — which the run asserts, rather than assuming.
//!
//! Each configuration is timed `REPEATS` times; the median is reported together
//! with the min/max spread, because a shared build box is a noisy place to
//! measure.
//!
//! ```text
//! cargo run --release -p tenrso-exec --example blocked_gemm_scaling
//! ```

use anyhow::Result;
use scirs2_core::numeric::Complex64;
use scirs2_core::parallel_ops::{current_num_threads, ThreadPoolBuilder};
use std::time::{Duration, Instant};
use tenrso_core::DenseND;
use tenrso_exec::ops::{
    execute_dense_contraction, execute_dense_contraction_accelerated,
    execute_dense_contraction_parallel,
};
use tenrso_planner::EinsumSpec;

/// Timing runs per configuration (odd, so the median is a real sample).
const REPEATS: usize = 5;

/// Median, min and max of a set of samples.
fn summarize(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    (median, min, max)
}

fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

/// Time `f` `REPEATS` times, returning (median, min, max).
fn time_it<F, T>(mut f: F) -> Result<(Duration, Duration, Duration, T)>
where
    F: FnMut() -> Result<T>,
{
    let mut samples = Vec::with_capacity(REPEATS);
    let mut last = None;
    for _ in 0..REPEATS {
        let start = Instant::now();
        let value = f()?;
        samples.push(start.elapsed());
        last = Some(value);
    }
    let (median, min, max) = summarize(&mut samples);
    let value = last.ok_or_else(|| anyhow::anyhow!("REPEATS must be non-zero"))?;
    Ok((median, min, max, value))
}

/// One `(batch, m, k, n)` contraction, serial vs parallel vs dispatched.
fn scale<T>(
    label: &str,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    fill: impl Fn(usize) -> T,
) -> Result<()>
where
    T: Clone
        + scirs2_core::numeric::Num
        + std::ops::AddAssign
        + std::default::Default
        + Send
        + Sync
        + PartialEq
        + std::fmt::Debug
        + 'static,
{
    let spec = EinsumSpec::parse("bij,bjk->bik")?;
    let a = DenseND::from_vec((0..batch * m * k).map(&fill).collect(), &[batch, m, k])?;
    let b = DenseND::from_vec((0..batch * k * n).map(&fill).collect(), &[batch, k, n])?;

    let (ser_med, ser_min, ser_max, serial) = time_it(|| execute_dense_contraction(&spec, &a, &b))?;
    let (par_med, par_min, par_max, parallel) =
        time_it(|| execute_dense_contraction_parallel(&spec, &a, &b))?;
    let (dis_med, dis_min, dis_max, dispatched) =
        time_it(|| execute_dense_contraction_accelerated(&spec, &a, &b))?;

    assert_eq!(
        parallel.as_slice(),
        serial.as_slice(),
        "{label}: parallel kernel is not bit-identical to the serial one"
    );
    assert_eq!(
        dispatched.as_slice(),
        serial.as_slice(),
        "{label}: dispatched kernel is not bit-identical to the serial one"
    );

    let fma = (batch * m * k * n) as f64;
    println!(
        "{label}  batch={batch} m={m} k={k} n={n}  ({:.1} M FMA)",
        fma / 1e6
    );
    println!(
        "  serial      median {:>8.1} ms   [{:.1} .. {:.1}]",
        secs(ser_med) * 1e3,
        secs(ser_min) * 1e3,
        secs(ser_max) * 1e3
    );
    println!(
        "  parallel    median {:>8.1} ms   [{:.1} .. {:.1}]   speedup {:.2}×",
        secs(par_med) * 1e3,
        secs(par_min) * 1e3,
        secs(par_max) * 1e3,
        secs(ser_med) / secs(par_med)
    );
    println!(
        "  accelerated median {:>8.1} ms   [{:.1} .. {:.1}]   speedup {:.2}×",
        secs(dis_med) * 1e3,
        secs(dis_min) * 1e3,
        secs(dis_max) * 1e3,
        secs(ser_med) / secs(dis_med)
    );
    println!();
    Ok(())
}

/// Batch-loop scaling of the **native** (`matrixmultiply`) GEMM.
///
/// The same contraction is run inside rayon pools of increasing size.  One thread
/// is the old behaviour — `matrixmultiply` is single-threaded, so the whole batch
/// was serial — and every larger pool is the batch loop actually forking.
fn native_batch_scaling(batch: usize, m: usize, k: usize, n: usize) -> Result<()> {
    let spec = EinsumSpec::parse("bij,bjk->bik")?;
    let a = DenseND::from_vec(
        (0..batch * m * k)
            .map(|i| (i % 17) as f64 * 0.5 - 4.0)
            .collect(),
        &[batch, m, k],
    )?;
    let b = DenseND::from_vec(
        (0..batch * k * n)
            .map(|i| (i % 13) as f64 * 0.25 - 1.0)
            .collect(),
        &[batch, k, n],
    )?;

    let fma = (batch * m * k * n) as f64;
    println!("native f64 GEMM, batch loop over rayon");
    println!(
        "  batch={batch} m={m} k={k} n={n}  ({:.1} M FMA)",
        fma / 1e6
    );

    let max_threads = current_num_threads().max(1);
    let mut baseline: Option<Duration> = None;
    let mut reference: Option<Vec<f64>> = None;

    // 1 thread is the pre-change behaviour; the rest show the batch loop forking.
    let mut ladder: Vec<usize> = [1, 2, 4, max_threads]
        .into_iter()
        .filter(|t| *t <= max_threads)
        .collect();
    ladder.sort_unstable();
    ladder.dedup();

    for threads in ladder {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        let (median, min, max, result) =
            pool.install(|| time_it(|| execute_dense_contraction_accelerated(&spec, &a, &b)))?;

        // Bit-identity across thread counts: no dot product was split, so no
        // output element's summation order can have moved.
        match &reference {
            None => reference = Some(result.as_slice().to_vec()),
            Some(want) => {
                for (got, want) in result.as_slice().iter().zip(want.iter()) {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "native batch GEMM is not bit-identical on {threads} threads"
                    );
                }
            }
        }

        let speedup = match baseline {
            None => {
                baseline = Some(median);
                1.0
            }
            Some(base) => secs(base) / secs(median),
        };
        println!(
            "  {threads:>2} thread(s)  median {:>8.2} ms   [{:.2} .. {:.2}]   speedup {speedup:.2}×   {:.1} GFLOP/s",
            secs(median) * 1e3,
            secs(min) * 1e3,
            secs(max) * 1e3,
            2.0 * fma / secs(median) / 1e9,
        );
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    println!(
        "rayon threads: {}   repeats per configuration: {REPEATS}\n",
        current_num_threads()
    );

    // The native GEMM's batch loop: the only source of parallelism for f32/f64.
    native_batch_scaling(32, 128, 128, 128)?;
    native_batch_scaling(8, 256, 256, 256)?;

    // Single large matrix: the row-block split is the *only* source of
    // parallelism (batch == 1).
    scale("Complex64 single ", 1, 512, 512, 512, |i| {
        Complex64::new((i % 17) as f64 * 0.5 - 4.0, (i % 11) as f64 * 0.25)
    })?;

    // Batched: parallelism comes from the batch as well.
    scale("Complex64 batched", 8, 192, 192, 192, |i| {
        Complex64::new((i % 13) as f64 - 6.0, (i % 7) as f64 * 0.5)
    })?;

    // Integers: a cheap element type, so the kernel is far more
    // memory-bandwidth-bound — the interesting stress case for scaling.
    scale("i64 single       ", 1, 512, 512, 512, |i| {
        (i % 19) as i64 - 9
    })?;

    scale("i64 batched      ", 8, 192, 192, 192, |i| {
        (i % 23) as i64 - 11
    })?;

    Ok(())
}
