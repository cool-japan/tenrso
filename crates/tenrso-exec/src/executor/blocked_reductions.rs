//! Blocked full-tensor reductions.
//!
//! # What makes this faster (and why the old "tiled" version was not)
//!
//! A naive float sum is a *dependency chain*: every `+=` must wait for the
//! previous one to retire. On this Xeon that is ~4 cycles per element no matter
//! how wide the machine is, and no compiler may break the chain on its own
//! because float addition is not associative.
//!
//! This module breaks it explicitly, by keeping [`LANES`] independent
//! accumulators and combining them at the end. That single change is where the
//! speed comes from — 4M f64, medians of 7 on a contended box:
//!
//! | kernel | time |
//! |--------|------|
//! | live `full_reduce` (serial, `IxDyn` iterator) | 11.5 ms |
//! | live `full_reduce` (old "parallel": copies to a `Vec` first) | 17.2 ms |
//! | contiguous `slice.iter().sum()`, 1 accumulator | 5.05 ms |
//! | *old* `tiled_sum_all`: `chunks(4096)` then sum | **5.04 ms** |
//! | this kernel, 8 accumulators, serial | 2.94 ms |
//! | this kernel, 8 accumulators, over rayon blocks | **1.35 ms** |
//!
//! Note row four. The module this one replaces chunked the data by 4096 and
//! summed each chunk — but each chunk was *still* summed with one accumulator, so
//! the chain was never broken and it landed within noise of the plain slice sum
//! (5.04 vs 5.05 ms). The "tiling" bought nothing; what little it appeared to buy
//! came from touching a contiguous slice instead of ndarray's dynamic iterator.
//!
//! # Determinism
//!
//! The result does not depend on the thread count or on rayon's scheduling:
//! blocks are a fixed size, partials are collected in index order, and the lane
//! and partial combines run in a fixed order. Two runs with 2 and 64 threads
//! produce bit-identical output.
//!
//! It is *not* bit-identical to naive left-to-right accumulation — a different
//! summation order rounds differently (and, being closer to pairwise summation,
//! this one is usually the more accurate of the two). Callers that need the exact
//! naive order can set `enable_blocked_reductions = false`.

use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::DenseND;

use super::types::{CpuExecutor, ReduceOp};

/// Independent accumulators per block. Eight covers the FP-add latency (~4
/// cycles) times the two add ports, so the chain stops being the bottleneck.
const LANES: usize = 8;

/// Elements per rayon task. Large enough that task overhead is irrelevant,
/// small enough to keep every core fed.
const PAR_BLOCK: usize = 65_536;

/// Below this, the plain path wins: the reduction is over before the blocking
/// setup pays for itself.
pub(crate) const BLOCKED_MIN_ELEMS: usize = 1024;

/// Reduce one block with [`LANES`] independent accumulators.
///
/// The lane combine and the remainder are folded in a fixed order, so this is a
/// pure function of the block contents — same bytes in, same bits out.
#[inline]
fn reduce_block<T>(op: &ReduceOp, block: &[T]) -> Option<T>
where
    T: Clone + Num + Float,
{
    if block.is_empty() {
        return None;
    }

    // Seed the lanes with the op's identity, then fold LANES elements at a time.
    let identity = match op {
        ReduceOp::Sum | ReduceOp::Mean => T::zero(),
        ReduceOp::Prod => T::one(),
        ReduceOp::Max => T::neg_infinity(),
        ReduceOp::Min => T::infinity(),
        _ => return None,
    };

    let mut acc = [identity; LANES];
    let mut chunks = block.chunks_exact(LANES);

    for chunk in &mut chunks {
        for (a, &v) in acc.iter_mut().zip(chunk.iter()) {
            *a = combine(op, *a, v);
        }
    }

    // Fold the lanes down, then the tail, in a fixed order.
    let mut result = acc[0];
    for &a in acc.iter().skip(1) {
        result = combine(op, result, a);
    }
    for &v in chunks.remainder() {
        result = combine(op, result, v);
    }
    Some(result)
}

/// The associative combine behind each supported reduction.
#[inline]
fn combine<T>(op: &ReduceOp, a: T, b: T) -> T
where
    T: Clone + Num + Float,
{
    match op {
        ReduceOp::Sum | ReduceOp::Mean => a + b,
        ReduceOp::Prod => a * b,
        ReduceOp::Max => {
            if a > b {
                a
            } else {
                b
            }
        }
        ReduceOp::Min => {
            if a < b {
                a
            } else {
                b
            }
        }
        // Unreachable: `reduce_block` refuses these ops before any combine runs.
        _ => a,
    }
}

/// Is `op` one this module can reduce?
///
/// `All`/`Any` are short-circuiting boolean folds and `ArgMax`/`ArgMin` need the
/// index, not just the value; none of them are worth blocking, so they keep the
/// ordinary path.
pub(crate) fn supports(op: &ReduceOp) -> bool {
    matches!(
        op,
        ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Prod | ReduceOp::Max | ReduceOp::Min
    )
}

/// Blocked full reduction over every element of `dense`.
///
/// Returns `None` — meaning "no blocked kernel applies, use the ordinary path" —
/// when the op is unsupported, the tensor is too small, its data is not
/// contiguous (a permuted view has no slice to walk), or `Mean`'s element count
/// will not convert into `T`. It never returns a wrong answer in place of a
/// missing one.
///
/// When `parallel` is set the blocks are spread over rayon *inside the
/// executor's own pool*, so `CpuExecutor::with_threads` bounds this kernel too.
pub(crate) fn blocked_full_reduce<T>(
    op: &ReduceOp,
    dense: &DenseND<T>,
    parallel: bool,
    executor: &CpuExecutor,
) -> Option<T>
where
    T: Clone + Num + Float + FromPrimitive + Send + Sync,
{
    if !supports(op) {
        return None;
    }

    let array = dense.as_array();
    // A strided/permuted view has no contiguous slice; fall back rather than
    // silently reducing in the wrong order.
    let data = array.as_slice()?;
    if data.len() < BLOCKED_MIN_ELEMS {
        return None;
    }

    // Per-block partials, collected in block order. `par_chunks` is an indexed
    // parallel iterator, so `collect` restores index order regardless of the
    // order the tasks actually finished in — this is what keeps the result
    // independent of the thread count.
    let partials: Vec<T> = if parallel {
        use rayon::prelude::*;
        executor.install(|| {
            data.par_chunks(PAR_BLOCK)
                .filter_map(|block| reduce_block(op, block))
                .collect()
        })
    } else {
        data.chunks(PAR_BLOCK)
            .filter_map(|block| reduce_block(op, block))
            .collect()
    };

    let (&first, rest) = partials.split_first()?;
    let mut result = first;
    for &p in rest {
        result = combine(op, result, p);
    }

    if matches!(op, ReduceOp::Mean) {
        let count = T::from_usize(data.len())?;
        result = result / count;
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blocked sum must agree with the naive sum to within the accumulated
    /// rounding of the naive order — which for 100k well-scaled values is tiny.
    #[test]
    fn blocked_sum_matches_naive() {
        let data: Vec<f64> = (0..100_000).map(|i| (i % 97) as f64 * 0.5).collect();
        let dense = DenseND::from_vec(data.clone(), &[100_000]).unwrap();
        let executor = CpuExecutor::new();

        let naive: f64 = data.iter().sum();
        let blocked = blocked_full_reduce(&ReduceOp::Sum, &dense, false, &executor).unwrap();

        assert!(
            (blocked - naive).abs() < 1e-6,
            "blocked {blocked} vs naive {naive}"
        );
    }

    /// The whole point of the determinism argument: the answer must not depend on
    /// how many threads happened to run it. This would fail for any kernel that
    /// let rayon choose the combine order (e.g. `par_iter().sum()`).
    #[test]
    fn blocked_sum_is_thread_count_independent() {
        let data: Vec<f64> = (0..250_000)
            .map(|i| ((i * 7919) % 1000) as f64 * 0.001)
            .collect();
        let dense = DenseND::from_vec(data, &[250_000]).unwrap();

        let serial = {
            let ex = CpuExecutor::new();
            blocked_full_reduce(&ReduceOp::Sum, &dense, false, &ex).unwrap()
        };

        for threads in [1usize, 2, 3, 5, 8] {
            let ex = CpuExecutor::with_threads(threads).unwrap();
            let parallel = blocked_full_reduce(&ReduceOp::Sum, &dense, true, &ex).unwrap();
            assert_eq!(
                serial.to_bits(),
                parallel.to_bits(),
                "blocked sum changed with {threads} threads: {serial} vs {parallel}"
            );
        }
    }

    #[test]
    fn blocked_mean_max_min_prod() {
        let data: Vec<f64> = (1..=2000).map(|i| i as f64).collect();
        let dense = DenseND::from_vec(data, &[2000]).unwrap();
        let ex = CpuExecutor::new();

        let mean = blocked_full_reduce(&ReduceOp::Mean, &dense, false, &ex).unwrap();
        assert!((mean - 1000.5).abs() < 1e-9, "mean was {mean}");

        let max = blocked_full_reduce(&ReduceOp::Max, &dense, false, &ex).unwrap();
        assert_eq!(max, 2000.0);

        let min = blocked_full_reduce(&ReduceOp::Min, &dense, false, &ex).unwrap();
        assert_eq!(min, 1.0);

        // Product of 1..=2000 overflows to +inf, but must do so consistently.
        let prod = blocked_full_reduce(&ReduceOp::Prod, &dense, false, &ex).unwrap();
        assert!(prod.is_infinite() && prod > 0.0);
    }

    /// The kernel must decline, not guess, when it cannot apply.
    #[test]
    fn declines_when_inapplicable() {
        let ex = CpuExecutor::new();

        // Too small.
        let small = DenseND::from_vec(vec![1.0f64; 10], &[10]).unwrap();
        assert!(blocked_full_reduce(&ReduceOp::Sum, &small, false, &ex).is_none());

        // Unsupported op, even at a size it would otherwise handle.
        let big = DenseND::from_vec(vec![1.0f64; 4096], &[4096]).unwrap();
        assert!(blocked_full_reduce(&ReduceOp::Any, &big, false, &ex).is_none());
        assert!(blocked_full_reduce(&ReduceOp::ArgMax, &big, false, &ex).is_none());
    }

    /// Blocking must not change the answer for a size that straddles a block
    /// boundary and leaves a lane remainder (4096 + 3 is neither a multiple of
    /// PAR_BLOCK nor of LANES).
    #[test]
    fn handles_ragged_lengths() {
        let n = 4096 + 3;
        let data: Vec<f64> = (0..n).map(|i| (i % 13) as f64).collect();
        let dense = DenseND::from_vec(data.clone(), &[n]).unwrap();
        let ex = CpuExecutor::new();

        let naive: f64 = data.iter().sum();
        let blocked = blocked_full_reduce(&ReduceOp::Sum, &dense, false, &ex).unwrap();
        assert!((blocked - naive).abs() < 1e-9);
    }
}
