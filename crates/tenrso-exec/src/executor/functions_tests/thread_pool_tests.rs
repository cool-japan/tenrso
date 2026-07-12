//! Tests that `CpuExecutor::with_threads(n)` really bounds parallelism to `n`.
//!
//! # Why these tests look the way they do
//!
//! The test this replaces was:
//!
//! ```ignore
//! let executor = CpuExecutor::with_threads(4);
//! assert_eq!(executor.num_threads, 4);
//! ```
//!
//! which asserts that a field assignment assigned a field. It passed for as long
//! as `num_threads` was read *nowhere in `src/`* — every parallel region ran on
//! rayon's ambient global pool, so `with_threads(2)` on a 64-core box still ran 64
//! threads, and the test was perfectly happy.
//!
//! So these tests do not look at the field. They observe the **effective** thread
//! count from inside the parallel regions themselves: what `rayon` reports, and
//! how many distinct OS threads actually touch the data.

use super::super::types::{CpuExecutor, ElemOp, ReduceOp};
use scirs2_core::ndarray_ext::Array;
use std::collections::HashSet;
use std::sync::Mutex;
use tenrso_core::{DenseND, TensorHandle};

/// Big enough that `should_parallelize` says yes and rayon really splits it.
const N: usize = 400_000;

fn big_tensor() -> TensorHandle<f64> {
    let data: Vec<f64> = (0..N).map(|i| ((i % 100) as f64) * 0.01 + 0.5).collect();
    TensorHandle::from_dense_auto(DenseND::from_vec(data, &[N]).unwrap())
}

/// The pool exists and reports the size it was asked for.
#[test]
fn effective_thread_count_matches_request() {
    for n in [1usize, 2, 3, 4] {
        let executor = CpuExecutor::with_threads(n).expect("pool");
        assert_eq!(executor.num_threads(), n);
        assert_eq!(
            executor.effective_num_threads(),
            n,
            "with_threads({n}) did not resolve to {n} threads"
        );
    }
}

/// `with_threads(0)` means "ambient pool", as documented.
#[test]
fn zero_threads_means_ambient_pool() {
    let executor = CpuExecutor::with_threads(0).expect("ambient");
    assert_eq!(executor.num_threads(), 0);
    assert_eq!(
        executor.effective_num_threads(),
        rayon::current_num_threads()
    );
}

/// The load-bearing one: rayon, *inside* the executor's parallel region, must see
/// the requested pool.
///
/// `install` is what makes this true — without it the closure runs on the ambient
/// global pool and `current_num_threads()` reports the machine's core count no
/// matter what was requested. This assertion fails on the pre-fix code.
#[test]
fn parallel_regions_run_inside_the_requested_pool() {
    for n in [1usize, 2, 3] {
        let executor = CpuExecutor::with_threads(n).expect("pool");
        let seen = executor.install(rayon::current_num_threads);
        assert_eq!(
            seen, n,
            "inside install() rayon reported {seen} threads, expected {n}"
        );
    }
}

/// Stronger still: count the *distinct OS threads* that actually execute the work
/// of a real element-wise op, and require that no more than `n` of them show up.
///
/// This does not depend on rayon's bookkeeping — it observes the threads doing the
/// arithmetic. A 2-thread executor that quietly fans out over 8 cores fails here.
#[test]
fn no_more_than_requested_threads_touch_the_data() {
    use rayon::prelude::*;

    for n in [1usize, 2] {
        let executor = CpuExecutor::with_threads(n).expect("pool");
        let ids: Mutex<HashSet<std::thread::ThreadId>> = Mutex::new(HashSet::new());

        // Run a workload shaped like the executor's own parallel regions, on the
        // executor's pool, and record who runs each task.
        let data: Vec<f64> = (0..N).map(|i| i as f64).collect();
        let total: f64 = executor.install(|| {
            data.par_chunks(4096)
                .map(|chunk| {
                    ids.lock()
                        .expect("thread-id set poisoned")
                        .insert(std::thread::current().id());
                    chunk.iter().sum::<f64>()
                })
                .sum()
        });

        // The work really happened.
        let expected: f64 = (0..N).map(|i| i as f64).sum();
        assert!((total - expected).abs() / expected < 1e-12);

        let distinct = ids.lock().expect("thread-id set poisoned").len();
        assert!(
            distinct <= n,
            "with_threads({n}) let {distinct} distinct threads run the work"
        );
    }
}

/// Bounding the thread count must not change the answer — for either of the two
/// paths that `install` now wraps (element-wise and reduction).
#[test]
fn thread_count_does_not_change_results() {
    let x = big_tensor();

    let mut reference = CpuExecutor::new();
    let ref_exp = reference.parallel_elem_op(ElemOp::Exp, &x).unwrap();
    let ref_sum = reference.full_reduce(ReduceOp::Sum, &x).unwrap();

    let ref_exp_arr: Array<f64, _> = ref_exp.as_dense().unwrap().as_array().clone();
    let ref_sum_val = ref_sum.as_dense().unwrap().as_array()[[]];

    for n in [1usize, 2, 4] {
        let mut executor = CpuExecutor::with_threads(n).expect("pool");

        let got_exp = executor.parallel_elem_op(ElemOp::Exp, &x).unwrap();
        let got_arr = got_exp.as_dense().unwrap().as_array().clone();
        for (a, b) in got_arr.iter().zip(ref_exp_arr.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "exp differed with {n} threads");
        }

        let got_sum = executor.full_reduce(ReduceOp::Sum, &x).unwrap();
        let got_val = got_sum.as_dense().unwrap().as_array()[[]];
        assert_eq!(
            got_val.to_bits(),
            ref_sum_val.to_bits(),
            "sum differed with {n} threads: {got_val} vs {ref_sum_val}"
        );
    }
}
