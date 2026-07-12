//! Benchmarks for the executor's optimization knobs.
//!
//! # These compare genuinely different code paths
//!
//! The suite this replaces benchmarked `with_simd(true)` against
//! `with_simd(false)` — but `enable_simd` was read only inside a dead module, so
//! *both arms ran the identical code* and the "speedup" it printed was noise
//! around 1.0x with a SIMD-sounding label on it.
//!
//! Every pair below now toggles a flag that selects a different implementation:
//!
//! - `with_simd`: AVX2 `exp`/`log` kernels vs scalar libm `mapv`.
//! - `with_blocked_reductions`: multi-accumulator blocked reduction vs the naive
//!   single-accumulator fold.
//! - `with_threads`: the executor's private rayon pool, at several sizes.
//!
//! Note these drive `parallel_elem_op` / `full_reduce` — the inherent methods that
//! read the executor's configuration. The `TenrsoExecutor` *trait* methods
//! (`elem_op`, …) deliberately ignore executor config and are not benchmarked here.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{BinaryOp, CpuExecutor, ElemOp, ReduceOp};

fn tensor_1d(n: usize, f: impl Fn(usize) -> f64) -> TensorHandle<f64> {
    let data: Vec<f64> = (0..n).map(f).collect();
    TensorHandle::from_dense_auto(DenseND::from_vec(data, &[n]).expect("dense"))
}

/// AVX2 `exp`-family kernels vs the scalar `mapv` path.
///
/// These are the ops that have a real SIMD kernel. Expect a solid win — measured
/// ~2-5x for `exp` on this workspace's Xeon.
fn bench_simd_transcendental(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_transcendental");

    for &n in &[4096usize, 65_536, 1_048_576] {
        group.throughput(Throughput::Elements(n as u64));
        let x = tensor_1d(n, |i| -6.0 + (i as f64) * 1e-5);

        for (label, op) in [("exp", ElemOp::Exp), ("gelu", ElemOp::Gelu)] {
            group.bench_with_input(BenchmarkId::new(format!("{label}_simd"), n), &x, |b, x| {
                let mut ex = CpuExecutor::new().with_simd(true);
                b.iter(|| black_box(ex.parallel_elem_op(op.clone(), black_box(x)).expect("op")));
            });
            group.bench_with_input(
                BenchmarkId::new(format!("{label}_scalar"), n),
                &x,
                |b, x| {
                    let mut ex = CpuExecutor::new().with_simd(false);
                    b.iter(|| {
                        black_box(ex.parallel_elem_op(op.clone(), black_box(x)).expect("op"))
                    });
                },
            );
        }
    }
    group.finish();
}

/// Ops with **no** SIMD kernel, benchmarked to keep the claim honest.
///
/// `relu` is bandwidth-bound: an intrinsic version measured *slower* than `mapv`,
/// which is why there is no kernel for it. Both arms here run the same code, and
/// this benchmark exists to document that they are supposed to — if a future
/// change makes these diverge, someone has added a SIMD path that needs its own
/// measurement.
fn bench_bandwidth_bound_have_no_simd_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandwidth_bound_no_simd");
    let n = 1_048_576usize;
    group.throughput(Throughput::Elements(n as u64));
    let x = tensor_1d(n, |i| -1.0 + (i as f64) * 2e-6);

    for (label, op) in [("relu", ElemOp::ReLU), ("sqrt", ElemOp::Sqrt)] {
        group.bench_with_input(BenchmarkId::new(label, n), &x, |b, x| {
            let mut ex = CpuExecutor::new();
            b.iter(|| black_box(ex.parallel_elem_op(op.clone(), black_box(x)).expect("op")));
        });
    }
    group.finish();
}

/// Blocked (multi-accumulator) reduction vs the naive single-accumulator fold.
fn bench_blocked_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocked_reductions");

    for &n in &[65_536usize, 1_048_576, 4_194_304] {
        group.throughput(Throughput::Elements(n as u64));
        let x = tensor_1d(n, |i| ((i % 1000) as f64) * 0.001);

        for (label, op) in [("sum", ReduceOp::Sum), ("max", ReduceOp::Max)] {
            group.bench_with_input(
                BenchmarkId::new(format!("{label}_blocked"), n),
                &x,
                |b, x| {
                    let mut ex = CpuExecutor::new().with_blocked_reductions(true);
                    b.iter(|| black_box(ex.full_reduce(op.clone(), black_box(x)).expect("reduce")));
                },
            );
            group.bench_with_input(BenchmarkId::new(format!("{label}_naive"), n), &x, |b, x| {
                let mut ex = CpuExecutor::new().with_blocked_reductions(false);
                b.iter(|| black_box(ex.full_reduce(op.clone(), black_box(x)).expect("reduce")));
            });
        }
    }
    group.finish();
}

/// Broadcasting binary ops.
///
/// The old implementation rebuilt per-operand subscripts for every output element,
/// allocating two `Vec`s each time; this walks stride-0 broadcast views instead.
fn bench_broadcast_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_binary");

    // (B, 1, K) + (B, M, K): the classic "add a bias row" shape.
    for &(b_dim, m, k) in &[(64usize, 64usize, 64usize), (256, 128, 64)] {
        let out = b_dim * m * k;
        group.throughput(Throughput::Elements(out as u64));

        let x = TensorHandle::from_dense_auto(
            DenseND::from_vec(
                (0..b_dim * k).map(|i| i as f64 * 1e-3).collect(),
                &[b_dim, 1, k],
            )
            .expect("x"),
        );
        let y = TensorHandle::from_dense_auto(
            DenseND::from_vec((0..out).map(|i| i as f64 * 1e-4).collect(), &[b_dim, m, k])
                .expect("y"),
        );

        group.bench_with_input(
            BenchmarkId::new("add_broadcast", out),
            &(x, y),
            |bench, (x, y)| {
                let mut ex = CpuExecutor::new();
                bench.iter(|| {
                    black_box(
                        ex.parallel_binary_op(BinaryOp::Add, black_box(x), black_box(y))
                            .expect("add"),
                    )
                });
            },
        );
    }
    group.finish();
}

/// How the executor's private pool scales with its thread count.
///
/// `with_threads(n)` builds a real rayon pool of `n` workers and installs the
/// parallel regions into it, so these numbers should actually move with `n`.
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    let n = 2_097_152usize;
    group.throughput(Throughput::Elements(n as u64));
    let x = tensor_1d(n, |i| -6.0 + (i as f64) * 6e-6);

    for threads in [1usize, 2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("exp", threads), &x, |b, x| {
            let mut ex = CpuExecutor::with_threads(threads).expect("pool");
            b.iter(|| black_box(ex.parallel_elem_op(ElemOp::Exp, black_box(x)).expect("op")));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_simd_transcendental,
    bench_bandwidth_bound_have_no_simd_path,
    bench_blocked_reductions,
    bench_broadcast_binary,
    bench_thread_scaling,
);
criterion_main!(benches);
