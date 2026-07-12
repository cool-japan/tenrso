//! `DenseND<T>` vs raw `scirs2_core::ndarray_ext` array baseline comparison.
//!
//! `DenseND<T>` is a thin wrapper around `scirs2_core::ndarray_ext::Array<T, IxDyn>`
//! (which is itself a direct re-export of `ndarray::Array`). This benchmark answers
//! one question, honestly: **how much overhead does the wrapper add over the raw
//! array it wraps?**
//!
//! For every core operation we run a `dense_nd/<op>` benchmark next to an
//! `ndarray/<op>` benchmark that performs the equivalent work directly on the raw
//! array type, using criterion's `BenchmarkGroup` so the two numbers land next to
//! each other in the report (`dense_nd_64/reshape` vs `ndarray_64/reshape`, etc.).
//!
//! Two size tiers are covered:
//! - `64`  -> a `[64, 64, 64]` tensor (262,144 elements, ~2 MiB of `f64`) — small
//!   enough that allocator/bookkeeping overhead is visible relative to the copy.
//! - `256` -> a `[256, 256, 256]` tensor (16,777,216 elements, ~128 MiB of `f64`)
//!   — large enough that operations are memory-bandwidth dominated.
//!
//! ## A note on `permute` (and `reshape`)
//!
//! `DenseND::permute` is implemented as `self.data.clone().permuted_axes(axes)`
//! (see `src/dense/shape_ops.rs`). `Array::permuted_axes` itself only rewrites
//! strides/shape metadata — it never moves data. So the *entire* cost of
//! `DenseND::permute` is the leading `.clone()`, an `O(n)` copy, whereas the
//! equivalent raw-`ndarray` operation (`array.view().permuted_axes(axes)`) is a
//! zero-copy view. We benchmark both:
//! - `ndarray_*/permute_owned` clones first, exactly mirroring `DenseND::permute`,
//!   to isolate pure wrapper overhead (expected: ~0 delta vs `dense_nd`).
//! - `ndarray_*/permute_view` skips the clone entirely, to show what's actually
//!   achievable in raw `ndarray` for the same conceptual operation.
//!
//! The same pattern applies to `reshape`: `DenseND::reshape` goes through
//! `self.data.view().into_shape_with_order(shape)` (zero-copy *if* it succeeds)
//! followed by an unconditional `.to_owned()` (always copies). We expose the
//! same three-way split: `dense_nd/reshape`, `ndarray/reshape` (mirrors the
//! wrapper), and `ndarray/reshape_zero_copy` (a *consuming* reshape of an owned,
//! contiguous array, which never has to copy at all).
//!
//! Run with:
//! ```bash
//! cargo bench --bench ndarray_baseline
//! # or, for a fast smoke run:
//! cargo bench --bench ndarray_baseline -- --sample-size 10
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use scirs2_core::ndarray_ext::{Array, Array2, IxDyn};
use std::hint::black_box;
use std::time::Duration;
use tenrso_core::DenseND;

/// Raw array type matching exactly what `DenseND<f64>` wraps internally.
type RawArray = Array<f64, IxDyn>;

/// `[64, 64, 64]` — 262,144 elements (~2 MiB of `f64`); allocation-overhead
/// dominated.
const SMALL_SHAPE: [usize; 3] = [64, 64, 64];
/// `[256, 256, 256]` — 16,777,216 elements (~128 MiB of `f64`); memory
/// bandwidth dominated.
const LARGE_SHAPE: [usize; 3] = [256, 256, 256];

/// Mode used for all `unfold` benchmarks (an interior axis, so the
/// permutation involved is non-trivial).
const UNFOLD_MODE: usize = 1;
/// Axis permutation used for all `permute` benchmarks (a full 3-cycle).
const PERMUTE_AXES: [usize; 3] = [2, 0, 1];

/// A hand-fused, single-pass mode-`mode` unfold implemented directly on the
/// raw array type. This is what an `ndarray` user who cares about performance
/// would actually write: build the permutation as a zero-copy *view*, then
/// copy the (now logically reordered) elements into the output matrix in one
/// pass. This performs exactly one `O(n)` copy.
///
/// Compare with `DenseND::unfold`, which is implemented as
/// `self.permute(&perm)?.reshape(&[rows, cols])?`: `permute` cop`ies once
/// (`.clone()`), and then, because the permuted array is no longer
/// C-contiguous, `reshape`'s zero-copy path fails and it falls back to a
/// second `O(n)` copy (`self.data.iter().cloned().collect()`). So
/// `DenseND::unfold` performs *two* full copies where one suffices.
fn raw_unfold_single_pass(raw: &RawArray, mode: usize) -> Array2<f64> {
    let shape = raw.shape().to_vec();
    let rows = shape[mode];
    let cols: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_, &s)| s)
        .product();

    let mut perm: Vec<usize> = vec![mode];
    perm.extend((0..mode).chain((mode + 1)..shape.len()));

    // Zero-copy: only strides/shape metadata change.
    let permuted_view = raw.view().permuted_axes(IxDyn(&perm));
    // Exactly one O(n) copy: read the (strided) view once into a fresh buffer.
    let flat: Vec<f64> = permuted_view.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), flat).expect("shape matches by construction")
}

#[allow(clippy::too_many_lines)]
fn bench_for_size(
    c: &mut Criterion,
    shape: &[usize],
    label: &str,
    sample_size: usize,
    meas_secs: u64,
) {
    let total: usize = shape.iter().product();
    let flat_data: Vec<f64> = (0..total).map(|i| i as f64).collect();

    let dense_tensor = DenseND::<f64>::from_vec(flat_data.clone(), shape).expect("valid shape");
    let raw_array = RawArray::from_shape_vec(IxDyn(shape), flat_data.clone()).expect("valid shape");

    let dense_group_name = format!("dense_nd_{label}");
    let raw_group_name = format!("ndarray_{label}");

    // ----------------------------------------------------------------
    // Construction: zeros / from_vec
    // ----------------------------------------------------------------
    {
        let mut group = c.benchmark_group(dense_group_name);
        group.sample_size(sample_size);
        group.measurement_time(Duration::from_secs(meas_secs));
        group.warm_up_time(Duration::from_millis(500));

        group.bench_function("construction_zeros", |b| {
            b.iter(|| {
                let t = DenseND::<f64>::zeros(black_box(shape));
                black_box(t);
            });
        });
        group.bench_function("construction_from_vec", |b| {
            b.iter_batched(
                || flat_data.clone(),
                |data| {
                    let t = DenseND::from_vec(black_box(data), black_box(shape)).unwrap();
                    black_box(t);
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_function("reshape", |b| {
            let new_shape = [shape[0] * shape[1], shape[2]];
            b.iter(|| {
                let r = dense_tensor.reshape(black_box(&new_shape)).unwrap();
                black_box(r);
            });
        });
        group.bench_function("permute", |b| {
            b.iter(|| {
                let p = dense_tensor.permute(black_box(&PERMUTE_AXES)).unwrap();
                black_box(p);
            });
        });
        group.bench_function("unfold", |b| {
            b.iter(|| {
                let u = dense_tensor.unfold(black_box(UNFOLD_MODE)).unwrap();
                black_box(u);
            });
        });
        group.bench_function("add", |b| {
            b.iter(|| {
                let s = &dense_tensor + &dense_tensor;
                black_box(s);
            });
        });
        group.bench_function("hadamard", |b| {
            b.iter(|| {
                let h = dense_tensor.hadamard(black_box(&dense_tensor)).unwrap();
                black_box(h);
            });
        });
        group.bench_function("mul_scalar", |b| {
            b.iter(|| {
                let m = &dense_tensor * black_box(2.0_f64);
                black_box(m);
            });
        });
        group.bench_function("sum", |b| {
            b.iter(|| {
                let s = dense_tensor.sum();
                black_box(s);
            });
        });
        group.bench_function("mean", |b| {
            b.iter(|| {
                let m = dense_tensor.mean();
                black_box(m);
            });
        });
        group.bench_function("view", |b| {
            b.iter(|| {
                let v = dense_tensor.view();
                black_box(v);
            });
        });
        group.bench_function("index_get", |b| {
            let idx = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
            b.iter(|| {
                let v = dense_tensor[black_box(&idx[..])];
                black_box(v);
            });
        });

        group.finish();
    }

    // ----------------------------------------------------------------
    // Raw scirs2_core::ndarray_ext equivalents
    // ----------------------------------------------------------------
    {
        let mut group = c.benchmark_group(raw_group_name);
        group.sample_size(sample_size);
        group.measurement_time(Duration::from_secs(meas_secs));
        group.warm_up_time(Duration::from_millis(500));

        group.bench_function("construction_zeros", |b| {
            b.iter(|| {
                let t = RawArray::zeros(IxDyn(black_box(shape)));
                black_box(t);
            });
        });
        group.bench_function("construction_from_vec", |b| {
            b.iter_batched(
                || flat_data.clone(),
                |data| {
                    let t = RawArray::from_shape_vec(IxDyn(black_box(shape)), data).unwrap();
                    black_box(t);
                },
                BatchSize::LargeInput,
            );
        });
        // Mirrors DenseND::reshape's actual code path: view + into_shape_with_order
        // (zero-copy when contiguous) followed by an unconditional `.to_owned()`
        // (always copies). Isolates wrapper overhead — expect ~0 delta vs dense_nd.
        group.bench_function("reshape", |b| {
            let new_shape = [shape[0] * shape[1], shape[2]];
            b.iter(|| {
                let r = raw_array
                    .view()
                    .into_shape_with_order(IxDyn(&new_shape))
                    .unwrap()
                    .to_owned();
                black_box(r);
            });
        });
        // The zero-copy path DenseND::reshape could take if it consumed `self` by
        // value instead of borrowing: reshaping a contiguous *owned* array reuses
        // its buffer in place; no data is copied. `iter_batched` excludes the
        // `.clone()` setup from the timed routine so only the reshape itself is
        // measured.
        group.bench_function("reshape_zero_copy", |b| {
            let new_shape = [shape[0] * shape[1], shape[2]];
            b.iter_batched(
                || raw_array.clone(),
                |owned| {
                    let r = owned.into_shape_with_order(IxDyn(&new_shape)).unwrap();
                    black_box(r);
                },
                BatchSize::LargeInput,
            );
        });
        // Mirrors DenseND::permute's actual code path: clone, then permute the
        // clone's metadata. Isolates wrapper overhead — expect ~0 delta vs dense_nd.
        group.bench_function("permute_owned", |b| {
            b.iter(|| {
                let p = raw_array.clone().permuted_axes(IxDyn(&PERMUTE_AXES));
                black_box(p);
            });
        });
        // What raw ndarray actually offers for "give me a permuted tensor": a
        // zero-copy view. This is the real, material gap DenseND::permute leaves
        // on the table by always materializing an owned copy.
        group.bench_function("permute_view", |b| {
            b.iter(|| {
                let p = raw_array.view().permuted_axes(IxDyn(&PERMUTE_AXES));
                black_box(p);
            });
        });
        // Hand-fused single-pass unfold (one copy) vs DenseND::unfold (two copies).
        group.bench_function("unfold", |b| {
            b.iter(|| {
                let u = raw_unfold_single_pass(&raw_array, black_box(UNFOLD_MODE));
                black_box(u);
            });
        });
        // No clone-then-add: `&Array + &Array` allocates exactly one result
        // buffer. Compare with dense_nd/add, whose `Add` impl clones *both*
        // operands even when shapes already match (see src/dense/algebra.rs).
        group.bench_function("add", |b| {
            b.iter(|| {
                let s = &raw_array + &raw_array;
                black_box(s);
            });
        });
        group.bench_function("hadamard", |b| {
            b.iter(|| {
                let h = &raw_array * &raw_array;
                black_box(h);
            });
        });
        group.bench_function("mul_scalar", |b| {
            b.iter(|| {
                let m = &raw_array * black_box(2.0_f64);
                black_box(m);
            });
        });
        group.bench_function("sum", |b| {
            b.iter(|| {
                let s = raw_array.sum();
                black_box(s);
            });
        });
        group.bench_function("mean", |b| {
            b.iter(|| {
                let m = raw_array.mean();
                black_box(m);
            });
        });
        group.bench_function("view", |b| {
            b.iter(|| {
                let v = raw_array.view();
                black_box(v);
            });
        });
        group.bench_function("index_get", |b| {
            let idx = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
            b.iter(|| {
                let v = raw_array[IxDyn(black_box(&idx))];
                black_box(v);
            });
        });

        group.finish();
    }
}

fn bench_small(c: &mut Criterion) {
    bench_for_size(c, &SMALL_SHAPE, "64", 30, 2);
}

fn bench_large(c: &mut Criterion) {
    bench_for_size(c, &LARGE_SHAPE, "256", 10, 3);
}

criterion_group!(benches, bench_small, bench_large);
criterion_main!(benches);
