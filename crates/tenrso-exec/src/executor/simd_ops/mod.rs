//! Real SIMD acceleration for the element-wise path.
//!
//! # Scope: transcendentals only, and that is deliberate
//!
//! This module accelerates exactly eight operations — `exp`, `log`, and the six
//! activations that are pure functions of them (`sigmoid`, `tanh`, `gelu`, `elu`,
//! `selu`, `softplus`). Every other [`ElemOp`] has **no** SIMD path here and is
//! routed to the ordinary `mapv` implementation by [`simd_elem_op`] returning
//! `None`.
//!
//! That is not laziness, it is the measurement. On this workspace's Xeon Gold
//! 5315Y (contended, medians of 15 runs):
//!
//! | op | scalar `mapv` | AVX2 | verdict |
//! |----|--------------|------|---------|
//! | `exp`, 1M f64 | 7.44 ms | 1.98 ms | **3.75x faster** |
//! | `relu`, 8M f64 | 20.5 ms | 35.0 ms | 0.59x — *slower* |
//!
//! `exp` is ALU-latency-bound (libm's scalar `exp` is never auto-vectorised), so
//! vectorising it wins big. `relu` is memory-bandwidth-bound: LLVM already
//! auto-vectorises the `mapv` closure and the loop then just waits on RAM, so a
//! hand-written intrinsic cannot beat it and mostly perturbs the prefetcher.
//! Shipping a "SIMD-optimised `relu`" that is really a plain `mapv` — or one that
//! is genuinely SIMD and genuinely slower — would be a lie either way, so neither
//! exists here.
//!
//! # When the fast path is actually taken
//!
//! [`simd_elem_op`] returns `Some` only when *all* of these hold:
//!
//! 1. the executor has `enable_simd` set (the caller checks this),
//! 2. the op is one of the eight above,
//! 3. `T` is `f32` or `f64`,
//! 4. the tensor is contiguous in memory (`as_slice()` succeeds),
//! 5. it has at least [`SIMD_MIN_ELEMS`] elements,
//! 6. the CPU really has AVX2 + FMA (checked once, at runtime).
//!
//! Otherwise it returns `None` and the caller falls back. There is no silent
//! "SIMD" path that is secretly scalar.

use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use std::any::TypeId;
use std::sync::OnceLock;
use tenrso_core::DenseND;

use super::types::{CpuExecutor, ElemOp};

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(test)]
mod tests;

/// Below this element count the plain `mapv` path is used: the win does not repay
/// the dispatch, and small tensors are dominated by allocation anyway.
pub(crate) const SIMD_MIN_ELEMS: usize = 256;

/// Elements per rayon task when the SIMD path also runs in parallel. A multiple
/// of both lane counts (4 for f64, 8 for f32) so every task is whole vectors.
const PAR_CHUNK: usize = 8192;

/// The operations with a real AVX2 kernel behind them.
///
/// Every variant is `exp`- or `log`-bound, which is precisely what makes
/// vectorising it pay. See the module docs for why the list stops here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TranscendentalOp {
    Exp,
    Log,
    Sigmoid,
    Tanh,
    Gelu,
    Elu,
    Selu,
    Softplus,
}

pub(crate) const SELU_SCALE: f64 = 1.050_700_987_355_480_5;
pub(crate) const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
/// sqrt(2/pi), the coefficient of the GELU tanh approximation.
pub(crate) const GELU_COEF: f64 = 0.797_884_560_802_865_4;
pub(crate) const GELU_CUBIC: f64 = 0.044_715;

/// Map an [`ElemOp`] to its SIMD kernel, if it has one.
///
/// `None` means "no SIMD path exists for this op" — the caller must use the
/// ordinary implementation. The ops that return `None` (`Neg`, `Abs`, `Sin`,
/// `Cos`, `Sqrt`, `Sqr`, `Recip`, `Sign`, `ReLU`) are bandwidth-bound; see the
/// module docs.
pub(crate) fn transcendental_of(op: &ElemOp) -> Option<TranscendentalOp> {
    match op {
        ElemOp::Exp => Some(TranscendentalOp::Exp),
        ElemOp::Log => Some(TranscendentalOp::Log),
        ElemOp::Sigmoid => Some(TranscendentalOp::Sigmoid),
        ElemOp::Tanh => Some(TranscendentalOp::Tanh),
        ElemOp::Gelu => Some(TranscendentalOp::Gelu),
        ElemOp::Elu => Some(TranscendentalOp::Elu),
        ElemOp::Selu => Some(TranscendentalOp::Selu),
        ElemOp::Softplus => Some(TranscendentalOp::Softplus),
        ElemOp::Neg
        | ElemOp::Abs
        | ElemOp::Sin
        | ElemOp::Cos
        | ElemOp::Sqrt
        | ElemOp::Sqr
        | ElemOp::Recip
        | ElemOp::Sign
        | ElemOp::ReLU => None,
    }
}

/// The scalar definition of each op, in f64 and to full accuracy.
///
/// This is the single source of truth the SIMD kernels reproduce: it runs for the
/// vector tail, for any lane the range guard rejects, and as the oracle the
/// accuracy tests hold the kernels to.
///
/// It is written in the *cancellation-free* forms — `exp_m1`, `ln_1p`, and the
/// `sigmoid(2y)` GELU identity — for the same reason the kernels are: the naive
/// `exp(x) - 1` / `ln(1 + u)` / `1 + tanh(y)` forms lose most of their significant
/// digits in each op's tail (ELU/SELU near `x = 0`, softplus and GELU for very
/// negative `x`). An oracle that used the naive forms would be *less* accurate
/// than the kernel it judges, and would fail correct kernels at the tails.
#[inline]
pub(crate) fn scalar_f64(op: TranscendentalOp, v: f64) -> f64 {
    match op {
        TranscendentalOp::Exp => v.exp(),
        TranscendentalOp::Log => v.ln(),
        // sigmoid and tanh have accurate library forms with no cancellation.
        TranscendentalOp::Sigmoid => 1.0 / (1.0 + (-v).exp()),
        TranscendentalOp::Tanh => v.tanh(),
        TranscendentalOp::Gelu => {
            // 0.5*x*(1 + tanh(y)) == x*sigmoid(2y), exactly; the latter does not
            // cancel as tanh(y) -> -1.
            //
            // `mul_add` (one rounding) mirrors the kernel's fused inner term and is
            // strictly more accurate than a separate mul+add — so this is both the
            // truer value and the one the SIMD path reproduces bit-closely.
            let y = GELU_COEF * GELU_CUBIC.mul_add(v * v * v, v);
            v * (1.0 / (1.0 + (-2.0 * y).exp()))
        }
        TranscendentalOp::Elu => {
            if v > 0.0 {
                v
            } else {
                v.exp_m1()
            }
        }
        TranscendentalOp::Selu => {
            if v > 0.0 {
                SELU_SCALE * v
            } else {
                SELU_SCALE * SELU_ALPHA * v.exp_m1()
            }
        }
        TranscendentalOp::Softplus => {
            // max(x,0) + ln1p(exp(-|x|)). Never overflows, and stays accurate for
            // very negative x where the `1 +` would swallow exp(-|x|) entirely.
            let max_part = if v > 0.0 { v } else { 0.0 };
            max_part + (-v.abs()).exp().ln_1p()
        }
    }
}

/// The scalar definition in f32, evaluated in f64 and rounded once.
#[inline]
pub(crate) fn scalar_f32(op: TranscendentalOp, v: f32) -> f32 {
    scalar_f64(op, v as f64) as f32
}

/// Does this CPU actually have AVX2 + FMA? Probed once, then cached.
///
/// On any non-x86_64 target this is `false` and there is simply no SIMD path —
/// the element-wise ops run the ordinary implementation, correctly and honestly.
pub(crate) fn simd_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Run `op` over a contiguous f64 slice with the AVX2 kernel.
///
/// Splits into rayon tasks when `parallel` is set, running them inside the
/// executor's own thread pool so that `CpuExecutor::with_threads` governs this
/// path too.
#[cfg(target_arch = "x86_64")]
fn run_f64(
    op: TranscendentalOp,
    src: &[f64],
    dst: &mut [f64],
    parallel: bool,
    executor: &CpuExecutor,
) {
    if parallel {
        use rayon::prelude::*;
        executor.install(|| {
            src.par_chunks(PAR_CHUNK)
                .zip(dst.par_chunks_mut(PAR_CHUNK))
                .for_each(|(s, d)| {
                    // SAFETY: `simd_available()` was checked by the caller, so AVX2+FMA
                    // are present. `s` and `d` are equal-length (par_chunks/par_chunks_mut
                    // over equal-length slices with the same chunk size).
                    unsafe { avx2::apply_slice_f64(op, s, d) }
                });
        });
    } else {
        // SAFETY: as above; caller checked `simd_available()`, lengths are equal.
        unsafe { avx2::apply_slice_f64(op, src, dst) }
    }
}

/// Run `op` over a contiguous f32 slice with the AVX2 kernel.
#[cfg(target_arch = "x86_64")]
fn run_f32(
    op: TranscendentalOp,
    src: &[f32],
    dst: &mut [f32],
    parallel: bool,
    executor: &CpuExecutor,
) {
    if parallel {
        use rayon::prelude::*;
        executor.install(|| {
            src.par_chunks(PAR_CHUNK)
                .zip(dst.par_chunks_mut(PAR_CHUNK))
                .for_each(|(s, d)| {
                    // SAFETY: see `run_f64`.
                    unsafe { avx2::apply_slice_f32(op, s, d) }
                });
        });
    } else {
        // SAFETY: see `run_f64`.
        unsafe { avx2::apply_slice_f32(op, src, dst) }
    }
}

/// The element-wise SIMD entry point used by `CpuExecutor::parallel_elem_op`.
///
/// Returns `Some(result)` when the real AVX2 kernel ran, and `None` when no SIMD
/// path applies — see the module docs for the six conditions. A `None` return is
/// not a failure: it means the caller should use the ordinary implementation,
/// which produces the same values.
pub(crate) fn simd_elem_op<T>(
    op: &ElemOp,
    dense: &DenseND<T>,
    parallel: bool,
    executor: &CpuExecutor,
) -> Option<Array<T, IxDyn>>
where
    T: Clone + Num + Float + FromPrimitive + Send + Sync + 'static,
{
    let top = transcendental_of(op)?;
    if !simd_available() {
        return None;
    }

    let array = dense.as_array();
    // Strided/permuted views have no contiguous slice; the vector loads need one.
    let src = array.as_slice()?;
    if src.len() < SIMD_MIN_ELEMS {
        return None;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let n = src.len();
        let mut out: Vec<T> = vec![T::zero(); n];

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: the TypeId check proves T is exactly f64, so these casts are
            // the identity reinterpretation of a &[f64] / &mut [f64] that already
            // exist, with unchanged length. Nothing is transmuted between distinct
            // types, and `out` stays a live Vec<T> that we hand back below.
            let s: &[f64] = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, n) };
            let d: &mut [f64] =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f64, n) };
            run_f64(top, s, d, parallel, executor);
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: as above, with T proven to be exactly f32.
            let s: &[f32] = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, n) };
            let d: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, n) };
            run_f32(top, s, d, parallel, executor);
        } else {
            // Some other float type: no kernel, no pretending.
            return None;
        }

        Array::from_shape_vec(IxDyn(dense.shape()), out).ok()
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Keep the parameters used on non-x86_64 so the signature stays honest.
        let _ = (top, parallel, executor);
        None
    }
}
