//! AVX2 + FMA kernels for transcendental element-wise operations.
//!
//! # Why only transcendentals
//!
//! Hand-written SIMD only pays for itself when an operation is *compute*-bound.
//! Bandwidth-bound kernels (`neg`, `abs`, `relu`, `sqrt`, `x*x`, `+`, `*`, …)
//! already run at memory speed: LLVM auto-vectorises the `mapv` closure and the
//! loop then waits on RAM, so an intrinsic version cannot go faster.  Measured on
//! this workspace's Xeon Gold 5315Y (4M f64, median of 15, contended box):
//! `relu` scalar 20.5 ms vs AVX2 35.0 ms — the intrinsic version is *slower*.
//!
//! `exp`/`log` are the opposite: libm's scalar `exp` is a ~20-40 cycle call per
//! element and is never auto-vectorised, so the loop is latency-bound on the ALU,
//! not on RAM.  Same machine, same protocol: `exp` scalar 7.44 ms vs AVX2 1.98 ms
//! (**3.75x**), holding 1.65x-5.2x across 100K/1M/8M and repeated runs.
//!
//! So this module implements *only* `exp` and `log` (plus the six activations
//! that are pure functions of them).  Everything else deliberately has no SIMD
//! path and goes through the plain `mapv` route — see [`super::simd_elem_op`].
//!
//! # Accuracy
//!
//! Both cores are correctly range-reduced (Cody-Waite) and use a polynomial long
//! enough that its first omitted term is below the unit roundoff, so the fast path
//! is accurate to ~1 ULP.  Measured max relative error vs libm over the tested
//! sweep: **2.2e-16 for f64** (= 1 ULP) and **1.2e-7 for f32**.
//!
//! # Edge cases are exactly libm's, by construction
//!
//! Every kernel guards its lanes: a SIMD block is only taken when *all* its lanes
//! are "tame" (finite, in a range where the vector math provably cannot overflow
//! or underflow).  Any block containing a NaN, an infinity, a zero/negative (for
//! `log`), or an extreme magnitude is computed with **scalar libm** instead.  The
//! vector path therefore never has to reproduce libm's special-case behaviour —
//! it never sees those inputs.

#![allow(clippy::excessive_precision)]

use super::{
    scalar_f32, scalar_f64, TranscendentalOp, GELU_COEF, GELU_CUBIC, SELU_ALPHA, SELU_SCALE,
};
use std::arch::x86_64::*;

// ─────────────────────────────── f64 constants ───────────────────────────────

/// `|x| <= EXP_LIMIT_F64` keeps `2^n` inside the normal double range.
const EXP_LIMIT_F64: f64 = 700.0;
/// `tanh` evaluates `exp(2x)`, so its input limit is half of `exp`'s.
const TANH_LIMIT_F64: f64 = 350.0;
/// `gelu` feeds `c*(x + 0.044715 x^3)` to `tanh`; `|x| <= 20` keeps that under
/// `TANH_LIMIT_F64` (`0.798 * (20 + 0.044715*8000) = 301 < 350`).
const GELU_LIMIT_F64: f64 = 20.0;

/// Cody-Waite split of ln(2): `ln2 = LN2_HI + LN2_LO` to > 100 bits.
const LN2_HI_F64: f64 = 6.93147180369123816490e-01;
const LN2_LO_F64: f64 = 1.90821492927058770002e-10;

/// Taylor coefficients `1/k!` for k = 13 down to 1.
///
/// After range reduction `|r| <= ln2/2 = 0.3466`, so the first omitted term is
/// `r^14/14! = 4.1e-18` — below the f64 unit roundoff (1.1e-16).
const EXP_COEF_F64: [f64; 13] = [
    1.60590438368216145993e-10, // 1/13!
    2.08767569878680989792e-09, // 1/12!
    2.50521083854417187751e-08, // 1/11!
    2.75573192239858906526e-07, // 1/10!
    2.75573192239858906526e-06, // 1/9!
    2.48015873015873015873e-05, // 1/8!
    1.98412698412698412698e-04, // 1/7!
    1.38888888888888888889e-03, // 1/6!
    8.33333333333333333333e-03, // 1/5!
    4.16666666666666666667e-02, // 1/4!
    1.66666666666666666667e-01, // 1/3!
    5.00000000000000000000e-01, // 1/2!
    1.00000000000000000000e+00, // 1/1!
];

/// atanh series coefficients `1/(2k+1)` for k = 11 down to 0.
///
/// With the mantissa normalised to `[sqrt(2)/2, sqrt(2))`, `|s| <= 0.1716` and
/// `z = s^2 <= 0.0295`, so the first omitted term `z^12/25 = 1.7e-20` is far
/// below the unit roundoff.
const LOG_COEF_F64: [f64; 12] = [
    4.34782608695652173913e-02, // 1/23
    4.76190476190476190476e-02, // 1/21
    5.26315789473684210526e-02, // 1/19
    5.88235294117647058824e-02, // 1/17
    6.66666666666666666667e-02, // 1/15
    7.69230769230769230769e-02, // 1/13
    9.09090909090909090909e-02, // 1/11
    1.11111111111111111111e-01, // 1/9
    1.42857142857142857143e-01, // 1/7
    2.00000000000000000000e-01, // 1/5
    3.33333333333333333333e-01, // 1/3
    1.00000000000000000000e+00, // 1/1
];

/// `1.5 * 2^52`. Adding a small signed integer to this double's *bit pattern*
/// lands the integer in the mantissa; subtracting the value back out recovers it
/// as an f64. Valid for `|e| < 2^51`, which every IEEE-754 exponent satisfies.
const I64_TO_F64_MAGIC: f64 = 6755399441055744.0;
const I64_TO_F64_MAGIC_BITS: i64 = 0x4338_0000_0000_0000u64 as i64;

// ─────────────────────────────── f32 constants ───────────────────────────────

/// `|x| <= EXP_LIMIT_F32` keeps `2^n` inside the normal float range.
const EXP_LIMIT_F32: f32 = 88.0;
const TANH_LIMIT_F32: f32 = 44.0;
const GELU_LIMIT_F32: f32 = 10.0;

const LN2_HI_F32: f32 = 6.9314575195e-01;
const LN2_LO_F32: f32 = 1.4286067653e-06;

/// Taylor `1/k!` for k = 6..1. First omitted term `r^7/7! = 3.4e-9`, below the
/// f32 unit roundoff (6.0e-8).
const EXP_COEF_F32: [f32; 6] = [
    1.3888888888e-03, // 1/6!
    8.3333333333e-03, // 1/5!
    4.1666666667e-02, // 1/4!
    1.6666666667e-01, // 1/3!
    5.0000000000e-01, // 1/2!
    1.0000000000e+00, // 1/1!
];

/// atanh `1/(2k+1)` for k = 5..0. First omitted term `z^6/13 = 5.0e-11`.
const LOG_COEF_F32: [f32; 6] = [
    9.0909090909e-02, // 1/11
    1.1111111111e-01, // 1/9
    1.4285714286e-01, // 1/7
    2.0000000000e-01, // 1/5
    3.3333333333e-01, // 1/3
    1.0000000000e+00, // 1/1
];

// ═══════════════════════════════ f64 vector cores ════════════════════════════

/// `exp(x)` for four lanes that the caller has already proven tame
/// (finite, `|x| <= EXP_LIMIT_F64`).
///
/// `exp(x) = 2^n * exp(r)` with `n = round(x * log2(e))` and
/// `r = x - n*ln2` evaluated in two Cody-Waite steps so that `r` keeps full
/// precision even when `n*ln2` cancels most of `x`.
///
/// # Safety
/// Requires AVX2 + FMA. Lanes must satisfy the tameness guard above; otherwise
/// the `2^n` exponent construction can overflow into a NaN/garbage bit pattern.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp_pd(x: __m256d) -> __m256d {
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        _mm256_set1_pd(std::f64::consts::LOG2_E),
    ));

    // r = x - n*LN2_HI - n*LN2_LO
    let mut r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_HI_F64), x);
    r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_LO_F64), r);

    // Horner over 1/k!, then the implicit leading 1.
    let mut p = _mm256_set1_pd(EXP_COEF_F64[0]);
    for &c in EXP_COEF_F64.iter().skip(1) {
        p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(c));
    }
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));

    // 2^n by writing n straight into the exponent field.
    let ni = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(n));
    let pow2n = _mm256_castsi256_pd(_mm256_slli_epi64::<52>(_mm256_add_epi64(
        ni,
        _mm256_set1_epi64x(1023),
    )));

    _mm256_mul_pd(p, pow2n)
}

/// `exp(x) - 1`, accurate all the way down to `x = 0`.
///
/// Computing `exp(x) - 1` directly is a classic numerical trap: for small `x`,
/// `exp(x)` rounds to something very near 1 and the subtraction cancels away most
/// of the significant digits (at `x = 0.003` it costs ~2.5 decimal digits).
///
/// The fix falls straight out of the way [`exp_pd`] is already written. Its Horner
/// loop produces `Q(r)` with `exp(r) = 1 + r*Q(r)`, so `expm1(r) = r*Q(r)` — a
/// product, with nothing to cancel. Undoing the range reduction keeps that
/// property:
///
/// ```text
/// expm1(x) = 2^n * exp(r) - 1
///          = 2^n * (1 + r*Q(r)) - 1
///          = (2^n - 1) + 2^n * r*Q(r)
/// ```
///
/// At `n = 0` the first term is exactly zero and the result is just `r*Q(r)`; for
/// large `n` the `2^n - 1` term dominates and is exact. No branch, no cancellation.
///
/// # Safety
/// Requires AVX2 + FMA and the same tameness guard as [`exp_pd`].
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn expm1_pd(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        _mm256_set1_pd(std::f64::consts::LOG2_E),
    ));

    let mut r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_HI_F64), x);
    r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_LO_F64), r);

    // q = Q(r): the same Horner as exp_pd, stopped one step early.
    let mut q = _mm256_set1_pd(EXP_COEF_F64[0]);
    for &c in EXP_COEF_F64.iter().skip(1) {
        q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(c));
    }
    let rq = _mm256_mul_pd(r, q); // expm1(r), exactly

    let ni = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(n));
    let pow2n = _mm256_castsi256_pd(_mm256_slli_epi64::<52>(_mm256_add_epi64(
        ni,
        _mm256_set1_epi64x(1023),
    )));

    // (2^n - 1) + 2^n * expm1(r)
    let res = _mm256_fmadd_pd(pow2n, rq, _mm256_sub_pd(pow2n, one));

    // expm1 is odd around 0, so `expm1(-0.0)` must stay `-0.0`. The arithmetic
    // above yields `+0.0` there (`1*(-0.0) + 0.0` rounds to `+0.0`), which would
    // flip the sign of the odd functions built on it (tanh, elu, selu). Restore
    // the input's sign for the exact-zero lanes; every other lane is untouched.
    let is_zero = _mm256_cmp_pd::<_CMP_EQ_OQ>(x, _mm256_setzero_pd());
    _mm256_blendv_pd(res, x, is_zero)
}

/// `ln(1 + u)` for `u` in `(0, 1]`, accurate as `u -> 0`.
///
/// Naively, `1 + u` rounds away the low bits of a small `u` and `ln` of the result
/// underflows to zero. The standard correction (Goldberg) evaluates `ln` at the
/// *rounded* `m = 1 + u` and rescales by `u / (m - 1)` — the same rounding error
/// appears in numerator and denominator and cancels out. Where `m` rounds to
/// exactly 1, `ln(1+u) = u` to full precision, and that lane is selected directly.
///
/// # Safety
/// Requires AVX2 + FMA. `u` must be non-negative and finite, so `m` lands in
/// `[1, 2]` — inside [`log_pd`]'s positive-normal domain.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn log1p_pd(u: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let m = _mm256_add_pd(one, u);
    let d = _mm256_sub_pd(m, one); // exactly representable

    let is_zero = _mm256_cmp_pd::<_CMP_EQ_OQ>(d, _mm256_setzero_pd());
    let safe_d = _mm256_blendv_pd(d, one, is_zero); // never divide by 0

    let scaled = _mm256_mul_pd(log_pd(m), _mm256_div_pd(u, safe_d));
    _mm256_blendv_pd(scaled, u, is_zero)
}

/// `ln(x)` for four lanes the caller has proven to be positive *normal* doubles.
///
/// `x = m * 2^e` with the mantissa renormalised to `[sqrt(2)/2, sqrt(2))`, then
/// `ln(m) = 2*atanh(s)` with `s = (m-1)/(m+1)`, and `ln(x) = ln(m) + e*ln2`
/// (the `e*ln2` product also uses the Cody-Waite split).
///
/// # Safety
/// Requires AVX2 + FMA. Lanes must be finite, positive and normal — subnormals,
/// zero, negatives, inf and NaN all break the exponent/mantissa extraction.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn log_pd(x: __m256d) -> __m256d {
    let xi = _mm256_castpd_si256(x);

    // e = unbiased exponent; m = mantissa forced into [1, 2).
    let e_raw = _mm256_sub_epi64(
        _mm256_srli_epi64::<52>(xi),
        _mm256_set1_epi64x(1023), // exponent field is 11 bits; sign bit is 0 (x > 0)
    );
    let m_bits = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi64x(0x000F_FFFF_FFFF_FFFFu64 as i64)),
        _mm256_set1_epi64x(0x3FF0_0000_0000_0000u64 as i64),
    );
    let m = _mm256_castsi256_pd(m_bits);

    // Renormalise m into [sqrt(2)/2, sqrt(2)) so |s| stays small.
    let gt = _mm256_cmp_pd::<_CMP_GT_OQ>(m, _mm256_set1_pd(std::f64::consts::SQRT_2));
    let m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), gt);
    let e = _mm256_add_epi64(
        e_raw,
        _mm256_and_si256(_mm256_castpd_si256(gt), _mm256_set1_epi64x(1)),
    );

    // i64 -> f64 via the mantissa-magic trick (exponents are far inside range).
    let ed = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_add_epi64(
            e,
            _mm256_set1_epi64x(I64_TO_F64_MAGIC_BITS),
        )),
        _mm256_set1_pd(I64_TO_F64_MAGIC),
    );

    let one = _mm256_set1_pd(1.0);
    let s = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
    let z = _mm256_mul_pd(s, s);

    let mut p = _mm256_set1_pd(LOG_COEF_F64[0]);
    for &c in LOG_COEF_F64.iter().skip(1) {
        p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(c));
    }
    // ln(m) = 2*s*p
    let ln_m = _mm256_mul_pd(_mm256_add_pd(s, s), p);

    // ln(x) = ln(m) + e*LN2_HI + e*LN2_LO
    let acc = _mm256_fmadd_pd(ed, _mm256_set1_pd(LN2_HI_F64), ln_m);
    _mm256_fmadd_pd(ed, _mm256_set1_pd(LN2_LO_F64), acc)
}

/// True iff every lane is finite with `|x| <= limit` (NaN and inf both fail).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tame_pd(x: __m256d, limit: f64) -> bool {
    let absx = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    // _CMP_LE_OQ is false for NaN, and |inf| > limit, so both are rejected.
    _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LE_OQ>(absx, _mm256_set1_pd(limit))) == 0b1111
}

/// True iff every lane is a positive normal double (the domain `log_pd` needs).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn log_domain_pd(x: __m256d) -> bool {
    let lo = _mm256_cmp_pd::<_CMP_GE_OQ>(x, _mm256_set1_pd(f64::MIN_POSITIVE));
    let hi = _mm256_cmp_pd::<_CMP_LE_OQ>(x, _mm256_set1_pd(f64::MAX));
    _mm256_movemask_pd(_mm256_and_pd(lo, hi)) == 0b1111
}

// ═══════════════════════════════ f32 vector cores ════════════════════════════

/// `exp(x)` for eight tame f32 lanes. Same algorithm as [`exp_pd`].
///
/// # Safety
/// Requires AVX2 + FMA; lanes must be finite with `|x| <= EXP_LIMIT_F32`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp_ps(x: __m256) -> __m256 {
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));

    let mut r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI_F32), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO_F32), r);

    let mut p = _mm256_set1_ps(EXP_COEF_F32[0]);
    for &c in EXP_COEF_F32.iter().skip(1) {
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
    }
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));

    let ni = _mm256_cvtps_epi32(n);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        ni,
        _mm256_set1_epi32(127),
    )));

    _mm256_mul_ps(p, pow2n)
}

/// `exp(x) - 1` for eight f32 lanes. Same identity as [`expm1_pd`].
///
/// # Safety
/// Requires AVX2 + FMA and the [`exp_ps`] tameness guard.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn expm1_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));

    let mut r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI_F32), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO_F32), r);

    let mut q = _mm256_set1_ps(EXP_COEF_F32[0]);
    for &c in EXP_COEF_F32.iter().skip(1) {
        q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(c));
    }
    let rq = _mm256_mul_ps(r, q);

    let ni = _mm256_cvtps_epi32(n);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        ni,
        _mm256_set1_epi32(127),
    )));

    let res = _mm256_fmadd_ps(pow2n, rq, _mm256_sub_ps(pow2n, one));

    // Preserve the sign of zero; see the f64 path.
    let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(res, x, is_zero)
}

/// `ln(1 + u)` for eight f32 lanes, `u` in `(0, 1]`. Same identity as [`log1p_pd`].
///
/// # Safety
/// Requires AVX2 + FMA; `u` must be non-negative and finite.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn log1p_ps(u: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let m = _mm256_add_ps(one, u);
    let d = _mm256_sub_ps(m, one);

    let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(d, _mm256_setzero_ps());
    let safe_d = _mm256_blendv_ps(d, one, is_zero);

    let scaled = _mm256_mul_ps(log_ps(m), _mm256_div_ps(u, safe_d));
    _mm256_blendv_ps(scaled, u, is_zero)
}

/// `ln(x)` for eight positive-normal f32 lanes. Same algorithm as [`log_pd`].
///
/// # Safety
/// Requires AVX2 + FMA; lanes must be finite, positive and normal.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn log_ps(x: __m256) -> __m256 {
    let xi = _mm256_castps_si256(x);

    let e_raw = _mm256_sub_epi32(_mm256_srli_epi32::<23>(xi), _mm256_set1_epi32(127));
    let m_bits = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007F_FFFF)),
        _mm256_set1_epi32(0x3F80_0000),
    );
    let m = _mm256_castsi256_ps(m_bits);

    let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(m, _mm256_set1_ps(std::f32::consts::SQRT_2));
    let m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5)), gt);
    let e = _mm256_add_epi32(
        e_raw,
        _mm256_and_si256(_mm256_castps_si256(gt), _mm256_set1_epi32(1)),
    );
    let ed = _mm256_cvtepi32_ps(e);

    let one = _mm256_set1_ps(1.0);
    let s = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
    let z = _mm256_mul_ps(s, s);

    let mut p = _mm256_set1_ps(LOG_COEF_F32[0]);
    for &c in LOG_COEF_F32.iter().skip(1) {
        p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(c));
    }
    let ln_m = _mm256_mul_ps(_mm256_add_ps(s, s), p);

    let acc = _mm256_fmadd_ps(ed, _mm256_set1_ps(LN2_HI_F32), ln_m);
    _mm256_fmadd_ps(ed, _mm256_set1_ps(LN2_LO_F32), acc)
}

/// True iff every lane is finite with `|x| <= limit`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tame_ps(x: __m256, limit: f32) -> bool {
    let absx = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LE_OQ>(absx, _mm256_set1_ps(limit))) == 0xFF
}

/// True iff every lane is a positive normal float.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn log_domain_ps(x: __m256) -> bool {
    let lo = _mm256_cmp_ps::<_CMP_GE_OQ>(x, _mm256_set1_ps(f32::MIN_POSITIVE));
    let hi = _mm256_cmp_ps::<_CMP_LE_OQ>(x, _mm256_set1_ps(f32::MAX));
    _mm256_movemask_ps(_mm256_and_ps(lo, hi)) == 0xFF
}

// ═══════════════════════════ per-op limits ═══════════════════════════════════

/// The input magnitude beyond which each op's vector math would overflow.
#[inline]
fn limit_f64(op: TranscendentalOp) -> f64 {
    match op {
        TranscendentalOp::Tanh => TANH_LIMIT_F64,
        TranscendentalOp::Gelu => GELU_LIMIT_F64,
        _ => EXP_LIMIT_F64,
    }
}

#[inline]
fn limit_f32(op: TranscendentalOp) -> f32 {
    match op {
        TranscendentalOp::Tanh => TANH_LIMIT_F32,
        TranscendentalOp::Gelu => GELU_LIMIT_F32,
        _ => EXP_LIMIT_F32,
    }
}

/// Evaluate `op` on four tame f64 lanes.
///
/// # Safety
/// Requires AVX2 + FMA. Lanes must have passed this op's guard
/// ([`tame_pd`] against [`limit_f64`], or [`log_domain_pd`] for `Log`).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn apply_pd(op: TranscendentalOp, x: __m256d) -> __m256d {
    let zero = _mm256_setzero_pd();
    match op {
        TranscendentalOp::Exp => exp_pd(x),
        TranscendentalOp::Log => log_pd(x),
        TranscendentalOp::Sigmoid => sigmoid_pd(x),
        TranscendentalOp::Tanh => tanh_pd(x),
        TranscendentalOp::Gelu => {
            // gelu(x) = 0.5*x*(1 + tanh(y)),  y = c*(x + 0.044715 x^3)
            //
            // evaluated as the exact identity  x * sigmoid(2y), because
            //   1 + tanh(y) == 2 / (1 + exp(-2y)) == 2*sigmoid(2y).
            // The `1 + tanh(y)` form cancels catastrophically once tanh(y) nears
            // -1 (around x = -5.5 it has already lost ~8 digits); `sigmoid` just
            // decays smoothly to 0 there, with nothing to cancel.
            let x3 = _mm256_mul_pd(_mm256_mul_pd(x, x), x);
            let y = _mm256_mul_pd(
                _mm256_set1_pd(GELU_COEF),
                _mm256_fmadd_pd(_mm256_set1_pd(GELU_CUBIC), x3, x),
            );
            _mm256_mul_pd(x, sigmoid_pd(_mm256_add_pd(y, y)))
        }
        TranscendentalOp::Elu => {
            let neg = expm1_pd(x);
            _mm256_blendv_pd(neg, x, _mm256_cmp_pd::<_CMP_GT_OQ>(x, zero))
        }
        TranscendentalOp::Selu => {
            let scale = _mm256_set1_pd(SELU_SCALE);
            let pos = _mm256_mul_pd(scale, x);
            let neg = _mm256_mul_pd(
                _mm256_mul_pd(scale, _mm256_set1_pd(SELU_ALPHA)),
                expm1_pd(x),
            );
            _mm256_blendv_pd(neg, pos, _mm256_cmp_pd::<_CMP_GT_OQ>(x, zero))
        }
        TranscendentalOp::Softplus => {
            // max(x,0) + ln1p(exp(-|x|)). `exp(-|x|)` lands in (0, 1], and using
            // ln1p rather than ln(1+u) keeps the answer accurate for very negative
            // x, where u underflows the `1 +` and a naive ln would return 0
            // instead of ~exp(x).
            let absx = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
            let u = exp_pd(_mm256_sub_pd(zero, absx));
            _mm256_add_pd(_mm256_max_pd(x, zero), log1p_pd(u))
        }
    }
}

/// `sigmoid(x) = 1 / (1 + exp(-x))`. No cancellation anywhere: the denominator
/// only ever grows.
///
/// # Safety
/// Requires AVX2 + FMA and `|x| <= EXP_LIMIT_F64`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sigmoid_pd(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let e = exp_pd(_mm256_sub_pd(_mm256_setzero_pd(), x));
    _mm256_div_pd(one, _mm256_add_pd(one, e))
}

/// `tanh(x) = u / (u + 2)` with `u = expm1(2x)`.
///
/// The textbook `1 - 2/(exp(2x)+1)` form is wrong to use here: near `x = 0` it
/// subtracts two nearly-equal numbers and bleeds precision in proportion to how
/// small the answer is (at `x = 0.003` it is already off by 3e-14 relative, versus
/// libm's 1 ULP). Routing through `expm1` removes the subtraction entirely — as
/// `x -> 0`, `u -> 2x` and the quotient tends to `x` with no cancellation.
///
/// # Safety
/// Requires AVX2 + FMA and the `TANH_LIMIT_F64` guard (so that `exp(2x)` is tame).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn tanh_pd(x: __m256d) -> __m256d {
    let two = _mm256_set1_pd(2.0);
    let u = expm1_pd(_mm256_mul_pd(two, x));
    _mm256_div_pd(u, _mm256_add_pd(u, two))
}

/// Evaluate `op` on eight tame f32 lanes.
///
/// # Safety
/// Requires AVX2 + FMA and this op's guard, as for [`apply_pd`].
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn apply_ps(op: TranscendentalOp, x: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    match op {
        TranscendentalOp::Exp => exp_ps(x),
        TranscendentalOp::Log => log_ps(x),
        TranscendentalOp::Sigmoid => sigmoid_ps(x),
        TranscendentalOp::Tanh => tanh_ps(x),
        TranscendentalOp::Gelu => {
            // x * sigmoid(2y); see the f64 path for why this beats 0.5*x*(1+tanh y).
            let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            let y = _mm256_mul_ps(
                _mm256_set1_ps(GELU_COEF as f32),
                _mm256_fmadd_ps(_mm256_set1_ps(GELU_CUBIC as f32), x3, x),
            );
            _mm256_mul_ps(x, sigmoid_ps(_mm256_add_ps(y, y)))
        }
        TranscendentalOp::Elu => {
            let neg = expm1_ps(x);
            _mm256_blendv_ps(neg, x, _mm256_cmp_ps::<_CMP_GT_OQ>(x, zero))
        }
        TranscendentalOp::Selu => {
            let scale = _mm256_set1_ps(SELU_SCALE as f32);
            let pos = _mm256_mul_ps(scale, x);
            let neg = _mm256_mul_ps(
                _mm256_mul_ps(scale, _mm256_set1_ps(SELU_ALPHA as f32)),
                expm1_ps(x),
            );
            _mm256_blendv_ps(neg, pos, _mm256_cmp_ps::<_CMP_GT_OQ>(x, zero))
        }
        TranscendentalOp::Softplus => {
            let absx = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
            let u = exp_ps(_mm256_sub_ps(zero, absx));
            _mm256_add_ps(_mm256_max_ps(x, zero), log1p_ps(u))
        }
    }
}

/// `sigmoid(x) = 1 / (1 + exp(-x))` for eight f32 lanes.
///
/// # Safety
/// Requires AVX2 + FMA and `|x| <= EXP_LIMIT_F32`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sigmoid_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let e = exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_div_ps(one, _mm256_add_ps(one, e))
}

/// `tanh(x) = u / (u + 2)` with `u = expm1(2x)`; see [`tanh_pd`].
///
/// # Safety
/// Requires AVX2 + FMA and the `TANH_LIMIT_F32` guard.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn tanh_ps(x: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0);
    let u = expm1_ps(_mm256_mul_ps(two, x));
    _mm256_div_ps(u, _mm256_add_ps(u, two))
}

// ══════════════════════════════ slice entry points ═══════════════════════════

/// Apply `op` to `src`, writing `dst`, four f64 lanes at a time.
///
/// Any 4-element block whose lanes are not all tame is computed with scalar libm,
/// so `inf`/`NaN`/out-of-range inputs get exactly libm's answer.
///
/// # Safety
/// The caller must have checked that AVX2 and FMA are available at runtime.
/// `src` and `dst` must have equal length.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn apply_slice_f64(op: TranscendentalOp, src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let limit = limit_f64(op);
    let is_log = op == TranscendentalOp::Log;

    let mut i = 0usize;
    while i + 4 <= n {
        let x = _mm256_loadu_pd(src.as_ptr().add(i));
        let ok = if is_log {
            log_domain_pd(x)
        } else {
            tame_pd(x, limit)
        };
        if ok {
            _mm256_storeu_pd(dst.as_mut_ptr().add(i), apply_pd(op, x));
        } else {
            for k in 0..4 {
                *dst.get_unchecked_mut(i + k) = scalar_f64(op, *src.get_unchecked(i + k));
            }
        }
        i += 4;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = scalar_f64(op, *src.get_unchecked(i));
        i += 1;
    }
}

/// Apply `op` to `src`, writing `dst`, eight f32 lanes at a time.
///
/// # Safety
/// As [`apply_slice_f64`], but for f32.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn apply_slice_f32(op: TranscendentalOp, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let limit = limit_f32(op);
    let is_log = op == TranscendentalOp::Log;

    let mut i = 0usize;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(src.as_ptr().add(i));
        let ok = if is_log {
            log_domain_ps(x)
        } else {
            tame_ps(x, limit)
        };
        if ok {
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), apply_ps(op, x));
        } else {
            for k in 0..8 {
                *dst.get_unchecked_mut(i + k) = scalar_f32(op, *src.get_unchecked(i + k));
            }
        }
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = scalar_f32(op, *src.get_unchecked(i));
        i += 1;
    }
}
