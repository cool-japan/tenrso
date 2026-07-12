//! Tests for the AVX2 element-wise kernels.
//!
//! Two things need proving, and they are different things:
//!
//! 1. **The kernels are correct** — every SIMD result matches scalar libm to
//!    within ~1 ULP on the fast path, and *exactly* on the guarded edge cases.
//! 2. **The kernels are actually reached** — `enable_simd` selects a genuinely
//!    different code path, and the ops with no kernel honestly say so rather than
//!    quietly running scalar code under a SIMD-sounding name.
//!
//! The old module could have passed (1) trivially: its "SIMD" `exp` was literally
//! `input.mapv(|v| v.exp())`, so it agreed with the scalar path bit-for-bit. Only
//! (2) catches that, which is why those tests are here.

use super::*;
use tenrso_core::TensorHandle;

/// Every op the SIMD layer claims to accelerate.
const ALL_OPS: [TranscendentalOp; 8] = [
    TranscendentalOp::Exp,
    TranscendentalOp::Log,
    TranscendentalOp::Sigmoid,
    TranscendentalOp::Tanh,
    TranscendentalOp::Gelu,
    TranscendentalOp::Elu,
    TranscendentalOp::Selu,
    TranscendentalOp::Softplus,
];

fn elem_op_of(op: TranscendentalOp) -> ElemOp {
    match op {
        TranscendentalOp::Exp => ElemOp::Exp,
        TranscendentalOp::Log => ElemOp::Log,
        TranscendentalOp::Sigmoid => ElemOp::Sigmoid,
        TranscendentalOp::Tanh => ElemOp::Tanh,
        TranscendentalOp::Gelu => ElemOp::Gelu,
        TranscendentalOp::Elu => ElemOp::Elu,
        TranscendentalOp::Selu => ElemOp::Selu,
        TranscendentalOp::Softplus => ElemOp::Softplus,
    }
}

/// A spread of inputs in the range each op is actually used over. `Log` needs
/// positives, so it gets its own sweep.
fn sweep_f64(op: TranscendentalOp, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            match op {
                TranscendentalOp::Log => 1e-8 + t * 1e4,
                // -12..12 covers the interesting part of every activation, and
                // crosses zero so the blends get exercised in both directions.
                _ => -12.0 + 24.0 * t,
            }
        })
        .collect()
}

fn max_rel_err(actual: &[f64], expect: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expect.iter())
        .map(|(&a, &e)| {
            if e == 0.0 {
                a.abs()
            } else {
                ((a - e) / e).abs()
            }
        })
        .fold(0.0f64, f64::max)
}

// ─────────────────────────── (1) the kernels are correct ─────────────────────

/// Each f64 kernel must track scalar libm to ~1 ULP across its working range.
///
/// A tolerance of 1e-14 is ~45x the f64 unit roundoff — loose enough not to be
/// brittle, but *far* tighter than a wrong polynomial or a botched range
/// reduction could ever sneak through (those miss by 1e-3 or worse).
#[test]
fn f64_kernels_match_libm() {
    if !simd_available() {
        return;
    }
    let n = 4096;
    let executor = CpuExecutor::new();

    for op in ALL_OPS {
        let src = sweep_f64(op, n);
        let dense = DenseND::from_vec(src.clone(), &[n]).unwrap();

        let got = simd_elem_op(&elem_op_of(op), &dense, false, &executor)
            .unwrap_or_else(|| panic!("{op:?}: expected the SIMD path to be taken"));
        let got = got.as_slice().expect("contiguous").to_vec();

        let want: Vec<f64> = src.iter().map(|&v| scalar_f64(op, v)).collect();
        let err = max_rel_err(&got, &want);
        assert!(err < 1e-14, "{op:?}: max relative error {err:.3e} vs libm");
    }
}

/// Same for f32, at f32 precision (unit roundoff 6e-8).
#[test]
fn f32_kernels_match_libm() {
    if !simd_available() {
        return;
    }
    let n = 4096;
    let executor = CpuExecutor::new();

    for op in ALL_OPS {
        let src: Vec<f32> = sweep_f64(op, n).into_iter().map(|v| v as f32).collect();
        let dense = DenseND::from_vec(src.clone(), &[n]).unwrap();

        let got = simd_elem_op(&elem_op_of(op), &dense, false, &executor)
            .unwrap_or_else(|| panic!("{op:?}: expected the SIMD path to be taken"));
        let got = got.as_slice().expect("contiguous").to_vec();

        for (i, (&a, &s)) in got.iter().zip(src.iter()).enumerate() {
            let e = scalar_f32(op, s);
            let rel = if e == 0.0 {
                a.abs()
            } else {
                ((a - e) / e).abs()
            };
            assert!(
                rel < 1e-5,
                "{op:?}[{i}] x={s}: got {a}, want {e}, rel {rel:.3e}"
            );
        }
    }
}

/// The per-op limit beyond which the kernel defers a lane to scalar libm.
/// Mirrors `avx2::limit_f64`; kept in the test so a drift between the guard and
/// what the test believes the guard is would show up as a failure.
fn guard_limit(op: TranscendentalOp) -> f64 {
    match op {
        TranscendentalOp::Tanh => 350.0,
        TranscendentalOp::Gelu => 20.0,
        _ => 700.0,
    }
}

/// Would the kernel route this lane through the vector math, or defer it to
/// scalar libm? Mirrors the runtime guard exactly.
fn goes_through_simd(op: TranscendentalOp, x: f64) -> bool {
    match op {
        // `log` needs a positive *normal* double.
        TranscendentalOp::Log => (f64::MIN_POSITIVE..=f64::MAX).contains(&x),
        // everything else needs finite and within the op's magnitude limit.
        _ => x.is_finite() && x.abs() <= guard_limit(op),
    }
}

/// The whole point of the range guard: NaN / inf / overflow / (for `log`)
/// non-positive inputs must never reach the vector math. Those lanes are deferred
/// to scalar libm and so must come back **bit-identical** to it — not merely
/// close. A guard that is off by a lane, or a "fast" path that silently produces
/// garbage for `exp(1e9)`, fails here.
///
/// Lanes the guard *accepts* (finite, in range) run the SIMD approximation and are
/// only ~1 ULP from libm — that is `f64_kernels_match_libm`'s job, and this test
/// deliberately does not demand bit-identity of them (it would be asserting that
/// two different algorithms round identically, which is false and would be a
/// dishonest test).
#[test]
fn guarded_edge_cases_are_bit_identical_to_libm() {
    if !simd_available() {
        return;
    }
    let executor = CpuExecutor::new();

    let edges = [
        0.0f64,
        -0.0,
        1.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        5e-324, // smallest subnormal
        1e9,
        -1e9,
        710.0,  // just past the exp guard
        -710.0, // just past it the other way
        700.0,  // exactly on it — accepted, so runs SIMD
        -700.0,
    ];

    for op in ALL_OPS {
        // Pad to a length that is neither a multiple of 4 nor of 8, so the vector
        // body, the guarded blocks and the scalar tail all run.
        let mut src = edges.to_vec();
        while src.len() < SIMD_MIN_ELEMS + 3 {
            src.push(0.5);
        }
        let dense = DenseND::from_vec(src.clone(), &[src.len()]).unwrap();

        let got = simd_elem_op(&elem_op_of(op), &dense, false, &executor)
            .unwrap_or_else(|| panic!("{op:?}: expected the SIMD path to be taken"));
        let got = got.as_slice().expect("contiguous");

        // At least one genuinely guarded lane must be exercised, or the test is
        // vacuous for this op.
        let mut guarded_seen = 0usize;

        for (i, (&a, &s)) in got.iter().zip(src.iter()).enumerate() {
            let e = scalar_f64(op, s);
            if goes_through_simd(op, s) {
                // In-range: SIMD approximation, ~1 ULP. (NaN result only if input
                // was NaN, which is not in-range, so `e` is finite here.)
                if e != 0.0 && e.is_finite() {
                    let rel = ((a - e) / e).abs();
                    assert!(rel < 1e-14, "{op:?}[{i}] x={s:e}: rel {rel:.3e}");
                }
            } else {
                guarded_seen += 1;
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "{op:?}[{i}] x={s:e}: guarded lane not bit-identical — \
                     SIMD gave {a:e}, libm gives {e:e}"
                );
            }
        }

        assert!(
            guarded_seen > 0,
            "{op:?}: no guarded lane was exercised — the bit-identity check is vacuous"
        );
    }
}

/// Signed zero of the odd activations must match libm: `tanh(-0.0)`, `elu(-0.0)`
/// and `selu(-0.0)` are all `-0.0`, not `+0.0`. The naive `2^n * expm1(r)`
/// reconstruction yields `+0.0` there, so this pins the sign-of-zero fix in
/// `expm1`. (`-0.0` is an *in-range* input, so it is not covered by the guarded
/// edge test above.)
#[test]
fn signed_zero_is_preserved_for_odd_activations() {
    if !simd_available() {
        return;
    }
    let executor = CpuExecutor::new();

    for op in [
        TranscendentalOp::Tanh,
        TranscendentalOp::Elu,
        TranscendentalOp::Selu,
        TranscendentalOp::Gelu,
    ] {
        let mut src = vec![-0.0f64, 0.0];
        while src.len() < SIMD_MIN_ELEMS + 1 {
            src.push(0.25);
        }
        let dense = DenseND::from_vec(src, &[SIMD_MIN_ELEMS + 1]).unwrap();
        let got = simd_elem_op(&elem_op_of(op), &dense, false, &executor).expect("SIMD");
        let got = got.as_slice().expect("contiguous");

        assert_eq!(
            got[0].to_bits(),
            scalar_f64(op, -0.0).to_bits(),
            "{op:?}(-0.0): got {} (bits {:x}), want {}",
            got[0],
            got[0].to_bits(),
            scalar_f64(op, -0.0)
        );
        assert_eq!(
            got[1].to_bits(),
            scalar_f64(op, 0.0).to_bits(),
            "{op:?}(+0.0) sign flipped"
        );
    }
}

/// Every block-alignment case: lengths that leave a lane remainder, straddle the
/// guard, or sit exactly on a vector boundary.
#[test]
fn ragged_lengths_are_handled() {
    if !simd_available() {
        return;
    }
    let executor = CpuExecutor::new();

    for len in [
        SIMD_MIN_ELEMS,
        SIMD_MIN_ELEMS + 1,
        SIMD_MIN_ELEMS + 3,
        SIMD_MIN_ELEMS + 7,
        1000,
        1024,
        1031,
    ] {
        let src: Vec<f64> = (0..len).map(|i| -5.0 + (i as f64) * 0.011).collect();
        let dense = DenseND::from_vec(src.clone(), &[len]).unwrap();

        let got = simd_elem_op(&ElemOp::Exp, &dense, false, &executor)
            .unwrap_or_else(|| panic!("len {len}: expected SIMD"));
        let got = got.as_slice().expect("contiguous");

        for (i, (&a, &s)) in got.iter().zip(src.iter()).enumerate() {
            let want = s.exp();
            let rel = ((a - want) / want).abs();
            assert!(rel < 1e-14, "len {len}, [{i}]: rel {rel:.3e}");
        }
    }
}

/// Splitting the work over rayon must not change a single bit: each task owns a
/// disjoint chunk and the kernel is a pure map, so the result is independent of
/// the thread count.
#[test]
fn parallel_simd_matches_serial_bit_for_bit() {
    if !simd_available() {
        return;
    }
    let n = 40_000;
    let src: Vec<f64> = (0..n).map(|i| -8.0 + (i as f64) * 4e-4).collect();
    let dense = DenseND::from_vec(src, &[n]).unwrap();

    let serial_exec = CpuExecutor::new();
    let serial = simd_elem_op(&ElemOp::Gelu, &dense, false, &serial_exec).expect("SIMD");

    for threads in [1usize, 2, 4] {
        let exec = CpuExecutor::with_threads(threads).unwrap();
        let parallel = simd_elem_op(&ElemOp::Gelu, &dense, true, &exec).expect("SIMD");
        for (i, (a, b)) in parallel.iter().zip(serial.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{threads} threads, element {i}: {a} vs {b}"
            );
        }
    }
}

// ──────────────────── (2) the kernels are actually reached ───────────────────

/// The dispatch must accept exactly the ops it has a kernel for, and honestly
/// decline the rest.
///
/// This is the anti-facade test. The module this replaced advertised a
/// `simd_relu`, a `simd_sqrt`, a `simd_abs`… all of which were plain `mapv`. Here
/// those ops have **no** SIMD path, and the dispatch says so out loud.
#[test]
fn dispatch_accepts_only_ops_with_a_real_kernel() {
    for op in [
        ElemOp::Exp,
        ElemOp::Log,
        ElemOp::Sigmoid,
        ElemOp::Tanh,
        ElemOp::Gelu,
        ElemOp::Elu,
        ElemOp::Selu,
        ElemOp::Softplus,
    ] {
        assert!(
            transcendental_of(&op).is_some(),
            "{op:?} should have a SIMD kernel"
        );
    }

    // Bandwidth-bound: measured no faster (or slower) with intrinsics, so there is
    // deliberately no kernel and no claim of one.
    for op in [
        ElemOp::Neg,
        ElemOp::Abs,
        ElemOp::Sqrt,
        ElemOp::Sqr,
        ElemOp::Recip,
        ElemOp::Sign,
        ElemOp::ReLU,
        ElemOp::Sin,
        ElemOp::Cos,
    ] {
        assert!(
            transcendental_of(&op).is_none(),
            "{op:?} must NOT claim a SIMD kernel it does not have"
        );
    }
}

/// `simd_elem_op` must decline — not silently run scalar code — whenever a
/// precondition fails.
#[test]
fn dispatch_declines_when_it_cannot_apply() {
    let executor = CpuExecutor::new();

    // Op with no kernel, at a size that would otherwise qualify.
    let big = DenseND::from_vec(vec![1.0f64; 4096], &[4096]).unwrap();
    assert!(simd_elem_op(&ElemOp::ReLU, &big, false, &executor).is_none());

    // Too small to be worth it.
    let small = DenseND::from_vec(vec![1.0f64; 8], &[8]).unwrap();
    assert!(simd_elem_op(&ElemOp::Exp, &small, false, &executor).is_none());

    // A contiguous tensor of the same size and op IS accepted — so the declines
    // below are about the precondition, not about the op being unsupported.
    let contiguous = DenseND::from_vec((0..4096).map(|i| i as f64).collect(), &[64, 64]).unwrap();
    assert!(simd_elem_op(&ElemOp::Exp, &contiguous, false, &executor).is_some());

    // Non-contiguous: `reversed_axes` keeps the same buffer but swaps the strides,
    // so there is no contiguous slice for the vector loads to walk. The dispatch
    // must decline rather than read the wrong elements.
    let permuted = DenseND::from_array(contiguous.as_array().clone().reversed_axes());
    assert!(
        permuted.as_array().as_slice().is_none(),
        "expected a strided array"
    );
    assert!(simd_elem_op(&ElemOp::Exp, &permuted, false, &executor).is_none());
}

/// `enable_simd` must change what actually runs.
///
/// The SIMD `exp` is a polynomial + `2^n` reconstruction; libm's `exp` is a
/// different algorithm. They agree to ~1 ULP but they are **not** the same
/// function bit-for-bit, so if the flag really switches code paths, some element
/// must differ in its low bit. If someone re-wires `enable_simd` back to a plain
/// `mapv`, this test fails — which is exactly what it is for.
#[test]
fn enable_simd_selects_a_genuinely_different_path() {
    if !simd_available() {
        return;
    }
    let n = 20_000;
    let src: Vec<f64> = (0..n).map(|i| -6.0 + (i as f64) * 6e-4).collect();
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(src, &[n]).unwrap());

    let mut with = CpuExecutor::new().with_simd(true);
    let mut without = CpuExecutor::new().with_simd(false);

    let a = with.parallel_elem_op(ElemOp::Exp, &handle).unwrap();
    let b = without.parallel_elem_op(ElemOp::Exp, &handle).unwrap();

    let av = a.as_dense().unwrap().as_array().clone();
    let bv = b.as_dense().unwrap().as_array().clone();

    let mut differing_bits = 0usize;
    let mut worst_rel = 0.0f64;
    for (&x, &y) in av.iter().zip(bv.iter()) {
        if x.to_bits() != y.to_bits() {
            differing_bits += 1;
        }
        let rel = ((x - y) / y).abs();
        worst_rel = worst_rel.max(rel);
    }

    // Numerically the same function...
    assert!(
        worst_rel < 1e-14,
        "SIMD and scalar exp disagree by {worst_rel:.3e} — that is a bug, not a rounding difference"
    );
    // ...but demonstrably not the same code.
    assert!(
        differing_bits > 0,
        "with_simd(true) and with_simd(false) produced bit-identical output for all {n} \
         elements — the SIMD flag is not selecting a different implementation"
    );
}
