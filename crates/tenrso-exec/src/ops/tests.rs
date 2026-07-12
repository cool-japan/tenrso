//! Tests for the pairwise contraction engine.
//!
//! The centrepiece is [`naive_einsum`] — a deliberately slow, deliberately dumb
//! reference implementation that indexes each operand *by index character*
//! (so a repeated character such as `"ii"` naturally selects the diagonal) and
//! brute-forces the sum over every non-output index.  It shares no code with the
//! engine, which makes the randomised cross-checks a genuine guard against
//! index/stride mapping mistakes rather than a tautology.

use super::*;
use scirs2_core::random::seeded_rng;
use std::collections::HashMap;
use tenrso_core::DenseND;

// ───────────────────────────── helpers ──────────────────────────────────────

/// Decode a row-major flat index into a multi-dimensional index.
fn decode(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    let mut rest = flat;
    for d in (0..shape.len()).rev() {
        idx[d] = rest % shape[d];
        rest /= shape[d];
    }
    idx
}

/// Brute-force reference einsum for two operands.
///
/// `out[o] = Σ_{s}  A[a-indices(o, s)] · B[b-indices(o, s)]` where `s` ranges
/// over every index character that does not appear in the output.  Operands are
/// indexed by *character list*, so `"ii"` reads `A[i, i]` — the generalised
/// diagonal — exactly as einsum requires.
fn naive_einsum(spec_str: &str, a: &DenseND<f64>, b: &DenseND<f64>) -> DenseND<f64> {
    let spec = EinsumSpec::parse(spec_str).unwrap();
    let sub_a: Vec<char> = spec.inputs[0].chars().collect();
    let sub_b: Vec<char> = spec.inputs[1].chars().collect();
    let sub_o: Vec<char> = spec.output.chars().collect();

    let mut dims: HashMap<char, usize> = HashMap::new();
    for (c, &d) in sub_a.iter().zip(a.shape()) {
        dims.insert(*c, d);
    }
    for (c, &d) in sub_b.iter().zip(b.shape()) {
        dims.insert(*c, d);
    }

    let mut summed: Vec<char> = dims
        .keys()
        .copied()
        .filter(|c| !sub_o.contains(c))
        .collect();
    summed.sort_unstable();

    let out_shape: Vec<usize> = sub_o.iter().map(|c| dims[c]).collect();
    let sum_shape: Vec<usize> = summed.iter().map(|c| dims[c]).collect();
    let out_total: usize = out_shape.iter().product();
    let sum_total: usize = sum_shape.iter().product();

    let a_view = a.view();
    let b_view = b.view();
    let mut out = vec![0.0f64; out_total];

    for (flat, slot) in out.iter_mut().enumerate() {
        let out_idx = decode(flat, &out_shape);
        let mut acc = 0.0f64;
        for s in 0..sum_total {
            let sum_idx = decode(s, &sum_shape);
            let mut value: HashMap<char, usize> = HashMap::new();
            for (c, v) in sub_o.iter().zip(&out_idx) {
                value.insert(*c, *v);
            }
            for (c, v) in summed.iter().zip(&sum_idx) {
                value.insert(*c, *v);
            }
            let ai: Vec<usize> = sub_a.iter().map(|c| value[c]).collect();
            let bi: Vec<usize> = sub_b.iter().map(|c| value[c]).collect();
            acc += a_view[ai.as_slice()] * b_view[bi.as_slice()];
        }
        *slot = acc;
    }

    DenseND::from_vec(out, &out_shape).unwrap()
}

/// Deterministic pseudo-random tensor (seeded through `scirs2_core::random`).
fn random_tensor(seed: u64, shape: &[usize]) -> DenseND<f64> {
    let total: usize = shape.iter().product();
    let mut rng = seeded_rng(seed);
    let data: Vec<f64> = (0..total).map(|_| rng.random_range(-2.0..2.0f64)).collect();
    DenseND::from_vec(data, shape).unwrap()
}

/// Assert two tensors have the same shape and (almost) the same elements.
fn assert_close(actual: &DenseND<f64>, expected: &DenseND<f64>, context: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{context}: shape mismatch"
    );
    for (i, (got, want)) in actual.view().iter().zip(expected.view().iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "{context}: element {i} = {got} != {want}"
        );
    }
}

/// Run a spec through both backends and compare against the naive reference.
fn check_against_naive(spec_str: &str, a: &DenseND<f64>, b: &DenseND<f64>) {
    let spec = EinsumSpec::parse(spec_str).unwrap();
    let expected = naive_einsum(spec_str, a, b);

    let blocked = execute_dense_contraction(&spec, a, b)
        .unwrap_or_else(|e| panic!("{spec_str}: blocked backend failed: {e}"));
    assert_close(&blocked, &expected, &format!("{spec_str} [blocked]"));

    let native = execute_dense_contraction_accelerated(&spec, a, b)
        .unwrap_or_else(|e| panic!("{spec_str}: native backend failed: {e}"));
    assert_close(&native, &expected, &format!("{spec_str} [native gemm]"));
}

// ─────────────────────── pre-existing behaviour ─────────────────────────────

#[test]
fn test_execute_matmul() {
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();

    // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
    //           [3*5+4*7, 3*6+4*8] = [43, 50]
    let expected = [19.0, 22.0, 43.0, 50.0];
    let result_view = c.view();

    for (i, &expected_val) in expected.iter().enumerate() {
        let row = i / 2;
        let col = i % 2;
        let diff: f64 = result_view[[row, col]] - expected_val;
        assert!(diff.abs() < 1e-10);
    }
}

#[test]
fn test_execute_dense_contraction_matmul() {
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);

    let result_view = c.view();
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    let diff: f64 = result_view[[0, 0]] - 58.0;
    assert!(diff.abs() < 1e-10);
}

/// 3D × 2D contraction: `"ijk,kl->ijl"` with an identity `b` must return `a`.
#[test]
fn test_general_einsum_3d_times_2d() {
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();
    let b = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

    let spec = EinsumSpec::parse("ijk,kl->ijl").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2, 2]);

    let a_view = a.view();
    let c_view = c.view();
    for i in 0..2 {
        for j in 0..2 {
            for l in 0..2 {
                let diff: f64 = c_view[[i, j, l]] - a_view[[i, j, l]];
                assert!(
                    diff.abs() < 1e-10,
                    "c[{i},{j},{l}] = {} != {} (expected a[{i},{j},{l}])",
                    c_view[[i, j, l]],
                    a_view[[i, j, l]]
                );
            }
        }
    }
}

/// `"ijk,jl->ikl"`: contraction over the *middle* axis of `a`.
#[test]
fn test_general_einsum_middle_contraction() {
    let a =
        DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();
    let b = DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let spec = EinsumSpec::parse("ijk,jl->ikl").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2, 3]);

    let a_view = a.view();
    let b_view = b.view();
    let c_view = c.view();
    for i in 0..2usize {
        for k in 0..2usize {
            for l in 0..3usize {
                let expected: f64 = (0..2).map(|j| a_view[[i, j, k]] * b_view[[j, l]]).sum();
                let diff = (c_view[[i, k, l]] - expected).abs();
                assert!(
                    diff < 1e-10,
                    "c[{i},{k},{l}] = {} != {expected}",
                    c_view[[i, k, l]]
                );
            }
        }
    }
}

// ──────────────────────── the correctness fix ───────────────────────────────

/// **Regression test for the diagonal bug.**
///
/// `"ii,ii->"` is `Σ_i A[i,i]·B[i,i]` — an `n`-term sum over the diagonal — and
/// *not* `Σ_{i,j} A[i,j]·B[i,j]`, which is what the old byte-equality fast path
/// silently computed.
#[test]
fn test_double_diagonal_full_contraction() {
    // A = [[1, 2, 3],      B = [[10, 20, 30],
    //      [4, 5, 6],           [40, 50, 60],
    //      [7, 8, 9]]           [70, 80, 90]]
    let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
        .unwrap();
    let b = DenseND::<f64>::from_vec(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        &[3, 3],
    )
    .unwrap();

    let spec = EinsumSpec::parse("ii,ii->").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();

    assert!(c.shape().is_empty(), "scalar output must have rank 0");

    // Correct: 1*10 + 5*50 + 9*90 = 10 + 250 + 810 = 1070.
    // The old (buggy) full-shape walk would have produced
    //   Σ_ij A[i,j]·B[i,j] = 10+40+90+160+250+360+490+640+810 = 2850.
    let value = c.view()[[]];
    assert!(
        (value - 1070.0).abs() < 1e-10,
        "expected 1070 (diagonal), got {value}"
    );
    assert!(
        (value - 2850.0).abs() > 1.0,
        "engine reproduced the old full-shape bug"
    );

    // And the native backend agrees.
    let native = execute_dense_contraction_accelerated(&spec, &a, &b).unwrap();
    assert!((native.view()[[]] - 1070.0).abs() < 1e-10);
}

/// Diagonal of one operand contracted with the other: `"ii,ij->j"`.
///
/// `out[j] = Σ_i A[i,i] · B[i,j]`.
#[test]
fn test_diagonal_contraction() {
    // A diag = [1, 5, 9]
    let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
        .unwrap();
    // B is 3×2
    let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

    let spec = EinsumSpec::parse("ii,ij->j").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();

    assert_eq!(c.shape(), &[2]);
    // out[0] = 1*1 + 5*3 + 9*5 = 1 + 15 + 45 = 61
    // out[1] = 1*2 + 5*4 + 9*6 = 2 + 20 + 54 = 76
    let view = c.view();
    assert!((view[[0]] - 61.0).abs() < 1e-10, "got {}", view[[0]]);
    assert!((view[[1]] - 76.0).abs() < 1e-10, "got {}", view[[1]]);

    check_against_naive("ii,ij->j", &a, &b);
}

/// Batched matmul `"bij,bjk->bik"`, verified against an independent per-batch
/// 2-D matmul.
#[test]
fn test_batched_matmul() {
    let batch = 3usize;
    let (m, k, n) = (4usize, 5usize, 2usize);
    let a = random_tensor(0xB0A7, &[batch, m, k]);
    let b = random_tensor(0xB0A8, &[batch, k, n]);

    let spec = EinsumSpec::parse("bij,bjk->bik").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();
    assert_eq!(c.shape(), &[batch, m, n]);

    let a_view = a.view();
    let b_view = b.view();
    let c_view = c.view();
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let expected: f64 = (0..k)
                    .map(|p| a_view[[bi, i, p]] * b_view[[bi, p, j]])
                    .sum();
                assert!(
                    (c_view[[bi, i, j]] - expected).abs() < 1e-9,
                    "c[{bi},{i},{j}] = {} != {expected}",
                    c_view[[bi, i, j]]
                );
            }
        }
    }

    // Both backends, against the brute-force reference.
    check_against_naive("bij,bjk->bik", &a, &b);
}

// ─────────────────── randomised cross-check vs naive ────────────────────────

/// Every structural shape the engine has to handle, cross-checked against the
/// brute-force reference on pseudo-random data with **distinct extents** (so a
/// transposed or mis-strided axis cannot accidentally still match).
#[test]
fn test_randomised_cross_check() {
    let cases: &[(&str, &[usize], &[usize])] = &[
        // plain matmul and its transposes
        ("ij,jk->ik", &[3, 4], &[4, 5]),
        ("ij,jk->ki", &[3, 4], &[4, 5]),
        ("ij,kj->ik", &[3, 4], &[5, 4]),
        ("ji,jk->ik", &[4, 3], &[4, 5]),
        // batched
        ("bij,bjk->bik", &[2, 3, 4], &[2, 4, 5]),
        ("bij,bjk->kib", &[2, 3, 4], &[2, 4, 5]),
        ("bij,bij->b", &[3, 4, 5], &[3, 4, 5]),
        // higher-rank operands
        ("ijk,kl->ijl", &[2, 3, 4], &[4, 5]),
        ("ijk,jl->ikl", &[2, 3, 4], &[3, 5]),
        ("ijkl,klm->ijm", &[2, 3, 4, 5], &[4, 5, 6]),
        ("ijk,ijk->", &[2, 3, 4], &[2, 3, 4]),
        // vectors / outer products / full reductions
        ("i,j->ij", &[4], &[5]),
        ("ij,j->i", &[3, 4], &[4]),
        ("i,ij->j", &[3], &[3, 4]),
        ("ij,ij->", &[3, 4], &[3, 4]),
        ("ij,ij->ij", &[3, 4], &[3, 4]),
        // summed-out axes (present in one operand only, absent from the output)
        ("ij,jk->k", &[3, 4], &[4, 5]),
        ("ij,jk->i", &[3, 4], &[4, 5]),
        ("ijl,jk->ik", &[3, 4, 2], &[4, 5]),
        // diagonals (repeated index inside one operand)
        ("ii,ii->", &[4, 4], &[4, 4]),
        ("ii,ij->j", &[3, 3], &[3, 5]),
        ("ii,jk->ik", &[3, 3], &[4, 5]),
        ("iij,jk->ik", &[3, 3, 4], &[4, 5]),
        ("iji,jk->ik", &[3, 4, 3], &[4, 5]),
        ("ii,ii->i", &[4, 4], &[4, 4]),
        ("iijj,jk->ik", &[2, 2, 3, 3], &[3, 4]),
    ];

    for (seed, (spec_str, shape_a, shape_b)) in cases.iter().enumerate() {
        let a = random_tensor(1000 + seed as u64, shape_a);
        let b = random_tensor(2000 + seed as u64, shape_b);
        check_against_naive(spec_str, &a, &b);
    }
}

/// Same cross-check, but the operands are fed in **non-contiguous** form (the
/// result of a `permute`), which exercises the logical-order flattening path and
/// the stride bookkeeping.
#[test]
fn test_non_contiguous_inputs() {
    // `a_t` is a permuted view of a 3×4 matrix: shape [4, 3], non-contiguous.
    let a_base = random_tensor(0x1234, &[3, 4]);
    let a_t = a_base.permute(&[1, 0]).unwrap();
    assert!(!a_t.is_contiguous(), "permute must yield a strided tensor");

    // Identity gather on a *non-contiguous* operand: only the flattening path
    // (logical-order materialisation) is exercised.
    let b = random_tensor(0x5678, &[3, 5]);
    check_against_naive("ij,jk->ik", &a_t, &b);

    // Non-identity gather *and* non-contiguous: the contracted axis of `a_t` is
    // its leading one, so the operand must be transposed as well as flattened.
    let b2 = random_tensor(0x5679, &[4, 5]);
    check_against_naive("ji,jk->ik", &a_t, &b2);

    // Both operands non-contiguous, rank 3.
    let x_base = random_tensor(0x9abc, &[2, 3, 4]);
    let x = x_base.permute(&[2, 0, 1]).unwrap(); // [4, 2, 3]
    assert!(!x.is_contiguous());
    let y_base = random_tensor(0xdef0, &[3, 4, 5]);
    let y = y_base.permute(&[1, 0, 2]).unwrap(); // [4, 3, 5]
    assert!(!y.is_contiguous());
    // x is [k=4, i=2, j=3], y is [k=4, j=3, l=5] → batched over k.
    check_against_naive("kij,kjl->kil", &x, &y);

    // Non-contiguous operand with a diagonal.
    let d_base = random_tensor(0x2468, &[4, 4, 2]);
    let d = d_base.permute(&[2, 0, 1]).unwrap(); // [2, 4, 4]
    assert!(!d.is_contiguous());
    let e = random_tensor(0x1357, &[4, 3]);
    // d is [j=2, i=4, i=4] → diagonal over i → (j, i); contract i with e[i, k].
    check_against_naive("jii,ik->jk", &d, &e);
}

// ─────────────────────────── degenerate shapes ──────────────────────────────

/// A zero-length contracted axis is a sum over the empty set: all-zero output.
#[test]
fn test_zero_length_contracted_axis() {
    let a = DenseND::<f64>::zeros(&[2, 0]);
    let b = DenseND::<f64>::zeros(&[0, 3]);
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

    let c = execute_dense_contraction(&spec, &a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert!(c.view().iter().all(|&v| v == 0.0));

    let c = execute_dense_contraction_accelerated(&spec, &a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert!(c.view().iter().all(|&v| v == 0.0));
}

/// A zero-length *free* axis yields an empty tensor.
#[test]
fn test_zero_length_free_axis() {
    let a = DenseND::<f64>::zeros(&[0, 3]);
    let b = DenseND::<f64>::zeros(&[3, 4]);
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

    let c = execute_dense_contraction(&spec, &a, &b).unwrap();
    assert_eq!(c.shape(), &[0, 4]);
    assert_eq!(c.len(), 0);
}

/// Single-element tensors: exercises the `m = n = 1` corner.
#[test]
fn test_scalar_sized_operands() {
    let a = DenseND::<f64>::from_vec(vec![3.0], &[1, 1]).unwrap();
    let b = DenseND::<f64>::from_vec(vec![4.0], &[1, 1]).unwrap();
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    let c = execute_dense_contraction(&spec, &a, &b).unwrap();
    assert_eq!(c.shape(), &[1, 1]);
    assert!((c.view()[[0, 0]] - 12.0).abs() < 1e-12);
}

// ───────────────────────────── error paths ──────────────────────────────────

#[test]
fn test_error_wrong_input_count() {
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
    let err = execute_dense_contraction(&spec, &a, &a).unwrap_err();
    assert!(
        err.to_string().contains("exactly 2 inputs"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_error_rank_mismatch() {
    let a = DenseND::<f64>::zeros(&[2, 3, 4]);
    let b = DenseND::<f64>::zeros(&[3, 4]);
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    let err = execute_dense_contraction(&spec, &a, &b).unwrap_err();
    assert!(err.to_string().contains("rank"), "unexpected error: {err}");
}

#[test]
fn test_error_extent_mismatch() {
    let a = DenseND::<f64>::zeros(&[2, 3]);
    let b = DenseND::<f64>::zeros(&[4, 5]); // j is 3 in A but 4 in B
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    let err = execute_dense_contraction(&spec, &a, &b).unwrap_err();
    assert!(
        err.to_string().contains("inconsistent extents"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_error_repeated_output_index() {
    let a = DenseND::<f64>::zeros(&[2, 3]);
    let b = DenseND::<f64>::zeros(&[3, 2]);
    let spec = EinsumSpec::parse("ij,jk->ii").unwrap();
    let err = execute_dense_contraction(&spec, &a, &b).unwrap_err();
    assert!(
        err.to_string().contains("repeats index"),
        "unexpected error: {err}"
    );
}

/// The diagonal of a non-square operand is not well defined.
#[test]
fn test_error_non_square_diagonal() {
    let a = DenseND::<f64>::zeros(&[3, 4]);
    let b = DenseND::<f64>::zeros(&[3, 4]);
    let spec = EinsumSpec::parse("ii,ij->j").unwrap();
    let err = execute_dense_contraction(&spec, &a, &b).unwrap_err();
    assert!(
        err.to_string().contains("inconsistent extents"),
        "unexpected error: {err}"
    );
}

// ──────────────────────── element-type coverage ─────────────────────────────

/// `f32` takes the native `matrixmultiply` path too.
#[test]
fn test_f32_native_path() {
    let a = DenseND::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = DenseND::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

    let c = execute_dense_contraction_accelerated(&spec, &a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    assert!((c.view()[[0, 0]] - 58.0).abs() < 1e-4);

    let blocked = execute_dense_contraction(&spec, &a, &b).unwrap();
    for (x, y) in c.view().iter().zip(blocked.view().iter()) {
        assert!((x - y).abs() < 1e-4);
    }
}

/// A non-float element type must still work: the accelerated entry point falls
/// back to the portable blocked kernel.
#[test]
fn test_integer_element_type_falls_back() {
    let a = DenseND::from_vec(vec![1i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let b = DenseND::from_vec(vec![7i64, 8, 9, 10, 11, 12], &[3, 2]).unwrap();
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

    for c in [
        execute_dense_contraction(&spec, &a, &b).unwrap(),
        execute_dense_contraction_accelerated(&spec, &a, &b).unwrap(),
    ] {
        assert_eq!(c.shape(), &[2, 2]);
        let v = c.view();
        assert_eq!(v[[0, 0]], 58); // 1*7 + 2*9 + 3*11
        assert_eq!(v[[0, 1]], 64); // 1*8 + 2*10 + 3*12
        assert_eq!(v[[1, 0]], 139); // 4*7 + 5*9 + 6*11
        assert_eq!(v[[1, 1]], 154); // 4*8 + 5*10 + 6*12
    }
}

// ───────────────── large / blocked-path exercise ────────────────────────────

/// Big enough that `matmul_cache_oblivious_sequence` actually subdivides (in
/// particular it splits `k`, so the accumulate-don't-overwrite contract of the
/// blocked kernel is exercised), and both backends must still agree exactly with
/// each other and with a straightforward triple loop.
#[test]
fn test_blocked_kernel_multi_block() {
    let (m, k, n) = (67usize, 129usize, 53usize); // deliberately non-power-of-two
    let a = random_tensor(0xFEED, &[m, k]);
    let b = random_tensor(0xF00D, &[k, n]);
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

    let blocked = execute_dense_contraction(&spec, &a, &b).unwrap();
    let native = execute_dense_contraction_accelerated(&spec, &a, &b).unwrap();
    assert_eq!(blocked.shape(), &[m, n]);

    let a_view = a.view();
    let b_view = b.view();
    let blocked_view = blocked.view();
    let native_view = native.view();
    for i in 0..m {
        for j in 0..n {
            let expected: f64 = (0..k).map(|p| a_view[[i, p]] * b_view[[p, j]]).sum();
            assert!(
                (blocked_view[[i, j]] - expected).abs() < 1e-8,
                "blocked[{i},{j}] = {} != {expected}",
                blocked_view[[i, j]]
            );
            assert!(
                (native_view[[i, j]] - expected).abs() < 1e-8,
                "native[{i},{j}] = {} != {expected}",
                native_view[[i, j]]
            );
        }
    }
}

// ───────────────── VJP-style adjoint specs (tenrso-ad contract) ─────────────

/// `tenrso-ad` builds backward specs by string surgery: for
/// `C = einsum("<A>,<B>-><O>")` it evaluates `grad_A = einsum("<O>,<B>-><A>")`
/// and `grad_B = einsum("<A>,<O>-><B>")`.  Those adjoint patterns must keep
/// working — including the transposed forms they generate.
#[test]
fn test_vjp_adjoint_spec_patterns() {
    let a = random_tensor(0xADA0, &[3, 4]);
    let b = random_tensor(0xADB0, &[4, 5]);
    let grad_c = random_tensor(0xADC0, &[3, 5]);

    // forward
    check_against_naive("ij,jk->ik", &a, &b);
    // grad_A = einsum("ik,jk->ij", grad_C, B)
    check_against_naive("ik,jk->ij", &grad_c, &b);
    // grad_B = einsum("ij,ik->jk", A, grad_C)
    check_against_naive("ij,ik->jk", &a, &grad_c);

    // The same for a batched forward op.
    let ba = random_tensor(0xBDA0, &[2, 3, 4]);
    let bb = random_tensor(0xBDB0, &[2, 4, 5]);
    let bg = random_tensor(0xBDC0, &[2, 3, 5]);
    check_against_naive("bik,bjk->bij", &bg, &bb);
    check_against_naive("bij,bik->bjk", &ba, &bg);
}

// ─────────────────────────── plan unit tests ────────────────────────────────

#[test]
fn test_plan_classification_batched_matmul() {
    let spec = EinsumSpec::parse("bij,bjk->bik").unwrap();
    let plan = plan::ContractionPlan::build(&spec, &[7, 3, 4], &[7, 4, 5]).unwrap();
    assert_eq!(plan.batch, 7);
    assert_eq!(plan.m, 3);
    assert_eq!(plan.k, 4);
    assert_eq!(plan.n, 5);
    assert!(plan.out_is_identity);
    assert_eq!(plan.output_shape, vec![7, 3, 5]);
    // Both operands are already in canonical (batch, m, k) / (batch, k, n) order.
    assert!(plan.a.is_identity(7 * 3 * 4));
    assert!(plan.b.is_identity(7 * 4 * 5));
}

#[test]
fn test_plan_diagonal_stride() {
    // "ii" over a 5×5 operand: the diagonal stride must be 5 + 1 = 6.
    let spec = EinsumSpec::parse("ii,ij->j").unwrap();
    let plan = plan::ContractionPlan::build(&spec, &[5, 5], &[5, 2]).unwrap();
    assert_eq!(plan.a.kept.len(), 1);
    assert_eq!(plan.a.kept[0].dim, 5);
    assert_eq!(plan.a.kept[0].stride, 6);
    assert!(plan.a.summed.is_empty());
    assert_eq!(plan.m, 1);
    assert_eq!(plan.k, 5);
    assert_eq!(plan.n, 2);
    // The diagonal gather is *not* the identity — it must copy.
    assert!(!plan.a.is_identity(25));
}

#[test]
fn test_plan_summed_axis() {
    // "ij,jk->k": i appears only in A and not in the output → summed out of A.
    let spec = EinsumSpec::parse("ij,jk->k").unwrap();
    let plan = plan::ContractionPlan::build(&spec, &[3, 4], &[4, 5]).unwrap();
    assert_eq!(plan.a.summed.len(), 1);
    assert_eq!(plan.a.summed[0].dim, 3);
    assert_eq!(plan.a.summed[0].stride, 4); // row stride of a 3×4 row-major matrix
    assert_eq!(plan.a.kept.len(), 1); // just j
    assert_eq!(plan.m, 1);
    assert_eq!(plan.k, 4);
    assert_eq!(plan.n, 5);
}

#[test]
fn test_row_major_strides() {
    assert_eq!(plan::row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    assert_eq!(plan::row_major_strides(&[5]), vec![1]);
    assert!(plan::row_major_strides(&[]).is_empty());
}

// ══════════════════════════ unary (single-operand) ═══════════════════════════
//
// `execute_unary_einsum` is the fix for a released-API bug: a one-input spec
// emits zero pairwise steps, so `execute_plan` used to hand the *input* back as
// the "result" — wrong shape, wrong values, no error, for every one of
// `"ij->ji"`, `"ii->i"`, `"ii->"`, `"ij->i"`, …

/// Brute-force reference einsum for **one** operand.
///
/// Shares no code with the engine: it indexes the operand by *index character*
/// (so `"ii"` naturally reads `A[i, i]` — the generalised diagonal) and sums
/// over every character absent from the output.
fn naive_unary_einsum(spec_str: &str, a: &DenseND<f64>) -> DenseND<f64> {
    let spec = EinsumSpec::parse(spec_str).unwrap();
    let sub_a: Vec<char> = spec.inputs[0].chars().collect();
    let sub_o: Vec<char> = spec.output.chars().collect();

    let mut dims: HashMap<char, usize> = HashMap::new();
    for (c, &d) in sub_a.iter().zip(a.shape()) {
        dims.insert(*c, d);
    }

    let mut summed: Vec<char> = dims
        .keys()
        .copied()
        .filter(|c| !sub_o.contains(c))
        .collect();
    summed.sort_unstable();

    let out_shape: Vec<usize> = sub_o.iter().map(|c| dims[c]).collect();
    let sum_shape: Vec<usize> = summed.iter().map(|c| dims[c]).collect();
    let out_total: usize = out_shape.iter().product();
    let sum_total: usize = sum_shape.iter().product();

    let a_view = a.view();
    let mut out = vec![0.0f64; out_total];

    for (flat, slot) in out.iter_mut().enumerate() {
        let out_idx = decode(flat, &out_shape);
        let mut acc = 0.0f64;
        for s in 0..sum_total {
            let sum_idx = decode(s, &sum_shape);
            let mut value: HashMap<char, usize> = HashMap::new();
            for (c, v) in sub_o.iter().zip(&out_idx) {
                value.insert(*c, *v);
            }
            for (c, v) in summed.iter().zip(&sum_idx) {
                value.insert(*c, *v);
            }
            let ai: Vec<usize> = sub_a.iter().map(|c| value[c]).collect();
            acc += a_view[ai.as_slice()];
        }
        *slot = acc;
    }

    DenseND::from_vec(out, &out_shape).unwrap()
}

/// Run a unary spec through the engine and compare against the naive reference.
fn check_unary_against_naive(spec_str: &str, a: &DenseND<f64>) {
    let spec = EinsumSpec::parse(spec_str).unwrap();
    let expected = naive_unary_einsum(spec_str, a);
    let actual = execute_unary_einsum(&spec, a)
        .unwrap_or_else(|e| panic!("{spec_str}: unary einsum failed: {e}"));
    assert_close(&actual, &expected, &format!("{spec_str} [unary]"));
}

// ─────────────────────── hand-computed expectations ─────────────────────────

#[test]
fn test_unary_transpose() {
    // [[1, 2, 3], [4, 5, 6]]  →  [[1, 4], [2, 5], [3, 6]]
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let spec = EinsumSpec::parse("ij->ji").unwrap();
    let t = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_unary_permute_3d() {
    // A[i,j,k] with shape 2×3×4; "ijk->kij" must give B[k,i,j] = A[i,j,k].
    let a = random_tensor(0x0117, &[2, 3, 4]);
    let spec = EinsumSpec::parse("ijk->kij").unwrap();
    let b = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(b.shape(), &[4, 2, 3]);
    let (av, bv) = (a.view(), b.view());
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(
                    bv[[k, i, j]],
                    av[[i, j, k]],
                    "kij mismatch at ({i},{j},{k})"
                );
            }
        }
    }
}

#[test]
fn test_unary_diagonal() {
    // [[1, 2], [3, 4]] → diag = [1, 4]
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let spec = EinsumSpec::parse("ii->i").unwrap();
    let d = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(d.shape(), &[2]);
    assert_eq!(d.as_slice(), &[1.0, 4.0]);
}

#[test]
fn test_unary_diagonal_keeps_third_axis() {
    // A[i,i,j] for a 2×2×3 tensor: "iij->ij" keeps the diagonal in (i, i).
    // Flat row-major indices: A[i,i,j] = flat[i*6 + i*3 + j].
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let a = DenseND::from_vec(a, &[2, 2, 3]).unwrap();
    let spec = EinsumSpec::parse("iij->ij").unwrap();
    let d = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(d.shape(), &[2, 3]);
    // i=0 → flat 0,1,2 ; i=1 → flat 9,10,11
    assert_eq!(d.as_slice(), &[0.0, 1.0, 2.0, 9.0, 10.0, 11.0]);
}

#[test]
fn test_unary_trace() {
    // trace([[1, 2], [3, 4]]) = 1 + 4 = 5, as a rank-0 tensor.
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let spec = EinsumSpec::parse("ii->").unwrap();
    let t = execute_unary_einsum(&spec, &a).unwrap();

    assert!(
        t.shape().is_empty(),
        "trace must be rank-0, got {:?}",
        t.shape()
    );
    assert_eq!(t.as_slice(), &[5.0]);
}

#[test]
fn test_unary_row_sum() {
    // [[1, 2, 3], [4, 5, 6]] → row sums [6, 15]
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let spec = EinsumSpec::parse("ij->i").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(s.shape(), &[2]);
    assert_eq!(s.as_slice(), &[6.0, 15.0]);
}

#[test]
fn test_unary_reduce_to_last_axis() {
    // "ijk->k" over a 2×3×4 ramp: column k gets Σ over the 6 (i,j) pairs.
    let a: Vec<f64> = (0..24).map(|v| v as f64).collect();
    let a = DenseND::from_vec(a, &[2, 3, 4]).unwrap();
    let spec = EinsumSpec::parse("ijk->k").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(s.shape(), &[4]);
    // k-th sum = Σ_{m=0}^{5} (4m + k) = 4·15 + 6k = 60 + 6k
    assert_eq!(s.as_slice(), &[60.0, 66.0, 72.0, 78.0]);
}

#[test]
fn test_unary_full_reduction() {
    // "ij->" is the scalar sum: 1+2+3+4+5+6 = 21.
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let spec = EinsumSpec::parse("ij->").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();

    assert!(s.shape().is_empty());
    assert_eq!(s.as_slice(), &[21.0]);
}

#[test]
fn test_unary_diagonal_then_reduce() {
    // "iij->j" = Σ_i A[i,i,j], the column sums of the "iij->ij" diagonal above.
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let a = DenseND::from_vec(a, &[2, 2, 3]).unwrap();
    let spec = EinsumSpec::parse("iij->j").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(s.shape(), &[3]);
    // [0+9, 1+10, 2+11]
    assert_eq!(s.as_slice(), &[9.0, 11.0, 13.0]);
}

#[test]
fn test_unary_split_diagonal_then_reduce() {
    // "iji->j" = Σ_i A[i,j,i]: the repeated index brackets the kept one.
    // Shape 2×3×2, row-major strides (6, 2, 1) → stride(i) = 6 + 1 = 7.
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let a = DenseND::from_vec(a, &[2, 3, 2]).unwrap();
    let spec = EinsumSpec::parse("iji->j").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(s.shape(), &[3]);
    // A[0,j,0] = flat[2j] = 0, 2, 4 ; A[1,j,1] = flat[7 + 2j] = 7, 9, 11
    assert_eq!(s.as_slice(), &[7.0, 11.0, 15.0]);
}

#[test]
fn test_unary_identity_is_exact_no_op() {
    let a = random_tensor(0x1DEA, &[3, 4]);
    let spec = EinsumSpec::parse("ij->ij").unwrap();
    let b = execute_unary_einsum(&spec, &a).unwrap();

    assert_eq!(b.shape(), a.shape());
    assert_eq!(b.as_slice(), a.as_slice());

    // …and it really is recognised as a passthrough, not a re-gather.
    let plan = plan::UnaryPlan::build(&spec, a.shape()).unwrap();
    assert!(plan.is_passthrough);
}

// ───────────────────────── randomised cross-check ───────────────────────────

#[test]
fn test_unary_matches_naive_reference() {
    // Every spec class, cross-checked element-wise against the naive reference.
    // Non-square extents where possible, so a transposed stride mapping cannot
    // pass by accident.
    let cases: &[(&str, &[usize])] = &[
        ("ij->ij", &[3, 4]),
        ("ij->ji", &[3, 4]),
        ("ijk->ijk", &[2, 3, 4]),
        ("ijk->kij", &[2, 3, 4]),
        ("ijk->jik", &[2, 3, 4]),
        ("ijk->kji", &[2, 3, 4]),
        ("ii->i", &[5, 5]),
        ("ii->", &[5, 5]),
        ("iij->ij", &[3, 3, 4]),
        ("iij->ji", &[3, 3, 4]),
        ("iij->j", &[3, 3, 4]),
        ("iji->j", &[3, 4, 3]),
        ("iji->ij", &[3, 4, 3]),
        ("jii->j", &[4, 3, 3]),
        ("iii->i", &[4, 4, 4]),
        ("iii->", &[4, 4, 4]),
        ("ij->i", &[3, 4]),
        ("ij->j", &[3, 4]),
        ("ij->", &[3, 4]),
        ("ijk->k", &[2, 3, 4]),
        ("ijk->ki", &[2, 3, 4]),
        ("ijk->", &[2, 3, 4]),
        ("i->i", &[6]),
        ("i->", &[6]),
        ("ijkl->ljki", &[2, 3, 2, 4]),
        ("iijj->ij", &[3, 3, 2, 2]),
        ("iijj->", &[3, 3, 2, 2]),
    ];

    for (seed, (spec_str, shape)) in cases.iter().enumerate() {
        let a = random_tensor(0x5EED_0000 + seed as u64, shape);
        check_unary_against_naive(spec_str, &a);
    }
}

/// Non-contiguous operand: `permute` returns a *view* with permuted strides, so
/// the raw buffer order no longer matches the logical row-major order.  The
/// gather strides are expressed against the logical layout, so the engine must
/// flatten first — if it read the raw buffer it would silently transpose.
#[test]
fn test_unary_non_contiguous_input() {
    let base = random_tensor(0xC0FFEE, &[2, 3, 4]);
    let permuted = base.permute(&[2, 0, 1]).unwrap(); // logical shape 4×2×3
    assert_eq!(permuted.shape(), &[4, 2, 3]);
    assert!(
        !permuted.is_contiguous(),
        "test precondition: the permuted operand must be non-contiguous"
    );

    // Every unary class, run on a non-contiguous operand.
    for spec_str in ["ijk->ijk", "ijk->kji", "ijk->jk", "ijk->i", "ijk->"] {
        check_unary_against_naive(spec_str, &permuted);
    }

    // …and a diagonal over a non-contiguous square operand.
    let square = random_tensor(0xBEEF, &[4, 4]).permute(&[1, 0]).unwrap();
    assert!(!square.is_contiguous());
    check_unary_against_naive("ii->i", &square);
    check_unary_against_naive("ii->", &square);
}

// ──────────────────────────── unary plan / errors ───────────────────────────

#[test]
fn test_unary_plan_diagonal_strides() {
    // "iij" over a 3×3×4 operand: row-major strides are (12, 4, 1), so the
    // diagonal stride of 'i' is 12 + 4 = 16 and 'j' keeps stride 1.
    let spec = EinsumSpec::parse("iij->j").unwrap();
    let plan = plan::UnaryPlan::build(&spec, &[3, 3, 4]).unwrap();

    assert_eq!(plan.output_shape, vec![4]);
    assert_eq!(plan.gather.kept.len(), 1);
    assert_eq!(plan.gather.kept[0].dim, 4);
    assert_eq!(plan.gather.kept[0].stride, 1);
    assert_eq!(plan.gather.summed.len(), 1);
    assert_eq!(plan.gather.summed[0].dim, 3);
    assert_eq!(plan.gather.summed[0].stride, 16);
    assert!(!plan.is_passthrough);
}

#[test]
fn test_unary_plan_rejects_bad_specs() {
    // A repeated index needs a square extent.
    let spec = EinsumSpec::parse("ii->i").unwrap();
    let err = plan::UnaryPlan::build(&spec, &[2, 3]).unwrap_err();
    assert!(err.to_string().contains("inconsistent extents"), "{err}");

    // Subscript length must match the operand rank.
    let spec = EinsumSpec::parse("ijk->i").unwrap();
    let err = plan::UnaryPlan::build(&spec, &[2, 3]).unwrap_err();
    assert!(err.to_string().contains("rank"), "{err}");

    // A repeated *output* index is not valid einsum.
    let spec = EinsumSpec::parse("ij->ii").unwrap();
    let err = plan::UnaryPlan::build(&spec, &[3, 3]).unwrap_err();
    assert!(err.to_string().contains("repeats index"), "{err}");

    // Two operands are not a unary einsum.
    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    let err = plan::UnaryPlan::build(&spec, &[2, 3]).unwrap_err();
    assert!(err.to_string().contains("exactly 1 input"), "{err}");
}

#[test]
fn test_unary_zero_sized_axis() {
    // A degenerate extent must not panic and must reduce to an empty sum.
    let a = DenseND::<f64>::zeros(&[0, 3]);
    let spec = EinsumSpec::parse("ij->j").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.as_slice(), &[0.0, 0.0, 0.0]);

    let spec = EinsumSpec::parse("ij->i").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();
    assert_eq!(s.shape(), &[0]);
    assert!(s.as_slice().is_empty());

    let spec = EinsumSpec::parse("ij->").unwrap();
    let s = execute_unary_einsum(&spec, &a).unwrap();
    assert!(s.shape().is_empty());
    assert_eq!(s.as_slice(), &[0.0]);
}
