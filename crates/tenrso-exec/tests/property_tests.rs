//! Property-based tests for `tenrso-exec`
//!
//! Verifies mathematical invariants of the execution engine across
//! randomly generated inputs.  The properties checked:
//!
//! 1. Identity:         transpose(A, [0,1]) = A  (no-op permutation)
//! 2. Transpose:        transpose(A, [1,0])[j,i] = A[i,j]
//! 3. Total sum:        reduce_sum(A, all_axes) = Σ A[i,j]
//! 4. Row-sum:          reduce_sum(A, axis=1)[i] = Σ_j A[i,j]
//! 5. Trace:            einsum("ij,jk->ik", A, Iₙ) diagonal = row norms
//!    (leverages einsum contraction correctness)
//! 6. Outer product:    einsum("i,j->ij", a, b)[i,j] = a[i]*b[j]
//! 7. Linearity:        einsum("ij,jk->ik", A, α·B) = α · einsum("ij,jk->ik", A, B)
//! 8. Matmul determinism: two identical einsum calls produce the same result
//!    (checks memory-pool correctness)
//! 9. Unary einsum:     einsum("ij->ji"), ("ii->i"), ("ii->"), ("ij->i"), …
//!    cross-checked against a naive reference and against the `transpose` /
//!    `reduce` executor methods.
//!
//! Notes on design
//! ---------------
//! Single-input unary specs (`"ij->ji"`, `"ii->i"`, `"ii->"`, `"ij->i"`, …) are
//! executed by [`tenrso_exec::ops::execute_unary_einsum`], which the executor
//! short-circuits to before planning: a `Plan` schedules *pairwise* contractions
//! and there is no pair to schedule for one operand.  Until that path existed,
//! a one-input spec produced zero contraction steps and the executor returned
//! the **input tensor unchanged** — silently wrong for every unary spec.  The
//! `unary_*` properties and `regression_single_input_einsum_is_not_a_passthrough`
//! below guard that.  Properties 1–4 still exercise the `transpose` / `reduce`
//! executor methods, which remain the direct API for those operations.

use proptest::prelude::*;
use std::collections::HashMap;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{einsum_ex, CpuExecutor, ExecHints, TenrsoExecutor};

// ─── helpers ────────────────────────────────────────────────────────────────

/// Build a `TensorHandle<f64>` from a flat `Vec` and shape.
fn make_tensor(data: Vec<f64>, shape: &[usize]) -> TensorHandle<f64> {
    TensorHandle::from_dense_auto(DenseND::from_vec(data, shape).unwrap())
}

/// Strategy for a single finite f64 that is well-behaved numerically.
fn finite_f64() -> impl Strategy<Value = f64> {
    prop::num::f64::ANY.prop_filter("must be finite and small", |v| {
        v.is_finite() && v.abs() < 1e6
    })
}

/// Strategy for a `Vec<f64>` with exactly `n` finite elements.
fn finite_vec(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(finite_f64(), n..=n)
}

/// Run a single-input spec through the *public* `einsum_ex` builder.
fn unary_einsum(spec: &str, data: Vec<f64>, shape: &[usize]) -> DenseND<f64> {
    let handle = make_tensor(data, shape);
    einsum_ex::<f64>(spec)
        .inputs(&[handle])
        .hints(&ExecHints::default())
        .run()
        .unwrap_or_else(|e| panic!("{spec}: einsum_ex failed: {e}"))
        .as_dense()
        .cloned()
        .unwrap_or_else(|| panic!("{spec}: einsum_ex returned a non-dense tensor"))
}

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

/// Deliberately-naive single-operand einsum, sharing no code with the engine.
///
/// Indexes the operand **by index character** — so a repeated character reads
/// the generalised diagonal (`"ii"` → `A[i, i]`) — and brute-forces the sum over
/// every character that does not occur in the output.  Slow and obviously
/// correct: the point is that a stride/permutation mistake in the engine cannot
/// also be present here.
fn naive_unary_einsum(spec: &str, data: &[f64], shape: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let (lhs, rhs) = spec
        .split_once("->")
        .unwrap_or_else(|| panic!("{spec}: not an explicit einsum spec"));
    let sub_in: Vec<char> = lhs.chars().collect();
    let sub_out: Vec<char> = rhs.chars().collect();
    assert_eq!(sub_in.len(), shape.len(), "{spec}: rank mismatch");

    let mut dims: HashMap<char, usize> = HashMap::new();
    for (c, &d) in sub_in.iter().zip(shape) {
        dims.insert(*c, d);
    }

    let mut summed: Vec<char> = dims
        .keys()
        .copied()
        .filter(|c| !sub_out.contains(c))
        .collect();
    summed.sort_unstable();

    let out_shape: Vec<usize> = sub_out.iter().map(|c| dims[c]).collect();
    let sum_shape: Vec<usize> = summed.iter().map(|c| dims[c]).collect();
    let out_total: usize = out_shape.iter().product();
    let sum_total: usize = sum_shape.iter().product();

    // Row-major strides of the *input*, used to read A[char-indexed position].
    let mut in_strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * shape[d + 1];
    }

    let mut out = vec![0.0f64; out_total];
    for (flat, slot) in out.iter_mut().enumerate() {
        let out_idx = decode(flat, &out_shape);
        let mut acc = 0.0f64;
        for s in 0..sum_total {
            let sum_idx = decode(s, &sum_shape);
            let mut value: HashMap<char, usize> = HashMap::new();
            for (c, v) in sub_out.iter().zip(&out_idx) {
                value.insert(*c, *v);
            }
            for (c, v) in summed.iter().zip(&sum_idx) {
                value.insert(*c, *v);
            }
            let offset: usize = sub_in
                .iter()
                .zip(&in_strides)
                .map(|(c, stride)| value[c] * stride)
                .sum();
            acc += data[offset];
        }
        *slot = acc;
    }

    (out, out_shape)
}

// ─── property 1: identity permutation ───────────────────────────────────────

proptest! {
    /// `transpose(A, [0, 1])` must equal `A` element-wise.
    ///
    /// This tests that the executor's permutation code produces no mutation when
    /// given the identity permutation.
    #[test]
    fn prop_identity_permutation(data in finite_vec(4 * 5)) {
        let a = make_tensor(data.clone(), &[4, 5]);

        let mut exec = CpuExecutor::new();
        let result = exec.transpose(&a, &[0, 1]).unwrap();

        let result_dense = result.as_dense().unwrap();
        prop_assert_eq!(result_dense.shape(), &[4, 5]);

        let result_view = result_dense.view();
        for i in 0..4_usize {
            for j in 0..5_usize {
                let expected = data[i * 5 + j];
                let actual = result_view[[i, j]];
                prop_assert!(
                    (actual - expected).abs() < 1e-10,
                    "identity-permutation failed at ({i},{j}): got {actual}, expected {expected}"
                );
            }
        }
    }
}

// ─── property 2: transpose ──────────────────────────────────────────────────

proptest! {
    /// `transpose(A, [1, 0])[j, i]` must equal `A[i, j]` for all i, j.
    #[test]
    fn prop_transpose_swap(data in finite_vec(4 * 5)) {
        let a = make_tensor(data.clone(), &[4, 5]);

        let mut exec = CpuExecutor::new();
        let result = exec.transpose(&a, &[1, 0]).unwrap();

        let result_dense = result.as_dense().unwrap();
        prop_assert_eq!(result_dense.shape(), &[5, 4]);

        let result_view = result_dense.view();
        for i in 0..4_usize {
            for j in 0..5_usize {
                let expected = data[i * 5 + j];
                let actual = result_view[[j, i]];
                prop_assert!(
                    (actual - expected).abs() < 1e-10,
                    "transpose failed at ({i},{j}): got {actual}, expected {expected}"
                );
            }
        }
    }
}

// ─── property 3: total sum via reduce ───────────────────────────────────────

proptest! {
    /// `reduce_sum(A, axis=[0, 1])` must equal the arithmetic sum of all elements.
    ///
    /// Reducing over both axes of a 3×4 matrix must collapse to a single value.
    #[test]
    fn prop_total_sum_reduce(data in finite_vec(3 * 4)) {
        let a = make_tensor(data.clone(), &[3, 4]);

        let mut exec = CpuExecutor::new();
        // Sum along axis 1 first, then axis 0
        let sum_axis1 = exec
            .reduce(tenrso_exec::executor::types::ReduceOp::Sum, &a, &[1])
            .unwrap();
        let sum_all = exec
            .reduce(tenrso_exec::executor::types::ReduceOp::Sum, &sum_axis1, &[0])
            .unwrap();

        let expected_sum: f64 = data.iter().sum();

        let sum_dense = sum_all.as_dense().unwrap();
        let sum_view = sum_dense.view();
        // Shape after reducing a [3] vector along axis 0 should be [] or [1]
        let actual_sum = if sum_dense.shape().is_empty() {
            sum_view[[]]
        } else {
            sum_view[[0]]
        };

        prop_assert!(
            (actual_sum - expected_sum).abs() < 1e-8,
            "total-sum failed: got {actual_sum}, expected {expected_sum}"
        );
    }
}

// ─── property 4: row sums via reduce ────────────────────────────────────────

proptest! {
    /// `reduce_sum(A, axis=[1])[i]` must equal Σ_j A[i,j].
    #[test]
    fn prop_row_sum_reduce(data in finite_vec(3 * 4)) {
        let a = make_tensor(data.clone(), &[3, 4]);

        let mut exec = CpuExecutor::new();
        let row_sums = exec
            .reduce(tenrso_exec::executor::types::ReduceOp::Sum, &a, &[1])
            .unwrap();

        let result_dense = row_sums.as_dense().unwrap();
        prop_assert_eq!(result_dense.shape(), &[3]);

        let result_view = result_dense.view();
        for i in 0..3_usize {
            let expected_row_sum: f64 = (0..4).map(|j| data[i * 4 + j]).sum();
            let actual = result_view[[i]];
            prop_assert!(
                (actual - expected_row_sum).abs() < 1e-8,
                "row-sum failed at row {i}: got {actual}, expected {expected_row_sum}"
            );
        }
    }
}

// ─── property 5: trace via einsum ───────────────────────────────────────────

proptest! {
    /// Trace of A equals einsum("ij,jk->ik", A, I)[i,i] summed.
    ///
    /// Multiplying a square matrix by the identity produces the same matrix, so
    /// Σ_i (A·I)[i,i] = Σ_i A[i,i] = trace(A).  This drives a full 4×4 matmul
    /// through the executor and then verifies the diagonal.
    ///
    /// Two properties are tested simultaneously:
    /// (a) matmul correctness: A·I = A
    /// (b) diagonal extraction: trace(A·I) = Σ_i A[i,i]
    #[test]
    fn prop_trace_via_identity_matmul(data in finite_vec(4 * 4)) {
        // Build A (4×4) and I₄ (identity matrix).
        let a = make_tensor(data.clone(), &[4, 4]);

        let mut identity_data = vec![0.0f64; 16];
        for k in 0..4 {
            identity_data[k * 4 + k] = 1.0;
        }
        let eye = make_tensor(identity_data, &[4, 4]);

        // A · I = A via einsum.
        let ai = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[a, eye])
            .hints(&ExecHints::default())
            .run()
            .unwrap();

        let ai_dense = ai.as_dense().unwrap();
        prop_assert_eq!(ai_dense.shape(), &[4, 4]);
        let ai_view = ai_dense.view();

        // (a) A·I should equal A everywhere.
        for i in 0..4_usize {
            for j in 0..4_usize {
                let expected = data[i * 4 + j];
                let actual = ai_view[[i, j]];
                prop_assert!(
                    (actual - expected).abs() < 1e-8,
                    "A·I ≠ A at ({i},{j}): got {actual}, expected {expected}"
                );
            }
        }

        // (b) trace(A·I) = trace(A) = Σ_i A[i,i].
        let trace_ai: f64 = (0..4).map(|i| ai_view[[i, i]]).sum();
        let trace_a: f64 = (0..4).map(|i| data[i * 4 + i]).sum();
        prop_assert!(
            (trace_ai - trace_a).abs() < 1e-8,
            "trace(A·I) ≠ trace(A): {trace_ai} vs {trace_a}"
        );
    }
}

// ─── property 6: outer product via einsum ───────────────────────────────────

proptest! {
    /// einsum("i,j->ij", a, b)[i,j] must equal a[i]*b[j]  (outer product).
    #[test]
    fn prop_outer_product(
        a_data in finite_vec(3),
        b_data in finite_vec(4)
    ) {
        let a = make_tensor(a_data.clone(), &[3]);
        let b = make_tensor(b_data.clone(), &[4]);

        let result = einsum_ex::<f64>("i,j->ij")
            .inputs(&[a, b])
            .run()
            .unwrap();

        let result_dense = result.as_dense().unwrap();
        prop_assert_eq!(result_dense.shape(), &[3, 4]);

        let result_view = result_dense.view();
        for i in 0..3_usize {
            for j in 0..4_usize {
                let expected = a_data[i] * b_data[j];
                let actual = result_view[[i, j]];
                prop_assert!(
                    (actual - expected).abs() < 1e-10,
                    "outer-product failed at ({i},{j}): got {actual}, expected {expected}"
                );
            }
        }
    }
}

// ─── property 7: linearity (scalar scaling) ─────────────────────────────────

proptest! {
    /// Linearity: einsum("ij,jk->ik", A, α·B) = α · einsum("ij,jk->ik", A, B).
    ///
    /// Uses adaptive tolerance: max(1e-6, 1e-8 * |rhs|) to handle error
    /// accumulation in 3×4×5 contractions with values up to ~1e6.
    #[test]
    fn prop_linearity_scalar_factor(
        a_data in finite_vec(3 * 4),
        b_data in finite_vec(4 * 5),
        alpha in (-10.0f64..10.0f64)
            .prop_filter("nonzero alpha", |v| v.abs() > 1e-3)
    ) {
        let a = make_tensor(a_data.clone(), &[3, 4]);
        let b = make_tensor(b_data.clone(), &[4, 5]);

        // Build α·B.
        let alpha_b_data: Vec<f64> = b_data.iter().map(|v| v * alpha).collect();
        let alpha_b = make_tensor(alpha_b_data, &[4, 5]);

        // einsum(A, α·B)
        let lhs = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[a.clone(), alpha_b])
            .hints(&ExecHints::default())
            .run()
            .unwrap();

        // α · einsum(A, B)
        let ab = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[a, b])
            .hints(&ExecHints::default())
            .run()
            .unwrap();

        let lhs_dense = lhs.as_dense().unwrap();
        let ab_dense = ab.as_dense().unwrap();

        prop_assert_eq!(lhs_dense.shape(), &[3, 5]);
        prop_assert_eq!(ab_dense.shape(), &[3, 5]);

        let lhs_view = lhs_dense.view();
        let ab_view = ab_dense.view();

        for i in 0..3_usize {
            for k in 0..5_usize {
                let left = lhs_view[[i, k]];
                let right = alpha * ab_view[[i, k]];
                let tol = 1e-6_f64.max(1e-8 * right.abs());
                prop_assert!(
                    (left - right).abs() <= tol,
                    "linearity failed at ({i},{k}): lhs={left}, α·rhs={right}, diff={}",
                    (left - right).abs()
                );
            }
        }
    }
}

// ─── property 8: matmul determinism / memory-pool safety ────────────────────

proptest! {
    /// Two identical `einsum("ij,jk->ik")` calls must return bit-identical results.
    ///
    /// The executor owns a memory pool; this property verifies that pool reuse
    /// does not corrupt successive computations.
    #[test]
    fn prop_matmul_determinism(
        a_data in finite_vec(3 * 4),
        b_data in finite_vec(4 * 5)
    ) {
        let a1 = make_tensor(a_data.clone(), &[3, 4]);
        let b1 = make_tensor(b_data.clone(), &[4, 5]);
        let a2 = make_tensor(a_data, &[3, 4]);
        let b2 = make_tensor(b_data, &[4, 5]);

        let result1 = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[a1, b1])
            .run()
            .unwrap();

        let result2 = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[a2, b2])
            .run()
            .unwrap();

        let r1_dense = result1.as_dense().unwrap();
        let r2_dense = result2.as_dense().unwrap();

        prop_assert_eq!(r1_dense.shape(), &[3, 5]);
        prop_assert_eq!(r2_dense.shape(), &[3, 5]);

        let v1 = r1_dense.view();
        let v2 = r2_dense.view();

        for i in 0..3_usize {
            for k in 0..5_usize {
                let a = v1[[i, k]];
                let b = v2[[i, k]];
                prop_assert!(
                    (a - b).abs() < 1e-10,
                    "determinism broken at ({i},{k}): first={a}, second={b}"
                );
            }
        }
    }
}

// ─── property 9: unary einsum vs the naive reference ────────────────────────

proptest! {
    /// Every unary spec class, cross-checked element-wise against
    /// [`naive_unary_einsum`] through the public `einsum_ex` builder.
    ///
    /// This is the strongest guard against an index-mapping error: the engine
    /// walks summed row-major strides, the reference walks characters, and the
    /// two share no code.
    #[test]
    fn prop_unary_einsum_matches_naive_reference(data in finite_vec(3 * 3 * 4)) {
        // 3×3×4 (square in the first two axes, so diagonal specs are legal).
        let shape = [3usize, 3, 4];
        let specs = [
            "ijk->ijk", "ijk->kij", "ijk->jik", "ijk->kji", "ijk->ikj",
            "ijk->ij", "ijk->k", "ijk->ki", "ijk->",
            "iij->ij", "iij->ji", "iij->i", "iij->j", "iij->",
        ];

        for spec in specs {
            let (expected, expected_shape) = naive_unary_einsum(spec, &data, &shape);
            let actual = unary_einsum(spec, data.clone(), &shape);

            prop_assert_eq!(
                actual.shape(),
                expected_shape.as_slice(),
                "{}: shape mismatch",
                spec
            );
            for (i, (got, want)) in actual.view().iter().zip(&expected).enumerate() {
                let tol = 1e-8_f64.max(1e-10 * want.abs());
                prop_assert!(
                    (got - want).abs() <= tol,
                    "{}: element {} = {} != {}",
                    spec,
                    i,
                    got,
                    want
                );
            }
        }
    }
}

// ─── property 10: unary einsum vs the transpose / reduce methods ────────────

proptest! {
    /// `einsum("ij->ji")` must agree with `CpuExecutor::transpose(A, [1, 0])`,
    /// and `einsum("ij->i")` with `reduce(Sum, A, [1])`.  The two APIs are
    /// documented to express the same operation; before the unary path existed
    /// they disagreed, because einsum returned `A` itself.
    #[test]
    fn prop_unary_einsum_agrees_with_executor_methods(data in finite_vec(4 * 5)) {
        let a = make_tensor(data.clone(), &[4, 5]);
        let mut exec = CpuExecutor::new();

        let transposed = exec.transpose(&a, &[1, 0]).unwrap();
        let transposed = transposed.as_dense().unwrap();
        let via_einsum = unary_einsum("ij->ji", data.clone(), &[4, 5]);
        prop_assert_eq!(via_einsum.shape(), transposed.shape());
        for (got, want) in via_einsum.view().iter().zip(transposed.view().iter()) {
            prop_assert!((got - want).abs() < 1e-10, "transpose: {} != {}", got, want);
        }

        let reduced = exec
            .reduce(tenrso_exec::executor::types::ReduceOp::Sum, &a, &[1])
            .unwrap();
        let reduced = reduced.as_dense().unwrap();
        let via_einsum = unary_einsum("ij->i", data, &[4, 5]);
        prop_assert_eq!(via_einsum.shape(), reduced.shape());
        for (got, want) in via_einsum.view().iter().zip(reduced.view().iter()) {
            let tol = 1e-8_f64.max(1e-10 * want.abs());
            prop_assert!((got - want).abs() <= tol, "row-sum: {} != {}", got, want);
        }
    }
}

// ─── property 11: trace and diagonal ────────────────────────────────────────

proptest! {
    /// `einsum("ii->i")` is the diagonal and `einsum("ii->")` its sum.
    #[test]
    fn prop_unary_diagonal_and_trace(data in finite_vec(4 * 4)) {
        let diag = unary_einsum("ii->i", data.clone(), &[4, 4]);
        prop_assert_eq!(diag.shape(), &[4]);
        let diag_view = diag.view();
        for i in 0..4_usize {
            prop_assert!(
                (diag_view[[i]] - data[i * 4 + i]).abs() < 1e-12,
                "diagonal[{}] = {} != {}",
                i,
                diag_view[[i]],
                data[i * 4 + i]
            );
        }

        let trace = unary_einsum("ii->", data.clone(), &[4, 4]);
        prop_assert!(trace.shape().is_empty(), "trace must be rank-0");
        let expected: f64 = (0..4).map(|i| data[i * 4 + i]).sum();
        let actual = trace.view()[[]];
        let tol = 1e-8_f64.max(1e-10 * expected.abs());
        prop_assert!(
            (actual - expected).abs() <= tol,
            "trace = {} != {}",
            actual,
            expected
        );
    }
}

// ─── regression: single-input einsum used to be a passthrough ───────────────

/// **Regression test for the silent single-input einsum bug.**
///
/// A one-operand spec emits zero pairwise contraction steps, so the executor's
/// plan loop had nothing to run and handed the *input tensor* back as the
/// "result": `einsum_ex("ij->ji")` on a 2×3 matrix returned that same 2×3 matrix
/// with its values untouched — wrong shape, wrong values, no error.  Shipped in
/// 0.1.0.
///
/// The assertions below are deliberately phrased as "the result is not the
/// input": they fail loudly against the old behaviour instead of merely checking
/// that the new behaviour is self-consistent.
#[test]
fn regression_single_input_einsum_is_not_a_passthrough() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // "ij->ji" on a 2×3 matrix: the shape alone catches the passthrough.
    let transposed = unary_einsum("ij->ji", data.clone(), &[2, 3]);
    assert_ne!(
        transposed.shape(),
        &[2, 3],
        "einsum_ex(\"ij->ji\") returned the input's shape — the passthrough bug is back"
    );
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // A *square* operand hides the shape symptom, so check the values.
    let square = vec![1.0, 2.0, 3.0, 4.0];
    let transposed = unary_einsum("ij->ji", square.clone(), &[2, 2]);
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_ne!(
        transposed.as_slice(),
        square.as_slice(),
        "einsum_ex(\"ij->ji\") returned the input unchanged — the passthrough bug is back"
    );
    assert_eq!(transposed.as_slice(), &[1.0, 3.0, 2.0, 4.0]);

    // The other unary classes: each used to return the full 2×2 input.
    let diagonal = unary_einsum("ii->i", square.clone(), &[2, 2]);
    assert_eq!(
        diagonal.shape(),
        &[2],
        "\"ii->i\" must extract the diagonal"
    );
    assert_eq!(diagonal.as_slice(), &[1.0, 4.0]);

    let trace = unary_einsum("ii->", square.clone(), &[2, 2]);
    assert!(trace.shape().is_empty(), "\"ii->\" must produce a scalar");
    assert_eq!(trace.as_slice(), &[5.0]);

    let row_sums = unary_einsum("ij->i", square.clone(), &[2, 2]);
    assert_eq!(row_sums.shape(), &[2], "\"ij->i\" must reduce an axis");
    assert_eq!(row_sums.as_slice(), &[3.0, 7.0]);

    // Identity is the one spec that *is* a passthrough — and must stay correct.
    let identity = unary_einsum("ij->ij", square.clone(), &[2, 2]);
    assert_eq!(identity.shape(), &[2, 2]);
    assert_eq!(identity.as_slice(), square.as_slice());
}
