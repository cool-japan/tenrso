//! Property-based tests for `tenrso-exec`
//!
//! Verifies mathematical invariants of the execution engine across
//! randomly generated inputs.  Eight properties are checked:
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
//!
//! Notes on design
//! ---------------
//! The executor's `einsum` API requires **≥ 2 inputs** — single-input unary
//! specs ("ij->ij", "ij->ji", "ij->") are not routed through the general
//! einsum path by this executor.  Properties 1–4 therefore test the
//! `transpose` and `reduce` executor methods, which are the documented way
//! to express those operations.  This is consistent with the existing unit
//! tests in `src/executor/functions_tests/`.

use proptest::prelude::*;
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
