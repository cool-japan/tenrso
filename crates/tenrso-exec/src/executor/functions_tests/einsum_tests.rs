//! Tests for the `einsum` trait method on `CpuExecutor`.

#![allow(clippy::unnecessary_cast)]

use super::super::{functions::TenrsoExecutor, types::CpuExecutor};
use crate::hints::ExecHints;
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_cpu_executor_matmul() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .einsum("ij,jk->ik", &[handle_a, handle_b], &ExecHints::default())
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    let diff: f64 = result_view[[0, 0]] - 58.0;
    assert!(diff.abs() < 1e-10);
}

#[test]
fn test_cpu_executor_input_validation() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.einsum("ij,jk->ik", &[handle_a], &ExecHints::default());
    assert!(result.is_err());
}

#[test]
fn test_cpu_executor_three_tensors() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let c = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let handle_c = TensorHandle::from_dense_auto(c);
    let result = executor
        .einsum(
            "ij,jk,kl->il",
            &[handle_a, handle_b, handle_c],
            &ExecHints::default(),
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    let val: f64 = result_view[[0, 0]];
    assert!(val.abs() > 0.0);
}

#[test]
fn test_cpu_executor_outer_then_contract() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let b = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let c = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let handle_c = TensorHandle::from_dense_auto(c);
    let result = executor
        .einsum(
            "i,j,ij->",
            &[handle_a, handle_b, handle_c],
            &ExecHints::default(),
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert!(result_dense.shape().is_empty() || result_dense.shape() == [1]);
    let result_view = result_dense.view();
    let result_val = if result_dense.shape().is_empty() {
        result_view[[]]
    } else {
        result_view[[0]]
    };
    let diff: f64 = result_val - 78.0;
    assert!(diff.abs() < 1e-10, "Expected 78.0, got {}", result_val);
}

// ═══════════════════════ single-operand (unary) einsum ═══════════════════════
//
// Regression suite for a released-API bug: a 1-input spec produces zero pairwise
// contraction steps, so the executor used to return the *input tensor* as the
// "result" — wrong shape, wrong values, and no error. Every assertion below
// fails against that behaviour.

/// Run a unary spec through the public executor API and return the dense result.
fn unary(spec: &str, data: Vec<f64>, shape: &[usize]) -> DenseND<f64> {
    let mut executor = CpuExecutor::new();
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, shape).unwrap());
    executor
        .einsum(spec, &[handle], &ExecHints::default())
        .unwrap_or_else(|e| panic!("{spec}: einsum failed: {e}"))
        .as_dense()
        .cloned()
        .unwrap_or_else(|| panic!("{spec}: executor returned a non-dense tensor"))
}

#[test]
fn test_cpu_executor_unary_transpose() {
    let t = unary("ij->ji", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_cpu_executor_unary_permute_3d() {
    // "ijk->kij" on a 2×2×2 ramp: B[k,i,j] = A[i,j,k].
    let t = unary(
        "ijk->kij",
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[2, 2, 2],
    );
    assert_eq!(t.shape(), &[2, 2, 2]);
    // k=0 → A[·,·,0] = [0, 2, 4, 6]; k=1 → A[·,·,1] = [1, 3, 5, 7]
    assert_eq!(t.as_slice(), &[0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn test_cpu_executor_unary_diagonal() {
    let t = unary("ii->i", vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(t.shape(), &[2]);
    assert_eq!(t.as_slice(), &[1.0, 4.0]);
}

#[test]
fn test_cpu_executor_unary_diagonal_keeps_axis() {
    // "iij->ij" over a 2×2×3 ramp: rows are A[0,0,·] and A[1,1,·].
    let data: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let t = unary("iij->ij", data, &[2, 2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.as_slice(), &[0.0, 1.0, 2.0, 9.0, 10.0, 11.0]);
}

#[test]
fn test_cpu_executor_unary_trace() {
    let t = unary("ii->", vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(
        t.shape().is_empty(),
        "trace must be rank-0, got {:?}",
        t.shape()
    );
    assert_eq!(t.as_slice(), &[5.0]);
}

#[test]
fn test_cpu_executor_unary_axis_reduction() {
    let t = unary("ij->i", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(t.shape(), &[2]);
    assert_eq!(t.as_slice(), &[6.0, 15.0]);

    let t = unary("ij->j", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_cpu_executor_unary_full_reduction() {
    let t = unary("ij->", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(t.shape().is_empty());
    assert_eq!(t.as_slice(), &[21.0]);
}

#[test]
fn test_cpu_executor_unary_diagonal_then_reduce() {
    let data: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let t = unary("iij->j", data, &[2, 2, 3]);
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.as_slice(), &[9.0, 11.0, 13.0]);
}

#[test]
fn test_cpu_executor_unary_identity() {
    let t = unary("ij->ij", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_cpu_executor_unary_agrees_with_transpose_and_reduce_methods() {
    // The two documented workarounds must now agree with the einsum path.
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = (0..12).map(|v| (v as f64) * 0.75 - 3.0).collect();
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data.clone(), &[3, 4]).unwrap());

    // `transpose` returns a permuted (non-contiguous) view, so compare through
    // the logical iteration order rather than the raw buffers.
    let via_transpose = executor.transpose(&handle, &[1, 0]).unwrap();
    let via_einsum = executor
        .einsum(
            "ij->ji",
            std::slice::from_ref(&handle),
            &ExecHints::default(),
        )
        .unwrap();
    assert_same_tensor(&via_einsum, &via_transpose, "ij->ji vs transpose");

    let via_reduce = executor
        .reduce(crate::executor::types::ReduceOp::Sum, &handle, &[1])
        .unwrap();
    let via_einsum = executor
        .einsum("ij->i", &[handle], &ExecHints::default())
        .unwrap();
    assert_same_tensor(&via_einsum, &via_reduce, "ij->i vs reduce_sum");
}

/// Assert two handles hold the same dense tensor (shape and logical elements).
fn assert_same_tensor(lhs: &TensorHandle<f64>, rhs: &TensorHandle<f64>, context: &str) {
    let lhs = lhs
        .as_dense()
        .unwrap_or_else(|| panic!("{context}: lhs not dense"));
    let rhs = rhs
        .as_dense()
        .unwrap_or_else(|| panic!("{context}: rhs not dense"));
    assert_eq!(lhs.shape(), rhs.shape(), "{context}: shape mismatch");
    for (i, (a, b)) in lhs.view().iter().zip(rhs.view().iter()).enumerate() {
        let diff: f64 = a - b;
        assert!(diff.abs() < 1e-12, "{context}: element {i}: {a} != {b}");
    }
}

#[test]
fn test_cpu_executor_unary_rejects_bad_spec() {
    let mut executor = CpuExecutor::new();
    // "ii" needs a square operand.
    let handle = TensorHandle::from_dense_auto(
        DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
    );
    let err = executor
        .einsum("ii->i", &[handle], &ExecHints::default())
        .unwrap_err();
    assert!(err.to_string().contains("inconsistent extents"), "{err}");
}

// ════════════════════ multi-operand step-semantics regressions ═══════════════
//
// The planner decides each step's output subscript from the two operands alone,
// dropping every index they *share* — which silently sums batch indices that are
// still needed downstream, and emits its indices alphabetically rather than in
// the requested order.  The executor now derives the step subscripts itself.

#[test]
fn test_three_operand_batch_index_is_not_summed() {
    // "bij,bjk,bkl->bil" — 'b' is shared by every operand and appears in the
    // output.  A pairwise step must NOT contract it.
    let mut executor = CpuExecutor::new();
    let b_dim = 2;

    let a: Vec<f64> = (0..b_dim * 2 * 2).map(|v| (v as f64) + 1.0).collect();
    let bmat: Vec<f64> = (0..b_dim * 2 * 2).map(|v| (v as f64) * 0.5).collect();
    let c: Vec<f64> = (0..b_dim * 2 * 2)
        .map(|v| 2.0 - (v as f64) * 0.25)
        .collect();

    let ha = TensorHandle::from_dense_auto(DenseND::from_vec(a.clone(), &[b_dim, 2, 2]).unwrap());
    let hb =
        TensorHandle::from_dense_auto(DenseND::from_vec(bmat.clone(), &[b_dim, 2, 2]).unwrap());
    let hc = TensorHandle::from_dense_auto(DenseND::from_vec(c.clone(), &[b_dim, 2, 2]).unwrap());

    let result = executor
        .einsum("bij,bjk,bkl->bil", &[ha, hb, hc], &ExecHints::default())
        .unwrap();
    let result = result.as_dense().unwrap();
    assert_eq!(result.shape(), &[b_dim, 2, 2]);

    // Reference: per-batch triple matrix product, computed by hand.
    let at = |b: usize, i: usize, j: usize| a[b * 4 + i * 2 + j];
    let bt = |b: usize, i: usize, j: usize| bmat[b * 4 + i * 2 + j];
    let ct = |b: usize, i: usize, j: usize| c[b * 4 + i * 2 + j];

    let view = result.view();
    for b in 0..b_dim {
        for i in 0..2 {
            for l in 0..2 {
                let expected: f64 = (0..2)
                    .map(|j| {
                        (0..2)
                            .map(|k| at(b, i, j) * bt(b, j, k) * ct(b, k, l))
                            .sum::<f64>()
                    })
                    .sum();
                let got = view[[b, i, l]];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "batch {b} [{i},{l}]: got {got}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn test_three_operand_output_index_order_is_respected() {
    // "ij,jk,kl->li" asks for the *transposed* chain product.
    let mut executor = CpuExecutor::new();
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
    let c: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0]; // identity → result = A·B, transposed

    let ha = TensorHandle::from_dense_auto(DenseND::from_vec(a, &[2, 2]).unwrap());
    let hb = TensorHandle::from_dense_auto(DenseND::from_vec(b, &[2, 2]).unwrap());
    let hc = TensorHandle::from_dense_auto(DenseND::from_vec(c, &[2, 2]).unwrap());

    let result = executor
        .einsum("ij,jk,kl->li", &[ha, hb, hc], &ExecHints::default())
        .unwrap();
    let result = result.as_dense().unwrap();

    // A·B = [[19, 22], [43, 50]] → transposed = [[19, 43], [22, 50]]
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), &[19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn test_three_operand_elementwise_product() {
    // Every index is shared by every operand *and* kept in the output: the
    // greedy planner cannot express this (its pairwise output is empty and
    // unparseable), so the executor falls back to a sequential order.
    let mut executor = CpuExecutor::new();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = vec![2.0, 2.0, 0.5, 0.5];

    let ha = TensorHandle::from_dense_auto(DenseND::from_vec(a.clone(), &[2, 2]).unwrap());
    let hb = TensorHandle::from_dense_auto(DenseND::from_vec(b.clone(), &[2, 2]).unwrap());
    let hc = TensorHandle::from_dense_auto(DenseND::from_vec(c.clone(), &[2, 2]).unwrap());

    let result = executor
        .einsum("ij,ij,ij->ij", &[ha, hb, hc], &ExecHints::default())
        .unwrap();
    let result = result.as_dense().unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let expected: Vec<f64> = (0..4).map(|i| a[i] * b[i] * c[i]).collect();
    for (got, want) in result.as_slice().iter().zip(&expected) {
        assert!((got - want).abs() < 1e-12, "got {got}, expected {want}");
    }
}

#[test]
fn test_three_operand_scalar_intermediate() {
    // "ij,ij,kl->kl": the first pair contracts to a scalar, which then scales
    // the third operand.  A step can legitimately produce a rank-0 intermediate.
    let mut executor = CpuExecutor::new();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let ha = TensorHandle::from_dense_auto(DenseND::from_vec(a.clone(), &[2, 2]).unwrap());
    let hb = TensorHandle::from_dense_auto(DenseND::from_vec(b, &[2, 2]).unwrap());
    let hc = TensorHandle::from_dense_auto(DenseND::from_vec(c.clone(), &[2, 3]).unwrap());

    let result = executor
        .einsum("ij,ij,kl->kl", &[ha, hb, hc], &ExecHints::default())
        .unwrap();
    let result = result.as_dense().unwrap();

    // Σ_ij A·B = 1+2+3+4 = 10, so the result is 10·C.
    assert_eq!(result.shape(), &[2, 3]);
    let expected: Vec<f64> = c.iter().map(|v| v * 10.0).collect();
    for (got, want) in result.as_slice().iter().zip(&expected) {
        let diff: f64 = got - want;
        assert!(diff.abs() < 1e-10, "got {got}, expected {want}");
    }
}
