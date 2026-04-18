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
