//! Tests for reduction operations on `CpuExecutor`: `reduce`, `softmax`,
//! `log_softmax`, `layer_norm`, `batch_norm`, `argmax`, `argmin`.

#![allow(clippy::unnecessary_cast)]

use super::super::{
    functions::TenrsoExecutor,
    types::{CpuExecutor, ElemOp, ReduceOp},
};
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_reduce_sum_single_axis() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Sum, &handle_a, &[0]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 5.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 7.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 9.0_f64).abs() < 1e-10);
}

#[test]
fn test_reduce_sum_axis_1() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Sum, &handle_a, &[1]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2]);
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 6.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 15.0_f64).abs() < 1e-10);
}

#[test]
fn test_reduce_max() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Max, &handle_a, &[0]).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 2.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 8.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 4.0_f64).abs() < 1e-10);
}

#[test]
fn test_reduce_min() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Min, &handle_a, &[0]).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 5.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 3.0_f64).abs() < 1e-10);
}

#[test]
fn test_reduce_mean() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Mean, &handle_a, &[0]).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 2.5_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 3.5_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 4.5_f64).abs() < 1e-10);
}

#[test]
fn test_reduce_multiple_axes() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Sum, &handle_a, &[0, 2]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert!(result_view[[0]] > 0.0);
}

#[test]
fn test_reduce_invalid_axis() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Sum, &handle_a, &[2]);
    assert!(result.is_err());
}

#[test]
fn test_reduce_no_axes() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reduce(ReduceOp::Sum, &handle_a, &[]);
    assert!(result.is_err());
}

#[test]
fn test_softmax_axis0() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.softmax(&handle_a, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    let col0_sum: f64 = result_view[[0, 0]] + result_view[[1, 0]];
    let col1_sum: f64 = result_view[[0, 1]] + result_view[[1, 1]];
    let col2_sum: f64 = result_view[[0, 2]] + result_view[[1, 2]];
    assert!((col0_sum - 1.0).abs() < 1e-10);
    assert!((col1_sum - 1.0).abs() < 1e-10);
    assert!((col2_sum - 1.0).abs() < 1e-10);
    for i in 0..2 {
        for j in 0..3 {
            assert!(result_view[[i, j]] > 0.0);
            assert!(result_view[[i, j]] < 1.0);
        }
    }
}

#[test]
fn test_softmax_axis1() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.softmax(&handle_a, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    let row0_sum: f64 = result_view[[0, 0]] + result_view[[0, 1]] + result_view[[0, 2]];
    let row1_sum: f64 = result_view[[1, 0]] + result_view[[1, 1]] + result_view[[1, 2]];
    assert!((row0_sum - 1.0).abs() < 1e-10);
    assert!((row1_sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_softmax_numerical_stability() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1000.0, 1001.0, 1002.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.softmax(&handle_a, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    let sum: f64 = result_view[[0]] + result_view[[1]] + result_view[[2]];
    assert!((sum - 1.0).abs() < 1e-10);
    assert!((result_view[[0]] as f64).is_finite());
    assert!((result_view[[1]] as f64).is_finite());
    assert!((result_view[[2]] as f64).is_finite());
}

#[test]
fn test_softmax_invalid_axis() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.softmax(&handle_a, 2);
    assert!(result.is_err());
}

#[test]
fn test_log_softmax_axis1() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.log_softmax(&handle_a, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    let row0_sum = (result_view[[0, 0]] as f64).exp()
        + (result_view[[0, 1]] as f64).exp()
        + (result_view[[0, 2]] as f64).exp();
    let row1_sum = (result_view[[1, 0]] as f64).exp()
        + (result_view[[1, 1]] as f64).exp()
        + (result_view[[1, 2]] as f64).exp();
    assert!((row0_sum - 1.0).abs() < 1e-10);
    assert!((row1_sum - 1.0).abs() < 1e-10);
    for i in 0..2 {
        for j in 0..3 {
            assert!(result_view[[i, j]] < 0.0 || (result_view[[i, j]] as f64).abs() < 1e-10);
        }
    }
}

#[test]
fn test_log_softmax_numerical_stability() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1000.0, 1001.0, 1002.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.log_softmax(&handle_a, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] as f64).is_finite());
    assert!((result_view[[1]] as f64).is_finite());
    assert!((result_view[[2]] as f64).is_finite());
    let sum: f64 = (result_view[[0]] as f64).exp()
        + (result_view[[1]] as f64).exp()
        + (result_view[[2]] as f64).exp();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_log_softmax_equivalence() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let softmax_result = executor.softmax(&handle_a, 1).unwrap();
    let log_of_softmax = executor.elem_op(ElemOp::Log, &softmax_result).unwrap();
    let los_dense = log_of_softmax.as_dense().unwrap();
    let log_softmax_result = executor.log_softmax(&handle_a, 1).unwrap();
    let ls_dense = log_softmax_result.as_dense().unwrap();
    let los_view = los_dense.view();
    let ls_view = ls_dense.view();
    for i in 0..2 {
        for j in 0..2 {
            let diff: f64 = los_view[[i, j]] - ls_view[[i, j]];
            assert!(diff.abs() < 1e-9);
        }
    }
}

#[test]
fn test_layer_norm_2d() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let eps = 1e-5;
    let result = executor.layer_norm(&handle_a, eps).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 3]);
    let result_view = result_dense.view();
    let row0_mean: f64 = (result_view[[0, 0]] + result_view[[0, 1]] + result_view[[0, 2]]) / 3.0;
    assert!(row0_mean.abs() < 1e-6);
    let row1_mean: f64 = (result_view[[1, 0]] + result_view[[1, 1]] + result_view[[1, 2]]) / 3.0;
    assert!(row1_mean.abs() < 1e-6);
}

#[test]
fn test_batch_norm_2d() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let eps = 1e-5;
    let result = executor.batch_norm(&handle_a, eps).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[4, 2]);
    let result_view = result_dense.view();
    let col0_mean: f64 =
        (result_view[[0, 0]] + result_view[[1, 0]] + result_view[[2, 0]] + result_view[[3, 0]])
            / 4.0;
    assert!(col0_mean.abs() < 1e-6);
    let col1_mean: f64 =
        (result_view[[0, 1]] + result_view[[1, 1]] + result_view[[2, 1]] + result_view[[3, 1]])
            / 4.0;
    assert!(col1_mean.abs() < 1e-6);
}

#[test]
fn test_argmax_1d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.argmax(&input_handle, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape().len(), 0); // Scalar result
    let result_view = result_dense.view();
    assert_eq!(result_view[[]] as usize, 1); // Index of max value (5.0)
}

#[test]
fn test_argmax_2d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 3.0, 2.0, 2.0, 4.0, 1.0], &[2, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.argmax(&input_handle, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]] as usize, 1); // Max in first row [1.0, 3.0, 2.0] is at index 1
    assert_eq!(result_view[[1]] as usize, 1); // Max in second row [2.0, 4.0, 1.0] is at index 1
}

#[test]
fn test_argmin_1d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![5.0, 1.0, 3.0, 2.0, 4.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.argmin(&input_handle, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[]] as usize, 1); // Index of min value (1.0)
}

#[test]
fn test_argmin_2d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![3.0, 1.0, 2.0, 4.0, 2.0, 5.0], &[2, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.argmin(&input_handle, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]] as usize, 1); // Min in first row is at index 1
    assert_eq!(result_view[[1]] as usize, 1); // Min in second row is at index 1
}
