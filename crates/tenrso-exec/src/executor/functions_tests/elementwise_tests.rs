//! Tests for element-wise operations on `CpuExecutor`: `elem_op`, `binary_op`,
//! `clip`, `modulo`, and `remainder`.

#![allow(clippy::unnecessary_cast)]

use super::super::{
    functions::TenrsoExecutor,
    types::{BinaryOp, CpuExecutor, ElemOp},
};
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_elem_op_neg() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Neg, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0, 0]] - -1.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 1]] - 2.0_f64).abs() < 1e-10);
    assert!((result_view[[1, 0]] - -3.0_f64).abs() < 1e-10);
    assert!((result_view[[1, 1]] - 4.0_f64).abs() < 1e-10);
}

#[test]
fn test_elem_op_abs() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Abs, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0, 0]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 1]] - 2.0_f64).abs() < 1e-10);
    assert!((result_view[[1, 0]] - 3.0_f64).abs() < 1e-10);
    assert!((result_view[[1, 1]] - 4.0_f64).abs() < 1e-10);
}

#[test]
fn test_elem_op_exp() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Exp, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] as f64 - std::f64::consts::E as f64).abs() < 1e-10);
    assert!((result_view[[2]] - std::f64::consts::E.powi(2)).abs() < 1e-9);
}

#[test]
fn test_elem_op_log() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(
        vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)],
        &[3],
    )
    .unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Log, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 2.0_f64).abs() < 1e-9);
}

#[test]
fn test_elem_op_sin() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(
        vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI],
        &[3],
    )
    .unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Sin, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 0.0_f64).abs() < 1e-10);
}

#[test]
fn test_elem_op_cos() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(
        vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI],
        &[3],
    )
    .unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Cos, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - -1.0_f64).abs() < 1e-10);
}

#[test]
fn test_elem_op_sqrt() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0], &[5]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Sqrt, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 2.0_f64).abs() < 1e-10);
    assert!((result_view[[3]] - 3.0_f64).abs() < 1e-10);
    assert!((result_view[[4]] - 4.0_f64).abs() < 1e-10);
}

#[test]
fn test_elem_op_sqr() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Sqr, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 4.0_f64).abs() < 1e-10);
    assert!((result_view[[3]] - 9.0_f64).abs() < 1e-10);
    assert!((result_view[[4]] - 16.0_f64).abs() < 1e-10);
}

#[test]
fn test_elem_op_recip() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 4.0, 10.0], &[4]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Recip, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 0.5_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 0.25_f64).abs() < 1e-10);
    assert!((result_view[[3]] - 0.1_f64).abs() < 1e-10);
}

#[test]
fn test_elem_op_tanh() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Tanh, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[2]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[0]] + result_view[[4]]).abs() < 1e-10);
    assert!((result_view[[1]] + result_view[[3]]).abs() < 1e-10);
    for i in 0..5 {
        assert!((result_view[[i]] as f64).abs() < 1.0);
    }
}

#[test]
fn test_elem_op_sigmoid() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-10.0, 0.0, 10.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Sigmoid, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[1]] - 0.5_f64).abs() < 1e-10);
    assert!(result_view[[0]] < 0.001);
    assert!(result_view[[2]] > 0.999);
    for i in 0..3 {
        assert!(result_view[[i]] > 0.0 && result_view[[i]] < 1.0);
    }
}

#[test]
fn test_elem_op_relu() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::ReLU, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 0.0);
    assert_eq!(result_view[[1]], 0.0);
    assert_eq!(result_view[[2]], 0.0);
    assert_eq!(result_view[[3]], 1.0);
    assert_eq!(result_view[[4]], 2.0);
}

#[test]
fn test_elem_op_gelu() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Gelu, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[1]] as f64).abs() < 0.01);
    assert!(result_view[[2]] > 0.8);
    assert!((result_view[[0]] as f64).abs() < 0.2);
}

#[test]
fn test_elem_op_elu() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Elu, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[2]], 1.0);
    assert_eq!(result_view[[3]], 2.0);
    assert_eq!(result_view[[1]], 0.0);
    assert!((result_view[[0]] - (std::f64::consts::E.recip() - 1.0)).abs() < 1e-9);
}

#[test]
fn test_elem_op_selu() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Selu, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    let scale = 1.050_700_987_355_480_5;
    let alpha = 1.673_263_242_354_377_2;
    assert!((result_view[[2]] as f64 - scale as f64).abs() < 1e-9);
    assert_eq!(result_view[[1]], 0.0);
    let expected_neg = scale * alpha * ((-1.0_f64).exp() - 1.0);
    assert!((result_view[[0]] as f64 - expected_neg as f64).abs() < 1e-9);
}

#[test]
fn test_elem_op_softplus() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-10.0, 0.0, 10.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Softplus, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[1]] - 2.0_f64.ln()).abs() < 1e-9);
    assert!(result_view[[0]] < 0.001);
    assert!((result_view[[2]] - 10.0_f64).abs() < 0.001);
}

#[test]
fn test_elem_op_sign() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-2.5, -1.0, 0.0, 1.0, 3.5], &[5]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.elem_op(ElemOp::Sign, &handle_a).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], -1.0);
    assert_eq!(result_view[[1]], -1.0);
    assert_eq!(result_view[[2]], 0.0);
    assert_eq!(result_view[[3]], 1.0);
    assert_eq!(result_view[[4]], 1.0);
}

#[test]
fn test_binary_op_add() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Add, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 6.0);
    assert_eq!(result_view[[0, 1]], 8.0);
    assert_eq!(result_view[[1, 0]], 10.0);
    assert_eq!(result_view[[1, 1]], 12.0);
}

#[test]
fn test_binary_op_sub() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Sub, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 9.0);
    assert_eq!(result_view[[0, 1]], 18.0);
    assert_eq!(result_view[[1, 0]], 27.0);
    assert_eq!(result_view[[1, 1]], 36.0);
}

#[test]
fn test_binary_op_mul() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Mul, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 2.0);
    assert_eq!(result_view[[0, 1]], 6.0);
    assert_eq!(result_view[[1, 0]], 12.0);
    assert_eq!(result_view[[1, 1]], 20.0);
}

#[test]
fn test_binary_op_div() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![2.0, 4.0, 5.0, 8.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Div, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 5.0);
    assert_eq!(result_view[[0, 1]], 5.0);
    assert_eq!(result_view[[1, 0]], 6.0);
    assert_eq!(result_view[[1, 1]], 5.0);
}

#[test]
fn test_binary_op_pow() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![2.0, 2.0, 3.0, 2.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Pow, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 4.0);
    assert_eq!(result_view[[0, 1]], 9.0);
    assert_eq!(result_view[[1, 0]], 64.0);
    assert_eq!(result_view[[1, 1]], 25.0);
}

#[test]
fn test_binary_op_maximum() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 7.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![2.0, 4.0, 6.0, 1.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Maximum, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 2.0);
    assert_eq!(result_view[[0, 1]], 5.0);
    assert_eq!(result_view[[1, 0]], 6.0);
    assert_eq!(result_view[[1, 1]], 7.0);
}

#[test]
fn test_binary_op_minimum() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 7.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![2.0, 4.0, 6.0, 1.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Minimum, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 1.0);
    assert_eq!(result_view[[0, 1]], 4.0);
    assert_eq!(result_view[[1, 0]], 3.0);
    assert_eq!(result_view[[1, 1]], 1.0);
}

#[test]
fn test_binary_op_broadcast_scalar() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![5.0], &[1]).unwrap();
    let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Add, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 6.0);
    assert_eq!(result_view[[0, 1]], 7.0);
    assert_eq!(result_view[[1, 0]], 8.0);
    assert_eq!(result_view[[1, 1]], 9.0);
}

#[test]
fn test_binary_op_broadcast_scalar_mul() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![2.0], &[1]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Mul, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 2.0);
    assert_eq!(result_view[[0, 1]], 4.0);
    assert_eq!(result_view[[1, 0]], 6.0);
    assert_eq!(result_view[[1, 1]], 8.0);
}

#[test]
fn test_binary_op_general_broadcast() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let b = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor
        .binary_op(BinaryOp::Add, &handle_a, &handle_b)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 11.0);
    assert_eq!(result_view[[0, 1]], 22.0);
    assert_eq!(result_view[[0, 2]], 33.0);
    assert_eq!(result_view[[1, 0]], 41.0);
    assert_eq!(result_view[[1, 1]], 52.0);
    assert_eq!(result_view[[1, 2]], 63.0);
}

#[test]
fn test_clip_basic() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![-5.0, -1.0, 0.0, 3.0, 10.0], &[5]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.clip(&handle_a, -2.0, 5.0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], -2.0);
    assert_eq!(result_view[[1]], -1.0);
    assert_eq!(result_view[[2]], 0.0);
    assert_eq!(result_view[[3]], 3.0);
    assert_eq!(result_view[[4]], 5.0);
}

#[test]
fn test_clip_no_change() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.clip(&handle_a, 0.0, 5.0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 1.0);
    assert_eq!(result_view[[1]], 2.0);
    assert_eq!(result_view[[2]], 3.0);
}

#[test]
fn test_clip_invalid_bounds() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.clip(&handle_a, 5.0, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_modulo_basic() {
    let mut executor = CpuExecutor::new();
    let data = DenseND::from_vec(vec![10.0, 11.0, 12.0, 13.0], &[2, 2]).unwrap();
    let handle = TensorHandle::from_dense_auto(data);
    let result = executor.modulo(&handle, 3.0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 1]] - 2.0_f64).abs() < 1e-10);
    assert!((result_view[[1, 0]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[1, 1]] - 1.0_f64).abs() < 1e-10);
}

#[test]
fn test_modulo_zero_divisor() {
    let mut executor = CpuExecutor::new();
    let data = DenseND::from_vec(vec![10.0, 20.0], &[2]).unwrap();
    let handle = TensorHandle::from_dense_auto(data);
    let result = executor.modulo(&handle, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_remainder_basic() {
    let mut executor = CpuExecutor::new();
    let data = DenseND::from_vec(vec![7.0, 8.0, 9.0], &[3]).unwrap();
    let handle = TensorHandle::from_dense_auto(data);
    let result = executor.remainder(&handle, 4.0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 3.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 0.0_f64).abs() < 1e-10);
    assert!((result_view[[2]] - 1.0_f64).abs() < 1e-10);
}
