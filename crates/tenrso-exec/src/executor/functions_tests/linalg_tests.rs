//! Tests for linear-algebra operations on `CpuExecutor`: `determinant`,
//! `matrix_inverse`, `solve`.

#![allow(clippy::unnecessary_cast)]

use super::super::{functions::TenrsoExecutor, types::CpuExecutor};
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_determinant_2x2() {
    let mut executor = CpuExecutor::new();
    let matrix = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.determinant(&matrix_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape().len(), 0);
    let result_view = result_dense.view();
    assert!((result_view[[]] - (-2.0_f64)).abs() < 1e-10);
}

#[test]
fn test_determinant_3x3() {
    let mut executor = CpuExecutor::new();
    let matrix =
        DenseND::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0], &[3, 3]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.determinant(&matrix_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert!((result_view[[]] - 24.0_f64).abs() < 1e-10);
}

#[test]
fn test_determinant_singular() {
    let mut executor = CpuExecutor::new();
    let matrix = DenseND::from_vec(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.determinant(&matrix_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    let val: f64 = result_view[[]];
    assert!(val.abs() < 1e-10);
}

#[test]
fn test_determinant_batched() {
    let mut executor = CpuExecutor::new();
    let matrices =
        DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0], &[2, 2, 2]).unwrap();
    let matrices_handle = TensorHandle::from_dense_auto(matrices);
    let result = executor.determinant(&matrices_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2]);
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
    assert!((result_view[[1]] - 6.0_f64).abs() < 1e-10);
}

#[test]
fn test_determinant_invalid_shape() {
    let mut executor = CpuExecutor::new();
    let matrix = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.determinant(&matrix_handle);
    assert!(result.is_err());
}

#[test]
fn test_matrix_inverse_2x2() {
    let mut executor = CpuExecutor::new();
    let matrix = DenseND::from_vec(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.matrix_inverse(&matrix_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0]] - 0.6_f64).abs() < 1e-10);
    assert!((result_view[[0, 1]] - (-0.7_f64)).abs() < 1e-10);
    assert!((result_view[[1, 0]] - (-0.2_f64)).abs() < 1e-10);
    assert!((result_view[[1, 1]] - 0.4_f64).abs() < 1e-10);
}

#[test]
fn test_matrix_inverse_identity() {
    let mut executor = CpuExecutor::new();
    let matrix =
        DenseND::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.matrix_inverse(&matrix_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3, 3]);
    let result_view = result_dense.view();
    for i in 0..3 {
        for j in 0..3 {
            let expected: f64 = if i == j { 1.0 } else { 0.0 };
            assert!((result_view[[i, j]] - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn test_matrix_inverse_singular() {
    let mut executor = CpuExecutor::new();
    let matrix = DenseND::from_vec(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.matrix_inverse(&matrix_handle);
    assert!(result.is_err());
}

#[test]
fn test_matrix_inverse_batched() {
    let mut executor = CpuExecutor::new();
    let matrices =
        DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]).unwrap();
    let matrices_handle = TensorHandle::from_dense_auto(matrices);
    let result = executor.matrix_inverse(&matrices_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2, 2]);
    let result_view = result_dense.view();
    for b in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                assert!((result_view[[b, i, j]] - expected).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn test_matrix_inverse_non_square() {
    let mut executor = CpuExecutor::new();
    let matrix = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let matrix_handle = TensorHandle::from_dense_auto(matrix);
    let result = executor.matrix_inverse(&matrix_handle);
    assert!(result.is_err());
}

#[test]
fn test_solve_2x2_simple() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]).unwrap();
    let a_handle = TensorHandle::from_dense_auto(a);
    let b = DenseND::from_vec(vec![5.0, 5.0], &[2]).unwrap();
    let b_handle = TensorHandle::from_dense_auto(b);
    let result = executor.solve(&a_handle, &b_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2]);
    let result_view = result_dense.view();
    assert!((result_view[[0]] - 2.0_f64).abs() < 1e-9);
    assert!((result_view[[1]] - 1.0_f64).abs() < 1e-9);
}

#[test]
fn test_solve_3x3() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0], &[3, 3]).unwrap();
    let a_handle = TensorHandle::from_dense_auto(a);
    let b = DenseND::from_vec(vec![6.0, 9.0, 12.0], &[3]).unwrap();
    let b_handle = TensorHandle::from_dense_auto(b);
    let result = executor.solve(&a_handle, &b_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    for i in 0..3 {
        assert!((result_view[[i]] - 3.0_f64).abs() < 1e-9);
    }
}

#[test]
fn test_solve_singular() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
    let a_handle = TensorHandle::from_dense_auto(a);
    let b = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let b_handle = TensorHandle::from_dense_auto(b);
    let result = executor.solve(&a_handle, &b_handle);
    assert!(result.is_err());
}

#[test]
fn test_solve_dimension_mismatch() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let a_handle = TensorHandle::from_dense_auto(a);
    let b = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b_handle = TensorHandle::from_dense_auto(b);
    let result = executor.solve(&a_handle, &b_handle);
    assert!(result.is_err());
}

#[test]
fn test_solve_non_square_matrix() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let a_handle = TensorHandle::from_dense_auto(a);
    let b = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let b_handle = TensorHandle::from_dense_auto(b);
    let result = executor.solve(&a_handle, &b_handle);
    assert!(result.is_err());
}
