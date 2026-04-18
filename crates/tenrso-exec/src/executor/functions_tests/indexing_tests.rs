//! Tests for indexing operations on `CpuExecutor`: `where_op`, `masked_select`,
//! `gather`, `scatter`, `advanced_gather`, `advanced_scatter`, `fancy_index_mask`.

#![allow(clippy::unnecessary_cast)]

use super::super::{
    functions::TenrsoExecutor,
    types::{CpuExecutor, ScatterMode},
};
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_where_op_basic() {
    let mut executor = CpuExecutor::new();
    let condition = DenseND::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
    let x = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let y = DenseND::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[2, 2]).unwrap();
    let cond_handle = TensorHandle::from_dense_auto(condition);
    let x_handle = TensorHandle::from_dense_auto(x);
    let y_handle = TensorHandle::from_dense_auto(y);
    let result = executor
        .where_op(&cond_handle, &x_handle, &y_handle)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 10.0);
    assert_eq!(result_view[[0, 1]], 200.0);
    assert_eq!(result_view[[1, 0]], 30.0);
    assert_eq!(result_view[[1, 1]], 400.0);
}

#[test]
fn test_where_op_shape_mismatch() {
    let mut executor = CpuExecutor::new();
    let condition = DenseND::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let x = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let y = DenseND::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[2, 2]).unwrap();
    let cond_handle = TensorHandle::from_dense_auto(condition);
    let x_handle = TensorHandle::from_dense_auto(x);
    let y_handle = TensorHandle::from_dense_auto(y);
    let result = executor.where_op(&cond_handle, &x_handle, &y_handle);
    assert!(result.is_err());
}

#[test]
fn test_masked_select_basic() {
    let mut executor = CpuExecutor::new();
    let data = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();
    let mask = DenseND::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();
    let data_handle = TensorHandle::from_dense_auto(data);
    let mask_handle = TensorHandle::from_dense_auto(mask);
    let result = executor.masked_select(&data_handle, &mask_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 10.0);
    assert_eq!(result_view[[1]], 30.0);
    assert_eq!(result_view[[2]], 50.0);
}

#[test]
fn test_masked_select_no_match() {
    let mut executor = CpuExecutor::new();
    let data = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let mask = DenseND::from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();
    let data_handle = TensorHandle::from_dense_auto(data);
    let mask_handle = TensorHandle::from_dense_auto(mask);
    let result = executor.masked_select(&data_handle, &mask_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[0]);
}

#[test]
fn test_gather_basic() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    )
    .unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let indices = DenseND::from_vec(vec![0.0, 2.0], &[2]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);
    let result = executor.gather(&input_handle, 0, &indices_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 1.0);
    assert_eq!(result_view[[0, 1]], 2.0);
    assert_eq!(result_view[[0, 2]], 3.0);
    assert_eq!(result_view[[1, 0]], 7.0);
    assert_eq!(result_view[[1, 1]], 8.0);
    assert_eq!(result_view[[1, 2]], 9.0);
}

#[test]
fn test_gather_out_of_bounds() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let indices = DenseND::from_vec(vec![0.0, 5.0], &[2]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);
    let result = executor.gather(&input_handle, 0, &indices_handle);
    assert!(result.is_err());
}

#[test]
fn test_gather_duplicates() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3, 1]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let indices = DenseND::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);
    let result = executor.gather(&input_handle, 0, &indices_handle).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3, 1]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 20.0);
    assert_eq!(result_view[[1, 0]], 20.0);
    assert_eq!(result_view[[2, 0]], 20.0);
}

#[test]
fn test_scatter_basic() {
    let mut executor = CpuExecutor::new();
    let indices = DenseND::from_vec(vec![1.0, 3.0], &[2]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);
    let values = DenseND::from_vec(vec![10.0, 11.0, 20.0, 21.0], &[2, 2]).unwrap();
    let values_handle = TensorHandle::from_dense_auto(values);
    let result = executor
        .scatter(&[5, 2], 0, &indices_handle, &values_handle)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[5, 2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 0.0);
    assert_eq!(result_view[[0, 1]], 0.0);
    assert_eq!(result_view[[1, 0]], 10.0);
    assert_eq!(result_view[[1, 1]], 11.0);
    assert_eq!(result_view[[2, 0]], 0.0);
    assert_eq!(result_view[[2, 1]], 0.0);
    assert_eq!(result_view[[3, 0]], 20.0);
    assert_eq!(result_view[[3, 1]], 21.0);
    assert_eq!(result_view[[4, 0]], 0.0);
    assert_eq!(result_view[[4, 1]], 0.0);
}

#[test]
fn test_scatter_out_of_bounds() {
    let mut executor = CpuExecutor::new();
    let indices = DenseND::from_vec(vec![0.0, 10.0], &[2]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);
    let values = DenseND::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let values_handle = TensorHandle::from_dense_auto(values);
    let result = executor.scatter(&[5, 1], 0, &indices_handle, &values_handle);
    assert!(result.is_err());
}

#[test]
fn test_scatter_shape_mismatch() {
    let mut executor = CpuExecutor::new();
    let indices = DenseND::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);
    let values = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let values_handle = TensorHandle::from_dense_auto(values);
    let result = executor.scatter(&[5, 2], 0, &indices_handle, &values_handle);
    assert!(result.is_err());
}

#[test]
fn test_advanced_gather_basic() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let indices = DenseND::from_vec(vec![0.0, 2.0, 4.0, 1.0], &[4]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);

    let result = executor
        .advanced_gather(&input_handle, 0, &indices_handle, false)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[4]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 10.0);
    assert_eq!(result_view[[1]], 30.0);
    assert_eq!(result_view[[2]], 50.0);
    assert_eq!(result_view[[3]], 20.0);
}

#[test]
fn test_advanced_gather_negative_indices() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let indices = DenseND::from_vec(vec![-1.0, -2.0], &[2]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);

    let result = executor
        .advanced_gather(&input_handle, 0, &indices_handle, true)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 50.0); // -1 -> index 4
    assert_eq!(result_view[[1]], 40.0); // -2 -> index 3
}

#[test]
fn test_advanced_scatter_replace() {
    let mut executor = CpuExecutor::new();
    let shape = vec![5];
    let indices = DenseND::from_vec(vec![0.0, 2.0, 4.0], &[3]).unwrap();
    let indices_handle = TensorHandle::from_dense_auto(indices);
    let values = DenseND::from_vec(vec![10.0, 30.0, 50.0], &[3]).unwrap();
    let values_handle = TensorHandle::from_dense_auto(values);

    let result = executor
        .advanced_scatter(
            &shape,
            0,
            &indices_handle,
            &values_handle,
            ScatterMode::Replace,
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[5]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 10.0);
    assert_eq!(result_view[[1]], 0.0);
    assert_eq!(result_view[[2]], 30.0);
    assert_eq!(result_view[[3]], 0.0);
    assert_eq!(result_view[[4]], 50.0);
}

#[test]
fn test_fancy_index_mask_basic() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let mask = DenseND::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0], &[5]).unwrap();
    let mask_handle = TensorHandle::from_dense_auto(mask);

    let result = executor
        .fancy_index_mask(&input_handle, &mask_handle)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 10.0);
    assert_eq!(result_view[[1]], 30.0);
    assert_eq!(result_view[[2]], 50.0);
}

#[test]
fn test_fancy_index_mask_all_false() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let mask = DenseND::from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();
    let mask_handle = TensorHandle::from_dense_auto(mask);

    let result = executor
        .fancy_index_mask(&input_handle, &mask_handle)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[0]);
}
