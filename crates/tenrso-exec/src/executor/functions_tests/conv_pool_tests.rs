//! Tests for convolution and pooling operations on `CpuExecutor`:
//! `max_pool_1d/2d`, `avg_pool_1d/2d`, `conv1d`, `conv2d`, `conv3d`.

#![allow(clippy::unnecessary_cast)]

use super::super::{functions::TenrsoExecutor, types::CpuExecutor};
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_max_pool_1d_basic() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0];
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[6]).unwrap());
    let result = executor.max_pool_1d(&handle, 2, 2).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 3.0);
    assert_eq!(result_view[[1]], 4.0);
    assert_eq!(result_view[[2]], 5.0);
}

#[test]
fn test_max_pool_1d_overlapping() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = vec![1.0, 5.0, 2.0, 8.0, 3.0];
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[5]).unwrap());
    let result = executor.max_pool_1d(&handle, 3, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 5.0);
    assert_eq!(result_view[[1]], 8.0);
    assert_eq!(result_view[[2]], 8.0);
}

#[test]
fn test_avg_pool_1d_basic() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 6.0, 2.0];
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[6]).unwrap());
    let result = executor.avg_pool_1d(&handle, 2, 2).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 2.0);
    assert_eq!(result_view[[1]], 3.0);
    assert_eq!(result_view[[2]], 4.0);
}

#[test]
fn test_max_pool_2d_basic() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[4, 4]).unwrap());
    let result = executor.max_pool_2d(&handle, (2, 2), (2, 2)).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 6.0);
    assert_eq!(result_view[[0, 1]], 8.0);
    assert_eq!(result_view[[1, 0]], 14.0);
    assert_eq!(result_view[[1, 1]], 16.0);
}

#[test]
fn test_avg_pool_2d_basic() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[4, 4]).unwrap());
    let result = executor.avg_pool_2d(&handle, (2, 2), (2, 2)).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 3.5);
    assert_eq!(result_view[[0, 1]], 5.5);
    assert_eq!(result_view[[1, 0]], 11.5);
    assert_eq!(result_view[[1, 1]], 13.5);
}

#[test]
fn test_max_pool_2d_non_square() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = (1..=18).map(|x| x as f64).collect();
    let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[3, 6]).unwrap());
    let result = executor.max_pool_2d(&handle, (3, 2), (3, 2)).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 3]);
}

#[test]
fn test_pooling_invalid_params() {
    let mut executor = CpuExecutor::new();
    let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let handle = TensorHandle::from_dense_auto(data);
    assert!(executor.max_pool_1d(&handle, 0, 1).is_err());
    assert!(executor.avg_pool_1d(&handle, 2, 0).is_err());
    assert!(executor.max_pool_1d(&handle, 10, 1).is_err());
}

#[test]
fn test_conv1d_basic() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0, 0.0, -1.0], &[1, 1, 3]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv1d(&input_handle, &kernel_handle, None, 1, (0, 0))
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 3]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0, 0]] - -2.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 1]] - -2.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 2]] - -2.0_f64).abs() < 1e-10);
}

#[test]
fn test_conv1d_with_bias() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0, 1.0], &[1, 1, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let bias = DenseND::from_vec(vec![10.0], &[1]).unwrap();
    let bias_handle = TensorHandle::from_dense_auto(bias);
    let result = executor
        .conv1d(&input_handle, &kernel_handle, Some(&bias_handle), 1, (0, 0))
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 3]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0, 0]] - 13.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 1]] - 15.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 2]] - 17.0_f64).abs() < 1e-10);
}

#[test]
fn test_conv1d_with_padding() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0, 1.0, 1.0], &[1, 1, 3]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv1d(&input_handle, &kernel_handle, None, 1, (1, 1))
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 3]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0, 0]] - 3.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 1]] - 6.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 2]] - 5.0_f64).abs() < 1e-10);
}

#[test]
fn test_conv1d_multi_channel() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv1d(&input_handle, &kernel_handle, None, 1, (0, 0))
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 2]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0, 0]] - 6.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 1]] - 8.0_f64).abs() < 1e-10);
}

#[test]
fn test_conv2d_basic() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    )
    .unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![0.25, 0.25, 0.25, 0.25], &[1, 1, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv2d(&input_handle, &kernel_handle, None, (1, 1), (0, 0, 0, 0))
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 2, 2]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0, 0, 0]] - 3.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 0, 1]] - 4.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 1, 0]] - 6.0_f64).abs() < 1e-10);
    assert!((result_view[[0, 0, 1, 1]] - 7.0_f64).abs() < 1e-10);
}

#[test]
fn test_conv2d_with_bias() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let bias = DenseND::from_vec(vec![100.0], &[1]).unwrap();
    let bias_handle = TensorHandle::from_dense_auto(bias);
    let result = executor
        .conv2d(
            &input_handle,
            &kernel_handle,
            Some(&bias_handle),
            (1, 1),
            (0, 0, 0, 0),
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 1, 1]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0, 0, 0]] - 105.0_f64).abs() < 1e-10);
}

#[test]
fn test_conv2d_stride() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec((0..16).map(|x| x as f64).collect(), &[1, 1, 4, 4]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv2d(&input_handle, &kernel_handle, None, (2, 2), (0, 0, 0, 0))
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 2, 2]);
}

#[test]
fn test_conv2d_invalid_channels() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec((0..18).map(|x| x as f64).collect(), &[1, 2, 3, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor.conv2d(&input_handle, &kernel_handle, None, (1, 1), (0, 0, 0, 0));
    assert!(result.is_err());
}

#[test]
fn test_conv3d_basic() {
    let mut executor = CpuExecutor::new();
    let input_data: Vec<f64> = (1..=27).map(|x| x as f64).collect();
    let input = DenseND::from_vec(input_data, &[1, 1, 3, 3, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel_data = vec![1.0; 8];
    let kernel = DenseND::from_vec(kernel_data, &[1, 1, 2, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv3d(
            &input_handle,
            &kernel_handle,
            None,
            (1, 1, 1),
            (0, 0, 0, 0, 0, 0),
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 2, 2, 2]);
    let result_view = result_dense.view();
    assert!(result_view[[0, 0, 0, 0, 0]] > 0.0);
}

#[test]
fn test_conv3d_with_bias() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let bias = DenseND::from_vec(vec![100.0], &[1]).unwrap();
    let bias_handle = TensorHandle::from_dense_auto(bias);
    let result = executor
        .conv3d(
            &input_handle,
            &kernel_handle,
            Some(&bias_handle),
            (1, 1, 1),
            (0, 0, 0, 0, 0, 0),
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 1, 1, 1]);
    let result_view = result_dense.view();
    assert!((result_view[[0, 0, 0, 0, 0]] - 108.0_f64).abs() < 1e-10);
}

#[test]
fn test_conv3d_with_stride() {
    let mut executor = CpuExecutor::new();
    let input_data: Vec<f64> = (0..64).map(|x| x as f64).collect();
    let input = DenseND::from_vec(input_data, &[1, 1, 4, 4, 4]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv3d(
            &input_handle,
            &kernel_handle,
            None,
            (2, 2, 2),
            (0, 0, 0, 0, 0, 0),
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 2, 2, 2]);
}

#[test]
fn test_conv3d_with_padding() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor
        .conv3d(
            &input_handle,
            &kernel_handle,
            None,
            (1, 1, 1),
            (1, 1, 1, 1, 1, 1),
        )
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 1, 3, 3, 3]);
}

#[test]
fn test_conv3d_invalid_dimensions() {
    let mut executor = CpuExecutor::new();
    let input_data: Vec<f64> = (0..54).map(|x| x as f64).collect();
    let input = DenseND::from_vec(input_data, &[1, 2, 3, 3, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);
    let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
    let kernel_handle = TensorHandle::from_dense_auto(kernel);
    let result = executor.conv3d(
        &input_handle,
        &kernel_handle,
        None,
        (1, 1, 1),
        (0, 0, 0, 0, 0, 0),
    );
    assert!(result.is_err());
}
