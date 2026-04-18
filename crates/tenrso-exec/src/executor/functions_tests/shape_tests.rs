//! Tests for shape operations on `CpuExecutor`: `transpose`, `reshape`,
//! `concatenate`, `split`, `tile`, `pad`, `flip`, `squeeze`, `unsqueeze`,
//! `stack`, `repeat`, `roll`.

#![allow(clippy::unnecessary_cast)]

use super::super::{functions::TenrsoExecutor, types::CpuExecutor};
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_transpose_2d() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.transpose(&handle_a, &[1, 0]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3, 2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 1.0);
    assert_eq!(result_view[[0, 1]], 4.0);
    assert_eq!(result_view[[1, 0]], 2.0);
    assert_eq!(result_view[[1, 1]], 5.0);
    assert_eq!(result_view[[2, 0]], 3.0);
    assert_eq!(result_view[[2, 1]], 6.0);
}

#[test]
fn test_transpose_3d() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let a = DenseND::from_vec(data, &[2, 3, 4]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.transpose(&handle_a, &[2, 0, 1]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[4, 2, 3]);
}

#[test]
fn test_transpose_invalid_axes() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.transpose(&handle_a, &[0]);
    assert!(result.is_err());
    let result = executor.transpose(&handle_a, &[0, 0]);
    assert!(result.is_err());
    let result = executor.transpose(&handle_a, &[0, 2]);
    assert!(result.is_err());
}

#[test]
fn test_reshape_2d_to_1d() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reshape(&handle_a, &[6]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[6]);
    let result_view = result_dense.view();
    for (i, &expected) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
        assert_eq!(result_view[[i]], expected);
    }
}

#[test]
fn test_reshape_1d_to_3d() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let a = DenseND::from_vec(data, &[24]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reshape(&handle_a, &[2, 3, 4]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 3, 4]);
}

#[test]
fn test_reshape_invalid_size() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.reshape(&handle_a, &[3, 3]);
    assert!(result.is_err());
}

#[test]
fn test_concatenate_axis0() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor.concatenate(&[handle_a, handle_b], 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[4, 2]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 1.0);
    assert_eq!(result_view[[1, 1]], 4.0);
    assert_eq!(result_view[[2, 0]], 5.0);
    assert_eq!(result_view[[3, 1]], 8.0);
}

#[test]
fn test_concatenate_axis1() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let result = executor.concatenate(&[handle_a, handle_b], 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 4]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 1.0);
    assert_eq!(result_view[[0, 1]], 2.0);
    assert_eq!(result_view[[0, 2]], 5.0);
    assert_eq!(result_view[[0, 3]], 6.0);
}

#[test]
fn test_concatenate_multiple() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let b = DenseND::from_vec(vec![3.0, 4.0], &[2]).unwrap();
    let c = DenseND::from_vec(vec![5.0, 6.0], &[2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let handle_c = TensorHandle::from_dense_auto(c);
    let result = executor
        .concatenate(&[handle_a, handle_b, handle_c], 0)
        .unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[6]);
    let result_view = result_dense.view();
    for (i, &expected) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
        assert_eq!(result_view[[i]], expected);
    }
}

#[test]
fn test_split_axis0() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = DenseND::from_vec(data, &[4, 3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let results = executor.split(&handle_a, 2, 0).unwrap();
    assert_eq!(results.len(), 2);
    let result0 = results[0].as_dense().unwrap();
    let result1 = results[1].as_dense().unwrap();
    assert_eq!(result0.shape(), &[2, 3]);
    assert_eq!(result1.shape(), &[2, 3]);
    let view0 = result0.view();
    let view1 = result1.view();
    assert_eq!(view0[[0, 0]], 0.0);
    assert_eq!(view0[[1, 2]], 5.0);
    assert_eq!(view1[[0, 0]], 6.0);
    assert_eq!(view1[[1, 2]], 11.0);
}

#[test]
fn test_split_axis1() {
    let mut executor = CpuExecutor::new();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = DenseND::from_vec(data, &[3, 4]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let results = executor.split(&handle_a, 2, 1).unwrap();
    assert_eq!(results.len(), 2);
    let result0 = results[0].as_dense().unwrap();
    let result1 = results[1].as_dense().unwrap();
    assert_eq!(result0.shape(), &[3, 2]);
    assert_eq!(result1.shape(), &[3, 2]);
}

#[test]
fn test_split_invalid() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let result = executor.split(&handle_a, 2, 0);
    assert!(result.is_err());
}

#[test]
fn test_tile_1d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.tile(&input_handle, &[3]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[9]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 1.0);
    assert_eq!(result_view[[1]], 2.0);
    assert_eq!(result_view[[2]], 3.0);
    assert_eq!(result_view[[3]], 1.0);
    assert_eq!(result_view[[4]], 2.0);
    assert_eq!(result_view[[5]], 3.0);
}

#[test]
fn test_tile_2d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.tile(&input_handle, &[2, 3]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[4, 6]);
    let result_view = result_dense.view();
    // Check first row of tiles
    assert_eq!(result_view[[0, 0]], 1.0);
    assert_eq!(result_view[[0, 1]], 2.0);
    assert_eq!(result_view[[0, 2]], 1.0);
    assert_eq!(result_view[[0, 3]], 2.0);
}

#[test]
fn test_tile_invalid_reps() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.tile(&input_handle, &[2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_pad_1d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.pad(&input_handle, &[(1, 2)], 0.0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[6]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 0.0);
    assert_eq!(result_view[[1]], 1.0);
    assert_eq!(result_view[[2]], 2.0);
    assert_eq!(result_view[[3]], 3.0);
    assert_eq!(result_view[[4]], 0.0);
    assert_eq!(result_view[[5]], 0.0);
}

#[test]
fn test_pad_2d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.pad(&input_handle, &[(1, 1), (1, 1)], 0.0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[4, 4]);
    let result_view = result_dense.view();
    // Check corners are padded
    assert_eq!(result_view[[0, 0]], 0.0);
    assert_eq!(result_view[[3, 3]], 0.0);
    // Check original values
    assert_eq!(result_view[[1, 1]], 1.0);
    assert_eq!(result_view[[1, 2]], 2.0);
    assert_eq!(result_view[[2, 1]], 3.0);
    assert_eq!(result_view[[2, 2]], 4.0);
}

#[test]
fn test_pad_invalid_width() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.pad(&input_handle, &[(1, 1), (1, 1)], 0.0);
    assert!(result.is_err());
}

#[test]
fn test_flip_1d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.flip(&input_handle, &[0]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[5]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 5.0);
    assert_eq!(result_view[[1]], 4.0);
    assert_eq!(result_view[[2]], 3.0);
    assert_eq!(result_view[[3]], 2.0);
    assert_eq!(result_view[[4]], 1.0);
}

#[test]
fn test_flip_2d_horizontal() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.flip(&input_handle, &[1]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 3]);
    let result_view = result_dense.view();
    // First row: [1, 2, 3] -> [3, 2, 1]
    assert_eq!(result_view[[0, 0]], 3.0);
    assert_eq!(result_view[[0, 1]], 2.0);
    assert_eq!(result_view[[0, 2]], 1.0);
    // Second row: [4, 5, 6] -> [6, 5, 4]
    assert_eq!(result_view[[1, 0]], 6.0);
    assert_eq!(result_view[[1, 1]], 5.0);
    assert_eq!(result_view[[1, 2]], 4.0);
}

#[test]
fn test_flip_2d_vertical() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.flip(&input_handle, &[0]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 3]);
    let result_view = result_dense.view();
    // Rows are flipped
    assert_eq!(result_view[[0, 0]], 4.0);
    assert_eq!(result_view[[0, 1]], 5.0);
    assert_eq!(result_view[[0, 2]], 6.0);
    assert_eq!(result_view[[1, 0]], 1.0);
    assert_eq!(result_view[[1, 1]], 2.0);
    assert_eq!(result_view[[1, 2]], 3.0);
}

#[test]
fn test_flip_2d_both_axes() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.flip(&input_handle, &[0, 1]).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
    let result_view = result_dense.view();
    // Original: [[1, 2], [3, 4]]
    // Flipped:  [[4, 3], [2, 1]]
    assert_eq!(result_view[[0, 0]], 4.0);
    assert_eq!(result_view[[0, 1]], 3.0);
    assert_eq!(result_view[[1, 0]], 2.0);
    assert_eq!(result_view[[1, 1]], 1.0);
}

#[test]
fn test_flip_invalid_axis() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.flip(&input_handle, &[5]);
    assert!(result.is_err());
}

#[test]
fn test_squeeze_all() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.squeeze(&input_handle, None).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3]);
}

#[test]
fn test_squeeze_specific_axis() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.squeeze(&input_handle, Some(&[0])).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2]);
}

#[test]
fn test_squeeze_invalid_axis() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.squeeze(&input_handle, Some(&[0]));
    assert!(result.is_err());
}

#[test]
fn test_unsqueeze_front() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.unsqueeze(&input_handle, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[1, 3]);
}

#[test]
fn test_unsqueeze_end() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.unsqueeze(&input_handle, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[3, 1]);
}

#[test]
fn test_unsqueeze_invalid_axis() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.unsqueeze(&input_handle, 5);
    assert!(result.is_err());
}

#[test]
fn test_stack_1d() {
    let mut executor = CpuExecutor::new();
    let t1 = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let t2 = DenseND::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    let h1 = TensorHandle::from_dense_auto(t1);
    let h2 = TensorHandle::from_dense_auto(t2);

    let result = executor.stack(&[h1, h2], 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 3]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0, 0]], 1.0);
    assert_eq!(result_view[[1, 0]], 4.0);
}

#[test]
fn test_stack_2d() {
    let mut executor = CpuExecutor::new();
    let t1 = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let t2 = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let h1 = TensorHandle::from_dense_auto(t1);
    let h2 = TensorHandle::from_dense_auto(t2);

    let result = executor.stack(&[h1, h2], 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 2, 2]);
}

#[test]
fn test_stack_shape_mismatch() {
    let mut executor = CpuExecutor::new();
    let t1 = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let t2 = DenseND::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
    let h1 = TensorHandle::from_dense_auto(t1);
    let h2 = TensorHandle::from_dense_auto(t2);

    let result = executor.stack(&[h1, h2], 0);
    assert!(result.is_err());
}

#[test]
fn test_repeat_1d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.repeat(&input_handle, 2, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[6]);
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 1.0);
    assert_eq!(result_view[[1]], 1.0);
    assert_eq!(result_view[[2]], 2.0);
    assert_eq!(result_view[[3]], 2.0);
}

#[test]
fn test_repeat_2d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.repeat(&input_handle, 3, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.shape(), &[2, 6]);
}

#[test]
fn test_roll_positive() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.roll(&input_handle, 2, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 4.0);
    assert_eq!(result_view[[1]], 5.0);
    assert_eq!(result_view[[2]], 1.0);
    assert_eq!(result_view[[3]], 2.0);
    assert_eq!(result_view[[4]], 3.0);
}

#[test]
fn test_roll_negative() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.roll(&input_handle, -2, 0).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    assert_eq!(result_view[[0]], 3.0);
    assert_eq!(result_view[[1]], 4.0);
    assert_eq!(result_view[[2]], 5.0);
    assert_eq!(result_view[[3]], 1.0);
    assert_eq!(result_view[[4]], 2.0);
}

#[test]
fn test_roll_2d() {
    let mut executor = CpuExecutor::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let input_handle = TensorHandle::from_dense_auto(input);

    let result = executor.roll(&input_handle, 1, 1).unwrap();
    let result_dense = result.as_dense().unwrap();
    let result_view = result_dense.view();
    // First row [1, 2, 3] -> [3, 1, 2]
    assert_eq!(result_view[[0, 0]], 3.0);
    assert_eq!(result_view[[0, 1]], 1.0);
    assert_eq!(result_view[[0, 2]], 2.0);
}
