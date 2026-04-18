//! Linear algebra helpers for the `TenrsoExecutor` implementation.
//!
//! Contains free functions supporting: `determinant`, `matrix_inverse`, `solve`.

use super::super::types::CpuExecutor;
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{DenseND, TensorHandle};

pub(super) fn determinant<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for determinant"))?;
    let shape = dense.shape();
    if shape.len() < 2 {
        return Err(anyhow!("Input must be at least 2D for determinant"));
    }
    let n = shape[shape.len() - 1];
    let m = shape[shape.len() - 2];
    if n != m {
        return Err(anyhow!(
            "Last two dimensions must be square for determinant, got {}x{}",
            m,
            n
        ));
    }
    if shape.len() == 2 {
        use scirs2_core::ndarray_ext::Array2;
        let view = dense.view();
        let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| view[[i, j]]);
        let det = executor.compute_determinant_2d(&matrix)?;
        return Ok(TensorHandle::from_dense_auto(DenseND::from_vec(
            vec![det],
            &[],
        )?));
    }
    let batch_size: usize = shape[..shape.len() - 2].iter().product();
    let mut determinants = Vec::with_capacity(batch_size);
    let view = dense.view();
    for batch_idx in 0..batch_size {
        let batch_multi = executor.flat_to_multidim(batch_idx, &shape[..shape.len() - 2]);
        use scirs2_core::ndarray_ext::Array2;
        let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| {
            let mut idx = batch_multi.clone();
            idx.push(i);
            idx.push(j);
            view[idx.as_slice()]
        });
        let det = executor.compute_determinant_2d(&matrix)?;
        determinants.push(det);
    }
    let output_shape = &shape[..shape.len() - 2];
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(output_shape), determinants)
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn matrix_inverse<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for matrix_inverse"))?;
    let shape = dense.shape();
    if shape.len() < 2 {
        return Err(anyhow!("Input must be at least 2D for matrix inverse"));
    }
    let n = shape[shape.len() - 1];
    let m = shape[shape.len() - 2];
    if n != m {
        return Err(anyhow!(
            "Last two dimensions must be square for matrix inverse, got {}x{}",
            m,
            n
        ));
    }
    if shape.len() == 2 {
        use scirs2_core::ndarray_ext::Array2;
        let view = dense.view();
        let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| view[[i, j]]);
        let inv = executor.compute_inverse_2d(&matrix)?;
        let inv_dyn = inv.into_dyn();
        return Ok(TensorHandle::from_dense_auto(DenseND::from_array(inv_dyn)));
    }
    let batch_size: usize = shape[..shape.len() - 2].iter().product();
    let output_size = batch_size * n * n;
    let mut output = Vec::with_capacity(output_size);
    let view = dense.view();
    for batch_idx in 0..batch_size {
        let batch_multi = executor.flat_to_multidim(batch_idx, &shape[..shape.len() - 2]);
        use scirs2_core::ndarray_ext::Array2;
        let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| {
            let mut idx = batch_multi.clone();
            idx.push(i);
            idx.push(j);
            view[idx.as_slice()]
        });
        let inv = executor.compute_inverse_2d(&matrix)?;
        output.extend(inv.iter().copied());
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(shape), output)
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn solve<T>(
    executor: &mut CpuExecutor,
    a: &TensorHandle<T>,
    b: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense_a = a
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for solve (A)"))?;
    let dense_b = b
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for solve (b)"))?;
    let a_shape = dense_a.shape();
    let b_shape = dense_b.shape();
    if a_shape.len() < 2 {
        return Err(anyhow!("Matrix A must be at least 2D"));
    }
    if b_shape.is_empty() {
        return Err(anyhow!("Vector/matrix b must be at least 1D"));
    }
    let n = a_shape[a_shape.len() - 1];
    let m = a_shape[a_shape.len() - 2];
    if n != m {
        return Err(anyhow!("Matrix A must be square, got {}x{}", m, n));
    }
    let b_rows = b_shape[b_shape.len()
        - (if b_shape.len() == a_shape.len() - 1 {
            1
        } else {
            2
        })];
    if b_rows != n {
        return Err(anyhow!(
            "Dimension mismatch: A is {}x{}, b has {} rows",
            m,
            n,
            b_rows
        ));
    }
    if a_shape.len() == 2 && b_shape.len() == 1 {
        use scirs2_core::ndarray_ext::{Array1, Array2};
        let a_view = dense_a.view();
        let b_view = dense_b.view();
        let a_matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| a_view[[i, j]]);
        let b_vector: Array1<T> = Array1::from_shape_fn(n, |i| b_view[[i]]);
        let x = executor.solve_2d_1d(&a_matrix, &b_vector)?;
        let x_dyn = x.into_dyn();
        return Ok(TensorHandle::from_dense_auto(DenseND::from_array(x_dyn)));
    }
    Err(anyhow!(
        "Solve only supports 2D matrix A with 1D vector b in this implementation"
    ))
}
