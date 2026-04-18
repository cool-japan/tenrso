//! Convolution and pooling helpers for the `TenrsoExecutor` implementation.
//!
//! Contains free functions supporting: `max_pool_1d`, `avg_pool_1d`, `max_pool_2d`,
//! `avg_pool_2d`, `conv1d`, `conv2d`, `conv3d`.

use super::super::types::CpuExecutor;
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{DenseND, TensorHandle};

pub(super) fn max_pool_1d<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    kernel_size: usize,
    stride: usize,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for max_pool_1d"))?;
    if kernel_size == 0 || stride == 0 {
        return Err(anyhow!("Kernel size and stride must be positive"));
    }
    let shape = dense.shape();
    if shape.len() != 1 {
        return Err(anyhow!(
            "Expected 1D tensor for max_pool_1d, got {:?}D",
            shape.len()
        ));
    }
    let input_len = shape[0];
    if kernel_size > input_len {
        return Err(anyhow!(
            "Kernel size {} larger than input length {}",
            kernel_size,
            input_len
        ));
    }
    let output_len = (input_len - kernel_size) / stride + 1;
    let mut output = Vec::with_capacity(output_len);
    let view = dense.view();
    for i in 0..output_len {
        let start = i * stride;
        let end = start + kernel_size;
        let max_val = (start..end)
            .map(|j| view[[j]])
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(T::default);
        output.push(max_val);
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&[output_len]), output)
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn avg_pool_1d<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    kernel_size: usize,
    stride: usize,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for avg_pool_1d"))?;
    if kernel_size == 0 || stride == 0 {
        return Err(anyhow!("Kernel size and stride must be positive"));
    }
    let shape = dense.shape();
    if shape.len() != 1 {
        return Err(anyhow!(
            "Expected 1D tensor for avg_pool_1d, got {:?}D",
            shape.len()
        ));
    }
    let input_len = shape[0];
    if kernel_size > input_len {
        return Err(anyhow!(
            "Kernel size {} larger than input length {}",
            kernel_size,
            input_len
        ));
    }
    let output_len = (input_len - kernel_size) / stride + 1;
    let mut output = Vec::with_capacity(output_len);
    let view = dense.view();
    let kernel_size_t = T::from_usize(kernel_size).unwrap_or_else(T::one);
    for i in 0..output_len {
        let start = i * stride;
        let end = start + kernel_size;
        let mut sum = T::zero();
        for j in start..end {
            sum += view[[j]];
        }
        let avg = sum / kernel_size_t;
        output.push(avg);
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&[output_len]), output)
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn max_pool_2d<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for max_pool_2d"))?;
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
        return Err(anyhow!("Kernel size and stride must be positive"));
    }
    let shape = dense.shape();
    if shape.len() != 2 {
        return Err(anyhow!(
            "Expected 2D tensor for max_pool_2d, got {:?}D",
            shape.len()
        ));
    }
    let (h, w) = (shape[0], shape[1]);
    if kh > h || kw > w {
        return Err(anyhow!(
            "Kernel size ({}, {}) larger than input ({}, {})",
            kh,
            kw,
            h,
            w
        ));
    }
    let out_h = (h - kh) / sh + 1;
    let out_w = (w - kw) / sw + 1;
    let output_shape = [out_h, out_w];

    // Use pooled buffer for max_pool_2d output (Phase 5: Automatic Pooling)
    let mut output = executor.acquire_pooled_generic::<T>(&output_shape);
    output.clear(); // Ensure buffer starts empty
    output.reserve(out_h * out_w);

    let view = dense.view();
    for i in 0..out_h {
        for j in 0..out_w {
            let start_h = i * sh;
            let start_w = j * sw;
            let mut max_val = T::default();
            let mut first = true;
            for di in 0..kh {
                for dj in 0..kw {
                    let val = view[[start_h + di, start_w + dj]];
                    if first || val > max_val {
                        max_val = val;
                        first = false;
                    }
                }
            }
            output.push(max_val);
        }
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output);
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn avg_pool_2d<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for avg_pool_2d"))?;
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
        return Err(anyhow!("Kernel size and stride must be positive"));
    }
    let shape = dense.shape();
    if shape.len() != 2 {
        return Err(anyhow!(
            "Expected 2D tensor for avg_pool_2d, got {:?}D",
            shape.len()
        ));
    }
    let (h, w) = (shape[0], shape[1]);
    if kh > h || kw > w {
        return Err(anyhow!(
            "Kernel size ({}, {}) larger than input ({}, {})",
            kh,
            kw,
            h,
            w
        ));
    }
    let out_h = (h - kh) / sh + 1;
    let out_w = (w - kw) / sw + 1;
    let output_shape = [out_h, out_w];

    // Use pooled buffer for avg_pool_2d output (Phase 5: Automatic Pooling)
    let mut output = executor.acquire_pooled_generic::<T>(&output_shape);
    output.clear(); // Ensure buffer starts empty
    output.reserve(out_h * out_w);

    let view = dense.view();
    let kernel_count = T::from_usize(kh * kw).unwrap_or_else(T::one);
    for i in 0..out_h {
        for j in 0..out_w {
            let start_h = i * sh;
            let start_w = j * sw;
            let mut sum = T::zero();
            for di in 0..kh {
                for dj in 0..kw {
                    sum += view[[start_h + di, start_w + dj]];
                }
            }
            let avg = sum / kernel_count;
            output.push(avg);
        }
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output);
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn conv1d<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    kernel: &TensorHandle<T>,
    bias: Option<&TensorHandle<T>>,
    stride: usize,
    padding: (usize, usize),
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense_x = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for conv1d"))?;
    let dense_kernel = kernel
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for conv1d kernel"))?;
    if stride == 0 {
        return Err(anyhow!("Stride must be positive"));
    }
    let x_shape = dense_x.shape();
    let k_shape = dense_kernel.shape();
    if x_shape.len() != 3 {
        return Err(anyhow!(
            "Expected 3D input tensor [batch, in_channels, length], got {:?}D",
            x_shape.len()
        ));
    }
    if k_shape.len() != 3 {
        return Err(anyhow!(
            "Expected 3D kernel tensor [out_channels, in_channels, kernel_size], got {:?}D",
            k_shape.len()
        ));
    }
    let (batch, in_channels, in_length) = (x_shape[0], x_shape[1], x_shape[2]);
    let (out_channels, k_in_channels, kernel_size) = (k_shape[0], k_shape[1], k_shape[2]);
    if in_channels != k_in_channels {
        return Err(anyhow!(
            "Input channels mismatch: input has {}, kernel expects {}",
            in_channels,
            k_in_channels
        ));
    }
    if let Some(bias_tensor) = bias {
        let bias_dense = bias_tensor
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for bias"))?;
        let bias_shape = bias_dense.shape();
        if bias_shape.len() != 1 || bias_shape[0] != out_channels {
            return Err(anyhow!(
                "Expected bias shape [{}], got {:?}",
                out_channels,
                bias_shape
            ));
        }
    }
    let (pad_left, pad_right) = padding;
    let padded_length = in_length + pad_left + pad_right;
    if kernel_size > padded_length {
        return Err(anyhow!(
            "Kernel size {} larger than padded input length {}",
            kernel_size,
            padded_length
        ));
    }
    let out_length = (padded_length - kernel_size) / stride + 1;
    let output_shape = [batch, out_channels, out_length];

    // Use pooled buffer for output allocation (Phase 5: Automatic Pooling)
    let mut output = executor.acquire_pooled_generic::<T>(&output_shape);
    output.clear(); // Ensure buffer starts empty
    output.resize(batch * out_channels * out_length, T::zero());

    let x_view = dense_x.view();
    let k_view = dense_kernel.view();
    for b in 0..batch {
        for oc in 0..out_channels {
            for o in 0..out_length {
                let mut sum = T::zero();
                let in_start = (o * stride) as isize - pad_left as isize;
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        let in_pos = in_start + k as isize;
                        if in_pos >= 0 && (in_pos as usize) < in_length {
                            let x_val = x_view[[b, ic, in_pos as usize]];
                            let k_val = k_view[[oc, ic, k]];
                            sum += x_val * k_val;
                        }
                    }
                }
                if let Some(bias_tensor) = bias {
                    let bias_dense = bias_tensor
                        .as_dense()
                        .ok_or_else(|| anyhow!("Bias must be dense tensor"))?;
                    let bias_view = bias_dense.view();
                    sum += bias_view[[oc]];
                }
                output[b * out_channels * out_length + oc * out_length + o] = sum;
            }
        }
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output);
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn conv2d<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    kernel: &TensorHandle<T>,
    bias: Option<&TensorHandle<T>>,
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense_x = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for conv2d"))?;
    let dense_kernel = kernel
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for conv2d kernel"))?;
    let (stride_h, stride_w) = stride;
    if stride_h == 0 || stride_w == 0 {
        return Err(anyhow!("Stride must be positive"));
    }
    let x_shape = dense_x.shape();
    let k_shape = dense_kernel.shape();
    if x_shape.len() != 4 {
        return Err(anyhow!(
            "Expected 4D input tensor [batch, in_channels, height, width], got {:?}D",
            x_shape.len()
        ));
    }
    if k_shape.len() != 4 {
        return Err(anyhow!(
            "Expected 4D kernel tensor [out_channels, in_channels, kernel_h, kernel_w], got {:?}D",
            k_shape.len()
        ));
    }
    let (batch, in_channels, in_h, in_w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (out_channels, k_in_channels, kernel_h, kernel_w) =
        (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    if in_channels != k_in_channels {
        return Err(anyhow!(
            "Input channels mismatch: input has {}, kernel expects {}",
            in_channels,
            k_in_channels
        ));
    }
    if let Some(bias_tensor) = bias {
        let bias_dense = bias_tensor
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for bias"))?;
        let bias_shape = bias_dense.shape();
        if bias_shape.len() != 1 || bias_shape[0] != out_channels {
            return Err(anyhow!(
                "Expected bias shape [{}], got {:?}",
                out_channels,
                bias_shape
            ));
        }
    }
    let (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right) = padding;
    let padded_h = in_h + pad_h_top + pad_h_bottom;
    let padded_w = in_w + pad_w_left + pad_w_right;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(anyhow!(
            "Kernel size ({}, {}) larger than padded input ({}, {})",
            kernel_h,
            kernel_w,
            padded_h,
            padded_w
        ));
    }
    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let output_shape = [batch, out_channels, out_h, out_w];

    // Use pooled buffer for output allocation (Phase 5: Automatic Pooling)
    let mut output = executor.acquire_pooled_generic::<T>(&output_shape);
    output.clear(); // Ensure buffer starts empty
    output.resize(batch * out_channels * out_h * out_w, T::zero());

    let x_view = dense_x.view();
    let k_view = dense_kernel.view();
    for b in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = T::zero();
                    let in_start_h = (oh * stride_h) as isize - pad_h_top as isize;
                    let in_start_w = (ow * stride_w) as isize - pad_w_left as isize;
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let in_h_pos = in_start_h + kh as isize;
                                let in_w_pos = in_start_w + kw as isize;
                                if in_h_pos >= 0
                                    && (in_h_pos as usize) < in_h
                                    && in_w_pos >= 0
                                    && (in_w_pos as usize) < in_w
                                {
                                    let x_val =
                                        x_view[[b, ic, in_h_pos as usize, in_w_pos as usize]];
                                    let k_val = k_view[[oc, ic, kh, kw]];
                                    sum += x_val * k_val;
                                }
                            }
                        }
                    }
                    if let Some(bias_tensor) = bias {
                        let bias_dense = bias_tensor
                            .as_dense()
                            .ok_or_else(|| anyhow!("Bias must be dense tensor"))?;
                        let bias_view = bias_dense.view();
                        sum += bias_view[[oc]];
                    }
                    let out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output);
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn conv3d<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    kernel: &TensorHandle<T>,
    bias: Option<&TensorHandle<T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize, usize, usize, usize),
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense_x = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for conv3d"))?;
    let dense_kernel = kernel
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for conv3d kernel"))?;
    let (stride_d, stride_h, stride_w) = stride;
    if stride_d == 0 || stride_h == 0 || stride_w == 0 {
        return Err(anyhow!("Stride must be positive"));
    }
    let x_shape = dense_x.shape();
    let k_shape = dense_kernel.shape();
    if x_shape.len() != 5 {
        return Err(anyhow!(
            "Expected 5D input tensor [batch, in_channels, depth, height, width], got {:?}D",
            x_shape.len()
        ));
    }
    if k_shape.len() != 5 {
        return Err(anyhow!(
            "Expected 5D kernel tensor [out_channels, in_channels, kernel_d, kernel_h, kernel_w], got {:?}D",
            k_shape.len()
        ));
    }
    let (batch, in_channels, in_d, in_h, in_w) =
        (x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]);
    let (out_channels, k_in_channels, kernel_d, kernel_h, kernel_w) =
        (k_shape[0], k_shape[1], k_shape[2], k_shape[3], k_shape[4]);
    if in_channels != k_in_channels {
        return Err(anyhow!(
            "Input channels mismatch: input has {}, kernel expects {}",
            in_channels,
            k_in_channels
        ));
    }
    if let Some(bias_tensor) = bias {
        let bias_dense = bias_tensor
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for bias"))?;
        let bias_shape = bias_dense.shape();
        if bias_shape.len() != 1 || bias_shape[0] != out_channels {
            return Err(anyhow!(
                "Expected bias shape [{}], got {:?}",
                out_channels,
                bias_shape
            ));
        }
    }
    let (pad_d_front, pad_d_back, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right) = padding;
    let padded_d = in_d + pad_d_front + pad_d_back;
    let padded_h = in_h + pad_h_top + pad_h_bottom;
    let padded_w = in_w + pad_w_left + pad_w_right;
    if kernel_d > padded_d || kernel_h > padded_h || kernel_w > padded_w {
        return Err(anyhow!(
            "Kernel size ({}, {}, {}) larger than padded input ({}, {}, {})",
            kernel_d,
            kernel_h,
            kernel_w,
            padded_d,
            padded_h,
            padded_w
        ));
    }
    let out_d = (padded_d - kernel_d) / stride_d + 1;
    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let output_shape = [batch, out_channels, out_d, out_h, out_w];

    // Use pooled buffer for output allocation (Phase 5: Automatic Pooling)
    let mut output = executor.acquire_pooled_generic::<T>(&output_shape);
    output.clear(); // Ensure buffer starts empty
    output.resize(batch * out_channels * out_d * out_h * out_w, T::zero());

    let x_view = dense_x.view();
    let k_view = dense_kernel.view();
    for b in 0..batch {
        for oc in 0..out_channels {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = T::zero();
                        let in_start_d = (od * stride_d) as isize - pad_d_front as isize;
                        let in_start_h = (oh * stride_h) as isize - pad_h_top as isize;
                        let in_start_w = (ow * stride_w) as isize - pad_w_left as isize;
                        for ic in 0..in_channels {
                            for kd in 0..kernel_d {
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let in_d_pos = in_start_d + kd as isize;
                                        let in_h_pos = in_start_h + kh as isize;
                                        let in_w_pos = in_start_w + kw as isize;
                                        if in_d_pos >= 0
                                            && (in_d_pos as usize) < in_d
                                            && in_h_pos >= 0
                                            && (in_h_pos as usize) < in_h
                                            && in_w_pos >= 0
                                            && (in_w_pos as usize) < in_w
                                        {
                                            let x_val = x_view[[
                                                b,
                                                ic,
                                                in_d_pos as usize,
                                                in_h_pos as usize,
                                                in_w_pos as usize,
                                            ]];
                                            let k_val = k_view[[oc, ic, kd, kh, kw]];
                                            sum += x_val * k_val;
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(bias_tensor) = bias {
                            let bias_dense = bias_tensor
                                .as_dense()
                                .ok_or_else(|| anyhow!("Bias must be dense tensor"))?;
                            let bias_view = bias_dense.view();
                            sum += bias_view[[oc]];
                        }
                        let out_idx =
                            ((((b * out_channels + oc) * out_d + od) * out_h + oh) * out_w) + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output);
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}
