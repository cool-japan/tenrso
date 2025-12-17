//! Tensor reduction operations
//!
//! This module provides comprehensive reduction operations for tensors, including
//! statistical measures, norms, and multivariate analysis along specified modes.
//!
//! # Operations
//!
//! ## Basic Statistics
//! - **Sum** - Sum along modes
//! - **Mean** - Average along modes
//! - **Variance/Std** - Statistical measures along modes (biased/unbiased)
//! - **Min/Max** - Extreme values along modes
//!
//! ## Distribution Analysis
//! - **Median** - Middle value (50th percentile)
//! - **Percentile** - Arbitrary percentile computation with interpolation
//! - **Skewness** - Distribution asymmetry (Fisher-Pearson coefficient)
//! - **Kurtosis** - Distribution tailedness (Pearson/excess kurtosis)
//!
//! ## Multivariate Statistics
//! - **Covariance** - Joint variability between two tensors
//! - **Correlation** - Pearson correlation coefficient (normalized covariance)
//!
//! ## Norms and Metrics
//! - **Frobenius Norm** - L2 (Euclidean) norm
//! - **P-Norms** - Generalized norms (L1, L2, L∞, arbitrary p)
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use crate::error::{KernelError, KernelResult};
use scirs2_core::ndarray_ext::{Array, ArrayView, Dimension, IxDyn};
use scirs2_core::numeric::{Float, Num, Zero};

/// Compute the sum of tensor elements along specified modes
///
/// Reduces the tensor by summing over the specified modes. This is a wrapper
/// around the contraction module's `sum_over_modes` with additional flexibility.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to sum over (0-indexed), can be empty to sum all elements
///
/// # Returns
///
/// Tensor with summed modes removed (or scalar if all modes summed)
///
/// # Errors
///
/// Returns error if mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::sum_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3, 4],
///     (0..24).map(|x| x as f64).collect()
/// ).unwrap();
///
/// // Sum over mode 1
/// let result = sum_along_modes(&tensor.view(), &[1]).unwrap();
/// assert_eq!(result.shape(), &[2, 4]);
///
/// // Sum all elements
/// let total = sum_along_modes(&tensor.view(), &[0, 1, 2]).unwrap();
/// assert_eq!(total.shape(), &[1]);
/// ```
pub fn sum_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Num + Zero,
{
    // Re-export from contractions module
    crate::sum_over_modes(tensor, modes)
}

/// Compute the mean (average) of tensor elements along specified modes
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to average over (0-indexed)
///
/// # Returns
///
/// Tensor with averaged modes removed
///
/// # Errors
///
/// Returns error if mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::mean_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ).unwrap();
///
/// // Mean over mode 1
/// let result = mean_along_modes(&tensor.view(), &[1]).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// assert!((result[[0]] - 2.0_f64).abs() < 1e-10); // (1+2+3)/3 = 2
/// assert!((result[[1]] - 5.0_f64).abs() < 1e-10); // (4+5+6)/3 = 5
/// ```
pub fn mean_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "mean_along_modes",
            ));
        }
    }

    // Compute sum
    let sum = sum_along_modes(tensor, modes)?;

    // Compute number of elements being averaged
    let count: usize = modes.iter().map(|&m| shape[m]).product();
    let count_t = T::from(count).unwrap();

    // Divide by count
    let mean = sum.mapv(|x| x / count_t);

    Ok(mean)
}

/// Compute variance of tensor elements along specified modes
///
/// Computes the unbiased sample variance (using Bessel's correction, n-1 denominator).
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to compute variance over (0-indexed)
/// * `ddof` - Delta degrees of freedom (0 for population variance, 1 for sample variance)
///
/// # Returns
///
/// Tensor with variance computed along specified modes
///
/// # Errors
///
/// Returns error if:
/// - Mode indices are out of bounds
/// - Sample size ≤ ddof (would cause division by zero)
///
/// # Complexity
///
/// Time: O(2 × tensor_size) (two passes)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::variance_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ).unwrap();
///
/// // Variance over mode 1 (sample variance, ddof=1)
/// let result = variance_along_modes(&tensor.view(), &[1], 1).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// ```
pub fn variance_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
    ddof: usize,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "variance_along_modes",
            ));
        }
    }

    // Compute number of elements
    let count: usize = modes.iter().map(|&m| shape[m]).product();

    if count <= ddof {
        return Err(KernelError::operation_error(
            "variance_along_modes",
            format!("Sample size {} must be > ddof {}", count, ddof),
        ));
    }

    // Compute mean
    let mean = mean_along_modes(tensor, modes)?;

    // Compute squared deviations
    // Need to broadcast mean back to original shape
    let variance = compute_variance_from_mean(tensor, &mean.view(), modes, count, ddof)?;

    Ok(variance)
}

/// Helper function to compute variance given mean
fn compute_variance_from_mean<T>(
    tensor: &ArrayView<T, IxDyn>,
    mean: &ArrayView<T, IxDyn>,
    modes: &[usize],
    count: usize,
    ddof: usize,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    let shape = tensor.shape();

    // Determine output shape
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    let final_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    let mut variance = Array::<T, _>::zeros(IxDyn(&final_shape));

    // Iterate over all tensor elements and accumulate squared deviations
    for (idx, &val) in tensor.indexed_iter() {
        // Map tensor index to mean index (remove summed modes)
        let mut mean_idx = Vec::new();
        for (i, &dim_idx) in idx.as_array_view().iter().enumerate() {
            if !modes.contains(&i) {
                mean_idx.push(dim_idx);
            }
        }

        if mean_idx.is_empty() {
            mean_idx.push(0);
        }

        let m = mean[&mean_idx[..]];
        let deviation = val - m;
        variance[&mean_idx[..]] = variance[&mean_idx[..]] + deviation * deviation;
    }

    // Divide by (count - ddof)
    let divisor = T::from(count - ddof).unwrap();
    variance.mapv_inplace(|x| x / divisor);

    Ok(variance)
}

/// Compute standard deviation along specified modes
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to compute std dev over (0-indexed)
/// * `ddof` - Delta degrees of freedom (0 for population std, 1 for sample std)
///
/// # Returns
///
/// Tensor with standard deviation computed along specified modes
///
/// # Errors
///
/// Returns error if mode indices are out of bounds or sample size ≤ ddof
///
/// # Complexity
///
/// Time: O(2 × tensor_size)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::std_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ).unwrap();
///
/// let result = std_along_modes(&tensor.view(), &[1], 1).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// ```
pub fn std_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
    ddof: usize,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    let variance = variance_along_modes(tensor, modes, ddof)?;
    Ok(variance.mapv(|x| x.sqrt()))
}

/// Compute Frobenius norm (L2 norm) of a tensor
///
/// Computes √(∑ᵢ xᵢ²), the square root of the sum of squared elements.
///
/// # Arguments
///
/// * `tensor` - Input tensor
///
/// # Returns
///
/// Scalar Frobenius norm
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{array, Array};
/// use tenrso_kernels::frobenius_norm_tensor;
///
/// let tensor = array![[3.0, 4.0], [0.0, 0.0]];
/// let tensor_dyn = tensor.into_dyn();
///
/// let norm = frobenius_norm_tensor(&tensor_dyn.view());
/// assert!((norm - 5.0_f64).abs() < 1e-10); // √(9 + 16) = 5
/// ```
pub fn frobenius_norm_tensor<T>(tensor: &ArrayView<T, IxDyn>) -> T
where
    T: Clone + Float,
{
    let sum_sq = tensor
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x);
    sum_sq.sqrt()
}

/// Compute p-norm along specified modes
///
/// Computes (∑ᵢ |xᵢ|ᵖ)^(1/p) along the specified modes.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to compute norm over
/// * `p` - Norm order (1 for L1, 2 for L2/Euclidean, etc.)
///
/// # Returns
///
/// Tensor with p-norm computed along specified modes
///
/// # Errors
///
/// Returns error if mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::pnorm_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3],
///     vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]
/// ).unwrap();
///
/// // L1 norm (sum of absolute values) along mode 1
/// let result = pnorm_along_modes(&tensor.view(), &[1], 1.0).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// assert!((result[[0]] - 6.0_f64).abs() < 1e-10); // |1| + |2| + |3| = 6
/// assert!((result[[1]] - 6.0_f64).abs() < 1e-10); // |-1| + |-2| + |-3| = 6
/// ```
pub fn pnorm_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
    p: T,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "pnorm_along_modes",
            ));
        }
    }

    // Determine output shape
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    let final_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    let mut result = Array::<T, _>::zeros(IxDyn(&final_shape));

    // Compute |x|^p for each element and accumulate
    for (idx, &val) in tensor.indexed_iter() {
        let mut out_idx = Vec::new();
        for (i, &dim_idx) in idx.as_array_view().iter().enumerate() {
            if !modes.contains(&i) {
                out_idx.push(dim_idx);
            }
        }

        if out_idx.is_empty() {
            out_idx.push(0);
        }

        let abs_val = val.abs();
        let powered = abs_val.powf(p);
        result[&out_idx[..]] = result[&out_idx[..]] + powered;
    }

    // Take p-th root
    let inv_p = T::one() / p;
    result.mapv_inplace(|x| x.powf(inv_p));

    Ok(result)
}

/// Find minimum value along specified modes
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to find minimum over
///
/// # Returns
///
/// Tensor containing minimum values with specified modes removed
///
/// # Errors
///
/// Returns error if mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::min_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3],
///     vec![3.0, 1.0, 4.0, 2.0, 5.0, 0.0]
/// ).unwrap();
///
/// let result = min_along_modes(&tensor.view(), &[1]).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// assert_eq!(result[[0]], 1.0); // min(3, 1, 4)
/// assert_eq!(result[[1]], 0.0); // min(2, 5, 0)
/// ```
pub fn min_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    reduce_with_op(tensor, modes, |a, b| if a < b { a } else { b })
}

/// Find maximum value along specified modes
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to find maximum over
///
/// # Returns
///
/// Tensor containing maximum values with specified modes removed
///
/// # Errors
///
/// Returns error if mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::max_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3],
///     vec![3.0, 1.0, 4.0, 2.0, 5.0, 0.0]
/// ).unwrap();
///
/// let result = max_along_modes(&tensor.view(), &[1]).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// assert_eq!(result[[0]], 4.0); // max(3, 1, 4)
/// assert_eq!(result[[1]], 5.0); // max(2, 5, 0)
/// ```
pub fn max_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    reduce_with_op(tensor, modes, |a, b| if a > b { a } else { b })
}

/// Compute median of tensor elements along specified modes
///
/// The median is the middle value when elements are sorted. For even-sized samples,
/// the median is the average of the two middle values.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to compute median over (0-indexed)
///
/// # Returns
///
/// Tensor with median computed along specified modes
///
/// # Errors
///
/// Returns error if mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(n log n) per output element (due to sorting)
/// Space: O(sample_size) temporary buffer
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::median_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 5],
///     vec![5.0, 1.0, 3.0, 2.0, 4.0,  // median = 3.0
///          10.0, 20.0, 15.0, 25.0, 30.0]  // median = 20.0
/// ).unwrap();
///
/// let result = median_along_modes(&tensor.view(), &[1]).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// assert!((result[[0]] - 3.0_f64).abs() < 1e-10);
/// assert!((result[[1]] - 20.0_f64).abs() < 1e-10);
/// ```
pub fn median_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float + PartialOrd,
{
    percentile_along_modes(tensor, modes, 50.0)
}

/// Compute percentile of tensor elements along specified modes
///
/// The p-th percentile is the value below which p% of the data falls.
/// Uses linear interpolation between sorted values.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to compute percentile over (0-indexed)
/// * `percentile` - Percentile to compute (0.0 to 100.0)
///
/// # Returns
///
/// Tensor with percentile computed along specified modes
///
/// # Errors
///
/// Returns error if:
/// - Mode indices are out of bounds
/// - Percentile is outside [0, 100]
///
/// # Complexity
///
/// Time: O(n log n) per output element (due to sorting)
/// Space: O(sample_size) temporary buffer
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::percentile_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 5],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0,
///          10.0, 20.0, 30.0, 40.0, 50.0]
/// ).unwrap();
///
/// // 25th percentile
/// let result = percentile_along_modes(&tensor.view(), &[1], 25.0).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// assert!((result[[0]] - 2.0_f64).abs() < 1e-10);
/// assert!((result[[1]] - 20.0_f64).abs() < 1e-10);
/// ```
pub fn percentile_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
    percentile: f64,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float + PartialOrd,
{
    if !(0.0..=100.0).contains(&percentile) {
        return Err(KernelError::operation_error(
            "percentile_along_modes",
            format!("Percentile must be in [0, 100], got {}", percentile),
        ));
    }

    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "percentile_along_modes",
            ));
        }
    }

    // Determine output shape
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    let final_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    // Initialize result array
    let mut result = Array::zeros(IxDyn(&final_shape));

    // Collect values for each output element
    let mut values_buffer = Vec::new();

    for (out_idx, result_val) in result.indexed_iter_mut() {
        values_buffer.clear();

        // Gather all values that contribute to this output element
        for (idx, &val) in tensor.indexed_iter() {
            // Check if this element maps to current output element
            let mut out_coords = Vec::new();
            for (i, _size) in shape.iter().enumerate() {
                if !modes.contains(&i) {
                    out_coords.push(idx[i]);
                }
            }

            // Match output coordinates
            let matches = if out_coords.is_empty() {
                true
            } else {
                out_coords
                    .iter()
                    .zip(out_idx.as_array_view().iter())
                    .all(|(a, b)| a == b)
            };

            if matches {
                values_buffer.push(val);
            }
        }

        // Sort values
        values_buffer.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute percentile with linear interpolation
        let n = values_buffer.len();
        if n == 0 {
            return Err(KernelError::operation_error(
                "percentile_along_modes",
                "No values to compute percentile from",
            ));
        }

        if n == 1 {
            *result_val = values_buffer[0];
        } else {
            // Linear interpolation: index = (n-1) * (percentile/100)
            let index = (n - 1) as f64 * (percentile / 100.0);
            let lower_idx = index.floor() as usize;
            let upper_idx = index.ceil() as usize;
            let fraction = T::from(index - index.floor()).unwrap();

            let lower_val = values_buffer[lower_idx];
            let upper_val = values_buffer[upper_idx];

            *result_val = lower_val + fraction * (upper_val - lower_val);
        }
    }

    Ok(result)
}

/// Compute skewness of tensor elements along specified modes
///
/// Skewness measures the asymmetry of the distribution. Positive skewness indicates
/// a distribution with a longer right tail, negative skewness indicates a longer left tail.
///
/// Uses the Fisher-Pearson coefficient of skewness:
/// g₁ = E[(X - μ)³] / σ³
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to compute skewness over (0-indexed)
/// * `bias` - If true, use biased estimator (divide by n); if false, use unbiased (divide by n-2)
///
/// # Returns
///
/// Tensor with skewness computed along specified modes
///
/// # Errors
///
/// Returns error if:
/// - Mode indices are out of bounds
/// - Sample size too small for unbiased estimate (n < 3)
///
/// # Complexity
///
/// Time: O(3 × tensor_size) (three passes: mean, variance, third moment)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::skewness_along_modes;
///
/// // Symmetric distribution has zero skewness
/// let tensor = Array::from_shape_vec(
///     vec![1, 5],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0]
/// ).unwrap();
///
/// let result = skewness_along_modes(&tensor.view(), &[1], true).unwrap();
/// assert!((result[[0]] as f64).abs() < 1e-6); // Near zero for symmetric data
/// ```
pub fn skewness_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
    bias: bool,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "skewness_along_modes",
            ));
        }
    }

    let n: usize = modes.iter().map(|&m| shape[m]).product();

    if !bias && n < 3 {
        return Err(KernelError::operation_error(
            "skewness_along_modes",
            "Sample size must be at least 3 for unbiased skewness",
        ));
    }

    // Compute mean
    let mean = mean_along_modes(tensor, modes)?;

    // Determine output shape
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    let final_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    // Compute third central moment and variance
    let mut third_moment = Array::<T, _>::zeros(IxDyn(&final_shape));
    let mut second_moment = Array::<T, _>::zeros(IxDyn(&final_shape));

    for (idx, &val) in tensor.indexed_iter() {
        // Get corresponding output index
        let out_coords: Vec<usize> = (0..shape.len())
            .filter(|m| !modes.contains(m))
            .map(|m| idx[m])
            .collect();

        let out_idx = if out_coords.is_empty() {
            vec![0]
        } else {
            out_coords
        };

        let mean_val = mean[IxDyn(&out_idx)];
        let diff = val - mean_val;

        third_moment[IxDyn(&out_idx)] = third_moment[IxDyn(&out_idx)] + diff * diff * diff;
        second_moment[IxDyn(&out_idx)] = second_moment[IxDyn(&out_idx)] + diff * diff;
    }

    // Compute skewness
    let n_t = T::from(n).unwrap();
    let one_point_five = T::from(1.5).unwrap();
    let result = if bias {
        // Biased estimator: g₁ = m₃ / m₂^(3/2)
        third_moment.mapv(|m3| m3 / n_t)
            / second_moment.mapv(|m2: T| (m2 / n_t).powf(one_point_five))
    } else {
        // Unbiased estimator: G₁ = √(n(n-1)) / (n-2) × g₁
        let adj_factor = T::from((n * (n - 1)) as f64).unwrap().sqrt() / T::from(n - 2).unwrap();
        (third_moment.mapv(|m3| m3 / n_t)
            / second_moment.mapv(|m2: T| (m2 / n_t).powf(one_point_five)))
        .mapv(|g1| g1 * adj_factor)
    };

    Ok(result)
}

/// Compute kurtosis of tensor elements along specified modes
///
/// Kurtosis measures the "tailedness" of the distribution. Higher kurtosis indicates
/// heavier tails and more outliers.
///
/// Uses the Pearson kurtosis:
/// g₂ = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to compute kurtosis over (0-indexed)
/// * `fisher` - If true, compute excess kurtosis (subtract 3); if false, raw kurtosis
/// * `bias` - If true, use biased estimator; if false, use unbiased
///
/// # Returns
///
/// Tensor with kurtosis computed along specified modes
///
/// # Errors
///
/// Returns error if:
/// - Mode indices are out of bounds
/// - Sample size too small for unbiased estimate (n < 4)
///
/// # Complexity
///
/// Time: O(3 × tensor_size) (three passes: mean, variance, fourth moment)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::kurtosis_along_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![1, 5],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0]
/// ).unwrap();
///
/// let result = kurtosis_along_modes(&tensor.view(), &[1], true, true).unwrap();
/// // Uniform-ish distribution has negative excess kurtosis
/// assert!(result[[0]] < 0.0);
/// ```
pub fn kurtosis_along_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
    fisher: bool,
    bias: bool,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
{
    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "kurtosis_along_modes",
            ));
        }
    }

    let n: usize = modes.iter().map(|&m| shape[m]).product();

    if !bias && n < 4 {
        return Err(KernelError::operation_error(
            "kurtosis_along_modes",
            "Sample size must be at least 4 for unbiased kurtosis",
        ));
    }

    // Compute mean
    let mean = mean_along_modes(tensor, modes)?;

    // Determine output shape
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    let final_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    // Compute fourth central moment and variance
    let mut fourth_moment = Array::<T, _>::zeros(IxDyn(&final_shape));
    let mut second_moment = Array::<T, _>::zeros(IxDyn(&final_shape));

    for (idx, &val) in tensor.indexed_iter() {
        // Get corresponding output index
        let out_coords: Vec<usize> = (0..shape.len())
            .filter(|m| !modes.contains(m))
            .map(|m| idx[m])
            .collect();

        let out_idx = if out_coords.is_empty() {
            vec![0]
        } else {
            out_coords
        };

        let mean_val = mean[IxDyn(&out_idx)];
        let diff = val - mean_val;
        let diff2 = diff * diff;

        fourth_moment[IxDyn(&out_idx)] = fourth_moment[IxDyn(&out_idx)] + diff2 * diff2;
        second_moment[IxDyn(&out_idx)] = second_moment[IxDyn(&out_idx)] + diff2;
    }

    // Compute kurtosis
    let n_t = T::from(n).unwrap();
    let three = T::from(3.0).unwrap();

    let kurt = if bias {
        // Biased estimator: g₂ = m₄ / m₂²
        fourth_moment.mapv(|m4| m4 / n_t)
            / second_moment.mapv(|m2| {
                let var = m2 / n_t;
                var * var
            })
    } else {
        // Unbiased estimator with correction factor
        let adj1 = T::from((n + 1) * n * (n - 1)).unwrap();
        let adj2 = T::from((n - 2) * (n - 3)).unwrap();
        let adj3 = T::from(3 * (n - 1) * (n - 1)).unwrap();
        let adj4 = T::from((n - 2) * (n - 3)).unwrap();

        (fourth_moment.mapv(|m4| m4 / n_t)
            / second_moment.mapv(|m2| {
                let var = m2 / n_t;
                var * var
            }))
        .mapv(|g2| (adj1 / adj2) * g2 - (adj3 / adj4))
    };

    // Apply Fisher correction if requested (excess kurtosis)
    let result = if fisher {
        kurt.mapv(|k| k - three)
    } else {
        kurt
    };

    Ok(result)
}

/// Helper function to reduce tensor with a binary operation
fn reduce_with_op<T, F>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
    op: F,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Float,
    F: Fn(T, T) -> T,
{
    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "reduce_with_op",
            ));
        }
    }

    // Determine output shape
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    let final_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    // Initialize with first element's values (or infinity for min/max)
    let mut result = Array::from_elem(IxDyn(&final_shape), T::infinity());
    let mut initialized = vec![false; result.len()];

    // Iterate and reduce
    for (idx, &val) in tensor.indexed_iter() {
        let mut out_idx = Vec::new();
        for (i, &dim_idx) in idx.as_array_view().iter().enumerate() {
            if !modes.contains(&i) {
                out_idx.push(dim_idx);
            }
        }

        if out_idx.is_empty() {
            out_idx.push(0);
        }

        let linear_idx = ravel_index(&out_idx, &final_shape);

        if !initialized[linear_idx] {
            result[&out_idx[..]] = val;
            initialized[linear_idx] = true;
        } else {
            let current = result[&out_idx[..]];
            result[&out_idx[..]] = op(current, val);
        }
    }

    Ok(result)
}

/// Convert multi-dimensional index to linear index
fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0;
    let mut stride = 1;

    for i in (0..indices.len()).rev() {
        linear += indices[i] * stride;
        stride *= shape[i];
    }

    linear
}

/// Compute covariance between two tensors along specified modes
///
/// Computes the covariance:
/// ```text
/// cov(X, Y) = E[(X - E[X])(Y - E[Y])]
/// ```
///
/// where E[\] denotes expectation (mean) along the specified modes.
///
/// # Arguments
///
/// * `tensor_x` - First input tensor
/// * `tensor_y` - Second input tensor (must have same shape as `tensor_x`)
/// * `modes` - Modes to compute covariance over (0-indexed)
/// * `ddof` - Degrees of freedom correction (0 for population, 1 for sample)
///
/// # Returns
///
/// Tensor with covariance values, with specified modes reduced
///
/// # Errors
///
/// * `DimensionMismatch` - If tensors have different shapes
/// * `InvalidMode` - If mode indices are out of bounds
/// * `InvalidParameter` - If ddof >= sample size
///
/// # Complexity
///
/// Time: O(3 × tensor_size) - three passes (mean_x, mean_y, covariance)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::covariance_along_modes;
///
/// let x = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let y = Array::from_shape_vec(vec![2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();
///
/// // Compute covariance along mode 1
/// let cov = covariance_along_modes(&x.view(), &y.view(), &[1], 1).unwrap();
/// assert_eq!(cov.shape(), &[2]);
/// // For perfectly correlated data: cov should be positive
/// assert!(cov[[0]] > 0.0);
/// assert!(cov[[1]] > 0.0);
/// ```
pub fn covariance_along_modes<T>(
    tensor_x: &ArrayView<T, IxDyn>,
    tensor_y: &ArrayView<T, IxDyn>,
    modes: &[usize],
    ddof: usize,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Num + Zero + Float,
{
    // Validate shapes match
    if tensor_x.shape() != tensor_y.shape() {
        return Err(KernelError::dimension_mismatch(
            "covariance_along_modes",
            tensor_x.shape().to_vec(),
            tensor_y.shape().to_vec(),
            "Tensors must have the same shape",
        ));
    }

    // Validate modes
    let ndim = tensor_x.ndim();
    for &mode in modes {
        if mode >= ndim {
            return Err(KernelError::invalid_mode(
                mode,
                ndim,
                "covariance_along_modes",
            ));
        }
    }

    // Compute sample size along reduction modes
    let mut sample_size = 1;
    for &mode in modes {
        sample_size *= tensor_x.shape()[mode];
    }

    if ddof >= sample_size {
        return Err(KernelError::operation_error(
            "covariance_along_modes",
            format!(
                "ddof ({}) must be less than sample size ({})",
                ddof, sample_size
            ),
        ));
    }

    // Compute means
    let mean_x = mean_along_modes(tensor_x, modes)?;
    let mean_y = mean_along_modes(tensor_y, modes)?;

    // Compute covariance: E[(X - mean_x)(Y - mean_y)]
    let shape = tensor_x.shape();
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    let final_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    let mut result = Array::zeros(IxDyn(&final_shape));

    // Accumulate (x - mean_x) * (y - mean_y)
    for (idx, &val_x) in tensor_x.indexed_iter() {
        let mut out_idx = Vec::new();
        for (i, &dim_idx) in idx.as_array_view().iter().enumerate() {
            if !modes.contains(&i) {
                out_idx.push(dim_idx);
            }
        }

        if out_idx.is_empty() {
            out_idx.push(0);
        }

        let val_y = tensor_y[idx.clone()];
        let mean_x_val = mean_x[&out_idx[..]];
        let mean_y_val = mean_y[&out_idx[..]];

        let dev_x = val_x - mean_x_val;
        let dev_y = val_y - mean_y_val;

        result[&out_idx[..]] = result[&out_idx[..]] + dev_x * dev_y;
    }

    // Divide by (n - ddof)
    let divisor = T::from(sample_size - ddof).unwrap();
    result.mapv_inplace(|x| x / divisor);

    Ok(result)
}

/// Compute Pearson correlation coefficient between two tensors along specified modes
///
/// Computes the correlation:
/// ```text
/// corr(X, Y) = cov(X, Y) / (std(X) * std(Y))
/// ```
///
/// where cov is covariance and std is standard deviation.
/// Correlation is normalized covariance ranging from -1 to 1.
///
/// # Arguments
///
/// * `tensor_x` - First input tensor
/// * `tensor_y` - Second input tensor (must have same shape as `tensor_x`)
/// * `modes` - Modes to compute correlation over (0-indexed)
///
/// # Returns
///
/// Tensor with correlation coefficients in \[-1, 1\], with specified modes reduced
///
/// # Errors
///
/// * `DimensionMismatch` - If tensors have different shapes
/// * `InvalidMode` - If mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(5 × tensor_size) - five passes (mean_x, mean_y, cov, std_x, std_y)
/// Space: O(output_size)
///
/// # Mathematical Properties
///
/// * -1 ≤ corr(X, Y) ≤ 1
/// * corr(X, Y) = 1 if Y = aX + b with a > 0 (perfect positive correlation)
/// * corr(X, Y) = -1 if Y = aX + b with a < 0 (perfect negative correlation)
/// * corr(X, Y) = 0 for uncorrelated variables
/// * corr(X, X) = 1 (perfect self-correlation)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::correlation_along_modes;
///
/// let x = Array::from_shape_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let y = Array::from_shape_vec(vec![2, 3], vec![2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();
///
/// // Compute correlation along mode 1 (y = 2*x, so perfect correlation)
/// let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
/// assert_eq!(corr.shape(), &[2]);
/// // Perfect positive correlation
/// assert!((corr[[0]] - 1.0_f64).abs() < 1e-10);
/// assert!((corr[[1]] - 1.0_f64).abs() < 1e-10);
/// ```
pub fn correlation_along_modes<T>(
    tensor_x: &ArrayView<T, IxDyn>,
    tensor_y: &ArrayView<T, IxDyn>,
    modes: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Num + Zero + Float,
{
    // Validate shapes match
    if tensor_x.shape() != tensor_y.shape() {
        return Err(KernelError::dimension_mismatch(
            "correlation_along_modes",
            tensor_x.shape().to_vec(),
            tensor_y.shape().to_vec(),
            "Tensors must have the same shape",
        ));
    }

    // Compute covariance (use ddof=0 for population, will normalize by std)
    let cov = covariance_along_modes(tensor_x, tensor_y, modes, 0)?;

    // Compute standard deviations
    let std_x = std_along_modes(tensor_x, modes, 0)?;
    let std_y = std_along_modes(tensor_y, modes, 0)?;

    // Compute correlation = cov / (std_x * std_y)
    let mut corr = cov;
    for (idx, val) in corr.indexed_iter_mut() {
        let sx = std_x[idx.clone()];
        let sy = std_y[idx.clone()];

        // Handle zero standard deviation case
        if sx.is_zero() || sy.is_zero() {
            *val = T::nan(); // Undefined correlation
        } else {
            *val = *val / (sx * sy);
        }
    }

    Ok(corr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::{array, Array};

    #[test]
    fn test_sum_along_modes() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = sum_along_modes(&tensor.view(), &[1]).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 6.0); // 1+2+3
        assert_eq!(result[[1]], 15.0); // 4+5+6
    }

    #[test]
    fn test_mean_along_modes() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = mean_along_modes(&tensor.view(), &[1]).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 2.0).abs() < 1e-10);
        assert!((result[[1]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_along_modes() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = variance_along_modes(&tensor.view(), &[1], 1).unwrap();
        assert_eq!(result.shape(), &[2]);

        // Variance of [1, 2, 3] = ((1-2)² + (2-2)² + (3-2)²) / 2 = 1.0
        assert!((result[[0]] - 1.0).abs() < 1e-10);
        assert!((result[[1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_along_modes() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = std_along_modes(&tensor.view(), &[1], 1).unwrap();
        assert_eq!(result.shape(), &[2]);

        // Std of [1, 2, 3] = √1.0 = 1.0
        assert!((result[[0]] - 1.0).abs() < 1e-10);
        assert!((result[[1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm() {
        let tensor = array![[3.0, 4.0], [0.0, 0.0]];
        let tensor_dyn = tensor.into_dyn();

        let norm = frobenius_norm_tensor(&tensor_dyn.view());
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pnorm_l1() {
        let tensor =
            Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]).unwrap();

        let result = pnorm_along_modes(&tensor.view(), &[1], 1.0).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 6.0).abs() < 1e-10);
        assert!((result[[1]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_pnorm_l2() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

        let result = pnorm_along_modes(&tensor.view(), &[1], 2.0).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 5.0).abs() < 1e-10); // √(9+16) = 5
        assert!((result[[1]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_along_modes() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![3.0, 1.0, 4.0, 2.0, 5.0, 0.0]).unwrap();

        let result = min_along_modes(&tensor.view(), &[1]).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 1.0);
        assert_eq!(result[[1]], 0.0);
    }

    #[test]
    fn test_max_along_modes() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![3.0, 1.0, 4.0, 2.0, 5.0, 0.0]).unwrap();

        let result = max_along_modes(&tensor.view(), &[1]).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 4.0);
        assert_eq!(result[[1]], 5.0);
    }

    #[test]
    fn test_mean_3d_tensor() {
        let tensor = Array::<f64, _>::ones(IxDyn(&[2, 3, 4]));

        let result = mean_along_modes(&tensor.view(), &[0, 2]).unwrap();
        assert_eq!(result.shape(), &[3]);

        // Mean of all 1.0s is 1.0
        for i in 0..3 {
            assert!((result[[i]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_variance_error_ddof_too_large() {
        let tensor = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Sample size is 2, ddof=2 should error
        let result = variance_along_modes(&tensor.view(), &[1], 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_mode_error() {
        let tensor = Array::<f64, _>::ones(IxDyn(&[2, 3]));

        let result = sum_along_modes(&tensor.view(), &[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_median_along_modes() {
        let tensor = Array::from_shape_vec(
            vec![2, 5],
            vec![
                5.0, 1.0, 3.0, 2.0, 4.0, // median = 3.0
                10.0, 20.0, 15.0, 25.0, 30.0, // median = 20.0
            ],
        )
        .unwrap();

        let result = median_along_modes(&tensor.view(), &[1]).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 3.0).abs() < 1e-10);
        assert!((result[[1]] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even_elements() {
        // Even number of elements: median = average of middle two
        let tensor = Array::from_shape_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = median_along_modes(&tensor.view(), &[1]).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert!((result[[0]] - 2.5).abs() < 1e-10); // (2+3)/2
    }

    #[test]
    fn test_percentile_along_modes() {
        let tensor = Array::from_shape_vec(
            vec![2, 5],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        )
        .unwrap();

        // 25th percentile
        let result = percentile_along_modes(&tensor.view(), &[1], 25.0).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 2.0).abs() < 1e-10);
        assert!((result[[1]] - 20.0).abs() < 1e-10);

        // 75th percentile
        let result = percentile_along_modes(&tensor.view(), &[1], 75.0).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 4.0).abs() < 1e-10);
        assert!((result[[1]] - 40.0).abs() < 1e-10);

        // 0th percentile (min)
        let result = percentile_along_modes(&tensor.view(), &[1], 0.0).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 1.0).abs() < 1e-10);
        assert!((result[[1]] - 10.0).abs() < 1e-10);

        // 100th percentile (max)
        let result = percentile_along_modes(&tensor.view(), &[1], 100.0).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result[[0]] - 5.0).abs() < 1e-10);
        assert!((result[[1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_invalid_range() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Percentile < 0
        let result = percentile_along_modes(&tensor.view(), &[1], -10.0);
        assert!(result.is_err());

        // Percentile > 100
        let result = percentile_along_modes(&tensor.view(), &[1], 150.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_skewness_symmetric_distribution() {
        // Symmetric distribution should have near-zero skewness
        let tensor = Array::from_shape_vec(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let result = skewness_along_modes(&tensor.view(), &[1], true).unwrap();
        assert_eq!(result.shape(), &[1]);
        // Perfectly symmetric, skewness should be very close to zero
        assert!(result[[0]].abs() < 1e-6);
    }

    #[test]
    fn test_skewness_right_skewed() {
        // Right-skewed distribution (positive skewness)
        let tensor = Array::from_shape_vec(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 100.0]).unwrap();

        let result = skewness_along_modes(&tensor.view(), &[1], true).unwrap();
        assert_eq!(result.shape(), &[1]);
        // Right-skewed data should have positive skewness
        assert!(result[[0]] > 0.0);
    }

    #[test]
    fn test_skewness_left_skewed() {
        // Left-skewed distribution (negative skewness)
        let tensor = Array::from_shape_vec(vec![1, 5], vec![-100.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let result = skewness_along_modes(&tensor.view(), &[1], true).unwrap();
        assert_eq!(result.shape(), &[1]);
        // Left-skewed data should have negative skewness
        assert!(result[[0]] < 0.0);
    }

    #[test]
    fn test_skewness_unbiased_sample_size_error() {
        // Sample size < 3 should error for unbiased estimator
        let tensor = Array::from_shape_vec(vec![1, 2], vec![1.0, 2.0]).unwrap();

        let result = skewness_along_modes(&tensor.view(), &[1], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_kurtosis_normal_like() {
        // Normal-like distribution should have kurtosis near 3 (excess kurtosis near 0)
        let tensor =
            Array::from_shape_vec(vec![1, 7], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();

        let result = kurtosis_along_modes(&tensor.view(), &[1], true, true).unwrap();
        assert_eq!(result.shape(), &[1]);
        // Uniform-like distribution has negative excess kurtosis
        assert!(result[[0]] < 0.0);
    }

    #[test]
    fn test_kurtosis_heavy_tails() {
        // Distribution with outliers should have high kurtosis
        let tensor =
            Array::from_shape_vec(vec![1, 7], vec![-100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0])
                .unwrap();

        let result = kurtosis_along_modes(&tensor.view(), &[1], true, true).unwrap();
        assert_eq!(result.shape(), &[1]);
        // Heavy tails should have positive excess kurtosis
        assert!(result[[0]] > 0.0);
    }

    #[test]
    fn test_kurtosis_fisher_vs_pearson() {
        let tensor = Array::from_shape_vec(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        // Fisher (excess) kurtosis
        let fisher = kurtosis_along_modes(&tensor.view(), &[1], true, true).unwrap();

        // Pearson (raw) kurtosis
        let pearson = kurtosis_along_modes(&tensor.view(), &[1], false, true).unwrap();

        // Pearson = Fisher + 3
        assert!((pearson[[0]] - (fisher[[0]] + 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_kurtosis_unbiased_sample_size_error() {
        // Sample size < 4 should error for unbiased estimator
        let tensor = Array::from_shape_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();

        let result = kurtosis_along_modes(&tensor.view(), &[1], true, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_median_3d_tensor() {
        let tensor = Array::from_shape_vec(
            vec![2, 2, 3],
            vec![
                1.0, 5.0, 3.0, // median = 3.0
                2.0, 4.0, 6.0, // median = 4.0
                10.0, 30.0, 20.0, // median = 20.0
                15.0, 25.0, 35.0, // median = 25.0
            ],
        )
        .unwrap();

        let result = median_along_modes(&tensor.view(), &[2]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result[[0, 0]] - 3.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 20.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_interpolation() {
        // Test linear interpolation for non-integer indices
        let tensor = Array::from_shape_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();

        // 50th percentile with 3 elements: index = (3-1) * 0.5 = 1.0
        let result = percentile_along_modes(&tensor.view(), &[1], 50.0).unwrap();
        assert!((result[[0]] - 2.0).abs() < 1e-10);

        // 25th percentile: index = (3-1) * 0.25 = 0.5 (interpolate between 1.0 and 2.0)
        let result = percentile_along_modes(&tensor.view(), &[1], 25.0).unwrap();
        assert!((result[[0]] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_skewness_multiple_modes() {
        let tensor =
            Array::<f64, _>::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect())
                .unwrap();

        let result = skewness_along_modes(&tensor.view(), &[1, 2], true).unwrap();
        assert_eq!(result.shape(), &[2]);
        // Should compute skewness for each of the 2 groups
    }

    #[test]
    fn test_kurtosis_constant_values() {
        // Constant values should have undefined (NaN or Inf) kurtosis due to zero variance
        let tensor = Array::from_shape_vec(vec![1, 5], vec![5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();

        let result = kurtosis_along_modes(&tensor.view(), &[1], true, true).unwrap();
        // Zero variance leads to division by zero, resulting in NaN or Inf
        assert!(result[[0]].is_nan() || result[[0]].is_infinite());
    }

    #[test]
    fn test_covariance_perfect_correlation() {
        // Perfect linear relationship: y = 2*x
        let x = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array::from_shape_vec(vec![2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();

        let cov = covariance_along_modes(&x.view(), &y.view(), &[1], 1).unwrap();
        assert_eq!(cov.shape(), &[2]);
        // Covariance should be positive for positively correlated data
        assert!(cov[[0]] > 0.0);
        assert!(cov[[1]] > 0.0);
    }

    #[test]
    fn test_covariance_negative_correlation() {
        // Negative correlation: y = -x
        let x = Array::from_shape_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array::from_shape_vec(vec![1, 4], vec![-1.0, -2.0, -3.0, -4.0]).unwrap();

        let cov = covariance_along_modes(&x.view(), &y.view(), &[1], 1).unwrap();
        assert_eq!(cov.shape(), &[1]);
        // Covariance should be negative
        assert!(cov[[0]] < 0.0);
    }

    #[test]
    fn test_covariance_self_is_variance() {
        // Covariance of a variable with itself equals its variance
        let x = Array::from_shape_vec(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let cov = covariance_along_modes(&x.view(), &x.view(), &[1], 1).unwrap();
        let var = variance_along_modes(&x.view(), &[1], 1).unwrap();

        assert_eq!(cov.shape(), var.shape());
        assert!((cov[[0]] - var[[0]]).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_dimension_mismatch() {
        let x = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array::from_shape_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();

        let result = covariance_along_modes(&x.view(), &y.view(), &[1], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_covariance_ddof_too_large() {
        let x = Array::from_shape_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array::from_shape_vec(vec![1, 3], vec![2.0, 4.0, 6.0]).unwrap();

        // ddof=3 with sample_size=3 should error
        let result = covariance_along_modes(&x.view(), &y.view(), &[1], 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_covariance_multiple_modes() {
        let x = Array::<f64, _>::from_shape_vec(vec![2, 3, 4], (0..24).map(|i| i as f64).collect())
            .unwrap();
        let y = Array::<f64, _>::from_shape_vec(
            vec![2, 3, 4],
            (0..24).map(|i| (i * 2) as f64).collect(),
        )
        .unwrap();

        let cov = covariance_along_modes(&x.view(), &y.view(), &[1, 2], 0).unwrap();
        assert_eq!(cov.shape(), &[2]);
        // Should compute covariance over 3×4=12 elements for each of 2 groups
    }

    #[test]
    fn test_correlation_perfect_positive() {
        // Perfect positive correlation: y = 2*x
        let x = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array::from_shape_vec(vec![2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
        assert_eq!(corr.shape(), &[2]);
        // Perfect correlation should be 1.0
        assert!((corr[[0]] - 1.0).abs() < 1e-10);
        assert!((corr[[1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        // Perfect negative correlation: y = -x
        let x = Array::from_shape_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array::from_shape_vec(vec![1, 4], vec![-1.0, -2.0, -3.0, -4.0]).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
        assert_eq!(corr.shape(), &[1]);
        // Perfect negative correlation should be -1.0
        assert!((corr[[0]] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_self_is_one() {
        // Correlation of a variable with itself should be 1.0
        let x = Array::from_shape_vec(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let corr = correlation_along_modes(&x.view(), &x.view(), &[1]).unwrap();
        assert_eq!(corr.shape(), &[1]);
        assert!((corr[[0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_uncorrelated() {
        // Orthogonal vectors should have correlation near zero
        let x = Array::from_shape_vec(vec![1, 4], vec![1.0, 0.0, -1.0, 0.0]).unwrap();
        let y = Array::from_shape_vec(vec![1, 4], vec![0.0, 1.0, 0.0, -1.0]).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
        assert_eq!(corr.shape(), &[1]);
        // Should be near zero (exactly zero for orthogonal)
        assert!(corr[[0]].abs() < 1e-10);
    }

    #[test]
    fn test_correlation_bounds() {
        // Correlation should always be in [-1, 1]
        let x = Array::from_shape_vec(vec![2, 5], (0..10).map(|i| i as f64).collect()).unwrap();
        let y =
            Array::from_shape_vec(vec![2, 5], (0..10).map(|i| (i * i) as f64).collect()).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
        assert_eq!(corr.shape(), &[2]);

        for val in corr.iter() {
            assert!(
                *val >= -1.0 && *val <= 1.0,
                "Correlation {} outside [-1, 1]",
                val
            );
        }
    }

    #[test]
    fn test_correlation_constant_array() {
        // Constant array has zero std, so correlation should be NaN
        let x = Array::from_shape_vec(vec![1, 4], vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        let y = Array::from_shape_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
        assert_eq!(corr.shape(), &[1]);
        assert!(corr[[0]].is_nan());
    }

    #[test]
    fn test_correlation_dimension_mismatch() {
        let x = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array::from_shape_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();

        let result = correlation_along_modes(&x.view(), &y.view(), &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_correlation_affine_relationship() {
        // y = 3*x + 2 should have perfect correlation
        let x = Array::from_shape_vec(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array::from_shape_vec(vec![1, 5], vec![5.0, 8.0, 11.0, 14.0, 17.0]).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
        assert_eq!(corr.shape(), &[1]);
        // Affine relationship should have perfect correlation
        assert!((corr[[0]] - 1.0).abs() < 1e-10);
    }
}
