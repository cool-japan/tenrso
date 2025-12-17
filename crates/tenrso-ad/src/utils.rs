//! Utility functions for gradient computation and manipulation
//!
//! This module provides helper functions for common gradient operations,
//! diagnostics, and debugging.

use anyhow::Result;
use scirs2_core::ndarray_ext::Array;
use scirs2_core::numeric::{Num, NumCast};
use std::fmt;
use tenrso_core::DenseND;

/// Statistics about a gradient tensor
///
/// All statistics are computed in f64 for numerical stability
#[derive(Debug, Clone)]
pub struct GradientStats<T = f64> {
    /// Mean of gradient values
    pub mean: T,
    /// Standard deviation of gradient values
    pub std: T,
    /// Minimum gradient value
    pub min: T,
    /// Maximum gradient value
    pub max: T,
    /// Number of zero gradients
    pub num_zeros: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// L2 norm of gradient
    pub l2_norm: T,
    /// L1 norm of gradient
    pub l1_norm: T,
}

impl<T: fmt::Display> fmt::Display for GradientStats<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Gradient Statistics:")?;
        writeln!(f, "  Mean: {}", self.mean)?;
        writeln!(f, "  Std:  {}", self.std)?;
        writeln!(f, "  Min:  {}", self.min)?;
        writeln!(f, "  Max:  {}", self.max)?;
        writeln!(f, "  L1 norm:  {}", self.l1_norm)?;
        writeln!(f, "  L2 norm:  {}", self.l2_norm)?;
        writeln!(
            f,
            "  Zeros: {}/{} ({:.2}%)",
            self.num_zeros,
            self.total_elements,
            (self.num_zeros as f64 / self.total_elements as f64) * 100.0
        )?;
        Ok(())
    }
}

/// Compute statistics for a gradient tensor
///
/// # Arguments
///
/// * `grad` - The gradient tensor to analyze
///
/// # Returns
///
/// Statistics including mean, std, min, max, norms, and sparsity
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::compute_gradient_stats;
/// use tenrso_core::DenseND;
///
/// let grad = DenseND::from_vec(vec![0.1, 0.0, 0.5, -0.3], &[2, 2])?;
/// let stats = compute_gradient_stats(&grad)?;
/// println!("{}", stats);
/// ```
pub fn compute_gradient_stats<T>(grad: &DenseND<T>) -> Result<GradientStats<f64>>
where
    T: Num + Clone + NumCast + PartialOrd,
{
    let arr = grad.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(anyhow::anyhow!("Cannot compute stats for empty gradient"));
    }

    let n_f64 = n as f64;

    // Convert all values to f64 for computation
    let mut values_f64: Vec<f64> = Vec::with_capacity(n);
    for val in arr.iter() {
        values_f64.push(num_traits::cast::cast(val.clone()).unwrap_or(0.0));
    }

    // Compute mean
    let sum: f64 = values_f64.iter().sum();
    let mean = sum / n_f64;

    // Compute std, min, max
    let mut variance = 0.0;
    let mut min_val = values_f64[0];
    let mut max_val = values_f64[0];
    let mut num_zeros = 0;
    let mut l1_norm = 0.0;
    let mut l2_norm_sq = 0.0;

    for &val_f64 in &values_f64 {
        let diff = val_f64 - mean;
        variance += diff * diff;

        if val_f64 < min_val {
            min_val = val_f64;
        }
        if val_f64 > max_val {
            max_val = val_f64;
        }

        l1_norm += val_f64.abs();
        l2_norm_sq += val_f64 * val_f64;

        if val_f64.abs() < 1e-15 {
            num_zeros += 1;
        }
    }

    let std = if n > 1 {
        (variance / (n_f64 - 1.0)).sqrt()
    } else {
        0.0
    };

    let l2_norm = l2_norm_sq.sqrt();

    Ok(GradientStats {
        mean,
        std,
        min: min_val,
        max: max_val,
        num_zeros,
        total_elements: n,
        l2_norm,
        l1_norm,
    })
}

/// Clip gradient values to a specified range
///
/// Useful for preventing gradient explosion. Values outside [min, max]
/// are clipped to the boundary values.
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::clip_gradient;
///
/// let grad = DenseND::from_vec(vec![10.0, -5.0, 0.5, 100.0], &[4])?;
/// let clipped = clip_gradient(&grad, -1.0, 1.0)?;
/// // clipped contains [1.0, -1.0, 0.5, 1.0]
/// ```
pub fn clip_gradient<T>(grad: &DenseND<T>, min: T, max: T) -> Result<DenseND<T>>
where
    T: Num + Clone + PartialOrd,
{
    if min > max {
        return Err(anyhow::anyhow!("min must be <= max"));
    }

    let arr = grad.as_array();
    let clipped = arr.mapv(|x| {
        if x < min {
            min.clone()
        } else if x > max {
            max.clone()
        } else {
            x
        }
    });

    Ok(DenseND::from_array(clipped))
}

/// Normalize gradient by its L2 norm
///
/// Useful for stabilizing training. Returns a gradient with L2 norm = 1.
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::normalize_gradient;
///
/// let grad = DenseND::from_vec(vec![3.0, 4.0], &[2])?;
/// let normalized = normalize_gradient(&grad)?;
/// // L2 norm of normalized is 1.0
/// ```
pub fn normalize_gradient<T>(grad: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Num + Clone + NumCast + From<f64>,
{
    let arr = grad.as_array();

    let norm_sq: T = arr
        .iter()
        .fold(T::zero(), |acc, x| acc + x.clone() * x.clone());
    let norm_sq_f64 = num_traits::cast::cast::<T, f64>(norm_sq).unwrap_or(0.0);
    let norm_f64 = norm_sq_f64.sqrt();

    if norm_f64 < 1e-12 {
        return Err(anyhow::anyhow!("Cannot normalize near-zero gradient"));
    }

    let norm = <T as From<f64>>::from(norm_f64);
    let normalized = arr.mapv(|x| x / norm.clone());

    Ok(DenseND::from_array(normalized))
}

/// Scale gradient by a constant factor
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::scale_gradient;
///
/// let grad = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
/// let scaled = scale_gradient(&grad, 0.5)?;
/// // scaled contains [0.5, 1.0, 1.5]
/// ```
pub fn scale_gradient<T>(grad: &DenseND<T>, scale: T) -> Result<DenseND<T>>
where
    T: Num + Clone,
{
    let arr = grad.as_array();
    let scaled = arr.mapv(|x| x * scale.clone());
    Ok(DenseND::from_array(scaled))
}

/// Check if gradient contains NaN or Inf values
///
/// Returns true if any element is NaN or infinite.
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::has_nan_or_inf;
///
/// let grad = DenseND::from_vec(vec![1.0, 2.0, f64::NAN], &[3])?;
/// assert!(has_nan_or_inf(&grad));
/// ```
pub fn has_nan_or_inf<T>(grad: &DenseND<T>) -> bool
where
    T: Num + Clone + NumCast,
{
    grad.as_array().iter().any(|x| {
        if let Some(f) = num_traits::cast::cast::<T, f64>(x.clone()) {
            f.is_nan() || f.is_infinite()
        } else {
            false
        }
    })
}

/// Replace NaN and Inf values with zeros
///
/// Useful for recovering from numerical instability.
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::sanitize_gradient;
///
/// let grad = DenseND::from_vec(vec![1.0, f64::NAN, f64::INFINITY], &[3])?;
/// let clean = sanitize_gradient(&grad)?;
/// // clean contains [1.0, 0.0, 0.0]
/// ```
pub fn sanitize_gradient<T>(grad: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Num + Clone + NumCast + From<f64>,
{
    let arr = grad.as_array();
    let sanitized = arr.mapv(|x| {
        if let Some(f) = num_traits::cast::cast::<T, f64>(x.clone()) {
            if f.is_nan() || f.is_infinite() {
                T::zero()
            } else {
                x
            }
        } else {
            x
        }
    });

    Ok(DenseND::from_array(sanitized))
}

/// Accumulate gradients from multiple sources
///
/// Computes the sum of all input gradients. All gradients must have
/// the same shape.
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::accumulate_gradients;
///
/// let grad1 = DenseND::from_vec(vec![1.0, 2.0], &[2])?;
/// let grad2 = DenseND::from_vec(vec![3.0, 4.0], &[2])?;
/// let total = accumulate_gradients(&[grad1, grad2])?;
/// // total contains [4.0, 6.0]
/// ```
pub fn accumulate_gradients<T>(grads: &[DenseND<T>]) -> Result<DenseND<T>>
where
    T: Num + Clone + std::ops::AddAssign,
{
    if grads.is_empty() {
        return Err(anyhow::anyhow!("Cannot accumulate empty gradient list"));
    }

    let shape = grads[0].shape();

    // Verify all have same shape
    for grad in grads.iter().skip(1) {
        if grad.shape() != shape {
            return Err(anyhow::anyhow!(
                "All gradients must have same shape: expected {:?}, got {:?}",
                shape,
                grad.shape()
            ));
        }
    }

    // Accumulate
    let mut result = Array::zeros(grads[0].as_array().raw_dim());
    for grad in grads {
        result += grad.as_array();
    }

    Ok(DenseND::from_array(result))
}

/// Compute the cosine similarity between two gradients
///
/// Returns a value in [-1, 1] indicating how similar the gradient
/// directions are. 1 = same direction, -1 = opposite direction, 0 = orthogonal.
///
/// # Example
///
/// ```rust,ignore
/// use tenrso_ad::utils::gradient_cosine_similarity;
///
/// let grad1 = DenseND::from_vec(vec![1.0, 0.0], &[2])?;
/// let grad2 = DenseND::from_vec(vec![0.0, 1.0], &[2])?;
/// let sim = gradient_cosine_similarity(&grad1, &grad2)?;
/// // sim is approximately 0.0 (orthogonal)
/// ```
pub fn gradient_cosine_similarity<T>(grad1: &DenseND<T>, grad2: &DenseND<T>) -> Result<f64>
where
    T: Num + Clone + NumCast,
{
    if grad1.shape() != grad2.shape() {
        return Err(anyhow::anyhow!(
            "Gradients must have same shape: {:?} vs {:?}",
            grad1.shape(),
            grad2.shape()
        ));
    }

    let arr1 = grad1.as_array();
    let arr2 = grad2.as_array();

    let mut dot: f64 = 0.0;
    let mut norm1_sq: f64 = 0.0;
    let mut norm2_sq: f64 = 0.0;

    for (x, y) in arr1.iter().zip(arr2.iter()) {
        let x_f64 = num_traits::cast::cast::<T, f64>(x.clone()).unwrap_or(0.0);
        let y_f64 = num_traits::cast::cast::<T, f64>(y.clone()).unwrap_or(0.0);

        dot += x_f64 * y_f64;
        norm1_sq += x_f64 * x_f64;
        norm2_sq += y_f64 * y_f64;
    }

    let norm1 = norm1_sq.sqrt();
    let norm2 = norm2_sq.sqrt();

    if norm1 < 1e-12 || norm2 < 1e-12 {
        return Err(anyhow::anyhow!(
            "Cannot compute similarity for near-zero gradients"
        ));
    }

    Ok(dot / (norm1 * norm2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_stats() {
        let grad = DenseND::from_vec(vec![1.0, 2.0, 0.0, -1.0, 0.0, 5.0], &[2, 3]).unwrap();
        let stats: GradientStats<f64> = compute_gradient_stats(&grad).unwrap();

        assert_eq!(stats.total_elements, 6);
        assert_eq!(stats.num_zeros, 2);
        assert_eq!(stats.min, -1.0);
        assert_eq!(stats.max, 5.0);
        let expected_mean: f64 = 7.0 / 6.0;
        assert!((stats.mean - expected_mean).abs() < 1e-10);
    }

    #[test]
    fn test_clip_gradient() {
        let grad = DenseND::from_vec(vec![10.0, -5.0, 0.5, 100.0], &[4]).unwrap();
        let clipped = clip_gradient(&grad, -1.0, 1.0).unwrap();

        assert_eq!(*clipped.get(&[0]).unwrap(), 1.0);
        assert_eq!(*clipped.get(&[1]).unwrap(), -1.0);
        assert_eq!(*clipped.get(&[2]).unwrap(), 0.5);
        assert_eq!(*clipped.get(&[3]).unwrap(), 1.0);
    }

    #[test]
    fn test_normalize_gradient() {
        let grad = DenseND::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let normalized = normalize_gradient(&grad).unwrap();

        let norm_sq: f64 = normalized
            .as_array()
            .iter()
            .fold(0.0, |acc, &x| acc + x * x);
        assert!((norm_sq.sqrt() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_gradient() {
        let grad = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let scaled = scale_gradient(&grad, 0.5).unwrap();

        assert_eq!(*scaled.get(&[0]).unwrap(), 0.5);
        assert_eq!(*scaled.get(&[1]).unwrap(), 1.0);
        assert_eq!(*scaled.get(&[2]).unwrap(), 1.5);
    }

    #[test]
    fn test_has_nan_or_inf() {
        let grad_normal = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(!has_nan_or_inf(&grad_normal));

        let grad_nan = DenseND::from_vec(vec![1.0, f64::NAN, 3.0], &[3]).unwrap();
        assert!(has_nan_or_inf(&grad_nan));

        let grad_inf = DenseND::from_vec(vec![1.0, f64::INFINITY, 3.0], &[3]).unwrap();
        assert!(has_nan_or_inf(&grad_inf));
    }

    #[test]
    fn test_sanitize_gradient() {
        let grad =
            DenseND::from_vec(vec![1.0, f64::NAN, f64::INFINITY, -f64::INFINITY], &[4]).unwrap();
        let clean = sanitize_gradient(&grad).unwrap();

        assert_eq!(*clean.get(&[0]).unwrap(), 1.0);
        assert_eq!(*clean.get(&[1]).unwrap(), 0.0);
        assert_eq!(*clean.get(&[2]).unwrap(), 0.0);
        assert_eq!(*clean.get(&[3]).unwrap(), 0.0);
    }

    #[test]
    fn test_accumulate_gradients() {
        let grad1 = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let grad2 = DenseND::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let grad3 = DenseND::from_vec(vec![5.0, 6.0], &[2]).unwrap();

        let total = accumulate_gradients(&[grad1, grad2, grad3]).unwrap();

        assert_eq!(*total.get(&[0]).unwrap(), 9.0);
        assert_eq!(*total.get(&[1]).unwrap(), 12.0);
    }

    #[test]
    fn test_gradient_cosine_similarity() {
        let grad1 = DenseND::from_vec(vec![1.0, 0.0], &[2]).unwrap();
        let grad2 = DenseND::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let grad3 = DenseND::from_vec(vec![1.0, 0.0], &[2]).unwrap();

        // Orthogonal gradients
        let sim1 = gradient_cosine_similarity(&grad1, &grad2).unwrap();
        assert!(sim1.abs() < 1e-10);

        // Same direction
        let sim2 = gradient_cosine_similarity(&grad1, &grad3).unwrap();
        assert!((sim2 - 1.0).abs() < 1e-10);
    }
}
