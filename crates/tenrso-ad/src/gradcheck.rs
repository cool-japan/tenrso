//! Gradient checking utilities
//!
//! This module provides utilities for verifying gradient implementations
//! using finite difference methods. This is essential for debugging and
//! testing custom VJP rules.
//!
//! # Finite Difference Methods
//!
//! We implement both:
//! - **Central difference**: `f'(x) ≈ [f(x+h) - f(x-h)] / (2h)` (more accurate)
//! - **Forward difference**: `f'(x) ≈ [f(x+h) - f(x)] / h` (faster)
//!
//! # Example
//!
//! ```rust,ignore
//! use tenrso_ad::gradcheck::check_gradient;
//!
//! // Define a function f: R^n -> R^m
//! let f = |x: &DenseND<f64>| {
//!     // Some tensor operation
//!     x.clone()
//! };
//!
//! // Define the analytical gradient
//! let df = |x: &DenseND<f64>, grad_out: &DenseND<f64>| {
//!     // Gradient computation
//!     grad_out.clone()
//! };
//!
//! // Check if analytical gradient matches numerical gradient
//! let x = DenseND::ones(&[3, 4]);
//! let grad_out = DenseND::ones(&[3, 4]);
//! let result = check_gradient(&f, &df, &x, &grad_out, 1e-5, 1e-3);
//! assert!(result.is_ok());
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::numeric::Float;
use tenrso_core::DenseND;

/// Gradient checking configuration
#[derive(Debug, Clone)]
pub struct GradCheckConfig {
    /// Step size for finite differences (default: 1e-5)
    pub epsilon: f64,

    /// Relative tolerance for gradient comparison (default: 1e-3)
    pub rtol: f64,

    /// Absolute tolerance for gradient comparison (default: 1e-5)
    pub atol: f64,

    /// Use central difference (more accurate but 2x slower)
    pub use_central_diff: bool,

    /// Print detailed error messages
    pub verbose: bool,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
            use_central_diff: true,
            verbose: false,
        }
    }
}

/// Result of gradient checking
#[derive(Debug)]
pub struct GradCheckResult {
    /// Maximum absolute difference between analytical and numerical gradients
    pub max_abs_diff: f64,

    /// Maximum relative difference
    pub max_rel_diff: f64,

    /// Whether the gradient check passed
    pub passed: bool,

    /// Number of elements checked
    pub num_elements: usize,

    /// Number of elements that failed the check
    pub num_failures: usize,
}

/// Check gradient using finite differences
///
/// Verifies that the analytical gradient matches the numerical gradient
/// computed via finite differences.
///
/// # Arguments
///
/// * `f` - Forward function: x -> y
/// * `df` - Gradient function: (x, grad_y) -> grad_x
/// * `x` - Input tensor to check gradients at
/// * `grad_y` - Upstream gradient (∂L/∂y)
/// * `config` - Gradient checking configuration
///
/// # Returns
///
/// Result containing gradient check statistics
pub fn check_gradient<T, F, G>(
    f: F,
    df: G,
    x: &DenseND<T>,
    grad_y: &DenseND<T>,
    config: &GradCheckConfig,
) -> Result<GradCheckResult>
where
    T: Float + std::fmt::Display,
    F: Fn(&DenseND<T>) -> Result<DenseND<T>>,
    G: Fn(&DenseND<T>, &DenseND<T>) -> Result<DenseND<T>>,
{
    // Compute analytical gradient
    let analytical_grad = df(x, grad_y)?;

    if analytical_grad.shape() != x.shape() {
        return Err(anyhow!(
            "Gradient shape {:?} doesn't match input shape {:?}",
            analytical_grad.shape(),
            x.shape()
        ));
    }

    // Compute numerical gradient using finite differences
    let numerical_grad = compute_numerical_gradient(f, x, grad_y, config)?;

    // Compare gradients
    compare_gradients(&analytical_grad, &numerical_grad, config)
}

/// Compute numerical gradient using finite differences
fn compute_numerical_gradient<T, F>(
    f: F,
    x: &DenseND<T>,
    grad_y: &DenseND<T>,
    config: &GradCheckConfig,
) -> Result<DenseND<T>>
where
    T: Float,
    F: Fn(&DenseND<T>) -> Result<DenseND<T>>,
{
    let epsilon = T::from(config.epsilon).ok_or_else(|| anyhow!("Failed to convert epsilon"))?;
    let shape = x.shape();
    let mut numerical_grad = DenseND::zeros(shape);

    // Iterate over all elements
    for idx in 0..x.len() {
        let multi_idx = x.linear_to_multi_index(idx);

        // Perturb x[idx] by +epsilon
        let mut x_plus = x.clone();
        let val = *x_plus
            .get(&multi_idx)
            .ok_or_else(|| anyhow!("Index error"))?;
        *x_plus
            .get_mut(&multi_idx)
            .ok_or_else(|| anyhow!("Index error"))? = val + epsilon;

        let y_plus = f(&x_plus)?;

        let grad_contribution = if config.use_central_diff {
            // Central difference: [f(x+h) - f(x-h)] / (2h)
            let mut x_minus = x.clone();
            let val = *x_minus
                .get(&multi_idx)
                .ok_or_else(|| anyhow!("Index error"))?;
            *x_minus
                .get_mut(&multi_idx)
                .ok_or_else(|| anyhow!("Index error"))? = val - epsilon;

            let y_minus = f(&x_minus)?;

            // Compute: dot(grad_y, (y_plus - y_minus) / (2*epsilon))
            let diff = y_plus.sub(&y_minus)?;
            let scaled_diff = diff.scalar_mul(T::one() / (epsilon + epsilon))?;
            dot_product(grad_y, &scaled_diff)?
        } else {
            // Forward difference: [f(x+h) - f(x)] / h
            let y = f(x)?;
            let diff = y_plus.sub(&y)?;
            let scaled_diff = diff.scalar_mul(T::one() / epsilon)?;
            dot_product(grad_y, &scaled_diff)?
        };

        *numerical_grad
            .get_mut(&multi_idx)
            .ok_or_else(|| anyhow!("Index error"))? = grad_contribution;
    }

    Ok(numerical_grad)
}

/// Compute dot product between two tensors (element-wise multiplication and sum)
fn dot_product<T>(a: &DenseND<T>, b: &DenseND<T>) -> Result<T>
where
    T: Float,
{
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "Shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        ));
    }

    let mut sum = T::zero();
    for idx in 0..a.len() {
        let multi_idx = a.linear_to_multi_index(idx);
        let a_val = a.get(&multi_idx).ok_or_else(|| anyhow!("Index error"))?;
        let b_val = b.get(&multi_idx).ok_or_else(|| anyhow!("Index error"))?;
        sum = sum + (*a_val * *b_val);
    }

    Ok(sum)
}

/// Compare analytical and numerical gradients
fn compare_gradients<T>(
    analytical: &DenseND<T>,
    numerical: &DenseND<T>,
    config: &GradCheckConfig,
) -> Result<GradCheckResult>
where
    T: Float + std::fmt::Display,
{
    let rtol = T::from(config.rtol).ok_or_else(|| anyhow!("Failed to convert rtol"))?;
    let atol = T::from(config.atol).ok_or_else(|| anyhow!("Failed to convert atol"))?;

    let mut max_abs_diff = 0.0_f64;
    let mut max_rel_diff = 0.0_f64;
    let mut num_failures = 0;

    for idx in 0..analytical.len() {
        let multi_idx = analytical.linear_to_multi_index(idx);

        let a_val = analytical
            .get(&multi_idx)
            .ok_or_else(|| anyhow!("Index error"))?;
        let n_val = numerical
            .get(&multi_idx)
            .ok_or_else(|| anyhow!("Index error"))?;

        let abs_diff = (*a_val - *n_val).abs();
        let rel_diff = if n_val.abs() > T::epsilon() {
            abs_diff / n_val.abs()
        } else {
            abs_diff
        };

        // Update max differences
        let abs_diff_f64 = abs_diff
            .to_f64()
            .ok_or_else(|| anyhow!("Conversion error"))?;
        let rel_diff_f64 = rel_diff
            .to_f64()
            .ok_or_else(|| anyhow!("Conversion error"))?;

        max_abs_diff = max_abs_diff.max(abs_diff_f64);
        max_rel_diff = max_rel_diff.max(rel_diff_f64);

        // Check if this element fails
        if abs_diff > atol && rel_diff > rtol {
            num_failures += 1;

            if config.verbose {
                println!(
                    "Gradient mismatch at {:?}: analytical={}, numerical={}, abs_diff={}, rel_diff={}",
                    multi_idx, a_val, n_val, abs_diff, rel_diff
                );
            }
        }
    }

    let passed = num_failures == 0;

    if config.verbose {
        if passed {
            println!("✓ Gradient check passed!");
        } else {
            println!(
                "✗ Gradient check failed: {}/{} elements exceeded tolerance",
                num_failures,
                analytical.len()
            );
        }
        println!("  Max absolute difference: {:.2e}", max_abs_diff);
        println!("  Max relative difference: {:.2e}", max_rel_diff);
    }

    Ok(GradCheckResult {
        max_abs_diff,
        max_rel_diff,
        passed,
        num_elements: analytical.len(),
        num_failures,
    })
}

// Helper trait for DenseND operations needed by gradcheck
trait GradCheckOps<T> {
    fn sub(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;
    fn scalar_mul(&self, scalar: T) -> Result<Self>
    where
        Self: Sized;
    fn linear_to_multi_index(&self, linear_idx: usize) -> Vec<usize>;
}

impl<T> GradCheckOps<T> for DenseND<T>
where
    T: Float,
{
    fn sub(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(anyhow!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let mut result = self.clone();
        for idx in 0..self.len() {
            let multi_idx = self.linear_to_multi_index(idx);
            let a = *self.get(&multi_idx).ok_or_else(|| anyhow!("Index error"))?;
            let b = *other
                .get(&multi_idx)
                .ok_or_else(|| anyhow!("Index error"))?;
            *result
                .get_mut(&multi_idx)
                .ok_or_else(|| anyhow!("Index error"))? = a - b;
        }

        Ok(result)
    }

    fn scalar_mul(&self, scalar: T) -> Result<Self> {
        let mut result = self.clone();
        for idx in 0..self.len() {
            let multi_idx = self.linear_to_multi_index(idx);
            let val = *self.get(&multi_idx).ok_or_else(|| anyhow!("Index error"))?;
            *result
                .get_mut(&multi_idx)
                .ok_or_else(|| anyhow!("Index error"))? = val * scalar;
        }
        Ok(result)
    }

    fn linear_to_multi_index(&self, linear_idx: usize) -> Vec<usize> {
        let shape = self.shape();
        let mut multi_idx = vec![0; shape.len()];
        let mut remaining = linear_idx;

        for (dim, &size) in shape.iter().enumerate().rev() {
            multi_idx[dim] = remaining % size;
            remaining /= size;
        }

        multi_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradcheck_identity() {
        // f(x) = x, df/dx = 1
        let f = |x: &DenseND<f64>| Ok(x.clone());
        let df = |_x: &DenseND<f64>, grad_y: &DenseND<f64>| Ok(grad_y.clone());

        let x = DenseND::ones(&[2, 3]);
        let grad_y = DenseND::ones(&[2, 3]);

        let config = GradCheckConfig {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
            use_central_diff: true,
            verbose: false,
        };

        let result = check_gradient(f, df, &x, &grad_y, &config).unwrap();
        assert!(result.passed, "Gradient check should pass for identity");
        assert!(
            result.max_abs_diff < 1e-6,
            "Max abs diff should be very small"
        );
    }

    #[test]
    fn test_gradcheck_square() {
        // f(x) = x^2, df/dx = 2x
        let f = |x: &DenseND<f64>| {
            let mut result = x.clone();
            for idx in 0..x.len() {
                let multi_idx = x.linear_to_multi_index(idx);
                let val = *x.get(&multi_idx).unwrap();
                *result.get_mut(&multi_idx).unwrap() = val * val;
            }
            Ok(result)
        };

        let df = |x: &DenseND<f64>, grad_y: &DenseND<f64>| {
            let mut grad_x = DenseND::zeros(x.shape());
            for idx in 0..x.len() {
                let multi_idx = x.linear_to_multi_index(idx);
                let x_val = *x.get(&multi_idx).unwrap();
                let gy_val = *grad_y.get(&multi_idx).unwrap();
                *grad_x.get_mut(&multi_idx).unwrap() = 2.0 * x_val * gy_val;
            }
            Ok(grad_x)
        };

        let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let grad_y = DenseND::ones(&[2, 2]);

        let config = GradCheckConfig::default();
        let result = check_gradient(f, df, &x, &grad_y, &config).unwrap();

        assert!(result.passed, "Gradient check should pass for f(x) = x^2");
        assert!(
            result.max_rel_diff < 1e-4,
            "Max rel diff should be small, got {}",
            result.max_rel_diff
        );
    }

    #[test]
    fn test_gradcheck_config_default() {
        let config = GradCheckConfig::default();
        assert_eq!(config.epsilon, 1e-5);
        assert_eq!(config.rtol, 1e-3);
        assert!(config.use_central_diff);
    }
}
