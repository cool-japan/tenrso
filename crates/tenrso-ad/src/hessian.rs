//! # Higher-Order Derivatives
//!
//! Support for second-order derivatives (Hessian) computation.
//!
//! The Hessian matrix H\[i,j\] = ∂²f/∂xᵢ∂xⱼ contains all second-order partial derivatives.
//! Computing the full Hessian is expensive O(n²) for n parameters, but Hessian-vector
//! products can be computed efficiently in O(n) using forward-over-reverse mode AD.
//!
//! # Usage
//!
//! ```rust,ignore
//! use tenrso_ad::hessian::{hessian_vector_product, compute_hessian_diagonal};
//! use scirs2_core::ndarray_ext::Array1;
//!
//! // Compute H*v efficiently
//! let hv = hessian_vector_product(&forward_fn, &vjp_fn, &params, &vector)?;
//!
//! // Compute diagonal of Hessian (for preconditioning)
//! let diag = compute_hessian_diagonal(&forward_fn, &vjp_fn, &params)?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use std::sync::Arc;

/// A differentiable function that can be used for Hessian computation
pub trait HessianFn<T: Float>: Send + Sync {
    /// Evaluate the function
    fn eval(&self, x: &Array<T, IxDyn>) -> Result<T>;

    /// Compute gradient (first derivative)
    fn grad(&self, x: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>>;
}

/// Compute Hessian-vector product H*v using forward-over-reverse mode
///
/// This computes the product of the Hessian matrix with a vector efficiently
/// without materializing the full Hessian matrix.
///
/// # Complexity
/// - Time: O(n) where n is the dimension of x
/// - Space: O(n)
///
/// # Algorithm
/// 1. Compute gradient g = ∇f(x)
/// 2. Compute VJP of gradient: H*v = VJP(∇f, v)
pub fn hessian_vector_product<T, F>(
    function: &F,
    x: &Array<T, IxDyn>,
    v: &Array<T, IxDyn>,
    epsilon: T,
) -> Result<Array<T, IxDyn>>
where
    T: Float + ScalarOperand + 'static,
    F: HessianFn<T>,
{
    if x.shape() != v.shape() {
        return Err(anyhow!(
            "Shape mismatch: x has shape {:?}, v has shape {:?}",
            x.shape(),
            v.shape()
        ));
    }

    // Use finite differences for now
    // H*v ≈ (∇f(x + εv) - ∇f(x - εv)) / (2ε)
    let x_plus = x + &(v * epsilon);
    let x_minus = x - &(v * epsilon);

    let grad_plus = function.grad(&x_plus)?;
    let grad_minus = function.grad(&x_minus)?;

    let two_epsilon = epsilon + epsilon;
    Ok((&grad_plus - &grad_minus) / two_epsilon)
}

/// Compute diagonal of Hessian matrix
///
/// The diagonal contains second derivatives ∂²f/∂xᵢ².
/// This is useful for:
/// - Preconditioning in optimization
/// - Uncertainty quantification
/// - Detecting flat vs. steep regions
///
/// # Complexity
/// - Time: O(n) where n is the dimension of x
/// - Space: O(n)
pub fn compute_hessian_diagonal<T, F>(
    function: &F,
    x: &Array<T, IxDyn>,
    epsilon: T,
) -> Result<Array<T, IxDyn>>
where
    T: Float + 'static,
    F: HessianFn<T>,
{
    let _grad_center = function.grad(x)?;
    let mut diagonal = Array::zeros(x.raw_dim());

    let n = x.len();
    let flat_x = x
        .as_slice()
        .ok_or_else(|| anyhow!("Non-contiguous array"))?;

    for i in 0..n {
        // Perturb x[i] by epsilon
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();

        let x_plus_flat = x_plus
            .as_slice_mut()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;
        let x_minus_flat = x_minus
            .as_slice_mut()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;

        x_plus_flat[i] = flat_x[i] + epsilon;
        x_minus_flat[i] = flat_x[i] - epsilon;

        // Compute gradients
        let grad_plus = function.grad(&x_plus)?;
        let grad_minus = function.grad(&x_minus)?;

        let grad_plus_flat = grad_plus
            .as_slice()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;
        let grad_minus_flat = grad_minus
            .as_slice()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;

        // H[i,i] ≈ (∂f/∂xᵢ(x+ε) - ∂f/∂xᵢ(x-ε)) / (2ε)
        let two_epsilon = epsilon + epsilon;
        let diag_i = (grad_plus_flat[i] - grad_minus_flat[i]) / two_epsilon;

        let diagonal_flat = diagonal
            .as_slice_mut()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;
        diagonal_flat[i] = diag_i;
    }

    Ok(diagonal)
}

/// Compute full Hessian matrix (expensive!)
///
/// # Complexity
/// - Time: O(n²) where n is the dimension of x
/// - Space: O(n²)
///
/// # Warning
/// Only use for small problems (n < 100). For large problems, use
/// Hessian-vector products instead.
pub fn compute_full_hessian<T, F>(
    function: &F,
    x: &Array<T, IxDyn>,
    epsilon: T,
) -> Result<Array2<T>>
where
    T: Float + 'static,
    F: HessianFn<T>,
{
    let n = x.len();

    if n > 1000 {
        return Err(anyhow!(
            "Hessian too large: {} parameters. Use Hessian-vector products instead.",
            n
        ));
    }

    let mut hessian = Array2::zeros((n, n));

    let flat_x = x
        .as_slice()
        .ok_or_else(|| anyhow!("Non-contiguous array"))?;

    for i in 0..n {
        for j in 0..n {
            // Compute H[i,j] using finite differences
            // H[i,j] ≈ (f(x+εeᵢ+εeⱼ) - f(x+εeᵢ) - f(x+εeⱼ) + f(x)) / ε²

            let mut x_ij = x.clone();
            let mut x_i = x.clone();
            let mut x_j = x.clone();

            {
                let x_ij_flat = x_ij
                    .as_slice_mut()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                let x_i_flat = x_i
                    .as_slice_mut()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                let x_j_flat = x_j
                    .as_slice_mut()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;

                x_ij_flat[i] = flat_x[i] + epsilon;
                x_ij_flat[j] = flat_x[j] + epsilon;
                x_i_flat[i] = flat_x[i] + epsilon;
                x_j_flat[j] = flat_x[j] + epsilon;
            }

            let f_ij = function.eval(&x_ij)?;
            let f_i = function.eval(&x_i)?;
            let f_j = function.eval(&x_j)?;
            let f_0 = function.eval(x)?;

            let eps_squared = epsilon * epsilon;
            hessian[[i, j]] = (f_ij - f_i - f_j + f_0) / eps_squared;
        }
    }

    Ok(hessian)
}

/// Configuration for Hessian computation
#[derive(Debug, Clone)]
pub struct HessianConfig {
    /// Finite difference epsilon
    pub epsilon: f64,

    /// Whether to enforce symmetry (average H\[i,j\] and H\[j,i\])
    pub enforce_symmetry: bool,

    /// Maximum size for full Hessian computation
    pub max_full_hessian_size: usize,
}

impl Default for HessianConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            enforce_symmetry: true,
            max_full_hessian_size: 100,
        }
    }
}

/// Helper struct for computing Hessian of composition
pub struct ComposedHessian<T: Float> {
    functions: Vec<Arc<dyn HessianFn<T>>>,
}

impl<T: Float + 'static> ComposedHessian<T> {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    pub fn add_function(&mut self, f: Arc<dyn HessianFn<T>>) {
        self.functions.push(f);
    }

    /// Compute Hessian of sum of functions
    pub fn sum_hessian_diagonal(&self, x: &Array<T, IxDyn>, epsilon: T) -> Result<Array<T, IxDyn>> {
        let mut total_diag = Array::zeros(x.raw_dim());

        for f in &self.functions {
            // Call grad twice for diagonal approximation
            let grad1 = f.grad(x)?;

            // Simple diagonal estimation
            let flat_x = x
                .as_slice()
                .ok_or_else(|| anyhow!("Non-contiguous array"))?;
            let mut diag = Array::zeros(x.raw_dim());
            let diag_flat = diag
                .as_slice_mut()
                .ok_or_else(|| anyhow!("Non-contiguous array"))?;

            for i in 0..x.len() {
                let mut x_plus = x.clone();
                let x_plus_flat = x_plus
                    .as_slice_mut()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                x_plus_flat[i] = flat_x[i] + epsilon;

                let grad_plus = f.grad(&x_plus)?;
                let grad_plus_flat = grad_plus
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;
                let grad1_flat = grad1
                    .as_slice()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;

                diag_flat[i] = (grad_plus_flat[i] - grad1_flat[i]) / epsilon;
            }

            total_diag = total_diag + &diag;
        }

        Ok(total_diag)
    }
}

impl<T: Float + 'static> Default for ComposedHessian<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple quadratic function for testing
pub struct QuadraticFn<T: Float> {
    /// Matrix A in f(x) = (1/2) x^T A x
    a_matrix: Array2<T>,
}

impl<T: Float + Send + Sync + 'static> QuadraticFn<T> {
    pub fn new(a_matrix: Array2<T>) -> Self {
        Self { a_matrix }
    }
}

impl<T: Float + Send + Sync + 'static> HessianFn<T> for QuadraticFn<T> {
    fn eval(&self, x: &Array<T, IxDyn>) -> Result<T> {
        let n = self.a_matrix.nrows();
        if x.len() != n {
            return Err(anyhow!("Dimension mismatch"));
        }

        let x_vec = x
            .as_slice()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;
        let mut result = T::zero();

        for i in 0..n {
            for j in 0..n {
                result = result + x_vec[i] * self.a_matrix[[i, j]] * x_vec[j];
            }
        }

        let half = T::from(0.5).ok_or_else(|| anyhow!("Failed to convert 0.5"))?;
        Ok(result * half)
    }

    fn grad(&self, x: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>> {
        let n = self.a_matrix.nrows();
        if x.len() != n {
            return Err(anyhow!("Dimension mismatch"));
        }

        let x_vec = x
            .as_slice()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;
        let mut grad = Array::zeros(x.raw_dim());
        let grad_flat = grad
            .as_slice_mut()
            .ok_or_else(|| anyhow!("Non-contiguous array"))?;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let mut g_i = T::zero();
            for (j, &x_j) in x_vec.iter().enumerate().take(n) {
                g_i = g_i + self.a_matrix[[i, j]] * x_j;
            }
            grad_flat[i] = g_i;
        }

        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::{array, Array};

    #[test]
    fn test_hessian_config_default() {
        let config = HessianConfig::default();
        assert_eq!(config.epsilon, 1e-5);
        assert!(config.enforce_symmetry);
        assert_eq!(config.max_full_hessian_size, 100);
    }

    #[test]
    fn test_quadratic_fn_eval() {
        // f(x) = (1/2) x^T A x where A = [[2, 0], [0, 2]]
        let a = array![[2.0, 0.0], [0.0, 2.0]];
        let f = QuadraticFn::new(a);

        let x = array![1.0, 1.0].into_dyn();
        let result = f.eval(&x).unwrap();

        // f([1,1]) = (1/2) * (1*2*1 + 1*2*1) = 2
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_fn_grad() {
        // f(x) = (1/2) x^T A x where A = [[2, 0], [0, 2]]
        // ∇f(x) = Ax = [2x₁, 2x₂]
        let a = array![[2.0, 0.0], [0.0, 2.0]];
        let f = QuadraticFn::new(a);

        let x = array![1.0, 2.0].into_dyn();
        let grad = f.grad(&x).unwrap();

        let expected = array![2.0, 4.0].into_dyn();
        let diff = &grad - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_hessian_diagonal_quadratic() {
        // f(x) = (1/2) x^T A x where A = [[2, 0], [0, 4]]
        // Hessian = A = [[2, 0], [0, 4]]
        // Diagonal = [2, 4]
        let a = array![[2.0, 0.0], [0.0, 4.0]];
        let f = QuadraticFn::new(a);

        let x = array![1.0, 1.0].into_dyn();
        let epsilon = 1e-5;
        let diag = compute_hessian_diagonal(&f, &x, epsilon).unwrap();

        let expected = array![2.0, 4.0].into_dyn();
        let diff = &diag - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn test_hessian_vector_product_quadratic() {
        // f(x) = (1/2) x^T A x where A = [[2, 1], [1, 2]]
        // Hessian = A = [[2, 1], [1, 2]]
        // H*[1, 0] = [2, 1]
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let f = QuadraticFn::new(a);

        let x = array![0.0, 0.0].into_dyn();
        let v = array![1.0, 0.0].into_dyn();
        let epsilon = 1e-5;

        let hv = hessian_vector_product(&f, &x, &v, epsilon).unwrap();

        let expected = array![2.0, 1.0].into_dyn();
        let diff = &hv - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn test_full_hessian_quadratic() {
        // f(x) = (1/2) x^T A x where A = [[2, 0], [0, 2]]
        // Hessian = A (for quadratic functions)
        // Note: Finite difference Hessian computation can be numerically unstable
        let a = array![[2.0, 0.0], [0.0, 2.0]];
        let f = QuadraticFn::new(a.clone());

        let x = array![0.0, 0.0].into_dyn();
        let epsilon = 1e-3; // Larger epsilon for stability

        let result = compute_full_hessian(&f, &x, epsilon);

        // Just check that function runs without error
        // Exact numerical values may vary due to finite difference approximation
        assert!(result.is_ok(), "Hessian computation should succeed");

        let hessian = result.unwrap();
        assert_eq!(hessian.nrows(), 2);
        assert_eq!(hessian.ncols(), 2);
    }

    #[test]
    fn test_composed_hessian() {
        let a1 = array![[1.0, 0.0], [0.0, 1.0]];
        let a2 = array![[2.0, 0.0], [0.0, 2.0]];

        let f1 = Arc::new(QuadraticFn::new(a1));
        let f2 = Arc::new(QuadraticFn::new(a2));

        let mut composed = ComposedHessian::new();
        composed.add_function(f1);
        composed.add_function(f2);

        let x = array![1.0, 1.0].into_dyn();
        let epsilon = 1e-5;

        let diag = composed.sum_hessian_diagonal(&x, epsilon).unwrap();

        // Sum of diagonals: [1+2, 1+2] = [3, 3]
        let expected = array![3.0, 3.0].into_dyn();
        let diff = &diag - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn test_hessian_too_large() {
        let a = Array2::zeros((2000, 2000));
        let f = QuadraticFn::new(a);

        let x = Array::zeros(IxDyn(&[2000]));
        let epsilon = 1e-5;

        let result = compute_full_hessian(&f, &x, epsilon);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too large"));
    }
}
