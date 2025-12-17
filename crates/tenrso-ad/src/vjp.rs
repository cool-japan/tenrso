//! Vector-Jacobian Product (VJP) rules for tensor operations
//!
//! This module implements custom VJP rules for efficient backward passes
//! through tensor contractions, avoiding AD tape blow-up.
//!
//! # Overview
//!
//! For a forward operation `y = f(x1, x2, ...)`, the VJP computes:
//! ```text
//! vjp(dy) = (∂L/∂x1, ∂L/∂x2, ...)
//! ```
//! where `dy = ∂L/∂y` is the incoming gradient (cotangent).
//!
//! # Einsum VJP
//!
//! For `C = einsum("spec", A, B)`, the gradients are computed by:
//! - `grad_A = einsum(adjoint_spec_A, grad_C, B)`
//! - `grad_B = einsum(adjoint_spec_B, A, grad_C)`
//!
//! where the adjoint specs are derived from the original einsum specification.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array, IxDyn, Zip};
use scirs2_core::numeric::{Num, NumCast};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction;
use tenrso_planner::EinsumSpec;

/// Trait for operations that support VJP (backward differentiation)
pub trait VjpOp<T>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
{
    /// Compute the VJP (backward pass) given the output gradient
    ///
    /// # Arguments
    ///
    /// * `output_grad` - Gradient w.r.t. the output (∂L/∂output)
    ///
    /// # Returns
    ///
    /// Gradients w.r.t. each input: (∂L/∂input1, ∂L/∂input2, ...)
    fn vjp(&self, output_grad: &DenseND<T>) -> Result<Vec<DenseND<T>>>;
}

/// VJP context for einsum contractions
///
/// Stores the forward pass inputs needed for efficient backward computation.
///
/// # Example
///
/// ```rust,ignore
/// // Forward pass
/// let spec = EinsumSpec::parse("ij,jk->ik")?;
/// let c = execute_dense_contraction(&spec, &a, &b)?;
///
/// // Create VJP context
/// let vjp_ctx = EinsumVjp::new(spec, a.clone(), b.clone());
///
/// // Backward pass
/// let grads = vjp_ctx.vjp(&grad_c)?;
/// let grad_a = &grads[0];
/// let grad_b = &grads[1];
/// ```
pub struct EinsumVjp<T>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
{
    /// Original einsum specification
    pub spec: EinsumSpec,
    /// First input tensor (saved from forward pass)
    pub input_a: DenseND<T>,
    /// Second input tensor (saved from forward pass)
    pub input_b: DenseND<T>,
}

impl<T> EinsumVjp<T>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
{
    /// Create a new einsum VJP context
    pub fn new(spec: EinsumSpec, input_a: DenseND<T>, input_b: DenseND<T>) -> Self {
        Self {
            spec,
            input_a,
            input_b,
        }
    }

    /// Compute the adjoint einsum specification for the first input
    ///
    /// For `C = einsum("ij,jk->ik", A, B)`, this computes the spec for:
    /// `grad_A = einsum(adjoint_spec, grad_C, B)`
    ///
    /// Algorithm:
    /// 1. Original: `spec_a, spec_b -> spec_out`
    /// 2. Adjoint for A: `spec_out, spec_b -> spec_a`
    fn adjoint_spec_for_input_a(&self) -> Result<EinsumSpec> {
        let spec_a = &self.spec.inputs[0];
        let spec_b = &self.spec.inputs[1];
        let spec_out = &self.spec.output;

        // Build the adjoint specification: grad_c, b -> grad_a
        // We need to contract grad_c with b to produce grad_a
        let adjoint_str = format!("{},{}->{}", spec_out, spec_b, spec_a);
        EinsumSpec::parse(&adjoint_str)
    }

    /// Compute the adjoint einsum specification for the second input
    ///
    /// For `C = einsum("ij,jk->ik", A, B)`, this computes the spec for:
    /// `grad_B = einsum(adjoint_spec, A, grad_C)`
    fn adjoint_spec_for_input_b(&self) -> Result<EinsumSpec> {
        let spec_a = &self.spec.inputs[0];
        let spec_b = &self.spec.inputs[1];
        let spec_out = &self.spec.output;

        // Build the adjoint specification: a, grad_c -> grad_b
        let adjoint_str = format!("{},{}->{}", spec_a, spec_out, spec_b);
        EinsumSpec::parse(&adjoint_str)
    }
}

impl<T> VjpOp<T> for EinsumVjp<T>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
{
    fn vjp(&self, output_grad: &DenseND<T>) -> Result<Vec<DenseND<T>>> {
        // Special case: if output is scalar, use broadcasting instead of einsum
        if self.spec.output.is_empty() {
            // For scalar output, gradient is just the scalar value broadcast to input shapes
            // grad_a = output_grad * input_b
            // grad_b = output_grad * input_a

            // Get the scalar gradient value
            if output_grad.shape().is_empty() {
                let scalar_grad = output_grad.as_array()[[].as_ref()].clone();

                // Broadcast to input shapes
                let mut grad_a_data = Array::zeros(IxDyn(self.input_a.shape()));
                let mut grad_b_data = Array::zeros(IxDyn(self.input_b.shape()));

                // For inner product i,i->, gradients are:
                // grad_a[i] = output_grad * input_b[i]
                // grad_b[i] = output_grad * input_a[i]
                Zip::from(&mut grad_a_data)
                    .and(self.input_b.as_array())
                    .for_each(|ga, b| *ga = scalar_grad.clone() * b.clone());

                Zip::from(&mut grad_b_data)
                    .and(self.input_a.as_array())
                    .for_each(|gb, a| *gb = scalar_grad.clone() * a.clone());

                return Ok(vec![
                    DenseND::from_array(grad_a_data),
                    DenseND::from_array(grad_b_data),
                ]);
            } else {
                return Err(anyhow!(
                    "Expected scalar output gradient for scalar einsum output"
                ));
            }
        }

        // Normal case: use adjoint einsum contractions
        let adjoint_spec_a = self.adjoint_spec_for_input_a()?;
        let grad_a = execute_dense_contraction(&adjoint_spec_a, output_grad, &self.input_b)?;

        // Compute gradient w.r.t. second input
        let adjoint_spec_b = self.adjoint_spec_for_input_b()?;
        let grad_b = execute_dense_contraction(&adjoint_spec_b, &self.input_a, output_grad)?;

        Ok(vec![grad_a, grad_b])
    }
}

/// VJP for element-wise unary operations
///
/// For operations like `y = f(x)` where `f` is applied element-wise,
/// the gradient is `grad_x = grad_y * f'(x)`.
pub struct ElementwiseUnaryVjp<T, F>
where
    T: Num + Clone,
    F: Fn(&T) -> T,
{
    /// Input tensor (saved from forward pass)
    pub input: DenseND<T>,
    /// Derivative function: f'(x)
    pub derivative: F,
}

impl<T, F> ElementwiseUnaryVjp<T, F>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
    F: Fn(&T) -> T,
{
    /// Create a new element-wise unary VJP context
    pub fn new(input: DenseND<T>, derivative: F) -> Self {
        Self { input, derivative }
    }
}

impl<T, F> VjpOp<T> for ElementwiseUnaryVjp<T, F>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
    F: Fn(&T) -> T,
{
    fn vjp(&self, output_grad: &DenseND<T>) -> Result<Vec<DenseND<T>>> {
        if self.input.shape() != output_grad.shape() {
            return Err(anyhow!(
                "Shape mismatch: input {:?} vs output_grad {:?}",
                self.input.shape(),
                output_grad.shape()
            ));
        }

        // Compute grad_x = grad_y * f'(x) element-wise
        let mut result = Array::zeros(IxDyn(self.input.shape()));

        Zip::from(&mut result)
            .and(self.input.as_array())
            .and(output_grad.as_array())
            .for_each(|r, x, g| {
                *r = (self.derivative)(x) * g.clone();
            });

        Ok(vec![DenseND::from_array(result)])
    }
}

/// VJP for element-wise binary operations
///
/// For operations like `z = f(x, y)`, computes gradients for both inputs.
pub struct ElementwiseBinaryVjp<T, Fx, Fy>
where
    T: Num + Clone,
    Fx: Fn(&T, &T) -> T,
    Fy: Fn(&T, &T) -> T,
{
    /// First input tensor
    pub input_x: DenseND<T>,
    /// Second input tensor
    pub input_y: DenseND<T>,
    /// Partial derivative w.r.t. x: ∂f/∂x
    pub derivative_x: Fx,
    /// Partial derivative w.r.t. y: ∂f/∂y
    pub derivative_y: Fy,
}

impl<T, Fx, Fy> ElementwiseBinaryVjp<T, Fx, Fy>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
    Fx: Fn(&T, &T) -> T,
    Fy: Fn(&T, &T) -> T,
{
    /// Create a new element-wise binary VJP context
    pub fn new(
        input_x: DenseND<T>,
        input_y: DenseND<T>,
        derivative_x: Fx,
        derivative_y: Fy,
    ) -> Self {
        Self {
            input_x,
            input_y,
            derivative_x,
            derivative_y,
        }
    }
}

impl<T, Fx, Fy> VjpOp<T> for ElementwiseBinaryVjp<T, Fx, Fy>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default,
    Fx: Fn(&T, &T) -> T,
    Fy: Fn(&T, &T) -> T,
{
    fn vjp(&self, output_grad: &DenseND<T>) -> Result<Vec<DenseND<T>>> {
        if self.input_x.shape() != output_grad.shape() {
            return Err(anyhow!(
                "Shape mismatch: input_x {:?} vs output_grad {:?}",
                self.input_x.shape(),
                output_grad.shape()
            ));
        }

        // Compute gradients for both inputs element-wise
        let mut grad_x = Array::zeros(IxDyn(self.input_x.shape()));
        let mut grad_y = Array::zeros(IxDyn(self.input_y.shape()));

        Zip::from(&mut grad_x)
            .and(&mut grad_y)
            .and(self.input_x.as_array())
            .and(self.input_y.as_array())
            .and(output_grad.as_array())
            .for_each(|gx, gy, x, y, g| {
                *gx = (self.derivative_x)(x, y) * g.clone();
                *gy = (self.derivative_y)(x, y) * g.clone();
            });

        Ok(vec![
            DenseND::from_array(grad_x),
            DenseND::from_array(grad_y),
        ])
    }
}

/// VJP for reduction operations
///
/// For operations like `y = sum(x, axis)` or `y = mean(x, axis)`,
/// the gradient is broadcasted back to the original shape.
pub struct ReductionVjp<T>
where
    T: Num + Clone,
{
    /// Original input shape (before reduction)
    pub input_shape: Vec<usize>,
    /// Axis that was reduced over
    pub axis: Option<usize>,
    /// Reduction type (sum, mean, etc.)
    pub reduction_type: ReductionType,
    /// Phantom data to use T
    _phantom: std::marker::PhantomData<T>,
}

/// Type of reduction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionType {
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// Max reduction (requires special handling)
    Max,
    /// Min reduction (requires special handling)
    Min,
}

impl<T> ReductionVjp<T>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default + NumCast,
{
    /// Create a new reduction VJP context
    pub fn new(
        input_shape: Vec<usize>,
        axis: Option<usize>,
        reduction_type: ReductionType,
    ) -> Self {
        Self {
            input_shape,
            axis,
            reduction_type,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Broadcast the gradient back to the original shape
    fn broadcast_grad(&self, grad: &DenseND<T>) -> Result<DenseND<T>> {
        match self.axis {
            None => {
                // Full reduction: broadcast scalar to full shape
                if grad.len() != 1 {
                    return Err(anyhow!(
                        "Expected scalar gradient for full reduction, got shape {:?}",
                        grad.shape()
                    ));
                }
                let scalar = grad.get(&vec![0; grad.rank()]).unwrap();

                // Apply scaling for mean reduction
                let scaled_scalar = match self.reduction_type {
                    ReductionType::Mean => {
                        let n_elements = self.input_shape.iter().product::<usize>();
                        let divisor = T::from(n_elements).ok_or_else(|| {
                            anyhow!("Failed to convert n_elements to numeric type")
                        })?;
                        scalar.clone() / divisor
                    }
                    _ => scalar.clone(),
                };

                Ok(DenseND::from_elem(&self.input_shape, scaled_scalar))
            }
            Some(_axis) => {
                // Partial reduction: broadcast along the reduced axis
                // Apply scaling for mean reduction
                let scale = match self.reduction_type {
                    ReductionType::Mean => {
                        let total: usize = self.input_shape.iter().product();
                        let grad_total: usize = grad.shape().iter().product();
                        let divisor = T::from(total / grad_total)
                            .ok_or_else(|| anyhow!("Failed to convert divisor to numeric type"))?;
                        T::one() / divisor
                    }
                    _ => T::one(),
                };

                // Use ndarray's broadcasting capabilities
                let grad_array = grad.as_array().to_owned();
                let broadcasted = grad_array
                    .broadcast(IxDyn(self.input_shape.as_slice()))
                    .ok_or_else(|| {
                        anyhow!(
                            "Failed to broadcast gradient from {:?} to {:?}",
                            grad.shape(),
                            self.input_shape
                        )
                    })?;

                // Apply scaling if needed
                let mut result = broadcasted.to_owned();
                if scale != T::one() {
                    result.mapv_inplace(|x| x * scale.clone());
                }

                Ok(DenseND::from_array(result))
            }
        }
    }
}

impl<T> VjpOp<T> for ReductionVjp<T>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default + NumCast,
{
    fn vjp(&self, output_grad: &DenseND<T>) -> Result<Vec<DenseND<T>>> {
        let grad_input = self.broadcast_grad(output_grad)?;
        Ok(vec![grad_input])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_einsum_vjp_matmul() {
        // Forward: C = A @ B (matrix multiplication)
        // C[i,k] = sum_j A[i,j] * B[j,k]
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let c = execute_dense_contraction(&spec, &a, &b).unwrap();

        // Backward: given grad_C, compute grad_A and grad_B
        let grad_c = DenseND::ones(c.shape());

        let vjp_ctx = EinsumVjp::new(spec, a.clone(), b.clone());
        let grads = vjp_ctx.vjp(&grad_c).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), a.shape());
        assert_eq!(grads[1].shape(), b.shape());
    }

    #[test]
    fn test_einsum_vjp_adjoint_spec_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let a = DenseND::<f64>::zeros(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[3, 4]);

        let vjp_ctx = EinsumVjp::new(spec, a, b);

        // Adjoint for A: grad_C (i,k), B (j,k) -> grad_A (i,j)
        // This should be: "ik,jk->ij"
        let adj_a = vjp_ctx.adjoint_spec_for_input_a().unwrap();
        assert_eq!(adj_a.inputs.len(), 2);

        // Adjoint for B: A (i,j), grad_C (i,k) -> grad_B (j,k)
        // This should be: "ij,ik->jk"
        let adj_b = vjp_ctx.adjoint_spec_for_input_b().unwrap();
        assert_eq!(adj_b.inputs.len(), 2);
    }

    #[test]
    fn test_elementwise_unary_vjp() {
        // Forward: y = x^2
        // Backward: grad_x = grad_y * 2x
        let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let derivative = |x: &f64| 2.0 * x;
        let vjp_ctx = ElementwiseUnaryVjp::new(x.clone(), derivative);

        let grad_y = DenseND::ones(x.shape());
        let grads = vjp_ctx.vjp(&grad_y).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), x.shape());

        // Check values: should be [2.0, 4.0, 6.0, 8.0]
        assert_eq!(*grads[0].get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*grads[0].get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(*grads[0].get(&[1, 0]).unwrap(), 6.0);
        assert_eq!(*grads[0].get(&[1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn test_reduction_vjp_sum() {
        let input_shape = vec![2, 3];
        let vjp_ctx = ReductionVjp::<f64>::new(input_shape.clone(), None, ReductionType::Sum);

        // Full reduction: scalar gradient should broadcast to full shape
        let grad_out = DenseND::from_elem(&[1, 1], 5.0);
        let grads = vjp_ctx.vjp(&grad_out).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), input_shape.as_slice());

        // All elements should be 5.0
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*grads[0].get(&[i, j]).unwrap(), 5.0);
            }
        }
    }

    #[test]
    fn test_einsum_vjp_scalar_output() {
        // Test inner product (scalar output): y = sum_i(a[i] * b[i])
        // Forward: y = a · b
        let spec = EinsumSpec::parse("i,i->").unwrap();

        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();

        let y = execute_dense_contraction(&spec, &a, &b).unwrap();

        // Result should be scalar: 1*5 + 2*6 + 3*7 + 4*8 = 70
        assert!(y.shape().is_empty());
        assert_eq!(*y.get(&[]).unwrap(), 70.0);

        // Backward: given grad_y (scalar), compute grad_a and grad_b
        // grad_a[i] = grad_y * b[i]
        // grad_b[i] = grad_y * a[i]
        let grad_y = DenseND::from_elem(&[], 1.0);

        let vjp_ctx = EinsumVjp::new(spec, a.clone(), b.clone());
        let grads = vjp_ctx.vjp(&grad_y).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), a.shape());
        assert_eq!(grads[1].shape(), b.shape());

        // Check grad_a = b
        for i in 0..4 {
            assert_eq!(*grads[0].get(&[i]).unwrap(), *b.get(&[i]).unwrap());
        }

        // Check grad_b = a
        for i in 0..4 {
            assert_eq!(*grads[1].get(&[i]).unwrap(), *a.get(&[i]).unwrap());
        }
    }
}
