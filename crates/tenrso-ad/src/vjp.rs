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

use anyhow::{anyhow, bail, Result};
use scirs2_core::ndarray_ext::{Array, IxDyn, Zip};
use scirs2_core::numeric::{Num, NumCast};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction_accelerated;
use tenrso_planner::EinsumSpec;

use crate::registry::reduction::ReductionLayout;
use crate::registry::{AdScalar, OpParams, OpRule, ReduceKind, ReductionRule};

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
    T: Num + Clone + std::ops::AddAssign + std::default::Default + 'static,
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

        // Normal case: use adjoint einsum contractions, routed through the
        // native-GEMM-accelerated contraction engine (the `+ 'static` bound
        // on this impl is exactly what makes that dispatch possible; see
        // `execute_dense_contraction_accelerated`'s docs).
        let adjoint_spec_a = self.adjoint_spec_for_input_a()?;
        let grad_a =
            execute_dense_contraction_accelerated(&adjoint_spec_a, output_grad, &self.input_b)?;

        // Compute gradient w.r.t. second input
        let adjoint_spec_b = self.adjoint_spec_for_input_b()?;
        let grad_b =
            execute_dense_contraction_accelerated(&adjoint_spec_b, &self.input_a, output_grad)?;

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
/// For operations like `y = sum(x, axes)`, `y = max(x, axes)`, etc., computes
/// the gradient w.r.t. `x` given the upstream gradient w.r.t. `y`.
///
/// # Implementation
///
/// [`ReductionVjp::vjp`] does not re-derive the reduction gradient math: it
/// validates and translates this context's axis selection --- `axis: Option<usize>`
/// (`None` = every axis, `Some(a)` = axis `a`) for [`ReductionVjp::new`] /
/// [`ReductionVjp::with_input`], or an arbitrary explicit set for
/// [`ReductionVjp::with_input_axes`] --- into
/// [`crate::registry::ReductionRule`]'s `axes: Vec<usize>` / `keepdims = false`
/// convention, then calls [`crate::registry::ReductionRule`]'s own
/// (finite-difference verified) VJP directly. There is exactly one
/// implementation of the sum/mean/max/min/product gradient math in this crate.
///
/// # Why two constructors
///
/// [`ReductionVjp::new`] stores only the input **shape**. That suffices for
/// [`ReductionType::Sum`] and [`ReductionType::Mean`]: their gradient depends
/// only on which output cell each input element folds into (a shape-level
/// fact), never on the input's actual values. It is **structurally**
/// insufficient for [`ReductionType::Max`], [`ReductionType::Min`] and
/// [`ReductionType::Product`] --- routing a max/min gradient to the
/// arg-extremum, or scaling a product gradient by "the product of the other
/// elements in the group", both require the real input values. Calling
/// [`VjpOp::vjp`] on a `Max`/`Min`/`Product` context built with `new` returns
/// an honest error instead of a silently wrong (uniformly broadcast)
/// gradient; use [`ReductionVjp::with_input`] (or
/// [`ReductionVjp::with_input_axes`]) instead, which save the input tensor.
///
/// # Tie-breaking (`Max` / `Min`)
///
/// When several elements of a reduction group tie for the extremum, the
/// gradient is routed to the **first** such element in row-major order --
/// the same documented, gradchecked rule [`crate::registry::ReductionRule`]
/// uses. This is deterministic and reproducible, and matches the sub-gradient
/// convention used by mainstream AD frameworks.
pub struct ReductionVjp<T>
where
    T: Num + Clone,
{
    /// Original input shape (before reduction).
    pub input_shape: Vec<usize>,
    /// Axis reduced over, for contexts built with [`ReductionVjp::new`] /
    /// [`ReductionVjp::with_input`] (`None` means "every axis"). For
    /// [`ReductionVjp::with_input_axes`] this reports `Some(a)` only when the
    /// given axis set happens to be the single axis `[a]`, and `None`
    /// otherwise (including the "every axis" and "more than one axis" cases) ---
    /// `vjp` always computes from the private, fully general `axes` field
    /// instead of this one.
    pub axis: Option<usize>,
    /// Reduction type (sum, mean, max, min, product).
    pub reduction_type: ReductionType,
    /// The fully general axis set actually used for computation. Derived from
    /// `axis` for `new`/`with_input`; given directly to `with_input_axes`.
    axes: Vec<usize>,
    /// Saved input values. `None` for contexts built with [`ReductionVjp::new`]
    /// (shape-only); `Some` for [`ReductionVjp::with_input`] /
    /// [`ReductionVjp::with_input_axes`].
    input: Option<DenseND<T>>,
}

/// Type of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionType {
    /// Sum reduction.
    Sum,
    /// Mean reduction.
    Mean,
    /// Max reduction (requires the input values; see [`ReductionVjp::with_input`]).
    Max,
    /// Min reduction (requires the input values; see [`ReductionVjp::with_input`]).
    Min,
    /// Product reduction (requires the input values; see [`ReductionVjp::with_input`]).
    Product,
}

impl ReductionType {
    /// Does this reduction's gradient require the input *values* (not just
    /// its shape) to be computed correctly?
    fn needs_input_values(self) -> bool {
        matches!(
            self,
            ReductionType::Max | ReductionType::Min | ReductionType::Product
        )
    }

    /// Translate to the registry's [`ReduceKind`], which this type mirrors
    /// one-for-one.
    fn to_reduce_kind(self) -> ReduceKind {
        match self {
            ReductionType::Sum => ReduceKind::Sum,
            ReductionType::Mean => ReduceKind::Mean,
            ReductionType::Max => ReduceKind::Max,
            ReductionType::Min => ReduceKind::Min,
            ReductionType::Product => ReduceKind::Product,
        }
    }
}

/// Expand the legacy `axis: Option<usize>` convention into an `axes` list:
/// `None` reduces every axis (full reduction), `Some(a)` reduces only axis `a`.
fn axes_from_axis(axis: Option<usize>, rank: usize) -> Vec<usize> {
    match axis {
        None => (0..rank).collect(),
        Some(a) => vec![a],
    }
}

impl<T> ReductionVjp<T>
where
    T: Num + Clone + std::ops::AddAssign + std::default::Default + NumCast,
{
    /// Create a new reduction VJP context from the input **shape** alone.
    ///
    /// Only [`ReductionType::Sum`] and [`ReductionType::Mean`] can be computed
    /// this way -- [`VjpOp::vjp`] returns an error for `Max`/`Min`/`Product`
    /// contexts constructed with this function. Use
    /// [`ReductionVjp::with_input`] for those.
    pub fn new(
        input_shape: Vec<usize>,
        axis: Option<usize>,
        reduction_type: ReductionType,
    ) -> Self {
        let axes = axes_from_axis(axis, input_shape.len());
        Self {
            input_shape,
            axis,
            reduction_type,
            axes,
            input: None,
        }
    }
}

impl<T> ReductionVjp<T>
where
    T: AdScalar,
{
    /// Create a reduction VJP context that saves the input tensor.
    ///
    /// Required for [`ReductionType::Max`] / [`ReductionType::Min`] /
    /// [`ReductionType::Product`]; also valid (and equally correct) for
    /// `Sum`/`Mean`. `axis = None` reduces every axis (full reduction);
    /// `axis = Some(a)` reduces only axis `a` (any position, not just leading).
    pub fn with_input(
        input: DenseND<T>,
        axis: Option<usize>,
        reduction_type: ReductionType,
    ) -> Self {
        let axes = axes_from_axis(axis, input.rank());
        Self {
            input_shape: input.shape().to_vec(),
            axis,
            reduction_type,
            axes,
            input: Some(input),
        }
    }

    /// Create a reduction VJP context reducing an explicit, arbitrary set of
    /// axes in one call -- e.g. axes `[0, 2]` of a rank-3 tensor while keeping
    /// axis `1`, in any order and not necessarily contiguous or leading.
    pub fn with_input_axes(
        input: DenseND<T>,
        axes: Vec<usize>,
        reduction_type: ReductionType,
    ) -> Self {
        let axis = match axes.as_slice() {
            [only] => Some(*only),
            _ => None,
        };
        Self {
            input_shape: input.shape().to_vec(),
            axis,
            reduction_type,
            axes,
            input: Some(input),
        }
    }
}

impl<T> VjpOp<T> for ReductionVjp<T>
where
    T: AdScalar,
{
    fn vjp(&self, output_grad: &DenseND<T>) -> Result<Vec<DenseND<T>>> {
        if self.reduction_type.needs_input_values() && self.input.is_none() {
            bail!(
                "ReductionVjp: {:?} needs the saved input values to route the gradient to \
                 the arg-extremum (or scale by the product of the other elements) -- the \
                 input shape alone, as given to `ReductionVjp::new`, is not enough. \
                 Construct this context with `ReductionVjp::with_input` or \
                 `ReductionVjp::with_input_axes` instead.",
                self.reduction_type
            );
        }

        let rule = ReductionRule::new(self.reduction_type.to_reduce_kind());
        let params = OpParams::axes(self.axes.clone());

        // Validate the axis set and compute the canonical (`keepdims = false`)
        // output shape -- the same layout `ReductionRule::vjp` itself builds
        // internally, reused here purely to translate `output_grad`'s shape
        // (this context's callers may pass a `keepdims = true`-shaped
        // gradient, e.g. `[1, 1]` for a full reduction, for
        // backward-compatibility with the pre-fix API).
        let layout = ReductionLayout::new(&self.input_shape, &self.axes, "ReductionVjp")?;
        let expected_shape = layout.output_shape();

        let squeezed_grad = if output_grad.shape() == expected_shape {
            output_grad.clone()
        } else {
            output_grad.reshape(expected_shape).map_err(|e| {
                anyhow!(
                    "ReductionVjp: output gradient shape {:?} ({} element(s)) cannot be \
                     reshaped to the reduced shape {:?} ({} element(s)): {e}",
                    output_grad.shape(),
                    output_grad.len(),
                    expected_shape,
                    expected_shape.iter().product::<usize>(),
                )
            })?
        };

        // `Sum`/`Mean` never read input values (only the shape-derived
        // layout), so a `new`-built context (no saved input) can still be
        // routed through the exact same math using a placeholder tensor.
        let owned_placeholder;
        let input_ref: &DenseND<T> = match &self.input {
            Some(input) => input,
            None => {
                owned_placeholder = DenseND::zeros(&self.input_shape);
                &owned_placeholder
            }
        };

        let mut grads = rule.vjp(std::slice::from_ref(input_ref), &squeezed_grad, &params)?;
        Ok(vec![grads.remove(0)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenrso_exec::ops::execute_dense_contraction;

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

    // ------------------------------------------------------------------
    // `ReductionVjp`: finite-difference verification of the fixed max/min
    // gradient bug, plus the non-leading / multi-axis fix.
    // ------------------------------------------------------------------

    use crate::gradcheck::{check_gradient, GradCheckConfig};

    /// Deterministic, tie-free probe values (no RNG, reproducible across
    /// runs).
    ///
    /// `Max`/`Min` gradcheck is only meaningful away from a tie: at a tie the
    /// reduction itself is non-differentiable (which of the tied elements
    /// "wins" is a modeling choice, not a limit), so a finite-difference
    /// check there would be comparing our analytical gradient against a
    /// numerically ill-defined quantity -- a coincidental pass would prove
    /// nothing.
    ///
    /// Element `i` is `offset + i` plus a bounded "wobble" of amplitude at
    /// most `0.2` (a deterministic function of `i` and `offset`, not a
    /// constant, so the sequence is not just a plain ramp). Since the integer
    /// part strictly increases by `1` per step and the wobble can move a
    /// value by at most `0.2` either way, any two *distinct* elements are
    /// separated by at least `1 - 2*0.2 = 0.6` -- five orders of magnitude
    /// above the `epsilon = 1e-5` central-difference step used by
    /// [`GradCheckConfig::default`], for *every* `shape`/`offset`. No
    /// perturbation performed by [`check_gradient`] can therefore ever flip
    /// which element is the arg-extremum of its group. The assertion below
    /// is a belt-and-suspenders check of that invariant, not a source of
    /// flakiness.
    fn probe_tensor(shape: &[usize], offset: f64) -> DenseND<f64> {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let wobble = 0.2 * ((i as f64) * 0.618_033_988_7 + offset).sin();
                offset + (i as f64) + wobble
            })
            .collect();

        for i in 0..data.len() {
            for j in (i + 1)..data.len() {
                assert!(
                    (data[i] - data[j]).abs() > 0.1,
                    "probe_tensor(offset={offset}) produced a near-tie at indices {i}/{j} \
                     ({} vs {}); adjust the generator constants",
                    data[i],
                    data[j]
                );
            }
        }

        DenseND::from_vec(data, shape).unwrap()
    }

    /// Gradcheck `ReductionVjp::with_input_axes(x, axes, kind)` against a
    /// central-difference numerical gradient of the *same* op's forward pass
    /// ([`ReductionRule::forward`], so forward and backward are unambiguously
    /// for the same reduction). Uses a non-uniform cotangent (also from
    /// [`probe_tensor`]) so a gradient that is only correct for an
    /// all-ones cotangent would be caught.
    fn check_reduction_vjp(x: &DenseND<f64>, axes: Vec<usize>, kind: ReductionType) {
        let rule = ReductionRule::new(kind.to_reduce_kind());
        let params = OpParams::axes(axes.clone());

        let forward = |probe: &DenseND<f64>| -> Result<DenseND<f64>> {
            rule.forward(std::slice::from_ref(probe), &params)
        };
        let backward = |probe: &DenseND<f64>, grad_y: &DenseND<f64>| -> Result<DenseND<f64>> {
            let ctx = ReductionVjp::with_input_axes(probe.clone(), axes.clone(), kind);
            let mut grads = ctx.vjp(grad_y)?;
            Ok(grads.remove(0))
        };

        let y = forward(x).expect("forward pass failed");
        let grad_y = probe_tensor(y.shape(), 12.3);

        let result = check_gradient(forward, backward, x, &grad_y, &GradCheckConfig::default())
            .unwrap_or_else(|e| panic!("gradcheck error for {kind:?} axes={axes:?}: {e:#}"));

        assert!(
            result.passed,
            "{kind:?} axes={axes:?}: gradient mismatch ({} / {} elements), \
             max_abs={:.3e}, max_rel={:.3e}",
            result.num_failures, result.num_elements, result.max_abs_diff, result.max_rel_diff
        );
    }

    #[test]
    fn gradcheck_reduction_vjp_sum_single_axis_leading_and_non_leading() {
        // A square shape so a broadcast along the *wrong* axis would not
        // even error -- exactly the historical silent-bug scenario.
        let x = probe_tensor(&[4, 4], 0.0);
        for axis in 0..2 {
            check_reduction_vjp(&x, vec![axis], ReductionType::Sum);
        }
    }

    #[test]
    fn gradcheck_reduction_vjp_mean_single_axis_leading_and_non_leading() {
        let x = probe_tensor(&[3, 5], 0.4);
        for axis in 0..2 {
            check_reduction_vjp(&x, vec![axis], ReductionType::Mean);
        }
    }

    #[test]
    fn gradcheck_reduction_vjp_max_single_axis_leading_and_non_leading() {
        let x = probe_tensor(&[4, 4], 0.9);
        for axis in 0..2 {
            check_reduction_vjp(&x, vec![axis], ReductionType::Max);
        }
    }

    #[test]
    fn gradcheck_reduction_vjp_min_single_axis_leading_and_non_leading() {
        let x = probe_tensor(&[4, 4], 1.7);
        for axis in 0..2 {
            check_reduction_vjp(&x, vec![axis], ReductionType::Min);
        }
    }

    #[test]
    fn gradcheck_reduction_vjp_product_single_axis_leading_and_non_leading() {
        let x = probe_tensor(&[3, 4], 2.3);
        for axis in 0..2 {
            check_reduction_vjp(&x, vec![axis], ReductionType::Product);
        }
    }

    #[test]
    fn gradcheck_reduction_vjp_multi_axis_and_full_reduction() {
        let x = probe_tensor(&[2, 3, 2], 3.1);
        for kind in [
            ReductionType::Sum,
            ReductionType::Mean,
            ReductionType::Max,
            ReductionType::Min,
            ReductionType::Product,
        ] {
            // Non-contiguous multi-axis: reduce axes 0 and 2, keep axis 1.
            check_reduction_vjp(&x, vec![0, 2], kind);
            // Full reduction: every axis at once.
            check_reduction_vjp(&x, vec![0, 1, 2], kind);
        }
    }

    #[test]
    fn reduction_vjp_new_fixes_non_leading_axis_regression() {
        // Regression test for the historical bug: the pre-fix `ReductionVjp`
        // broadcast a *squeezed* gradient back over the input shape using
        // `ndarray`'s positional broadcast rules without regard to which
        // axis was actually reduced. For a *square* input the bug was
        // silent -- broadcasting a shape-`[3]` gradient over a `[3, 3]`
        // target always succeeds -- but it stretched the gradient along the
        // wrong axis whenever the reduced axis was not the leading one.
        //
        // Sum-reduce axis 1 (trailing) of a 3x3 matrix using the legacy,
        // shape-only `new` constructor and check every element by hand:
        // d(sum_axis=1)/dx[i, j] = grad_out[i] for every j (broadcasts along
        // axis 1), never grad_out[j].
        let input_shape = vec![3, 3];
        let ctx = ReductionVjp::<f64>::new(input_shape.clone(), Some(1), ReductionType::Sum);

        let grad_out = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let grads = ctx.vjp(&grad_out).unwrap();

        assert_eq!(grads[0].shape(), input_shape.as_slice());
        for i in 0..3 {
            let expected = *grad_out.get(&[i]).unwrap();
            for j in 0..3 {
                assert_eq!(
                    *grads[0].get(&[i, j]).unwrap(),
                    expected,
                    "row {i} should broadcast grad_out[{i}] = {expected}, not grad_out[{j}]"
                );
            }
        }
    }

    #[test]
    fn reduction_vjp_max_zero_gradient_on_non_extremum() {
        // The core bug this task fixes: the old `ReductionVjp` broadcast the
        // upstream gradient to *every* element for `Max`/`Min`, which is only
        // correct for `Sum`. The correct gradient routes 100% of the
        // upstream gradient to the arg-max and exactly `0` everywhere else.
        let x = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 9.0, 4.0], &[2, 3]).unwrap();
        let ctx = ReductionVjp::with_input(x, Some(1), ReductionType::Max);

        // Row 0 = [1, 5, 3]: max 5.0 at column 1.
        // Row 1 = [2, 9, 4]: max 9.0 at column 1.
        let grad_out = DenseND::from_vec(vec![7.0, 11.0], &[2]).unwrap();
        let grads = ctx.vjp(&grad_out).unwrap();

        let expected = [[0.0, 7.0, 0.0], [0.0, 11.0, 0.0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &want) in row.iter().enumerate() {
                assert_eq!(
                    *grads[0].get(&[i, j]).unwrap(),
                    want,
                    "element [{i}, {j}] should be {want} (zero unless it is the row's arg-max)"
                );
            }
        }
    }

    #[test]
    fn reduction_vjp_new_errors_instead_of_wrong_gradient_for_max_min_product() {
        // `ReductionVjp::new` only has the input *shape*; routing a max/min
        // gradient to the arg-extremum (or scaling a product gradient by the
        // product of the other elements) needs the actual values. Rather
        // than silently broadcasting (the historical bug), `vjp` must return
        // an honest error pointing the caller at `with_input`.
        for kind in [
            ReductionType::Max,
            ReductionType::Min,
            ReductionType::Product,
        ] {
            let ctx = ReductionVjp::<f64>::new(vec![2, 3], None, kind);
            let grad_out = DenseND::from_elem(&[], 1.0);
            let err = ctx.vjp(&grad_out).unwrap_err();
            assert!(
                err.to_string().contains("with_input"),
                "{kind:?}: expected the error to point at `with_input`, got: {err}"
            );
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
