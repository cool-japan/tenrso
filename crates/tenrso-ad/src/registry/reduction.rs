//! Reduction rules (sum / mean / max / min / product).
//!
//! # Semantics
//!
//! `reduce(op, x, axes)` collapses every axis in `axes` and **removes** it from
//! the output shape (`keepdims = false`). This matches the Tensorlogic
//! `ReduceOp` contract (`ndarray::sum_axis` / `mean_axis` / `fold_axis`
//! repeatedly applied), so a Tensorlogic executor can be backed by these rules
//! directly. An empty `axes` list is the identity.
//!
//! # Relationship with [`crate::vjp::ReductionVjp`]
//!
//! [`crate::vjp::ReductionVjp`] is the historical, framework-internal entry
//! point for reduction gradients; this module is the canonical, input-aware
//! implementation. `ReductionVjp::with_input` / `with_input_axes` no longer
//! re-derive the math: they validate and translate `ReductionVjp`'s own
//! `axis: Option<usize>` (single-axis-or-full-reduction) convention into this
//! module's `axes: Vec<usize>` / `keepdims = false` convention (via a private
//! `ReductionLayout` helper, reused unchanged) and then call
//! [`ReductionRule::vjp`] directly. The legacy shape-only `ReductionVjp::new`
//! constructor is kept for `Sum`/`Mean` (whose gradient never reads input
//! values, so a shape-matched zero tensor can stand in for the real input) and
//! honestly errors for `Max`/`Min`/`Product`, which cannot be computed from a
//! shape alone.
//!
//! [`ReductionRule`] implements the full, input-aware VJP:
//!
//! | op | ∂L/∂x\[i\] |
//! |----|-----------|
//! | sum | `g[out(i)]` |
//! | mean | `g[out(i)] / n` |
//! | max / min | `g[out(i)]` if `i` is the **first** extremum of its group, else `0` |
//! | product | `g[out(i)] * Π_{j≠i, j∈group} x[j]` (computed without division, so zeros are exact) |
//!
//! Ties in `max`/`min` are broken deterministically towards the first element of
//! the group in row-major order (documented, reproducible, and consistent with
//! the sub-gradient convention used by mainstream AD frameworks).

use anyhow::{bail, Result};
use tenrso_core::DenseND;

use super::{linear_to_multi, row_major_strides, AdScalar, Arity, OpParams, OpRule};

/// Reduction kinds supported by the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    /// Sum over the reduced axes.
    Sum,
    /// Arithmetic mean over the reduced axes.
    Mean,
    /// Maximum over the reduced axes.
    Max,
    /// Minimum over the reduced axes.
    Min,
    /// Product over the reduced axes.
    Product,
}

impl ReduceKind {
    /// Every reduction kind, in registration order.
    pub const ALL: [ReduceKind; 5] = [
        ReduceKind::Sum,
        ReduceKind::Mean,
        ReduceKind::Max,
        ReduceKind::Min,
        ReduceKind::Product,
    ];

    /// Registry name of this op.
    pub fn name(&self) -> &'static str {
        match self {
            ReduceKind::Sum => "reduce_sum",
            ReduceKind::Mean => "reduce_mean",
            ReduceKind::Max => "reduce_max",
            ReduceKind::Min => "reduce_min",
            ReduceKind::Product => "reduce_product",
        }
    }

    /// Look a kind up by its registry name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// Identity element used to seed an accumulator.
    fn identity<T: AdScalar>(&self) -> T {
        match self {
            ReduceKind::Sum | ReduceKind::Mean => T::zero(),
            ReduceKind::Max => T::neg_infinity(),
            ReduceKind::Min => T::infinity(),
            ReduceKind::Product => T::one(),
        }
    }

    /// Fold one element into an accumulator.
    fn accumulate<T: AdScalar>(&self, acc: T, value: T) -> T {
        match self {
            ReduceKind::Sum | ReduceKind::Mean => acc + value,
            ReduceKind::Max => {
                if value > acc {
                    value
                } else {
                    acc
                }
            }
            ReduceKind::Min => {
                if value < acc {
                    value
                } else {
                    acc
                }
            }
            ReduceKind::Product => acc * value,
        }
    }
}

/// Registry rule for a reduction over an arbitrary axis set.
#[derive(Debug, Clone, Copy)]
pub struct ReductionRule {
    kind: ReduceKind,
}

impl ReductionRule {
    /// Create a rule for `kind`.
    pub fn new(kind: ReduceKind) -> Self {
        Self { kind }
    }

    /// The reduction this rule implements.
    pub fn kind(&self) -> ReduceKind {
        self.kind
    }
}

/// Pre-computed layout of a reduction: which axes collapse, what the output
/// shape is, and how an input linear index maps to an output linear index.
///
/// `pub(crate)` because [`crate::vjp::ReductionVjp`] reuses it (rather than
/// re-deriving the same axis validation and output-shape arithmetic) to
/// translate its own legacy `axis: Option<usize>` convention into the
/// `axes: Vec<usize>` shape this module expects before delegating to
/// [`ReductionRule::vjp`].
pub(crate) struct ReductionLayout {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    /// `true` for every reduced axis.
    reduced: Vec<bool>,
    /// Row-major strides of the output shape.
    output_strides: Vec<usize>,
    /// Number of input elements folded into each output element.
    group_size: usize,
}

impl ReductionLayout {
    pub(crate) fn new(input_shape: &[usize], axes: &[usize], op_name: &str) -> Result<Self> {
        let rank = input_shape.len();
        let mut reduced = vec![false; rank];

        for &axis in axes {
            if axis >= rank {
                bail!(
                    "'{op_name}': axis {axis} is out of bounds for a rank-{rank} tensor \
                     (shape {input_shape:?})"
                );
            }
            if reduced[axis] {
                bail!("'{op_name}': axis {axis} is listed more than once");
            }
            reduced[axis] = true;
        }

        let output_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| !reduced[*axis])
            .map(|(_, &size)| size)
            .collect();

        let group_size: usize = input_shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| reduced[*axis])
            .map(|(_, &size)| size)
            .product();

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_strides: row_major_strides(&output_shape),
            output_shape,
            reduced,
            group_size,
        })
    }

    /// Number of elements in the reduced output.
    fn output_len(&self) -> usize {
        self.output_shape.iter().product()
    }

    /// The (`keepdims = false`) output shape: `input_shape` with every
    /// reduced axis removed.
    pub(crate) fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Number of elements in the input.
    fn input_len(&self) -> usize {
        self.input_shape.iter().product()
    }

    /// Map an input linear index to the linear index of its output cell.
    fn output_index(&self, input_multi: &[usize]) -> usize {
        let mut out = 0usize;
        let mut out_axis = 0usize;
        for (axis, &idx) in input_multi.iter().enumerate() {
            if !self.reduced[axis] {
                out += idx * self.output_strides[out_axis];
                out_axis += 1;
            }
        }
        out
    }
}

impl<T> OpRule<T> for ReductionRule
where
    T: AdScalar,
{
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn arity(&self) -> Arity {
        Arity::Exact(1)
    }

    fn forward(&self, inputs: &[DenseND<T>], params: &OpParams) -> Result<DenseND<T>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let x = &inputs[0];

        if params.axes.is_empty() {
            // No axes given: identity, matching the Tensorlogic reduce contract.
            return Ok(x.clone());
        }

        let layout = ReductionLayout::new(x.shape(), &params.axes, self.kind.name())?;
        let mut acc = vec![self.kind.identity::<T>(); layout.output_len()];

        let data = x.as_array();
        let mut multi = vec![0usize; x.rank()];
        for (linear, &value) in data.iter().enumerate() {
            linear_to_multi(linear, &layout.input_shape, &mut multi);
            let out = layout.output_index(&multi);
            acc[out] = self.kind.accumulate(acc[out], value);
        }

        if self.kind == ReduceKind::Mean {
            let count = T::from_usize(layout.group_size)
                .ok_or_else(|| anyhow::anyhow!("cannot represent group size as scalar"))?;
            for value in acc.iter_mut() {
                *value = *value / count;
            }
        }

        DenseND::from_vec(acc, &layout.output_shape)
    }

    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let x = &inputs[0];

        if params.axes.is_empty() {
            if output_grad.shape() != x.shape() {
                bail!(
                    "'{}': identity reduction expects a gradient of shape {:?}, got {:?}",
                    self.kind.name(),
                    x.shape(),
                    output_grad.shape()
                );
            }
            return Ok(vec![output_grad.clone()]);
        }

        let layout = ReductionLayout::new(x.shape(), &params.axes, self.kind.name())?;
        if output_grad.shape() != layout.output_shape.as_slice() {
            bail!(
                "'{}': output gradient shape {:?} does not match the reduced shape {:?}",
                self.kind.name(),
                output_grad.shape(),
                layout.output_shape
            );
        }

        let grad_out = output_grad.as_array().as_slice().map(<[T]>::to_vec);
        let grad_out = match grad_out {
            Some(values) => values,
            None => output_grad.as_array().iter().copied().collect(),
        };

        let input_values: Vec<T> = x.as_array().iter().copied().collect();
        let mut grad_in = vec![T::zero(); layout.input_len()];
        let mut multi = vec![0usize; x.rank()];

        match self.kind {
            ReduceKind::Sum => {
                for (linear, grad) in grad_in.iter_mut().enumerate() {
                    linear_to_multi(linear, &layout.input_shape, &mut multi);
                    *grad = grad_out[layout.output_index(&multi)];
                }
            }
            ReduceKind::Mean => {
                let count = T::from_usize(layout.group_size)
                    .ok_or_else(|| anyhow::anyhow!("cannot represent group size as scalar"))?;
                for (linear, grad) in grad_in.iter_mut().enumerate() {
                    linear_to_multi(linear, &layout.input_shape, &mut multi);
                    *grad = grad_out[layout.output_index(&multi)] / count;
                }
            }
            ReduceKind::Max | ReduceKind::Min => {
                // First pass: the extremum of every group.
                let mut extremum = vec![self.kind.identity::<T>(); layout.output_len()];
                for (linear, value) in input_values.iter().enumerate() {
                    linear_to_multi(linear, &layout.input_shape, &mut multi);
                    let out = layout.output_index(&multi);
                    extremum[out] = self.kind.accumulate(extremum[out], *value);
                }

                // Second pass: route the gradient to the first extremum element
                // of each group (deterministic tie-breaking, row-major order).
                let mut claimed = vec![false; layout.output_len()];
                for (linear, value) in input_values.iter().enumerate() {
                    linear_to_multi(linear, &layout.input_shape, &mut multi);
                    let out = layout.output_index(&multi);
                    if !claimed[out] && *value == extremum[out] {
                        claimed[out] = true;
                        grad_in[linear] = grad_out[out];
                    }
                }
            }
            ReduceKind::Product => {
                // Product of the non-zero elements of each group, plus a zero
                // count. This yields the exact "product of the others" without
                // ever dividing by zero.
                let mut nonzero_product = vec![T::one(); layout.output_len()];
                let mut zero_count = vec![0usize; layout.output_len()];

                for (linear, value) in input_values.iter().enumerate() {
                    linear_to_multi(linear, &layout.input_shape, &mut multi);
                    let out = layout.output_index(&multi);
                    if *value == T::zero() {
                        zero_count[out] += 1;
                    } else {
                        nonzero_product[out] = nonzero_product[out] * *value;
                    }
                }

                for (linear, value) in input_values.iter().enumerate() {
                    linear_to_multi(linear, &layout.input_shape, &mut multi);
                    let out = layout.output_index(&multi);

                    let others = match zero_count[out] {
                        // No zeros: product of the others = total / x_i.
                        0 => nonzero_product[out] / *value,
                        // Exactly one zero: only that element has a non-zero
                        // partial derivative.
                        1 => {
                            if *value == T::zero() {
                                nonzero_product[out]
                            } else {
                                T::zero()
                            }
                        }
                        // Two or more zeros: every partial derivative vanishes.
                        _ => T::zero(),
                    };

                    grad_in[linear] = grad_out[out] * others;
                }
            }
        }

        Ok(vec![DenseND::from_vec(grad_in, &layout.input_shape)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_2x3() -> DenseND<f64> {
        DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap()
    }

    #[test]
    fn test_reduce_names() {
        for kind in ReduceKind::ALL {
            assert_eq!(ReduceKind::from_name(kind.name()), Some(kind));
        }
    }

    #[test]
    fn test_sum_forward_leading_axis() {
        let rule = ReductionRule::new(ReduceKind::Sum);
        let y = rule
            .forward(&[tensor_2x3()], &OpParams::axes(vec![0]))
            .unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_forward_trailing_axis() {
        // This is the case `ReductionVjp` cannot broadcast back.
        let rule = ReductionRule::new(ReduceKind::Sum);
        let y = rule
            .forward(&[tensor_2x3()], &OpParams::axes(vec![1]))
            .unwrap();
        assert_eq!(y.shape(), &[2]);
        assert_eq!(y.as_slice(), &[6.0, 15.0]);

        let grad = DenseND::from_vec(vec![2.0, 3.0], &[2]).unwrap();
        let grads = rule
            .vjp(&[tensor_2x3()], &grad, &OpParams::axes(vec![1]))
            .unwrap();
        assert_eq!(grads[0].shape(), &[2, 3]);
        assert_eq!(grads[0].as_slice(), &[2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_full_reduction_all_axes() {
        let rule = ReductionRule::new(ReduceKind::Mean);
        let y = rule
            .forward(&[tensor_2x3()], &OpParams::axes(vec![0, 1]))
            .unwrap();
        assert!(y.shape().is_empty());
        assert!((y.as_slice()[0] - 3.5).abs() < 1e-12);

        let grad = DenseND::from_elem(&[], 6.0);
        let grads = rule
            .vjp(&[tensor_2x3()], &grad, &OpParams::axes(vec![0, 1]))
            .unwrap();
        // 6 / 6 elements = 1 everywhere.
        assert!(grads[0].as_slice().iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_max_routes_gradient_to_argmax_only() {
        let rule = ReductionRule::new(ReduceKind::Max);
        let x = tensor_2x3();
        let y = rule
            .forward(std::slice::from_ref(&x), &OpParams::axes(vec![1]))
            .unwrap();
        assert_eq!(y.as_slice(), &[3.0, 6.0]);

        let grad = DenseND::from_vec(vec![1.0, 10.0], &[2]).unwrap();
        let grads = rule.vjp(&[x], &grad, &OpParams::axes(vec![1])).unwrap();
        assert_eq!(grads[0].as_slice(), &[0.0, 0.0, 1.0, 0.0, 0.0, 10.0]);
    }

    #[test]
    fn test_min_ties_go_to_first_element() {
        let rule = ReductionRule::new(ReduceKind::Min);
        let x = DenseND::from_vec(vec![2.0, 2.0, 5.0], &[3]).unwrap();
        let grad = DenseND::from_elem(&[], 1.0);
        let grads = rule.vjp(&[x], &grad, &OpParams::axes(vec![0])).unwrap();
        assert_eq!(grads[0].as_slice(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_product_handles_zeros_exactly() {
        let rule = ReductionRule::new(ReduceKind::Product);
        let params = OpParams::axes(vec![0]);

        // One zero: only the zero element has a non-zero derivative (2*4 = 8).
        let x = DenseND::from_vec(vec![2.0, 0.0, 4.0], &[3]).unwrap();
        let y = rule.forward(std::slice::from_ref(&x), &params).unwrap();
        assert_eq!(y.as_slice(), &[0.0]);

        let grad = DenseND::from_elem(&[], 1.0);
        let grads = rule.vjp(&[x], &grad, &params).unwrap();
        assert_eq!(grads[0].as_slice(), &[0.0, 8.0, 0.0]);

        // Two zeros: every derivative vanishes.
        let x = DenseND::from_vec(vec![0.0, 0.0, 4.0], &[3]).unwrap();
        let grads = rule.vjp(&[x], &grad, &params).unwrap();
        assert_eq!(grads[0].as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_empty_axes_is_identity() {
        let rule = ReductionRule::new(ReduceKind::Sum);
        let x = tensor_2x3();
        let y = rule
            .forward(std::slice::from_ref(&x), &OpParams::none())
            .unwrap();
        assert_eq!(y.shape(), x.shape());
        assert_eq!(y.as_slice(), x.as_slice());
    }

    #[test]
    fn test_axis_validation() {
        let rule = ReductionRule::new(ReduceKind::Sum);
        let x = tensor_2x3();
        assert!(rule
            .forward(std::slice::from_ref(&x), &OpParams::axes(vec![5]))
            .is_err());
        assert!(rule.forward(&[x], &OpParams::axes(vec![0, 0])).is_err());
    }
}
