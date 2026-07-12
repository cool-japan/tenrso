//! Execution of a [`FusionPlan`]: forward kernels and reverse-mode VJPs.
//!
//! # Fused kernels
//!
//! Each fused kernel writes exactly **one** buffer. `MatMulBiasReLU`, for
//! instance, runs the GEMM into its output buffer and then applies bias and
//! ReLU **in place, in a single pass** — where the unfused graph allocates and
//! traverses three buffers (`A@B`, `+bias`, `relu`).
//!
//! # Fused VJPs
//!
//! No fused VJP needs an interior value:
//!
//! - `ReLU`'s mask is recovered from the fused **output**: for `y = ReLU(p)`,
//!   `p > 0 ⟺ y > 0` (and `p ≤ 0 ⟹ y = 0`), so the mask is `y > 0`. This is
//!   what makes eliminating the pre-activation buffer sound, in the backward
//!   pass as well as the forward one.
//! - The matmul VJPs need only the live operands `A` and `B`.
//! - The bias VJP is the output gradient summed back over whichever axes the
//!   bias was broadcast along (`unbroadcast`).
//!
//! # Broadcasting
//!
//! Elementwise binary ops broadcast their operands (as `ComputationGraph`'s
//! do, since both use ndarray's co-broadcasting `+`/`*` operators). Their VJPs
//! therefore *un*-broadcast: an operand gradient is summed over every axis the
//! operand was expanded along, so it always comes back with the operand's own
//! shape. `bias: [n]` against `A@B: [m, n]` is the common case.

use super::fusion::{FusedOperation, FusionPlan, PlanStep};
use crate::graph::{NodeId, Operation};
use anyhow::{anyhow, Context, Result};
use scirs2_core::ndarray_ext::{ArrayD, ArrayViewD, Axis, Ix2, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;

/// Values produced by one forward execution of a [`FusionPlan`].
///
/// Borrows the feed map, so leaf tensors are never copied: everything counted
/// by [`PlanExecution::buffers_allocated`] is a tensor the plan itself
/// materialized.
#[derive(Debug)]
pub struct PlanExecution<'f, T> {
    feeds: &'f HashMap<NodeId, ArrayD<T>>,
    computed: HashMap<NodeId, ArrayD<T>>,
    buffers_allocated: usize,
    elements_allocated: usize,
}

impl<T> PlanExecution<'_, T> {
    /// Value of any leaf or step node.
    pub fn value(&self, id: NodeId) -> Result<&ArrayD<T>> {
        self.computed
            .get(&id)
            .or_else(|| self.feeds.get(&id))
            .ok_or_else(|| anyhow!("PlanExecution: node {} was not computed by this plan", id))
    }

    /// Number of intermediate tensors this forward pass materialized.
    ///
    /// Exactly one per plan step, so a fused plan's count is lower than the
    /// unfused plan's by the number of interior nodes the fusions eliminated.
    /// This is a measurement, not an estimate.
    pub fn buffers_allocated(&self) -> usize {
        self.buffers_allocated
    }

    /// Total element count across the materialized intermediates.
    pub fn elements_allocated(&self) -> usize {
        self.elements_allocated
    }
}

/// Gradients keyed by node.
pub type GradientMap<T> = HashMap<NodeId, ArrayD<T>>;

impl FusionPlan {
    /// Execute the plan forward.
    ///
    /// `feeds` must contain a value for every leaf; see
    /// [`FusionPlan::leaf_ids`]. Feeding fresh values re-executes the compiled
    /// graph, which is the point of compiling it: fusion cannot speed up the
    /// eager pass that already ran.
    ///
    /// # Complexity
    ///
    /// One pass per step; identical FLOPs to the unfused graph, minus the
    /// eliminated buffer traffic.
    pub fn forward<'f, T>(
        &self,
        feeds: &'f HashMap<NodeId, ArrayD<T>>,
    ) -> Result<PlanExecution<'f, T>>
    where
        T: Float + ScalarOperand + FromPrimitive,
    {
        for leaf in &self.leaves {
            if !feeds.contains_key(&leaf.id) {
                return Err(anyhow!(
                    "FusionPlan::forward: no feed value for leaf {}; every leaf returned by \
                     `leaf_ids()` must be fed",
                    leaf.id
                ));
            }
        }

        let mut exec = PlanExecution {
            feeds,
            computed: HashMap::with_capacity(self.steps.len()),
            buffers_allocated: 0,
            elements_allocated: 0,
        };

        for step in &self.steps {
            let out = match step {
                PlanStep::Base { op, .. } => eval_base(op, &exec)?,
                PlanStep::Fused { op, .. } => eval_fused(op, &exec)?,
            };
            exec.buffers_allocated += 1;
            exec.elements_allocated += out.len();
            exec.computed.insert(step.id(), out);
        }

        Ok(exec)
    }

    /// Reverse-mode differentiation of the plan.
    ///
    /// Seeds `output` with `seed` (which must have `output`'s shape) and walks
    /// the steps in reverse, accumulating gradients into every node that
    /// transitively depends on a `requires_grad` leaf. Fused steps use fused
    /// VJPs, so no interior gradient buffer is allocated either.
    ///
    /// The returned map contains gradients for leaves *and* intermediates;
    /// [`FusionPlan::requires_grad_leaves`] lists the ones a caller normally
    /// wants.
    pub fn backward<T>(
        &self,
        exec: &PlanExecution<'_, T>,
        output: NodeId,
        seed: &ArrayD<T>,
    ) -> Result<GradientMap<T>>
    where
        T: Float + ScalarOperand + FromPrimitive,
    {
        let out_value = exec.value(output)?;
        if out_value.shape() != seed.shape() {
            return Err(anyhow!(
                "FusionPlan::backward: seed shape {:?} does not match output {} shape {:?}",
                seed.shape(),
                output,
                out_value.shape()
            ));
        }
        if !self.needs_grad.contains(&output) {
            return Err(anyhow!(
                "FusionPlan::backward: output {} does not depend on any leaf with requires_grad",
                output
            ));
        }

        let mut grads: GradientMap<T> = HashMap::new();
        grads.insert(output, seed.clone());

        for step in self.steps.iter().rev() {
            let id = step.id();
            if !self.needs_grad.contains(&id) {
                continue;
            }
            let grad_out = match grads.get(&id) {
                Some(g) => g.clone(),
                // Not on the path between `output` and the leaves.
                None => continue,
            };

            let parent_grads = match step {
                PlanStep::Base { op, .. } => backward_base(op, exec, &grad_out)?,
                PlanStep::Fused { op, .. } => backward_fused(op, exec, &grad_out, id)?,
            };

            for (parent, grad) in parent_grads {
                if !self.needs_grad.contains(&parent) {
                    continue;
                }
                accumulate(&mut grads, parent, grad)?;
            }
        }

        Ok(grads)
    }
}

fn accumulate<T: Float>(grads: &mut GradientMap<T>, id: NodeId, grad: ArrayD<T>) -> Result<()> {
    match grads.get_mut(&id) {
        Some(existing) => {
            if existing.shape() != grad.shape() {
                return Err(anyhow!(
                    "gradient accumulation shape mismatch at {}: {:?} vs {:?}",
                    id,
                    existing.shape(),
                    grad.shape()
                ));
            }
            *existing = &*existing + &grad;
        }
        None => {
            grads.insert(id, grad);
        }
    }
    Ok(())
}

// ============================================================================
// Shape helpers
// ============================================================================

/// NumPy broadcasting: the shape `a` and `b` broadcast to.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for (i, dim) in out.iter_mut().enumerate() {
        // Right-align both shapes; missing leading axes are implicitly 1.
        let da = if i + a.len() >= n {
            a[i + a.len() - n]
        } else {
            1
        };
        let db = if i + b.len() >= n {
            b[i + b.len() - n]
        } else {
            1
        };
        *dim = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(anyhow!("shapes {:?} and {:?} do not broadcast", a, b));
        };
    }
    Ok(out)
}

/// Sum `grad` back down to `target`, undoing a broadcast.
///
/// Every axis that was expanded (size 1 in the operand, > 1 in the result) and
/// every leading axis that the operand did not have is summed over. This is
/// the VJP of broadcasting, and it is what gives a bias of shape `[n]` a
/// gradient of shape `[n]` rather than `[m, n]`.
pub(crate) fn unbroadcast<T: Float>(grad: ArrayD<T>, target: &[usize]) -> Result<ArrayD<T>> {
    if grad.shape() == target {
        return Ok(grad);
    }
    let ndim_out = grad.ndim();
    if target.len() > ndim_out {
        return Err(anyhow!(
            "cannot unbroadcast {:?} down to {:?}",
            grad.shape(),
            target
        ));
    }
    let pad = ndim_out - target.len();
    let padded: Vec<usize> = std::iter::repeat_n(1usize, pad)
        .chain(target.iter().copied())
        .collect();
    let out_shape = grad.shape().to_vec();

    let mut g = grad;
    for ax in 0..ndim_out {
        if padded[ax] == 1 && out_shape[ax] > 1 {
            let summed = g.sum_axis(Axis(ax));
            let mut keepdim = summed.shape().to_vec();
            keepdim.insert(ax, 1);
            g = summed
                .to_shape(IxDyn(&keepdim))
                .context("unbroadcast: keepdim reshape failed")?
                .to_owned();
        } else if padded[ax] != out_shape[ax] {
            return Err(anyhow!(
                "cannot unbroadcast {:?} down to {:?}: axis {} mismatches",
                out_shape,
                target,
                ax
            ));
        }
    }
    Ok(g.to_shape(IxDyn(target))
        .context("unbroadcast: final reshape failed")?
        .to_owned())
}

/// Zero-copy broadcast view of `v` at `shape`.
fn view_as<'a, T>(v: &'a ArrayD<T>, shape: &[usize]) -> Result<ArrayViewD<'a, T>> {
    v.broadcast(IxDyn(shape))
        .ok_or_else(|| anyhow!("cannot broadcast {:?} to {:?}", v.shape(), shape))
}

// ============================================================================
// Fused kernels (forward)
// ============================================================================

fn eval_fused<T>(op: &FusedOperation, exec: &PlanExecution<'_, T>) -> Result<ArrayD<T>>
where
    T: Float + ScalarOperand + FromPrimitive,
{
    match op {
        FusedOperation::MatMulBias { lhs, rhs, bias } => {
            matmul_bias_relu(exec, *lhs, *rhs, *bias, false)
        }
        FusedOperation::MatMulBiasReLU { lhs, rhs, bias } => {
            matmul_bias_relu(exec, *lhs, *rhs, *bias, true)
        }
        FusedOperation::MulAdd { x, y, c } => {
            let xv = exec.value(*x)?;
            let yv = exec.value(*y)?;
            let cv = exec.value(*c)?;
            let shape = broadcast_shape(&broadcast_shape(xv.shape(), yv.shape())?, cv.shape())?;
            let xb = view_as(xv, &shape)?;
            let yb = view_as(yv, &shape)?;
            let cb = view_as(cv, &shape)?;
            // One buffer, one pass: x * y + c.
            Ok(Zip::from(&xb)
                .and(&yb)
                .and(&cb)
                .map_collect(|&a, &b, &c| a * b + c))
        }
        FusedOperation::AddReLU { lhs, rhs } => {
            let lv = exec.value(*lhs)?;
            let rv = exec.value(*rhs)?;
            let shape = broadcast_shape(lv.shape(), rv.shape())?;
            let lb = view_as(lv, &shape)?;
            let rb = view_as(rv, &shape)?;
            // One buffer, one pass: relu(x + y).
            Ok(Zip::from(&lb).and(&rb).map_collect(|&a, &b| {
                let v = a + b;
                if v > T::zero() {
                    v
                } else {
                    T::zero()
                }
            }))
        }
    }
}

/// `A @ B + bias`, optionally with ReLU, into a single buffer.
///
/// The GEMM output *is* the result buffer: bias and ReLU are applied in place
/// in one pass over it. The unfused chain allocates and traverses two more
/// buffers of the same size.
fn matmul_bias_relu<T>(
    exec: &PlanExecution<'_, T>,
    lhs: NodeId,
    rhs: NodeId,
    bias: NodeId,
    relu: bool,
) -> Result<ArrayD<T>>
where
    T: Float + ScalarOperand + FromPrimitive,
{
    let lv = exec.value(lhs)?;
    let rv = exec.value(rhs)?;
    let bv = exec.value(bias)?;

    if lv.ndim() != 2 || rv.ndim() != 2 {
        return Err(anyhow!(
            "fused matmul supports 2-D operands only, got {:?} and {:?}",
            lv.shape(),
            rv.shape()
        ));
    }
    let a = lv.view().into_dimensionality::<Ix2>()?;
    let b = rv.view().into_dimensionality::<Ix2>()?;

    // The one and only buffer this kernel allocates.
    let mut out = a.dot(&b);

    let out_shape = out.shape().to_vec();
    let bias_view = view_as(bv, &out_shape)?;
    let bias_2d = bias_view.into_dimensionality::<Ix2>()?;

    if relu {
        Zip::from(&mut out).and(&bias_2d).for_each(|o, &bias_v| {
            let v = *o + bias_v;
            *o = if v > T::zero() { v } else { T::zero() };
        });
    } else {
        Zip::from(&mut out).and(&bias_2d).for_each(|o, &bias_v| {
            *o = *o + bias_v;
        });
    }

    Ok(out.into_dyn())
}

// ============================================================================
// Fused kernels (backward)
// ============================================================================

fn backward_fused<T>(
    op: &FusedOperation,
    exec: &PlanExecution<'_, T>,
    grad_out: &ArrayD<T>,
    out_id: NodeId,
) -> Result<Vec<(NodeId, ArrayD<T>)>>
where
    T: Float + ScalarOperand + FromPrimitive,
{
    match op {
        FusedOperation::MatMulBias { lhs, rhs, bias } => {
            matmul_bias_backward(exec, *lhs, *rhs, *bias, grad_out.clone())
        }
        FusedOperation::MatMulBiasReLU { lhs, rhs, bias } => {
            // ReLU mask straight from the fused output: y > 0 ⟺ pre-activation
            // > 0. The pre-activation buffer never existed, and never needs to.
            let y = exec.value(out_id)?;
            let masked = relu_mask_apply(grad_out, y)?;
            matmul_bias_backward(exec, *lhs, *rhs, *bias, masked)
        }
        FusedOperation::MulAdd { x, y, c } => {
            let xv = exec.value(*x)?;
            let yv = exec.value(*y)?;
            let cv = exec.value(*c)?;
            let shape = grad_out.shape().to_vec();
            let xb = view_as(xv, &shape)?;
            let yb = view_as(yv, &shape)?;
            let grad_x = Zip::from(grad_out).and(&yb).map_collect(|&g, &yy| g * yy);
            let grad_y = Zip::from(grad_out).and(&xb).map_collect(|&g, &xx| g * xx);
            Ok(vec![
                (*x, unbroadcast(grad_x, xv.shape())?),
                (*y, unbroadcast(grad_y, yv.shape())?),
                (*c, unbroadcast(grad_out.clone(), cv.shape())?),
            ])
        }
        FusedOperation::AddReLU { lhs, rhs } => {
            let out = exec.value(out_id)?;
            let masked = relu_mask_apply(grad_out, out)?;
            let lv = exec.value(*lhs)?;
            let rv = exec.value(*rhs)?;
            Ok(vec![
                (*lhs, unbroadcast(masked.clone(), lv.shape())?),
                (*rhs, unbroadcast(masked, rv.shape())?),
            ])
        }
    }
}

/// `grad * (y > 0)` where `y` is a ReLU output.
fn relu_mask_apply<T: Float>(grad: &ArrayD<T>, y: &ArrayD<T>) -> Result<ArrayD<T>> {
    if grad.shape() != y.shape() {
        return Err(anyhow!(
            "relu mask shape mismatch: {:?} vs {:?}",
            grad.shape(),
            y.shape()
        ));
    }
    Ok(Zip::from(grad)
        .and(y)
        .map_collect(|&g, &out| if out > T::zero() { g } else { T::zero() }))
}

/// VJP of `A @ B + bias` given the (already ReLU-masked, if applicable) output
/// gradient. Needs no interior value.
fn matmul_bias_backward<T>(
    exec: &PlanExecution<'_, T>,
    lhs: NodeId,
    rhs: NodeId,
    bias: NodeId,
    grad_out: ArrayD<T>,
) -> Result<Vec<(NodeId, ArrayD<T>)>>
where
    T: Float + ScalarOperand + FromPrimitive,
{
    let lv = exec.value(lhs)?;
    let rv = exec.value(rhs)?;
    let bv = exec.value(bias)?;

    let g = grad_out.view().into_dimensionality::<Ix2>()?;
    let a = lv.view().into_dimensionality::<Ix2>()?;
    let b = rv.view().into_dimensionality::<Ix2>()?;

    let grad_lhs = g.dot(&b.t()).into_dyn();
    let grad_rhs = a.t().dot(&g).into_dyn();
    let grad_bias = unbroadcast(grad_out, bv.shape())?;

    Ok(vec![(lhs, grad_lhs), (rhs, grad_rhs), (bias, grad_bias)])
}

// ============================================================================
// Base operations (forward)
// ============================================================================

fn eval_base<T>(op: &Operation, exec: &PlanExecution<'_, T>) -> Result<ArrayD<T>>
where
    T: Float + ScalarOperand + FromPrimitive,
{
    let v = |id: NodeId| exec.value(id);
    match op {
        Operation::Input => Err(anyhow!(
            "eval_base: Operation::Input is a leaf and must come from the feed map"
        )),
        Operation::Add { lhs, rhs } => Ok(v(*lhs)? + v(*rhs)?),
        Operation::Sub { lhs, rhs } => Ok(v(*lhs)? - v(*rhs)?),
        Operation::Mul { lhs, rhs } => Ok(v(*lhs)? * v(*rhs)?),
        Operation::Div { lhs, rhs } => Ok(v(*lhs)? / v(*rhs)?),
        Operation::MatMul { lhs, rhs } => {
            let a = v(*lhs)?;
            let b = v(*rhs)?;
            if a.ndim() != 2 || b.ndim() != 2 {
                return Err(anyhow!(
                    "MatMul supports 2-D operands only, got {:?} and {:?}",
                    a.shape(),
                    b.shape()
                ));
            }
            let a2 = a.view().into_dimensionality::<Ix2>()?;
            let b2 = b.view().into_dimensionality::<Ix2>()?;
            Ok(a2.dot(&b2).into_dyn())
        }
        Operation::Neg { input } => Ok(v(*input)?.mapv(|x| -x)),
        Operation::Exp { input } => Ok(v(*input)?.mapv(|x| x.exp())),
        Operation::Log { input } => Ok(v(*input)?.mapv(|x| x.ln())),
        Operation::Pow { input, exponent } => {
            let e = T::from(*exponent).ok_or_else(|| anyhow!("Pow: cannot convert exponent"))?;
            Ok(v(*input)?.mapv(|x| x.powf(e)))
        }
        Operation::Sum { input, axis } => {
            let x = v(*input)?;
            Ok(match axis {
                Some(ax) => x.sum_axis(Axis(*ax)),
                None => {
                    let s = x.iter().fold(T::zero(), |acc, &e| acc + e);
                    ArrayD::from_elem(IxDyn(&[]), s)
                }
            })
        }
        Operation::Mean { input, axis } => {
            let x = v(*input)?;
            Ok(match axis {
                Some(ax) => x
                    .mean_axis(Axis(*ax))
                    .ok_or_else(|| anyhow!("Mean: empty axis {}", ax))?,
                None => {
                    let s = x.iter().fold(T::zero(), |acc, &e| acc + e);
                    let n = T::from(x.len()).ok_or_else(|| anyhow!("Mean: cannot convert len"))?;
                    ArrayD::from_elem(IxDyn(&[]), s / n)
                }
            })
        }
        Operation::ReLU { input } => {
            Ok(v(*input)?.mapv(|x| if x > T::zero() { x } else { T::zero() }))
        }
        Operation::Sigmoid { input } => Ok(v(*input)?.mapv(|x| T::one() / (T::one() + (-x).exp()))),
        Operation::Tanh { input } => Ok(v(*input)?.mapv(|x| x.tanh())),
        Operation::Transpose { input, axes } => {
            Ok(v(*input)?.view().permuted_axes(IxDyn(axes)).to_owned())
        }
        Operation::Slice { input, ranges } => slice_forward(v(*input)?, ranges),
        Operation::Reshape { .. } | Operation::Broadcast { .. } => Err(anyhow!(
            "eval_base: Reshape/Broadcast are rejected at compile time and cannot appear in a plan"
        )),
    }
}

fn slice_forward<T: Float>(x: &ArrayD<T>, ranges: &[(usize, usize)]) -> Result<ArrayD<T>> {
    let ndim = x.ndim();
    if ranges.len() != ndim {
        return Err(anyhow!(
            "Slice: {} ranges for a {}-D tensor",
            ranges.len(),
            ndim
        ));
    }
    let shape: Vec<usize> = ranges.iter().map(|&(s, e)| e - s).collect();
    let flat: usize = shape.iter().product();
    let mut out = Vec::with_capacity(flat);
    for lin in 0..flat {
        let idx = unravel(lin, &shape);
        let src: Vec<usize> = idx
            .iter()
            .enumerate()
            .map(|(ax, &i)| ranges[ax].0 + i)
            .collect();
        out.push(x[src.as_slice()]);
    }
    ArrayD::from_shape_vec(IxDyn(&shape), out).context("Slice: output construction failed")
}

fn unravel(mut lin: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        if shape[d] > 0 {
            idx[d] = lin % shape[d];
            lin /= shape[d];
        }
    }
    idx
}

// ============================================================================
// Base operations (backward)
// ============================================================================

fn backward_base<T>(
    op: &Operation,
    exec: &PlanExecution<'_, T>,
    grad_out: &ArrayD<T>,
) -> Result<Vec<(NodeId, ArrayD<T>)>>
where
    T: Float + ScalarOperand + FromPrimitive,
{
    let v = |id: NodeId| exec.value(id);
    match op {
        Operation::Input => Ok(vec![]),

        Operation::Add { lhs, rhs } => Ok(vec![
            (*lhs, unbroadcast(grad_out.clone(), v(*lhs)?.shape())?),
            (*rhs, unbroadcast(grad_out.clone(), v(*rhs)?.shape())?),
        ]),

        Operation::Sub { lhs, rhs } => Ok(vec![
            (*lhs, unbroadcast(grad_out.clone(), v(*lhs)?.shape())?),
            (*rhs, unbroadcast(grad_out.mapv(|g| -g), v(*rhs)?.shape())?),
        ]),

        Operation::Mul { lhs, rhs } => {
            let a = v(*lhs)?;
            let b = v(*rhs)?;
            Ok(vec![
                (*lhs, unbroadcast(grad_out * b, a.shape())?),
                (*rhs, unbroadcast(grad_out * a, b.shape())?),
            ])
        }

        Operation::Div { lhs, rhs } => {
            let a = v(*lhs)?;
            let b = v(*rhs)?;
            let grad_lhs = grad_out / b;
            let grad_rhs = -(grad_out * a) / &(b * b);
            Ok(vec![
                (*lhs, unbroadcast(grad_lhs, a.shape())?),
                (*rhs, unbroadcast(grad_rhs, b.shape())?),
            ])
        }

        Operation::MatMul { lhs, rhs } => {
            let a = v(*lhs)?;
            let b = v(*rhs)?;
            let g = grad_out.view().into_dimensionality::<Ix2>()?;
            let a2 = a.view().into_dimensionality::<Ix2>()?;
            let b2 = b.view().into_dimensionality::<Ix2>()?;
            Ok(vec![
                (*lhs, g.dot(&b2.t()).into_dyn()),
                (*rhs, a2.t().dot(&g).into_dyn()),
            ])
        }

        Operation::Neg { input } => Ok(vec![(*input, grad_out.mapv(|g| -g))]),

        Operation::Exp { input } => {
            let x = v(*input)?;
            Ok(vec![(*input, grad_out * &x.mapv(|e| e.exp()))])
        }

        Operation::Log { input } => {
            let x = v(*input)?;
            Ok(vec![(*input, grad_out / x)])
        }

        Operation::Pow { input, exponent } => {
            let x = v(*input)?;
            let n = T::from(*exponent).ok_or_else(|| anyhow!("Pow: cannot convert exponent"))?;
            let n1 = T::from(exponent - 1.0)
                .ok_or_else(|| anyhow!("Pow: cannot convert exponent - 1"))?;
            Ok(vec![(*input, grad_out * &x.mapv(|e| n * e.powf(n1)))])
        }

        Operation::Sum { input, axis } => {
            let shape = v(*input)?.shape().to_vec();
            Ok(vec![(
                *input,
                expand_reduction(grad_out, &shape, *axis, T::one())?,
            )])
        }

        Operation::Mean { input, axis } => {
            let shape = v(*input)?.shape().to_vec();
            let n = match axis {
                Some(ax) => shape[*ax],
                None => shape.iter().product::<usize>().max(1),
            };
            let scale = T::one()
                / T::from(n).ok_or_else(|| anyhow!("Mean backward: cannot convert count"))?;
            Ok(vec![(
                *input,
                expand_reduction(grad_out, &shape, *axis, scale)?,
            )])
        }

        Operation::ReLU { input } => {
            let x = v(*input)?;
            Ok(vec![(
                *input,
                Zip::from(grad_out).and(x).map_collect(
                    |&g, &e| {
                        if e > T::zero() {
                            g
                        } else {
                            T::zero()
                        }
                    },
                ),
            )])
        }

        Operation::Sigmoid { input } => {
            let x = v(*input)?;
            Ok(vec![(
                *input,
                Zip::from(grad_out).and(x).map_collect(|&g, &e| {
                    let s = T::one() / (T::one() + (-e).exp());
                    g * s * (T::one() - s)
                }),
            )])
        }

        Operation::Tanh { input } => {
            let x = v(*input)?;
            Ok(vec![(
                *input,
                Zip::from(grad_out).and(x).map_collect(|&g, &e| {
                    let t = e.tanh();
                    g * (T::one() - t * t)
                }),
            )])
        }

        Operation::Transpose { input, axes } => {
            let mut inv = vec![0usize; axes.len()];
            for (i, &ax) in axes.iter().enumerate() {
                inv[ax] = i;
            }
            Ok(vec![(
                *input,
                grad_out.view().permuted_axes(IxDyn(&inv)).to_owned(),
            )])
        }

        Operation::Slice { input, ranges } => {
            let shape = v(*input)?.shape().to_vec();
            let mut grad_in = ArrayD::<T>::zeros(IxDyn(&shape));
            let slice_shape: Vec<usize> = ranges.iter().map(|&(s, e)| e - s).collect();
            let flat: usize = slice_shape.iter().product();
            for lin in 0..flat {
                let idx = unravel(lin, &slice_shape);
                let dst: Vec<usize> = idx
                    .iter()
                    .enumerate()
                    .map(|(ax, &i)| ranges[ax].0 + i)
                    .collect();
                grad_in[dst.as_slice()] = grad_in[dst.as_slice()] + grad_out[idx.as_slice()];
            }
            Ok(vec![(*input, grad_in)])
        }

        Operation::Reshape { .. } | Operation::Broadcast { .. } => Err(anyhow!(
            "backward_base: Reshape/Broadcast are rejected at compile time"
        )),
    }
}

/// Broadcast a reduction's output gradient back over the reduced axis (or the
/// whole tensor, for a full reduction), scaled by `scale` (1 for `Sum`, 1/n for
/// `Mean`).
fn expand_reduction<T>(
    grad_out: &ArrayD<T>,
    input_shape: &[usize],
    axis: Option<usize>,
    scale: T,
) -> Result<ArrayD<T>>
where
    T: Float + ScalarOperand + FromPrimitive,
{
    let expanded = match axis {
        None => ArrayD::from_elem(IxDyn(input_shape), grad_out[[]]),
        Some(ax) => {
            let mut keepdim = grad_out.shape().to_vec();
            keepdim.insert(ax, 1);
            grad_out
                .clone()
                .to_shape(IxDyn(&keepdim))
                .context("reduction backward: keepdim reshape failed")?
                .broadcast(IxDyn(input_shape))
                .ok_or_else(|| anyhow!("reduction backward: broadcast failed"))?
                .to_owned()
        }
    };
    Ok(expanded * scale)
}
