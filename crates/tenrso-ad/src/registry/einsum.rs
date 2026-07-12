//! Einsum rule.
//!
//! The einsum op is *spec-parameterized*: a single registered rule instance
//! serves every contraction, and the spec travels in [`OpParams::spec`].
//!
//! # Forward
//!
//! | operands | path |
//! |----------|------|
//! | 1 | native single-operand einsum (permute / diagonal / sum) |
//! | 2 | [`tenrso_exec::ops::execute_dense_contraction_accelerated`] |
//! | ≥3 | [`tenrso_exec::einsum_ex`] (cost-based contraction path from the planner) |
//!
//! A single-operand einsum is implemented here because the executor's planner
//! has no pairwise contraction to schedule for one input and would return the
//! operand unchanged — silently wrong for `"ij->ji"`.
//!
//! The two-operand path deliberately uses the *accelerated* entry point rather
//! than the plain [`tenrso_exec::ops::execute_dense_contraction`]: for `f32` /
//! `f64` it downcasts through [`std::any::Any`] and hands the inner product to
//! `ndarray`'s `Array2::dot` (pure-Rust `matrixmultiply`, packed and
//! register-blocked), and for every other standard scalar it runs the blocked
//! kernel in parallel. [`AdScalar`] already requires `'static`, which is the
//! only extra bound that dispatch needs. [`crate::vjp::EinsumVjp`] — the
//! backward pass — has always taken that path; the forward pass now matches it,
//! so a contraction and its adjoint run on the same kernel.
//!
//! # Backward
//!
//! For `C = einsum(s_0,…,s_{n-1} -> s_out; X_0,…,X_{n-1})` the adjoint of input
//! `k` is itself an einsum:
//!
//! ```text
//! grad_k = einsum(s_out, s_0, …, ŝ_k, … -> s_k ; grad_C, X_0, …, X̂_k, …)
//! ```
//!
//! * **2 operands**, non-scalar output, no broadcast index → delegated verbatim
//!   to [`crate::vjp::EinsumVjp`].
//! * **scalar output** (`"i,i->"`) → the adjoint drops the (empty) output
//!   operand and the result is scaled by the scalar cotangent.
//! * **broadcast indices** — an index that occurs *only* in operand `k` (summed
//!   inside that operand, absent from the output and from every other operand)
//!   cannot appear in the adjoint spec. The contraction is then performed over
//!   the remaining indices and the result is expanded back over the missing axes,
//!   which is exactly the derivative of a summed-away axis.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::ops::execute_dense_contraction_accelerated;
use tenrso_exec::{einsum_ex, ExecHints};
use tenrso_planner::EinsumSpec;

use super::{linear_to_multi, row_major_strides, AdScalar, Arity, OpParams, OpRule};
use crate::vjp::{EinsumVjp, VjpOp};

/// Registry rule for `einsum`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsumRule;

impl EinsumRule {
    /// Create the rule (registered as `"einsum"`).
    pub fn new() -> Self {
        Self
    }
}

impl<T> OpRule<T> for EinsumRule
where
    T: AdScalar,
{
    fn name(&self) -> &str {
        "einsum"
    }

    fn arity(&self) -> Arity {
        Arity::AtLeast(1)
    }

    fn forward(&self, inputs: &[DenseND<T>], params: &OpParams) -> Result<DenseND<T>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let spec_str = params.require_spec("einsum")?;
        let spec = EinsumSpec::parse(spec_str)?;
        let dims = validate_spec(&spec, inputs)?;

        match inputs.len() {
            1 => unary_einsum_forward(&spec.inputs[0], &spec.output, &inputs[0], &dims),
            2 => execute_dense_contraction_accelerated(&spec, &inputs[0], &inputs[1]),
            _ => nary_einsum(spec_str, inputs),
        }
    }

    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let spec_str = params.require_spec("einsum")?;
        let spec = EinsumSpec::parse(spec_str)?;
        let dims = validate_spec(&spec, inputs)?;

        let output_shape = subscript_shape(&spec.output, &dims)?;
        if output_grad.shape() != output_shape.as_slice() {
            bail!(
                "einsum '{spec_str}': output gradient shape {:?} does not match the contraction \
                 output shape {output_shape:?}",
                output_grad.shape()
            );
        }

        // Single operand: permutation / diagonal / summation adjoint.
        if inputs.len() == 1 {
            return Ok(vec![unary_einsum_vjp(
                &spec.inputs[0],
                &spec.output,
                inputs[0].shape(),
                output_grad,
                &dims,
            )?]);
        }

        // Binary contraction with a tensor output and no broadcast-only index:
        // this is exactly what `EinsumVjp` implements.
        if inputs.len() == 2 && !spec.output.is_empty() && !needs_broadcast_adjoint(&spec) {
            let ctx = EinsumVjp::new(spec, inputs[0].clone(), inputs[1].clone());
            return ctx.vjp(output_grad);
        }

        // General n-ary adjoint (also covers scalar outputs and broadcast axes).
        let mut grads = Vec::with_capacity(inputs.len());
        for k in 0..inputs.len() {
            grads.push(adjoint_for_input(&spec, inputs, output_grad, k, &dims)?);
        }
        Ok(grads)
    }
}

/// Validate the spec against the actual operand shapes and return the extent of
/// every index.
///
/// This closes a real hole in the executor: its dense-contraction entry points
/// fall back to an extent of `1` for an index they cannot find, which would
/// silently produce a wrong-shaped result instead of an error.
fn validate_spec<T: AdScalar>(
    spec: &EinsumSpec,
    inputs: &[DenseND<T>],
) -> Result<HashMap<char, usize>> {
    if spec.inputs.len() != inputs.len() {
        bail!(
            "einsum spec declares {} operand(s) but {} tensor(s) were supplied",
            spec.inputs.len(),
            inputs.len()
        );
    }

    let mut dims: HashMap<char, usize> = HashMap::new();

    for (operand, (subscript, tensor)) in spec.inputs.iter().zip(inputs.iter()).enumerate() {
        if subscript.chars().count() != tensor.rank() {
            bail!(
                "einsum operand {operand} has subscript '{subscript}' ({} index/indices) but the \
                 tensor has rank {}",
                subscript.chars().count(),
                tensor.rank()
            );
        }

        let mut seen: HashMap<char, usize> = HashMap::new();
        for (axis, index) in subscript.chars().enumerate() {
            let extent = tensor.shape()[axis];

            if let Some(&previous_axis) = seen.get(&index) {
                if inputs.len() > 1 {
                    bail!(
                        "einsum operand {operand} repeats index '{index}' (axes {previous_axis} \
                         and {axis}); diagonal extraction inside a multi-operand contraction is \
                         not supported by the TenRSo executor"
                    );
                }
                let previous_extent = tensor.shape()[previous_axis];
                if previous_extent != extent {
                    bail!(
                        "einsum operand {operand} repeats index '{index}' with different extents \
                         ({previous_extent} and {extent})"
                    );
                }
            } else {
                seen.insert(index, axis);
            }

            match dims.get(&index) {
                Some(&known) if known != extent => {
                    bail!(
                        "einsum index '{index}' has extent {known} elsewhere but {extent} in \
                         operand {operand}"
                    );
                }
                _ => {
                    dims.insert(index, extent);
                }
            }
        }
    }

    let mut output_seen: Vec<char> = Vec::new();
    for index in spec.output.chars() {
        if output_seen.contains(&index) {
            bail!("einsum output repeats index '{index}', which is not a valid contraction");
        }
        output_seen.push(index);
        if !dims.contains_key(&index) {
            bail!("einsum output index '{index}' does not appear in any operand");
        }
    }

    Ok(dims)
}

/// Shape of a subscript given the index extents.
fn subscript_shape(subscript: &str, dims: &HashMap<char, usize>) -> Result<Vec<usize>> {
    subscript
        .chars()
        .map(|index| {
            dims.get(&index)
                .copied()
                .ok_or_else(|| anyhow!("unknown einsum index '{index}'"))
        })
        .collect()
}

/// Does any operand carry an index that appears nowhere else (neither in the
/// output nor in another operand)? Such an index is summed away inside its own
/// operand and cannot be expressed in an adjoint spec.
fn needs_broadcast_adjoint(spec: &EinsumSpec) -> bool {
    (0..spec.inputs.len()).any(|k| !missing_indices(spec, k).is_empty())
}

/// Indices of operand `k` that appear neither in the output nor in any other operand.
fn missing_indices(spec: &EinsumSpec, k: usize) -> Vec<char> {
    let mut missing = Vec::new();
    for index in spec.inputs[k].chars() {
        if missing.contains(&index) {
            continue;
        }
        let in_output = spec.output.contains(index);
        let in_others = spec
            .inputs
            .iter()
            .enumerate()
            .any(|(j, subscript)| j != k && subscript.contains(index));
        if !in_output && !in_others {
            missing.push(index);
        }
    }
    missing
}

/// n-ary einsum forward through the executor (cost-based contraction path).
fn nary_einsum<T: AdScalar>(spec: &str, inputs: &[DenseND<T>]) -> Result<DenseND<T>> {
    let handles: Vec<TensorHandle<T>> = inputs
        .iter()
        .map(|tensor| TensorHandle::from_dense_auto(tensor.clone()))
        .collect();

    let result = einsum_ex::<T>(spec)
        .inputs(&handles)
        .hints(&ExecHints::default())
        .run()?;

    result
        .as_dense()
        .cloned()
        .ok_or_else(|| anyhow!("einsum '{spec}': executor returned a non-dense tensor"))
}

/// Contract an arbitrary operand list into `output_subscript`.
///
/// Mirrors [`EinsumRule::forward`]'s dispatch exactly, including the native-GEMM
/// two-operand path, so a general adjoint is never slower than the forward it
/// differentiates.
fn contract<T: AdScalar>(
    subscripts: &[String],
    operands: &[DenseND<T>],
    output_subscript: &str,
    dims: &HashMap<char, usize>,
) -> Result<DenseND<T>> {
    let spec_str = format!("{}->{}", subscripts.join(","), output_subscript);
    let spec = EinsumSpec::parse(&spec_str)?;

    match operands.len() {
        0 => bail!("cannot contract an empty operand list ('{spec_str}')"),
        1 => unary_einsum_forward(&spec.inputs[0], &spec.output, &operands[0], dims),
        2 => execute_dense_contraction_accelerated(&spec, &operands[0], &operands[1]),
        _ => nary_einsum(&spec_str, operands),
    }
}

/// Adjoint of input `k`, valid for any arity, scalar outputs and broadcast axes.
fn adjoint_for_input<T: AdScalar>(
    spec: &EinsumSpec,
    inputs: &[DenseND<T>],
    output_grad: &DenseND<T>,
    k: usize,
    dims: &HashMap<char, usize>,
) -> Result<DenseND<T>> {
    let target_subscript = &spec.inputs[k];

    // Indices summed away inside operand k: they cannot appear in the adjoint
    // spec, so contract without them and expand afterwards.
    let missing = missing_indices(spec, k);
    let reduced_subscript: String = {
        let mut seen: Vec<char> = Vec::new();
        target_subscript
            .chars()
            .filter(|index| !missing.contains(index))
            .filter(|index| {
                // Guard against a repeated index inside operand k. Multi-operand
                // specs reject repeats in `validate_spec`, so this can only be a
                // no-op, but it keeps the produced subscript well-formed.
                if seen.contains(index) {
                    false
                } else {
                    seen.push(*index);
                    true
                }
            })
            .collect()
    };

    // Operands of the adjoint: the cotangent (unless the output is a scalar)
    // plus every input except k.
    let mut subscripts: Vec<String> = Vec::with_capacity(inputs.len());
    let mut operands: Vec<DenseND<T>> = Vec::with_capacity(inputs.len());

    let scalar_output = spec.output.is_empty();
    if !scalar_output {
        subscripts.push(spec.output.clone());
        operands.push(output_grad.clone());
    }
    for (j, input) in inputs.iter().enumerate() {
        if j != k {
            subscripts.push(spec.inputs[j].clone());
            operands.push(input.clone());
        }
    }

    if operands.is_empty() {
        bail!(
            "einsum: cannot build an adjoint for operand {k} of '{}' — no operand remains",
            spec.inputs.join(",")
        );
    }

    let mut grad = contract(&subscripts, &operands, &reduced_subscript, dims)?;

    // Scalar output: the cotangent is a scalar multiplier.
    if scalar_output {
        let scale = *output_grad
            .as_array()
            .iter()
            .next()
            .ok_or_else(|| anyhow!("einsum: empty scalar cotangent"))?;
        grad = DenseND::from_array(grad.as_array().mapv(|value| value * scale));
    }

    // Expand over the axes that were summed away inside operand k.
    if !missing.is_empty() {
        let target_shape = subscript_shape(target_subscript, dims)?;
        grad = expand_subscript(&grad, &reduced_subscript, target_subscript, &target_shape)?;
    }

    Ok(grad)
}

/// Broadcast a tensor whose axes are labelled `source_subscript` onto the axis
/// layout of `target_subscript` (every extra target axis is constant).
fn expand_subscript<T: AdScalar>(
    source: &DenseND<T>,
    source_subscript: &str,
    target_subscript: &str,
    target_shape: &[usize],
) -> Result<DenseND<T>> {
    let source_chars: Vec<char> = source_subscript.chars().collect();
    let target_chars: Vec<char> = target_subscript.chars().collect();

    if source.rank() != source_chars.len() {
        bail!(
            "einsum adjoint: contraction produced rank {} for subscript '{source_subscript}'",
            source.rank()
        );
    }

    let source_strides = row_major_strides(source.shape());
    let source_values: Vec<T> = source.as_array().iter().copied().collect();

    let total: usize = target_shape.iter().product();
    let mut values = Vec::with_capacity(total);
    let mut multi = vec![0usize; target_chars.len()];

    for linear in 0..total {
        linear_to_multi(linear, target_shape, &mut multi);

        let mut source_linear = 0usize;
        for (axis, index) in source_chars.iter().enumerate() {
            let target_axis = target_chars
                .iter()
                .position(|candidate| candidate == index)
                .ok_or_else(|| {
                    anyhow!("einsum adjoint: index '{index}' is absent from '{target_subscript}'")
                })?;
            source_linear += multi[target_axis] * source_strides[axis];
        }

        values.push(source_values[source_linear]);
    }

    DenseND::from_vec(values, target_shape)
}

/// Single-operand einsum: permutation, diagonal extraction and summation.
fn unary_einsum_forward<T: AdScalar>(
    input_subscript: &str,
    output_subscript: &str,
    input: &DenseND<T>,
    dims: &HashMap<char, usize>,
) -> Result<DenseND<T>> {
    let input_chars: Vec<char> = input_subscript.chars().collect();
    let output_chars: Vec<char> = output_subscript.chars().collect();
    let output_shape = subscript_shape(output_subscript, dims)?;
    let output_strides = row_major_strides(&output_shape);

    let input_shape = input.shape().to_vec();
    let mut values = vec![T::zero(); output_shape.iter().product::<usize>()];
    let mut multi = vec![0usize; input_chars.len()];

    for (linear, &value) in input.as_array().iter().enumerate() {
        linear_to_multi(linear, &input_shape, &mut multi);
        if !on_diagonal(&input_chars, &multi) {
            continue;
        }
        let out_linear = output_linear_index(&input_chars, &output_chars, &multi, &output_strides)?;
        values[out_linear] += value;
    }

    DenseND::from_vec(values, &output_shape)
}

/// Adjoint of a single-operand einsum.
fn unary_einsum_vjp<T: AdScalar>(
    input_subscript: &str,
    output_subscript: &str,
    input_shape: &[usize],
    output_grad: &DenseND<T>,
    dims: &HashMap<char, usize>,
) -> Result<DenseND<T>> {
    let input_chars: Vec<char> = input_subscript.chars().collect();
    let output_chars: Vec<char> = output_subscript.chars().collect();
    let output_shape = subscript_shape(output_subscript, dims)?;
    let output_strides = row_major_strides(&output_shape);

    let grad_values: Vec<T> = output_grad.as_array().iter().copied().collect();

    let total: usize = input_shape.iter().product();
    let mut values = vec![T::zero(); total];
    let mut multi = vec![0usize; input_chars.len()];

    for (linear, value) in values.iter_mut().enumerate() {
        linear_to_multi(linear, input_shape, &mut multi);
        if !on_diagonal(&input_chars, &multi) {
            // Off-diagonal entries never contribute to the output.
            continue;
        }
        let out_linear = output_linear_index(&input_chars, &output_chars, &multi, &output_strides)?;
        *value = grad_values[out_linear];
    }

    DenseND::from_vec(values, input_shape)
}

/// For a repeated index, only the "diagonal" entries (equal positions) contribute.
fn on_diagonal(input_chars: &[char], multi: &[usize]) -> bool {
    for (axis, index) in input_chars.iter().enumerate() {
        for (other_axis, other_index) in input_chars.iter().enumerate().skip(axis + 1) {
            if index == other_index && multi[axis] != multi[other_axis] {
                return false;
            }
        }
    }
    true
}

/// Row-major linear index of the output cell an input element contributes to.
fn output_linear_index(
    input_chars: &[char],
    output_chars: &[char],
    multi: &[usize],
    output_strides: &[usize],
) -> Result<usize> {
    let mut linear = 0usize;
    for (out_axis, index) in output_chars.iter().enumerate() {
        let in_axis = input_chars
            .iter()
            .position(|candidate| candidate == index)
            .ok_or_else(|| anyhow!("einsum output index '{index}' is absent from the operand"))?;
        linear += multi[in_axis] * output_strides[out_axis];
    }
    Ok(linear)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, offset: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i as f64) * 0.53 + offset).cos() + 1.25)
            .collect()
    }

    /// Fully independent naive two-operand einsum: one accumulator per output
    /// cell, a plain nested walk over the whole index space, no blocking, no
    /// GEMM, no code shared with `tenrso-exec`.
    ///
    /// This is the oracle for the native-GEMM forward path: it pins down operand
    /// orientation (a transposed operand cannot survive it) *and* the value.
    fn naive_binary_einsum(spec_str: &str, a: &DenseND<f64>, b: &DenseND<f64>) -> DenseND<f64> {
        let spec = EinsumSpec::parse(spec_str).expect("spec");
        let subscripts = [spec.inputs[0].as_str(), spec.inputs[1].as_str()];
        let operands = [a, b];

        // Extent of every index, taken straight from the operands.
        let mut dims: HashMap<char, usize> = HashMap::new();
        for (subscript, tensor) in subscripts.iter().zip(operands.iter()) {
            for (axis, index) in subscript.chars().enumerate() {
                dims.insert(index, tensor.shape()[axis]);
            }
        }

        // Every index, output ones first so the loop nest is easy to reason about.
        let mut all: Vec<char> = spec.output.chars().collect();
        for subscript in &subscripts {
            for index in subscript.chars() {
                if !all.contains(&index) {
                    all.push(index);
                }
            }
        }
        let extents: Vec<usize> = all.iter().map(|index| dims[index]).collect();

        let output_shape: Vec<usize> = spec.output.chars().map(|index| dims[&index]).collect();
        let output_strides = row_major_strides(&output_shape);
        let mut values = vec![0.0f64; output_shape.iter().product::<usize>().max(1)];

        let total: usize = extents.iter().product();
        let mut assignment = vec![0usize; all.len()];
        for linear in 0..total {
            linear_to_multi(linear, &extents, &mut assignment);
            let position = |index: char| -> usize {
                assignment[all
                    .iter()
                    .position(|candidate| *candidate == index)
                    .expect("index is in `all`")]
            };

            // Row-major offset of this assignment inside each operand.
            let offset = |operand: usize| -> usize {
                let strides = row_major_strides(operands[operand].shape());
                subscripts[operand]
                    .chars()
                    .enumerate()
                    .map(|(axis, index)| position(index) * strides[axis])
                    .sum()
            };

            let out_linear: usize = spec
                .output
                .chars()
                .enumerate()
                .map(|(axis, index)| position(index) * output_strides[axis])
                .sum();

            values[out_linear] += a.as_slice()[offset(0)] * b.as_slice()[offset(1)];
        }

        DenseND::from_vec(values, &output_shape).expect("reference tensor")
    }

    /// The native-GEMM forward must stay inside the *classical* forward-error
    /// bound for a dot product.
    ///
    /// Switching the two-operand forward from the blocked triple loop to
    /// `matrixmultiply` changes the order in which the `k` products are summed, so
    /// the last bits of the result move. What must *not* change is that the kernel
    /// is backward stable: for `c = Σ_k a_k · b_k` computed in any order,
    ///
    /// ```text
    /// |ĉ − c| ≤ γ_k · Σ_k |a_k · b_k|,   γ_k = k·u / (1 − k·u),   u = 2^-53
    /// ```
    ///
    /// (Higham, *Accuracy and Stability of Numerical Algorithms*, §3.1). This
    /// holds for *every* summation order, so it is a tolerance that does not have
    /// to be re-tuned when the kernel changes — unlike an `assert!(a == b)` on two
    /// kernels, which would forbid the switch outright, and unlike a magic epsilon,
    /// which would just be re-tuned until green.
    ///
    /// The reference is Kahan-compensated, whose own error is `O(u)` independent of
    /// `k` — effectively exact at this size.
    #[test]
    fn test_binary_forward_is_backward_stable() {
        let rule = EinsumRule::new();

        let (m, k, n) = (64usize, 96usize, 80usize);
        let a = DenseND::from_vec(ramp(m * k, 0.31), &[m, k]).unwrap();
        let b = DenseND::from_vec(ramp(k * n, 1.87), &[k, n]).unwrap();

        let out = rule
            .forward(&[a.clone(), b.clone()], &OpParams::einsum("ij,jk->ik"))
            .expect("forward");

        let unit_roundoff = f64::EPSILON / 2.0;
        let gamma_k = (k as f64) * unit_roundoff / (1.0 - (k as f64) * unit_roundoff);

        let (sa, sb) = (a.as_slice(), b.as_slice());
        let mut worst_ratio = 0.0f64;

        for i in 0..m {
            for j in 0..n {
                // Kahan-compensated reference, and the magnitude sum that scales
                // the bound.
                let mut sum = 0.0f64;
                let mut compensation = 0.0f64;
                let mut magnitude = 0.0f64;
                for kk in 0..k {
                    let term = sa[i * k + kk] * sb[kk * n + j];
                    magnitude += term.abs();
                    let y = term - compensation;
                    let t = sum + y;
                    compensation = (t - sum) - y;
                    sum = t;
                }

                let error = (out.as_slice()[i * n + j] - sum).abs();
                let bound = gamma_k * magnitude;
                assert!(
                    error <= bound,
                    "({i},{j}): |error| {error:.3e} exceeds the backward-stability bound \
                     {bound:.3e} (gamma_k={gamma_k:.3e}) — the GEMM is not merely reassociating, \
                     it is wrong"
                );
                worst_ratio = worst_ratio.max(error / bound.max(f64::MIN_POSITIVE));
            }
        }

        // Sanity: the bound must not be vacuous. A kernel that summed in a truly
        // pathological order would sit near 1.0; a sane one sits orders below.
        assert!(
            worst_ratio < 1.0,
            "error/bound ratio {worst_ratio} must be below 1"
        );
    }

    #[test]
    fn test_binary_forward_matches_naive_reference() {
        let rule = EinsumRule::new();

        // Shapes large enough that `matrixmultiply` actually reaches its packed
        // micro-kernel (m, n, k all > its 8×4 register block), and every operand
        // extent is distinct so a transposed operand cannot accidentally agree.
        let cases: [(&str, Vec<usize>, Vec<usize>); 6] = [
            ("ij,jk->ik", vec![13, 17], vec![17, 11]),
            // Non-identity output permutation: the GEMM emits `ik`, the caller
            // wants `ki`, so the gather afterwards must transpose.
            ("ij,jk->ki", vec![13, 17], vec![17, 11]),
            // Contract over the *leading* axis of both operands.
            ("ij,ik->jk", vec![19, 7], vec![19, 5]),
            // Batched.
            ("bij,bjk->bik", vec![3, 9, 12], vec![3, 12, 6]),
            // Batched with a permuted output.
            ("bij,bjk->kbi", vec![3, 9, 12], vec![3, 12, 6]),
            // Full contraction to a scalar.
            ("ij,ij->", vec![11, 13], vec![11, 13]),
        ];

        for (spec, shape_a, shape_b) in cases {
            let a = DenseND::from_vec(ramp(shape_a.iter().product(), 0.31), &shape_a).unwrap();
            let b = DenseND::from_vec(ramp(shape_b.iter().product(), 1.87), &shape_b).unwrap();

            let actual = rule
                .forward(&[a.clone(), b.clone()], &OpParams::einsum(spec))
                .unwrap_or_else(|e| panic!("forward '{spec}': {e:#}"));
            let expected = naive_binary_einsum(spec, &a, &b);

            assert_eq!(actual.shape(), expected.shape(), "shape for '{spec}'");

            // The two differ only in summation order. The contracted extents here
            // are ≤ 17 and the values are O(1), so a relative bound of 1e-13 is
            // orders of magnitude above the ~k·eps ≈ 4e-15 worst case and still
            // tight enough that a genuinely wrong (e.g. transposed) result — which
            // is O(1) away — cannot slip through.
            for (index, (lhs, rhs)) in actual
                .as_slice()
                .iter()
                .zip(expected.as_slice().iter())
                .enumerate()
            {
                let scale = rhs.abs().max(1.0);
                assert!(
                    (lhs - rhs).abs() <= 1e-13 * scale,
                    "'{spec}' element {index}: {lhs} vs reference {rhs}"
                );
            }
        }
    }

    #[test]
    fn test_unary_transpose() {
        let rule = EinsumRule::new();
        let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let y = rule
            .forward(&[x], &OpParams::einsum("ij->ji"))
            .expect("transpose");

        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_unary_sum_and_trace() {
        let rule = EinsumRule::new();
        let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let row_sums = rule
            .forward(std::slice::from_ref(&x), &OpParams::einsum("ij->i"))
            .unwrap();
        assert_eq!(row_sums.as_slice(), &[6.0, 15.0]);

        let square = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let trace = rule
            .forward(std::slice::from_ref(&square), &OpParams::einsum("ii->"))
            .unwrap();
        assert!(trace.shape().is_empty());
        assert_eq!(trace.as_slice(), &[5.0]);

        let diag = rule.forward(&[square], &OpParams::einsum("ii->i")).unwrap();
        assert_eq!(diag.as_slice(), &[1.0, 4.0]);
    }

    #[test]
    fn test_unary_diagonal_vjp_is_zero_off_diagonal() {
        let rule = EinsumRule::new();
        let square = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let grad = DenseND::from_elem(&[], 1.0);

        let grads = rule
            .vjp(&[square], &grad, &OpParams::einsum("ii->"))
            .unwrap();
        assert_eq!(grads[0].as_slice(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_binary_matmul_forward() {
        let rule = EinsumRule::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let c = rule
            .forward(&[a, b], &OpParams::einsum("ij,jk->ik"))
            .unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_ternary_forward_matches_sequential_contraction() {
        let rule = EinsumRule::new();
        let a = DenseND::from_vec(ramp(6, 0.1), &[2, 3]).unwrap();
        let b = DenseND::from_vec(ramp(12, 0.9), &[3, 4]).unwrap();
        let c = DenseND::from_vec(ramp(8, 1.4), &[4, 2]).unwrap();

        let abc = rule
            .forward(
                &[a.clone(), b.clone(), c.clone()],
                &OpParams::einsum("ij,jk,kl->il"),
            )
            .unwrap();

        let ab = rule
            .forward(&[a, b], &OpParams::einsum("ij,jk->ik"))
            .unwrap();
        let expected = rule
            .forward(&[ab, c], &OpParams::einsum("ik,kl->il"))
            .unwrap();

        assert_eq!(abc.shape(), expected.shape());
        for (lhs, rhs) in abc.as_slice().iter().zip(expected.as_slice().iter()) {
            assert!((lhs - rhs).abs() < 1e-12, "{lhs} vs {rhs}");
        }
    }

    #[test]
    fn test_spec_validation_catches_shape_errors() {
        let rule = EinsumRule::new();
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::ones(&[4, 2]); // 'j' would be 3 and 4

        let err = rule
            .forward(&[a.clone(), b], &OpParams::einsum("ij,jk->ik"))
            .unwrap_err();
        assert!(err.to_string().contains("extent"), "{err}");

        // Subscript length must match the tensor rank.
        let err = rule
            .forward(std::slice::from_ref(&a), &OpParams::einsum("ijk->i"))
            .unwrap_err();
        assert!(err.to_string().contains("rank"), "{err}");

        // Missing spec.
        let err = rule.forward(&[a], &OpParams::none()).unwrap_err();
        assert!(err.to_string().contains("requires an einsum spec"), "{err}");
    }

    #[test]
    fn test_repeated_index_rejected_in_multi_operand() {
        let rule = EinsumRule::new();
        let a = DenseND::<f64>::ones(&[2, 2]);
        let b = DenseND::<f64>::ones(&[2, 2]);
        let err = rule
            .forward(&[a, b], &OpParams::einsum("ii,ij->j"))
            .unwrap_err();
        assert!(err.to_string().contains("diagonal"), "{err}");
    }

    #[test]
    fn test_broadcast_adjoint_expands_summed_axis() {
        // 'j' appears only in operand 0 and not in the output: the forward sums
        // it away, so grad_a is constant along j.
        let rule = EinsumRule::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![10.0, 20.0], &[2]).unwrap();

        let y = rule
            .forward(&[a.clone(), b.clone()], &OpParams::einsum("ij,i->i"))
            .unwrap();
        // y[i] = b[i] * sum_j a[i,j]
        assert_eq!(y.as_slice(), &[60.0, 300.0]);

        let grad = DenseND::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let grads = rule
            .vjp(&[a, b], &grad, &OpParams::einsum("ij,i->i"))
            .unwrap();

        // dY/da[i,j] = b[i] for every j.
        assert_eq!(grads[0].shape(), &[2, 3]);
        assert_eq!(grads[0].as_slice(), &[10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
        // dY/db[i] = sum_j a[i,j]
        assert_eq!(grads[1].as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn test_scalar_output_vjp() {
        let rule = EinsumRule::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseND::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let y = rule
            .forward(&[a.clone(), b.clone()], &OpParams::einsum("i,i->"))
            .unwrap();
        assert_eq!(y.as_slice(), &[32.0]);

        let grad = DenseND::from_elem(&[], 2.0);
        let grads = rule
            .vjp(&[a, b], &grad, &OpParams::einsum("i,i->"))
            .unwrap();
        assert_eq!(grads[0].as_slice(), &[8.0, 10.0, 12.0]);
        assert_eq!(grads[1].as_slice(), &[2.0, 4.0, 6.0]);
    }
}
