//! Index classification and canonicalisation planning for pairwise einsum.
//!
//! This module turns a two-operand [`EinsumSpec`] plus the two operand shapes
//! into a *gather program* that reduces the contraction to a single batched
//! GEMM.  No tensor data is touched here — the plan is pure metadata and is
//! therefore cheap to build and easy to unit-test.
//!
//! # The canonical form
//!
//! Every two-operand einsum can be written as
//!
//! ```text
//! OUT[batch, m, n] = Σ_k  A'[batch, m, k] · B'[batch, k, n]
//! ```
//!
//! where `A'` and `B'` are *derived* operands obtained from `A` and `B` by
//! (a) taking diagonals, (b) summing out axes, and (c) permuting.  The four
//! canonical index groups are:
//!
//! | group        | appears in A | appears in B | appears in output |
//! |--------------|--------------|--------------|-------------------|
//! | `batch`      | yes          | yes          | yes               |
//! | `contracted` | yes          | yes          | no                |
//! | `free_a` (m) | yes          | no           | yes               |
//! | `free_b` (n) | no           | yes          | yes               |
//! | `sum_a`      | yes          | no           | no                |
//! | `sum_b`      | no           | yes          | no                |
//!
//! `sum_a` / `sum_b` are axes that occur in exactly one operand and not in the
//! output; einsum semantics say they are *summed out* of that operand before
//! the contraction (e.g. `"ij,jk->k"` sums `A` over `i` first).  They are
//! folded into the gather so no separate reduction pass is required.
//!
//! # Repeated indices (diagonals)
//!
//! When an index character occurs more than once **within one operand**, einsum
//! semantics select the *generalised diagonal*: `"ii,ii->"` means
//! `Σ_i A[i,i]·B[i,i]`, **not** `Σ_{i,j} A[i,j]·B[i,j]`.
//!
//! The diagonal is expressed exactly — and for free — in the gather program by
//! *summing the strides* of every position at which the character occurs.  If
//! the row-major strides of `A` are `s₀ … s_{r-1}` and character `c` occurs at
//! positions `P(c)`, then advancing the canonical axis for `c` by one step must
//! advance **all** of those positions by one, i.e. the buffer offset advances by
//!
//! ```text
//! stride(c) = Σ_{p ∈ P(c)} s_p
//! ```
//!
//! For `A[i,i]` with shape `n×n` the row-major strides are `(n, 1)` so
//! `stride(i) = n + 1`, which walks `A[0,0], A[1,1], …` — exactly the diagonal.
//! The same rule applies to summed-out axes, so `"ii,jk->k"` correctly computes
//! `(Σ_i A[i,i]) · (Σ_j B[j,k])`.
//!
//! # Complexity
//!
//! Plan construction is `O(rank_a + rank_b + |output|)` with a small `HashMap`
//! keyed by index character.  It allocates only the per-axis vectors.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use tenrso_planner::EinsumSpec;

/// One axis of a *gather program*.
///
/// Semantically: "this axis has extent `dim`; incrementing its index by one
/// advances the flat offset into the (row-major) source buffer by `stride`".
///
/// `stride` is a **sum** of the source's row-major strides over every position
/// at which the corresponding index character occurs, which is what implements
/// diagonal extraction (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GatherAxis {
    /// Extent of this axis.
    pub dim: usize,
    /// Flat-offset increment per unit step along this axis.
    pub stride: usize,
}

/// The gather program for a single operand.
///
/// The materialised buffer has shape `kept` (in order) and each element is the
/// sum of the source elements obtained by ranging over every axis in `summed`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct OperandGather {
    /// Axes that survive into the canonical operand, in canonical order.
    pub kept: Vec<GatherAxis>,
    /// Axes that are summed out while gathering (may be empty).
    pub summed: Vec<GatherAxis>,
}

impl OperandGather {
    /// Is this gather the identity on a row-major source buffer of `src_len`
    /// elements?
    ///
    /// The gather is the identity exactly when it sums nothing and its strides
    /// are precisely the row-major strides implied by its own kept extents, and
    /// those extents cover the whole source buffer.  In that case
    /// `dst[flat] == src[flat]` for every `flat`, so the copy can be skipped and
    /// the source borrowed directly.
    ///
    /// This is conservative: a `false` result only costs a copy, never
    /// correctness.
    pub(crate) fn is_identity(&self, src_len: usize) -> bool {
        if !self.summed.is_empty() {
            return false;
        }
        let mut expected = 1usize;
        for ax in self.kept.iter().rev() {
            if ax.stride != expected {
                return false;
            }
            expected = match expected.checked_mul(ax.dim) {
                Some(v) => v,
                None => return false,
            };
        }
        expected == src_len
    }
}

/// A fully-resolved plan for one pairwise contraction.
///
/// Executing the plan is:
///
/// 1. gather `A` with [`ContractionPlan::a`] → row-major `(batch, m, k)`,
/// 2. gather `B` with [`ContractionPlan::b`] → row-major `(batch, k, n)`,
/// 3. batched GEMM → row-major `(batch, m, n)`,
/// 4. gather the result with [`ContractionPlan::out`] → the requested index
///    order (skipped when [`ContractionPlan::out_is_identity`]).
#[derive(Debug, Clone)]
pub(crate) struct ContractionPlan {
    /// Gather program for the left operand (produces `(batch, m, k)`).
    pub a: OperandGather,
    /// Gather program for the right operand (produces `(batch, k, n)`).
    pub b: OperandGather,
    /// Flattened batch extent (product of all batch axes; `1` if none).
    pub batch: usize,
    /// Flattened free-A extent (`1` if none).
    pub m: usize,
    /// Flattened contracted extent (`1` if none — i.e. an outer product).
    pub k: usize,
    /// Flattened free-B extent (`1` if none).
    pub n: usize,
    /// Gather program mapping the `(batch, m, n)` result into output order.
    pub out: OperandGather,
    /// `true` when [`ContractionPlan::out`] is a no-op and can be skipped.
    pub out_is_identity: bool,
    /// Shape of the final tensor (extents of the output subscripts, in order).
    pub output_shape: Vec<usize>,
}

impl ContractionPlan {
    /// Build the plan for `spec` applied to operands of the given shapes.
    ///
    /// # Errors
    ///
    /// * the spec does not have exactly two inputs,
    /// * a subscript string's length does not match its operand's rank,
    /// * an index character is used with two different extents,
    /// * an output subscript is repeated (`"ij,jk->ii"` is not valid einsum),
    /// * an output subscript does not occur in either input,
    /// * a shape product overflows `usize`.
    pub(crate) fn build(spec: &EinsumSpec, shape_a: &[usize], shape_b: &[usize]) -> Result<Self> {
        if spec.num_inputs() != 2 {
            bail!(
                "Pairwise contraction requires exactly 2 inputs, got {}",
                spec.num_inputs()
            );
        }

        let sub_a: Vec<char> = spec.inputs[0].chars().collect();
        let sub_b: Vec<char> = spec.inputs[1].chars().collect();
        let sub_out: Vec<char> = spec.output.chars().collect();

        if sub_a.len() != shape_a.len() {
            bail!(
                "Subscript '{}' has {} indices but operand A has rank {} (shape {:?})",
                spec.inputs[0],
                sub_a.len(),
                shape_a.len(),
                shape_a
            );
        }
        if sub_b.len() != shape_b.len() {
            bail!(
                "Subscript '{}' has {} indices but operand B has rank {} (shape {:?})",
                spec.inputs[1],
                sub_b.len(),
                shape_b.len(),
                shape_b
            );
        }

        // ── Extents, with consistency checking (also across repeats). ────────
        let mut dims: HashMap<char, usize> = HashMap::new();
        for (&c, &d) in sub_a.iter().zip(shape_a).chain(sub_b.iter().zip(shape_b)) {
            match dims.get(&c) {
                Some(&prev) if prev != d => bail!(
                    "Index '{}' is used with inconsistent extents: {} and {}",
                    c,
                    prev,
                    d
                ),
                Some(_) => {}
                None => {
                    dims.insert(c, d);
                }
            }
        }

        // ── Output validation. ───────────────────────────────────────────────
        for (i, &c) in sub_out.iter().enumerate() {
            if sub_out[..i].contains(&c) {
                bail!(
                    "Output subscript '{}' repeats index '{}'; repeated output indices are not \
                     valid einsum",
                    spec.output,
                    c
                );
            }
            if !dims.contains_key(&c) {
                bail!("Output index '{}' does not appear in any input", c);
            }
        }

        // ── Row-major strides + per-character gather strides. ─────────────────
        let rm_a = row_major_strides(shape_a);
        let rm_b = row_major_strides(shape_b);

        let uniq_a = unique_in_order(&sub_a);
        let uniq_b = unique_in_order(&sub_b);

        let in_a = |c: char| uniq_a.contains(&c);
        let in_b = |c: char| uniq_b.contains(&c);
        let in_out = |c: char| sub_out.contains(&c);

        // Canonical group orders.
        //
        // batch / m / n follow the *output* order so that the final gather is
        // the identity for the overwhelmingly common specs (`ij,jk->ik`,
        // `bij,bjk->bik`, `ijk,kl->ijl`, …).  `k` follows A's order, which is
        // an arbitrary but deterministic choice.
        let batch_chars: Vec<char> = sub_out
            .iter()
            .copied()
            .filter(|&c| in_a(c) && in_b(c))
            .collect();
        let m_chars: Vec<char> = sub_out
            .iter()
            .copied()
            .filter(|&c| in_a(c) && !in_b(c))
            .collect();
        let n_chars: Vec<char> = sub_out
            .iter()
            .copied()
            .filter(|&c| !in_a(c) && in_b(c))
            .collect();
        let k_chars: Vec<char> = uniq_a
            .iter()
            .copied()
            .filter(|&c| in_b(c) && !in_out(c))
            .collect();
        let sum_a_chars: Vec<char> = uniq_a
            .iter()
            .copied()
            .filter(|&c| !in_b(c) && !in_out(c))
            .collect();
        let sum_b_chars: Vec<char> = uniq_b
            .iter()
            .copied()
            .filter(|&c| !in_a(c) && !in_out(c))
            .collect();

        let extent = |c: char| -> Result<usize> {
            dims.get(&c)
                .copied()
                .ok_or_else(|| anyhow!("BUG: index '{}' has no recorded extent", c))
        };

        // `axes_for` builds the gather axes for a list of characters against one
        // operand's subscripts / strides.  The stride of a character is the sum
        // of the strides of every position it occupies (the diagonal rule).
        let axes_for = |chars: &[char], subs: &[char], rm: &[usize]| -> Result<Vec<GatherAxis>> {
            chars
                .iter()
                .map(|&c| {
                    let stride = subs
                        .iter()
                        .zip(rm)
                        .filter(|(&sc, _)| sc == c)
                        .map(|(_, &s)| s)
                        .sum::<usize>();
                    Ok(GatherAxis {
                        dim: extent(c)?,
                        stride,
                    })
                })
                .collect()
        };

        let chain = |x: &[char], y: &[char], z: &[char]| -> Vec<char> {
            x.iter().chain(y).chain(z).copied().collect()
        };

        let a_kept_chars = chain(&batch_chars, &m_chars, &k_chars);
        let b_kept_chars = chain(&batch_chars, &k_chars, &n_chars);

        let gather_a = OperandGather {
            kept: axes_for(&a_kept_chars, &sub_a, &rm_a)?,
            summed: axes_for(&sum_a_chars, &sub_a, &rm_a)?,
        };
        let gather_b = OperandGather {
            kept: axes_for(&b_kept_chars, &sub_b, &rm_b)?,
            summed: axes_for(&sum_b_chars, &sub_b, &rm_b)?,
        };

        // Flattened extent of a character group (panic-free: every character is
        // known to have a recorded extent, but we propagate rather than index).
        let group_extent = |chars: &[char]| -> Result<usize> {
            let group: Vec<usize> = chars.iter().map(|&c| extent(c)).collect::<Result<_>>()?;
            checked_product(group.into_iter())
        };

        let batch = group_extent(&batch_chars)?;
        let m = group_extent(&m_chars)?;
        let k = group_extent(&k_chars)?;
        let n = group_extent(&n_chars)?;

        // ── Result → output gather. ──────────────────────────────────────────
        //
        // The GEMM emits axes in the order `batch ++ m ++ n`; the caller asked
        // for `sub_out`.  Both are permutations of the same character set, so a
        // stride-gather over the result buffer reorders them.
        let res_chars = chain(&batch_chars, &m_chars, &n_chars);
        let res_shape: Vec<usize> = res_chars
            .iter()
            .map(|&c| extent(c))
            .collect::<Result<_>>()?;
        let rm_res = row_major_strides(&res_shape);

        let out_axes: Vec<GatherAxis> = sub_out
            .iter()
            .map(|&c| {
                let pos = res_chars
                    .iter()
                    .position(|&rc| rc == c)
                    .ok_or_else(|| anyhow!("BUG: output index '{}' missing from result", c))?;
                Ok(GatherAxis {
                    dim: extent(c)?,
                    stride: rm_res[pos],
                })
            })
            .collect::<Result<_>>()?;

        let out = OperandGather {
            kept: out_axes,
            summed: Vec::new(),
        };
        let out_is_identity = res_chars == sub_out;

        let output_shape: Vec<usize> = sub_out.iter().map(|&c| extent(c)).collect::<Result<_>>()?;

        Ok(Self {
            a: gather_a,
            b: gather_b,
            batch,
            m,
            k,
            n,
            out,
            out_is_identity,
            output_shape,
        })
    }

    /// Number of elements in the GEMM result buffer (`batch · m · n`).
    pub(crate) fn result_len(&self) -> Result<usize> {
        checked_product([self.batch, self.m, self.n].into_iter())
    }
}

/// A fully-resolved plan for a **single-operand** einsum.
///
/// A unary einsum (`"ij->ji"`, `"ii->i"`, `"ii->"`, `"ijk->k"`, …) is a strict
/// subset of the pairwise machinery: it is one [`OperandGather`] and nothing
/// else.  Every index character of the operand is classified into exactly one of
/// two groups:
///
/// | group    | in the output | meaning                                    |
/// |----------|---------------|--------------------------------------------|
/// | `kept`   | yes           | survives; its position follows the output   |
/// | `summed` | no            | reduced away while gathering                |
///
/// and a character *repeated within the operand* selects the generalised
/// diagonal, expressed — exactly as in [`ContractionPlan`] — by **summing the
/// row-major strides** of every position at which it occurs.  That single rule
/// covers permutation, diagonal extraction, trace, axis reduction and every
/// combination of them:
///
/// | spec       | kept (dim, stride) for `A: n×n×m`      | summed                 |
/// |------------|----------------------------------------|------------------------|
/// | `ij->ji`   | `j:(m,1)`, `i:(n,m)`                   | —                      |
/// | `ii->i`    | `i:(n, n+1)`                           | —                      |
/// | `ii->`     | —                                      | `i:(n, n+1)`           |
/// | `iij->ij`  | `i:(n, n·m+m)`, `j:(m,1)`              | —                      |
/// | `iij->j`   | `j:(m,1)`                              | `i:(n, n·m+m)`         |
/// | `ij->`     | —                                      | `i:(n,m)`, `j:(m,1)`   |
///
/// # Complexity
///
/// Plan construction is `O(rank + |output|)`; execution is one pass over the
/// elements the output actually depends on.
#[derive(Debug, Clone)]
pub(crate) struct UnaryPlan {
    /// The gather program: `kept` axes in output order, `summed` axes reduced.
    pub gather: OperandGather,
    /// Shape of the final tensor (extents of the output subscripts, in order).
    pub output_shape: Vec<usize>,
    /// `true` when the gather is the identity on the operand's row-major buffer
    /// (`"ij->ij"`), so the flat buffer can be handed straight through.
    pub is_passthrough: bool,
}

impl UnaryPlan {
    /// Build the plan for a one-input `spec` applied to an operand of `shape`.
    ///
    /// # Errors
    ///
    /// * the spec does not have exactly one input,
    /// * the subscript's length does not match the operand's rank,
    /// * an index character is repeated with two different extents (`"ii"` on a
    ///   non-square matrix),
    /// * an output subscript is repeated (`"ij->ii"` is not valid einsum),
    /// * an output subscript does not occur in the input,
    /// * a shape product overflows `usize`.
    pub(crate) fn build(spec: &EinsumSpec, shape: &[usize]) -> Result<Self> {
        if spec.num_inputs() != 1 {
            bail!(
                "Unary einsum requires exactly 1 input, got {}",
                spec.num_inputs()
            );
        }

        let sub_in: Vec<char> = spec.inputs[0].chars().collect();
        let sub_out: Vec<char> = spec.output.chars().collect();

        if sub_in.len() != shape.len() {
            bail!(
                "Subscript '{}' has {} indices but the operand has rank {} (shape {:?})",
                spec.inputs[0],
                sub_in.len(),
                shape.len(),
                shape
            );
        }

        // ── Extents, with consistency checking across repeats. ────────────────
        //
        // `"ii"` on a 2×3 matrix has no diagonal of length 2 *and* 3; reject it
        // rather than silently walking whichever extent happens to come first.
        let mut dims: HashMap<char, usize> = HashMap::new();
        for (&c, &d) in sub_in.iter().zip(shape) {
            match dims.get(&c) {
                Some(&prev) if prev != d => bail!(
                    "Index '{}' is used with inconsistent extents: {} and {}",
                    c,
                    prev,
                    d
                ),
                Some(_) => {}
                None => {
                    dims.insert(c, d);
                }
            }
        }

        // ── Output validation. ───────────────────────────────────────────────
        for (i, &c) in sub_out.iter().enumerate() {
            if sub_out[..i].contains(&c) {
                bail!(
                    "Output subscript '{}' repeats index '{}'; repeated output indices are not \
                     valid einsum",
                    spec.output,
                    c
                );
            }
            if !dims.contains_key(&c) {
                bail!("Output index '{}' does not appear in the input", c);
            }
        }

        // ── Strides: the diagonal rule (sum over every occurrence). ──────────
        let rm = row_major_strides(shape);
        let stride_of = |c: char| -> usize {
            sub_in
                .iter()
                .zip(&rm)
                .filter(|(&sc, _)| sc == c)
                .map(|(_, &s)| s)
                .sum()
        };
        let extent = |c: char| -> Result<usize> {
            dims.get(&c)
                .copied()
                .ok_or_else(|| anyhow!("BUG: index '{}' has no recorded extent", c))
        };
        let axis_for = |c: char| -> Result<GatherAxis> {
            Ok(GatherAxis {
                dim: extent(c)?,
                stride: stride_of(c),
            })
        };

        // Kept axes follow the *output* order; summed axes follow the operand's
        // order of first appearance (arbitrary but deterministic — the sum is
        // commutative, and a fixed order keeps results bit-reproducible).
        let kept: Vec<GatherAxis> = sub_out
            .iter()
            .map(|&c| axis_for(c))
            .collect::<Result<_>>()?;
        let summed: Vec<GatherAxis> = unique_in_order(&sub_in)
            .into_iter()
            .filter(|c| !sub_out.contains(c))
            .map(axis_for)
            .collect::<Result<_>>()?;

        let output_shape: Vec<usize> = sub_out.iter().map(|&c| extent(c)).collect::<Result<_>>()?;

        let gather = OperandGather { kept, summed };
        let src_len = checked_product(shape.iter().copied())?;
        let is_passthrough = gather.is_identity(src_len);

        Ok(Self {
            gather,
            output_shape,
            is_passthrough,
        })
    }
}

/// Row-major (C-order) strides for `shape`.
///
/// `strides[d] = Π shape[d+1..]`, and `strides[rank-1] == 1`.  For a zero-length
/// shape this is the empty vector (a rank-0 tensor holds exactly one element).
pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1].saturating_mul(shape[d + 1]);
    }
    strides
}

/// Product of `dims`, erroring on `usize` overflow instead of wrapping.
pub(crate) fn checked_product(mut dims: impl Iterator<Item = usize>) -> Result<usize> {
    dims.try_fold(1usize, |acc, d| {
        acc.checked_mul(d)
            .ok_or_else(|| anyhow!("Tensor extent product overflows usize"))
    })
}

/// The distinct characters of `chars`, in order of first appearance.
fn unique_in_order(chars: &[char]) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    for &c in chars {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}
