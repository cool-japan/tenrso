//! Operand materialisation: strided gather with diagonal extraction and
//! fused axis reduction.
//!
//! A [`ContractionPlan`](super::plan::ContractionPlan) describes each operand as
//! a list of *kept* axes and a list of *summed* axes, each carrying an extent
//! and a flat-offset stride into the operand's row-major buffer.  This module
//! executes that description, producing a densely packed, row-major buffer in
//! canonical `(batch, m, k)` / `(batch, k, n)` layout ready to be fed to a GEMM.
//!
//! # The gather
//!
//! With kept axes `(d₀,g₀) … (d_{r-1},g_{r-1})` and summed axes
//! `(e₀,h₀) … (e_{s-1},h_{s-1})`, the gathered buffer is
//!
//! ```text
//! dst[i₀, …, i_{r-1}] = Σ_{j₀ < e₀} … Σ_{j_{s-1} < e_{s-1}}
//!                           src[ Σ_p i_p·g_p  +  Σ_q j_q·h_q ]
//! ```
//!
//! Both index spaces are walked with an **odometer**: the flat source offset is
//! carried incrementally and updated in `O(1)` amortised time per element
//! (a counter is bumped, and on wrap the stride contribution is subtracted).
//! Nothing is allocated inside the loop — in particular there is no per-element
//! `HashMap` lookup and no per-element index `Vec`, which is what made the
//! previous naive implementation so slow.
//!
//! # Complexity
//!
//! `O(kept_len · summed_len)` element reads — i.e. exactly one pass over the
//! part of the source that is actually needed — and `O(kept_len)` writes.
//!
//! # Non-contiguous inputs
//!
//! Strides are always computed against the **row-major** layout of the operand's
//! logical shape.  A tensor whose memory is not C-contiguous (e.g. the result of
//! `DenseND::permute`) is therefore first flattened in logical row-major order
//! by [`flatten_row_major`], which is a no-op borrow when the tensor already is
//! contiguous.

use anyhow::{anyhow, Result};
use scirs2_core::numeric::Num;
use std::borrow::Cow;
use tenrso_core::DenseND;

use super::plan::{GatherAxis, OperandGather};

/// A canonical operand: either a direct borrow of a tensor whose memory is
/// already exactly the canonical buffer, or an owned gathered buffer.
///
/// The `Direct` variant is what makes the common case (`"ij,jk->ik"`,
/// `"bij,bjk->bik"`, …) zero-copy: no data is touched between the caller's
/// tensor and the GEMM kernel.
///
/// Keeping the *tensor* (rather than a `&[T]`) in the borrowed variant is
/// deliberate — it lets the accelerated backend recover a concrete
/// `&DenseND<f64>` / `&DenseND<f32>` through `std::any::Any` without any
/// `unsafe` pointer casting (see [`super::gemm`]).
pub(crate) enum Operand<'a, T> {
    /// The tensor's own contiguous buffer already is the canonical buffer.
    Direct(&'a DenseND<T>),
    /// A freshly gathered, densely packed row-major buffer.
    Owned(Vec<T>),
}

impl<T> Operand<'_, T>
where
    T: Clone + Num,
{
    /// The canonical buffer as a flat row-major slice.
    ///
    /// # Errors
    ///
    /// Only if a `Direct` operand turns out not to be contiguous, which the
    /// construction in [`prepare_operand`] rules out; it is reported as an error
    /// rather than a panic so that a logic bug can never take the process down.
    pub(crate) fn as_slice(&self) -> Result<&[T]> {
        match self {
            Operand::Direct(tensor) => tensor
                .try_as_slice()
                .ok_or_else(|| anyhow!("internal error: direct einsum operand is not contiguous")),
            Operand::Owned(buf) => Ok(buf.as_slice()),
        }
    }
}

/// Flatten a tensor to a row-major slice, borrowing when possible.
///
/// * contiguous (C-order) tensor → zero-copy borrow,
/// * otherwise → one pass copying the elements in **logical** row-major order
///   (`ndarray`'s iterator order), which is exactly the layout the gather
///   strides are expressed against.
pub(crate) fn flatten_row_major<T>(tensor: &DenseND<T>) -> Cow<'_, [T]>
where
    T: Clone + Num,
{
    match tensor.try_as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(tensor.view().iter().cloned().collect()),
    }
}

/// Materialise one operand in canonical layout.
///
/// Skips all work when the gather is the identity on an already-contiguous
/// tensor; otherwise flattens (if needed) and runs the strided gather.
pub(crate) fn prepare_operand<'a, T>(
    tensor: &'a DenseND<T>,
    gather_program: &OperandGather,
) -> Result<Operand<'a, T>>
where
    T: Clone + Num + std::ops::AddAssign,
{
    if tensor.is_contiguous() && gather_program.is_identity(tensor.len()) {
        return Ok(Operand::Direct(tensor));
    }
    let flat = flatten_row_major(tensor);
    let buf = gather(&flat, &gather_program.kept, &gather_program.summed)?;
    Ok(Operand::Owned(buf))
}

/// Execute a gather program against a row-major source buffer.
///
/// See the module documentation for the exact mapping.  Every source access is
/// bounds-checked and reported as an error (never a panic, never `unsafe`).
pub(crate) fn gather<T>(src: &[T], kept: &[GatherAxis], summed: &[GatherAxis]) -> Result<Vec<T>>
where
    T: Clone + Num + std::ops::AddAssign,
{
    let total = super::plan::checked_product(kept.iter().map(|ax| ax.dim))?;
    let mut dst: Vec<T> = Vec::with_capacity(total);
    if total == 0 {
        return Ok(dst);
    }

    let summed_total = super::plan::checked_product(summed.iter().map(|ax| ax.dim))?;

    // Odometer state for the kept (destination) axes.
    let mut counter = vec![0usize; kept.len()];
    // Odometer state for the summed axes, reused across destination positions.
    let mut summed_counter = vec![0usize; summed.len()];
    let mut base = 0usize;

    for _ in 0..total {
        let value = if summed.is_empty() {
            fetch(src, base)?
        } else {
            // Reduce over the summed sub-space rooted at `base`.
            let mut acc = T::zero();
            let mut offset = base;
            summed_counter.iter_mut().for_each(|c| *c = 0);
            for _ in 0..summed_total {
                acc += fetch(src, offset)?;
                advance(&mut summed_counter, summed, &mut offset);
            }
            acc
        };
        dst.push(value);
        advance(&mut counter, kept, &mut base);
    }

    Ok(dst)
}

/// Read `src[offset]`, turning an out-of-bounds access (only reachable via a
/// planning bug) into a descriptive error instead of a panic.
#[inline]
fn fetch<T: Clone>(src: &[T], offset: usize) -> Result<T> {
    src.get(offset).cloned().ok_or_else(|| {
        anyhow!(
            "internal error: einsum gather read offset {} of a {}-element buffer",
            offset,
            src.len()
        )
    })
}

/// Advance a mixed-radix odometer by one step and update the flat offset.
///
/// Digits are ordered most-significant first (row-major), so the last axis is
/// the fastest-varying one.  On a carry the wrapped axis' full stride span is
/// subtracted, which keeps `offset` exactly equal to `Σ counter[d]·stride[d]`
/// at all times — the invariant that makes this `O(1)` amortised.
#[inline]
fn advance(counter: &mut [usize], axes: &[GatherAxis], offset: &mut usize) {
    for d in (0..axes.len()).rev() {
        counter[d] += 1;
        *offset += axes[d].stride;
        if counter[d] < axes[d].dim {
            return;
        }
        counter[d] = 0;
        *offset -= axes[d].stride * axes[d].dim;
    }
}
