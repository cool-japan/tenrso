//! Batched GEMM backends for the pairwise contraction engine.
//!
//! Once [`plan`](super::plan) and [`gather`](super::gather) have canonicalised
//! the operands, *every* two-operand einsum is the same kernel:
//!
//! ```text
//! OUT[bi, i, j] = Σ_{p < k}  A[bi, i, p] · B[bi, p, j]        (row-major)
//! ```
//!
//! Two backends implement it.
//!
//! # `BlockedBackend` — portable, any element type
//!
//! A cache-oblivious blocked kernel built on
//! [`tenrso_planner::tiling::matmul_cache_oblivious_sequence`].  The recursion
//! halves the largest of `m`, `k`, `n` until the `A+B+C` working set of a block
//! fits the cache-aware block size returned by
//! [`tenrso_planner::tiling::compute_matmul_block_sizes`], giving
//! `O(m·n·k / (L·√Z))` cache misses (Frigo et al., FOCS 1999) instead of the
//! `O(m·n·k / L)` of a naive triple loop.
//!
//! **Splitting `k` produces several blocks that write the same output tile**, so
//! the kernel *accumulates* into a pre-zeroed `C` — it never overwrites.  The
//! innermost loop is an `axpy` (`c_row += a_scalar · b_row`) over two contiguous
//! slices, which has no loop-carried dependency and vectorises well.
//!
//! # `DispatchBackend` — native GEMM for `f32` / `f64`
//!
//! Routes to `ndarray`'s `Array2::dot`, which is backed by the pure-Rust
//! `matrixmultiply` crate (register-blocked, packed micro-kernels).  No BLAS,
//! no C/Fortran.
//!
//! ## Why two entry points instead of one
//!
//! Selecting a per-type kernel requires *type identity*, and type identity in
//! stable Rust (`TypeId`/`Any`) requires `T: 'static`.  The public
//! [`execute_dense_contraction`](super::execute_dense_contraction) is bounded
//! only by `Clone + Num + AddAssign + Default` and is called from `tenrso-ad`
//! from a context that does not carry `'static`; adding the bound there would
//! break that crate.  So the `'static` requirement is confined to
//! [`execute_dense_contraction_accelerated`](super::execute_dense_contraction_accelerated),
//! which every caller inside `tenrso-exec` can satisfy (the executor chain
//! already requires `Float + FromPrimitive + 'static`).
//!
//! The downcast itself is done through `std::any::Any` on the **owning**
//! containers (`&DenseND<T>` and `Vec<T>`, both `Sized + 'static`), so the fast
//! path is reached with **no `unsafe` and no copy**.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::numeric::Num;
use std::any::Any;
use tenrso_core::DenseND;
use tenrso_planner::tiling::{
    compute_matmul_block_sizes, matmul_cache_oblivious_sequence, CacheConfig,
    MatmulCacheObliviousBlock,
};

use super::gather::Operand;
use super::plan::checked_product;

/// Element types with a native, `matrixmultiply`-backed GEMM.
///
/// Crate-private, hence sealed: downstream crates cannot add implementations
/// and thereby change dispatch behaviour.
pub(crate) trait GemmScalar: Sized {
    /// `(m×k) · (k×n) → (m×n)`.
    fn gemm(a: &ArrayView2<Self>, b: &ArrayView2<Self>) -> Array2<Self>;
}

impl GemmScalar for f32 {
    #[inline]
    fn gemm(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
        a.dot(b)
    }
}

impl GemmScalar for f64 {
    #[inline]
    fn gemm(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
        a.dot(b)
    }
}

/// A batched-GEMM strategy.
///
/// Implemented by zero-sized selector types so the engine can be generic over
/// the backend without any runtime indirection.
pub(crate) trait BatchedGemm<T> {
    /// `out[bi] (m×n) = a[bi] (m×k) · b[bi] (k×n)` for `bi < batch`, all
    /// row-major and densely packed; returns the `batch·m·n` result buffer.
    fn batched_gemm(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &Operand<'_, T>,
        b: &Operand<'_, T>,
    ) -> Result<Vec<T>>;
}

/// Portable cache-oblivious blocked backend (works for every supported `T`).
pub(crate) struct BlockedBackend;

/// Native-GEMM backend: `matrixmultiply` for `f32`/`f64`, blocked otherwise.
pub(crate) struct DispatchBackend;

impl<T> BatchedGemm<T> for BlockedBackend
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default,
{
    fn batched_gemm(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &Operand<'_, T>,
        b: &Operand<'_, T>,
    ) -> Result<Vec<T>> {
        blocked_batched_gemm(batch, m, k, n, a.as_slice()?, b.as_slice()?)
    }
}

impl<T> BatchedGemm<T> for DispatchBackend
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + 'static,
{
    fn batched_gemm(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &Operand<'_, T>,
        b: &Operand<'_, T>,
    ) -> Result<Vec<T>> {
        // `T` is a single concrete type, so if `a` downcasts to `f64` then `b`
        // does too; the pair-match keeps that fact local and total.
        if let (Some(a64), Some(b64)) = (concrete_slice::<T, f64>(a), concrete_slice::<T, f64>(b)) {
            let out = scalar_batched_gemm::<f64>(batch, m, k, n, a64, b64)?;
            return retype_vec::<f64, T>(out);
        }
        if let (Some(a32), Some(b32)) = (concrete_slice::<T, f32>(a), concrete_slice::<T, f32>(b)) {
            let out = scalar_batched_gemm::<f32>(batch, m, k, n, a32, b32)?;
            return retype_vec::<f32, T>(out);
        }
        <BlockedBackend as BatchedGemm<T>>::batched_gemm(batch, m, k, n, a, b)
    }
}

/// View an operand's canonical buffer as `&[U]` when `T` *is* `U`.
///
/// Safe and zero-copy: the downcast is performed by `std::any::Any` on the
/// owning container (`DenseND<T>` or `Vec<T>`), both of which are `Sized` and
/// `'static`.  Returns `None` when `T != U`.
fn concrete_slice<'x, T, U>(operand: &'x Operand<'_, T>) -> Option<&'x [U]>
where
    T: Clone + Num + 'static,
    U: Clone + Num + 'static,
{
    match operand {
        Operand::Direct(tensor) => {
            let erased: &dyn Any = *tensor;
            erased
                .downcast_ref::<DenseND<U>>()
                .and_then(DenseND::try_as_slice)
        }
        Operand::Owned(buffer) => {
            let erased: &dyn Any = buffer;
            erased.downcast_ref::<Vec<U>>().map(Vec::as_slice)
        }
    }
}

/// Move a `Vec<U>` back out as the caller's `Vec<T>` when `T` *is* `U`.
///
/// Safe and allocation-preserving (the buffer is moved, not copied).
fn retype_vec<U, T>(values: Vec<U>) -> Result<Vec<T>>
where
    U: 'static,
    T: 'static,
{
    let erased: Box<dyn Any> = Box::new(values);
    erased
        .downcast::<Vec<T>>()
        .map(|boxed| *boxed)
        .map_err(|_| anyhow!("internal error: GEMM backend produced a mismatched element type"))
}

/// Validate that the canonical buffers have exactly the lengths the plan says.
fn check_operand_lengths(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a_len: usize,
    b_len: usize,
) -> Result<(usize, usize, usize)> {
    let a_stride = checked_product([m, k].into_iter())?;
    let b_stride = checked_product([k, n].into_iter())?;
    let c_stride = checked_product([m, n].into_iter())?;
    let expect_a = checked_product([batch, a_stride].into_iter())?;
    let expect_b = checked_product([batch, b_stride].into_iter())?;
    if a_len != expect_a {
        return Err(anyhow!(
            "internal error: canonical A buffer has {} elements, expected batch·m·k = {}",
            a_len,
            expect_a
        ));
    }
    if b_len != expect_b {
        return Err(anyhow!(
            "internal error: canonical B buffer has {} elements, expected batch·k·n = {}",
            b_len,
            expect_b
        ));
    }
    Ok((a_stride, b_stride, c_stride))
}

/// `matrixmultiply`-backed batched GEMM for the native scalar types.
fn scalar_batched_gemm<U>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[U],
    b: &[U],
) -> Result<Vec<U>>
where
    U: GemmScalar + Clone + Num,
{
    let (a_stride, b_stride, c_stride) = check_operand_lengths(batch, m, k, n, a.len(), b.len())?;
    let out_len = checked_product([batch, c_stride].into_iter())?;

    // `k == 0` is a sum over the empty set: the result is exactly zero.  Handled
    // explicitly so no degenerate matrix ever reaches the GEMM.
    if out_len == 0 || k == 0 {
        return Ok(vec![U::zero(); out_len]);
    }

    let mut out: Vec<U> = Vec::with_capacity(out_len);
    for bi in 0..batch {
        let a_mat = ArrayView2::from_shape((m, k), &a[bi * a_stride..(bi + 1) * a_stride])
            .map_err(|e| anyhow!("GEMM: cannot view A batch {} as {}×{}: {}", bi, m, k, e))?;
        let b_mat = ArrayView2::from_shape((k, n), &b[bi * b_stride..(bi + 1) * b_stride])
            .map_err(|e| anyhow!("GEMM: cannot view B batch {} as {}×{}: {}", bi, k, n, e))?;
        let c_mat = U::gemm(&a_mat, &b_mat);
        let c_slice = c_mat
            .as_slice()
            .ok_or_else(|| anyhow!("GEMM: result matrix is not contiguous"))?;
        out.extend_from_slice(c_slice);
    }
    Ok(out)
}

/// Portable cache-oblivious blocked batched GEMM.
fn blocked_batched_gemm<T>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
) -> Result<Vec<T>>
where
    T: Clone + Num + std::ops::AddAssign,
{
    let (a_stride, b_stride, c_stride) = check_operand_lengths(batch, m, k, n, a.len(), b.len())?;
    let out_len = checked_product([batch, c_stride].into_iter())?;

    // Pre-zeroed: the blocked kernel *accumulates*, because a `k`-split emits
    // several blocks that target the same output tile.
    let mut out = vec![T::zero(); out_len];
    if out_len == 0 || k == 0 {
        return Ok(out);
    }

    // The block schedule depends only on (m, k, n), so it is computed once and
    // reused for every batch element.
    let blocks = block_schedule::<T>(m, k, n);

    for bi in 0..batch {
        let a_mat = &a[bi * a_stride..(bi + 1) * a_stride];
        let b_mat = &b[bi * b_stride..(bi + 1) * b_stride];
        let c_mat = &mut out[bi * c_stride..(bi + 1) * c_stride];
        for block in &blocks {
            accumulate_block(k, n, a_mat, b_mat, c_mat, block);
        }
    }
    Ok(out)
}

/// Cache-oblivious block schedule for an `m×k · k×n` product.
///
/// The recursion threshold is expressed in the same metric the splitter uses —
/// the combined `A+B+C` footprint `bm·bk + bk·bn + bm·bn` — and is derived from
/// the cache-aware block sizes so that a leaf block's working set lands in L2.
fn block_schedule<T>(m: usize, k: usize, n: usize) -> Vec<MatmulCacheObliviousBlock> {
    let config = CacheConfig::default();
    // `max(1)`: a zero-sized element type would otherwise divide by zero inside
    // the cache model.
    let bytes_per_element = std::mem::size_of::<T>().max(1);
    let (block_m, block_k, block_n) =
        compute_matmul_block_sizes(m, k, n, bytes_per_element, &config);
    let threshold = block_m
        .saturating_mul(block_k)
        .saturating_add(block_k.saturating_mul(block_n))
        .saturating_add(block_m.saturating_mul(block_n))
        .max(1);
    matmul_cache_oblivious_sequence(m, k, n, threshold)
}

/// `C[m_range, n_range] += A[m_range, k_range] · B[k_range, n_range]`.
///
/// `k` and `n` are the *full* row strides of `A` and `B`/`C` respectively; the
/// block ranges select the sub-matrices.  The inner loop is an `axpy` over two
/// contiguous slices — no reduction dependency, so it auto-vectorises.
#[inline]
fn accumulate_block<T>(
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
    c: &mut [T],
    block: &MatmulCacheObliviousBlock,
) where
    T: Clone + Num + std::ops::AddAssign,
{
    let (m0, m1) = block.m_range;
    let (k0, k1) = block.k_range;
    let (n0, n1) = block.n_range;

    for i in m0..m1 {
        let a_row = &a[i * k + k0..i * k + k1];
        let c_row = &mut c[i * n + n0..i * n + n1];
        for (p, a_val) in (k0..k1).zip(a_row.iter()) {
            let b_row = &b[p * n + n0..p * n + n1];
            for (c_val, b_val) in c_row.iter_mut().zip(b_row.iter()) {
                *c_val += a_val.clone() * b_val.clone();
            }
        }
    }
}
