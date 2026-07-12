//! Batched GEMM backends for the pairwise contraction engine.
//!
//! Once [`plan`](super::plan) and [`gather`](super::gather) have canonicalised
//! the operands, *every* two-operand einsum is the same kernel:
//!
//! ```text
//! OUT[bi, i, j] = Σ_{p < k}  A[bi, i, p] · B[bi, p, j]        (row-major)
//! ```
//!
//! Three backends implement it.
//!
//! # `BlockedBackend` — portable, any element type, serial
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
//! # `ParallelBlockedBackend` — the same kernel, over a row-block partition
//!
//! Identical mathematics, distributed over rayon
//! (`scirs2_core::parallel_ops`).  The output is cut into `(batch element, row
//! block)` tiles; **each tile is written by exactly one task**, so there is no
//! reduction across threads and no atomics.  Every task walks the *whole* block
//! schedule in the *same* order as the serial kernel and simply clips each
//! block's `m`-range to its own rows, so for a fixed output element the sequence
//! of `+= a·b` operations — and therefore its floating-point rounding — is
//! **bit-identical to the serial kernel, for any thread count and any tiling**.
//! [`BlockedGemm::accumulate_rows`] is the single kernel both paths call, which
//! makes that property structural rather than a promise.
//!
//! Requires `T: Send + Sync`; see the note on dispatch below for how that is
//! reached without widening any public bound.
//!
//! # `DispatchBackend` — native GEMM for `f32` / `f64`
//!
//! Routes to `ndarray`'s `Array2::dot`, which is backed by the pure-Rust
//! `matrixmultiply` crate (register-blocked, packed micro-kernels).  No BLAS,
//! no C/Fortran.  Any other element type falls back to the blocked kernel —
//! *parallel* for the standard scalar types (integers and `Complex`), serial
//! for anything else.
//!
//! `matrixmultiply` is **single-threaded**, so a batched contraction
//! (`"bij,bjk->bik"`) would otherwise pin the whole batch to one core no matter
//! how many elements it has.  [`scalar_batched_gemm`] therefore runs the batch
//! loop over rayon: batch element `bi` reads `A[bi]`/`B[bi]` and writes `C[bi]`,
//! and `par_chunks_mut` hands each task its own disjoint output slice, so the
//! disjointness is enforced by the borrow checker.  Nothing is reduced across
//! tasks and no dot product's summation order changes, which makes the result
//! **bit-identical to the serial batch loop at any thread count** — the same
//! guarantee the blocked kernel gives.  Below [`PARALLEL_MIN_BATCH_FMA`] the
//! batch runs serially so a small problem does not pay for a fork/join.
//!
//! # Which kernel a type gets, and how that is kept honest
//!
//! [`dispatch_kind`] answers that question as a value ([`DispatchKind`]), and
//! [`DispatchBackend::batched_gemm`] then *executes* the arm it names.  The two
//! are not independent descriptions that could drift: the list of parallel-capable
//! scalars lives once, in [`with_parallel_scalars`], and is expanded by both.
//! Delete a type from the list and the probe stops claiming `ParallelBlocked` for
//! it (a test fails); delete the executing arm and the match stops compiling;
//! delete the macro invocation inside the arm and the GEMM returns a hard error
//! instead of quietly falling back to the serial kernel.  A silent regression to
//! serial is therefore not expressible.
//!
//! ## Why three entry points instead of one
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
//!
//! ## Reaching a parallel kernel from a generic function
//!
//! Rayon needs `T: Send + Sync`, and stable Rust cannot *detect* an auto trait
//! from inside a generic function: method resolution happens once, on the
//! generic body, so the "autoref specialization" trick (a specific impl on
//! `&Wrap<T> where T: Send + Sync`, a blanket one on `Wrap<T>`) silently picks
//! the blanket impl even when the eventual `T` is `f64`.  Real specialization is
//! unstable.  Two routes are left, and this module takes both:
//!
//! 1. **Concrete dispatch.**  Inside `DispatchBackend` the `Any` downcast
//!    already produces a *concrete* element type, where `Send + Sync` is known
//!    statically.  Extending the existing `f32`/`f64` match with the remaining
//!    scalar types (`i8..i128`, `u8..u128`, `isize`, `usize`, `Complex32`,
//!    `Complex64`) hands every one of them to the parallel kernel — with **no
//!    change to any public bound**, so nothing that compiles today stops
//!    compiling.
//! 2. **An additive entry point.**  A caller with a *user-defined* scalar (a
//!    dual number, a fixed-point type, an interval) is outside that list, so
//!    [`execute_dense_contraction_parallel`](super::execute_dense_contraction_parallel)
//!    takes `T: Send + Sync` explicitly and always runs the parallel kernel.
//!    It is a new function, not a changed one.
//!
//! An element type that is neither in the list nor passed through the parallel
//! entry point (i.e. one that is genuinely `!Send`/`!Sync`) keeps the serial
//! kernel, which is the only correct answer for it.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::numeric::{Complex32, Complex64, Num};
use scirs2_core::parallel_ops::{
    current_num_threads, IndexedParallelIterator, ParallelIterator, ParallelSliceMut,
};
use std::any::{Any, TypeId};
use tenrso_core::DenseND;
use tenrso_planner::tiling::{
    compute_matmul_block_sizes, matmul_cache_oblivious_sequence, CacheConfig,
    MatmulCacheObliviousBlock,
};

use super::gather::Operand;
use super::plan::checked_product;

/// Fused multiply–adds below which the parallel kernel runs itself serially.
///
/// A rayon fork/join costs a few microseconds; a 64 Ki-FMA blocked GEMM costs
/// tens of microseconds even for a cheap element type, so this is the point at
/// which the scheduling overhead is safely amortised.  Below it the parallel
/// kernel is *not* wrong, just slower — hence a threshold and not an assertion.
const PARALLEL_MIN_FMA: u128 = 1 << 16;

/// Fused multiply–adds below which the *native* GEMM runs its batch serially.
///
/// Four times [`PARALLEL_MIN_FMA`], because `matrixmultiply` retires an FMA far
/// faster than the portable kernel does: 256 Ki FMA is ~30 µs of native GEMM at
/// the ~18 GFLOP/s this kernel sustains on one core, which comfortably covers a
/// rayon fork/join.  Splitting less work than that across cores loses.
const PARALLEL_MIN_BATCH_FMA: u128 = 1 << 18;

/// The element types [`DispatchBackend`] hands to the **parallel** blocked kernel.
///
/// This list exists exactly once and is expanded twice — by [`dispatch_kind`],
/// which *reports* the kernel a type resolves to, and by
/// [`DispatchBackend::batched_gemm`], which *runs* it.  Keeping one list means
/// the report cannot drift away from the reality it describes; see the module
/// docs.
macro_rules! with_parallel_scalars {
    ($callback:ident) => {
        $callback!(
            i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, Complex32, Complex64,
        )
    };
}

/// Which batched-GEMM kernel [`DispatchBackend`] resolves an element type to.
///
/// Produced by [`dispatch_kind`].  Making the choice a *value* is what lets a
/// test assert that `i64` really does reach the parallel kernel — an assertion on
/// the results alone cannot tell the two blocked kernels apart, since they are
/// bit-identical by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchKind {
    /// `matrixmultiply`-backed native GEMM, batch loop over rayon (`f32`, `f64`).
    NativeGemm,
    /// Portable blocked kernel, parallel over `(batch element, row block)`.
    ParallelBlocked,
    /// Portable blocked kernel, serial — the only correct kernel for a type that
    /// is not known to be `Send + Sync`.
    SerialBlocked,
}

/// The kernel [`DispatchBackend`] will use for element type `T`.
///
/// Pure: it inspects nothing but `T`'s [`TypeId`], so a test can ask the question
/// without running a contraction.
pub(crate) fn dispatch_kind<T: 'static>() -> DispatchKind {
    if TypeId::of::<T>() == TypeId::of::<f64>() || TypeId::of::<T>() == TypeId::of::<f32>() {
        return DispatchKind::NativeGemm;
    }

    macro_rules! parallel_kind_arm {
        ($($scalar:ty),+ $(,)?) => {$(
            if TypeId::of::<T>() == TypeId::of::<$scalar>() {
                return DispatchKind::ParallelBlocked;
            }
        )+};
    }
    with_parallel_scalars!(parallel_kind_arm);

    DispatchKind::SerialBlocked
}

/// Row blocks per batch element, chosen for `~4 · threads` tasks in total.
///
/// Over-decomposing (4× rather than 1× the thread count) lets rayon's
/// work-stealing hide the load imbalance that a ragged final block causes.
/// `batch` already supplies parallelism, so a large batch keeps whole matrices
/// intact (`chunks_per_batch == 1` → `rows_per_task == m`), which is both the
/// cheapest schedule and the friendliest to the cache-oblivious block order.
pub(super) fn parallel_row_chunk(batch: usize, m: usize, threads: usize) -> usize {
    let target_tasks = threads.saturating_mul(4).max(1);
    let chunks_per_batch = target_tasks.div_ceil(batch.max(1)).max(1);
    m.div_ceil(chunks_per_batch).max(1)
}

/// Element types with a native, `matrixmultiply`-backed GEMM.
///
/// Crate-private, hence sealed: downstream crates cannot add implementations
/// and thereby change dispatch behaviour.  `Send + Sync` is a supertrait because
/// the batch loop over this GEMM is a rayon loop.
pub(crate) trait GemmScalar: Sized + Send + Sync {
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

/// Portable cache-oblivious blocked backend, serial (works for every `T`).
pub(crate) struct BlockedBackend;

/// The same blocked kernel over a row-block partition; needs `T: Send + Sync`.
pub(crate) struct ParallelBlockedBackend;

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

impl<T> BatchedGemm<T> for ParallelBlockedBackend
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Send + Sync,
{
    fn batched_gemm(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &Operand<'_, T>,
        b: &Operand<'_, T>,
    ) -> Result<Vec<T>> {
        parallel_blocked_batched_gemm(batch, m, k, n, a.as_slice()?, b.as_slice()?)
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
        // One `match`, so every kernel this backend can run is named by
        // `DispatchKind` — and `DispatchKind` is what the tests assert on.
        // Removing an arm is a compile error; emptying one is a hard error at
        // runtime (below).  Neither can degrade quietly to the serial kernel.
        match dispatch_kind::<T>() {
            DispatchKind::NativeGemm => {
                // `T` is a single concrete type, so if `a` downcasts to `f64`
                // then `b` does too; the pair-match keeps that fact local.
                if let (Some(a64), Some(b64)) =
                    (concrete_slice::<T, f64>(a), concrete_slice::<T, f64>(b))
                {
                    let out = scalar_batched_gemm::<f64>(batch, m, k, n, a64, b64)?;
                    return retype_vec::<f64, T>(out);
                }
                if let (Some(a32), Some(b32)) =
                    (concrete_slice::<T, f32>(a), concrete_slice::<T, f32>(b))
                {
                    let out = scalar_batched_gemm::<f32>(batch, m, k, n, a32, b32)?;
                    return retype_vec::<f32, T>(out);
                }
                Err(anyhow!(
                    "internal error: dispatch_kind reported NativeGemm for an element type with \
                     no native GEMM"
                ))
            }

            DispatchKind::ParallelBlocked => {
                // No native GEMM for this type — but the downcast makes
                // `Send + Sync` statically known, so the blocked kernel can run
                // in parallel.  See the module docs: this is the only way to
                // reach an auto-trait-bounded kernel from a generic function on
                // stable Rust without widening `T`'s bounds.
                macro_rules! parallel_blocked_arm {
                    ($($scalar:ty),+ $(,)?) => {$(
                        if let Some(result) =
                            try_parallel_concrete::<T, $scalar>(batch, m, k, n, a, b)
                        {
                            return result;
                        }
                    )+};
                }
                with_parallel_scalars!(parallel_blocked_arm);

                // `dispatch_kind` said this type is parallel-capable but no arm
                // claimed it, so the two expansions of `with_parallel_scalars`
                // have been made to disagree.  Fail loudly: silently running the
                // serial kernel here is precisely the regression this structure
                // exists to prevent.
                Err(anyhow!(
                    "internal error: dispatch_kind reported ParallelBlocked for an element type \
                     with no parallel arm"
                ))
            }

            // A genuinely exotic element type (possibly `!Send`): serial is the
            // only correct kernel.
            DispatchKind::SerialBlocked => {
                <BlockedBackend as BatchedGemm<T>>::batched_gemm(batch, m, k, n, a, b)
            }
        }
    }
}

/// Run the parallel blocked kernel if `T` *is* the concrete scalar `U`.
///
/// `None` when `T != U`, so the caller can try the next candidate type.  The
/// element buffers are borrowed and the result buffer is moved — no copy on
/// either side of the downcast.
fn try_parallel_concrete<T, U>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &Operand<'_, T>,
    b: &Operand<'_, T>,
) -> Option<Result<Vec<T>>>
where
    T: Clone + Num + 'static,
    U: Clone + Num + std::ops::AddAssign + Send + Sync + 'static,
{
    let a_concrete = concrete_slice::<T, U>(a)?;
    let b_concrete = concrete_slice::<T, U>(b)?;
    Some(
        parallel_blocked_batched_gemm::<U>(batch, m, k, n, a_concrete, b_concrete)
            .and_then(retype_vec::<U, T>),
    )
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
///
/// The batch loop runs over rayon above [`PARALLEL_MIN_BATCH_FMA`], because
/// `matrixmultiply` itself is single-threaded: without this a `"bij,bjk->bik"`
/// with a batch of 64 would use exactly one core.  Batch element `bi` is an
/// independent `A[bi] · B[bi] → C[bi]`, and `par_chunks_mut` gives each task a
/// disjoint `&mut` output slice, so no output element is written twice and no
/// dot product is split across tasks: the summation order inside every dot
/// product is untouched and the result is **bit-identical to the serial batch
/// loop, at any thread count**.
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

    // Zero-filled rather than `with_capacity` + `extend`: a task writes *its*
    // chunk, so the buffer must already exist at full length.  The memset costs
    // `O(batch·m·n)` against the GEMM's `O(batch·m·k·n)` and is not measurable
    // for any `k` the parallel path is taken for.
    let mut out = vec![U::zero(); out_len];

    let total_fma = (batch as u128) * (m as u128) * (k as u128) * (n as u128);
    let parallel = batch > 1 && current_num_threads() > 1 && total_fma >= PARALLEL_MIN_BATCH_FMA;

    let gemm_into = |bi: usize, c_out: &mut [U]| -> Result<()> {
        let a_mat = ArrayView2::from_shape((m, k), &a[bi * a_stride..(bi + 1) * a_stride])
            .map_err(|e| anyhow!("GEMM: cannot view A batch {} as {}×{}: {}", bi, m, k, e))?;
        let b_mat = ArrayView2::from_shape((k, n), &b[bi * b_stride..(bi + 1) * b_stride])
            .map_err(|e| anyhow!("GEMM: cannot view B batch {} as {}×{}: {}", bi, k, n, e))?;
        let c_mat = U::gemm(&a_mat, &b_mat);
        let c_slice = c_mat
            .as_slice()
            .ok_or_else(|| anyhow!("GEMM: result matrix is not contiguous"))?;
        c_out.clone_from_slice(c_slice);
        Ok(())
    };

    if parallel {
        out.par_chunks_mut(c_stride)
            .enumerate()
            .try_for_each(|(bi, c_out)| gemm_into(bi, c_out))?;
    } else {
        out.chunks_mut(c_stride)
            .enumerate()
            .try_for_each(|(bi, c_out)| gemm_into(bi, c_out))?;
    }
    Ok(out)
}

/// Portable cache-oblivious blocked batched GEMM, serial.
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
    let gemm = BlockedGemm::new(batch, m, k, n, a, b)?;
    // Pre-zeroed: the blocked kernel *accumulates*, because a `k`-split emits
    // several blocks that target the same output tile.
    let mut out = vec![T::zero(); gemm.out_len()?];
    gemm.run_serial(&mut out);
    Ok(out)
}

/// Portable cache-oblivious blocked batched GEMM, parallel over `(batch, rows)`.
///
/// Bit-identical to [`blocked_batched_gemm`] — see [`BlockedGemm`].
fn parallel_blocked_batched_gemm<T>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
) -> Result<Vec<T>>
where
    T: Clone + Num + std::ops::AddAssign + Send + Sync,
{
    let gemm = BlockedGemm::new(batch, m, k, n, a, b)?;
    let mut out = vec![T::zero(); gemm.out_len()?];
    gemm.run_parallel(&mut out);
    Ok(out)
}

/// One batched blocked GEMM: the operands, their strides, and the block schedule.
///
/// Both the serial and the parallel driver call exactly one kernel method,
/// [`Self::accumulate_rows`], differing only in *which rows of which batch
/// element* they hand it.  Since a row block of the output is owned by a single
/// task, and the block schedule is walked in the same order regardless of how
/// the rows were cut, every output element accumulates its `k`-terms in one
/// fixed order — the results are bit-identical across thread counts.
struct BlockedGemm<'x, T> {
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    /// `m·k` — elements per batch element of `A`.
    a_stride: usize,
    /// `k·n` — elements per batch element of `B`.
    b_stride: usize,
    /// `m·n` — elements per batch element of `C`.
    c_stride: usize,
    a: &'x [T],
    b: &'x [T],
    /// Shared by every batch element: the schedule depends only on `(m, k, n)`.
    blocks: Vec<MatmulCacheObliviousBlock>,
}

impl<'x, T> BlockedGemm<'x, T>
where
    T: Clone + Num + std::ops::AddAssign,
{
    /// Validate the operand lengths and build the block schedule.
    fn new(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &'x [T],
        b: &'x [T],
    ) -> Result<BlockedGemm<'x, T>> {
        let (a_stride, b_stride, c_stride) =
            check_operand_lengths(batch, m, k, n, a.len(), b.len())?;
        // A degenerate extent means an empty (or all-zero, for `k == 0`) result:
        // no block is emitted, and both drivers then leave the zeroed buffer as
        // it is.  Keeping the schedule empty also keeps the cache model — which
        // divides by the extents — away from a zero.
        let degenerate = batch == 0 || m == 0 || k == 0 || n == 0;
        let blocks = if degenerate {
            Vec::new()
        } else {
            block_schedule::<T>(m, k, n)
        };
        Ok(BlockedGemm {
            batch,
            m,
            k,
            n,
            a_stride,
            b_stride,
            c_stride,
            a,
            b,
            blocks,
        })
    }

    /// `batch · m · n`, checked.
    fn out_len(&self) -> Result<usize> {
        checked_product([self.batch, self.c_stride].into_iter())
    }

    /// Fused multiply–adds the whole batched GEMM performs.
    fn total_fma(&self) -> u128 {
        (self.batch as u128) * (self.m as u128) * (self.k as u128) * (self.n as u128)
    }

    /// The kernel: accumulate rows `[row_lo, row_hi)` of batch element `bi`.
    ///
    /// `c_rows` holds row `row_lo` at offset `0` and is `(row_hi - row_lo)·n`
    /// elements long; `row_lo == 0` with the full row count is the serial case.
    /// Blocks are walked in schedule order and clipped to the row window, so an
    /// output element sees the same `k`-blocks, in the same order, whatever the
    /// row partition is.
    ///
    /// The innermost loop is an `axpy` over two contiguous slices: no
    /// loop-carried dependency, so it auto-vectorises.
    fn accumulate_rows(&self, bi: usize, row_lo: usize, row_hi: usize, c_rows: &mut [T]) {
        let (k, n) = (self.k, self.n);
        let a_mat = &self.a[bi * self.a_stride..(bi + 1) * self.a_stride];
        let b_mat = &self.b[bi * self.b_stride..(bi + 1) * self.b_stride];

        for block in &self.blocks {
            let (k0, k1) = block.k_range;
            let (n0, n1) = block.n_range;
            let m0 = block.m_range.0.max(row_lo);
            let m1 = block.m_range.1.min(row_hi);

            for i in m0..m1 {
                let a_row = &a_mat[i * k + k0..i * k + k1];
                let c_base = (i - row_lo) * n;
                let c_row = &mut c_rows[c_base + n0..c_base + n1];
                for (p, a_val) in (k0..k1).zip(a_row.iter()) {
                    let b_row = &b_mat[p * n + n0..p * n + n1];
                    for (c_val, b_val) in c_row.iter_mut().zip(b_row.iter()) {
                        *c_val += a_val.clone() * b_val.clone();
                    }
                }
            }
        }
    }

    /// Whole batch, one thread.
    fn run_serial(&self, out: &mut [T]) {
        if self.blocks.is_empty() {
            return;
        }
        for (bi, c_mat) in out.chunks_mut(self.c_stride).enumerate() {
            self.accumulate_rows(bi, 0, self.m, c_mat);
        }
    }
}

impl<T> BlockedGemm<'_, T>
where
    T: Clone + Num + std::ops::AddAssign + Send + Sync,
{
    /// Whole batch, one task per `(batch element, row block)` tile.
    ///
    /// `par_chunks_mut` hands each task a disjoint `&mut` slice of the output,
    /// so the "one writer per tile" invariant is enforced by the borrow checker
    /// rather than by convention — no atomics, no reduction, no `unsafe`.
    fn run_parallel(&self, out: &mut [T]) {
        if self.blocks.is_empty() {
            return;
        }
        let threads = current_num_threads().max(1);
        if threads == 1 || self.total_fma() < PARALLEL_MIN_FMA {
            self.run_serial(out);
            return;
        }

        let rows_per_task = parallel_row_chunk(self.batch, self.m, threads);
        // Non-zero: `blocks` is non-empty, so every extent is ≥ 1.
        let elems_per_task = rows_per_task * self.n;

        out.par_chunks_mut(self.c_stride)
            .enumerate()
            .for_each(|(bi, c_mat)| {
                c_mat
                    .par_chunks_mut(elems_per_task)
                    .enumerate()
                    .for_each(|(task, c_rows)| {
                        let row_lo = task * rows_per_task;
                        let row_hi = (row_lo + rows_per_task).min(self.m);
                        self.accumulate_rows(bi, row_lo, row_hi, c_rows);
                    });
            });
    }
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
