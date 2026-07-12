//! Dense tensor contraction (einsum).
//!
//! This module implements a general **blocked / batched contraction engine**.
//! Multi-operand einsum is decomposed into a sequence of pairwise contractions
//! by the caller (see `executor::types::execute_plan`); this module is the
//! kernel each of those steps runs.
//!
//! Two entry points:
//!
//! * [`execute_unary_einsum`] — one operand (`"ij->ji"`, `"ii->i"`, `"ii->"`,
//!   `"ij->i"`, …).  A gather, no GEMM.
//! * [`execute_dense_contraction`] / [`execute_dense_contraction_accelerated`] —
//!   two operands.  Gather, batched GEMM, gather.
//!
//! The unary case is not a special case bolted on: it is the *same* index
//! classification and the *same* stride-based gather program with the B operand
//! removed, so the two paths agree on diagonals, summed axes and permutations by
//! construction.
//!
//! # The reduction to batched GEMM
//!
//! Given `"<A>,<B>-><O>"` every index character falls into exactly one of six
//! classes, determined by *where it appears*:
//!
//! ```text
//!               in A   in B   in O
//!   batch        ✓      ✓      ✓     kept, iterated (outer / batch dimension)
//!   contracted   ✓      ✓      ✗     summed over (the GEMM's k)
//!   free-A (m)   ✓      ✗      ✓     kept, rows of the GEMM
//!   free-B (n)   ✗      ✓      ✓     kept, columns of the GEMM
//!   sum-A        ✓      ✗      ✗     summed out of A before contracting
//!   sum-B        ✗      ✓      ✗     summed out of B before contracting
//! ```
//!
//! Additionally, a character repeated **within one operand** selects that
//! operand's generalised diagonal (`"ii"` → `A[i,i]`).
//!
//! Canonicalising each operand — take diagonals, sum out the single-operand
//! axes, permute, and flatten each group to one axis — turns *any* two-operand
//! contraction into
//!
//! ```text
//! OUT[batch, m, n] = Σ_{k}  A'[batch, m, k] · B'[batch, k, n]
//! ```
//!
//! which is exactly a batched GEMM.  Every example in the test-suite lands here:
//!
//! | spec              | batch | m       | k     | n     |
//! |-------------------|-------|---------|-------|-------|
//! | `ij,jk->ik`       | 1     | `i`     | `j`   | `k`   |
//! | `bij,bjk->bik`    | `b`   | `i`     | `j`   | `k`   |
//! | `ijk,kl->ijl`     | 1     | `i·j`   | `k`   | `l`   |
//! | `ijk,jl->ikl`     | 1     | `i·k`   | `j`   | `l`   |
//! | `i,j->ij`         | 1     | `i`     | 1     | `j`   |
//! | `ij,ij->`         | 1     | 1       | `i·j` | 1     |
//! | `ii,ii->`         | 1     | 1       | `i`   | 1     |
//! | `ii,ij->j`        | 1     | 1       | `i`   | `j`   |
//!
//! Note the last two rows: because the diagonal is taken *first*, the contracted
//! extent of `"ii,ii->"` is `i` (an `n`-term sum over the diagonal) and **not**
//! `i·i` (an `n²`-term sum over the whole matrix).  A previous fast path in this
//! module got that wrong — it fired on a byte-equality check of the two
//! subscript strings and then walked the raw shape — and silently returned
//! `Σ_{i,j} A[i,j]·B[i,j]`.  The canonicalisation makes the correct semantics
//! structural rather than a special case.
//!
//! # Pipeline
//!
//! 1. `plan::ContractionPlan::build` (internal) — classify indices, validate
//!    shapes, and emit a stride-based *gather program* per operand.  Pure
//!    metadata.
//! 2. `gather::prepare_operand` (internal) — materialise each operand in
//!    canonical `(batch, m, k)` / `(batch, k, n)` layout.  A no-op borrow when
//!    the operand is already contiguous and in canonical order (the common
//!    case).
//! 3. `gemm` (internal) — batched GEMM.
//! 4. `gather::gather` (internal) — permute the `(batch, m, n)` result into
//!    the requested output order (skipped when it is already correct).
//!
//! # Complexity
//!
//! `O(|A| + |B|)` for canonicalisation plus `O(batch · m · k · n)` fused
//! multiply-adds — asymptotically optimal for a pairwise contraction, versus the
//! `O(|output| · |contracted|)` *with a `HashMap` lookup and a `Vec` allocation
//! per element* of the previous implementation.
//!
//! # Ellipsis
//!
//! Not supported (`tenrso_planner::EinsumSpec` does not parse `...`).

mod gather;
mod gemm;
mod plan;

#[cfg(test)]
mod tests;

use anyhow::Result;
use scirs2_core::numeric::Num;
use tenrso_core::DenseND;
use tenrso_planner::EinsumSpec;

use gather::{flatten_row_major, gather, prepare_operand};
use gemm::{BatchedGemm, BlockedBackend, DispatchBackend, ParallelBlockedBackend};
use plan::{ContractionPlan, UnaryPlan};

/// Execute a **single-operand** dense einsum.
///
/// This is the canonical unary einsum for TenRSo: permutation, generalised
/// diagonal extraction, trace, axis reduction, and any combination of them.
/// It is a strict subset of the pairwise engine — one stride-based gather with
/// a fused sum-out (see the internal `plan::UnaryPlan`) and no GEMM.
///
/// | spec       | operation                                      |
/// |------------|-----------------------------------------------|
/// | `ij->ij`   | identity (buffer handed straight through)      |
/// | `ij->ji`   | transpose                                      |
/// | `ijk->kij` | permutation                                    |
/// | `ii->i`    | diagonal extraction                            |
/// | `iij->ij`  | diagonal over a repeated index, other axes kept|
/// | `ii->`     | trace                                          |
/// | `ij->i`    | reduction over an axis                         |
/// | `ij->`     | full reduction (rank-0 result)                 |
/// | `iij->j`   | diagonal, then reduction                       |
///
/// Non-contiguous operands (e.g. the result of `DenseND::permute`) are handled:
/// the gather strides are expressed against the operand's *logical* row-major
/// layout, and the internal `gather::flatten_row_major` materialises that
/// layout when the memory is not already C-contiguous.
///
/// # Arguments
///
/// * `spec` — einsum specification with exactly one input, e.g. `"ii->i"`
/// * `a` — the input tensor
///
/// # Errors
///
/// Returns an error if the spec does not have exactly one input, if the
/// subscript length does not match the operand's rank, if a repeated index is
/// used with two different extents (`"ii"` on a non-square matrix), if the
/// output repeats an index, or if an output index does not appear in the input.
///
/// # Complexity
///
/// `O(|output| · |summed|)` element reads and `O(|output|)` writes — one pass
/// over exactly the part of the operand the output depends on.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_exec::ops::execute_unary_einsum;
/// use tenrso_planner::EinsumSpec;
///
/// // Diagonal of a 2×2 matrix.
/// let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
/// let spec = EinsumSpec::parse("ii->i")?;
/// let d = execute_unary_einsum(&spec, &a)?;
/// assert_eq!(d.shape(), &[2]);
/// assert_eq!(d.as_slice(), &[1.0, 4.0]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn execute_unary_einsum<T>(spec: &EinsumSpec, a: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign,
{
    let plan = UnaryPlan::build(spec, a.shape())?;
    let flat = flatten_row_major(a);

    // `is_passthrough` implies the gather sums nothing and its kept extents are
    // exactly the source's row-major layout, so `Π output_shape == flat.len()`
    // and the buffer can be handed through untouched.
    let data = if plan.is_passthrough {
        flat.into_owned()
    } else {
        gather(&flat, &plan.gather.kept, &plan.gather.summed)?
    };

    DenseND::from_vec(data, &plan.output_shape)
}

/// Execute a pairwise dense tensor contraction.
///
/// Handles the full two-operand einsum grammar: batch indices, contracted
/// indices, free indices, indices repeated within an operand (diagonals), and
/// indices that appear in one operand only and not in the output (summed out).
///
/// Uses the portable cache-oblivious blocked kernel, **serially**: the bound on
/// `T` here is the weakest of the three entry points, and rayon needs
/// `T: Send + Sync`.  Prefer, in order:
///
/// * [`execute_dense_contraction_accelerated`] (`T: 'static`) — native
///   `matrixmultiply` GEMM for `f32`/`f64`, and the *parallel* blocked kernel
///   for every other standard scalar (integers, `Complex32`, `Complex64`);
/// * [`execute_dense_contraction_parallel`] (`T: Send + Sync`) — the parallel
///   blocked kernel for a user-defined element type;
/// * this function — for an element type that is genuinely `!Send`/`!Sync`.
///
/// # Arguments
///
/// * `spec` — einsum specification with exactly two inputs, e.g. `"bij,bjk->bik"`
/// * `a` — first input tensor
/// * `b` — second input tensor
///
/// # Errors
///
/// Returns an error if the spec does not have exactly two inputs, if a subscript
/// length does not match its operand's rank, if an index is used with two
/// different extents, or if the output repeats an index.
///
/// # Complexity
///
/// `O(|A| + |B| + batch · m · k · n)`; see the module documentation.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_exec::ops::execute_dense_contraction;
/// use tenrso_planner::EinsumSpec;
///
/// let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
/// let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
/// let spec = EinsumSpec::parse("ij,jk->ik")?;
/// let c = execute_dense_contraction(&spec, &a, &b)?;
/// assert_eq!(c.shape(), &[2, 2]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn execute_dense_contraction<T>(
    spec: &EinsumSpec,
    a: &DenseND<T>,
    b: &DenseND<T>,
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default,
{
    contract::<T, BlockedBackend>(spec, a, b)
}

/// Execute a pairwise dense tensor contraction with a **row-block-parallel**
/// blocked kernel.
///
/// Bit-for-bit identical results to [`execute_dense_contraction`] — not merely
/// "identical up to rounding".  The output is cut into `(batch element, row
/// block)` tiles and each tile is written by exactly one rayon task, which walks
/// the *whole* cache-oblivious block schedule in the same order as the serial
/// kernel and clips each block to its own rows.  Every output element therefore
/// accumulates its `k`-terms in one fixed order, independent of the thread count
/// and of the tiling, so this is a drop-in replacement even for code that
/// compares floating-point results exactly.
///
/// Small problems (fewer than 64 Ki fused multiply–adds) run serially: the
/// fork/join would cost more than the work.
///
/// Use this when the element type is *not* one of the standard scalars —
/// [`execute_dense_contraction_accelerated`] already parallelises those (and
/// sends `f32`/`f64` to a native GEMM, which is faster still).  A dual number, a
/// fixed-point type or an interval arithmetic type is exactly the case this
/// entry point exists for.
///
/// # Arguments
///
/// * `spec` — einsum specification with exactly two inputs, e.g. `"bij,bjk->bik"`
/// * `a` — first input tensor
/// * `b` — second input tensor
///
/// # Errors
///
/// Same as [`execute_dense_contraction`].
///
/// # Complexity
///
/// `O(|A| + |B| + batch · m · k · n)` work, `O(batch · m · k · n / p)` span on
/// `p` threads.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_exec::ops::{execute_dense_contraction, execute_dense_contraction_parallel};
/// use tenrso_planner::EinsumSpec;
///
/// let a = DenseND::from_vec((0..64 * 64).map(|i| i as f64).collect(), &[64, 64])?;
/// let b = DenseND::from_vec((0..64 * 64).map(|i| (i % 7) as f64).collect(), &[64, 64])?;
/// let spec = EinsumSpec::parse("ij,jk->ik")?;
///
/// let parallel = execute_dense_contraction_parallel(&spec, &a, &b)?;
/// let serial = execute_dense_contraction(&spec, &a, &b)?;
/// assert_eq!(parallel.as_slice(), serial.as_slice()); // bit-identical
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn execute_dense_contraction_parallel<T>(
    spec: &EinsumSpec,
    a: &DenseND<T>,
    b: &DenseND<T>,
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Send + Sync,
{
    contract::<T, ParallelBlockedBackend>(spec, a, b)
}

/// Execute a pairwise dense tensor contraction, using a native GEMM when the
/// element type is `f32` or `f64`.
///
/// Semantically identical to [`execute_dense_contraction`] — same specs, same
/// results — but dispatches the inner product to `ndarray`'s `Array2::dot`,
/// which is backed by the pure-Rust `matrixmultiply` crate (packed,
/// register-blocked micro-kernels).
///
/// `matrixmultiply` is single-threaded, so the **batch loop** around it runs over
/// rayon: a batched spec such as `"bij,bjk->bik"` uses every core, not one.  Each
/// batch element writes a disjoint slice of the output and no dot product is
/// split across tasks, so the result is bit-identical to a serial batch loop at
/// any thread count.  A batch whose total work is under ~256 Ki FMA stays serial,
/// where the fork/join would cost more than it saves.
///
/// Any other element type falls back to the portable blocked kernel, which runs
/// **in parallel** whenever the downcast identifies a standard scalar
/// (`i8`…`i128`, `u8`…`u128`, `isize`, `usize`, `Complex32`, `Complex64`) and
/// serially otherwise.  Results do not depend on which of those happens: the
/// blocked kernel is bit-identical serial or parallel.
///
/// The extra `'static` bound is what makes the dispatch possible: type identity
/// in stable Rust goes through `std::any::Any`.  It is kept off
/// [`execute_dense_contraction`] so that callers without a `'static` element
/// type (notably `tenrso-ad`'s VJP rules) keep compiling unchanged.
///
/// # Errors
///
/// Same as [`execute_dense_contraction`].
pub fn execute_dense_contraction_accelerated<T>(
    spec: &EinsumSpec,
    a: &DenseND<T>,
    b: &DenseND<T>,
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + 'static,
{
    contract::<T, DispatchBackend>(spec, a, b)
}

/// The engine, generic over the GEMM backend.
fn contract<T, B>(spec: &EinsumSpec, a: &DenseND<T>, b: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default,
    B: BatchedGemm<T>,
{
    let plan = ContractionPlan::build(spec, a.shape(), b.shape())?;

    let operand_a = prepare_operand(a, &plan.a)?;
    let operand_b = prepare_operand(b, &plan.b)?;

    let result = B::batched_gemm(plan.batch, plan.m, plan.k, plan.n, &operand_a, &operand_b)?;

    // The GEMM emits `(batch ++ free_a ++ free_b)`; the caller may have asked for
    // a different interleaving (e.g. `"ij,jk->ki"`).  Reorder if so.
    let data = if plan.out_is_identity {
        result
    } else {
        debug_assert_eq!(result.len(), plan.result_len().unwrap_or(usize::MAX));
        gather(&result, &plan.out.kept, &plan.out.summed)?
    };

    DenseND::from_vec(data, &plan.output_shape)
}
