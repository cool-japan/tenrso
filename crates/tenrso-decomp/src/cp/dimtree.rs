//! **Gauss-Seidel-preserving dimension tree** for CP-ALS.
//!
//! This module answers the question the dense dimension-tree MTTKRP in
//! [`tenrso_kernels::DimTree`] deliberately leaves open: *a dimension tree computes
//! all `N` mode-MTTKRPs against **one frozen factor set**, but classical ALS is
//! **Gauss-Seidel** — mode `k`'s MTTKRP must see the factors `0..k` that were
//! already refreshed earlier in the same sweep. Can the tree's shared partial
//! contractions be reused without silently turning ALS into Jacobi?*
//!
//! **Yes — and at exactly the same cost.** The proof is below; the consequence is
//! that [`cp_als`](crate::cp_als) keeps its *iterate-for-iterate* Gauss-Seidel
//! semantics while its per-sweep work drops from `N·nnz·R` to `2·nnz·R` (plus it
//! sheds `N` full tensor copies and `N` full complement Khatri-Rao
//! materializations).
//!
//! # Notation
//!
//! `X ∈ R^{I_0 × … × I_{N-1}}` row-major, factors `A_k ∈ R^{I_k × R}`. For a
//! **contiguous** mode range `S = [lo, hi)` the *partial contraction* is
//!
//! ```text
//! P_S[i_S, r] = Σ_{i_j : j ∉ S}  X[i] · Π_{j ∉ S} A_j[i_j, r]
//! ```
//!
//! with `i_S` the row-major mixed-radix index of the modes in `S`. Two facts:
//!
//! * `P_{{k}} = M_k`, the mode-`k` MTTKRP (a leaf *is* the answer);
//! * with `S = S₁ ⊎ S₂` (contiguous, `S₁` first) and `KR(S) = ⊙_{j∈S} A_j`,
//!
//!   ```text
//!   P_{S₁}[i₁, r] = Σ_{i₂} P_S[i₁·|S₂| + i₂, r] · KR(S₂)[i₂, r]
//!   P_{S₂}[i₂, r] = Σ_{i₁} P_S[i₁·|S₂| + i₂, r] · KR(S₁)[i₁, r]
//!   ```
//!
//! # The invariant that makes Gauss-Seidel work
//!
//! The crux is **which factors a node actually depends on**. `P_S` is a function of
//! the factors in the *complement* of `S` only — the factors *inside* `S` are
//! folded in later, on the way down to the leaves. Because tree nodes are
//! **contiguous ranges**, the complement of `S = [lo, hi)` splits cleanly into a
//! **prefix** `[0, lo)` and a **suffix** `[hi, N)`.
//!
//! Now run the sweep in mode order `0, 1, …, N-1` (which is what Gauss-Seidel ALS
//! does) and consider any mode `k ∈ S`:
//!
//! * every `j ∈ [0, lo)` satisfies `j < lo ≤ k`, so `A_j` has **already** been
//!   updated this sweep — Gauss-Seidel wants the **new** value;
//! * every `j ∈ [hi, N)` satisfies `j ≥ hi > k`, so `A_j` has **not yet** been
//!   updated — Gauss-Seidel wants the **old** value.
//!
//! This is *independent of `k`*. So **one** partial `P_S`, built with new prefix
//! factors and old suffix factors, is simultaneously the correct Gauss-Seidel
//! parent for **every** mode in `S`.
//!
//! ## Which partials does an update dirty?
//!
//! Updating `A_k` dirties exactly the nodes `S` with `k ∉ S` — i.e. all the
//! *siblings-to-the-right* along the root path, and nothing on the path itself.
//! Concretely, walking the tree **in-order** (left subtree fully processed —
//! *including its factor updates* — before the right child's partial is formed):
//!
//! 1. entering `S` we hold `P_S` (valid, by the invariant above);
//! 2. `P_{S₁} = contract(P_S, KR(S₂))` with `S₂`'s factors read **now** — they are
//!    all *after* `S₁`'s modes, hence still old. ✔
//! 3. recurse into `S₁`: its leaves update `A_j` for `j ∈ S₁`. `P_S` is *not*
//!    dirtied by this, because `S₁ ⊆ S` and `P_S` does not depend on factors in
//!    `S`. **This is the whole trick.**
//! 4. `P_{S₂} = contract(P_S, KR(S₁))` with `S₁`'s factors read **now** — they are
//!    all *before* `S₂`'s modes, hence freshly updated. ✔
//! 5. recurse into `S₂`.
//!
//! The root is the degenerate case: `P_{[0,N)}` would be `X` replicated `R` times,
//! so we never form it; its two children come straight from two GEMMs against the
//! row-major matricization `X₂` (which *is* the root's data and is never dirtied).
//!
//! ## Cost, honestly
//!
//! Every node's partial is still computed **exactly once**, so the FLOP count is
//! identical to the Jacobi tree: `2·nnz·R + O(√nnz·R)`. The only price of
//! Gauss-Seidel is that a node's two children can no longer be produced by one
//! *fused* pass over the parent (they are needed at different times), so deeper
//! nodes make two streaming passes instead of one — over `O(√nnz·R)` bytes, i.e.
//! noise. Peak memory is unchanged: the DFS stack holds one partial per level,
//! geometrically shrinking from `O(√nnz·R)`.
//!
//! Compared with the Jacobi tree, Gauss-Seidel is in fact *cheaper end-to-end*:
//! Jacobi's post-sweep factors are all new, so `⟨X, [[A]]⟩` (needed for the fit)
//! requires a fresh MTTKRP, whereas the Gauss-Seidel sweep's **last** leaf `M_{N-1}`
//! is already taken against the final `A_0 … A_{N-2}`, making the fit free.
//! See [`AlsDimTree::gauss_seidel_sweep`], which returns that leaf.
//!
//! # Why this lives here and not in `tenrso-kernels`
//!
//! The traversal must **call back into the least-squares solve** between forming a
//! node's left and right child partials. That is an ALS concern, not an MTTKRP
//! concern: a pure kernel cannot express it without taking a solver callback.
//!
//! # SciRS2 Integration
//!
//! All array operations go through `scirs2_core::ndarray_ext`; parallelism through
//! `scirs2_core::parallel_ops`. Direct `ndarray`/`rand` use is forbidden per
//! SCIRS2_INTEGRATION_POLICY.md.

use super::types::CpError;
use scirs2_core::ndarray_ext::{Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::Float;
use std::borrow::Cow;

// ─── Tree structure ─────────────────────────────────────────────────────────

/// One node of the dimension tree: a contiguous mode range `[lo, hi)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Node {
    /// First mode of this node (inclusive).
    lo: usize,
    /// One past the last mode of this node (exclusive).
    hi: usize,
    /// Split point (`lo < split < hi`); `None` for a leaf.
    split: Option<usize>,
    /// Arena indices of the `[lo, split)` and `[split, hi)` children.
    children: Option<(usize, usize)>,
}

/// A reusable balanced binary dimension tree, specialised for **Gauss-Seidel**
/// CP-ALS sweeps.
///
/// Build once per tensor shape and reuse across every ALS iteration: the tree is
/// factor-independent, `O(N)` to build, and holds no tensor data.
#[derive(Debug, Clone)]
pub(crate) struct AlsDimTree {
    /// Tensor shape this tree was built for.
    shape: Vec<usize>,
    /// Node arena; `nodes[0]` is the root `[0, N)`.
    nodes: Vec<Node>,
    /// Exclusive prefix products (`prefix[i] = Π_{j<i} shape[j]`).
    prefix: Vec<usize>,
}

impl AlsDimTree {
    /// Build a dimension tree for a tensor of the given shape.
    ///
    /// # Errors
    ///
    /// * `shape.len() < 2` — MTTKRP needs at least two factor matrices;
    /// * the dimension product overflows `usize`.
    pub(crate) fn new(shape: &[usize]) -> Result<Self, CpError> {
        let ndim = shape.len();
        if ndim < 2 {
            return Err(CpError::ShapeMismatch(format!(
                "CP-ALS dimension tree requires a tensor of order >= 2 (got order {ndim})"
            )));
        }

        let mut prefix = Vec::with_capacity(ndim + 1);
        prefix.push(1usize);
        for (axis, &dim) in shape.iter().enumerate() {
            let acc = prefix[axis].checked_mul(dim).ok_or_else(|| {
                CpError::ShapeMismatch(format!(
                    "tensor dimension product overflows usize at axis {axis}"
                ))
            })?;
            prefix.push(acc);
        }

        let mut tree = AlsDimTree {
            shape: shape.to_vec(),
            nodes: Vec::with_capacity(2 * ndim),
            prefix,
        };
        tree.build_subtree(0, ndim);
        Ok(tree)
    }

    /// Recursively build `[lo, hi)` and return its arena index.
    fn build_subtree(&mut self, lo: usize, hi: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            lo,
            hi,
            split: None,
            children: None,
        });

        if hi - lo == 1 {
            return idx;
        }

        let split = self.choose_split(lo, hi);
        let left = self.build_subtree(lo, split);
        let right = self.build_subtree(split, hi);

        self.nodes[idx].split = Some(split);
        self.nodes[idx].children = Some((left, right));
        idx
    }

    /// Split `[lo, hi)` so as to minimise `max(Π_{lo..mid}, Π_{mid..hi})`.
    ///
    /// Balancing the dimension *products* (not the mode counts) keeps both the
    /// Khatri-Rao materialisations and the intermediate partials at `O(√nnz·R)` and
    /// equalises the two root GEMM shapes. Comparison is in log-space so it stays
    /// well defined for astronomically large half-products; ties resolve to the
    /// smallest `mid`, which makes the tree deterministic.
    fn choose_split(&self, lo: usize, hi: usize) -> usize {
        let log_dim: Vec<f64> = self.shape[lo..hi]
            .iter()
            .map(|&d| (d.max(1) as f64).ln())
            .collect();
        let total: f64 = log_dim.iter().sum();

        let mut best_mid = lo + 1;
        let mut best_cost = f64::INFINITY;
        let mut left_log = 0.0f64;
        for (offset, &l) in log_dim.iter().enumerate().take(hi - lo - 1) {
            left_log += l;
            let cost = left_log.max(total - left_log);
            if cost < best_cost - 1e-12 {
                best_cost = cost;
                best_mid = lo + offset + 1;
            }
        }
        best_mid
    }

    /// The tensor order `N`.
    #[inline]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements of the tensor this tree was built for.
    #[inline]
    fn numel(&self) -> usize {
        self.prefix[self.shape.len()]
    }

    /// `Π_{j ∈ [lo, hi)} I_j`.
    #[inline]
    fn range_prod(&self, lo: usize, hi: usize) -> usize {
        debug_assert!(lo <= hi && hi <= self.shape.len());
        if self.prefix[lo] == 0 {
            return self.shape[lo..hi].iter().product();
        }
        self.prefix[hi] / self.prefix[lo]
    }

    /// Validate tensor/factor shapes and return the CP rank `R`.
    fn validate<T>(
        &self,
        tensor: &ArrayView<T, IxDyn>,
        factors: &[Array2<T>],
    ) -> Result<usize, CpError>
    where
        T: Copy,
    {
        if tensor.shape() != self.shape.as_slice() {
            return Err(CpError::ShapeMismatch(format!(
                "tensor shape {:?} does not match the shape the dimension tree was built for ({:?})",
                tensor.shape(),
                self.shape
            )));
        }
        if factors.len() != self.shape.len() {
            return Err(CpError::ShapeMismatch(format!(
                "number of factor matrices ({}) must match tensor order ({})",
                factors.len(),
                self.shape.len()
            )));
        }
        let cp_rank = factors[0].shape()[1];
        for (i, factor) in factors.iter().enumerate() {
            if factor.shape()[1] != cp_rank {
                return Err(CpError::ShapeMismatch(format!(
                    "factor matrix {i} has {} columns, expected {cp_rank}",
                    factor.shape()[1]
                )));
            }
            if factor.shape()[0] != self.shape[i] {
                return Err(CpError::ShapeMismatch(format!(
                    "factor matrix {i} has {} rows, expected {} (tensor mode-{i} size)",
                    factor.shape()[0],
                    self.shape[i]
                )));
            }
        }
        Ok(cp_rank)
    }
}

// ─── The Gauss-Seidel sweep ─────────────────────────────────────────────────

impl AlsDimTree {
    /// Run **one exact Gauss-Seidel ALS sweep**, sharing partial contractions.
    ///
    /// `update(mode, mttkrp, factors)` is invoked once per mode, in the order
    /// `0, 1, …, N-1`. When it is called for `mode`, `mttkrp` is *bit-for-bit the
    /// quantity classical Gauss-Seidel ALS would compute there*: the mode-`mode`
    /// MTTKRP taken against `A_0 … A_{mode-1}` **as this sweep already rewrote
    /// them** and `A_{mode+1} … A_{N-1}` **as the previous sweep left them** (up to
    /// floating-point summation order, which the tree changes — see the module
    /// docs). The callback owns the whole `factors` slice, so it can build the Gram
    /// Hadamard product, solve, project/regularise, and write `factors[mode]` back.
    ///
    /// # Returns
    ///
    /// The mode-`(N-1)` MTTKRP, i.e. the last leaf. Because the sweep updates mode
    /// `N-1` last, `A_0 … A_{N-2}` are final at that point, so
    ///
    /// ```text
    /// ⟨X, [[A]]⟩ = Σ_{i, r} M_{N-1}[i, r] · A_{N-1}[i, r]
    /// ```
    ///
    /// holds for the *post-sweep* factors — the fit therefore costs **no extra
    /// MTTKRP**. See [`compute_fit_from_mttkrp`](super::helpers::compute_fit_from_mttkrp).
    ///
    /// # Errors
    ///
    /// Shape/rank validation failures, a reshape failure on a pathological tensor
    /// layout, or any error raised by `update`.
    ///
    /// # Complexity
    ///
    /// `2·nnz·R + O(√nnz·R)` multiply-adds and `O(√nnz·R)` peak temporary memory,
    /// versus `N·nnz·R` plus `N` full tensor copies for the naive per-mode loop.
    pub(crate) fn gauss_seidel_sweep<T, F>(
        &self,
        tensor: &ArrayView<T, IxDyn>,
        factors: &mut Vec<Array2<T>>,
        update: &mut F,
    ) -> Result<Array2<T>, CpError>
    where
        T: Float + Send + Sync + 'static,
        F: FnMut(usize, &Array2<T>, &mut Vec<Array2<T>>) -> Result<(), CpError>,
    {
        let cp_rank = self.validate(tensor, factors)?;
        let ndim = self.ndim();

        if self.numel() == 0 {
            // An empty sum is exactly zero — not a shortcut, the correct answer.
            let mut last = Array2::<T>::zeros((self.shape[ndim - 1], cp_rank));
            for mode in 0..ndim {
                let zeros = Array2::<T>::zeros((self.shape[mode], cp_rank));
                update(mode, &zeros, factors)?;
                if mode + 1 == ndim {
                    last = zeros;
                }
            }
            return Ok(last);
        }

        let root = self.nodes[0];
        let mid = root
            .split
            .ok_or_else(|| CpError::ShapeMismatch("malformed dimension tree root".into()))?;
        let (left_idx, right_idx) = root
            .children
            .ok_or_else(|| CpError::ShapeMismatch("malformed dimension tree root".into()))?;

        let rows = self.range_prod(0, mid);
        let cols = self.range_prod(mid, ndim);

        // `X₂`: the row-major matricization. Zero-copy when `X` is C-contiguous
        // (the normal case for `DenseND`); at worst ONE copy per sweep, where the
        // naive per-mode loop makes N.
        let standard = tensor.as_standard_layout();
        let x2: ArrayView2<T> = standard
            .view()
            .into_shape_with_order((rows, cols))
            .map_err(|e| {
                CpError::ShapeMismatch(format!(
                    "could not matricize tensor for dimension tree: {e}"
                ))
            })?;

        // ── Root, left child: complement is [mid, N), all still OLD. ──
        let partial_left = {
            let kr_right = khatri_rao_range(factors, mid, ndim, cp_rank);
            gemm_rows(&x2, kr_right.as_ref())
        };
        let mut last: Option<Array2<T>> = None;
        self.descend(left_idx, partial_left, factors, cp_rank, update, &mut last)?;

        // ── Root, right child: complement is [0, mid), which the left descent has
        //    just finished updating — read it NOW to get the Gauss-Seidel state. ──
        let partial_right = {
            let kr_left = khatri_rao_range(factors, 0, mid, cp_rank);
            gemm_cols(&x2, kr_left.as_ref())
        };
        self.descend(
            right_idx,
            partial_right,
            factors,
            cp_rank,
            update,
            &mut last,
        )?;

        last.ok_or_else(|| {
            CpError::ShapeMismatch("dimension tree never reached the last mode (malformed)".into())
        })
    }

    /// In-order descent. `partial` is `P_S` for `self.nodes[node_idx]`, already
    /// correct in the Gauss-Seidel sense (new prefix factors, old suffix factors).
    ///
    /// `partial` is deliberately kept alive across the left-subtree recursion: the
    /// left subtree only rewrites factors *inside* `S`, and `P_S` does not depend on
    /// those, so it survives untouched and can still seed the right child.
    fn descend<T, F>(
        &self,
        node_idx: usize,
        partial: Array2<T>,
        factors: &mut Vec<Array2<T>>,
        cp_rank: usize,
        update: &mut F,
        last: &mut Option<Array2<T>>,
    ) -> Result<(), CpError>
    where
        T: Float + Send + Sync + 'static,
        F: FnMut(usize, &Array2<T>, &mut Vec<Array2<T>>) -> Result<(), CpError>,
    {
        let node = self.nodes[node_idx];

        let Some(split) = node.split else {
            // Leaf: `P_{{k}} = M_k` — exactly the Gauss-Seidel MTTKRP for mode `k`.
            update(node.lo, &partial, factors)?;
            if node.lo + 1 == self.ndim() {
                *last = Some(partial);
            }
            return Ok(());
        };
        let (left, right) = node
            .children
            .ok_or_else(|| CpError::ShapeMismatch("malformed dimension tree node".into()))?;

        let left_dim = self.range_prod(node.lo, split);
        let right_dim = self.range_prod(split, node.hi);

        // Left child: fold away S₂ = [split, hi). Those modes all come *after* every
        // mode of S₁, so their factors are still the previous sweep's — which is
        // precisely what Gauss-Seidel prescribes for S₁'s updates.
        let child_left = {
            let kr_right = khatri_rao_range(factors, split, node.hi, cp_rank);
            contract_left_child(&partial, left_dim, right_dim, cp_rank, kr_right.as_ref())
        };
        self.descend(left, child_left, factors, cp_rank, update, last)?;

        // Right child: fold away S₁ = [lo, split). The recursion above just rewrote
        // exactly those factors, and S₁'s modes all come *before* S₂'s — so reading
        // them now yields the fresh values Gauss-Seidel prescribes.
        let child_right = {
            let kr_left = khatri_rao_range(factors, node.lo, split, cp_rank);
            contract_right_child(&partial, left_dim, right_dim, cp_rank, kr_left.as_ref())
        };
        drop(partial);
        self.descend(right, child_right, factors, cp_rank, update, last)
    }

    /// Compute a single mode's MTTKRP by walking only the root→leaf path.
    ///
    /// Costs `nnz·R + O(√nnz·R)` — the same FLOPs as the flat
    /// [`tenrso_kernels::mttkrp`] but without its full permuted tensor copy and full
    /// complement Khatri-Rao. Used for the Jacobi path's fit, where the post-sweep
    /// factors are *all* new and no sweep leaf can supply `⟨X, [[A]]⟩`.
    ///
    /// # Errors
    ///
    /// Shape/rank validation failures, `mode >= N`, or a matricization failure.
    pub(crate) fn mttkrp_mode<T>(
        &self,
        tensor: &ArrayView<T, IxDyn>,
        factors: &[Array2<T>],
        mode: usize,
    ) -> Result<Array2<T>, CpError>
    where
        T: Float + Send + Sync + 'static,
    {
        let cp_rank = self.validate(tensor, factors)?;
        let ndim = self.ndim();
        if mode >= ndim {
            return Err(CpError::ShapeMismatch(format!(
                "mode {mode} out of bounds for a tensor of order {ndim}"
            )));
        }
        if self.numel() == 0 {
            return Ok(Array2::<T>::zeros((self.shape[mode], cp_rank)));
        }

        let root = self.nodes[0];
        let mid = root
            .split
            .ok_or_else(|| CpError::ShapeMismatch("malformed dimension tree root".into()))?;
        let (left_idx, right_idx) = root
            .children
            .ok_or_else(|| CpError::ShapeMismatch("malformed dimension tree root".into()))?;

        let rows = self.range_prod(0, mid);
        let cols = self.range_prod(mid, ndim);
        let standard = tensor.as_standard_layout();
        let x2: ArrayView2<T> = standard
            .view()
            .into_shape_with_order((rows, cols))
            .map_err(|e| {
                CpError::ShapeMismatch(format!(
                    "could not matricize tensor for dimension tree: {e}"
                ))
            })?;

        let (mut node_idx, mut partial) = if mode < mid {
            let kr_right = khatri_rao_range(factors, mid, ndim, cp_rank);
            (left_idx, gemm_rows(&x2, kr_right.as_ref()))
        } else {
            let kr_left = khatri_rao_range(factors, 0, mid, cp_rank);
            (right_idx, gemm_cols(&x2, kr_left.as_ref()))
        };
        drop(standard);

        while let Some(split) = self.nodes[node_idx].split {
            let node = self.nodes[node_idx];
            let (left, right) = node
                .children
                .ok_or_else(|| CpError::ShapeMismatch("malformed dimension tree node".into()))?;
            let left_dim = self.range_prod(node.lo, split);
            let right_dim = self.range_prod(split, node.hi);

            if mode < split {
                let kr_right = khatri_rao_range(factors, split, node.hi, cp_rank);
                partial =
                    contract_left_child(&partial, left_dim, right_dim, cp_rank, kr_right.as_ref());
                node_idx = left;
            } else {
                let kr_left = khatri_rao_range(factors, node.lo, split, cp_rank);
                partial =
                    contract_right_child(&partial, left_dim, right_dim, cp_rank, kr_left.as_ref());
                node_idx = right;
            }
        }

        Ok(partial)
    }
}

// ─── Flat-buffer helpers ────────────────────────────────────────────────────

/// Row-major flat view of a matrix, copying only if it is not standard-layout.
///
/// Every matrix this module builds is standard-layout, and so is every CP factor
/// produced by the solvers, so the `Owned` arm is a safety net rather than a cost
/// centre — but it keeps the kernels total (no panic, no unwrap).
fn as_flat<T: Copy>(matrix: &Array2<T>) -> Cow<'_, [T]> {
    match matrix.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(matrix.iter().copied().collect()),
    }
}

/// `KR([lo, hi)) = A_lo ⊙ … ⊙ A_{hi-1}` in **forward** mode order, so that row `i`
/// is indexed by the row-major mixed-radix index over the modes `[lo, hi)` — which
/// is exactly the column order of the row-major matricizations used throughout.
///
/// Borrows (zero copy) for a single mode.
///
/// # Complexity
///
/// `O(Π_{j∈[lo,hi)} I_j · R)` sequential multiply-writes.
fn khatri_rao_range<T>(
    factors: &[Array2<T>],
    lo: usize,
    hi: usize,
    cp_rank: usize,
) -> Cow<'_, Array2<T>>
where
    T: Float + 'static,
{
    debug_assert!(lo < hi);
    if hi - lo == 1 {
        return Cow::Borrowed(&factors[lo]);
    }

    let mut acc: Array2<T> = factors[lo].clone();
    for factor in &factors[lo + 1..hi] {
        acc = khatri_rao_row_major(&acc, factor, cp_rank);
    }
    Cow::Owned(acc)
}

/// Row-major Khatri-Rao: `out[(i·J + j), :] = a[i, :] * b[j, :]`.
fn khatri_rao_row_major<T>(a: &Array2<T>, b: &Array2<T>, cp_rank: usize) -> Array2<T>
where
    T: Float + 'static,
{
    let rows_a = a.shape()[0];
    let rows_b = b.shape()[0];
    let a_data = as_flat(a);
    let b_data = as_flat(b);

    let mut out = vec![T::zero(); rows_a * rows_b * cp_rank];
    for row_a in 0..rows_a {
        let a_row = &a_data[row_a * cp_rank..(row_a + 1) * cp_rank];
        let slab = &mut out[row_a * rows_b * cp_rank..(row_a + 1) * rows_b * cp_rank];
        for row_b in 0..rows_b {
            let b_row = &b_data[row_b * cp_rank..(row_b + 1) * cp_rank];
            let out_row = &mut slab[row_b * cp_rank..(row_b + 1) * cp_rank];
            for ((o, &av), &bv) in out_row.iter_mut().zip(a_row.iter()).zip(b_row.iter()) {
                *o = av * bv;
            }
        }
    }

    from_flat(rows_a * rows_b, cp_rank, out)
}

/// Wrap a flat row-major buffer of exactly `rows * cols` elements as an `Array2`.
///
/// The length always matches by construction at every call site; the fallback
/// allocates a correctly shaped zero matrix rather than panicking (no-unwrap
/// policy), which is unreachable in practice.
#[inline]
fn from_flat<T: Float>(rows: usize, cols: usize, data: Vec<T>) -> Array2<T> {
    debug_assert_eq!(data.len(), rows * cols);
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::<T>::zeros((rows, cols)))
}

// ─── Contraction kernels ────────────────────────────────────────────────────

/// `q[i₁, r] = Σ_{i₂} P[i₁·b + i₂, r] · KR₂[i₂, r]` — the **left** child.
///
/// # Complexity
///
/// `a·b·R` multiply-adds; one streaming pass over `P`, accumulator row hot in L1.
fn contract_left_child<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_right: &Array2<T>,
) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    #[cfg(feature = "parallel")]
    if left_dim * right_dim * cp_rank >= PARALLEL_WORK_THRESHOLD && left_dim > 1 {
        return contract_left_child_parallel(partial, left_dim, right_dim, cp_rank, kr_right);
    }

    let p_data = as_flat(partial);
    let k_data = as_flat(kr_right);
    let mut acc = vec![T::zero(); left_dim * cp_rank];

    for row_left in 0..left_dim {
        let slab = &p_data[row_left * right_dim * cp_rank..(row_left + 1) * right_dim * cp_rank];
        let out_row = &mut acc[row_left * cp_rank..(row_left + 1) * cp_rank];
        for row_right in 0..right_dim {
            let p_row = &slab[row_right * cp_rank..(row_right + 1) * cp_rank];
            let k_row = &k_data[row_right * cp_rank..(row_right + 1) * cp_rank];
            for ((o, &pv), &kv) in out_row.iter_mut().zip(p_row.iter()).zip(k_row.iter()) {
                *o = *o + pv * kv;
            }
        }
    }

    from_flat(left_dim, cp_rank, acc)
}

/// `q[i₂, r] = Σ_{i₁} P[i₁·b + i₂, r] · KR₁[i₁, r]` — the **right** child.
///
/// # Complexity
///
/// `a·b·R` multiply-adds; one streaming pass over `P`, the `KR₁` row stays hot.
fn contract_right_child<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_left: &Array2<T>,
) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    #[cfg(feature = "parallel")]
    if left_dim * right_dim * cp_rank >= PARALLEL_WORK_THRESHOLD && right_dim > 1 {
        return contract_right_child_parallel(partial, left_dim, right_dim, cp_rank, kr_left);
    }

    let p_data = as_flat(partial);
    let k_data = as_flat(kr_left);
    let mut acc = vec![T::zero(); right_dim * cp_rank];

    for row_left in 0..left_dim {
        let slab = &p_data[row_left * right_dim * cp_rank..(row_left + 1) * right_dim * cp_rank];
        let k_row = &k_data[row_left * cp_rank..(row_left + 1) * cp_rank];
        for row_right in 0..right_dim {
            let p_row = &slab[row_right * cp_rank..(row_right + 1) * cp_rank];
            let out_row = &mut acc[row_right * cp_rank..(row_right + 1) * cp_rank];
            for ((o, &pv), &kv) in out_row.iter_mut().zip(p_row.iter()).zip(k_row.iter()) {
                *o = *o + pv * kv;
            }
        }
    }

    from_flat(right_dim, cp_rank, acc)
}

// ─── GEMM wrappers ──────────────────────────────────────────────────────────

/// `X₂ · KR` — the root's *left* child. Serial `dot` dispatches to the pure-Rust
/// `matrixmultiply` micro-kernel for `f32`/`f64`.
#[cfg(not(feature = "parallel"))]
fn gemm_rows<T>(x2: &ArrayView2<T>, kr: &Array2<T>) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    x2.dot(kr)
}

/// `X₂ᵀ · KR` — the root's *right* child.
#[cfg(not(feature = "parallel"))]
fn gemm_cols<T>(x2: &ArrayView2<T>, kr: &Array2<T>) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    x2.t().dot(kr)
}

/// Below this many multiply-adds, Rayon's fork/join overhead dominates.
#[cfg(feature = "parallel")]
const PARALLEL_WORK_THRESHOLD: usize = 1 << 16;

/// Row-block size targeting ~4 blocks per worker thread so Rayon can steal.
#[cfg(feature = "parallel")]
fn parallel_chunk_rows(rows: usize) -> usize {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    rows.div_ceil(threads.saturating_mul(4).max(1)).max(1)
}

/// `X₂ · KR`, blocked over the output rows.
///
/// This GEMM and [`gemm_cols`] carry `2·nnz·R` of the sweep's `2·nnz·R + O(√nnz·R)`
/// total work, so this is the only place where thread scaling really matters.
#[cfg(feature = "parallel")]
fn gemm_rows<T>(x2: &ArrayView2<T>, kr: &Array2<T>) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    use scirs2_core::ndarray_ext::s;
    use scirs2_core::parallel_ops::*;

    let rows = x2.shape()[0];
    let cp_rank = kr.shape()[1];
    if rows * x2.shape()[1] * cp_rank < PARALLEL_WORK_THRESHOLD || rows < 2 {
        return x2.dot(kr);
    }

    let chunk = parallel_chunk_rows(rows);
    let n_chunks = rows.div_ceil(chunk);

    let blocks: Vec<Array2<T>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk;
            let end = (start + chunk).min(rows);
            x2.slice(s![start..end, ..]).dot(kr)
        })
        .collect();

    let mut out = Array2::<T>::zeros((rows, cp_rank));
    for (c, block) in blocks.into_iter().enumerate() {
        let start = c * chunk;
        let end = (start + chunk).min(rows);
        out.slice_mut(s![start..end, ..]).assign(&block);
    }
    out
}

/// `X₂ᵀ · KR`, blocked over the output rows (i.e. over the *columns* of `X₂`).
#[cfg(feature = "parallel")]
fn gemm_cols<T>(x2: &ArrayView2<T>, kr: &Array2<T>) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    use scirs2_core::ndarray_ext::s;
    use scirs2_core::parallel_ops::*;

    let cols = x2.shape()[1];
    let cp_rank = kr.shape()[1];
    if x2.shape()[0] * cols * cp_rank < PARALLEL_WORK_THRESHOLD || cols < 2 {
        return x2.t().dot(kr);
    }

    let chunk = parallel_chunk_rows(cols);
    let n_chunks = cols.div_ceil(chunk);

    let blocks: Vec<Array2<T>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk;
            let end = (start + chunk).min(cols);
            x2.slice(s![.., start..end]).t().dot(kr)
        })
        .collect();

    let mut out = Array2::<T>::zeros((cols, cp_rank));
    for (c, block) in blocks.into_iter().enumerate() {
        let start = c * chunk;
        let end = (start + chunk).min(cols);
        out.slice_mut(s![start..end, ..]).assign(&block);
    }
    out
}

/// Parallel [`contract_left_child`]: the output rows are independent, so block `i₁`.
#[cfg(feature = "parallel")]
fn contract_left_child_parallel<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_right: &Array2<T>,
) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    use scirs2_core::parallel_ops::*;

    let p_data = as_flat(partial);
    let k_data = as_flat(kr_right);
    let chunk = parallel_chunk_rows(left_dim);
    let n_chunks = left_dim.div_ceil(chunk);

    let blocks: Vec<Vec<T>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk;
            let end = (start + chunk).min(left_dim);
            let mut acc = vec![T::zero(); (end - start) * cp_rank];
            for row_left in start..end {
                let slab =
                    &p_data[row_left * right_dim * cp_rank..(row_left + 1) * right_dim * cp_rank];
                let out_row =
                    &mut acc[(row_left - start) * cp_rank..(row_left - start + 1) * cp_rank];
                for row_right in 0..right_dim {
                    let p_row = &slab[row_right * cp_rank..(row_right + 1) * cp_rank];
                    let k_row = &k_data[row_right * cp_rank..(row_right + 1) * cp_rank];
                    for ((o, &pv), &kv) in out_row.iter_mut().zip(p_row.iter()).zip(k_row.iter()) {
                        *o = *o + pv * kv;
                    }
                }
            }
            acc
        })
        .collect();

    let mut acc = Vec::with_capacity(left_dim * cp_rank);
    for block in blocks {
        acc.extend_from_slice(&block);
    }
    from_flat(left_dim, cp_rank, acc)
}

/// Parallel [`contract_right_child`]: the reduction runs over `i₁`, so we block the
/// *output* rows `i₂` and let every worker sweep all of `i₁` within its band.
#[cfg(feature = "parallel")]
fn contract_right_child_parallel<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_left: &Array2<T>,
) -> Array2<T>
where
    T: Float + Send + Sync + 'static,
{
    use scirs2_core::parallel_ops::*;

    let p_data = as_flat(partial);
    let k_data = as_flat(kr_left);
    let chunk = parallel_chunk_rows(right_dim);
    let n_chunks = right_dim.div_ceil(chunk);

    let blocks: Vec<Vec<T>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk;
            let end = (start + chunk).min(right_dim);
            let mut acc = vec![T::zero(); (end - start) * cp_rank];
            for row_left in 0..left_dim {
                let k_row = &k_data[row_left * cp_rank..(row_left + 1) * cp_rank];
                let slab =
                    &p_data[row_left * right_dim * cp_rank..(row_left + 1) * right_dim * cp_rank];
                for row_right in start..end {
                    let p_row = &slab[row_right * cp_rank..(row_right + 1) * cp_rank];
                    let out_row =
                        &mut acc[(row_right - start) * cp_rank..(row_right - start + 1) * cp_rank];
                    for ((o, &pv), &kv) in out_row.iter_mut().zip(p_row.iter()).zip(k_row.iter()) {
                        *o = *o + pv * kv;
                    }
                }
            }
            acc
        })
        .collect();

    let mut acc = Vec::with_capacity(right_dim * cp_rank);
    for block in blocks {
        acc.extend_from_slice(&block);
    }
    from_flat(right_dim, cp_rank, acc)
}
