//! Dense **dimension-tree (memoized) MTTKRP** — all N mode-MTTKRPs of one CP-ALS
//! sweep computed with shared partial contractions.
//!
//! This is the dense analogue of the CSF fiber-tree walk in
//! [`crate::mttkrp_sparse_csf()`]: instead of amortizing partial Khatri-Rao products
//! across shared *fiber prefixes* of a sparse tensor, we amortize *partial
//! contractions of the tensor itself* across the modes of a balanced binary tree
//! (the Phan–Tichavský–Cichocki "fast ALS" / `DTREE` scheme).
//!
//! # The math
//!
//! For a tensor `X ∈ R^{I_0 × … × I_{N-1}}` (row-major / C order) and factor
//! matrices `A_k ∈ R^{I_k × R}`, the mode-`k` MTTKRP is
//!
//! ```text
//! M_k[i_k, r] = Σ_{i_j : j ≠ k}  X[i_0,…,i_{N-1}] · Π_{j ≠ k} A_j[i_j, r]
//! ```
//!
//! which is exactly what [`crate::mttkrp()`] computes as
//! `unfold_k(X) · (A_0 ⊙ … ⊙ A_{k-1} ⊙ A_{k+1} ⊙ … ⊙ A_{N-1})`.
//!
//! ## Partial contractions
//!
//! For a **contiguous** set of modes `S = [lo, hi)` define the *partial
//! contraction* (a matrix with `∏_{j∈S} I_j` rows and `R` columns, whose row index
//! is the row-major mixed-radix index of the modes in `S`):
//!
//! ```text
//! P_S[i_S, r] = Σ_{i_j : j ∉ S}  X[i] · Π_{j ∉ S} A_j[i_j, r]
//! ```
//!
//! Two facts make the tree work:
//!
//! 1. **Leaves are answers.** `P_{{k}} = M_k`, by definition.
//! 2. **Children come from their parent.** If `S = S₁ ⊎ S₂` (contiguous, `S₁`
//!    first), then with `KR(S) = ⊙_{j∈S} A_j` (Khatri-Rao in forward mode order,
//!    so its row index is again the row-major index over `S`):
//!
//!    ```text
//!    P_{S₁}[i₁, r] = Σ_{i₂} P_S[i₁·|S₂| + i₂, r] · KR(S₂)[i₂, r]
//!    P_{S₂}[i₂, r] = Σ_{i₁} P_S[i₁·|S₂| + i₂, r] · KR(S₁)[i₁, r]
//!    ```
//!
//!    (Proof: substitute the definition of `P_S` and note `S₁ᶜ = Sᶜ ⊎ S₂`.)
//!
//! The root `S = [0, N)` has an empty complement, so `P_root` would be `X`
//! replicated `R` times; we never form it. Instead the root's two children are
//! obtained directly with **two GEMMs** against the row-major matricization
//! `X₂ ∈ R^{(∏_{j<mid} I_j) × (∏_{j≥mid} I_j)}` (a zero-copy reshape of `X` when
//! `X` is C-contiguous):
//!
//! ```text
//! P_left  = X₂  · KR([mid, N))      (∏_left × R)
//! P_right = X₂ᵀ · KR([0, mid))      (∏_right × R)
//! ```
//!
//! Every deeper node is then a pure element-wise-in-`r` reduction (no GEMM shape
//! exists there because `r` is shared, not contracted), fused so that a node's two
//! children are produced in a **single streaming pass** over the parent's partial.
//!
//! # Complexity (honest accounting)
//!
//! Let `nnz = ∏_j I_j`, `p_S = ∏_{j∈S} I_j`.
//!
//! | stage | multiply-adds |
//! |---|---|
//! | root (2 GEMMs) | `2·nnz·R` |
//! | root Khatri-Rao builds | `(p_left + p_right)·R` |
//! | internal node `S` | `2·p_S·R` (both children, one pass) |
//! | all deeper levels | `Σ_{S internal, S≠root} 2·p_S·R` |
//!
//! Because the split is chosen to balance the *dimension products*, the level-1
//! nodes have `p_S ≈ √nnz`, and every deeper level shrinks geometrically. The
//! total is therefore
//!
//! ```text
//! 2·nnz·R + O(√nnz·R)   ≈  2·nnz·R
//! ```
//!
//! versus the naive loop `for k in 0..N { mttkrp(X, A, k) }` which costs
//!
//! ```text
//! N·nnz·R   (GEMMs)
//! + Σ_k (nnz/I_k)·R      (materializing a full complement Khatri-Rao per mode)
//! + N·nnz element copies  (a full permuted copy of X per mode, in `unfold_tensor`)
//! ```
//!
//! So the **FLOP** ratio is `N/2` (1.5× for `N = 3`, 2× for `N = 4`, 2.5× for
//! `N = 5`, …), and on top of that this kernel eliminates `N-1` full tensor copies
//! and all `N` full complement-Khatri-Rao materializations. Peak working memory is
//! `O(max_S p_S · R) ≈ O(√nnz · R)` instead of the naive `O(nnz + (nnz/I_min)·R)`.
//!
//! # Numerical stability
//!
//! The tree changes the *summation order* relative to the flat reference kernel
//! (which reduces over one flattened complement index of length `nnz/I_k`). The
//! tree instead reduces in nested stages of length `p_{S₂}`, `p_{S₁}`, … — a
//! pairwise-style split whose worst-case rounding error grows like
//! `O(ε · Σ_levels p)` rather than `O(ε · nnz/I_k)`, i.e. it is never worse and is
//! typically better conditioned than the reference. Additionally the largest
//! Khatri-Rao product it ever materializes spans only half the modes, which halves
//! the number of factor entries multiplied together before any tensor value damps
//! them — reducing the dynamic range of the intermediates (relevant for high-order
//! tensors where `Π_{j≠k} A_j[i_j, r]` can under/overflow).
//!
//! Results therefore agree with [`crate::mttkrp()`] to floating-point tolerance, not
//! bit-for-bit.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`; parallelism uses
//! `scirs2_core::parallel_ops`. Direct use of `ndarray`/`rand` is forbidden per
//! SCIRS2_INTEGRATION_POLICY.md.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::{Num, One, Zero};
use std::borrow::Cow;

// ─── Tree structure ─────────────────────────────────────────────────────────

/// One node of the dimension tree: a contiguous mode range `[lo, hi)`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DimTreeNode {
    /// First mode of this node (inclusive).
    lo: usize,
    /// One past the last mode of this node (exclusive).
    hi: usize,
    /// Split point: `lo < split < hi`. `None` for a leaf (`hi - lo == 1`).
    split: Option<usize>,
    /// Arena indices of the `[lo, split)` and `[split, hi)` children.
    children: Option<(usize, usize)>,
}

/// A reusable balanced binary **dimension tree** over the modes of a tensor.
///
/// Build it once for a fixed tensor shape and reuse it across every CP-ALS
/// iteration (the tensor is fixed; only the factor matrices change). The tree
/// itself is `O(N)` to build and holds no tensor data.
///
/// # Split rule (deterministic)
///
/// Each internal node `[lo, hi)` is split at the point that **minimizes the larger
/// of the two child dimension-products**, `max(∏_{lo..mid}, ∏_{mid..hi})`, with
/// ties broken toward the smaller `mid`. Balancing the *products* (not the mode
/// counts) is what keeps both the Khatri-Rao materializations and the intermediate
/// partials at `O(√nnz · R)`; it also equalizes the two root GEMM shapes.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2, IxDyn};
/// use tenrso_kernels::{mttkrp, DimTree};
///
/// let shape = [4usize, 3, 5];
/// let n: usize = shape.iter().product();
/// let x = Array::from_shape_vec(IxDyn(&shape), (0..n).map(|v| v as f64).collect())?;
/// let factors: Vec<Array2<f64>> = shape
///     .iter()
///     .map(|&d| Array2::from_shape_fn((d, 2), |(i, r)| (i + r) as f64))
///     .collect();
/// let views: Vec<_> = factors.iter().map(|f| f.view()).collect();
///
/// let tree = DimTree::new(&shape)?;
/// let all = tree.mttkrp_all(&x.view(), &views)?;
///
/// for mode in 0..3 {
///     let reference = mttkrp(&x.view(), &views, mode)?;
///     for (a, b) in all[mode].iter().zip(reference.iter()) {
///         assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()));
///     }
/// }
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct DimTree {
    /// Tensor shape this tree was built for.
    shape: Vec<usize>,
    /// Node arena; `nodes[0]` is the root (`[0, N)`).
    nodes: Vec<DimTreeNode>,
    /// Exclusive prefix products of `shape` (`prefix[i] = ∏_{j<i} shape[j]`).
    prefix: Vec<usize>,
}

impl DimTree {
    /// Build a dimension tree for a tensor of the given shape.
    ///
    /// # Errors
    ///
    /// * `shape.len() < 2` — MTTKRP is undefined for order-1 tensors (there is no
    ///   Khatri-Rao product to take), matching [`crate::mttkrp()`]'s "need at least 2
    ///   factor matrices" contract.
    /// * the dimension product overflows `usize`.
    ///
    /// # Complexity
    ///
    /// Time and space `O(N)` (`N` = tensor order); independent of the tensor size.
    pub fn new(shape: &[usize]) -> Result<Self> {
        let ndim = shape.len();
        if ndim < 2 {
            anyhow::bail!(
                "DimTree requires a tensor of order >= 2 (got order {}); MTTKRP needs at least 2 factor matrices",
                ndim
            );
        }

        // Exclusive prefix products, checked for overflow.
        let mut prefix = Vec::with_capacity(ndim + 1);
        prefix.push(1usize);
        for (axis, &dim) in shape.iter().enumerate() {
            let acc = prefix[axis].checked_mul(dim).ok_or_else(|| {
                anyhow::anyhow!("tensor dimension product overflows usize at axis {}", axis)
            })?;
            prefix.push(acc);
        }

        let mut tree = DimTree {
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
        self.nodes.push(DimTreeNode {
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

    /// Pick the split of `[lo, hi)` minimizing `max(∏_{lo..mid}, ∏_{mid..hi})`.
    ///
    /// Comparison is done in log-space (`f64`) so that it is well defined even when
    /// an individual half-product is astronomically large; ties (including the
    /// perfectly balanced case) resolve to the smallest `mid`, which makes the tree
    /// deterministic and reproducible.
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
            let right_log = total - left_log;
            // Strictly-less keeps the smallest `mid` on ties.
            let cost = left_log.max(right_log);
            if cost < best_cost - 1e-12 {
                best_cost = cost;
                best_mid = lo + offset + 1;
            }
        }
        best_mid
    }

    /// The tensor shape this tree was built for.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The tensor order `N`.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements of the tensor this tree was built for.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.prefix[self.shape.len()]
    }

    /// `∏_{j ∈ [lo, hi)} I_j`.
    #[inline]
    fn range_prod(&self, lo: usize, hi: usize) -> usize {
        debug_assert!(lo <= hi && hi <= self.shape.len());
        if self.prefix[lo] == 0 {
            // A zero-sized leading dimension makes prefix products collapse; fall
            // back to the direct product (the empty-tensor path never uses this).
            return self.shape[lo..hi].iter().product();
        }
        self.prefix[hi] / self.prefix[lo]
    }

    /// Validate tensor/factor shapes and return the CP rank `R`.
    fn validate<T>(&self, tensor: &ArrayView<T, IxDyn>, factors: &[ArrayView2<T>]) -> Result<usize>
    where
        T: Copy,
    {
        if tensor.shape() != self.shape.as_slice() {
            anyhow::bail!(
                "tensor shape {:?} does not match the shape this DimTree was built for ({:?})",
                tensor.shape(),
                self.shape
            );
        }
        if factors.len() != self.shape.len() {
            anyhow::bail!(
                "Number of factor matrices ({}) must match tensor rank ({})",
                factors.len(),
                self.shape.len()
            );
        }
        let cp_rank = factors[0].shape()[1];
        for (i, factor) in factors.iter().enumerate() {
            if factor.shape()[1] != cp_rank {
                anyhow::bail!(
                    "Factor matrix {} has {} columns, expected {}",
                    i,
                    factor.shape()[1],
                    cp_rank
                );
            }
            if factor.shape()[0] != self.shape[i] {
                anyhow::bail!(
                    "Factor matrix {} has {} rows, expected {} (tensor mode-{} size)",
                    i,
                    factor.shape()[0],
                    self.shape[i],
                    i
                );
            }
        }
        Ok(cp_rank)
    }

    /// All-zero MTTKRP results, used when the tensor has no elements (an empty sum
    /// is exactly zero, so this is not a shortcut but the correct answer).
    fn empty_results<T>(&self, cp_rank: usize) -> Vec<Array2<T>>
    where
        T: Copy + Zero,
    {
        self.shape
            .iter()
            .map(|&dim| Array2::<T>::zeros((dim, cp_rank)))
            .collect()
    }

    /// Shape of the root matricization `X₂`: `(∏_{j<mid} I_j, ∏_{j≥mid} I_j)`.
    ///
    /// The reshape of `X` into `X₂` is zero-copy when `X` is already C-contiguous;
    /// otherwise a single standard-layout copy is made (once per sweep — the naive
    /// kernel makes one such copy *per mode*).
    #[inline]
    fn root_matrix_shape(&self, mid: usize) -> (usize, usize) {
        (
            self.range_prod(0, mid),
            self.range_prod(mid, self.shape.len()),
        )
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Compute **all N mode-MTTKRPs** of one CP-ALS sweep, sharing partial contractions.
///
/// `result[k]` equals `mttkrp(tensor, factors, k)` (to floating-point tolerance)
/// but the whole sweep costs `≈ 2·nnz·R` multiply-adds instead of `N·nnz·R`, and
/// never materializes a full complement Khatri-Rao product or a full tensor copy
/// per mode. See the [module docs](self) for the derivation and the honest
/// complexity accounting.
///
/// Convenience wrapper around [`DimTree::new`] + [`DimTree::mttkrp_all`]; if you
/// run many ALS iterations on a fixed tensor, build the [`DimTree`] once instead.
///
/// # Errors
///
/// Same contract as [`crate::mttkrp()`]: tensor order `< 2`, factor count mismatch,
/// inconsistent CP rank, or a factor whose row count disagrees with the tensor
/// dimension.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2, IxDyn};
/// use tenrso_kernels::mttkrp_all_modes;
///
/// let x = Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|v| v as f64).collect())?;
/// let a0 = Array2::<f64>::from_shape_fn((2, 2), |(i, r)| (i + r) as f64);
/// let a1 = Array2::<f64>::from_shape_fn((3, 2), |(i, r)| (i * r + 1) as f64);
/// let a2 = Array2::<f64>::from_shape_fn((4, 2), |(i, r)| (i as f64) - (r as f64));
///
/// let all = mttkrp_all_modes(&x.view(), &[a0.view(), a1.view(), a2.view()])?;
/// assert_eq!(all.len(), 3);
/// assert_eq!(all[0].shape(), &[2, 2]);
/// assert_eq!(all[1].shape(), &[3, 2]);
/// assert_eq!(all[2].shape(), &[4, 2]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn mttkrp_all_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
) -> Result<Vec<Array2<T>>>
where
    T: Copy + Num + One + Zero + 'static,
{
    let tree = DimTree::new(tensor.shape())?;
    tree.mttkrp_all(tensor, factors)
}

/// Parallel (Rayon) counterpart of [`mttkrp_all_modes`].
///
/// See [`DimTree::mttkrp_all_parallel`] for what exactly is parallelized.
///
/// # Errors
///
/// Same as [`mttkrp_all_modes`].
#[cfg(feature = "parallel")]
pub fn mttkrp_all_modes_parallel<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
) -> Result<Vec<Array2<T>>>
where
    T: Copy + Num + One + Zero + Send + Sync + 'static,
{
    let tree = DimTree::new(tensor.shape())?;
    tree.mttkrp_all_parallel(tensor, factors)
}

impl DimTree {
    /// Compute all `N` mode-MTTKRPs with shared partial contractions.
    ///
    /// # Errors
    ///
    /// Tensor shape mismatch with the tree, factor count mismatch, inconsistent CP
    /// rank, or factor row/dimension mismatch.
    ///
    /// # Complexity
    ///
    /// `2·nnz·R + O(√nnz·R)` multiply-adds; `O(√nnz·R)` peak temporary memory
    /// (plus one standard-layout copy of the tensor iff it is not C-contiguous).
    pub fn mttkrp_all<T>(
        &self,
        tensor: &ArrayView<T, IxDyn>,
        factors: &[ArrayView2<T>],
    ) -> Result<Vec<Array2<T>>>
    where
        T: Copy + Num + One + Zero + 'static,
    {
        let cp_rank = self.validate(tensor, factors)?;
        if self.numel() == 0 {
            return Ok(self.empty_results(cp_rank));
        }

        let owned = standard_factors(factors);
        let mut out: Vec<Option<Array2<T>>> = vec![None; self.shape.len()];

        let root = &self.nodes[0];
        let mid = root
            .split
            .expect("root of an order>=2 tree is always internal");
        let (left_idx, right_idx) = root
            .children
            .expect("root of an order>=2 tree always has children");

        let (rows, cols) = self.root_matrix_shape(mid);
        let standard = tensor.as_standard_layout();
        let x2: ArrayView2<T> = standard.view().into_shape_with_order((rows, cols))?;

        // P_left = X₂ · KR([mid, N))  — a (rows × cols) · (cols × R) GEMM.
        let partial_left = {
            let kr_right = khatri_rao_range(&owned, mid, self.shape.len(), cp_rank);
            x2.dot(kr_right.as_ref())
        };
        // P_right = X₂ᵀ · KR([0, mid))  — a (cols × rows) · (rows × R) GEMM.
        let partial_right = {
            let kr_left = khatri_rao_range(&owned, 0, mid, cp_rank);
            x2.t().dot(kr_left.as_ref())
        };

        self.descend(left_idx, partial_left, &owned, cp_rank, &mut out);
        self.descend(right_idx, partial_right, &owned, cp_rank, &mut out);

        collect_leaves(out, self.shape.len())
    }

    /// Compute the MTTKRP for a single `mode`, walking only the root→leaf path.
    ///
    /// Useful when a caller wants one mode at a time but still wants to avoid the
    /// naive kernel's full tensor copy and full complement Khatri-Rao. Costs
    /// `nnz·R + O(√nnz·R)` multiply-adds — about half of [`mttkrp_all`] and about
    /// the same FLOPs as one call to [`crate::mttkrp()`], but with far less memory
    /// traffic. To get *all* modes, call [`mttkrp_all`]: it is ~`N/2` times cheaper
    /// than calling this `N` times.
    ///
    /// [`mttkrp_all`]: DimTree::mttkrp_all
    ///
    /// # Errors
    ///
    /// As [`DimTree::mttkrp_all`], plus `mode >= N`.
    pub fn mttkrp_mode<T>(
        &self,
        tensor: &ArrayView<T, IxDyn>,
        factors: &[ArrayView2<T>],
        mode: usize,
    ) -> Result<Array2<T>>
    where
        T: Copy + Num + One + Zero + 'static,
    {
        let cp_rank = self.validate(tensor, factors)?;
        let ndim = self.shape.len();
        if mode >= ndim {
            anyhow::bail!("Mode {} out of bounds for tensor with rank {}", mode, ndim);
        }
        if self.numel() == 0 {
            return Ok(Array2::<T>::zeros((self.shape[mode], cp_rank)));
        }

        let owned = standard_factors(factors);

        let root = &self.nodes[0];
        let mid = root
            .split
            .expect("root of an order>=2 tree is always internal");
        let (left_idx, right_idx) = root
            .children
            .expect("root of an order>=2 tree always has children");

        let (rows, cols) = self.root_matrix_shape(mid);
        let standard = tensor.as_standard_layout();
        let x2: ArrayView2<T> = standard.view().into_shape_with_order((rows, cols))?;

        let (mut node_idx, mut partial) = if mode < mid {
            let kr_right = khatri_rao_range(&owned, mid, ndim, cp_rank);
            (left_idx, x2.dot(kr_right.as_ref()))
        } else {
            let kr_left = khatri_rao_range(&owned, 0, mid, cp_rank);
            (right_idx, x2.t().dot(kr_left.as_ref()))
        };
        drop(standard);

        // Walk down the single path to the leaf, contracting away one sibling
        // subtree per level.
        while let Some(split) = self.nodes[node_idx].split {
            let node = self.nodes[node_idx].clone();
            let (left, right) = node
                .children
                .expect("internal node always has children by construction");
            let left_dim = self.range_prod(node.lo, split);
            let right_dim = self.range_prod(split, node.hi);

            if mode < split {
                let kr_right = khatri_rao_range(&owned, split, node.hi, cp_rank);
                partial =
                    contract_left_child(&partial, left_dim, right_dim, cp_rank, kr_right.as_ref());
                node_idx = left;
            } else {
                let kr_left = khatri_rao_range(&owned, node.lo, split, cp_rank);
                partial =
                    contract_right_child(&partial, left_dim, right_dim, cp_rank, kr_left.as_ref());
                node_idx = right;
            }
        }

        Ok(partial)
    }

    /// Depth-first descent: turn a node's partial contraction into its children's,
    /// storing leaves (= the mode MTTKRPs) into `out`.
    fn descend<T>(
        &self,
        node_idx: usize,
        partial: Array2<T>,
        owned: &[Array2<T>],
        cp_rank: usize,
        out: &mut [Option<Array2<T>>],
    ) where
        T: Copy + Num + One + Zero + 'static,
    {
        let node = &self.nodes[node_idx];
        let Some(split) = node.split else {
            // Leaf: `P_{{k}} = M_k`.
            out[node.lo] = Some(partial);
            return;
        };
        let (left, right) = node
            .children
            .expect("internal node always has children by construction");
        let (lo, hi) = (node.lo, node.hi);
        let left_dim = self.range_prod(lo, split);
        let right_dim = self.range_prod(split, hi);

        let kr_left = khatri_rao_range(owned, lo, split, cp_rank);
        let kr_right = khatri_rao_range(owned, split, hi, cp_rank);

        // One streaming pass over `partial` produces BOTH children.
        let (child_left, child_right) = contract_both_children(
            &partial,
            left_dim,
            right_dim,
            cp_rank,
            kr_left.as_ref(),
            kr_right.as_ref(),
        );
        drop(partial);
        drop(kr_left);
        drop(kr_right);

        self.descend(left, child_left, owned, cp_rank, out);
        self.descend(right, child_right, owned, cp_rank, out);
    }
}

// ─── Private kernels (serial) ───────────────────────────────────────────────

/// Materialize the factors as standard-layout owned matrices so that every inner
/// kernel can work on flat `&[T]` slices (the input views may be transposed or
/// otherwise strided). Cost `Σ_k I_k·R` — negligible next to `nnz·R`.
fn standard_factors<T>(factors: &[ArrayView2<T>]) -> Vec<Array2<T>>
where
    T: Copy + 'static,
{
    factors
        .iter()
        .map(|f| f.as_standard_layout().into_owned())
        .collect()
}

/// `KR([lo, hi)) = A_lo ⊙ A_{lo+1} ⊙ … ⊙ A_{hi-1}` in **forward** mode order, so
/// that row `i` of the result is indexed by the row-major mixed-radix index of the
/// modes `[lo, hi)` — exactly matching the column order of the row-major
/// matricization used everywhere in this module (and of [`crate::mttkrp()`]'s
/// `unfold_tensor`).
///
/// Borrows (zero copy) when the range is a single mode.
///
/// # Complexity
///
/// `O(∏_{j∈[lo,hi)} I_j · R)` multiply-writes, all sequential in memory — the fold
/// writes whole rows, unlike the column-wise `khatri_rao` in [`crate::khatri_rao`]
/// which strides by `R` on every store.
fn khatri_rao_range<T>(
    owned: &[Array2<T>],
    lo: usize,
    hi: usize,
    cp_rank: usize,
) -> Cow<'_, Array2<T>>
where
    T: Copy + Num + Zero + 'static,
{
    debug_assert!(lo < hi);
    if hi - lo == 1 {
        return Cow::Borrowed(&owned[lo]);
    }

    let mut acc: Array2<T> = owned[lo].clone();
    for factor in &owned[lo + 1..hi] {
        acc = khatri_rao_row_major(&acc, factor, cp_rank);
    }
    Cow::Owned(acc)
}

/// Row-major Khatri-Rao: `out[(i·J + j), :] = a[i, :] * b[j, :]`.
fn khatri_rao_row_major<T>(a: &Array2<T>, b: &Array2<T>, cp_rank: usize) -> Array2<T>
where
    T: Copy + Num + Zero + 'static,
{
    let rows_a = a.shape()[0];
    let rows_b = b.shape()[0];
    let a_data = a
        .as_slice()
        .expect("khatri_rao_row_major: `a` is built standard-layout by this module");
    let b_data = b
        .as_slice()
        .expect("khatri_rao_row_major: `b` is a standard-layout owned factor");

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

    Array2::from_shape_vec((rows_a * rows_b, cp_rank), out)
        .expect("khatri_rao_row_major: shape matches the buffer length by construction")
}

/// Flat row-major slice of a matrix that this module built (partial contractions,
/// Khatri-Rao products and the standardized factors are all standard-layout).
fn flat_slice<T>(matrix: &Array2<T>) -> &[T]
where
    T: Copy,
{
    matrix
        .as_slice()
        .expect("dimension-tree intermediates are standard-layout by construction")
}

/// Turn the per-mode leaf slots into the ordered result vector.
///
/// Every mode of the tree is a leaf and `descend` fills each leaf exactly once, so
/// a missing slot would mean the tree was malformed; we report that as an error
/// rather than panicking.
fn collect_leaves<T>(slots: Vec<Option<Array2<T>>>, ndim: usize) -> Result<Vec<Array2<T>>>
where
    T: Copy,
{
    let results: Vec<Array2<T>> = slots.into_iter().flatten().collect();
    if results.len() != ndim {
        anyhow::bail!(
            "dimension tree produced {} of {} mode results (malformed tree)",
            results.len(),
            ndim
        );
    }
    Ok(results)
}

/// `q[i₁, r] = Σ_{i₂} P[i₁·b + i₂, r] · KR₂[i₂, r]` — the *left* child.
///
/// # Complexity
///
/// `a·b·R` multiply-adds, one streaming pass over `P` (`P` rows and `KR₂` rows are
/// both read sequentially; the accumulator row stays hot in L1).
fn contract_left_child<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_right: &Array2<T>,
) -> Array2<T>
where
    T: Copy + Num + Zero + 'static,
{
    let p_data = flat_slice(partial);
    let k_data = flat_slice(kr_right);
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

    Array2::from_shape_vec((left_dim, cp_rank), acc)
        .expect("contract_left_child: shape matches the buffer length by construction")
}

/// `q[i₂, r] = Σ_{i₁} P[i₁·b + i₂, r] · KR₁[i₁, r]` — the *right* child.
///
/// # Complexity
///
/// `a·b·R` multiply-adds, one streaming pass over `P`; the accumulator is swept
/// sequentially in the inner loop (`b·R` elements), the `KR₁` row stays hot.
fn contract_right_child<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_left: &Array2<T>,
) -> Array2<T>
where
    T: Copy + Num + Zero + 'static,
{
    let p_data = flat_slice(partial);
    let k_data = flat_slice(kr_left);
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

    Array2::from_shape_vec((right_dim, cp_rank), acc)
        .expect("contract_right_child: shape matches the buffer length by construction")
}

/// Fused version of [`contract_left_child`] + [`contract_right_child`]: produces
/// **both** children in a single pass over the parent partial.
///
/// This is the memoization payoff — the parent's `a·b·R` elements are streamed once
/// and feed `2·a·b·R` multiply-adds, halving the memory traffic of the two separate
/// passes.
fn contract_both_children<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_left: &Array2<T>,
    kr_right: &Array2<T>,
) -> (Array2<T>, Array2<T>)
where
    T: Copy + Num + Zero + 'static,
{
    let p_data = flat_slice(partial);
    let kl_data = flat_slice(kr_left);
    let kr_data = flat_slice(kr_right);

    let mut acc_left = vec![T::zero(); left_dim * cp_rank];
    let mut acc_right = vec![T::zero(); right_dim * cp_rank];

    for row_left in 0..left_dim {
        let slab = &p_data[row_left * right_dim * cp_rank..(row_left + 1) * right_dim * cp_rank];
        let kl_row = &kl_data[row_left * cp_rank..(row_left + 1) * cp_rank];
        let left_row = &mut acc_left[row_left * cp_rank..(row_left + 1) * cp_rank];

        for row_right in 0..right_dim {
            let p_row = &slab[row_right * cp_rank..(row_right + 1) * cp_rank];
            let kr_row = &kr_data[row_right * cp_rank..(row_right + 1) * cp_rank];
            let right_row = &mut acc_right[row_right * cp_rank..(row_right + 1) * cp_rank];

            let zipped = left_row
                .iter_mut()
                .zip(right_row.iter_mut())
                .zip(p_row.iter())
                .zip(kl_row.iter())
                .zip(kr_row.iter());
            for ((((lv, rv), &pv), &klv), &krv) in zipped {
                *lv = *lv + pv * krv;
                *rv = *rv + pv * klv;
            }
        }
    }

    let left = Array2::from_shape_vec((left_dim, cp_rank), acc_left)
        .expect("contract_both_children: left shape matches the buffer length by construction");
    let right = Array2::from_shape_vec((right_dim, cp_rank), acc_right)
        .expect("contract_both_children: right shape matches the buffer length by construction");
    (left, right)
}

// ─── Private kernels (parallel) ─────────────────────────────────────────────

/// Row-chunk size targeting ~4 chunks per worker thread (Rayon steals the rest).
#[cfg(feature = "parallel")]
fn parallel_chunk_rows(rows: usize) -> usize {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let target_chunks = threads.saturating_mul(4).max(1);
    rows.div_ceil(target_chunks).max(1)
}

/// Below this many multiply-adds the Rayon fork/join overhead dominates.
#[cfg(feature = "parallel")]
const PARALLEL_WORK_THRESHOLD: usize = 1 << 16;

/// Parallel `X₂ · KR` (row-blocked over the output rows of `X₂`).
#[cfg(feature = "parallel")]
fn gemm_rows_parallel<T>(x2: &ArrayView2<T>, kr: &Array2<T>) -> Array2<T>
where
    T: Copy + Num + One + Zero + Send + Sync + 'static,
{
    use scirs2_core::ndarray_ext::s;
    use scirs2_core::parallel_ops::*;

    let rows = x2.shape()[0];
    let cp_rank = kr.shape()[1];
    let chunk = parallel_chunk_rows(rows);
    let n_chunks = rows.div_ceil(chunk.max(1));

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

/// Parallel `X₂ᵀ · KR` (blocked over the output rows, i.e. the *columns* of `X₂`).
#[cfg(feature = "parallel")]
fn gemm_cols_parallel<T>(x2: &ArrayView2<T>, kr: &Array2<T>) -> Array2<T>
where
    T: Copy + Num + One + Zero + Send + Sync + 'static,
{
    use scirs2_core::ndarray_ext::s;
    use scirs2_core::parallel_ops::*;

    let cols = x2.shape()[1];
    let cp_rank = kr.shape()[1];
    let chunk = parallel_chunk_rows(cols);
    let n_chunks = cols.div_ceil(chunk.max(1));

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

/// Parallel [`contract_left_child`]: the output rows are independent, so we simply
/// block over `i₁`.
#[cfg(feature = "parallel")]
fn contract_left_child_parallel<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_right: &Array2<T>,
) -> Array2<T>
where
    T: Copy + Num + Zero + Send + Sync + 'static,
{
    use scirs2_core::parallel_ops::*;

    if left_dim * right_dim * cp_rank < PARALLEL_WORK_THRESHOLD {
        return contract_left_child(partial, left_dim, right_dim, cp_rank, kr_right);
    }

    let p_data = flat_slice(partial);
    let k_data = flat_slice(kr_right);
    let chunk = parallel_chunk_rows(left_dim);
    let n_chunks = left_dim.div_ceil(chunk.max(1));

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
    Array2::from_shape_vec((left_dim, cp_rank), acc)
        .expect("contract_left_child_parallel: shape matches the buffer length by construction")
}

/// Parallel [`contract_right_child`]: the reduction runs over `i₁`, so we block
/// over the *output* rows `i₂` instead and let every worker sweep all of `i₁`
/// (each element of `P` is still read exactly once per worker's column band).
#[cfg(feature = "parallel")]
fn contract_right_child_parallel<T>(
    partial: &Array2<T>,
    left_dim: usize,
    right_dim: usize,
    cp_rank: usize,
    kr_left: &Array2<T>,
) -> Array2<T>
where
    T: Copy + Num + Zero + Send + Sync + 'static,
{
    use scirs2_core::parallel_ops::*;

    if left_dim * right_dim * cp_rank < PARALLEL_WORK_THRESHOLD {
        return contract_right_child(partial, left_dim, right_dim, cp_rank, kr_left);
    }

    let p_data = flat_slice(partial);
    let k_data = flat_slice(kr_left);
    let chunk = parallel_chunk_rows(right_dim);
    let n_chunks = right_dim.div_ceil(chunk.max(1));

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
    Array2::from_shape_vec((right_dim, cp_rank), acc)
        .expect("contract_right_child_parallel: shape matches the buffer length by construction")
}

#[cfg(feature = "parallel")]
impl DimTree {
    /// Parallel counterpart of [`DimTree::mttkrp_all`].
    ///
    /// Parallelism:
    /// * the two root GEMMs are blocked over their output rows (this is where
    ///   `2·nnz·R` of the `2·nnz·R + O(√nnz·R)` total work lives);
    /// * each deeper node's two children are produced by two parallel reductions
    ///   (blocked over the respective output rows), falling back to the fused
    ///   single-pass serial kernel below a fixed per-node work threshold
    ///   (`PARALLEL_WORK_THRESHOLD`, an internal constant) multiply-adds.
    ///
    /// Note the deeper levels are *deliberately* not fused here: fusing them would
    /// serialize the `i₁` reduction of the right child, and they account for
    /// `O(√nnz·R)` of the work anyway.
    ///
    /// # Errors
    ///
    /// As [`DimTree::mttkrp_all`].
    pub fn mttkrp_all_parallel<T>(
        &self,
        tensor: &ArrayView<T, IxDyn>,
        factors: &[ArrayView2<T>],
    ) -> Result<Vec<Array2<T>>>
    where
        T: Copy + Num + One + Zero + Send + Sync + 'static,
    {
        let cp_rank = self.validate(tensor, factors)?;
        if self.numel() == 0 {
            return Ok(self.empty_results(cp_rank));
        }

        let owned = standard_factors(factors);
        let mut out: Vec<Option<Array2<T>>> = vec![None; self.shape.len()];

        let root = &self.nodes[0];
        let mid = root
            .split
            .expect("root of an order>=2 tree is always internal");
        let (left_idx, right_idx) = root
            .children
            .expect("root of an order>=2 tree always has children");

        let (rows, cols) = self.root_matrix_shape(mid);
        let standard = tensor.as_standard_layout();
        let x2: ArrayView2<T> = standard.view().into_shape_with_order((rows, cols))?;

        let partial_left = {
            let kr_right = khatri_rao_range(&owned, mid, self.shape.len(), cp_rank);
            gemm_rows_parallel(&x2, kr_right.as_ref())
        };
        let partial_right = {
            let kr_left = khatri_rao_range(&owned, 0, mid, cp_rank);
            gemm_cols_parallel(&x2, kr_left.as_ref())
        };
        drop(standard);

        self.descend_parallel(left_idx, partial_left, &owned, cp_rank, &mut out);
        self.descend_parallel(right_idx, partial_right, &owned, cp_rank, &mut out);

        collect_leaves(out, self.shape.len())
    }

    /// Parallel counterpart of [`DimTree::mttkrp_mode`].
    ///
    /// # Errors
    ///
    /// As [`DimTree::mttkrp_mode`].
    pub fn mttkrp_mode_parallel<T>(
        &self,
        tensor: &ArrayView<T, IxDyn>,
        factors: &[ArrayView2<T>],
        mode: usize,
    ) -> Result<Array2<T>>
    where
        T: Copy + Num + One + Zero + Send + Sync + 'static,
    {
        let cp_rank = self.validate(tensor, factors)?;
        let ndim = self.shape.len();
        if mode >= ndim {
            anyhow::bail!("Mode {} out of bounds for tensor with rank {}", mode, ndim);
        }
        if self.numel() == 0 {
            return Ok(Array2::<T>::zeros((self.shape[mode], cp_rank)));
        }

        let owned = standard_factors(factors);

        let root = &self.nodes[0];
        let mid = root
            .split
            .expect("root of an order>=2 tree is always internal");
        let (left_idx, right_idx) = root
            .children
            .expect("root of an order>=2 tree always has children");

        let (rows, cols) = self.root_matrix_shape(mid);
        let standard = tensor.as_standard_layout();
        let x2: ArrayView2<T> = standard.view().into_shape_with_order((rows, cols))?;

        let (mut node_idx, mut partial) = if mode < mid {
            let kr_right = khatri_rao_range(&owned, mid, ndim, cp_rank);
            (left_idx, gemm_rows_parallel(&x2, kr_right.as_ref()))
        } else {
            let kr_left = khatri_rao_range(&owned, 0, mid, cp_rank);
            (right_idx, gemm_cols_parallel(&x2, kr_left.as_ref()))
        };
        drop(standard);

        while let Some(split) = self.nodes[node_idx].split {
            let node = self.nodes[node_idx].clone();
            let (left, right) = node
                .children
                .expect("internal node always has children by construction");
            let left_dim = self.range_prod(node.lo, split);
            let right_dim = self.range_prod(split, node.hi);

            if mode < split {
                let kr_right = khatri_rao_range(&owned, split, node.hi, cp_rank);
                partial = contract_left_child_parallel(
                    &partial,
                    left_dim,
                    right_dim,
                    cp_rank,
                    kr_right.as_ref(),
                );
                node_idx = left;
            } else {
                let kr_left = khatri_rao_range(&owned, node.lo, split, cp_rank);
                partial = contract_right_child_parallel(
                    &partial,
                    left_dim,
                    right_dim,
                    cp_rank,
                    kr_left.as_ref(),
                );
                node_idx = right;
            }
        }

        Ok(partial)
    }

    /// Parallel depth-first descent (see [`DimTree::descend`]).
    fn descend_parallel<T>(
        &self,
        node_idx: usize,
        partial: Array2<T>,
        owned: &[Array2<T>],
        cp_rank: usize,
        out: &mut [Option<Array2<T>>],
    ) where
        T: Copy + Num + One + Zero + Send + Sync + 'static,
    {
        let node = &self.nodes[node_idx];
        let Some(split) = node.split else {
            out[node.lo] = Some(partial);
            return;
        };
        let (left, right) = node
            .children
            .expect("internal node always has children by construction");
        let (lo, hi) = (node.lo, node.hi);
        let left_dim = self.range_prod(lo, split);
        let right_dim = self.range_prod(split, hi);

        let kr_left = khatri_rao_range(owned, lo, split, cp_rank);
        let kr_right = khatri_rao_range(owned, split, hi, cp_rank);

        let child_left =
            contract_left_child_parallel(&partial, left_dim, right_dim, cp_rank, kr_right.as_ref());
        let child_right =
            contract_right_child_parallel(&partial, left_dim, right_dim, cp_rank, kr_left.as_ref());
        drop(partial);
        drop(kr_left);
        drop(kr_right);

        self.descend_parallel(left, child_left, owned, cp_rank, out);
        self.descend_parallel(right, child_right, owned, cp_rank, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mttkrp::mttkrp;
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{SeedableRng, StdRng};

    // ── helpers ────────────────────────────────────────────────────────────

    fn rand_tensor(shape: &[usize], seed: u64) -> Array<f64, IxDyn> {
        let n: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Array::from_shape_vec(IxDyn(shape), data).expect("test shape valid")
    }

    fn rand_factors(shape: &[usize], cp_rank: usize, seed: u64) -> Vec<Array2<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        shape
            .iter()
            .map(|&dim| {
                let data: Vec<f64> = (0..dim * cp_rank)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect();
                Array2::from_shape_vec((dim, cp_rank), data).expect("test shape valid")
            })
            .collect()
    }

    fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>, label: &str) {
        assert_eq!(actual.shape(), expected.shape(), "shape mismatch ({label})");
        for i in 0..expected.shape()[0] {
            for j in 0..expected.shape()[1] {
                let a = actual[[i, j]];
                let b = expected[[i, j]];
                let tol = 1e-9 * (1.0 + b.abs());
                assert!(
                    (a - b).abs() <= tol,
                    "{label}: mismatch at [{i},{j}]: {a} vs {b} (diff {:.3e})",
                    (a - b).abs()
                );
            }
        }
    }

    /// Brute-force MTTKRP straight from the mathematical definition:
    /// `M[i_k, r] = Σ_{i≠k} X[i] · Π_{j≠k} A_j[i_j, r]`.
    /// Independent of both the reference kernel and the dimension tree, so a shared
    /// bug cannot hide behind it.
    fn brute_force_mttkrp(
        tensor: &Array<f64, IxDyn>,
        factors: &[Array2<f64>],
        mode: usize,
    ) -> Array2<f64> {
        let shape = tensor.shape().to_vec();
        let cp_rank = factors[0].shape()[1];
        let mut out = Array2::<f64>::zeros((shape[mode], cp_rank));
        let numel: usize = shape.iter().product();

        for flat in 0..numel {
            // Row-major (C order) unflattening.
            let mut idx = vec![0usize; shape.len()];
            let mut rest = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rest % shape[axis];
                rest /= shape[axis];
            }
            let value = tensor[IxDyn(&idx)];
            for r in 0..cp_rank {
                let mut prod = value;
                for (axis, &i) in idx.iter().enumerate() {
                    if axis != mode {
                        prod *= factors[axis][[i, r]];
                    }
                }
                out[[idx[mode], r]] += prod;
            }
        }
        out
    }

    fn check_all_modes(shape: &[usize], cp_rank: usize, seed: u64) {
        let tensor = rand_tensor(shape, seed);
        let factors = rand_factors(shape, cp_rank, seed.wrapping_add(1));
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let all = mttkrp_all_modes(&tensor.view(), &views).expect("dimtree mttkrp");
        assert_eq!(all.len(), shape.len());

        let tree = DimTree::new(shape).expect("tree");
        for (mode, computed) in all.iter().enumerate() {
            let reference = mttkrp(&tensor.view(), &views, mode).expect("reference mttkrp");
            let label = format!("shape={shape:?} rank={cp_rank} mode={mode}");
            assert_eq!(computed.shape(), &[shape[mode], cp_rank]);
            assert_close(computed, &reference, &format!("all_modes {label}"));

            // Single-mode path must agree too.
            let single = tree
                .mttkrp_mode(&tensor.view(), &views, mode)
                .expect("dimtree single mode");
            assert_close(&single, &reference, &format!("mttkrp_mode {label}"));

            #[cfg(feature = "parallel")]
            {
                let par = tree
                    .mttkrp_mode_parallel(&tensor.view(), &views, mode)
                    .expect("dimtree single mode parallel");
                assert_close(&par, &reference, &format!("mttkrp_mode_parallel {label}"));
            }
        }

        #[cfg(feature = "parallel")]
        {
            let all_par =
                mttkrp_all_modes_parallel(&tensor.view(), &views).expect("dimtree parallel");
            for (mode, (par, serial)) in all_par.iter().zip(all.iter()).enumerate() {
                assert_close(
                    par,
                    serial,
                    &format!("parallel vs serial shape={shape:?} mode={mode}"),
                );
            }
        }
    }

    // ── correctness vs. the mathematical definition ────────────────────────

    #[test]
    fn test_matches_brute_force_definition_3d() {
        let shape = [3usize, 2, 4];
        let tensor = rand_tensor(&shape, 7);
        let factors = rand_factors(&shape, 3, 8);
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let all = mttkrp_all_modes(&tensor.view(), &views).expect("dimtree");
        for (mode, computed) in all.iter().enumerate() {
            let brute = brute_force_mttkrp(&tensor, &factors, mode);
            assert_close(computed, &brute, &format!("brute-force mode {mode}"));
        }
    }

    #[test]
    fn test_matches_brute_force_definition_4d_noncubic() {
        let shape = [4usize, 2, 3, 2];
        let tensor = rand_tensor(&shape, 11);
        let factors = rand_factors(&shape, 2, 12);
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let all = mttkrp_all_modes(&tensor.view(), &views).expect("dimtree");
        for (mode, computed) in all.iter().enumerate() {
            let brute = brute_force_mttkrp(&tensor, &factors, mode);
            assert_close(computed, &brute, &format!("brute-force 4d mode {mode}"));
        }
    }

    // ── correctness vs. the existing reference kernel ──────────────────────

    #[test]
    fn test_matches_reference_3d_cubic() {
        for &cp_rank in &[1usize, 2, 5] {
            check_all_modes(&[4, 4, 4], cp_rank, 100 + cp_rank as u64);
        }
    }

    #[test]
    fn test_matches_reference_3d_noncubic() {
        for &cp_rank in &[1usize, 3, 8] {
            check_all_modes(&[7, 5, 3], cp_rank, 200 + cp_rank as u64);
        }
    }

    #[test]
    fn test_matches_reference_4d_noncubic() {
        // Asymmetric shape: catches row-major index-mapping bugs a cube would hide.
        for &cp_rank in &[1usize, 2, 6] {
            check_all_modes(&[7, 5, 3, 4], cp_rank, 300 + cp_rank as u64);
        }
    }

    #[test]
    fn test_matches_reference_5d_stress() {
        check_all_modes(&[3, 4, 2, 5, 3], 4, 404);
    }

    #[test]
    fn test_rank_larger_than_every_mode_dimension() {
        // R = 9 > max(I_k) = 4: exercises the R > I_k regime of the leaf kernels.
        check_all_modes(&[2, 3, 4], 9, 505);
    }

    #[test]
    fn test_rank_one() {
        check_all_modes(&[5, 3, 6, 2], 1, 606);
    }

    #[test]
    fn test_matches_reference_medium_scale() {
        // Large enough that the root GEMMs actually go through `matrixmultiply`'s
        // blocked micro-kernels and the parallel variants cross
        // `PARALLEL_WORK_THRESHOLD` — the small tests above stay in the scalar
        // fallbacks and would not catch a blocking/threshold bug.
        check_all_modes(&[20, 18, 22], 12, 777);
        check_all_modes(&[12, 10, 8, 6], 5, 778);
        check_all_modes(&[9, 7, 6, 5, 4], 8, 779);
    }

    #[test]
    fn test_order_two_degenerates_to_matmul() {
        // N = 2: mode-0 MTTKRP is X·A_1 and mode-1 is Xᵀ·A_0 — the tree's two root
        // GEMMs are the whole computation.
        let shape = [6usize, 4];
        let tensor = rand_tensor(&shape, 707);
        let factors = rand_factors(&shape, 3, 708);
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let all = mttkrp_all_modes(&tensor.view(), &views).expect("dimtree");
        let x2 = tensor
            .view()
            .into_shape_with_order((6, 4))
            .expect("2d reshape");
        assert_close(&all[0], &x2.dot(&factors[1]), "N=2 mode 0");
        assert_close(&all[1], &x2.t().dot(&factors[0]), "N=2 mode 1");
    }

    #[test]
    fn test_singleton_dimensions() {
        // Dimensions of size 1 make several sub-products degenerate to 1.
        check_all_modes(&[1, 5, 1, 4], 3, 808);
    }

    #[test]
    fn test_non_contiguous_input_view() {
        // A permuted (non C-contiguous) view must give the same answer as the
        // reference on the same view — exercises the `as_standard_layout` path.
        let base = rand_tensor(&[4, 5, 3], 909);
        let permuted = base.view().permuted_axes(IxDyn(&[2, 0, 1])); // shape 3×4×5
        let shape: Vec<usize> = permuted.shape().to_vec();
        let factors = rand_factors(&shape, 3, 910);
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let all = mttkrp_all_modes(&permuted, &views).expect("dimtree on permuted view");
        for (mode, computed) in all.iter().enumerate() {
            let reference = mttkrp(&permuted, &views, mode).expect("reference");
            assert_close(computed, &reference, &format!("permuted view mode {mode}"));
        }
    }

    #[test]
    fn test_transposed_factor_views() {
        // Factor views that are not standard layout (transposed) must work too.
        let shape = [4usize, 3, 5];
        let tensor = rand_tensor(&shape, 1001);
        let transposed: Vec<Array2<f64>> = shape
            .iter()
            .map(|&dim| {
                Array2::<f64>::from_shape_fn((2, dim), |(r, i)| (i as f64) - 0.5 * r as f64)
            })
            .collect();
        let views: Vec<ArrayView2<f64>> = transposed.iter().map(|f| f.t()).collect();

        let all = mttkrp_all_modes(&tensor.view(), &views).expect("dimtree");
        for (mode, computed) in all.iter().enumerate() {
            let reference = mttkrp(&tensor.view(), &views, mode).expect("reference");
            assert_close(computed, &reference, &format!("transposed factors {mode}"));
        }
    }

    #[test]
    fn test_randomized_sweep_against_reference() {
        // Many random shapes/ranks, all modes, seeded for reproducibility.
        let shapes: [&[usize]; 6] = [
            &[2, 2, 2],
            &[9, 2, 5],
            &[3, 7, 2, 4],
            &[2, 3, 2, 3, 2],
            &[6, 1, 7],
            &[5, 5, 2, 2],
        ];
        for (case, shape) in shapes.iter().enumerate() {
            for &cp_rank in &[1usize, 4, 7] {
                check_all_modes(shape, cp_rank, 5000 + (case as u64) * 13 + cp_rank as u64);
            }
        }
    }

    #[test]
    fn test_f32_matches_reference() {
        let shape = [5usize, 4, 3];
        let numel: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(2024);
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let tensor = Array::from_shape_vec(IxDyn(&shape), data).expect("shape");
        let factors: Vec<Array2<f32>> = shape
            .iter()
            .map(|&dim| {
                let d: Vec<f32> = (0..dim * 3).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                Array2::from_shape_vec((dim, 3), d).expect("shape")
            })
            .collect();
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let all = mttkrp_all_modes(&tensor.view(), &views).expect("dimtree f32");
        for (mode, computed) in all.iter().enumerate() {
            let reference = mttkrp(&tensor.view(), &views, mode).expect("reference f32");
            for (a, b) in computed.iter().zip(reference.iter()) {
                assert!(
                    (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                    "f32 mode {mode}: {a} vs {b}"
                );
            }
        }
    }

    // ── tree structure ─────────────────────────────────────────────────────

    #[test]
    fn test_split_balances_dimension_products() {
        // 256 × 4 × 4 × 256: the balanced-product split must be after mode 1
        // (256·4 = 1024 vs 4·256 = 1024), not the naive midpoint by mode count
        // (which happens to coincide here) — check a case where they differ:
        // 100 × 2 × 2 × 2 × 100 → products by mid: 100 | 800 (mid=1),
        // 200 | 400 (mid=2), 400 | 200 (mid=3), 800 | 100 (mid=4).
        // Balanced max is 400, attained first at mid=2.
        let tree = DimTree::new(&[100, 2, 2, 2, 100]).expect("tree");
        assert_eq!(tree.nodes[0].split, Some(2));
        assert_eq!(tree.ndim(), 5);
        assert_eq!(tree.numel(), 100 * 2 * 2 * 2 * 100);
    }

    #[test]
    fn test_tree_has_expected_node_count() {
        // A binary tree with N leaves has exactly 2N-1 nodes.
        for ndim in 2..8usize {
            let shape: Vec<usize> = (0..ndim).map(|i| i + 2).collect();
            let tree = DimTree::new(&shape).expect("tree");
            assert_eq!(tree.nodes.len(), 2 * ndim - 1, "ndim={ndim}");
            let leaves = tree.nodes.iter().filter(|n| n.split.is_none()).count();
            assert_eq!(leaves, ndim);
            // Every leaf covers exactly one mode, and every mode exactly once.
            let mut covered = vec![0usize; ndim];
            for node in tree.nodes.iter().filter(|n| n.split.is_none()) {
                assert_eq!(node.hi - node.lo, 1);
                covered[node.lo] += 1;
            }
            assert!(covered.iter().all(|&c| c == 1));
        }
    }

    #[test]
    fn test_dimtree_reuse_across_iterations() {
        // The tree is factor-independent: reuse it across an ALS-like sweep.
        let shape = [5usize, 4, 6];
        let tensor = rand_tensor(&shape, 1234);
        let tree = DimTree::new(&shape).expect("tree");

        for iteration in 0..3u64 {
            let factors = rand_factors(&shape, 3, 2000 + iteration);
            let views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let all = tree.mttkrp_all(&tensor.view(), &views).expect("all");
            for (mode, computed) in all.iter().enumerate() {
                let reference = mttkrp(&tensor.view(), &views, mode).expect("reference");
                assert_close(
                    computed,
                    &reference,
                    &format!("iter {iteration} mode {mode}"),
                );
            }
        }
    }

    // ── edge cases and errors ──────────────────────────────────────────────

    #[test]
    fn test_empty_tensor_gives_zero_results() {
        let shape = [3usize, 0, 4];
        let tensor = Array::<f64, _>::zeros(IxDyn(&shape));
        let factors = rand_factors(&shape, 2, 1);
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let all = mttkrp_all_modes(&tensor.view(), &views).expect("empty tensor");
        assert_eq!(all[0].shape(), &[3, 2]);
        assert_eq!(all[1].shape(), &[0, 2]);
        assert_eq!(all[2].shape(), &[4, 2]);
        assert!(all[0].iter().all(|&v| v == 0.0));
        assert!(all[2].iter().all(|&v| v == 0.0));

        let tree = DimTree::new(&shape).expect("tree");
        let single = tree
            .mttkrp_mode(&tensor.view(), &views, 0)
            .expect("empty single");
        assert!(single.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_order_one_rejected() {
        let err = DimTree::new(&[5]).expect_err("order-1 must be rejected");
        assert!(err.to_string().contains("order >= 2"));
    }

    #[test]
    fn test_mode_out_of_bounds() {
        let shape = [3usize, 4, 2];
        let tensor = rand_tensor(&shape, 1);
        let factors = rand_factors(&shape, 2, 2);
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let tree = DimTree::new(&shape).expect("tree");
        let err = tree
            .mttkrp_mode(&tensor.view(), &views, 3)
            .expect_err("mode 3 is out of bounds");
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn test_factor_count_mismatch() {
        let shape = [3usize, 4, 2];
        let tensor = rand_tensor(&shape, 1);
        let factors = rand_factors(&shape, 2, 2);
        let views: Vec<_> = factors.iter().take(2).map(|f| f.view()).collect();
        let err = mttkrp_all_modes(&tensor.view(), &views).expect_err("factor count mismatch");
        assert!(err.to_string().contains("must match tensor rank"));
    }

    #[test]
    fn test_inconsistent_cp_rank() {
        let shape = [3usize, 4, 2];
        let tensor = rand_tensor(&shape, 1);
        let a0 = Array2::<f64>::zeros((3, 2));
        let a1 = Array2::<f64>::zeros((4, 3)); // wrong rank
        let a2 = Array2::<f64>::zeros((2, 2));
        let views = [a0.view(), a1.view(), a2.view()];
        let err = mttkrp_all_modes(&tensor.view(), &views).expect_err("rank mismatch");
        assert!(err.to_string().contains("columns"));
    }

    #[test]
    fn test_factor_row_mismatch() {
        let shape = [3usize, 4, 2];
        let tensor = rand_tensor(&shape, 1);
        let a0 = Array2::<f64>::zeros((3, 2));
        let a1 = Array2::<f64>::zeros((5, 2)); // wrong row count
        let a2 = Array2::<f64>::zeros((2, 2));
        let views = [a0.view(), a1.view(), a2.view()];
        let err = mttkrp_all_modes(&tensor.view(), &views).expect_err("row mismatch");
        assert!(err.to_string().contains("rows"));
    }

    #[test]
    fn test_shape_mismatch_against_tree() {
        let tree = DimTree::new(&[3, 4, 2]).expect("tree");
        let tensor = rand_tensor(&[3, 4, 3], 1);
        let factors = rand_factors(&[3, 4, 3], 2, 2);
        let views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let err = tree
            .mttkrp_all(&tensor.view(), &views)
            .expect_err("shape mismatch");
        assert!(err.to_string().contains("does not match"));
    }
}
