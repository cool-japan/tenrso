//! CSF-based sparse MTTKRP kernel
//!
//! Implements MTTKRP (Matricized Tensor Times Khatri-Rao Product) for the CSF
//! (Compressed Sparse Fiber) sparse tensor format, using a depth-first fiber-tree
//! walk that amortizes partial Khatri-Rao products across shared fiber prefixes.
//!
//! # CSF Invariants (verified from tenrso-sparse csf.rs)
//!
//! For an N-mode tensor:
//! - `fptr[l]` has `|fids[l]| + 1` entries
//! - `fptr[l][f]..fptr[l][f+1]` = range of **leaf values** reachable from fiber `f` at level `l`
//!   (all fptr arrays are value-range pointers, NOT children-range pointers)
//! - `fids[l]` stores the mode index for each fiber at level `l`
//! - `vals` has `nnz` entries; `vals[v]` = nonzero value number `v`
//! - `fids[ndim-1]` is co-indexed with `vals`: `fids[ndim-1][v]` = mode index for value `v`
//!
//! # Navigation (key insight)
//!
//! To find children of fiber `f` at level `l` (where `l < ndim-1`):
//! - Compute val-range: `vs = fptr[l][f]`, `ve = fptr[l][f+1]`
//! - If `l+1 < ndim-1`: children fibers at level `l+1` are `g_start..g_end` where
//!   `g_start = fptr[l+1].partition_point(|&p| p < vs)` and
//!   `g_end   = fptr[l+1].partition_point(|&p| p < ve)`
//! - If `l+1 == ndim-1`: the leaf VALUE indices are directly `vs..ve`
//!   (and `fids[ndim-1][v]` + `vals[v]` for `v in vs..ve`)
//!
//! # Complexity
//!
//! Time: `O(nnz·R + Σ_L |fibers_L|·R)` vs `O(nnz·(N-1)·R)` for the COO variant.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::numeric::Float;
use tenrso_sparse::csf::CsfTensor;

use crate::mttkrp_sparse::validate_sparse_mttkrp_inputs;

// ─── Private helpers ────────────────────────────────────────────────────────

/// Process the leaf level of the CSF fiber tree for MTTKRP.
///
/// The leaf level stores values directly: `fids[ndim-1][v]` gives the mode index
/// and `vals[v]` gives the value. `vals_range` is a range into both arrays.
///
/// # Arguments
///
/// * `tensor`       - The CSF tensor
/// * `factors`      - Factor matrices in natural mode order
/// * `mode`         - Target mode
/// * `vals_range`   - Range of value indices to process at the leaf level
/// * `partial`      - Accumulated partial KR product from ancestor levels
/// * `out_row`      - Output row determined by target-mode ancestor (if any)
/// * `cp_rank`      - CP rank `R`
/// * `result`       - Output matrix to accumulate into
#[allow(clippy::too_many_arguments)]
#[inline]
fn csf_process_leaf<T: Float>(
    tensor: &CsfTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
    vals_range: std::ops::Range<usize>,
    partial: &[T],
    out_row: Option<usize>,
    cp_rank: usize,
    result: &mut Array2<T>,
) {
    let ndim = tensor.ndim();
    let leaf_level = ndim - 1;
    let leaf_mode = tensor.mode_order()[leaf_level];

    for v in vals_range {
        let leaf_fid = tensor.fids(leaf_level)[v];
        let val = tensor.vals()[v];

        let effective_out_row = if leaf_mode == mode {
            // The leaf mode IS the target mode: this value's output row is leaf_fid
            leaf_fid
        } else {
            // The leaf mode is NOT the target mode: out_row must have been set by an ancestor
            match out_row {
                Some(row) => row,
                None => {
                    // Safety: if out_row is None and we're at the leaf, the target mode
                    // wasn't found in any ancestor. This is a structural error; skip.
                    continue;
                }
            }
        };

        if leaf_mode == mode {
            // partial already has the full non-leaf product; just accumulate
            for r in 0..cp_rank {
                result[[effective_out_row, r]] = result[[effective_out_row, r]] + val * partial[r];
            }
        } else {
            // Multiply leaf factor row into partial (on-the-fly) and accumulate
            let leaf_row = factors[leaf_mode].row(leaf_fid);
            for r in 0..cp_rank {
                result[[effective_out_row, r]] =
                    result[[effective_out_row, r]] + val * partial[r] * leaf_row[r];
            }
        }
    }
}

/// Depth-first tree-walk for CSF MTTKRP — intermediate levels.
///
/// Traverses fibers at `level` (a non-leaf, non-root internal level or any level
/// in the range `[1, ndim-2]`). Each fiber's value range is used to:
/// 1. Compute the updated partial KR product (or record the output row if this
///    level's mode is the target mode).
/// 2. Navigate to children at the next level using binary search on `fptr[level+1]`.
///
/// # Key invariant
///
/// `fiber_range` contains valid indices into `fids[level]`. For any `f` in
/// `fiber_range`, `fptr[level][f]..fptr[level][f+1]` is a valid range into `vals`.
///
/// # Arguments
///
/// * `tensor`      - The CSF tensor
/// * `factors`     - Factor matrices in natural mode order
/// * `mode`        - Target mode
/// * `level`       - Current level (1 <= level <= ndim-2); at ndim-2 calls csf_process_leaf
/// * `fiber_range` - Range of fiber indices at `level`
/// * `partial`     - Accumulated partial KR product from levels above
/// * `out_row`     - Output row (set once the target-mode fiber has been seen)
/// * `cp_rank`     - CP rank R
/// * `result`      - Output matrix to accumulate into
#[allow(clippy::too_many_arguments)]
fn csf_walk_internal<T: Float>(
    tensor: &CsfTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
    level: usize,
    fiber_range: std::ops::Range<usize>,
    partial: &[T],
    out_row: Option<usize>,
    cp_rank: usize,
    result: &mut Array2<T>,
) {
    let ndim = tensor.ndim();
    let next_level = level + 1;
    let is_next_leaf = next_level == ndim - 1;
    let mode_at_level = tensor.mode_order()[level];

    for f in fiber_range {
        let fid = tensor.fids(level)[f];

        // Compute updated partial product / output row for this fiber
        let (new_partial, new_out_row) = update_partial(
            factors, mode, mode_at_level, fid, partial, out_row, cp_rank,
        );

        // Determine the value range for this fiber's subtree
        let vs = tensor.fptr(level)[f];
        let ve = tensor.fptr(level)[f + 1];

        if vs == ve {
            // Empty subtree (shouldn't happen in a valid CSF, but be safe)
            continue;
        }

        if is_next_leaf {
            // Next level is leaf: value range is vs..ve directly
            csf_process_leaf(
                tensor,
                factors,
                mode,
                vs..ve,
                &new_partial,
                new_out_row,
                cp_rank,
                result,
            );
        } else {
            // Find children fibers at next_level by binary searching fptr[next_level]
            let fptr_next = tensor.fptr(next_level);
            let child_start = fptr_next.partition_point(|&p| p < vs);
            let child_end = fptr_next.partition_point(|&p| p < ve);
            // child_start..child_end are fiber indices at next_level
            csf_walk_internal(
                tensor,
                factors,
                mode,
                next_level,
                child_start..child_end,
                &new_partial,
                new_out_row,
                cp_rank,
                result,
            );
        }
    }
}

/// Compute the updated partial KR product and output row for a fiber.
///
/// If this level's mode IS the target mode: record `fid` as `out_row` and
/// do NOT multiply into `partial`.
/// Otherwise: multiply `factors[mode_at_level][fid, :]` into `partial`.
#[inline]
fn update_partial<T: Float>(
    factors: &[ArrayView2<T>],
    mode: usize,
    mode_at_level: usize,
    fid: usize,
    partial: &[T],
    out_row: Option<usize>,
    cp_rank: usize,
) -> (Vec<T>, Option<usize>) {
    if mode_at_level == mode {
        (partial.to_vec(), Some(fid))
    } else {
        let factor_row = factors[mode_at_level].row(fid);
        let mut p = partial.to_vec();
        for (pi, &fi) in p.iter_mut().zip(factor_row.iter()).take(cp_rank) {
            *pi = *pi * fi;
        }
        (p, out_row)
    }
}

/// Naive oracle for CSF MTTKRP: iterates all nonzeros via the CSF iterator
/// and applies the COO-style KR product computation.
///
/// This is bit-identical to `mttkrp_sparse_coo` applied to the same data,
/// and serves as a ground-truth reference for the optimized tree-walk.
#[allow(dead_code)]
fn mttkrp_csf_naive<T: Float>(
    tensor: &CsfTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>> {
    let cp_rank = validate_sparse_mttkrp_inputs(tensor.shape(), factors, mode)?;
    let mode_size = tensor.shape()[mode];
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    if tensor.nnz() == 0 {
        return Ok(result);
    }

    let ndim = tensor.ndim();
    let mut kr_buf = vec![T::zero(); cp_rank];

    for (idx, val) in tensor.iter() {
        let out_row = idx[mode];

        // Compute KR product: ∏_{k ≠ mode} factors[k][idx[k], r]
        kr_buf.iter_mut().for_each(|x| *x = T::one());
        for n in 0..ndim {
            if n == mode {
                continue;
            }
            let row = factors[n].row(idx[n]);
            for (b, &f) in kr_buf.iter_mut().zip(row.iter()).take(cp_rank) {
                *b = *b * f;
            }
        }

        // Accumulate
        for r in 0..cp_rank {
            result[[out_row, r]] = result[[out_row, r]] + val * kr_buf[r];
        }
    }

    Ok(result)
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Computes MTTKRP (Matricized Tensor × Khatri-Rao Product) for a CSF sparse tensor.
///
/// Uses a depth-first fiber-tree walk that amortizes partial Khatri-Rao products
/// across shared fiber prefixes, giving `O(nnz·R + Σ_L |fibers_L|·R)` complexity
/// vs `O(nnz·(N-1)·R)` for the COO variant (Smith & Karypis, 2015).
///
/// For target mode `m`, computes:
/// ```text
/// result[i_m, r] = Σ_{nnz: idx[m]=i_m} val × ∏_{k≠m} factors[k][idx[k], r]
/// ```
///
/// # Arguments
///
/// * `tensor` - CSF sparse tensor with any mode ordering
/// * `factors` - Slice of N factor matrices in natural mode order `[F_0, ..., F_{N-1}]`,
///   each shape `(I_n, R)`
/// * `mode` - Target mode; output has shape `(I_mode, R)`
///
/// # Complexity
///
/// Time: `O(nnz·R + Σ_L (number_of_fibers_at_level_L)·R)`
/// Space: `O(I_mode·R)` output + `O(ndim·R)` stack scratch per recursion level
///
/// # Errors
///
/// Returns an error if `mode` is out of bounds, `factors.len() != ndim`, or
/// factor shapes don't match tensor dimensions.
///
/// # Examples
///
/// ```rust,ignore
/// use tenrso_sparse::{CooTensor, CsfTensor};
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::mttkrp_sparse_csf::mttkrp_sparse_csf;
///
/// let mut coo = CooTensor::zeros(vec![3, 4, 5]).unwrap();
/// coo.push(vec![0, 1, 2], 1.0_f64).unwrap();
/// coo.push(vec![1, 2, 3], 2.0_f64).unwrap();
///
/// let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
/// let f0 = Array2::<f64>::ones((3, 2));
/// let f1 = Array2::<f64>::ones((4, 2));
/// let f2 = Array2::<f64>::ones((5, 2));
///
/// let result = mttkrp_sparse_csf(&csf, &[f0.view(), f1.view(), f2.view()], 1).unwrap();
/// assert_eq!(result.shape(), &[4, 2]);
/// ```
pub fn mttkrp_sparse_csf<T>(
    tensor: &CsfTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Float,
{
    let cp_rank = validate_sparse_mttkrp_inputs(tensor.shape(), factors, mode)?;
    let mode_size = tensor.shape()[mode];
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    if tensor.nnz() == 0 {
        return Ok(result);
    }

    let ndim = tensor.ndim();
    let initial_partial = vec![T::one(); cp_rank];

    // Handle 1D tensors separately (degenerate case: just sum values into output rows)
    if ndim == 1 {
        for f in 0..tensor.fids(0).len() {
            let fid = tensor.fids(0)[f];
            let vs = tensor.fptr(0)[f];
            let ve = tensor.fptr(0)[f + 1];
            for v in vs..ve {
                let val = tensor.vals()[v];
                for r in 0..cp_rank {
                    result[[fid, r]] = result[[fid, r]] + val;
                }
            }
        }
        return Ok(result);
    }

    let n_root_fibers = tensor.fids(0).len();
    let root_mode = tensor.mode_order()[0];

    // Process each root fiber
    for f in 0..n_root_fibers {
        let fid = tensor.fids(0)[f];
        let vs = tensor.fptr(0)[f];
        let ve = tensor.fptr(0)[f + 1];

        if vs == ve {
            continue;
        }

        let (partial, out_row) = update_partial(
            factors, mode, root_mode, fid, &initial_partial, None, cp_rank,
        );

        if ndim == 2 {
            // Special case: 2D tensor — root is level 0, leaf is level 1
            // val range for this root fiber is vs..ve directly
            csf_process_leaf(
                tensor,
                factors,
                mode,
                vs..ve,
                &partial,
                out_row,
                cp_rank,
                &mut result,
            );
        } else {
            // General case: find children at level 1
            let is_next_leaf = ndim == 2; // level 1 = ndim-1 only if ndim=2
            let _ = is_next_leaf; // already handled above

            // ndim >= 3: use csf_walk_internal starting at level 1
            let fptr_1 = tensor.fptr(1);
            let child_start = fptr_1.partition_point(|&p| p < vs);
            let child_end = fptr_1.partition_point(|&p| p < ve);

            if ndim == 2 {
                // Already handled, but keep for symmetry
                unreachable!();
            } else if ndim == 3 {
                // Level 1 is the last non-leaf before the leaf (level 2)
                // children at level 1 are in child_start..child_end
                // Then we go directly to leaf
                let next_is_leaf = true; // level 1+1 = level 2 = ndim-1 for ndim=3
                for g in child_start..child_end {
                    let gid = tensor.fids(1)[g];
                    let gvs = tensor.fptr(1)[g];
                    let gve = tensor.fptr(1)[g + 1];

                    let (partial2, out_row2) = update_partial(
                        factors, mode, tensor.mode_order()[1], gid, &partial, out_row, cp_rank,
                    );

                    if next_is_leaf {
                        csf_process_leaf(
                            tensor,
                            factors,
                            mode,
                            gvs..gve,
                            &partial2,
                            out_row2,
                            cp_rank,
                            &mut result,
                        );
                    }
                }
            } else {
                // ndim >= 4: use general recursive walker
                csf_walk_internal(
                    tensor,
                    factors,
                    mode,
                    1,
                    child_start..child_end,
                    &partial,
                    out_row,
                    cp_rank,
                    &mut result,
                );
            }
        }
    }

    Ok(result)
}

/// Parallel MTTKRP for CSF sparse tensors using a bucket-collect pattern.
///
/// Parallelizes over root fibers (level-0 fibers). Each root fiber's subtree is
/// independent, so each worker runs the tree-walk into a **local scratch** buffer.
/// The local buffers are then serially reduced (summed) into the final result.
///
/// This is race-free: each root fiber processes a disjoint subtree and writes
/// exclusively to its own local buffer.
///
/// # Arguments
///
/// * `tensor` - CSF sparse tensor with any mode ordering
/// * `factors` - Slice of N factor matrices in natural mode order, each shape `(I_n, R)`
/// * `mode` - Target mode; output has shape `(I_mode, R)`
///
/// # Complexity
///
/// Time: `O((nnz·R + Σ_L |fibers_L|·R) / P)` parallel + `O(P · I_mode · R)` reduce
/// Space: `O(P · I_mode · R)` total scratch across all workers
///
/// # Errors
///
/// Returns an error if `mode` is out of bounds, `factors.len() != ndim`, or
/// factor shapes don't match tensor dimensions.
///
/// # Examples
///
/// ```rust,ignore
/// use tenrso_sparse::{CooTensor, CsfTensor};
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::mttkrp_sparse_csf::mttkrp_sparse_csf_parallel;
///
/// let mut coo = CooTensor::zeros(vec![4, 5, 6]).unwrap();
/// coo.push(vec![0, 1, 2], 1.0_f64).unwrap();
/// let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
/// let f0 = Array2::<f64>::ones((4, 2));
/// let f1 = Array2::<f64>::ones((5, 2));
/// let f2 = Array2::<f64>::ones((6, 2));
///
/// let result = mttkrp_sparse_csf_parallel(&csf, &[f0.view(), f1.view(), f2.view()], 1).unwrap();
/// assert_eq!(result.shape(), &[5, 2]);
/// ```
#[cfg(feature = "parallel")]
pub fn mttkrp_sparse_csf_parallel<T>(
    tensor: &CsfTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Float + Send + Sync,
{
    // Import parallel ops only inside this function body to avoid unused-import warnings
    // when the parallel feature is inactive.
    use scirs2_core::parallel_ops::*;

    let cp_rank = validate_sparse_mttkrp_inputs(tensor.shape(), factors, mode)?;
    let mode_size = tensor.shape()[mode];

    if tensor.nnz() == 0 {
        return Ok(Array2::<T>::zeros((mode_size, cp_rank)));
    }

    let ndim = tensor.ndim();
    let initial_partial = vec![T::one(); cp_rank];
    let n_root_fibers = tensor.fids(0).len();
    let root_mode = tensor.mode_order()[0];

    // Each root fiber's subtree is independent; process in parallel
    let partials: Vec<Array2<T>> = (0..n_root_fibers)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|f| {
            let mut local_result = Array2::<T>::zeros((mode_size, cp_rank));

            let fid = tensor.fids(0)[f];
            let vs = tensor.fptr(0)[f];
            let ve = tensor.fptr(0)[f + 1];

            if vs == ve {
                return local_result;
            }

            let (partial, out_row) = update_partial(
                factors, mode, root_mode, fid, &initial_partial, None, cp_rank,
            );

            if ndim == 1 {
                // 1D: shouldn't reach here (already handled in the non-parallel path)
                for v in vs..ve {
                    let val = tensor.vals()[v];
                    for r in 0..cp_rank {
                        local_result[[fid, r]] = local_result[[fid, r]] + val;
                    }
                }
            } else if ndim == 2 {
                csf_process_leaf(
                    tensor,
                    factors,
                    mode,
                    vs..ve,
                    &partial,
                    out_row,
                    cp_rank,
                    &mut local_result,
                );
            } else if ndim == 3 {
                let fptr_1 = tensor.fptr(1);
                let child_start = fptr_1.partition_point(|&p| p < vs);
                let child_end = fptr_1.partition_point(|&p| p < ve);

                for g in child_start..child_end {
                    let gid = tensor.fids(1)[g];
                    let gvs = tensor.fptr(1)[g];
                    let gve = tensor.fptr(1)[g + 1];

                    let (partial2, out_row2) = update_partial(
                        factors, mode, tensor.mode_order()[1], gid, &partial, out_row, cp_rank,
                    );

                    csf_process_leaf(
                        tensor,
                        factors,
                        mode,
                        gvs..gve,
                        &partial2,
                        out_row2,
                        cp_rank,
                        &mut local_result,
                    );
                }
            } else {
                // ndim >= 4: use general recursive walker
                let fptr_1 = tensor.fptr(1);
                let child_start = fptr_1.partition_point(|&p| p < vs);
                let child_end = fptr_1.partition_point(|&p| p < ve);

                csf_walk_internal(
                    tensor,
                    factors,
                    mode,
                    1,
                    child_start..child_end,
                    &partial,
                    out_row,
                    cp_rank,
                    &mut local_result,
                );
            }

            local_result
        })
        .collect();

    // Serial reduce: sum all partial results into the final output
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));
    for partial in partials {
        result = result + partial;
    }

    Ok(result)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array2;
    use scirs2_core::random::{SeedableRng, StdRng};
    use tenrso_core::DenseND;
    use tenrso_sparse::coo::CooTensor;
    use tenrso_sparse::csf::CsfTensor;

    // ── Test helpers ────────────────────────────────────────────────────────

    /// Build a random sparse tensor in three representations simultaneously.
    fn build_sparse_test_tensor_3d(
        seed: u64,
        shape: &[usize],
        nnz_target: usize,
    ) -> (DenseND<f64>, CooTensor<f64>, CsfTensor<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let ndim = shape.len();
        let total: usize = shape.iter().product();

        // Build a flat data buffer and zero most entries
        let mut sparse_data = vec![0.0_f64; total];
        let mut all_flat: Vec<usize> = (0..total).collect();

        // Reservoir-sample nnz_target indices
        let actual_nnz = nnz_target.min(total);
        for i in 0..actual_nnz {
            let j = rng.gen_range(i..total);
            all_flat.swap(i, j);
        }
        for i in 0..actual_nnz {
            sparse_data[all_flat[i]] = rng.gen_range(-5.0_f64..5.0_f64);
        }

        // Build DenseND
        let dense =
            DenseND::from_vec(sparse_data.clone(), shape).expect("DenseND creation failed");

        // Compute strides for flat → multi-index conversion
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Build COO from the sparse data
        let mut coo =
            CooTensor::<f64>::zeros(shape.to_vec()).expect("CooTensor creation failed");
        for (flat_idx, &val) in sparse_data.iter().enumerate() {
            if val != 0.0 {
                let mut idx = vec![0usize; ndim];
                let mut remaining = flat_idx;
                for d in 0..ndim {
                    idx[d] = remaining / strides[d];
                    remaining %= strides[d];
                }
                coo.push(idx, val).expect("COO push failed");
            }
        }

        // Build CSF from COO with natural mode order
        let mode_order: Vec<usize> = (0..ndim).collect();
        let csf =
            CsfTensor::from_coo(&coo, &mode_order).expect("CsfTensor creation failed");

        (dense, coo, csf)
    }

    /// Generate random factor matrices.
    fn random_factors(shape: &[usize], cp_rank: usize, seed: u64) -> Vec<Array2<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        shape
            .iter()
            .map(|&dim| {
                let data: Vec<f64> = (0..dim * cp_rank)
                    .map(|_| rng.gen_range(-2.0_f64..2.0_f64))
                    .collect();
                Array2::from_shape_vec((dim, cp_rank), data)
                    .expect("factor matrix creation failed")
            })
            .collect()
    }

    /// Check that two Array2 results agree within the given tolerance.
    fn assert_arrays_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64, label: &str) {
        assert_eq!(a.shape(), b.shape(), "{}: shape mismatch", label);
        for i in 0..a.shape()[0] {
            for r in 0..a.shape()[1] {
                let diff = (a[[i, r]] - b[[i, r]]).abs();
                assert!(
                    diff < tol,
                    "{}: mismatch at [{}][{}]: left={:.15e} right={:.15e} diff={:.3e}",
                    label,
                    i,
                    r,
                    a[[i, r]],
                    b[[i, r]],
                    diff
                );
            }
        }
    }

    // ── Dense oracle for CSF MTTKRP ─────────────────────────────────────────

    /// Dense MTTKRP oracle using the existing `mttkrp` kernel.
    fn dense_mttkrp_oracle(
        dense: &DenseND<f64>,
        factors: &[Array2<f64>],
        mode: usize,
    ) -> Array2<f64> {
        use crate::mttkrp::mttkrp;
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        mttkrp(&dense.view(), &factor_views, mode).expect("dense mttkrp failed")
    }

    // ── Tests: CSF vs Dense ─────────────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_vs_dense_mode0() {
        let shape = [4, 5, 6];
        let (dense, _, csf) = build_sparse_test_tensor_3d(1001, &shape, 30);
        let factors = random_factors(&shape, 4, 2001);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result_csf = mttkrp_sparse_csf(&csf, &factor_views, 0).unwrap();
        let result_dense = dense_mttkrp_oracle(&dense, &factors, 0);
        assert_arrays_close(&result_csf, &result_dense, 1e-10, "mode0 vs dense");
    }

    #[test]
    fn test_csf_mttkrp_vs_dense_mode1() {
        let shape = [4, 5, 6];
        let (dense, _, csf) = build_sparse_test_tensor_3d(1002, &shape, 30);
        let factors = random_factors(&shape, 4, 2002);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result_csf = mttkrp_sparse_csf(&csf, &factor_views, 1).unwrap();
        let result_dense = dense_mttkrp_oracle(&dense, &factors, 1);
        assert_arrays_close(&result_csf, &result_dense, 1e-10, "mode1 vs dense");
    }

    #[test]
    fn test_csf_mttkrp_vs_dense_mode2() {
        let shape = [4, 5, 6];
        let (dense, _, csf) = build_sparse_test_tensor_3d(1003, &shape, 30);
        let factors = random_factors(&shape, 4, 2003);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result_csf = mttkrp_sparse_csf(&csf, &factor_views, 2).unwrap();
        let result_dense = dense_mttkrp_oracle(&dense, &factors, 2);
        assert_arrays_close(&result_csf, &result_dense, 1e-10, "mode2 vs dense");
    }

    // ── Test: CSF tree-walk vs COO ───────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_vs_coo() {
        use crate::mttkrp_sparse::mttkrp_sparse_coo;

        let shape = [4, 5, 6];
        let (_, coo, csf) = build_sparse_test_tensor_3d(1004, &shape, 40);
        let factors = random_factors(&shape, 4, 2004);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        for mode in 0..3 {
            let result_csf = mttkrp_sparse_csf(&csf, &factor_views, mode).unwrap();
            let result_coo = mttkrp_sparse_coo(&coo, &factor_views, mode).unwrap();
            assert_arrays_close(
                &result_csf,
                &result_coo,
                1e-10,
                &format!("csf vs coo mode={}", mode),
            );
        }
    }

    // ── Test: tree-walk vs naive oracle ─────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_vs_naive() {
        use crate::mttkrp_sparse::mttkrp_sparse_coo;

        let shape = [4, 5, 6];
        let (_, coo, csf) = build_sparse_test_tensor_3d(1005, &shape, 35);
        let factors = random_factors(&shape, 3, 2005);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        for mode in 0..3 {
            let result_walk = mttkrp_sparse_csf(&csf, &factor_views, mode).unwrap();
            let result_naive = mttkrp_csf_naive(&csf, &factor_views, mode).unwrap();
            assert_arrays_close(
                &result_walk,
                &result_naive,
                1e-12,
                &format!("tree-walk vs naive mode={}", mode),
            );

            // Also verify naive is consistent with COO (sanity check)
            let result_coo = mttkrp_sparse_coo(&coo, &factor_views, mode).unwrap();
            assert_arrays_close(
                &result_naive,
                &result_coo,
                1e-12,
                &format!("naive vs coo mode={}", mode),
            );
        }
    }

    // ── Tests: mode_order variants (same data, same target mode) ────────────

    #[test]
    fn test_csf_mttkrp_mode_order_012() {
        let shape = [4, 5, 6];
        let (_, coo, _) = build_sparse_test_tensor_3d(1006, &shape, 30);
        let factors = random_factors(&shape, 3, 2006);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let result = mttkrp_sparse_csf(&csf, &factor_views, 1).unwrap();
        assert_eq!(result.shape(), &[5, 3]);
    }

    #[test]
    fn test_csf_mttkrp_mode_order_210() {
        let shape = [4, 5, 6];
        let (_, coo, _) = build_sparse_test_tensor_3d(1006, &shape, 30);
        let factors = random_factors(&shape, 3, 2006);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let csf = CsfTensor::from_coo(&coo, &[2, 1, 0]).unwrap();
        let result = mttkrp_sparse_csf(&csf, &factor_views, 1).unwrap();
        assert_eq!(result.shape(), &[5, 3]);
    }

    #[test]
    fn test_csf_mttkrp_mode_order_102() {
        let shape = [4, 5, 6];
        let (_, coo, _) = build_sparse_test_tensor_3d(1006, &shape, 30);
        let factors = random_factors(&shape, 3, 2006);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        // Build all three CSF orderings and ensure they agree on mode=1
        let csf_012 = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let csf_210 = CsfTensor::from_coo(&coo, &[2, 1, 0]).unwrap();
        let csf_102 = CsfTensor::from_coo(&coo, &[1, 0, 2]).unwrap();

        let result_012 = mttkrp_sparse_csf(&csf_012, &factor_views, 1).unwrap();
        let result_210 = mttkrp_sparse_csf(&csf_210, &factor_views, 1).unwrap();
        let result_102 = mttkrp_sparse_csf(&csf_102, &factor_views, 1).unwrap();

        assert_arrays_close(&result_012, &result_210, 1e-10, "012 vs 210");
        assert_arrays_close(&result_012, &result_102, 1e-10, "012 vs 102");
    }

    // ── Test: 4D tensor ─────────────────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_4d_interior_mode() {
        let shape = [3, 4, 5, 6];
        let ndim = shape.len();
        let mut rng = StdRng::seed_from_u64(7777);

        // Build COO directly
        let mut coo = CooTensor::<f64>::zeros(shape.to_vec()).unwrap();
        for _ in 0..40 {
            let idx: Vec<usize> = shape.iter().map(|&s| rng.gen_range(0..s)).collect();
            let val: f64 = rng.gen_range(-3.0_f64..3.0_f64);
            let _ = coo.push(idx, val);
        }

        // Use mode_order [2, 0, 3, 1] (custom order)
        let csf = CsfTensor::from_coo(&coo, &[2, 0, 3, 1]).unwrap();
        let factors = random_factors(&shape, 4, 8888);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        // Target mode 2 (interior)
        let result_csf = mttkrp_sparse_csf(&csf, &factor_views, 2).unwrap();

        // Verify against naive
        let result_naive = mttkrp_csf_naive(&csf, &factor_views, 2).unwrap();
        assert_arrays_close(
            &result_csf,
            &result_naive,
            1e-10,
            "4d interior mode csf vs naive",
        );

        // Verify against COO
        let result_coo = {
            use crate::mttkrp_sparse::mttkrp_sparse_coo;
            mttkrp_sparse_coo(&coo, &factor_views, 2).unwrap()
        };
        assert_arrays_close(
            &result_csf,
            &result_coo,
            1e-10,
            "4d interior mode csf vs coo",
        );

        // Check output shape: mode 2 has size 5, rank 4
        assert_eq!(result_csf.shape(), &[5, 4]);
        let _ = ndim;
    }

    // ── Test: empty tensor ───────────────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_empty() {
        let coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        let f0 = Array2::<f64>::ones((4, 2));
        let f1 = Array2::<f64>::ones((5, 2));
        let f2 = Array2::<f64>::ones((6, 2));
        let factor_views = [f0.view(), f1.view(), f2.view()];

        let result = mttkrp_sparse_csf(&csf, &factor_views, 1).unwrap();
        assert_eq!(result.shape(), &[5, 2]);

        // All zeros
        for i in 0..5 {
            for r in 0..2 {
                assert_eq!(result[[i, r]], 0.0, "empty tensor: result should be zero");
            }
        }
    }

    // ── Test: single nonzero ─────────────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_single_nonzero() {
        // Single nonzero at (1, 2, 3) with value 5.0
        let mut coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
        coo.push(vec![1, 2, 3], 5.0_f64).unwrap();
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        // Factor matrices with known values
        let mut f0 = Array2::<f64>::zeros((4, 2));
        f0[[1, 0]] = 2.0;
        f0[[1, 1]] = 3.0;
        let mut f1 = Array2::<f64>::zeros((5, 2));
        f1[[2, 0]] = 4.0;
        f1[[2, 1]] = 5.0;
        let mut f2 = Array2::<f64>::zeros((6, 2));
        f2[[3, 0]] = 6.0;
        f2[[3, 1]] = 7.0;

        let factor_views = [f0.view(), f1.view(), f2.view()];

        // mode=0: result[1, r] = 5.0 * f1[2,r] * f2[3,r]
        let result_m0 = mttkrp_sparse_csf(&csf, &factor_views, 0).unwrap();
        assert!(
            (result_m0[[1, 0]] - 5.0 * 4.0 * 6.0).abs() < 1e-12,
            "mode=0, r=0: expected {}, got {}",
            5.0 * 4.0 * 6.0,
            result_m0[[1, 0]]
        );
        assert!(
            (result_m0[[1, 1]] - 5.0 * 5.0 * 7.0).abs() < 1e-12,
            "mode=0, r=1: expected {}, got {}",
            5.0 * 5.0 * 7.0,
            result_m0[[1, 1]]
        );

        // mode=1: result[2, r] = 5.0 * f0[1,r] * f2[3,r]
        let result_m1 = mttkrp_sparse_csf(&csf, &factor_views, 1).unwrap();
        assert!(
            (result_m1[[2, 0]] - 5.0 * 2.0 * 6.0).abs() < 1e-12,
            "mode=1, r=0: expected {}, got {}",
            5.0 * 2.0 * 6.0,
            result_m1[[2, 0]]
        );
        assert!(
            (result_m1[[2, 1]] - 5.0 * 3.0 * 7.0).abs() < 1e-12,
            "mode=1, r=1: expected {}, got {}",
            5.0 * 3.0 * 7.0,
            result_m1[[2, 1]]
        );

        // mode=2: result[3, r] = 5.0 * f0[1,r] * f1[2,r]
        let result_m2 = mttkrp_sparse_csf(&csf, &factor_views, 2).unwrap();
        assert!(
            (result_m2[[3, 0]] - 5.0 * 2.0 * 4.0).abs() < 1e-12,
            "mode=2, r=0: expected {}, got {}",
            5.0 * 2.0 * 4.0,
            result_m2[[3, 0]]
        );
        assert!(
            (result_m2[[3, 1]] - 5.0 * 3.0 * 5.0).abs() < 1e-12,
            "mode=2, r=1: expected {}, got {}",
            5.0 * 3.0 * 5.0,
            result_m2[[3, 1]]
        );
    }

    // ── Test: rank-1 ─────────────────────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_rank1() {
        let shape = [3, 4, 5];
        let (_, coo, csf) = build_sparse_test_tensor_3d(1010, &shape, 20);
        let factors = random_factors(&shape, 1, 2010);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        for (mode, &mode_size) in shape.iter().enumerate() {
            let result = mttkrp_sparse_csf(&csf, &factor_views, mode).unwrap();
            assert_eq!(result.shape()[0], mode_size);
            assert_eq!(result.shape()[1], 1);

            // Verify against COO
            let result_coo = {
                use crate::mttkrp_sparse::mttkrp_sparse_coo;
                mttkrp_sparse_coo(&coo, &factor_views, mode).unwrap()
            };
            assert_arrays_close(&result, &result_coo, 1e-12, &format!("rank1 mode={}", mode));
        }
    }

    // ── Test: rank-128 ───────────────────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_rank128() {
        let shape = [4, 5, 6];
        let (_, coo, csf) = build_sparse_test_tensor_3d(1013, &shape, 40);
        let factors = random_factors(&shape, 128, 2013);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        for mode in 0..3 {
            let result_csf = mttkrp_sparse_csf(&csf, &factor_views, mode).unwrap();
            let result_coo = {
                use crate::mttkrp_sparse::mttkrp_sparse_coo;
                mttkrp_sparse_coo(&coo, &factor_views, mode).unwrap()
            };
            // Looser tolerance for R=128 due to FMA accumulation
            assert_arrays_close(
                &result_csf,
                &result_coo,
                1e-8,
                &format!("rank128 mode={}", mode),
            );
        }
    }

    // ── Test: error cases ────────────────────────────────────────────────────

    #[test]
    fn test_csf_mttkrp_error_invalid_mode() {
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).unwrap();
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        let f0 = Array2::<f64>::ones((3, 2));
        let f1 = Array2::<f64>::ones((4, 2));
        let f2 = Array2::<f64>::ones((5, 2));
        let factor_views = [f0.view(), f1.view(), f2.view()];

        let result = mttkrp_sparse_csf(&csf, &factor_views, 10);
        assert!(result.is_err(), "should error on invalid mode");
        let msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            msg.contains("mode"),
            "error message should mention 'mode': {}",
            msg
        );
    }

    #[test]
    fn test_csf_mttkrp_error_wrong_num_factors() {
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).unwrap();
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        let f0 = Array2::<f64>::ones((3, 2));
        let f1 = Array2::<f64>::ones((4, 2));
        // Only 2 factors for a 3D tensor
        let factor_views = [f0.view(), f1.view()];

        let result = mttkrp_sparse_csf(&csf, &factor_views, 0);
        assert!(result.is_err(), "should error on wrong number of factors");
    }

    // ── Parallel tests ───────────────────────────────────────────────────────

    #[cfg(feature = "parallel")]
    #[test]
    fn test_csf_mttkrp_parallel_parity_root_mode() {
        // Target mode = root mode (mode_order[0])
        let shape = [4, 5, 6];
        let (_, coo, _) = build_sparse_test_tensor_3d(2000, &shape, 40);
        let factors = random_factors(&shape, 4, 3000);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        // Use [0,1,2] ordering so root mode = 0, target mode = 0
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let result_serial = mttkrp_sparse_csf(&csf, &factor_views, 0).unwrap();
        let result_parallel = mttkrp_sparse_csf_parallel(&csf, &factor_views, 0).unwrap();

        assert_arrays_close(
            &result_serial,
            &result_parallel,
            1e-12,
            "parallel parity root mode",
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_csf_mttkrp_parallel_parity_nonroot_mode() {
        // Target mode != root mode (mode_order[0])
        let shape = [4, 5, 6];
        let (_, coo, _) = build_sparse_test_tensor_3d(2001, &shape, 40);
        let factors = random_factors(&shape, 4, 3001);
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        // Use [0,1,2] ordering so root mode = 0, target mode = 1 (non-root)
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let result_serial = mttkrp_sparse_csf(&csf, &factor_views, 1).unwrap();
        let result_parallel = mttkrp_sparse_csf_parallel(&csf, &factor_views, 1).unwrap();

        assert_arrays_close(
            &result_serial,
            &result_parallel,
            1e-12,
            "parallel parity nonroot mode",
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_csf_mttkrp_parallel_empty() {
        let coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        let f0 = Array2::<f64>::ones((4, 2));
        let f1 = Array2::<f64>::ones((5, 2));
        let f2 = Array2::<f64>::ones((6, 2));
        let factor_views = [f0.view(), f1.view(), f2.view()];

        let result = mttkrp_sparse_csf_parallel(&csf, &factor_views, 1).unwrap();
        assert_eq!(result.shape(), &[5, 2]);

        for i in 0..5 {
            for r in 0..2 {
                assert_eq!(
                    result[[i, r]],
                    0.0,
                    "empty tensor parallel: result should be zero"
                );
            }
        }
    }
}
