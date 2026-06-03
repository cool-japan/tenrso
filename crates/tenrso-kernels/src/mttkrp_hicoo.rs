//! HiCOO-based sparse MTTKRP kernel
//!
//! Implements MTTKRP (Matricized Tensor Times Khatri-Rao Product) for the HiCOO
//! (Hierarchical COO) sparse tensor format. HiCOO groups nonzeros into rectangular
//! blocks, improving cache locality by keeping block-local coordinate data together.
//!
//! # Algorithm Overview
//!
//! The serial algorithm iterates over each block `b`, then over every nonzero `i`
//! in that block. The global row in the output matrix is:
//!
//! ```text
//! global_row = block_coords[b][mode] * block_shape[mode] + local_coords[i][mode]
//! ```
//!
//! The contribution to `result[global_row, r]` for every CP rank `r` is:
//!
//! ```text
//! val[i] × ∏_{k ≠ mode} factors[k][block_coords[b][k]*block_shape[k]+local_coords[i][k], r]
//! ```
//!
//! # Parallel Algorithm
//!
//! Blocks are grouped by `block_coords[b][mode]` — the *output block-row key*.
//! All blocks with the same key write exclusively into the row range
//! `[key*block_shape[mode], min((key+1)*block_shape[mode], shape[mode]))`.
//! Because key groups are disjoint in the output, they can be processed in parallel
//! without write races. A final scatter (serial, O(nnz) total) assembles the
//! per-group local buffers into the result.
//!
//! # Complexity
//!
//! - Serial:   `O(nnz · (N-1) · R)` — one KR-row per nonzero
//! - Parallel: `O(nnz · (N-1) · R / P)` compute + `O(nnz)` scatter
//! - Space:    `O(shape[mode] · R)` output + `O(R)` scratch per nonzero
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` or `rand` is forbidden per SCIRS2_INTEGRATION_POLICY.md.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::numeric::Float;
use tenrso_sparse::hicoo::HiCooTensor;

use crate::mttkrp_sparse::validate_sparse_mttkrp_inputs;

// ─── Private helpers ────────────────────────────────────────────────────────

/// Compute the Khatri-Rao row for a single HiCOO nonzero.
///
/// Returns a length-R vector `∏_{k ≠ mode} factors[k][bc[k]*bs[k]+lc[k], r]`.
///
/// # Arguments
///
/// * `block_coord` - Block coordinates of block `b`
/// * `local_coord` - Within-block coordinates of nonzero `i`
/// * `factors`     - Factor matrices in natural mode order
/// * `mode`        - Target MTTKRP mode (excluded from the product)
/// * `cp_rank`     - CP rank R
/// * `block_shape` - Block shape for global index reconstruction
#[inline]
fn kr_row_hicoo<T: Float>(
    block_coord: &[usize],
    local_coord: &[usize],
    factors: &[ArrayView2<T>],
    mode: usize,
    cp_rank: usize,
    block_shape: &[usize],
) -> Vec<T> {
    let n_modes = factors.len();
    let mut kr_row = vec![T::one(); cp_rank];
    let mut initialized = false;

    for k in 0..n_modes {
        if k == mode {
            continue;
        }
        let global_k = block_coord[k] * block_shape[k] + local_coord[k];
        let factor_row = factors[k].row(global_k);
        if !initialized {
            for r in 0..cp_rank {
                kr_row[r] = factor_row[r];
            }
            initialized = true;
        } else {
            for r in 0..cp_rank {
                kr_row[r] = kr_row[r] * factor_row[r];
            }
        }
    }

    kr_row
}

/// Accumulate one HiCOO block into an output row-slice.
///
/// `row_offset` is the absolute first row of this block in the mode dimension
/// (i.e. `block_coords[b][mode] * block_shape[mode]`). The slice `out` covers
/// exactly `local_rows` rows starting at `row_offset`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_block_into_slice<T: Float>(
    block_coord: &[usize],
    block_start: usize,
    block_end: usize,
    local_coords: &[Vec<usize>],
    values: &[T],
    factors: &[ArrayView2<T>],
    mode: usize,
    cp_rank: usize,
    block_shape: &[usize],
    row_offset: usize,
    out: &mut Array2<T>,
) {
    for i in block_start..block_end {
        let lc = &local_coords[i];
        let local_row = lc[mode];
        let global_row = row_offset + local_row;
        let val = values[i];
        let kr = kr_row_hicoo(block_coord, lc, factors, mode, cp_rank, block_shape);
        for r in 0..cp_rank {
            out[[global_row, r]] = out[[global_row, r]] + val * kr[r];
        }
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Compute MTTKRP (Matricized Tensor × Khatri-Rao Product) for a HiCOO tensor.
///
/// Iterates over every non-empty block and every nonzero within that block,
/// accumulating contributions into the output matrix. The block structure
/// improves cache locality compared to plain COO by keeping related coordinate
/// and value data together.
///
/// # Arguments
///
/// * `tensor`  - N-dimensional sparse tensor in HiCOO format
/// * `factors` - Factor matrices, one per mode. `factors[k]` has shape
///   `(tensor.shape()[k], R)`.
/// * `mode`    - Target mode; output has shape `(tensor.shape()[mode], R)`
///
/// # Returns
///
/// Matrix with shape `(tensor.shape()[mode], R)` where `R` is the CP rank.
///
/// # Errors
///
/// - `mode >= tensor.ndim()`
/// - `factors.len() != tensor.ndim()`
/// - Factor matrices have inconsistent column counts (non-uniform rank)
/// - `factors[k].shape()[0] != tensor.shape()[k]` for any `k`
///
/// # Complexity
///
/// - Time:  `O(nnz · (N-1) · R)`
/// - Space: `O(shape[mode] · R)` output + `O(R)` scratch per nonzero
///
/// # Examples
///
/// ```rust,ignore
/// use tenrso_sparse::{CooTensor, HiCooTensor};
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::mttkrp_hicoo::mttkrp_hicoo;
///
/// let mut coo = CooTensor::zeros(vec![4, 5, 6]).unwrap();
/// coo.push(vec![0, 1, 2], 1.0_f64).unwrap();
/// let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
/// let f0 = Array2::<f64>::ones((4, 3));
/// let f1 = Array2::<f64>::ones((5, 3));
/// let f2 = Array2::<f64>::ones((6, 3));
/// let result = mttkrp_hicoo(&hicoo, &[f0.view(), f1.view(), f2.view()], 0).unwrap();
/// assert_eq!(result.shape(), &[4, 3]);
/// ```
pub fn mttkrp_hicoo<T>(
    tensor: &HiCooTensor<T>,
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

    let block_shape = tensor.block_shape();
    let block_coords = tensor.block_coords();
    let block_ptrs = tensor.block_ptrs();
    let local_coords = tensor.local_coords();
    let values = tensor.values();

    for b in 0..tensor.num_blocks() {
        let bc = &block_coords[b];
        let row_offset = bc[mode] * block_shape[mode];
        let b_start = block_ptrs[b];
        let b_end = block_ptrs[b + 1];
        accumulate_block_into_slice(
            bc,
            b_start,
            b_end,
            local_coords,
            values,
            factors,
            mode,
            cp_rank,
            block_shape,
            row_offset,
            &mut result,
        );
    }

    Ok(result)
}

/// Parallel MTTKRP for HiCOO sparse tensors.
///
/// Groups blocks by `block_coords[b][mode]` (the output block-row key). All
/// blocks sharing the same key write exclusively into the row range
/// `[key*block_shape[mode], min((key+1)*block_shape[mode], shape[mode]))`.
/// These groups are disjoint in the output, so they are processed in parallel
/// without write races. After the parallel phase the per-group local buffers are
/// scattered (serially) into the global result array.
///
/// # Arguments
///
/// * `tensor`  - N-dimensional sparse tensor in HiCOO format
/// * `factors` - Factor matrices, one per mode. `factors[k]` has shape
///   `(tensor.shape()[k], R)`.
/// * `mode`    - Target mode; output has shape `(tensor.shape()[mode], R)`
///
/// # Returns
///
/// Matrix with shape `(tensor.shape()[mode], R)` where `R` is the CP rank.
///
/// # Errors
///
/// Same as [`mttkrp_hicoo`].
///
/// # Complexity
///
/// - Time:  `O(nnz·(N-1)·R / P)` parallel + `O(nnz)` scatter
/// - Space: `O(shape[mode]·R)` total across all groups + `O(R)` scratch
#[cfg(feature = "parallel")]
pub fn mttkrp_hicoo_parallel<T>(
    tensor: &HiCooTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Float + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::collections::BTreeMap;

    let cp_rank = validate_sparse_mttkrp_inputs(tensor.shape(), factors, mode)?;
    let mode_size = tensor.shape()[mode];

    if tensor.nnz() == 0 {
        return Ok(Array2::<T>::zeros((mode_size, cp_rank)));
    }

    let block_shape = tensor.block_shape();
    let bs_mode = block_shape[mode];
    let block_coords = tensor.block_coords();

    // Group block indices by their mode-dimension block-row key.
    // Blocks with key k write into rows [k*bs_mode, min((k+1)*bs_mode, mode_size)).
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (b, bc) in block_coords.iter().enumerate() {
        let key = bc[mode];
        groups.entry(key).or_default().push(b);
    }

    // Process each group in parallel; groups write to disjoint row ranges.
    let partial_results: Vec<(usize, Array2<T>)> = groups
        .into_par_iter()
        .map(|(key, block_list)| {
            let row_start = key * bs_mode;
            let row_end = (row_start + bs_mode).min(mode_size);
            let local_rows = row_end - row_start;
            let mut local = Array2::<T>::zeros((local_rows, cp_rank));

            let b_shape = tensor.block_shape();
            let b_coords = tensor.block_coords();
            let b_ptrs = tensor.block_ptrs();
            let l_coords = tensor.local_coords();
            let vals = tensor.values();

            for b in &block_list {
                let b = *b;
                let bc = &b_coords[b];
                let b_start = b_ptrs[b];
                let b_end = b_ptrs[b + 1];
                for i in b_start..b_end {
                    let lc = &l_coords[i];
                    let local_row = lc[mode];
                    let val = vals[i];
                    let kr = kr_row_hicoo(bc, lc, factors, mode, cp_rank, b_shape);
                    for r in 0..cp_rank {
                        local[[local_row, r]] = local[[local_row, r]] + val * kr[r];
                    }
                }
            }

            (row_start, local)
        })
        .collect();

    // Serial scatter: write each group's local buffer into the result.
    // Writes are disjoint so this is a plain assign, not an add.
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));
    for (row_start, local) in partial_results {
        let local_rows = local.shape()[0];
        for lr in 0..local_rows {
            for r in 0..cp_rank {
                result[[row_start + lr, r]] = local[[lr, r]];
            }
        }
    }

    Ok(result)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "csf"))]
mod tests {
    use super::*;
    use crate::mttkrp_sparse::mttkrp_sparse_coo;
    use scirs2_core::ndarray_ext::Array2;
    use scirs2_core::random::{SeedableRng, StdRng};
    use tenrso_sparse::coo::CooTensor;
    use tenrso_sparse::hicoo::HiCooTensor;

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// Build a random sparse 3-D COO tensor with approximately `nnz` nonzeros.
    fn build_sparse_3d_f64(shape: [usize; 3], nnz: usize, seed: u64) -> CooTensor<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(nnz);
        let mut values: Vec<f64> = Vec::with_capacity(nnz);
        let mut used = std::collections::HashSet::new();
        let mut count = 0usize;
        let mut safety = 0usize;
        while count < nnz && safety < nnz * 200 {
            safety += 1;
            let i: usize = rng.gen_range(0..shape[0]);
            let j: usize = rng.gen_range(0..shape[1]);
            let k: usize = rng.gen_range(0..shape[2]);
            if !used.insert((i, j, k)) {
                continue;
            }
            let v: f64 = rng.gen_range(-2.0_f64..2.0_f64);
            indices.push(vec![i, j, k]);
            values.push(v);
            count += 1;
        }
        CooTensor::new(indices, values, shape.to_vec()).expect("valid COO")
    }

    /// Generate a random matrix with the given shape.
    fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.gen_range(-2.0_f64..2.0_f64))
            .collect();
        Array2::from_shape_vec((rows, cols), data).expect("shape product matches data len")
    }

    /// Maximum absolute difference between two same-shaped matrices.
    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        assert_eq!(a.shape(), b.shape(), "shape mismatch in max_abs_diff");
        let mut m = 0.0f64;
        for i in 0..a.shape()[0] {
            for r in 0..a.shape()[1] {
                let d = (a[[i, r]] - b[[i, r]]).abs();
                if d > m {
                    m = d;
                }
            }
        }
        m
    }

    /// Oracle: convert HiCOO → COO, then run mttkrp_sparse_coo.
    fn mttkrp_hicoo_naive(
        tensor: &HiCooTensor<f64>,
        factors: &[ArrayView2<f64>],
        mode: usize,
    ) -> Result<Array2<f64>> {
        let coo = tensor.to_coo()?;
        mttkrp_sparse_coo(&coo, factors, mode)
    }

    // ── 1. mode 0/1/2 on 4×5×6, block [2,2,2], rank 4: vs dense oracle ──────

    #[test]
    fn test_hicoo_mode0_vs_dense() {
        let shape = [4, 5, 6];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 20, 11);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 4, 100);
        let u1 = random_matrix_f64(5, 4, 101);
        let u2 = random_matrix_f64(6, 4, 102);
        let factors = [u0.view(), u1.view(), u2.view()];

        let got = mttkrp_hicoo(&hicoo, &factors, 0).unwrap();
        let want = mttkrp_hicoo_naive(&hicoo, &factors, 0).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-10, "mode0 vs dense oracle");
    }

    #[test]
    fn test_hicoo_mode1_vs_dense() {
        let shape = [4, 5, 6];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 20, 21);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 4, 200);
        let u1 = random_matrix_f64(5, 4, 201);
        let u2 = random_matrix_f64(6, 4, 202);
        let factors = [u0.view(), u1.view(), u2.view()];

        let got = mttkrp_hicoo(&hicoo, &factors, 1).unwrap();
        let want = mttkrp_hicoo_naive(&hicoo, &factors, 1).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-10, "mode1 vs dense oracle");
    }

    #[test]
    fn test_hicoo_mode2_vs_dense() {
        let shape = [4, 5, 6];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 20, 31);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 4, 300);
        let u1 = random_matrix_f64(5, 4, 301);
        let u2 = random_matrix_f64(6, 4, 302);
        let factors = [u0.view(), u1.view(), u2.view()];

        let got = mttkrp_hicoo(&hicoo, &factors, 2).unwrap();
        let want = mttkrp_hicoo_naive(&hicoo, &factors, 2).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-10, "mode2 vs dense oracle");
    }

    // ── 2. mode 0/1/2: hicoo vs mttkrp_sparse_coo oracle ────────────────────

    #[test]
    fn test_hicoo_mode0_vs_coo() {
        let shape = [4, 5, 6];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 25, 41);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 4, 400);
        let u1 = random_matrix_f64(5, 4, 401);
        let u2 = random_matrix_f64(6, 4, 402);
        let factors = [u0.view(), u1.view(), u2.view()];

        let got = mttkrp_hicoo(&hicoo, &factors, 0).unwrap();
        let want = mttkrp_sparse_coo(&coo, &factors, 0).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-12, "mode0 vs coo oracle");
    }

    #[test]
    fn test_hicoo_mode1_vs_coo() {
        let shape = [4, 5, 6];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 25, 51);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 4, 500);
        let u1 = random_matrix_f64(5, 4, 501);
        let u2 = random_matrix_f64(6, 4, 502);
        let factors = [u0.view(), u1.view(), u2.view()];

        let got = mttkrp_hicoo(&hicoo, &factors, 1).unwrap();
        let want = mttkrp_sparse_coo(&coo, &factors, 1).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-12, "mode1 vs coo oracle");
    }

    #[test]
    fn test_hicoo_mode2_vs_coo() {
        let shape = [4, 5, 6];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 25, 61);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 4, 600);
        let u1 = random_matrix_f64(5, 4, 601);
        let u2 = random_matrix_f64(6, 4, 602);
        let factors = [u0.view(), u1.view(), u2.view()];

        let got = mttkrp_hicoo(&hicoo, &factors, 2).unwrap();
        let want = mttkrp_sparse_coo(&coo, &factors, 2).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-12, "mode2 vs coo oracle");
    }

    // ── 3. block_size = 1 (every element is its own block) ───────────────────

    #[test]
    fn test_hicoo_block_size_one_equals_coo() {
        let shape = [4, 5, 6];
        // block_shape = [1,1,1] means every nonzero is its own block
        let block = [1, 1, 1];
        let coo = build_sparse_3d_f64(shape, 18, 71);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 3, 700);
        let u1 = random_matrix_f64(5, 3, 701);
        let u2 = random_matrix_f64(6, 3, 702);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            let want = mttkrp_sparse_coo(&coo, &factors, mode).unwrap();
            assert!(
                max_abs_diff(&got, &want) < 1e-12,
                "block_size=1 mode={} should equal COO exactly",
                mode
            );
        }
    }

    // ── 4. block_size covers whole tensor (single block) ─────────────────────

    #[test]
    fn test_hicoo_single_block_covers_whole_tensor() {
        let shape = [4, 5, 6];
        // Block shape larger than the tensor → everything in one block
        let block = [8, 8, 8];
        let coo = build_sparse_3d_f64(shape, 15, 81);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        assert_eq!(hicoo.num_blocks(), 1, "expected exactly one block");

        let u0 = random_matrix_f64(4, 2, 800);
        let u1 = random_matrix_f64(5, 2, 801);
        let u2 = random_matrix_f64(6, 2, 802);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            let want = mttkrp_sparse_coo(&coo, &factors, mode).unwrap();
            assert!(
                max_abs_diff(&got, &want) < 1e-12,
                "single block mode={} should match COO",
                mode
            );
        }
    }

    // ── 5. 4th-order tensor, interior mode 2, block [1,1,2,2] ────────────────

    #[test]
    fn test_hicoo_4th_order_interior_mode() {
        let shape = [2, 3, 4, 5];
        let block = [1, 1, 2, 2];
        let mut rng = StdRng::seed_from_u64(91);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut used = std::collections::HashSet::new();
        for _ in 0..20 {
            let i0: usize = rng.gen_range(0..shape[0]);
            let i1: usize = rng.gen_range(0..shape[1]);
            let i2: usize = rng.gen_range(0..shape[2]);
            let i3: usize = rng.gen_range(0..shape[3]);
            if !used.insert((i0, i1, i2, i3)) {
                continue;
            }
            let v: f64 = rng.gen_range(-3.0..3.0);
            indices.push(vec![i0, i1, i2, i3]);
            values.push(v);
        }
        let coo = CooTensor::new(indices, values, shape.to_vec()).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(2, 3, 900);
        let u1 = random_matrix_f64(3, 3, 901);
        let u2 = random_matrix_f64(4, 3, 902);
        let u3 = random_matrix_f64(5, 3, 903);
        let factors = [u0.view(), u1.view(), u2.view(), u3.view()];

        let got = mttkrp_hicoo(&hicoo, &factors, 2).unwrap();
        let want = mttkrp_sparse_coo(&coo, &factors, 2).unwrap();
        assert_eq!(got.shape(), &[4, 3]);
        assert!(
            max_abs_diff(&got, &want) < 1e-12,
            "4th-order mode=2 mismatch"
        );
    }

    // ── 6. Empty tensor returns zeros ─────────────────────────────────────────

    #[test]
    fn test_hicoo_empty_tensor_returns_zeros() {
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        let u0 = random_matrix_f64(3, 4, 1000);
        let u1 = random_matrix_f64(4, 4, 1001);
        let u2 = random_matrix_f64(5, 4, 1002);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            assert_eq!(got.shape()[1], 4);
            for &v in got.iter() {
                assert_eq!(v, 0.0, "empty tensor: expected zero result, mode={}", mode);
            }
        }
    }

    // ── 7. Single nonzero hand-check ─────────────────────────────────────────

    #[test]
    fn test_hicoo_single_nonzero_hand_check() {
        // Nonzero at (1, 2, 3) with value 5.0
        let mut coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
        coo.push(vec![1, 2, 3], 5.0_f64).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();

        let mut f0 = Array2::<f64>::zeros((4, 2));
        f0[[1, 0]] = 2.0;
        f0[[1, 1]] = 3.0;
        let mut f1 = Array2::<f64>::zeros((5, 2));
        f1[[2, 0]] = 4.0;
        f1[[2, 1]] = 5.0;
        let mut f2 = Array2::<f64>::zeros((6, 2));
        f2[[3, 0]] = 6.0;
        f2[[3, 1]] = 7.0;
        let factors = [f0.view(), f1.view(), f2.view()];

        // mode=0: result[1, r] = 5 * f1[2,r] * f2[3,r]
        let r0 = mttkrp_hicoo(&hicoo, &factors, 0).unwrap();
        assert!((r0[[1, 0]] - 5.0 * 4.0 * 6.0).abs() < 1e-12);
        assert!((r0[[1, 1]] - 5.0 * 5.0 * 7.0).abs() < 1e-12);
        // All other rows must be zero
        for i in [0, 2, 3] {
            assert_eq!(r0[[i, 0]], 0.0);
            assert_eq!(r0[[i, 1]], 0.0);
        }

        // mode=1: result[2, r] = 5 * f0[1,r] * f2[3,r]
        let r1 = mttkrp_hicoo(&hicoo, &factors, 1).unwrap();
        assert!((r1[[2, 0]] - 5.0 * 2.0 * 6.0).abs() < 1e-12);
        assert!((r1[[2, 1]] - 5.0 * 3.0 * 7.0).abs() < 1e-12);

        // mode=2: result[3, r] = 5 * f0[1,r] * f1[2,r]
        let r2 = mttkrp_hicoo(&hicoo, &factors, 2).unwrap();
        assert!((r2[[3, 0]] - 5.0 * 2.0 * 4.0).abs() < 1e-12);
        assert!((r2[[3, 1]] - 5.0 * 3.0 * 5.0).abs() < 1e-12);
    }

    // ── 8. Error cases ────────────────────────────────────────────────────────

    #[test]
    fn test_hicoo_error_mode_oob() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        let u0 = Array2::<f64>::from_elem((2, 1), 1.0);
        let u1 = Array2::<f64>::from_elem((3, 1), 1.0);
        let u2 = Array2::<f64>::from_elem((4, 1), 1.0);
        let err = mttkrp_hicoo(&hicoo, &[u0.view(), u1.view(), u2.view()], 5);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.to_lowercase().contains("mode"),
            "expected 'mode' in error: {}",
            msg
        );
    }

    #[test]
    fn test_hicoo_error_wrong_factor_count() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        let u0 = Array2::<f64>::from_elem((2, 1), 1.0);
        let u1 = Array2::<f64>::from_elem((3, 1), 1.0);
        // Only 2 factors for a 3D tensor
        let err = mttkrp_hicoo(&hicoo, &[u0.view(), u1.view()], 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_hicoo_error_inconsistent_column_count() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        let u0 = Array2::<f64>::from_elem((2, 2), 1.0); // rank=2
        let u1 = Array2::<f64>::from_elem((3, 1), 1.0); // rank=1 (mismatch)
        let u2 = Array2::<f64>::from_elem((4, 2), 1.0);
        let err = mttkrp_hicoo(&hicoo, &[u0.view(), u1.view(), u2.view()], 0);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("columns"),
            "expected 'columns' in error: {}",
            msg
        );
    }

    #[test]
    fn test_hicoo_error_wrong_row_count() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        // u0 has 5 rows but tensor mode-0 has size 2
        let u0 = Array2::<f64>::from_elem((5, 1), 1.0);
        let u1 = Array2::<f64>::from_elem((3, 1), 1.0);
        let u2 = Array2::<f64>::from_elem((4, 1), 1.0);
        let err = mttkrp_hicoo(&hicoo, &[u0.view(), u1.view(), u2.view()], 0);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("rows"), "expected 'rows' in error: {}", msg);
    }

    // ── 9. Parallel parity ───────────────────────────────────────────────────

    #[cfg(feature = "parallel")]
    #[test]
    fn test_hicoo_parallel_matches_serial() {
        let shape = [4, 5, 6];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 30, 2020);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(4, 4, 2100);
        let u1 = random_matrix_f64(5, 4, 2101);
        let u2 = random_matrix_f64(6, 4, 2102);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let serial = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            let parallel = mttkrp_hicoo_parallel(&hicoo, &factors, mode).unwrap();
            assert_eq!(serial.shape(), parallel.shape());
            assert!(
                max_abs_diff(&serial, &parallel) < 1e-12,
                "serial vs parallel differ at mode {}",
                mode
            );
        }
    }

    // ── 10. High rank (rank=64, 8×8×8, block [4,4,4]) ────────────────────────

    #[test]
    fn test_hicoo_high_rank_stress() {
        let shape = [8, 8, 8];
        let block = [4, 4, 4];
        let coo = build_sparse_3d_f64(shape, 60, 3030);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(8, 64, 3100);
        let u1 = random_matrix_f64(8, 64, 3101);
        let u2 = random_matrix_f64(8, 64, 3102);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            let want = mttkrp_sparse_coo(&coo, &factors, mode).unwrap();
            assert_eq!(got.shape(), &[8, 64]);
            assert!(
                max_abs_diff(&got, &want) < 1e-8,
                "high rank mode={} exceeded tolerance",
                mode
            );
        }
    }

    // ── 11. Rank-1 edge case ─────────────────────────────────────────────────

    #[test]
    fn test_hicoo_rank1() {
        let shape = [3, 4, 5];
        let block = [2, 2, 2];
        let coo = build_sparse_3d_f64(shape, 10, 4040);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(3, 1, 4100);
        let u1 = random_matrix_f64(4, 1, 4101);
        let u2 = random_matrix_f64(5, 1, 4102);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            let want = mttkrp_sparse_coo(&coo, &factors, mode).unwrap();
            assert_eq!(got.shape()[1], 1);
            assert!(
                max_abs_diff(&got, &want) < 1e-12,
                "rank-1 mode={} mismatch",
                mode
            );
        }
    }

    // ── 12. All-nonzero tensor (block fully dense) ────────────────────────────

    #[test]
    fn test_hicoo_fully_dense_matches_coo() {
        // Build a fully populated small tensor so every possible index is nonzero
        let shape = [3, 3, 3];
        let block = [2, 2, 2];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut rng = StdRng::seed_from_u64(5050);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    indices.push(vec![i, j, k]);
                    values.push(rng.gen_range(-1.0_f64..1.0_f64));
                }
            }
        }
        let coo = CooTensor::new(indices, values, shape.to_vec()).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(3, 3, 5100);
        let u1 = random_matrix_f64(3, 3, 5101);
        let u2 = random_matrix_f64(3, 3, 5102);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            let want = mttkrp_sparse_coo(&coo, &factors, mode).unwrap();
            assert!(
                max_abs_diff(&got, &want) < 1e-12,
                "fully-dense mode={} mismatch",
                mode
            );
        }
    }

    // ── 13. Asymmetric block shape ────────────────────────────────────────────

    #[test]
    fn test_hicoo_asymmetric_block_shape() {
        let shape = [6, 4, 9];
        let block = [3, 2, 4]; // asymmetric blocks, won't tile evenly on dim 2
        let coo = build_sparse_3d_f64(shape, 30, 6060);
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let u0 = random_matrix_f64(6, 5, 6100);
        let u1 = random_matrix_f64(4, 5, 6101);
        let u2 = random_matrix_f64(9, 5, 6102);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo(&hicoo, &factors, mode).unwrap();
            let want = mttkrp_sparse_coo(&coo, &factors, mode).unwrap();
            assert!(
                max_abs_diff(&got, &want) < 1e-12,
                "asymmetric block mode={} mismatch",
                mode
            );
        }
    }

    // ── 14. Parallel with empty tensor ────────────────────────────────────────

    #[cfg(feature = "parallel")]
    #[test]
    fn test_hicoo_parallel_empty_tensor() {
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        let u0 = random_matrix_f64(3, 2, 7100);
        let u1 = random_matrix_f64(4, 2, 7101);
        let u2 = random_matrix_f64(5, 2, 7102);
        let factors = [u0.view(), u1.view(), u2.view()];

        for mode in 0..3 {
            let got = mttkrp_hicoo_parallel(&hicoo, &factors, mode).unwrap();
            for &v in got.iter() {
                assert_eq!(
                    v, 0.0,
                    "parallel empty: expected zero at mode {}",
                    mode
                );
            }
        }
    }

    // ── 15. 5-D tensor ────────────────────────────────────────────────────────

    #[test]
    fn test_hicoo_5d_tensor() {
        let shape = [2, 3, 4, 3, 2];
        let block = [1, 2, 2, 2, 1];
        let mut rng = StdRng::seed_from_u64(8080);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut used = std::collections::HashSet::new();
        for _ in 0..30 {
            let idx: Vec<usize> = shape.iter().map(|&s| rng.gen_range(0..s)).collect();
            let key = (idx[0], idx[1], idx[2], idx[3], idx[4]);
            if used.insert(key) {
                let v: f64 = rng.gen_range(-2.0..2.0);
                indices.push(idx);
                values.push(v);
            }
        }
        let coo = CooTensor::new(indices, values, shape.to_vec()).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &block).unwrap();
        let factors: Vec<Array2<f64>> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| random_matrix_f64(s, 3, 8100 + i as u64))
            .collect();
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        for mode in 0..shape.len() {
            let got = mttkrp_hicoo(&hicoo, &factor_views, mode).unwrap();
            let want = mttkrp_sparse_coo(&coo, &factor_views, mode).unwrap();
            assert!(
                max_abs_diff(&got, &want) < 1e-12,
                "5D tensor mode={} mismatch",
                mode
            );
        }
    }
}
