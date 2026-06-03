//! Sparse MTTKRP (Matricized Tensor Times Khatri-Rao Product)
//!
//! Sparse variants of the MTTKRP operation that operate directly on sparse
//! tensor formats instead of densifying. For tensors with high sparsity
//! (>90% zeros, common in recommender systems and graph analytics) this
//! reduces MTTKRP cost from `O(I·J·K·R)` to `O(nnz·R)` — often a speedup
//! of **several orders of magnitude**.
//!
//! # Mathematical definition
//!
//! For a 3rd-order tensor `T` with shape `(I, J, K)` and factor matrices
//! `B ∈ R^{J×R}`, `C ∈ R^{K×R}`, the mode-0 MTTKRP computes `M ∈ R^{I×R}`:
//!
//! ```text
//! M[i, r] = sum_{j,k} T[i, j, k] * B[j, r] * C[k, r]
//! ```
//!
//! For a sparse tensor stored as `(indices, value)` triples, this becomes:
//!
//! ```text
//! for each nonzero (idx, val) in T:
//!     i = idx[mode]
//!     M[i, :] += val * prod_{k != mode} factors[k][idx[k], :]
//! ```
//!
//! The generalization to N-mode tensors is immediate: iterate all non-mode
//! factor columns.
//!
//! # Supported formats
//!
//! - **COO** (`CooTensor`): general N-D sparse tensor via
//!   [`mttkrp_sparse_coo`] and [`mttkrp_sparse_coo_parallel`].
//! - **CSR** is not implemented in this module — the `CsrMatrix` in
//!   `tenrso-sparse` is 2D-only (rows × cols), whereas MTTKRP needs an
//!   N-dimensional sparse format. Converting COO → matricized CSR and then
//!   doing SpMM would materialize the full Khatri-Rao product and defeat
//!   the sparse win. A future N-D CSF-based MTTKRP is tracked in TODO.md.
//!
//! # Correctness invariant
//!
//! Sparse MTTKRP **MUST** produce the same output as the dense MTTKRP on
//! the densified tensor (up to floating-point accumulation order). This
//! is verified in tests against [`crate::mttkrp::mttkrp`].
//!
//! # Duplicate-index semantics
//!
//! If the COO tensor contains multiple entries at the same coordinate,
//! this implementation accumulates them additively. This matches the
//! semantics of [`tenrso_sparse::coo::CooTensor::deduplicate`] (summed
//! values) and of the "densify then MTTKRP" baseline if duplicates are
//! summed at densify time.
//!
//! # SciRS2 integration
//!
//! All array operations use [`scirs2_core::ndarray_ext`]. Direct use of
//! `ndarray` or `rand` is forbidden per SCIRS2_INTEGRATION_POLICY.md.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::numeric::{Num, One, Zero};
use tenrso_sparse::coo::CooTensor;

/// Validate that a list of factor matrices is consistent with the target
/// sparse tensor shape and mode.
///
/// Returns the CP rank `R` on success.
pub(crate) fn validate_sparse_mttkrp_inputs<T>(
    shape: &[usize],
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<usize> {
    let rank_tensor = shape.len();

    if mode >= rank_tensor {
        anyhow::bail!(
            "Mode {} out of bounds for tensor with rank {}",
            mode,
            rank_tensor
        );
    }

    if factors.len() != rank_tensor {
        anyhow::bail!(
            "Number of factor matrices ({}) must match tensor rank ({})",
            factors.len(),
            rank_tensor
        );
    }

    if factors.is_empty() {
        anyhow::bail!("Need at least 1 factor matrix for MTTKRP");
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
        if factor.shape()[0] != shape[i] {
            anyhow::bail!(
                "Factor matrix {} has {} rows, expected {} (tensor mode-{} size)",
                i,
                factor.shape()[0],
                shape[i],
                i
            );
        }
    }

    Ok(cp_rank)
}

/// Compute the Khatri-Rao row for a single nonzero.
///
/// Given a nonzero at multi-index `idx`, the value contribution to output
/// row `idx[mode]` is `val * prod_{k != mode} factors[k][idx[k], :]`.
/// This function returns the rank-length vector
/// `prod_{k != mode} factors[k][idx[k], :]`.
///
/// The product is accumulated in FORWARD mode order (skipping `mode`) to
/// match the convention used by the dense [`crate::mttkrp::mttkrp`] kernel.
#[inline]
fn kr_row_for_nonzero<T>(
    idx: &[usize],
    factors: &[ArrayView2<T>],
    mode: usize,
    cp_rank: usize,
) -> Vec<T>
where
    T: Clone + Num + One,
{
    let n_modes = factors.len();
    let mut kr_row: Vec<T> = vec![T::one(); cp_rank];
    let mut initialized = false;

    for (k, factor_k) in factors.iter().enumerate().take(n_modes) {
        if k == mode {
            continue;
        }
        let coord = idx[k];
        if !initialized {
            // First non-mode factor: overwrite the ones with factor values.
            for r in 0..cp_rank {
                kr_row[r] = factor_k[[coord, r]].clone();
            }
            initialized = true;
        } else {
            for r in 0..cp_rank {
                kr_row[r] = kr_row[r].clone() * factor_k[[coord, r]].clone();
            }
        }
    }

    kr_row
}

/// Compute sparse MTTKRP on a COO-format tensor.
///
/// Equivalent to the dense [`crate::mttkrp::mttkrp`] on `tensor.to_dense()`
/// but iterates only over the nonzeros, reducing cost from
/// `O(∏ I_i · R)` to `O(nnz · (N-1) · R)`.
///
/// # Arguments
///
/// * `tensor` - N-dimensional sparse tensor in COO format
/// * `factors` - Factor matrices, one per mode. `factors[k]` has shape
///   `(tensor.shape()[k], R)`.
/// * `mode` - The mode to compute MTTKRP for (0-indexed, `< tensor.rank()`)
///
/// # Returns
///
/// Matrix with shape `(tensor.shape()[mode], R)` where `R` is the CP rank
/// (the number of columns in every factor matrix).
///
/// # Errors
///
/// - `mode >= tensor.rank()`
/// - `factors.len() != tensor.rank()`
/// - factor matrices have inconsistent column counts (non-uniform rank)
/// - `factors[k].shape()[0] != tensor.shape()[k]` for any `k`
///
/// # Complexity
///
/// - Time: `O(nnz · (N-1) · R)` for iterating nonzeros and accumulating
///   rank-R contributions through the per-mode factors.
/// - Space: `O(I_mode · R)` for the output, plus `O(R)` scratch per
///   nonzero (the Khatri-Rao row).
///
/// # Duplicate indices
///
/// If `tensor` contains multiple entries at the same coordinate their
/// contributions are summed additively. This matches
/// [`tenrso_sparse::coo::CooTensor::deduplicate`] semantics.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::mttkrp_sparse_coo;
/// use tenrso_sparse::coo::CooTensor;
///
/// // 2x3x4 sparse tensor with 2 nonzeros
/// let indices = vec![vec![0, 1, 2], vec![1, 2, 3]];
/// let values = vec![5.0_f64, 7.0];
/// let coo = CooTensor::new(indices, values, vec![2, 3, 4]).unwrap();
///
/// let u1 = Array2::<f64>::from_elem((2, 3), 1.0);
/// let u2 = Array2::<f64>::from_elem((3, 3), 1.0);
/// let u3 = Array2::<f64>::from_elem((4, 3), 1.0);
///
/// let m = mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 0).unwrap();
/// assert_eq!(m.shape(), &[2, 3]);
/// // Row 0: KR = 1*1 = 1 for every r, contribution = 5, total M[0,r] = 5
/// // Row 1: KR = 1*1 = 1 for every r, contribution = 7, total M[1,r] = 7
/// for r in 0..3 {
///     assert!((m[[0, r]] - 5.0).abs() < 1e-12);
///     assert!((m[[1, r]] - 7.0).abs() < 1e-12);
/// }
/// ```
pub fn mttkrp_sparse_coo<T>(
    tensor: &CooTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Clone + Num + One + Zero,
{
    let shape = tensor.shape();
    let cp_rank = validate_sparse_mttkrp_inputs(shape, factors, mode)?;

    let mode_size = shape[mode];
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    let indices = tensor.indices();
    let values = tensor.values();

    for (idx, val) in indices.iter().zip(values.iter()) {
        let i = idx[mode];
        let kr_row = kr_row_for_nonzero(idx, factors, mode, cp_rank);
        // result[i, :] += val * kr_row
        for r in 0..cp_rank {
            let contribution = val.clone() * kr_row[r].clone();
            result[[i, r]] = result[[i, r]].clone() + contribution;
        }
    }

    Ok(result)
}

/// Parallel variant of [`mttkrp_sparse_coo`] using Rayon.
///
/// Groups nonzeros by their target output row (`indices[mode]`) so each
/// output row can be accumulated independently — no atomic updates, no
/// write races. For large `nnz` this typically scales to ~core-count
/// speedup.
///
/// For very small `nnz` (< a few thousand) the serial variant is
/// usually faster due to thread-pool overhead.
///
/// # Panics
///
/// Does not panic on valid inputs. See [`mttkrp_sparse_coo`] for the
/// error conditions.
///
/// # Correctness
///
/// Produces bit-identical output to the serial [`mttkrp_sparse_coo`]
/// **when summation order within a row is preserved**. Because we sort
/// nonzeros by target row (stable order within each row) and accumulate
/// sequentially per row, this holds.
///
/// # Complexity
///
/// - Time: `O(nnz · log(nnz) + nnz · (N-1) · R / P)` for `P` threads.
///   The `log(nnz)` factor is the row-grouping sort.
/// - Space: `O(nnz + I_mode · R)`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::mttkrp_sparse_coo_parallel;
/// use tenrso_sparse::coo::CooTensor;
///
/// let indices = vec![vec![0, 0, 0], vec![1, 1, 1]];
/// let values = vec![2.0_f64, 3.0];
/// let coo = CooTensor::new(indices, values, vec![2, 2, 2]).unwrap();
///
/// let u1 = Array2::<f64>::from_elem((2, 2), 1.0);
/// let u2 = Array2::<f64>::from_elem((2, 2), 1.0);
/// let u3 = Array2::<f64>::from_elem((2, 2), 1.0);
///
/// let m = mttkrp_sparse_coo_parallel(&coo, &[u1.view(), u2.view(), u3.view()], 0).unwrap();
/// assert_eq!(m.shape(), &[2, 2]);
/// ```
#[cfg(feature = "parallel")]
pub fn mttkrp_sparse_coo_parallel<T>(
    tensor: &CooTensor<T>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Clone + Num + One + Zero + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let shape = tensor.shape();
    let cp_rank = validate_sparse_mttkrp_inputs(shape, factors, mode)?;

    let mode_size = shape[mode];
    let indices = tensor.indices();
    let values = tensor.values();
    let nnz = indices.len();

    // Short-circuit for empty tensors.
    if nnz == 0 {
        return Ok(Array2::<T>::zeros((mode_size, cp_rank)));
    }

    // Bucket nonzeros by target row. Each bucket holds (idx, value) pairs.
    let mut buckets: Vec<Vec<(&Vec<usize>, &T)>> = (0..mode_size).map(|_| Vec::new()).collect();
    for (idx, val) in indices.iter().zip(values.iter()) {
        let row = idx[mode];
        buckets[row].push((idx, val));
    }

    // Process each row in parallel. Each row is an independent accumulator.
    let rows: Vec<Vec<T>> = buckets
        .par_iter()
        .map(|bucket| {
            let mut row_acc: Vec<T> = vec![T::zero(); cp_rank];
            for (idx, val) in bucket.iter() {
                let kr_row = kr_row_for_nonzero(idx, factors, mode, cp_rank);
                for r in 0..cp_rank {
                    let contribution = (*val).clone() * kr_row[r].clone();
                    row_acc[r] = row_acc[r].clone() + contribution;
                }
            }
            row_acc
        })
        .collect();

    // Assemble into Array2.
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));
    for (i, row_vals) in rows.iter().enumerate() {
        for r in 0..cp_rank {
            result[[i, r]] = row_vals[r].clone();
        }
    }

    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mttkrp::mttkrp;
    use scirs2_core::ndarray_ext::{array, Array, Array2};
    use scirs2_core::numeric::Float;
    use scirs2_core::random::{SeedableRng, StdRng};

    // --- Helpers -------------------------------------------------------------

    /// Densify a COO tensor into an `IxDyn` dense array by summing duplicates.
    fn coo_to_dense<T>(coo: &CooTensor<T>) -> Array<T, scirs2_core::ndarray_ext::IxDyn>
    where
        T: Clone + Num + Zero,
    {
        let shape = coo.shape().to_vec();
        let mut dense = Array::<T, _>::zeros(scirs2_core::ndarray_ext::IxDyn(&shape));
        for (idx, val) in coo.indices().iter().zip(coo.values().iter()) {
            let prev = dense[&idx[..]].clone();
            dense[&idx[..]] = prev + val.clone();
        }
        dense
    }

    /// Build a random dense tensor + matching COO with every element present.
    fn build_random_dense_matching_coo<T>(
        shape: &[usize],
        seed: u64,
    ) -> (Array<T, scirs2_core::ndarray_ext::IxDyn>, CooTensor<T>)
    where
        T: Clone + Num + Zero + Float,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let total: usize = shape.iter().product();
        let mut data: Vec<T> = Vec::with_capacity(total);
        for _ in 0..total {
            let v: f64 = rng.random_f64();
            let t_val = T::from(v).unwrap_or_else(T::zero);
            data.push(t_val);
        }
        let dense =
            Array::<T, _>::from_shape_vec(scirs2_core::ndarray_ext::IxDyn(shape), data.clone())
                .expect("shape product == data len");

        // Build COO from every element (including zero-valued ones).
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(total);
        let mut values: Vec<T> = Vec::with_capacity(total);
        for (flat, v) in data.iter().enumerate().take(total) {
            let mut remaining = flat;
            let mut multi = vec![0usize; shape.len()];
            for (d, &s) in shape.iter().enumerate().rev() {
                multi[d] = remaining % s;
                remaining /= s;
            }
            indices.push(multi);
            values.push(*v);
        }
        let coo = CooTensor::new(indices, values, shape.to_vec()).expect("valid COO");
        (dense, coo)
    }

    /// Build a sparse 3rd-order tensor with ~nnz random nonzeros.
    fn build_sparse_3d_f64(
        shape: [usize; 3],
        nnz: usize,
        seed: u64,
    ) -> (Array<f64, scirs2_core::ndarray_ext::IxDyn>, CooTensor<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dense = Array::<f64, _>::zeros(scirs2_core::ndarray_ext::IxDyn(&shape));
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(nnz);
        let mut values: Vec<f64> = Vec::with_capacity(nnz);
        let mut used = std::collections::HashSet::new();

        let mut count = 0usize;
        let mut safety = 0usize;
        while count < nnz && safety < nnz * 100 {
            safety += 1;
            let i: usize = rng.gen_range(0..shape[0]);
            let j: usize = rng.gen_range(0..shape[1]);
            let k: usize = rng.gen_range(0..shape[2]);
            if !used.insert((i, j, k)) {
                continue;
            }
            let v: f64 = rng.random_f64() * 2.0 - 1.0;
            dense[[i, j, k]] = v;
            indices.push(vec![i, j, k]);
            values.push(v);
            count += 1;
        }
        let coo = CooTensor::new(indices, values, shape.to_vec()).expect("valid COO");
        (dense, coo)
    }

    fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(rng.random_f64() * 2.0 - 1.0);
        }
        Array2::<f64>::from_shape_vec((rows, cols), data).expect("shape product == data len")
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        assert_eq!(a.shape(), b.shape());
        let mut m = 0.0f64;
        for ((i, j), x) in a.indexed_iter() {
            let d = (x - b[[i, j]]).abs();
            if d > m {
                m = d;
            }
        }
        m
    }

    // --- Correctness tests (vs dense MTTKRP oracle) --------------------------

    #[test]
    fn test_sparse_coo_matches_dense_mode0() {
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_3d_f64(shape, 10, 42);
        let u1 = random_matrix_f64(4, 3, 1);
        let u2 = random_matrix_f64(5, 3, 2);
        let u3 = random_matrix_f64(6, 3, 3);

        let dense_result =
            mttkrp(&dense.view(), &[u1.view(), u2.view(), u3.view()], 0).expect("dense mttkrp");
        let sparse_result =
            mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 0).expect("sparse mttkrp");

        assert_eq!(sparse_result.shape(), dense_result.shape());
        assert!(
            max_abs_diff(&sparse_result, &dense_result) < 1e-10,
            "sparse mode-0 result diverged from dense"
        );
    }

    #[test]
    fn test_sparse_coo_matches_dense_mode1() {
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_3d_f64(shape, 15, 123);
        let u1 = random_matrix_f64(4, 4, 11);
        let u2 = random_matrix_f64(5, 4, 12);
        let u3 = random_matrix_f64(6, 4, 13);

        let dense_result =
            mttkrp(&dense.view(), &[u1.view(), u2.view(), u3.view()], 1).expect("dense mttkrp");
        let sparse_result =
            mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 1).expect("sparse mttkrp");

        assert!(max_abs_diff(&sparse_result, &dense_result) < 1e-10);
    }

    #[test]
    fn test_sparse_coo_matches_dense_mode2() {
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_3d_f64(shape, 20, 321);
        let u1 = random_matrix_f64(4, 2, 21);
        let u2 = random_matrix_f64(5, 2, 22);
        let u3 = random_matrix_f64(6, 2, 23);

        let dense_result =
            mttkrp(&dense.view(), &[u1.view(), u2.view(), u3.view()], 2).expect("dense mttkrp");
        let sparse_result =
            mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 2).expect("sparse mttkrp");

        assert!(max_abs_diff(&sparse_result, &dense_result) < 1e-10);
    }

    #[test]
    fn test_sparse_coo_4th_order() {
        // 4D tensor I×J×K×L, mode=1
        let shape = [3, 4, 5, 3];
        let mut rng = StdRng::seed_from_u64(9999);
        let mut dense = Array::<f64, _>::zeros(scirs2_core::ndarray_ext::IxDyn(&shape));
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut used = std::collections::HashSet::new();
        let target_nnz = 12usize;
        let mut count = 0;
        while count < target_nnz {
            let i: usize = rng.gen_range(0..shape[0]);
            let j: usize = rng.gen_range(0..shape[1]);
            let k: usize = rng.gen_range(0..shape[2]);
            let l: usize = rng.gen_range(0..shape[3]);
            if !used.insert((i, j, k, l)) {
                continue;
            }
            let v: f64 = rng.random_f64();
            dense[[i, j, k, l]] = v;
            indices.push(vec![i, j, k, l]);
            values.push(v);
            count += 1;
        }
        let coo = CooTensor::new(indices, values, shape.to_vec()).expect("valid COO");
        let u0 = random_matrix_f64(3, 5, 100);
        let u1 = random_matrix_f64(4, 5, 101);
        let u2 = random_matrix_f64(5, 5, 102);
        let u3 = random_matrix_f64(3, 5, 103);
        let factors = [u0.view(), u1.view(), u2.view(), u3.view()];

        let dense_res = mttkrp(&dense.view(), &factors, 1).expect("dense");
        let sparse_res = mttkrp_sparse_coo(&coo, &factors, 1).expect("sparse");

        assert_eq!(sparse_res.shape(), &[4, 5]);
        assert!(max_abs_diff(&sparse_res, &dense_res) < 1e-10);
    }

    #[test]
    fn test_sparse_coo_all_zeros() {
        // A sparse tensor with no nonzeros should produce an all-zero M.
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).expect("zeros");
        let u1 = random_matrix_f64(3, 4, 1);
        let u2 = random_matrix_f64(4, 4, 2);
        let u3 = random_matrix_f64(5, 4, 3);
        let m =
            mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 0).expect("sparse mttkrp");
        assert_eq!(m.shape(), &[3, 4]);
        for &v in m.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_sparse_coo_fully_dense_matches_exact() {
        // Every element represented in COO ⇒ result == dense result.
        let shape = [3usize, 4, 3];
        let (dense, coo) = build_random_dense_matching_coo::<f64>(&shape, 7);
        let u1 = random_matrix_f64(3, 2, 201);
        let u2 = random_matrix_f64(4, 2, 202);
        let u3 = random_matrix_f64(3, 2, 203);
        let factors = [u1.view(), u2.view(), u3.view()];

        let dense_res = mttkrp(&dense.view(), &factors, 0).expect("dense");
        let sparse_res = mttkrp_sparse_coo(&coo, &factors, 0).expect("sparse");

        assert!(max_abs_diff(&sparse_res, &dense_res) < 1e-10);
    }

    #[test]
    fn test_sparse_coo_rank1() {
        // Rank-1 factor matrices edge case.
        let shape = [3, 4, 5];
        let (dense, coo) = build_sparse_3d_f64(shape, 8, 55);
        let u1 = random_matrix_f64(3, 1, 10);
        let u2 = random_matrix_f64(4, 1, 11);
        let u3 = random_matrix_f64(5, 1, 12);
        let factors = [u1.view(), u2.view(), u3.view()];

        let dense_res = mttkrp(&dense.view(), &factors, 0).expect("dense");
        let sparse_res = mttkrp_sparse_coo(&coo, &factors, 0).expect("sparse");
        assert_eq!(sparse_res.shape(), &[3, 1]);
        assert!(max_abs_diff(&sparse_res, &dense_res) < 1e-10);
    }

    #[test]
    fn test_sparse_coo_rank128_stress() {
        // Stress test with a large R. Small tensor with moderate nnz.
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_3d_f64(shape, 30, 777);
        let u1 = random_matrix_f64(4, 128, 1001);
        let u2 = random_matrix_f64(5, 128, 1002);
        let u3 = random_matrix_f64(6, 128, 1003);
        let factors = [u1.view(), u2.view(), u3.view()];

        let dense_res = mttkrp(&dense.view(), &factors, 1).expect("dense");
        let sparse_res = mttkrp_sparse_coo(&coo, &factors, 1).expect("sparse");
        assert_eq!(sparse_res.shape(), &[5, 128]);
        assert!(max_abs_diff(&sparse_res, &dense_res) < 1e-8);
    }

    // --- Edge / error cases --------------------------------------------------

    #[test]
    fn test_sparse_coo_invalid_mode() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).expect("zeros");
        let u1 = Array2::<f64>::from_elem((2, 1), 1.0);
        let u2 = Array2::<f64>::from_elem((3, 1), 1.0);
        let u3 = Array2::<f64>::from_elem((4, 1), 1.0);
        let err = mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 5);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("error present"));
        assert!(msg.contains("Mode"), "got: {}", msg);
    }

    #[test]
    fn test_sparse_coo_wrong_num_factors() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).expect("zeros");
        let u1 = Array2::<f64>::from_elem((2, 1), 1.0);
        let u2 = Array2::<f64>::from_elem((3, 1), 1.0);
        let err = mttkrp_sparse_coo(&coo, &[u1.view(), u2.view()], 0);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("error present"));
        assert!(msg.contains("Number of factor matrices"), "got: {}", msg);
    }

    #[test]
    fn test_sparse_coo_factor_rank_mismatch() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).expect("zeros");
        let u1 = Array2::<f64>::from_elem((2, 2), 1.0); // rank 2
        let u2 = Array2::<f64>::from_elem((3, 1), 1.0); // rank 1
        let u3 = Array2::<f64>::from_elem((4, 2), 1.0); // rank 2
        let err = mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 0);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("error present"));
        assert!(msg.contains("columns"), "got: {}", msg);
    }

    #[test]
    fn test_sparse_coo_factor_shape_mismatch() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).expect("zeros");
        // u1 has wrong row count for mode 0 (expected 2).
        let u1 = array![[1.0_f64, 2.0], [1.0, 2.0], [3.0, 4.0]];
        let u2 = array![[1.0_f64, 2.0], [1.0, 2.0], [3.0, 4.0]];
        let u3 = array![[1.0_f64, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let err = mttkrp_sparse_coo(&coo, &[u1.view(), u2.view(), u3.view()], 0);
        assert!(err.is_err());
        let msg = format!("{}", err.expect_err("error present"));
        assert!(msg.contains("rows"), "got: {}", msg);
    }

    #[test]
    fn test_sparse_coo_duplicate_indices_aggregate() {
        // Two entries at the same coordinate should accumulate additively.
        // Build a dense tensor where that coordinate holds the summed value.
        let indices: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![0, 1, 2], vec![1, 0, 1]];
        let values = vec![3.0_f64, 4.0, 5.0];
        let shape = vec![2, 3, 4];
        let coo = CooTensor::new(indices.clone(), values.clone(), shape.clone())
            .expect("valid COO with dups");

        // Reference dense via densify-with-sum (this is how CooTensor::to_dense
        // would behave if it summed; here we replicate that explicitly).
        let mut dense = Array::<f64, _>::zeros(scirs2_core::ndarray_ext::IxDyn(&shape));
        for (idx, &v) in indices.iter().zip(values.iter()) {
            let prev = dense[&idx[..]];
            dense[&idx[..]] = prev + v;
        }

        let u1 = random_matrix_f64(2, 3, 33);
        let u2 = random_matrix_f64(3, 3, 34);
        let u3 = random_matrix_f64(4, 3, 35);
        let factors = [u1.view(), u2.view(), u3.view()];

        let dense_res = mttkrp(&dense.view(), &factors, 0).expect("dense");
        let sparse_res = mttkrp_sparse_coo(&coo, &factors, 0).expect("sparse");
        assert!(max_abs_diff(&sparse_res, &dense_res) < 1e-10);
    }

    #[test]
    fn test_coo_to_dense_helper_matches_from_sparse() {
        // Sanity: our test helper coo_to_dense sums duplicates.
        let indices = vec![vec![0, 0], vec![0, 0], vec![1, 1]];
        let values = vec![1.0_f64, 2.0, 5.0];
        let coo = CooTensor::new(indices, values, vec![2, 2]).expect("valid");
        let dense = coo_to_dense(&coo);
        assert_eq!(dense[[0, 0]], 3.0);
        assert_eq!(dense[[1, 1]], 5.0);
        assert_eq!(dense[[0, 1]], 0.0);
    }

    // --- Parallel parity -----------------------------------------------------

    #[test]
    #[cfg(feature = "parallel")]
    fn test_sparse_coo_parallel_matches_serial() {
        let shape = [5, 7, 9];
        let (_, coo) = build_sparse_3d_f64(shape, 40, 2020);
        let u1 = random_matrix_f64(5, 4, 1);
        let u2 = random_matrix_f64(7, 4, 2);
        let u3 = random_matrix_f64(9, 4, 3);
        let factors = [u1.view(), u2.view(), u3.view()];

        for mode in 0..3 {
            let serial = mttkrp_sparse_coo(&coo, &factors, mode).expect("serial sparse mttkrp");
            let parallel =
                mttkrp_sparse_coo_parallel(&coo, &factors, mode).expect("parallel sparse mttkrp");
            assert_eq!(serial.shape(), parallel.shape());
            assert!(
                max_abs_diff(&serial, &parallel) < 1e-12,
                "serial vs parallel differ at mode {}",
                mode
            );
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_sparse_coo_parallel_matches_dense() {
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_3d_f64(shape, 20, 3131);
        let u1 = random_matrix_f64(4, 8, 501);
        let u2 = random_matrix_f64(5, 8, 502);
        let u3 = random_matrix_f64(6, 8, 503);
        let factors = [u1.view(), u2.view(), u3.view()];

        let dense_res = mttkrp(&dense.view(), &factors, 2).expect("dense");
        let parallel_res = mttkrp_sparse_coo_parallel(&coo, &factors, 2).expect("parallel sparse");
        assert!(max_abs_diff(&parallel_res, &dense_res) < 1e-10);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_sparse_coo_parallel_empty() {
        let coo = CooTensor::<f64>::zeros(vec![2, 3, 4]).expect("zeros");
        let u1 = random_matrix_f64(2, 3, 1);
        let u2 = random_matrix_f64(3, 3, 2);
        let u3 = random_matrix_f64(4, 3, 3);
        let m = mttkrp_sparse_coo_parallel(&coo, &[u1.view(), u2.view(), u3.view()], 0)
            .expect("parallel");
        assert_eq!(m.shape(), &[2, 3]);
        for &v in m.iter() {
            assert_eq!(v, 0.0);
        }
    }
}
