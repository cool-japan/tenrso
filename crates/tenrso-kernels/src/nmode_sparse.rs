//! Sparse N-mode product kernel
//!
//! Computes the N-mode product `X ×_n M` where `X` is a sparse COO tensor
//! and `M` is a dense matrix. This is a **mixed sparse/dense** operation:
//! the input tensor is sparse, the projection matrix is dense, and the output
//! is dense (every entry of the output may be non-zero even when the input is
//! sparse, because each non-zero contributes to `M.nrows()` output cells).
//!
//! # Mathematical definition
//!
//! For tensor `X` with shape `(I₀, …, Iₙ₋₁)` and matrix `M ∈ ℝ^{J × Iₘ}`,
//! the mode-m product `Y = X ×_m M` has shape
//! `(I₀, …, Iₘ₋₁, J, Iₘ₊₁, …, Iₙ₋₁)` and entries
//!
//! ```text
//! Y[i₀, …, j, …, iₙ₋₁] = ∑_{iₘ=0}^{Iₘ-1} X[i₀, …, iₘ, …, iₙ₋₁] · M[j, iₘ]
//! ```
//!
//! # Algorithm — scatter (no unfold, no GEMM)
//!
//! For each nonzero `(idx, val)` in the COO tensor and each output row
//! `j ∈ 0..M.nrows()`:
//!
//! ```text
//! out[idx with mode-m replaced by j] += M[j, idx[m]] * val
//! ```
//!
//! Time complexity: `O(nnz × M.nrows())`
//! Space complexity: `O(∏ output_shape)`
//!
//! This satisfies **both** open TODOs in `tenrso-kernels/TODO.md`:
//! - "Sparse n-mode product"
//! - "Mixed sparse/dense operations"
//!
//! # SciRS2 integration
//!
//! All array operations use [`scirs2_core::ndarray_ext`].
//! Direct use of `ndarray`, `rand`, `rayon`, or `num-traits` is forbidden per
//! `SCIRS2_INTEGRATION_POLICY.md`.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array, ArrayView2, IxDyn};
use scirs2_core::numeric::{Num, One, Zero};
use tenrso_sparse::coo::CooTensor;

// =============================================================================
// Validation helper
// =============================================================================

/// Validate inputs for sparse n-mode product and return the output shape.
///
/// # Errors
///
/// - `mode >= tensor_shape.len()`
/// - `matrix.ncols() != tensor_shape[mode]`
fn validate_nmode_sparse_inputs<T>(
    tensor_shape: &[usize],
    matrix: &ArrayView2<T>,
    mode: usize,
) -> Result<Vec<usize>> {
    let rank = tensor_shape.len();

    if mode >= rank {
        anyhow::bail!(
            "Mode {} out of bounds for tensor with rank {}",
            mode,
            rank
        );
    }

    let matrix_cols = matrix.shape()[1];
    let mode_size = tensor_shape[mode];

    if matrix_cols != mode_size {
        anyhow::bail!(
            "Matrix columns ({}) must match tensor mode-{} size ({})",
            matrix_cols,
            mode,
            mode_size
        );
    }

    let matrix_rows = matrix.shape()[0];
    let mut new_shape = tensor_shape.to_vec();
    new_shape[mode] = matrix_rows;
    Ok(new_shape)
}

// =============================================================================
// Serial implementation
// =============================================================================

/// Compute the sparse N-mode product `X ×_n M` where `X` is a sparse COO
/// tensor and `M` is a dense matrix.
///
/// Multiplies tensor `X` along mode `mode` by matrix `M`, producing a **dense**
/// output tensor. The output shape equals `X.shape()` with the `mode`-th
/// dimension replaced by `M.nrows()`.
///
/// This is both the "sparse n-mode product" and a "mixed sparse/dense"
/// operation. After this product the result is dense; use the dense
/// [`crate::nmode::nmode_products_seq`] or the Tucker operator for subsequent
/// mode products.
///
/// # Algorithm
///
/// Scatter — no unfold, no GEMM. For each nonzero `(idx, val)` and each
/// output row `j ∈ 0..M.nrows()`:
///
/// ```text
/// out[idx with mode replaced by j] += M[j, idx[mode]] * val
/// ```
///
/// # Arguments
///
/// * `tensor` – N-dimensional sparse tensor in COO format
/// * `matrix` – Dense matrix with shape `(J, tensor.shape()[mode])`
/// * `mode`   – The mode along which to perform the product (0-indexed)
///
/// # Returns
///
/// Dense tensor with shape `(I₀, …, Iₘ₋₁, J, Iₘ₊₁, …, Iₙ₋₁)`.
///
/// # Errors
///
/// Returns an error if `mode >= tensor.rank()` or
/// `matrix.ncols() != tensor.shape()[mode]`.
///
/// # Complexity
///
/// - Time: `O(nnz × M.nrows())`
/// - Space: `O(∏ output_shape)`
///
/// # Duplicate indices
///
/// If `tensor` contains multiple entries at the same coordinate their
/// contributions are accumulated additively, matching
/// [`tenrso_sparse::coo::CooTensor::deduplicate`] semantics.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::nmode_product_sparse_coo;
/// use tenrso_sparse::coo::CooTensor;
///
/// // 2×3×4 sparse tensor with 2 nonzeros
/// let indices = vec![vec![0usize, 1, 2], vec![1, 2, 3]];
/// let values = vec![1.0_f64, 2.0];
/// let coo = CooTensor::new(indices, values, vec![2, 3, 4]).unwrap();
///
/// // 5×3 matrix — mode 1 has size 3
/// let m = Array2::<f64>::from_elem((5, 3), 1.0);
/// let result = nmode_product_sparse_coo(&coo, &m.view(), 1).unwrap();
/// assert_eq!(result.shape(), &[2, 5, 4]);
/// ```
pub fn nmode_product_sparse_coo<T>(
    tensor: &CooTensor<T>,
    matrix: &ArrayView2<T>,
    mode: usize,
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num + One + Zero,
{
    let new_shape = validate_nmode_sparse_inputs(tensor.shape(), matrix, mode)?;
    let mut output = Array::<T, IxDyn>::zeros(IxDyn(&new_shape));

    let nnz = tensor.nnz();
    if nnz == 0 {
        return Ok(output);
    }

    let matrix_rows = matrix.shape()[0];
    let indices = tensor.indices();
    let values = tensor.values();
    let tensor_rank = tensor.rank();

    // Reuse a single scratch buffer per nonzero to avoid per-j allocation.
    let mut out_idx = vec![0usize; tensor_rank];

    for (idx, val) in indices.iter().zip(values.iter()) {
        let m_col = idx[mode];
        // Copy source coordinates; we will patch element [mode] below.
        out_idx.copy_from_slice(idx);

        for j in 0..matrix_rows {
            out_idx[mode] = j;
            let contrib = matrix[[j, m_col]].clone() * val.clone();
            let prev = output[out_idx.as_slice()].clone();
            output[out_idx.as_slice()] = prev + contrib;
        }
    }

    Ok(output)
}

// =============================================================================
// Parallel implementation
// =============================================================================

/// Parallel version of [`nmode_product_sparse_coo`] using Rayon (via
/// `scirs2_core::parallel_ops`).
///
/// Buckets nonzeros by *fiber key* — the multi-index with the `mode`-th
/// coordinate removed. Nonzeros in the same fiber write to the same output
/// *column* (a contiguous slice along mode `mode`), so bucket processing is
/// race-free: each bucket independently accumulates a set of disjoint output
/// entries.
///
/// After parallel accumulation, contributions are serialised back into the
/// output array. For large `nnz` this typically scales to roughly core-count
/// speedup.
///
/// Falls back to serial handling for empty tensors.
///
/// # Correctness
///
/// Produces numerically equivalent output to [`nmode_product_sparse_coo`]
/// (up to floating-point accumulation order within a fiber bucket, which is
/// deterministic here because bucket elements are processed in the order they
/// appear in the COO tensor).
///
/// # Complexity
///
/// - Time: `O(nnz × M.nrows() / P)` for `P` threads (excluding bucketing)
/// - Space: `O(nnz + ∏ output_shape)`
///
/// # Errors
///
/// Same as [`nmode_product_sparse_coo`].
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::nmode_product_sparse_coo_parallel;
/// use tenrso_sparse::coo::CooTensor;
///
/// let indices = vec![vec![0usize, 0, 0], vec![1, 1, 1]];
/// let values = vec![2.0_f64, 3.0];
/// let coo = CooTensor::new(indices, values, vec![2, 2, 2]).unwrap();
///
/// let m = Array2::<f64>::from_elem((3, 2), 1.0);
/// let result = nmode_product_sparse_coo_parallel(&coo, &m.view(), 0).unwrap();
/// assert_eq!(result.shape(), &[3, 2, 2]);
/// ```
#[cfg(feature = "parallel")]
pub fn nmode_product_sparse_coo_parallel<T>(
    tensor: &CooTensor<T>,
    matrix: &ArrayView2<T>,
    mode: usize,
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num + One + Zero + Send + Sync,
{
    // Parallel imports live inside the function body so they do not trigger
    // unused-import warnings when this function is absent (feature off).
    use scirs2_core::parallel_ops::*;
    use std::collections::HashMap;

    let new_shape = validate_nmode_sparse_inputs(tensor.shape(), matrix, mode)?;
    let mut output = Array::<T, IxDyn>::zeros(IxDyn(&new_shape));

    let nnz = tensor.nnz();
    if nnz == 0 {
        return Ok(output);
    }

    let matrix_rows = matrix.shape()[0];
    let indices = tensor.indices();
    let values = tensor.values();
    let tensor_rank = tensor.rank();

    // --- Bucket nonzeros by fiber key ----------------------------------------
    //
    // fiber_key = idx with element at position `mode` removed.
    // All nonzeros in the same fiber share the same mode-`mode` slice of the
    // output tensor, so their contributions can be computed independently.
    //
    // HashMap<fiber_key, Vec<(full_idx_ref, val_ref)>>
    let mut buckets: HashMap<Vec<usize>, Vec<(&Vec<usize>, &T)>> = HashMap::new();
    for (idx, val) in indices.iter().zip(values.iter()) {
        // Build fiber key by dropping element at position `mode`.
        let fiber_key: Vec<usize> = idx
            .iter()
            .enumerate()
            .filter_map(|(d, &c)| if d != mode { Some(c) } else { None })
            .collect();
        buckets.entry(fiber_key).or_default().push((idx, val));
    }

    // --- Process buckets in parallel -----------------------------------------
    //
    // Each bucket yields a list of (output_idx, scalar_contribution) pairs.
    // Output indices across different buckets are disjoint by construction
    // (they differ in at least one non-mode coordinate, or the same bucket
    // handles all j values for the same fiber).
    //
    // Within a bucket the output indices ARE shared across different j values,
    // but the bucket is processed by a single thread, so there are no races.
    let contributions: Vec<Vec<(Vec<usize>, T)>> = buckets
        .par_iter()
        .map(|(_fiber_key, bucket)| {
            let mut local: Vec<(Vec<usize>, T)> = Vec::with_capacity(bucket.len() * matrix_rows);
            // Reuse a single scratch buffer for output index construction.
            let mut out_idx = vec![0usize; tensor_rank];

            for (full_idx, val) in bucket.iter() {
                let m_col = full_idx[mode];
                out_idx.copy_from_slice(full_idx);

                for j in 0..matrix_rows {
                    out_idx[mode] = j;
                    let contrib = matrix[[j, m_col]].clone() * (*val).clone();
                    local.push((out_idx.clone(), contrib));
                }
            }
            local
        })
        .collect();

    // --- Scatter contributions into output (serial; no races) ----------------
    //
    // Different buckets CAN share the same output cell when two nonzeros lie
    // in the same fiber (same fiber_key → same bucket → already handled above).
    // However, contributions from DIFFERENT buckets that map to the SAME output
    // cell are possible when two distinct fiber_keys both map to the same output
    // slice — this can happen if two nonzeros differ only in their mode-`mode`
    // coordinate (they get different fiber_keys but produce overlapping j-output
    // cells). We therefore scatter serially here to correctly accumulate.
    for bucket_contributions in contributions {
        for (out_idx, contrib) in bucket_contributions {
            let prev = output[out_idx.as_slice()].clone();
            output[out_idx.as_slice()] = prev + contrib;
        }
    }

    Ok(output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nmode::nmode_product;
    use scirs2_core::ndarray_ext::{Array, Array2, IxDyn};
    use scirs2_core::random::{SeedableRng, StdRng};
    use tenrso_sparse::coo::CooTensor;

    // -------------------------------------------------------------------------
    // Local RNG helpers (mirror the pattern in mttkrp_sparse.rs tests)
    // -------------------------------------------------------------------------

    /// Fill a flat Vec<f64> with uniform [0, 1) values from a seeded RNG.
    fn random_f64_vec(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random_f64()).collect()
    }

    /// Build a random dense matrix.
    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let data: Vec<f64> = random_f64_vec(seed, rows * cols)
            .into_iter()
            .map(|v| v * 2.0 - 1.0)
            .collect();
        Array2::<f64>::from_shape_vec((rows, cols), data)
            .expect("shape product == data len")
    }

    // -------------------------------------------------------------------------
    // COO / dense pair builders
    // -------------------------------------------------------------------------

    /// Build a COO tensor with `nnz` distinct random nonzeros and the
    /// corresponding dense array (for oracle comparison via `nmode_product`).
    fn build_sparse_nd(
        shape: &[usize],
        nnz: usize,
        seed: u64,
    ) -> (Array<f64, IxDyn>, CooTensor<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dense = Array::<f64, IxDyn>::zeros(IxDyn(shape));
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(nnz);
        let mut values: Vec<f64> = Vec::with_capacity(nnz);
        let mut used = std::collections::HashSet::new();

        let ndim = shape.len();
        let mut count = 0usize;
        let mut safety = 0usize;
        let max_attempts = nnz * 200;

        while count < nnz && safety < max_attempts {
            safety += 1;
            let mut idx = vec![0usize; ndim];
            for (d, &s) in shape.iter().enumerate() {
                idx[d] = rng.gen_range(0..s);
            }
            if !used.insert(idx.clone()) {
                continue;
            }
            let v: f64 = rng.random_f64() * 2.0 - 1.0;
            dense[&idx[..]] = v;
            indices.push(idx);
            values.push(v);
            count += 1;
        }

        let coo = CooTensor::new(indices, values, shape.to_vec()).expect("valid COO");
        (dense, coo)
    }

    // -------------------------------------------------------------------------
    // max_diff: element-wise maximum absolute difference between two
    // IxDyn arrays with the same shape.
    // -------------------------------------------------------------------------

    fn max_diff(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> f64 {
        assert_eq!(a.shape(), b.shape(), "shape mismatch in max_diff");
        let mut m = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > m {
                m = d;
            }
        }
        m
    }

    // =========================================================================
    // Tests 1-4: correctness vs dense oracle for various modes / ranks
    // =========================================================================

    #[test]
    fn test_nmode_sparse_coo_mode0() {
        // 4x5x6 tensor, mode 0, 3x4 matrix -> output [3, 5, 6]
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_nd(&shape, 25, 42);
        let matrix = random_matrix(3, 4, 1);

        let sparse_res =
            nmode_product_sparse_coo(&coo, &matrix.view(), 0).expect("sparse nmode mode0");
        let dense_res =
            nmode_product(&dense.view(), &matrix.view(), 0).expect("dense nmode mode0");

        assert_eq!(sparse_res.shape(), &[3, 5, 6]);
        assert_eq!(sparse_res.shape(), dense_res.shape());
        assert!(
            max_diff(&sparse_res, &dense_res) < 1e-10,
            "mode0 sparse vs dense diff too large: {}",
            max_diff(&sparse_res, &dense_res)
        );
    }

    #[test]
    fn test_nmode_sparse_coo_mode1() {
        // 4x5x6 tensor, mode 1, 7x5 matrix -> output [4, 7, 6]
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_nd(&shape, 30, 43);
        let matrix = random_matrix(7, 5, 2);

        let sparse_res =
            nmode_product_sparse_coo(&coo, &matrix.view(), 1).expect("sparse nmode mode1");
        let dense_res =
            nmode_product(&dense.view(), &matrix.view(), 1).expect("dense nmode mode1");

        assert_eq!(sparse_res.shape(), &[4, 7, 6]);
        assert_eq!(sparse_res.shape(), dense_res.shape());
        assert!(
            max_diff(&sparse_res, &dense_res) < 1e-10,
            "mode1 sparse vs dense diff: {}",
            max_diff(&sparse_res, &dense_res)
        );
    }

    #[test]
    fn test_nmode_sparse_coo_mode2() {
        // 4x5x6 tensor, mode 2, 3x6 matrix -> output [4, 5, 3]
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_nd(&shape, 35, 44);
        let matrix = random_matrix(3, 6, 3);

        let sparse_res =
            nmode_product_sparse_coo(&coo, &matrix.view(), 2).expect("sparse nmode mode2");
        let dense_res =
            nmode_product(&dense.view(), &matrix.view(), 2).expect("dense nmode mode2");

        assert_eq!(sparse_res.shape(), &[4, 5, 3]);
        assert_eq!(sparse_res.shape(), dense_res.shape());
        assert!(
            max_diff(&sparse_res, &dense_res) < 1e-10,
            "mode2 sparse vs dense diff: {}",
            max_diff(&sparse_res, &dense_res)
        );
    }

    #[test]
    fn test_nmode_sparse_coo_4d_mode2() {
        // 3x4x5x6 tensor, mode 2, 3x5 matrix -> output [3, 4, 3, 6]
        let shape = [3, 4, 5, 6];
        let (dense, coo) = build_sparse_nd(&shape, 40, 45);
        let matrix = random_matrix(3, 5, 4);

        let sparse_res =
            nmode_product_sparse_coo(&coo, &matrix.view(), 2).expect("sparse nmode 4d mode2");
        let dense_res =
            nmode_product(&dense.view(), &matrix.view(), 2).expect("dense nmode 4d mode2");

        assert_eq!(sparse_res.shape(), &[3, 4, 3, 6]);
        assert_eq!(sparse_res.shape(), dense_res.shape());
        assert!(
            max_diff(&sparse_res, &dense_res) < 1e-10,
            "4d mode2 sparse vs dense diff: {}",
            max_diff(&sparse_res, &dense_res)
        );
    }

    // =========================================================================
    // Tests 5-6: shape-change and shape-preserving
    // =========================================================================

    #[test]
    fn test_nmode_sparse_dim_change() {
        // Verify output shape changes correctly when nrows != ncols.
        let shape = [2, 8, 4];
        let (_, coo) = build_sparse_nd(&shape, 10, 100);
        // 6x8 matrix: mode-1 dim 8 -> 6
        let matrix = random_matrix(6, 8, 10);
        let result = nmode_product_sparse_coo(&coo, &matrix.view(), 1)
            .expect("dim change");
        assert_eq!(result.shape(), &[2, 6, 4]);
    }

    #[test]
    fn test_nmode_sparse_dim_preserving() {
        // Square matrix (same nrows as ncols) preserves the mode dimension.
        let shape = [3, 5, 4];
        let (dense, coo) = build_sparse_nd(&shape, 20, 101);
        // 5x5 identity -- result must equal the original tensor (in dense form).
        let eye = Array2::<f64>::eye(5);
        let sparse_res = nmode_product_sparse_coo(&coo, &eye.view(), 1)
            .expect("dim preserving");
        let dense_res = nmode_product(&dense.view(), &eye.view(), 1)
            .expect("dense preserving");

        assert_eq!(sparse_res.shape(), &[3, 5, 4]);
        assert!(
            max_diff(&sparse_res, &dense_res) < 1e-12,
            "identity product should match dense: {}",
            max_diff(&sparse_res, &dense_res)
        );
    }

    // =========================================================================
    // Tests 7-9: special-case inputs
    // =========================================================================

    #[test]
    fn test_nmode_sparse_all_zeros() {
        // Empty COO (nnz = 0) -> output is all zeros.
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).expect("zeros COO");
        let matrix = random_matrix(2, 4, 7);
        let result =
            nmode_product_sparse_coo(&coo, &matrix.view(), 1).expect("empty tensor");
        assert_eq!(result.shape(), &[3, 2, 5]);
        for &v in result.iter() {
            assert_eq!(v, 0.0, "expected zero output for empty tensor");
        }
    }

    #[test]
    fn test_nmode_sparse_single_nonzero() {
        // Single nonzero at position (1, 2, 3) with value 5.0 in a 2x4x5 tensor.
        // Mode-1 product with a 3x4 matrix should scatter M[:,2]*5 into
        // output[:,0..3,3] at position (1, j, 3) for j in 0..3.
        let indices = vec![vec![1usize, 2, 3]];
        let values = vec![5.0_f64];
        let shape = vec![2, 4, 5];
        let coo = CooTensor::new(indices, values, shape).expect("single nonzero COO");

        // 3x4 matrix
        let mat = random_matrix(3, 4, 99);
        let result = nmode_product_sparse_coo(&coo, &mat.view(), 1).expect("single nz");
        assert_eq!(result.shape(), &[2, 3, 5]);

        // Verify only slice (1, :, 3) is non-zero.
        for i in 0..2usize {
            for j in 0..3usize {
                for k in 0..5usize {
                    let expected = if i == 1 && k == 3 {
                        mat[[j, 2]] * 5.0
                    } else {
                        0.0
                    };
                    assert!(
                        (result[&[i, j, k][..]] - expected).abs() < 1e-14,
                        "mismatch at [{i},{j},{k}]: got {}, expected {}",
                        result[&[i, j, k][..]],
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_nmode_sparse_duplicate_indices() {
        // Two COO entries at the same coordinate should sum their contributions.
        // Build COO with duplicate (0,1,2) -> values 3.0 and 4.0; and (1,0,1).
        let indices = vec![vec![0usize, 1, 2], vec![0, 1, 2], vec![1, 0, 1]];
        let values = vec![3.0_f64, 4.0, 7.0];
        let shape = vec![2, 3, 4];

        let coo = CooTensor::new(indices.clone(), values.clone(), shape.clone())
            .expect("COO with duplicates");

        // Build matching dense (summing duplicates).
        let mut dense = Array::<f64, IxDyn>::zeros(IxDyn(&shape));
        for (idx, &v) in indices.iter().zip(values.iter()) {
            let prev = dense[&idx[..]];
            dense[&idx[..]] = prev + v;
        }

        let matrix = random_matrix(5, 3, 200);

        let sparse_res = nmode_product_sparse_coo(&coo, &matrix.view(), 1)
            .expect("dup indices nmode");
        let dense_res =
            nmode_product(&dense.view(), &matrix.view(), 1).expect("dense dup");

        assert_eq!(sparse_res.shape(), &[2, 5, 4]);
        assert!(
            max_diff(&sparse_res, &dense_res) < 1e-10,
            "duplicate-index test: max_diff = {}",
            max_diff(&sparse_res, &dense_res)
        );
    }

    // =========================================================================
    // Tests 10-11: error-path tests
    // =========================================================================

    #[test]
    fn test_nmode_sparse_error_invalid_mode() {
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).expect("zeros COO");
        let matrix = Array2::<f64>::from_elem((2, 3), 1.0);
        // mode = 5 is out of bounds for a rank-3 tensor.
        let err = nmode_product_sparse_coo(&coo, &matrix.view(), 5);
        assert!(err.is_err(), "expected error for mode >= rank");
        let msg = format!("{}", err.expect_err("error"));
        assert!(
            msg.contains("mode") || msg.contains("Mode"),
            "error message should mention 'mode': got: {msg}"
        );
    }

    #[test]
    fn test_nmode_sparse_error_matrix_cols() {
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).expect("zeros COO");
        // Mode 1 has size 4; matrix has 7 columns -- mismatch.
        let matrix = Array2::<f64>::from_elem((2, 7), 1.0);
        let err = nmode_product_sparse_coo(&coo, &matrix.view(), 1);
        assert!(err.is_err(), "expected error for column mismatch");
        let msg = format!("{}", err.expect_err("error"));
        assert!(
            msg.to_lowercase().contains("col") || msg.to_lowercase().contains("match"),
            "error message should mention columns/match: got: {msg}"
        );
    }

    // =========================================================================
    // Tests 12-14: parallel parity
    // =========================================================================

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nmode_sparse_parallel_vs_serial() {
        let shape = [5, 7, 9];
        let (_, coo) = build_sparse_nd(&shape, 50, 2020);
        let matrix = random_matrix(4, 7, 300);

        let serial = nmode_product_sparse_coo(&coo, &matrix.view(), 1)
            .expect("serial nmode");
        let parallel = nmode_product_sparse_coo_parallel(&coo, &matrix.view(), 1)
            .expect("parallel nmode");

        assert_eq!(serial.shape(), parallel.shape());
        assert!(
            max_diff(&serial, &parallel) < 1e-12,
            "parallel vs serial diverged: {}",
            max_diff(&serial, &parallel)
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nmode_sparse_parallel_vs_dense_oracle() {
        let shape = [4, 5, 6];
        let (dense, coo) = build_sparse_nd(&shape, 40, 3131);
        let matrix = random_matrix(8, 6, 400);

        let parallel_res = nmode_product_sparse_coo_parallel(&coo, &matrix.view(), 2)
            .expect("parallel nmode");
        let dense_res =
            nmode_product(&dense.view(), &matrix.view(), 2).expect("dense nmode");

        assert_eq!(parallel_res.shape(), &[4, 5, 8]);
        assert!(
            max_diff(&parallel_res, &dense_res) < 1e-10,
            "parallel vs dense oracle: {}",
            max_diff(&parallel_res, &dense_res)
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nmode_sparse_parallel_empty() {
        // Empty tensor: parallel path must produce all-zero output.
        let coo = CooTensor::<f64>::zeros(vec![3, 4, 5]).expect("zeros COO");
        let matrix = Array2::<f64>::from_elem((2, 4), 1.0);
        let result = nmode_product_sparse_coo_parallel(&coo, &matrix.view(), 1)
            .expect("parallel empty");
        assert_eq!(result.shape(), &[3, 2, 5]);
        for &v in result.iter() {
            assert_eq!(v, 0.0, "expected zero from parallel empty tensor");
        }
    }
}
