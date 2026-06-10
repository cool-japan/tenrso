//! Parallel and fused Tucker mode application extensions.
//!
//! This module contains the parallel N-mode product variants and fused Tucker
//! reconstruction functions split from `nmode.rs` to keep files under 2000 lines.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
#[cfg(feature = "parallel")]
use scirs2_core::ndarray_ext::Axis;
use scirs2_core::ndarray_ext::{Array, Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::{Num, One, Zero};

use crate::nmode::{fold_matrix, unfold_tensor};

/// Parallel N-mode product using Rayon row-parallelism
///
/// Computes Y = X ×ₖ M identically to [`nmode_product`], but parallelises the
/// matrix-matrix multiplication over output rows.  Each row of `M` is assigned
/// to its own Rayon task, giving near-linear speedup when
/// `matrix.nrows() ≥ available_threads`.
///
/// # Bounds
///
/// `T` must be [`Send`] + [`Sync`] in addition to the serial requirements.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::nmode_product_parallel;
///
/// let tensor = Array::from_shape_vec(
///     vec![4, 5, 6],
///     (0..120).map(|x| x as f64).collect()
/// ).unwrap();
/// let matrix = Array::from_shape_vec((3, 5), vec![1.0; 15]).unwrap();
///
/// let result = nmode_product_parallel(&tensor.view(), &matrix.view(), 1).unwrap();
/// assert_eq!(result.shape(), &[4, 3, 6]);
/// ```
#[cfg(feature = "parallel")]
pub fn nmode_product_parallel<T>(
    tensor: &ArrayView<T, IxDyn>,
    matrix: &ArrayView2<T>,
    mode: usize,
) -> Result<Array<T, IxDyn>>
where
    T: Copy + Num + One + Zero + 'static + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let tensor_shape = tensor.shape();
    let rank = tensor_shape.len();

    if mode >= rank {
        anyhow::bail!("Mode {} out of bounds for tensor with rank {}", mode, rank);
    }

    let mode_size = tensor_shape[mode];
    let (matrix_rows, matrix_cols) = (matrix.shape()[0], matrix.shape()[1]);

    if matrix_cols != mode_size {
        anyhow::bail!(
            "Matrix columns ({}) must match tensor mode-{} size ({})",
            matrix_cols,
            mode,
            mode_size
        );
    }

    // Unfold tensor along mode k: shape [mode_size, rest_size] (row-major)
    let unfolded = unfold_tensor(tensor, mode)?;
    let rest_size = unfolded.shape()[1];

    // Parallel: compute each output row independently.
    // For row j: out[j, c] = Σ_k M[j, k] * unfolded[k, c]
    // We accumulate over k (iterating unfolded row-k, which is C-contiguous).
    let mut result_unfolded = Array2::<T>::zeros((matrix_rows, rest_size));

    result_unfolded
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut out_row)| {
            let m_row = matrix.row(j);
            out_row.fill(T::zero());
            for ki in 0..mode_size {
                let scale = m_row[ki];
                let unf_row = unfolded.row(ki);
                for c in 0..rest_size {
                    out_row[c] = out_row[c] + scale * unf_row[c];
                }
            }
        });

    let mut new_shape = tensor_shape.to_vec();
    new_shape[mode] = matrix_rows;

    fold_matrix(&result_unfolded.view(), &new_shape, mode)
}

/// Parallel multi-mode products
///
/// Applies each `(matrix, mode)` pair in order using [`nmode_product_parallel`].
/// Each individual mode application is parallelised; the applications are still
/// sequential because each step's output is the next step's input.
///
/// Prefer this over [`nmode_products_seq`] when matrices are large.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::nmode_products_parallel;
///
/// let tensor = Array::from_shape_vec(
///     vec![4, 5, 6],
///     (0..120).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let m0 = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];  // 2×4
/// let m2 = array![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
///                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]];  // 3×6
///
/// let result = nmode_products_parallel(
///     &tensor.view(),
///     &[(&m0.view(), 0), (&m2.view(), 2)]
/// ).unwrap();
/// assert_eq!(result.shape(), &[2, 5, 3]);
/// ```
#[cfg(feature = "parallel")]
pub fn nmode_products_parallel<T>(
    tensor: &ArrayView<T, IxDyn>,
    matrices: &[(&ArrayView2<T>, usize)],
) -> Result<Array<T, IxDyn>>
where
    T: Copy + Num + One + Zero + 'static + Send + Sync,
{
    let mut result = tensor.to_owned();
    for (matrix, mode) in matrices {
        result = nmode_product_parallel(&result.view(), matrix, *mode)?;
    }
    Ok(result)
}

/// Parallel Tucker operator
///
/// Like [`tucker_operator`] (applies each mode in cost-minimising order), but
/// delegates each mode application to [`nmode_product_parallel`].
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use std::collections::HashMap;
/// use tenrso_kernels::tucker_operator_parallel;
///
/// let tensor = Array::from_shape_vec(
///     vec![4, 5, 6],
///     (0..120).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let m0 = Array::from_shape_vec((2, 4), vec![1.0; 8]).unwrap();
/// let m1 = Array::from_shape_vec((3, 5), vec![1.0; 15]).unwrap();
/// let m2 = Array::from_shape_vec((4, 6), vec![1.0; 24]).unwrap();
///
/// let mut factors = HashMap::new();
/// factors.insert(0, m0.view());
/// factors.insert(1, m1.view());
/// factors.insert(2, m2.view());
///
/// let result = tucker_operator_parallel(&tensor.view(), &factors).unwrap();
/// assert_eq!(result.shape(), &[2, 3, 4]);
/// ```
#[cfg(feature = "parallel")]
pub fn tucker_operator_parallel<T>(
    tensor: &ArrayView<T, IxDyn>,
    factor_matrices: &std::collections::HashMap<usize, ArrayView2<T>>,
) -> Result<Array<T, IxDyn>>
where
    T: Copy + Num + One + Zero + 'static + Send + Sync,
{
    if factor_matrices.is_empty() {
        return Ok(tensor.to_owned());
    }

    let rank = tensor.shape().len();

    for &mode in factor_matrices.keys() {
        if mode >= rank {
            anyhow::bail!("Mode {} out of bounds for tensor with rank {}", mode, rank);
        }
    }

    // Apply modes in ascending reduction-ratio order (smallest output/input first)
    let mut modes: Vec<usize> = factor_matrices.keys().copied().collect();
    modes.sort_by(|&a, &b| {
        let ratio_a = factor_matrices[&a].shape()[0] as f64 / tensor.shape()[a] as f64;
        let ratio_b = factor_matrices[&b].shape()[0] as f64 / tensor.shape()[b] as f64;
        ratio_a
            .partial_cmp(&ratio_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = tensor.to_owned();
    for original_mode in modes {
        let matrix = factor_matrices[&original_mode];
        result = nmode_product_parallel(&result.view(), &matrix, original_mode)?;
    }

    Ok(result)
}

// ─── Fused Tucker reconstruction ───────────────────────────────────────────

/// Row-major strides for a given shape (C-order)
#[inline]
pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for k in (0..n.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// Fused Tucker reconstruction without intermediate tensor allocations
///
/// Computes `Y = G ×₁ U₁ ×₂ U₂ … ×_N U_N` directly without materialising the
/// N-1 intermediate tensors that sequential application would produce.
///
/// # Algorithm
///
/// For each output position `(i₀, …, i_{N-1})`:
///
/// ```text
/// Y[i₀,…,i_{N-1}] = Σ_{r₀,…,r_{N-1}} G[r₀,…,r_{N-1}] · Π_k U_k[i_k, r_k]
/// ```
///
/// The core is traversed in row-major (C) order with an amortised partial-product
/// prefix that avoids redundant factor multiplications.
///
/// # When to prefer this over `tucker_reconstruct`
///
/// | Scenario | Recommend |
/// |---|---|
/// | Small core (R ≤ ~16), large output | **fused** — core fits in L1 cache |
/// | Large core or I/R ≫ 1 | `tucker_reconstruct` — fewer FLOPS |
/// | Memory-constrained environment | **fused** — O(1) intermediate allocations |
///
/// The fused version is O(∏Iₙ · ∏Rₙ) FLOPS vs the sequential O(∏Iₙ · Rmax)
/// for Tucker reconstruction, so it is asymptotically more expensive for large R.
///
/// # Errors
///
/// Returns an error if:
/// - The number of factor matrices does not equal the core's order
/// - Factor matrix columns don't match the corresponding core mode size
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::tucker_reconstruct_fused;
///
/// let core = Array::from_shape_vec(
///     vec![2, 2, 2],
///     vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
/// ).unwrap();
/// let u1 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];  // 3×2
/// let u2 = array![[1.0, 0.0], [0.0, 1.0]];               // 2×2
/// let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];   // 3×2
///
/// let tensor = tucker_reconstruct_fused(
///     &core.view(), &[u1.view(), u2.view(), u3.view()]
/// ).unwrap();
/// assert_eq!(tensor.shape(), &[3, 2, 3]);
/// ```
pub fn tucker_reconstruct_fused<T>(
    core: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
) -> Result<Array<T, IxDyn>>
where
    T: Copy + Num + One + Zero + 'static,
{
    let n = factors.len();
    let core_ranks: Vec<usize> = core.shape().to_vec();

    if core_ranks.len() != n {
        anyhow::bail!(
            "Number of factor matrices ({}) must match core order ({})",
            n,
            core_ranks.len()
        );
    }
    for (k, factor) in factors.iter().enumerate() {
        if factor.shape()[1] != core_ranks[k] {
            anyhow::bail!(
                "Factor {} has {} columns but core mode {} has size {}",
                k,
                factor.shape()[1],
                k,
                core_ranks[k]
            );
        }
    }

    let out_shape: Vec<usize> = factors.iter().map(|f| f.shape()[0]).collect();
    let total_out: usize = out_shape.iter().product();
    let total_core: usize = core_ranks.iter().product();

    if total_out == 0 || total_core == 0 {
        return Ok(Array::zeros(IxDyn(&out_shape)));
    }

    // C-order strides for multi-index iteration
    let out_strides = row_major_strides(&out_shape);

    // Flatten core to a contiguous Vec for direct index access
    let core_std = core.as_standard_layout();
    let core_flat: Vec<T> = core_std.iter().copied().collect();

    let mut out_flat = vec![T::zero(); total_out];

    // For each output position: sum over all core positions with amortised
    // partial product updates (prefix[k] = Π_{j<k} U_j[i_j, r_j]).
    let mut out_idx = vec![0usize; n];
    let mut core_idx = vec![0usize; n];
    let mut prefix = vec![T::one(); n + 1];

    for (out_pos, out_val) in out_flat.iter_mut().enumerate() {
        // Decode current output multi-index
        let mut rem = out_pos;
        for k in 0..n {
            out_idx[k] = rem / out_strides[k];
            rem %= out_strides[k];
        }

        // Initialise core cursor and prefix products at (0,…,0)
        core_idx.fill(0);
        prefix[0] = T::one();
        for k in 0..n {
            prefix[k + 1] = prefix[k] * factors[k][[out_idx[k], 0]];
        }

        let mut acc = T::zero();

        for (core_pos, &core_val) in core_flat.iter().enumerate() {
            acc = acc + core_val * prefix[n];

            if core_pos + 1 < total_core {
                // Advance core multi-index (row-major: rightmost digit first).
                // Track which level changed so we only recompute the suffix of
                // the prefix chain from that level onward.
                let mut carry_level = n; // sentinel (exhausted — not reached within loop)
                for k in (0..n).rev() {
                    core_idx[k] += 1;
                    if core_idx[k] < core_ranks[k] {
                        carry_level = k;
                        break;
                    }
                    core_idx[k] = 0;
                }
                // Recompute prefix[carry_level+1 ..= n]
                // prefix[carry_level] is still valid (core_idx[0..carry_level] unchanged).
                if carry_level < n {
                    for j in carry_level..n {
                        prefix[j + 1] = prefix[j] * factors[j][[out_idx[j], core_idx[j]]];
                    }
                }
            }
        }

        *out_val = acc;
    }

    Ok(Array::from_shape_vec(IxDyn(&out_shape), out_flat)?)
}

/// Parallel fused Tucker reconstruction
///
/// Like [`tucker_reconstruct_fused`] but parallelises over all output positions
/// using Rayon.  Each thread independently iterates the full core for its
/// assigned output element — ideal when `∏Iₙ` is much larger than `∏Rₙ`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::tucker_reconstruct_fused_parallel;
///
/// let core = Array::from_shape_vec(
///     vec![2, 2, 2],
///     vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
/// ).unwrap();
/// let u1 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];  // 3×2
/// let u2 = array![[1.0, 0.0], [0.0, 1.0]];               // 2×2
/// let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];   // 3×2
///
/// let tensor = tucker_reconstruct_fused_parallel(
///     &core.view(), &[u1.view(), u2.view(), u3.view()]
/// ).unwrap();
/// assert_eq!(tensor.shape(), &[3, 2, 3]);
/// ```
#[cfg(feature = "parallel")]
pub fn tucker_reconstruct_fused_parallel<T>(
    core: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
) -> Result<Array<T, IxDyn>>
where
    T: Copy + Num + One + Zero + 'static + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let n = factors.len();
    let core_ranks: Vec<usize> = core.shape().to_vec();

    if core_ranks.len() != n {
        anyhow::bail!(
            "Number of factor matrices ({}) must match core order ({})",
            n,
            core_ranks.len()
        );
    }
    for (k, factor) in factors.iter().enumerate() {
        if factor.shape()[1] != core_ranks[k] {
            anyhow::bail!(
                "Factor {} has {} columns but core mode {} has size {}",
                k,
                factor.shape()[1],
                k,
                core_ranks[k]
            );
        }
    }

    let out_shape: Vec<usize> = factors.iter().map(|f| f.shape()[0]).collect();
    let total_out: usize = out_shape.iter().product();
    let total_core: usize = core_ranks.iter().product();

    if total_out == 0 || total_core == 0 {
        return Ok(Array::zeros(IxDyn(&out_shape)));
    }

    let out_strides = row_major_strides(&out_shape);

    // Flatten core once and share across threads (immutable)
    let core_std = core.as_standard_layout();
    let core_flat: Vec<T> = core_std.iter().copied().collect();

    // Each thread handles one output position independently.
    let out_flat: Vec<T> = (0..total_out)
        .into_par_iter()
        .map(|out_pos| {
            // Decode output multi-index
            let mut out_idx = vec![0usize; n];
            let mut rem = out_pos;
            for k in 0..n {
                out_idx[k] = rem / out_strides[k];
                rem %= out_strides[k];
            }

            // Per-thread core cursor and prefix products
            let mut core_idx = vec![0usize; n];
            let mut prefix = vec![T::one(); n + 1];
            for k in 0..n {
                prefix[k + 1] = prefix[k] * factors[k][[out_idx[k], 0]];
            }

            let mut acc = T::zero();
            for (core_pos, &core_val) in core_flat.iter().enumerate() {
                acc = acc + core_val * prefix[n];

                if core_pos + 1 < total_core {
                    let mut carry_level = n;
                    for k in (0..n).rev() {
                        core_idx[k] += 1;
                        if core_idx[k] < core_ranks[k] {
                            carry_level = k;
                            break;
                        }
                        core_idx[k] = 0;
                    }
                    if carry_level < n {
                        for j in carry_level..n {
                            prefix[j + 1] = prefix[j] * factors[j][[out_idx[j], core_idx[j]]];
                        }
                    }
                }
            }
            acc
        })
        .collect();

    Ok(Array::from_shape_vec(IxDyn(&out_shape), out_flat)?)
}

// ─── Tests for parallel Tucker mode application ────────────────────────────

#[cfg(all(test, feature = "parallel"))]
mod tucker_parallel_tests {
    use super::*;
    use scirs2_core::ndarray_ext::{array, Array};
    use std::collections::HashMap;

    fn max_abs_diff_dyn(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    #[test]
    fn test_nmode_product_parallel_matches_serial_mode0() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let m = Array::from_shape_vec((3, 4), (0..12).map(|x| x as f64 * 0.1).collect()).unwrap();

        let serial = crate::nmode::nmode_product(&tensor.view(), &m.view(), 0).unwrap();
        let par = nmode_product_parallel(&tensor.view(), &m.view(), 0).unwrap();

        assert_eq!(serial.shape(), par.shape());
        assert!(max_abs_diff_dyn(&serial, &par) < 1e-10);
    }

    #[test]
    fn test_nmode_product_parallel_matches_serial_mode1() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let m = Array::from_shape_vec((7, 5), (0..35).map(|x| x as f64 * 0.05).collect()).unwrap();

        let serial = crate::nmode::nmode_product(&tensor.view(), &m.view(), 1).unwrap();
        let par = nmode_product_parallel(&tensor.view(), &m.view(), 1).unwrap();

        assert_eq!(serial.shape(), par.shape());
        assert!(max_abs_diff_dyn(&serial, &par) < 1e-10);
    }

    #[test]
    fn test_nmode_product_parallel_matches_serial_mode2() {
        let tensor =
            Array::from_shape_vec(vec![3, 4, 6], (0..72).map(|x| x as f64).collect()).unwrap();
        let m = Array::from_shape_vec((2, 6), (0..12).map(|x| x as f64).collect()).unwrap();

        let serial = crate::nmode::nmode_product(&tensor.view(), &m.view(), 2).unwrap();
        let par = nmode_product_parallel(&tensor.view(), &m.view(), 2).unwrap();

        assert_eq!(serial.shape(), par.shape());
        assert!(max_abs_diff_dyn(&serial, &par) < 1e-10);
    }

    #[test]
    fn test_nmode_products_parallel_matches_seq() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let m0 = Array::from_shape_vec((3, 4), (0..12).map(|x| x as f64 * 0.1).collect()).unwrap();
        let m2 = Array::from_shape_vec((4, 6), (0..24).map(|x| x as f64 * 0.2).collect()).unwrap();

        let serial =
            crate::nmode::nmode_products_seq(&tensor.view(), &[(&m0.view(), 0), (&m2.view(), 2)])
                .unwrap();
        let par =
            nmode_products_parallel(&tensor.view(), &[(&m0.view(), 0), (&m2.view(), 2)]).unwrap();

        assert_eq!(serial.shape(), par.shape());
        assert!(max_abs_diff_dyn(&serial, &par) < 1e-10);
    }

    #[test]
    fn test_tucker_operator_parallel_matches_serial() {
        let tensor =
            Array::from_shape_vec(vec![6, 5, 4], (0..120).map(|x| x as f64).collect()).unwrap();
        let m0 = Array::from_shape_vec((3, 6), (0..18).map(|x| x as f64 * 0.1).collect()).unwrap();
        let m1 = Array::from_shape_vec((4, 5), (0..20).map(|x| x as f64 * 0.2).collect()).unwrap();
        let m2 = Array::from_shape_vec((2, 4), (0..8).map(|x| x as f64 * 0.3).collect()).unwrap();

        let mut factors = HashMap::new();
        factors.insert(0, m0.view());
        factors.insert(1, m1.view());
        factors.insert(2, m2.view());

        let serial = crate::nmode::tucker_operator(&tensor.view(), &factors).unwrap();
        let par = tucker_operator_parallel(&tensor.view(), &factors).unwrap();

        assert_eq!(serial.shape(), par.shape());
        assert!(max_abs_diff_dyn(&serial, &par) < 1e-10);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_nmode_product_parallel_invalid_mode() {
        let tensor = Array::from_shape_vec(vec![3, 4], vec![1.0; 12]).unwrap();
        let m = array![[1.0]];
        nmode_product_parallel(&tensor.view(), &m.view(), 5).unwrap();
    }

    #[test]
    #[should_panic(expected = "columns")]
    fn test_nmode_product_parallel_col_mismatch() {
        let tensor = Array::from_shape_vec(vec![3, 4], vec![1.0; 12]).unwrap();
        let m = Array::from_shape_vec((2, 5), vec![1.0; 10]).unwrap(); // 5 ≠ 4
        nmode_product_parallel(&tensor.view(), &m.view(), 1).unwrap();
    }
}

// ─── Tests for fused Tucker reconstruction ─────────────────────────────────

#[cfg(test)]
mod tucker_fused_tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    fn max_abs_diff_dyn(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    #[test]
    fn test_fused_matches_sequential_3mode() {
        let core =
            Array::from_shape_vec(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();
        let u1 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let u2 = array![[1.0, 0.0], [0.0, 1.0]];
        let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];

        let seq =
            crate::nmode::tucker_reconstruct(&core.view(), &[u1.view(), u2.view(), u3.view()])
                .unwrap();
        let fused =
            tucker_reconstruct_fused(&core.view(), &[u1.view(), u2.view(), u3.view()]).unwrap();

        assert_eq!(seq.shape(), fused.shape());
        assert!(
            max_abs_diff_dyn(&seq, &fused) < 1e-12,
            "max diff = {}",
            max_abs_diff_dyn(&seq, &fused)
        );
    }

    #[test]
    fn test_fused_matches_sequential_random_like() {
        // Use a deterministic "random-like" core and factors via LCG
        let mut state: u64 = 0xDEAD_BEEF_1234;
        let mut next_f64 = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 32) as f64 / u32::MAX as f64 - 0.5
        };

        let core_data: Vec<f64> = (0..27).map(|_| next_f64()).collect(); // 3×3×3
        let u0_data: Vec<f64> = (0..15).map(|_| next_f64()).collect(); // 5×3
        let u1_data: Vec<f64> = (0..12).map(|_| next_f64()).collect(); // 4×3
        let u2_data: Vec<f64> = (0..18).map(|_| next_f64()).collect(); // 6×3

        let core = Array::from_shape_vec(vec![3, 3, 3], core_data).unwrap();
        let u0 = Array::from_shape_vec((5, 3), u0_data).unwrap();
        let u1 = Array::from_shape_vec((4, 3), u1_data).unwrap();
        let u2 = Array::from_shape_vec((6, 3), u2_data).unwrap();

        let seq =
            crate::nmode::tucker_reconstruct(&core.view(), &[u0.view(), u1.view(), u2.view()])
                .unwrap();
        let fused =
            tucker_reconstruct_fused(&core.view(), &[u0.view(), u1.view(), u2.view()]).unwrap();

        assert_eq!(seq.shape(), fused.shape());
        assert!(
            max_abs_diff_dyn(&seq, &fused) < 1e-10,
            "max diff = {}",
            max_abs_diff_dyn(&seq, &fused)
        );
    }

    #[test]
    fn test_fused_2d_identity_core() {
        let core = Array::from_shape_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let u0 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]; // 3×2
        let u1 = array![[2.0, 0.0], [0.0, 2.0]]; // 2×2

        let seq = crate::nmode::tucker_reconstruct(&core.view(), &[u0.view(), u1.view()]).unwrap();
        let fused = tucker_reconstruct_fused(&core.view(), &[u0.view(), u1.view()]).unwrap();

        assert_eq!(seq.shape(), fused.shape());
        assert!(max_abs_diff_dyn(&seq, &fused) < 1e-12);
    }

    #[test]
    fn test_fused_4mode() {
        let mut state: u64 = 0xCAFE_BABE;
        let mut v = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 32) as f64 / u32::MAX as f64 - 0.5
        };

        let core_data: Vec<f64> = (0..16).map(|_| v()).collect(); // 2×2×2×2
        let u0_data: Vec<f64> = (0..6).map(|_| v()).collect(); // 3×2
        let u1_data: Vec<f64> = (0..6).map(|_| v()).collect(); // 3×2
        let u2_data: Vec<f64> = (0..4).map(|_| v()).collect(); // 2×2 (identity-ish)
        let u3_data: Vec<f64> = (0..8).map(|_| v()).collect(); // 4×2

        let core = Array::from_shape_vec(vec![2, 2, 2, 2], core_data).unwrap();
        let u0 = Array::from_shape_vec((3, 2), u0_data).unwrap();
        let u1 = Array::from_shape_vec((3, 2), u1_data).unwrap();
        let u2 = Array::from_shape_vec((2, 2), u2_data).unwrap();
        let u3 = Array::from_shape_vec((4, 2), u3_data).unwrap();

        let seq = crate::nmode::tucker_reconstruct(
            &core.view(),
            &[u0.view(), u1.view(), u2.view(), u3.view()],
        )
        .unwrap();
        let fused =
            tucker_reconstruct_fused(&core.view(), &[u0.view(), u1.view(), u2.view(), u3.view()])
                .unwrap();

        assert_eq!(seq.shape(), fused.shape());
        assert!(
            max_abs_diff_dyn(&seq, &fused) < 1e-10,
            "4-mode max diff = {}",
            max_abs_diff_dyn(&seq, &fused)
        );
    }

    #[test]
    fn test_fused_empty_output() {
        // zero mode-0 output dimension
        let core = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let u0 = Array::from_shape_vec((0, 2), vec![]).unwrap(); // 0×2 → empty output
        let u1 = Array::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();

        let result = tucker_reconstruct_fused(&core.view(), &[u0.view(), u1.view()]).unwrap();
        assert_eq!(result.shape(), &[0, 4]);
    }

    #[test]
    #[should_panic(expected = "Number of factor matrices")]
    fn test_fused_wrong_num_factors() {
        let core = Array::from_shape_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let u = array![[1.0, 0.0], [0.0, 1.0]];
        // Only 1 factor for 3-mode core
        tucker_reconstruct_fused(&core.view(), &[u.view()]).unwrap();
    }

    #[test]
    #[should_panic(expected = "columns")]
    fn test_fused_column_mismatch() {
        let core = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let u0 = Array::from_shape_vec((4, 2), vec![1.0; 8]).unwrap(); // OK
        let u1 = Array::from_shape_vec((4, 2), vec![1.0; 8]).unwrap(); // wrong: core mode 1 has size 3
        tucker_reconstruct_fused(&core.view(), &[u0.view(), u1.view()]).unwrap();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_fused_parallel_matches_serial_3mode() {
        let mut state: u64 = 0xABCD_EF01;
        let mut v = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 32) as f64 / u32::MAX as f64 - 0.5
        };

        let core_data: Vec<f64> = (0..12).map(|_| v()).collect(); // 2×2×3
        let u0_data: Vec<f64> = (0..8).map(|_| v()).collect(); // 4×2
        let u1_data: Vec<f64> = (0..6).map(|_| v()).collect(); // 3×2
        let u2_data: Vec<f64> = (0..15).map(|_| v()).collect(); // 5×3

        let core = Array::from_shape_vec(vec![2, 2, 3], core_data).unwrap();
        let u0 = Array::from_shape_vec((4, 2), u0_data).unwrap();
        let u1 = Array::from_shape_vec((3, 2), u1_data).unwrap();
        let u2 = Array::from_shape_vec((5, 3), u2_data).unwrap();

        let serial =
            tucker_reconstruct_fused(&core.view(), &[u0.view(), u1.view(), u2.view()]).unwrap();
        let par =
            tucker_reconstruct_fused_parallel(&core.view(), &[u0.view(), u1.view(), u2.view()])
                .unwrap();

        assert_eq!(serial.shape(), par.shape());
        assert!(
            max_abs_diff_dyn(&serial, &par) < 1e-12,
            "parallel vs serial max diff = {}",
            max_abs_diff_dyn(&serial, &par)
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_fused_parallel_matches_tucker_reconstruct() {
        let mut state: u64 = 0xFEED_FACE;
        let mut v = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 32) as f64 / u32::MAX as f64 - 0.5
        };

        let core_data: Vec<f64> = (0..8).map(|_| v()).collect(); // 2×2×2
        let u0_data: Vec<f64> = (0..10).map(|_| v()).collect(); // 5×2
        let u1_data: Vec<f64> = (0..8).map(|_| v()).collect(); // 4×2
        let u2_data: Vec<f64> = (0..6).map(|_| v()).collect(); // 3×2

        let core = Array::from_shape_vec(vec![2, 2, 2], core_data).unwrap();
        let u0 = Array::from_shape_vec((5, 2), u0_data).unwrap();
        let u1 = Array::from_shape_vec((4, 2), u1_data).unwrap();
        let u2 = Array::from_shape_vec((3, 2), u2_data).unwrap();

        let seq =
            crate::nmode::tucker_reconstruct(&core.view(), &[u0.view(), u1.view(), u2.view()])
                .unwrap();
        let par =
            tucker_reconstruct_fused_parallel(&core.view(), &[u0.view(), u1.view(), u2.view()])
                .unwrap();

        assert_eq!(seq.shape(), par.shape());
        assert!(
            max_abs_diff_dyn(&seq, &par) < 1e-10,
            "fused-parallel vs reconstruct max diff = {}",
            max_abs_diff_dyn(&seq, &par)
        );
    }
}
