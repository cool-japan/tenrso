//! TT construction algorithms: TT-SVD and TT-rounding.
//!
//! `tt_svd` builds a `TTDecomp` from a dense tensor via sequential SVD with
//! rank truncation. `tt_round` reduces the ranks of an existing `TTDecomp`
//! through right-to-left orthogonalization followed by left-to-right
//! truncation.
//!
//! All SVD/QR operations are delegated to `scirs2_linalg`; arrays use
//! `scirs2_core::ndarray_ext`.

use scirs2_core::ndarray_ext::{Array2, Array3, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, NumCast};
use scirs2_linalg::svd;
use std::iter::Sum;
use tenrso_core::DenseND;

use super::types::{TTDecomp, TTError};

/// Compute TT-SVD decomposition with rank truncation
///
/// Decomposes a tensor into Tensor Train format using sequential SVD.
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `max_ranks` - Maximum TT-ranks [r₁, r₂, ..., rₙ₋₁] (or single value for all)
/// * `tol` - Truncation tolerance (keep singular values > tol * σ_max)
///
/// # Returns
///
/// TTDecomp containing TT-cores and ranks
///
/// # Errors
///
/// Returns error if:
/// - Tensor has less than 2 modes
/// - Max ranks are invalid
/// - SVD computation fails
///
/// # Complexity
///
/// Time: O(N × I³ × R²) where I = max mode size, R = max TT-rank
/// Space: O(I² × R)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::tt_svd;
///
/// // Create a 10×10×10×10 tensor
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10, 10], 0.0, 1.0);
///
/// // Decompose with max TT-ranks [5, 5, 5]
/// let tt = tt_svd(&tensor, &[5, 5, 5], 1e-10).unwrap();
///
/// println!("TT-ranks: {:?}", tt.ranks);
/// println!("Compression ratio: {:.2}x", tt.compression_ratio());
/// ```
pub fn tt_svd<T>(tensor: &DenseND<T>, max_ranks: &[usize], tol: f64) -> Result<TTDecomp<T>, TTError>
where
    T: Float + NumCast + NumAssign + Sum + Send + Sync + ScalarOperand + std::fmt::Debug + 'static,
{
    let shape = tensor.shape().to_vec();
    let n_modes = shape.len();

    // Validation
    if n_modes < 2 {
        return Err(TTError::InvalidTensor(format!(
            "Tensor must have at least 2 modes, got {}",
            n_modes
        )));
    }

    if max_ranks.len() != n_modes - 1 {
        return Err(TTError::InvalidRanks(format!(
            "Expected {} max ranks, got {}",
            n_modes - 1,
            max_ranks.len()
        )));
    }

    // Validate max ranks
    for (k, &r) in max_ranks.iter().enumerate() {
        if r == 0 {
            return Err(TTError::InvalidRanks(format!("Max rank {} is zero", k)));
        }
    }

    let mut cores = Vec::with_capacity(n_modes);
    let mut actual_ranks = Vec::with_capacity(n_modes - 1);
    let tol_t: T = NumCast::from(tol).ok_or_else(|| {
        TTError::ShapeMismatch(format!("could not convert tol={tol} to scalar type"))
    })?;

    // Initialize C as the full tensor reshaped
    let mut c_data = tensor.view().iter().cloned().collect::<Vec<_>>();

    // Left rank for current iteration
    let mut r_left = 1;

    // TT-SVD iterations
    for k in 0..n_modes - 1 {
        let i_k = shape[k]; // Use original mode size, not c_shape[0]
        let i_rest: usize = shape[k + 1..].iter().product();

        // Reshape C to (r_left * i_k, i_rest)
        let rows = r_left * i_k;
        let cols = i_rest;

        let c_matrix = Array2::from_shape_vec((rows, cols), c_data)
            .map_err(|e| TTError::ShapeMismatch(format!("Matrix reshape failed: {}", e)))?;

        let target_rank = max_ranks[k];

        // Choose SVD strategy based on matrix shape:
        //  ① Short-fat extreme (rows ≤ 64, cols ≥ 1M): Gram SVD avoids allocating a
        //    cols×(k+oversampling) Gaussian Omega that would be GBs for very wide
        //    TT unfoldings (e.g. first step of 32^6 has cols ≈ 33M).
        //    Gram SVD costs O(m²n) — two matrix multiplies — which is heavier than
        //    full SVD for moderate sizes; hence the 1M lower bound on cols.
        //  ② Otherwise large: randomized SVD (Halko-Martinsson-Tropp).
        //  ③ Small/balanced: full thin SVD.
        let (u, s, vt) = if rows <= 64 && cols >= 1_000_000 {
            crate::utils::thin_svd_via_gram(&c_matrix.view(), target_rank)
                .map_err(|e| TTError::SvdError(format!("Gram SVD failed at mode {}: {}", k, e)))?
        } else if crate::utils::should_use_randomized_svd(rows, cols, target_rank) {
            crate::utils::randomized_svd_truncated(&c_matrix.view(), target_rank, 10, 2).map_err(
                |e| TTError::SvdError(format!("Randomized SVD failed at mode {}: {}", k, e)),
            )?
        } else {
            svd(&c_matrix.view(), false, None)
                .map_err(|e| TTError::SvdError(format!("SVD failed at mode {}: {}", k, e)))?
        };

        // Determine actual rank (truncate by max_rank and tolerance)
        let max_r = max_ranks[k].min(s.len());
        let s_max = s[0];
        let threshold = tol_t * s_max;

        let mut r_right = 0;
        for (idx, &sigma) in s.iter().enumerate().take(max_r) {
            if sigma > threshold {
                r_right = idx + 1;
            } else {
                break;
            }
        }

        if r_right == 0 {
            r_right = 1; // Keep at least one singular value
        }

        actual_ranks.push(r_right);

        // Extract TT-core: reshape U[:, :r_right] to (r_left, i_k, r_right)
        let u_trunc = u
            .slice(scirs2_core::ndarray_ext::s![.., ..r_right])
            .to_owned();

        let core_data: Vec<T> = u_trunc.iter().cloned().collect();
        let core_3d = Array3::from_shape_vec((r_left, i_k, r_right), core_data)
            .map_err(|e| TTError::ShapeMismatch(format!("Core reshape failed: {}", e)))?;

        cores.push(core_3d);

        // Compute new C = diag(S[:r_right]) @ Vᵀ[:r_right, :]
        let s_trunc = s.slice(scirs2_core::ndarray_ext::s![..r_right]);
        let vt_trunc = vt
            .slice(scirs2_core::ndarray_ext::s![..r_right, ..])
            .to_owned();

        // Multiply rows by singular values using efficient row scaling
        let mut c_next = vt_trunc.to_owned();
        for i in 0..r_right {
            let scale = s_trunc[i];
            for j in 0..cols {
                c_next[[i, j]] = scale * c_next[[i, j]];
            }
        }

        // Update for next iteration
        c_data = c_next.iter().cloned().collect();
        r_left = r_right;
    }

    // Final core: C_{n-1} has shape (r_{n-1}, I_n)
    // Reshape to (r_{n-1}, I_n, 1)
    let last_rows = r_left;
    let last_cols = shape[n_modes - 1];

    if c_data.len() != last_rows * last_cols {
        return Err(TTError::ShapeMismatch(format!(
            "Final core size mismatch: expected {}, got {}",
            last_rows * last_cols,
            c_data.len()
        )));
    }

    let last_core = Array3::from_shape_vec((last_rows, last_cols, 1), c_data)
        .map_err(|e| TTError::ShapeMismatch(format!("Last core reshape failed: {}", e)))?;

    cores.push(last_core);

    Ok(TTDecomp {
        cores,
        ranks: actual_ranks,
        shape,
        error: None,
    })
}

/// TT-rounding: reduce TT-ranks of an existing TT decomposition
///
/// Applies a right-to-left orthogonalization followed by left-to-right truncation
/// to reduce the TT-ranks while controlling the approximation error.
///
/// This is useful for:
/// - Memory optimization after TT operations (addition, multiplication)
/// - Post-processing to reduce storage while maintaining accuracy
/// - Controlling approximation error more tightly
///
/// # Arguments
///
/// * `tt` - Input TT decomposition to round
/// * `max_ranks` - Maximum TT-ranks after rounding
/// * `tol` - Truncation tolerance (relative error control)
///
/// # Returns
///
/// New TTDecomp with reduced ranks
///
/// # Algorithm
///
/// 1. Right-to-left orthogonalization (QR decompositions)
/// 2. Left-to-right truncation (SVD with rank reduction)
///
/// # Complexity
///
/// Time: O(N × R³) where R = max TT-rank
/// Space: O(R³)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::{tt_svd, tt_round};
///
/// // Create TT decomposition
/// let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6, 6], 0.0, 1.0);
/// let tt = tt_svd(&tensor, &[8, 8, 8], 1e-10).unwrap();
///
/// // Round to smaller ranks
/// let tt_rounded = tt_round(&tt, &[4, 4, 4], 1e-6).unwrap();
/// println!("Original ranks: {:?}", tt.ranks);
/// println!("Rounded ranks: {:?}", tt_rounded.ranks);
/// # assert_eq!(tt.cores.len(), 4);
/// # assert_eq!(tt_rounded.cores.len(), 4);
/// ```
pub fn tt_round<T>(tt: &TTDecomp<T>, max_ranks: &[usize], tol: f64) -> Result<TTDecomp<T>, TTError>
where
    T: Float + NumCast + NumAssign + Sum + Send + Sync + ScalarOperand + std::fmt::Debug + 'static,
{
    let n_modes = tt.cores.len();

    // Validation
    if max_ranks.len() != n_modes - 1 {
        return Err(TTError::InvalidRanks(format!(
            "Expected {} max ranks, got {}",
            n_modes - 1,
            max_ranks.len()
        )));
    }

    // Clone cores for modification
    let mut cores = tt.cores.clone();
    let tol_t: T = NumCast::from(tol).ok_or_else(|| {
        TTError::ShapeMismatch(format!("could not convert tol={tol} to scalar type"))
    })?;

    // Step 1: Right-to-left orthogonalization using QR decomposition
    for k in (1..n_modes).rev() {
        let core = &cores[k];
        let (r_left, n_k, r_right) = core.dim();

        // Reshape core to matrix (n_k * r_right, r_left) - transposed for QR
        let core_mat = core
            .clone()
            .into_shape_with_order((r_left, n_k * r_right))
            .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?;

        // Transpose to get (n_k * r_right, r_left) for QR
        let core_mat_t = core_mat.t().to_owned();

        // QR decomposition: core_mat^T = Q * R
        use scirs2_linalg::qr;
        let (q, r_mat) = qr(&core_mat_t.view(), None)
            .map_err(|e| TTError::SvdError(format!("QR failed: {}", e)))?;

        // Q has shape (n_k * r_right, r_left)
        // Take only first r_left columns (thin QR)
        use scirs2_core::ndarray_ext::s;
        let min_dim = r_left.min(n_k * r_right);
        let q_thin = q.slice(s![.., ..min_dim]).to_owned();

        // Transpose Q back and reshape to core shape (r_left, n_k, r_right)
        let q_t = q_thin.t().to_owned();
        cores[k] = Array3::from_shape_vec(
            (min_dim, n_k, r_right),
            q_t.into_shape_with_order(min_dim * n_k * r_right)
                .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?
                .to_vec(),
        )
        .map_err(|e| TTError::ShapeMismatch(format!("Core reshape failed: {}", e)))?;

        // Absorb R into previous core
        // R has shape (r_left, r_left) or smaller
        if k > 0 {
            let prev_core = &cores[k - 1];
            let (r_prev_left, n_prev, r_prev_right) = prev_core.dim();

            // Reshape previous core to matrix (r_prev_left * n_prev, r_prev_right)
            let prev_mat = prev_core
                .clone()
                .into_shape_with_order((r_prev_left * n_prev, r_prev_right))
                .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?;

            // Multiply: prev_mat * R^T
            // R has shape (min_dim, min_dim), we need (r_prev_right, min_dim)
            let r_slice = r_mat.slice(s![..min_dim, ..min_dim]).to_owned();
            let result = prev_mat.dot(&r_slice.t());

            // Reshape back
            cores[k - 1] = Array3::from_shape_vec(
                (r_prev_left, n_prev, min_dim),
                result
                    .into_shape_with_order(r_prev_left * n_prev * min_dim)
                    .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?
                    .to_vec(),
            )
            .map_err(|e| TTError::ShapeMismatch(format!("Core reshape failed: {}", e)))?;
        }
    }

    // Step 2: Left-to-right truncation using SVD
    let mut new_cores = Vec::with_capacity(n_modes);
    let mut new_ranks = Vec::with_capacity(n_modes - 1);
    let mut r_left = 1;

    for k in 0..n_modes - 1 {
        let core = &cores[k];
        let (_r_l, n_k, r_right) = core.dim();

        // Reshape to matrix (r_left * n_k, r_right)
        let core_mat = core
            .clone()
            .into_shape_with_order((r_left * n_k, r_right))
            .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?;

        // SVD with truncation
        let (u, s, vt) = svd(&core_mat.view(), true, None)
            .map_err(|e| TTError::SvdError(format!("SVD failed: {}", e)))?;

        // Determine truncation rank
        let max_rank_k = max_ranks[k];
        let s_max = s[0];
        let threshold = tol_t * s_max;

        let mut trunc_rank = 0;
        for (i, &sigma) in s.iter().enumerate() {
            if sigma > threshold && i < max_rank_k {
                trunc_rank = i + 1;
            } else {
                break;
            }
        }
        trunc_rank = trunc_rank.max(1).min(max_rank_k).min(s.len());

        // Truncate U, S, VT
        use scirs2_core::ndarray_ext::s;
        let u_trunc = u.slice(s![.., ..trunc_rank]).to_owned();
        let s_trunc = s.slice(s![..trunc_rank]).to_owned();
        let vt_trunc = vt.slice(s![..trunc_rank, ..]).to_owned();

        // Create new core: reshape U to (r_left, n_k, trunc_rank)
        let new_core = Array3::from_shape_vec(
            (r_left, n_k, trunc_rank),
            u_trunc
                .into_shape_with_order(r_left * n_k * trunc_rank)
                .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?
                .to_vec(),
        )
        .map_err(|e| TTError::ShapeMismatch(format!("Core reshape failed: {}", e)))?;

        new_cores.push(new_core);
        new_ranks.push(trunc_rank);

        // Absorb S * VT into next core
        let s_vt = {
            let mut result = Array2::zeros((trunc_rank, vt_trunc.ncols()));
            for i in 0..trunc_rank {
                for j in 0..vt_trunc.ncols() {
                    result[[i, j]] = s_trunc[i] * vt_trunc[[i, j]];
                }
            }
            result
        };

        // Multiply with next core
        let next_core = &cores[k + 1];
        let (next_r_left, next_n, next_r_right) = next_core.dim();

        let next_mat = next_core
            .clone()
            .into_shape_with_order((next_r_left, next_n * next_r_right))
            .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?;

        let result = s_vt.dot(&next_mat);

        cores[k + 1] = Array3::from_shape_vec(
            (trunc_rank, next_n, next_r_right),
            result
                .into_shape_with_order(trunc_rank * next_n * next_r_right)
                .map_err(|e| TTError::ShapeMismatch(format!("Reshape failed: {}", e)))?
                .to_vec(),
        )
        .map_err(|e| TTError::ShapeMismatch(format!("Core reshape failed: {}", e)))?;

        r_left = trunc_rank;
    }

    // Add last core
    new_cores.push(cores[n_modes - 1].clone());

    Ok(TTDecomp {
        cores: new_cores,
        ranks: new_ranks,
        shape: tt.shape.clone(),
        error: None,
    })
}
