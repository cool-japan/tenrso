//! Tensor Train decomposition (TT-SVD and TT-rounding)
//!
//! The Tensor Train (TT) decomposition represents an N-way tensor as a sequence
//! of 3-way tensors (TT-cores):
//!
//! X(i₁, i₂, ..., iₙ) = G₁\[i₁\] × G₂\[i₂\] × ... × Gₙ\[iₙ\]
//!
//! Where:
//! - Gₖ is a TT-core with shape (rₖ₋₁, iₖ, rₖ)
//! - r₀ = rₙ = 1 (boundary conditions)
//! - r₁, r₂, ..., rₙ₋₁ are TT-ranks
//!
//! # Algorithms
//!
//! ## TT-SVD
//! Computes TT decomposition via sequential SVD with rank truncation.
//! Time: O(N × I³ × R²) where I = max mode size, R = max TT-rank
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! SVD operations use `scirs2_linalg::decomposition`.
//! Direct use of `ndarray` or `num_traits` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, Array3, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, NumCast};
use scirs2_linalg::svd;
use std::iter::Sum;
use tenrso_core::DenseND;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TTError {
    #[error("Invalid ranks: {0}")]
    InvalidRanks(String),

    #[error("SVD failed: {0}")]
    SvdError(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Invalid tensor: {0}")]
    InvalidTensor(String),
}

/// Tensor Train decomposition result
///
/// Represents a tensor as a sequence of TT-cores G₁, G₂, ..., Gₙ
///
/// # Structure
///
/// Each core Gₖ has shape (rₖ₋₁, iₖ, rₖ) where:
/// - rₖ₋₁ is the left TT-rank
/// - iₖ is the mode size
/// - rₖ is the right TT-rank
///
/// Boundary conditions: r₀ = rₙ = 1
#[derive(Clone)]
pub struct TTDecomp<T>
where
    T: Clone + Float,
{
    /// TT-cores: each core is a 3-way tensor (r_{k-1}, I_k, r_k)
    pub cores: Vec<Array3<T>>,

    /// TT-ranks: [r₁, r₂, ..., rₙ₋₁]
    pub ranks: Vec<usize>,

    /// Original tensor shape
    pub shape: Vec<usize>,

    /// Reconstruction error (if computed)
    pub error: Option<T>,
}

impl<T> TTDecomp<T>
where
    T: Float + NumCast + 'static,
{
    /// Reconstruct the original tensor from TT decomposition
    ///
    /// Computes X(i₁, ..., iₙ) = G₁\[i₁\] × G₂\[i₂\] × ... × Gₙ\[iₙ\]
    ///
    /// # Complexity
    ///
    /// Time: O(∏ᵢ Iᵢ × R²) where R = max TT-rank
    /// Space: O(∏ᵢ Iᵢ)
    pub fn reconstruct(&self) -> Result<DenseND<T>> {
        use scirs2_core::ndarray_ext::{ArrayD, Axis, IxDyn};

        let n_modes = self.cores.len();

        if n_modes == 0 {
            return Err(anyhow::anyhow!("Empty TT decomposition"));
        }

        // Start with first core reshaped to (I₁, r₁)
        let first_core = &self.cores[0];
        let shape_0 = first_core.shape();

        if shape_0[0] != 1 {
            return Err(anyhow::anyhow!(
                "First core must have left rank 1, got {}",
                shape_0[0]
            ));
        }

        // Initialize accumulator as (I₁, r₁) in dynamic dimension
        let first_2d = first_core.index_axis(Axis(0), 0).to_owned();
        let mut acc: ArrayD<T> = first_2d.into_dyn();

        // Contract with each subsequent core
        for k in 1..n_modes {
            let core = &self.cores[k];
            let core_shape = core.shape();
            let (r_left, i_k, r_right) = (core_shape[0], core_shape[1], core_shape[2]);

            // acc has shape (..., r_left)
            // core has shape (r_left, i_k, r_right)
            // Result will have shape (..., i_k, r_right)

            let acc_shape = acc.shape().to_vec();
            let prod_size: usize = acc_shape[..acc_shape.len() - 1].iter().product();
            let acc_last = acc_shape[acc_shape.len() - 1];

            // Reshape acc to (prod_size, r_left)
            let acc_2d = acc
                .into_shape_with_order((prod_size, acc_last))
                .map_err(|e| anyhow::anyhow!("Reshape failed: {}", e))?;

            // Contract: (prod_size, r_left) × (r_left, i_k * r_right) = (prod_size, i_k * r_right)
            let core_2d = core
                .view()
                .into_shape_with_order((r_left, i_k * r_right))
                .map_err(|e| anyhow::anyhow!("Core reshape failed: {}", e))?;

            let contracted = acc_2d.dot(&core_2d);

            // Reshape to (..., i_k, r_right)
            let mut new_shape = acc_shape[..acc_shape.len() - 1].to_vec();
            new_shape.push(i_k);
            new_shape.push(r_right);

            acc = contracted
                .into_shape_with_order(IxDyn(new_shape.as_slice()))
                .map_err(|e| anyhow::anyhow!("Result reshape failed: {}", e))?;
        }

        // Final core should have r_right = 1, so squeeze last dimension
        let final_shape = acc.shape().to_vec();
        if final_shape[final_shape.len() - 1] != 1 {
            return Err(anyhow::anyhow!("Last core must have right rank 1"));
        }

        let result_shape = &final_shape[..final_shape.len() - 1];
        let squeezed = acc
            .into_shape_with_order(IxDyn(result_shape))
            .map_err(|e| anyhow::anyhow!("Final squeeze failed: {}", e))?;

        // Convert to DenseND
        let result = DenseND::from_array(squeezed);
        Ok(result)
    }

    /// Compute reconstruction error: ||X - X_reconstructed|| / ||X||
    pub fn compute_error(&mut self, original: &DenseND<T>) -> Result<T> {
        let reconstructed = self.reconstruct()?;

        let mut error_sq = T::zero();
        let mut norm_sq = T::zero();

        let orig_view = original.view();
        let recon_view = reconstructed.view();

        for (orig_val, recon_val) in orig_view.iter().zip(recon_view.iter()) {
            let diff = *orig_val - *recon_val;
            error_sq = error_sq + diff * diff;
            norm_sq = norm_sq + (*orig_val) * (*orig_val);
        }

        let error = (error_sq / norm_sq).sqrt();
        self.error = Some(error);
        Ok(error)
    }

    /// Get number of parameters in TT representation
    pub fn num_parameters(&self) -> usize {
        self.cores.iter().map(|core| core.len()).sum()
    }

    /// Get compression ratio compared to full tensor
    pub fn compression_ratio(&self) -> f64 {
        let full_size: usize = self.shape.iter().product();
        let tt_size = self.num_parameters();
        full_size as f64 / tt_size as f64
    }
}

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

        // Compute SVD
        let (u, s, vt) = svd(&c_matrix.view(), false, None)
            .map_err(|e| TTError::SvdError(format!("SVD failed at mode {}: {}", k, e)))?;

        // Determine actual rank (truncate by max_rank and tolerance)
        let max_r = max_ranks[k].min(s.len());
        let s_max = s[0];
        let threshold = T::from(tol).unwrap() * s_max;

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

        // Multiply rows by singular values
        let mut c_next = Array2::<T>::zeros((r_right, cols));
        for i in 0..r_right {
            for j in 0..cols {
                c_next[[i, j]] = s_trunc[i] * vt_trunc[[i, j]];
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
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10, 10], 0.0, 1.0);
/// let tt = tt_svd(&tensor, &[20, 20, 20], 1e-10).unwrap();
///
/// // Round to smaller ranks
/// let tt_rounded = tt_round(&tt, &[5, 5, 5], 1e-6).unwrap();
/// println!("Original ranks: {:?}", tt.ranks);
/// println!("Rounded ranks: {:?}", tt_rounded.ranks);
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
        let tol_t: T = NumCast::from(tol).unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_svd_basic() {
        // Small tensor for quick test
        let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
        let result = tt_svd(&tensor, &[2, 2], 1e-10);

        if result.is_err() {
            eprintln!("TT-SVD error: {:?}", result.err());
            panic!("TT-SVD failed");
        }

        let tt = result.unwrap();

        assert_eq!(tt.cores.len(), 3);
        assert_eq!(tt.ranks.len(), 2);

        // Check core shapes
        assert_eq!(tt.cores[0].shape(), &[1, 3, tt.ranks[0]]);
        assert_eq!(tt.cores[1].shape(), &[tt.ranks[0], 4, tt.ranks[1]]);
        assert_eq!(tt.cores[2].shape(), &[tt.ranks[1], 5, 1]);
    }

    #[test]
    fn test_tt_reconstruction() {
        let tensor = DenseND::<f64>::random_uniform(&[3, 4, 5], 0.0, 1.0);
        let mut tt = tt_svd(&tensor, &[3, 4], 1e-10).unwrap();

        let reconstructed = tt.reconstruct();
        assert!(reconstructed.is_ok());

        let error = tt.compute_error(&tensor);
        assert!(error.is_ok());
        assert!(error.unwrap() < 0.5); // Reasonable reconstruction
    }

    #[test]
    fn test_tt_compression() {
        let tensor = DenseND::<f64>::ones(&[10, 10, 10, 10]);
        let tt = tt_svd(&tensor, &[5, 5, 5], 1e-10).unwrap();

        let full_size = 10 * 10 * 10 * 10;
        let tt_size = tt.num_parameters();

        assert!(tt_size < full_size);
        assert!(tt.compression_ratio() > 1.0);
    }

    #[test]
    fn test_tt_round_basic() {
        // Create a TT decomposition with larger ranks (reduced size for speed)
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6, 6], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[4, 4, 4], 1e-10).unwrap();

        // Round to smaller ranks
        let tt_rounded = tt_round(&tt, &[2, 2, 2], 1e-6).unwrap();

        // Check that ranks are reduced
        for (i, &rank) in tt_rounded.ranks.iter().enumerate() {
            assert!(rank <= 2, "Rounded rank {} is {}, expected <= 2", i, rank);
        }

        // Check core shapes are valid
        assert_eq!(tt_rounded.cores.len(), 4);
        assert_eq!(tt_rounded.cores[0].shape()[0], 1); // First core left rank = 1
        assert_eq!(tt_rounded.cores[3].shape()[2], 1); // Last core right rank = 1
    }

    #[test]
    fn test_tt_round_reconstruction() {
        // Create a low-rank tensor (rank-1 tensor = outer product)
        let tensor = DenseND::<f64>::random_uniform(&[5, 6, 7], 0.0, 1.0);

        // Decompose with larger ranks
        let tt = tt_svd(&tensor, &[4, 5], 1e-10).unwrap();

        // Round to smaller ranks
        let mut tt_rounded = tt_round(&tt, &[2, 2], 1e-6).unwrap();

        // Verify reconstruction is reasonable
        let reconstructed = tt_rounded.reconstruct().unwrap();
        assert_eq!(reconstructed.shape(), tensor.shape());

        let error = tt_rounded.compute_error(&tensor).unwrap();
        assert!(error < 1.0, "Reconstruction error too large: {}", error);
    }

    #[test]
    fn test_tt_round_preserves_accuracy() {
        // Test that rounding with high tolerance preserves accuracy
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[5, 5], 1e-10).unwrap();

        // Round with very loose tolerance (should preserve accuracy)
        let mut tt_rounded = tt_round(&tt, &[5, 5], 1e-3).unwrap();

        let error = tt_rounded.compute_error(&tensor).unwrap();
        assert!(error < 0.3, "Error after rounding is too large: {}", error);
    }

    #[test]
    fn test_tt_round_compression() {
        // Verify that rounding reduces storage (reduced size for speed)
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6, 6], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[5, 5, 5], 1e-10).unwrap();
        let original_params = tt.num_parameters();

        // Round to smaller ranks
        let tt_rounded = tt_round(&tt, &[3, 3, 3], 1e-6).unwrap();
        let rounded_params = tt_rounded.num_parameters();

        assert!(
            rounded_params < original_params,
            "Rounding should reduce parameters: {} >= {}",
            rounded_params,
            original_params
        );

        // Compression ratio should increase
        assert!(
            tt_rounded.compression_ratio() > tt.compression_ratio(),
            "Rounded compression ratio {} should be > original {}",
            tt_rounded.compression_ratio(),
            tt.compression_ratio()
        );
    }

    #[test]
    fn test_tt_round_ranks_not_exceed_max() {
        // Test that rounded ranks never exceed max_ranks (reduced size for speed)
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6, 6], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[5, 5, 5], 1e-12).unwrap();

        let max_ranks = vec![2, 3, 2];
        let tt_rounded = tt_round(&tt, &max_ranks, 1e-8).unwrap();

        for (i, &rank) in tt_rounded.ranks.iter().enumerate() {
            assert!(
                rank <= max_ranks[i],
                "Rank {} is {}, exceeds max {}",
                i,
                rank,
                max_ranks[i]
            );
        }
    }
}
