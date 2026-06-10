//! Tucker decomposition (HOSVD and HOOI)
//!
//! The Tucker decomposition factorizes a tensor X into a core tensor G and factor matrices:
//!
//! X ≈ G ×₁ U₁ ×₂ U₂ ×₃ ... ×ₙ Uₙ
//!
//! Where:
//! - G is the core tensor with shape (R₁, R₂, ..., Rₙ)
//! - Uᵢ are orthogonal factor matrices with shape (Iᵢ, Rᵢ)
//! - ×ᵢ denotes the i-mode product
//!
//! # Algorithms
//!
//! ## HOSVD (Higher-Order SVD)
//! One-pass algorithm based on SVD of mode-n unfoldings. Fast but suboptimal.
//!
//! ## HOOI (Higher-Order Orthogonal Iteration)
//! Iterative refinement of HOSVD using ALS-like updates. Better approximation.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! SVD operations use `scirs2_linalg::decomposition`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, NumCast};
use scirs2_linalg::svd;
use std::iter::Sum;
use tenrso_core::DenseND;
use tenrso_kernels::nmode_product;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TuckerError {
    #[error("Invalid ranks: {0}")]
    InvalidRanks(String),

    #[error("SVD failed: {0}")]
    SvdError(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
}

/// Tucker decomposition result
///
/// Represents a tensor as G ×₁ U₁ ×₂ U₂ ×₃ ... ×ₙ Uₙ
#[derive(Clone, Debug)]
pub struct TuckerDecomp<T>
where
    T: Clone + Float,
{
    /// Core tensor with shape (R₁, R₂, ..., Rₙ)
    pub core: DenseND<T>,

    /// Factor matrices, one for each mode
    /// Each matrix Uᵢ has shape (Iᵢ, Rᵢ) and is orthogonal
    pub factors: Vec<Array2<T>>,

    /// Reconstruction error (if computed)
    pub error: Option<T>,

    /// Number of iterations (for HOOI)
    pub iters: usize,
}

impl<T> TuckerDecomp<T>
where
    T: Float + NumCast,
{
    /// Reconstruct the original tensor from Tucker decomposition
    ///
    /// Computes X ≈ G ×₁ U₁ ×₂ U₂ ×₃ ... ×ₙ Uₙ
    ///
    /// Uses optimized Tucker reconstruction from tenrso-kernels.
    ///
    /// # Complexity
    ///
    /// Time: O(N × ∏ᵢ Rᵢ × Iᵢ)
    /// Space: O(∏ᵢ Iᵢ)
    pub fn reconstruct(&self) -> Result<DenseND<T>> {
        // Use optimized kernel reconstruction
        let factor_views: Vec<_> = self.factors.iter().map(|f| f.view()).collect();
        let core_view = self.core.view();

        let reconstructed = tenrso_kernels::tucker_reconstruct(&core_view, &factor_views)?;

        // Wrap in DenseND
        Ok(DenseND::from_array(reconstructed))
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

    /// Compute compression ratio: original_elements / tucker_elements
    ///
    /// Tucker storage: core (∏ᵢ Rᵢ) + factors (∑ᵢ Iᵢ × Rᵢ)
    pub fn compression_ratio(&self) -> f64 {
        // Original tensor size
        let original_shape = self.factors.iter().map(|f| f.nrows()).collect::<Vec<_>>();
        let original_elements: usize = original_shape.iter().product();

        // Core tensor size
        let core_elements: usize = self.core.shape().iter().product();

        // Factor matrices size: ∑ᵢ (Iᵢ × Rᵢ)
        let factors_elements: usize = self.factors.iter().map(|f| f.nrows() * f.ncols()).sum();

        let tucker_elements = core_elements + factors_elements;

        original_elements as f64 / tucker_elements as f64
    }
}

/// Strategy for automatic rank selection in Tucker decomposition
#[derive(Debug, Clone)]
pub enum TuckerRankSelection {
    /// Keep components that preserve a fraction of the energy
    /// Value should be in (0, 1), e.g., 0.9 means keep 90% of energy
    Energy(f64),

    /// Keep singular values above threshold × max_singular_value
    /// Value should be in (0, 1), e.g., 0.01 means keep σ > 0.01 * σ_max
    Threshold(f64),

    /// Keep first k singular values for each mode
    Fixed(Vec<usize>),
}

/// Compute Tucker-HOSVD decomposition with automatic rank selection
///
/// Automatically determines Tucker ranks based on energy preservation or singular value thresholds.
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `selection` - Rank selection strategy
///
/// # Returns
///
/// TuckerDecomp containing core tensor and factor matrices with automatically determined ranks
///
/// # Errors
///
/// Returns error if:
/// - Invalid selection parameters
/// - SVD computation fails
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tucker::{tucker_hosvd_auto, TuckerRankSelection};
///
/// // Create a 10×10×10 tensor
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
///
/// // Decompose preserving 90% of energy in each mode
/// let tucker = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.9)).unwrap();
///
/// println!("Auto-selected ranks: {:?}",
///     tucker.factors.iter().map(|f| f.ncols()).collect::<Vec<_>>());
/// ```
pub fn tucker_hosvd_auto<T>(
    tensor: &DenseND<T>,
    selection: TuckerRankSelection,
) -> Result<TuckerDecomp<T>, TuckerError>
where
    T: Float + NumCast + NumAssign + Sum + Send + Sync + ScalarOperand + std::fmt::Debug + 'static,
{
    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Compute SVD for each mode and determine ranks
    let mut ranks = Vec::with_capacity(n_modes);
    let mut factors = Vec::with_capacity(n_modes);

    for mode in 0..n_modes {
        let unfolded = tensor
            .unfold(mode)
            .map_err(|e| TuckerError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

        // Compute SVD: X_(mode) = U Σ Vᵀ
        let (u, s, _vt) = svd(&unfolded.view(), false, None)
            .map_err(|e| TuckerError::SvdError(format!("SVD failed for mode {}: {}", mode, e)))?;

        // Determine rank based on selection strategy
        let rank = match selection {
            TuckerRankSelection::Energy(threshold) => {
                if !(0.0..1.0).contains(&threshold) {
                    return Err(TuckerError::InvalidRanks(format!(
                        "Energy threshold must be in (0, 1), got {}",
                        threshold
                    )));
                }

                // Compute cumulative energy
                let total_energy: T = s.iter().map(|&sigma| sigma * sigma).sum();
                let target_energy = total_energy * NumCast::from(threshold).unwrap();

                let mut cumulative_energy = T::zero();
                let mut rank = 1;

                for (i, &sigma) in s.iter().enumerate() {
                    cumulative_energy += sigma * sigma;
                    rank = i + 1;

                    if cumulative_energy >= target_energy {
                        break;
                    }
                }

                rank.min(shape[mode])
            }
            TuckerRankSelection::Threshold(threshold) => {
                if !(0.0..1.0).contains(&threshold) {
                    return Err(TuckerError::InvalidRanks(format!(
                        "Threshold must be in (0, 1), got {}",
                        threshold
                    )));
                }

                let s_max = s[0];
                let cutoff = s_max * NumCast::from(threshold).unwrap();

                let mut rank = 1;
                for (i, &sigma) in s.iter().enumerate() {
                    if sigma > cutoff {
                        rank = i + 1;
                    } else {
                        break;
                    }
                }

                rank.min(shape[mode])
            }
            TuckerRankSelection::Fixed(ref fixed_ranks) => {
                if fixed_ranks.len() != n_modes {
                    return Err(TuckerError::InvalidRanks(format!(
                        "Expected {} fixed ranks, got {}",
                        n_modes,
                        fixed_ranks.len()
                    )));
                }
                fixed_ranks[mode].min(shape[mode])
            }
        };

        ranks.push(rank);

        // Extract first 'rank' columns of U
        let factor = extract_columns(&u, rank);
        factors.push(factor);
    }

    // Compute core tensor
    let core = compute_core_tensor(tensor, &factors)?;

    Ok(TuckerDecomp {
        core,
        factors,
        error: None,
        iters: 0,
    })
}

/// Compute Tucker-HOSVD decomposition
///
/// One-pass algorithm based on SVD of mode-n unfoldings.
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `ranks` - Target ranks for each mode [R₁, R₂, ..., Rₙ]
///
/// # Returns
///
/// TuckerDecomp containing core tensor and factor matrices
///
/// # Errors
///
/// Returns error if:
/// - Number of ranks doesn't match tensor rank
/// - Any rank exceeds corresponding mode size
/// - SVD computation fails
///
/// # Complexity
///
/// Time: O(N × Imax² × ∏ᵢ Iᵢ) for SVD computations
/// Space: O(Imax² + ∏ᵢ Rᵢ) for unfolding and core tensor
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tucker::tucker_hosvd;
///
/// // Create a 10×10×10 tensor
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
///
/// // Decompose to (5,5,5) core
/// let tucker = tucker_hosvd(&tensor, &[5, 5, 5]).unwrap();
///
/// println!("Core shape: {:?}", tucker.core.shape());
/// ```
pub fn tucker_hosvd<T>(tensor: &DenseND<T>, ranks: &[usize]) -> Result<TuckerDecomp<T>, TuckerError>
where
    T: Float + NumCast + NumAssign + Sum + Send + Sync + ScalarOperand + std::fmt::Debug + 'static,
{
    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Validation
    if ranks.len() != n_modes {
        return Err(TuckerError::InvalidRanks(format!(
            "Expected {} ranks, got {}",
            n_modes,
            ranks.len()
        )));
    }

    for (i, (&rank, &mode_size)) in ranks.iter().zip(shape.iter()).enumerate() {
        if rank > mode_size {
            return Err(TuckerError::InvalidRanks(format!(
                "Rank {} ({}) exceeds mode-{} size ({})",
                i, rank, i, mode_size
            )));
        }
        if rank == 0 {
            return Err(TuckerError::InvalidRanks(format!("Rank {} is zero", i)));
        }
    }

    // Step 1: Compute factor matrices via SVD of mode-n unfoldings
    let mut factors = Vec::with_capacity(n_modes);

    #[allow(clippy::needless_range_loop)]
    for mode in 0..n_modes {
        let rank = ranks[mode];
        let unfolded = tensor
            .unfold(mode)
            .map_err(|e| TuckerError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

        // Compute SVD: X_(mode) = U Σ Vᵀ
        // We only need the first 'rank' left singular vectors
        let (u, _s, _vt) = svd(&unfolded.view(), false, None)
            .map_err(|e| TuckerError::SvdError(format!("SVD failed for mode {}: {}", mode, e)))?;

        // Extract first 'rank' columns of U
        let factor = extract_columns(&u, rank);
        factors.push(factor);
    }

    // Step 2: Compute core tensor: G = X ×₁ U₁ᵀ ×₂ U₂ᵀ ... ×ₙ Uₙᵀ
    let core = compute_core_tensor(tensor, &factors)?;

    Ok(TuckerDecomp {
        core,
        factors,
        error: None,
        iters: 0,
    })
}

/// Compute Tucker-HOOI decomposition
///
/// Iterative refinement of HOSVD using alternating least squares.
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `ranks` - Target ranks for each mode [R₁, R₂, ..., Rₙ]
/// * `max_iters` - Maximum number of iterations
/// * `tol` - Convergence tolerance on relative error change
///
/// # Returns
///
/// TuckerDecomp containing optimized core tensor and factor matrices
///
/// # Examples
///
/// ```no_run
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tucker::tucker_hooi;
///
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
/// let tucker = tucker_hooi(&tensor, &[5, 5, 5], 50, 1e-4).unwrap();
/// ```
pub fn tucker_hooi<T>(
    tensor: &DenseND<T>,
    ranks: &[usize],
    max_iters: usize,
    tol: f64,
) -> Result<TuckerDecomp<T>, TuckerError>
where
    T: Float + NumCast + NumAssign + Sum + Send + Sync + ScalarOperand + std::fmt::Debug + 'static,
{
    // Initialize with HOSVD
    let mut decomp = tucker_hosvd(tensor, ranks)?;
    let n_modes = tensor.rank();

    // Compute initial error
    let mut prev_error = decomp
        .compute_error(tensor)
        .map_err(|e| TuckerError::ShapeMismatch(format!("Error computation failed: {}", e)))?;

    // HOOI iterations
    let mut actual_iters = 0;
    for iter in 0..max_iters {
        actual_iters = iter + 1;

        // Update each factor matrix while keeping others fixed
        #[allow(clippy::needless_range_loop)]
        for mode in 0..n_modes {
            // Compute Y = X ×₁ U₁ᵀ ... ×ₘ₋₁ Uₘ₋₁ᵀ ×ₘ₊₁ Uₘ₊₁ᵀ ... ×ₙ Uₙᵀ
            let y = compute_mode_unfolding_contraction(tensor, &decomp.factors, mode)?;

            // Unfold Y along mode and compute SVD
            let y_unfolded = y
                .unfold(mode)
                .map_err(|e| TuckerError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

            let (u, _s, _vt) = svd(&y_unfolded.view(), false, None)
                .map_err(|e| TuckerError::SvdError(format!("SVD failed: {}", e)))?;

            decomp.factors[mode] = extract_columns(&u, ranks[mode]);
        }

        // Recompute core tensor
        decomp.core = compute_core_tensor(tensor, &decomp.factors)?;

        // Check convergence
        let error = decomp
            .compute_error(tensor)
            .map_err(|e| TuckerError::ShapeMismatch(format!("Error computation failed: {}", e)))?;

        let error_change = (prev_error - error).abs() / prev_error;
        if error_change < NumCast::from(tol).unwrap() {
            break;
        }

        prev_error = error;
    }

    decomp.iters = actual_iters;
    Ok(decomp)
}

/// Extract first k columns from a matrix
pub(crate) fn extract_columns<T>(matrix: &Array2<T>, k: usize) -> Array2<T>
where
    T: Clone + Float,
{
    let rows = matrix.shape()[0];
    let k = k.min(matrix.shape()[1]);

    let mut result = Array2::<T>::zeros((rows, k));
    for i in 0..rows {
        for j in 0..k {
            result[[i, j]] = matrix[[i, j]];
        }
    }
    result
}

/// Compute core tensor: G = X ×₁ U₁ᵀ ×₂ U₂ᵀ ... ×ₙ Uₙᵀ
pub(crate) fn compute_core_tensor<T>(
    tensor: &DenseND<T>,
    factors: &[Array2<T>],
) -> Result<DenseND<T>, TuckerError>
where
    T: Float + NumCast,
{
    let mut result = tensor.clone();

    // Apply transposed factor matrices in sequence
    for (mode, factor) in factors.iter().enumerate() {
        let result_view = result.view();

        // Transpose factor matrix (Uᵀ)
        let factor_t = transpose_matrix(factor);

        let contracted = nmode_product(&result_view, &factor_t.view(), mode)
            .map_err(|e| TuckerError::ShapeMismatch(format!("N-mode product failed: {}", e)))?;

        result = DenseND::from_array(contracted);
    }

    Ok(result)
}

/// Transpose a matrix
fn transpose_matrix<T>(matrix: &Array2<T>) -> Array2<T>
where
    T: Clone + Float,
{
    let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
    let mut result = Array2::<T>::zeros((cols, rows));

    for i in 0..rows {
        for j in 0..cols {
            result[[j, i]] = matrix[[i, j]];
        }
    }
    result
}

/// Compute Y = X ×₁ U₁ᵀ ... ×ₘ₋₁ Uₘ₋₁ᵀ ×ₘ₊₁ Uₘ₊₁ᵀ ... ×ₙ Uₙᵀ (skip mode m)
pub(crate) fn compute_mode_unfolding_contraction<T>(
    tensor: &DenseND<T>,
    factors: &[Array2<T>],
    skip_mode: usize,
) -> Result<DenseND<T>, TuckerError>
where
    T: Float + NumCast,
{
    let mut result = tensor.clone();

    for (mode, factor) in factors.iter().enumerate() {
        if mode == skip_mode {
            continue;
        }

        let result_view = result.view();
        let factor_t = transpose_matrix(factor);

        // nmode_product doesn't remove dimensions, it just changes their size
        // So mode indices don't shift - we always contract along the same mode index
        let contracted = nmode_product(&result_view, &factor_t.view(), mode)
            .map_err(|e| TuckerError::ShapeMismatch(format!("Contraction failed: {}", e)))?;

        result = DenseND::from_array(contracted);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tucker_hosvd_basic() {
        // Small tensor for quick test
        let tensor = DenseND::<f64>::ones(&[4, 5, 6]);
        let result = tucker_hosvd(&tensor, &[2, 3, 3]);

        assert!(result.is_ok());
        let tucker = result.unwrap();

        assert_eq!(tucker.core.shape(), &[2, 3, 3]);
        assert_eq!(tucker.factors.len(), 3);
        assert_eq!(tucker.factors[0].shape(), &[4, 2]);
        assert_eq!(tucker.factors[1].shape(), &[5, 3]);
        assert_eq!(tucker.factors[2].shape(), &[6, 3]);
    }

    #[test]
    fn test_tucker_reconstruction() {
        let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
        let mut tucker = tucker_hosvd(&tensor, &[2, 2, 2]).unwrap();

        let reconstructed = tucker.reconstruct();
        assert!(reconstructed.is_ok());

        let error = tucker.compute_error(&tensor);
        assert!(error.is_ok());
        let err_val = error.unwrap();
        assert!((0.0..=1.0).contains(&err_val));
    }

    #[test]
    fn test_extract_columns() {
        use scirs2_core::ndarray_ext::array;

        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let extracted = extract_columns(&matrix, 2);

        assert_eq!(extracted.shape(), &[2, 2]);
        assert_eq!(extracted[[0, 0]], 1.0);
        assert_eq!(extracted[[0, 1]], 2.0);
        assert_eq!(extracted[[1, 0]], 4.0);
        assert_eq!(extracted[[1, 1]], 5.0);
    }

    // ========================================================================
    // Automatic Rank Selection Tests
    // ========================================================================

    #[test]
    fn test_tucker_auto_rank_energy() {
        use super::TuckerRankSelection;

        // Create a low-rank tensor (approximately)
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);

        // Auto-select ranks based on 90% energy preservation
        let tucker = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.9)).unwrap();

        // Check that decomposition succeeded
        assert_eq!(tucker.factors.len(), 3);

        // Verify reconstruction quality
        let mut tucker_mut = tucker;
        let error = tucker_mut.compute_error(&tensor).unwrap();

        // With 90% energy, error should be reasonable
        assert!(error < 0.5, "Auto-rank error too large: {}", error);

        // Ranks should be less than full rank
        for factor in &tucker_mut.factors {
            assert!(factor.ncols() <= 10);
        }
    }

    #[test]
    fn test_tucker_auto_rank_threshold() {
        use super::TuckerRankSelection;

        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);

        // Auto-select ranks with singular value threshold
        let tucker = tucker_hosvd_auto(&tensor, TuckerRankSelection::Threshold(0.1)).unwrap();

        // Check that decomposition succeeded
        assert_eq!(tucker.factors.len(), 3);

        // All ranks should be at least 1
        for factor in &tucker.factors {
            assert!(
                factor.ncols() >= 1,
                "Rank should be at least 1, got {}",
                factor.ncols()
            );
        }
    }

    #[test]
    fn test_tucker_auto_rank_fixed() {
        use super::TuckerRankSelection;

        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let fixed_ranks = vec![5, 6, 7];

        // Auto-select with fixed ranks (should behave like tucker_hosvd)
        let tucker =
            tucker_hosvd_auto(&tensor, TuckerRankSelection::Fixed(fixed_ranks.clone())).unwrap();

        // Check that exact ranks are used
        for (i, factor) in tucker.factors.iter().enumerate() {
            assert_eq!(
                factor.ncols(),
                fixed_ranks[i],
                "Mode {} rank should be {}",
                i,
                fixed_ranks[i]
            );
        }
    }

    #[test]
    fn test_tucker_auto_rank_invalid_params() {
        use super::TuckerRankSelection;

        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);

        // Invalid energy threshold (> 1)
        let result = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(1.5));
        assert!(result.is_err());

        // Invalid singular value threshold (< 0)
        let result = tucker_hosvd_auto(&tensor, TuckerRankSelection::Threshold(-0.1));
        assert!(result.is_err());

        // Wrong number of fixed ranks
        let result = tucker_hosvd_auto(&tensor, TuckerRankSelection::Fixed(vec![5, 6])); // Need 3
        assert!(result.is_err());
    }

    #[test]
    fn test_tucker_auto_rank_comparison() {
        use super::TuckerRankSelection;

        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);

        // Compare energy-based vs fixed rank
        let tucker_auto = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.95)).unwrap();
        let tucker_fixed = tucker_hosvd(&tensor, &[3, 3, 3]).unwrap();

        // Both should produce valid decompositions
        assert!(tucker_auto.reconstruct().is_ok());
        assert!(tucker_fixed.reconstruct().is_ok());

        // Auto-selected ranks might be different from fixed
        let auto_ranks: Vec<_> = tucker_auto.factors.iter().map(|f| f.ncols()).collect();
        println!("Auto-selected ranks: {:?}", auto_ranks);
        println!("Fixed ranks: [3, 3, 3]");
    }

}
