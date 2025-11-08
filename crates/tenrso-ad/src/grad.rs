//! Gradient rules for tensor decompositions
//!
//! This module implements gradient rules for CP-ALS, Tucker-HOOI, and TT-SVD
//! decompositions. These gradients enable end-to-end differentiation through
//! tensor decomposition layers.
//!
//! # Approach
//!
//! For decomposition operations, we provide gradients for the reconstruction
//! operator rather than the decomposition algorithm itself. This is because:
//!
//! 1. Decomposition algorithms (ALS, HOOI) are iterative and stateful
//! 2. Their gradients would require unrolling the entire optimization
//! 3. In practice, we usually only need gradients w.r.t. reconstructed tensors
//!
//! # Example
//!
//! ```rust,ignore
//! // Forward: CP decomposition
//! let cp = cp_als(&tensor, rank, max_iters, tol)?;
//! let reconstructed = cp.reconstruct(tensor.shape())?;
//!
//! // Backward: gradient w.r.t. factors
//! let grad_ctx = CpReconstructionGrad::new(cp.factors.clone(), cp.weights.clone());
//! let factor_grads = grad_ctx.compute_factor_gradients(&grad_reconstructed)?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{s, Array1, Array2};
use scirs2_core::numeric::Float;
use tenrso_core::DenseND;

/// Gradient context for CP reconstruction
///
/// Computes gradients of the reconstructed tensor w.r.t. CP factors.
///
/// For reconstruction: `X = Σᵣ λᵣ (u₁ᵣ ⊗ u₂ᵣ ⊗ ... ⊗ uₙᵣ)`
///
/// The gradient w.r.t. factor matrix Uₙ involves contracting the gradient tensor
/// with all other factor matrices via the Khatri-Rao product.
pub struct CpReconstructionGrad<T>
where
    T: Float,
{
    /// CP factor matrices from forward pass
    pub factors: Vec<Array2<T>>,
    /// Component weights from forward pass
    pub weights: Option<Array1<T>>,
}

impl<T> CpReconstructionGrad<T>
where
    T: Float + 'static,
{
    /// Create a new CP reconstruction gradient context
    pub fn new(factors: Vec<Array2<T>>, weights: Option<Array1<T>>) -> Self {
        Self { factors, weights }
    }

    /// Compute gradients w.r.t. all factor matrices
    ///
    /// Given `grad_output` (gradient w.r.t. reconstructed tensor), computes
    /// gradients w.r.t. each factor matrix.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient w.r.t. the reconstructed tensor (∂L/∂X)
    ///
    /// # Returns
    ///
    /// Vector of gradients, one for each factor matrix: `[∂L/∂U₁, ∂L/∂U₂, ...]`
    ///
    /// # Algorithm
    ///
    /// For each mode n:
    /// 1. Unfold gradient tensor along mode n: `G₍ₙ₎ ∈ ℝ^(Iₙ × ∏ᵢ≠ₙ Iᵢ)`
    /// 2. Compute Khatri-Rao product of all factors except n: `V = U₍₁₎ ⊙ ... ⊙ U₍ₙ₋₁₎ ⊙ U₍ₙ₊₁₎ ⊙ ... ⊙ U₍ₙ₎`
    /// 3. Gradient: `∂L/∂Uₙ = G₍ₙ₎ × V`
    ///
    /// This is essentially the MTTKRP operation used in forward CP-ALS.
    pub fn compute_factor_gradients(&self, grad_output: &DenseND<T>) -> Result<Vec<Array2<T>>> {
        let n_modes = self.factors.len();

        // Verify gradient shape matches factor dimensions
        if grad_output.shape().len() != n_modes {
            return Err(anyhow!(
                "Gradient rank {} doesn't match number of factors {}",
                grad_output.shape().len(),
                n_modes
            ));
        }

        let mut factor_grads = Vec::with_capacity(n_modes);

        for mode in 0..n_modes {
            // Unfold gradient tensor along current mode
            let grad_unfolded = grad_output.unfold(mode)?;

            // Compute Khatri-Rao product of all factors except current mode
            // This is the "V" matrix in MTTKRP
            let kr_factors = self.khatri_rao_except(mode)?;

            // Gradient is: grad_unfolded @ kr_factors
            // Shape: (I_mode, rank)
            let grad_factor: Array2<T> = grad_unfolded.dot(&kr_factors);

            // Apply weight scaling if weights are present
            if let Some(ref weights) = self.weights {
                let weighted_grad = &grad_factor
                    * &weights
                        .view()
                        .insert_axis(scirs2_core::ndarray_ext::Axis(0));
                factor_grads.push(weighted_grad);
            } else {
                factor_grads.push(grad_factor);
            }
        }

        Ok(factor_grads)
    }

    /// Compute Khatri-Rao product of all factors except the given mode
    ///
    /// For CP gradient computation, we need: V = U₁ ⊙ U₂ ⊙ ... ⊙ Uₙ₋₁ ⊙ Uₙ₊₁ ⊙ ... ⊙ Uₘ
    /// where mode n is excluded.
    fn khatri_rao_except(&self, except_mode: usize) -> Result<Array2<T>> {
        // Collect all factors except the specified mode
        let factors_to_multiply: Vec<_> = self
            .factors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != except_mode)
            .map(|(_, f)| f)
            .collect();

        if factors_to_multiply.is_empty() {
            return Err(anyhow!("Need at least one factor for Khatri-Rao product"));
        }

        // Start with the first factor
        let mut result = factors_to_multiply[0].clone();

        // Compute Khatri-Rao product iteratively
        for factor in &factors_to_multiply[1..] {
            result = khatri_rao(&result.view(), &factor.view())?;
        }

        Ok(result)
    }
}

/// Compute Khatri-Rao product of two matrices
///
/// The Khatri-Rao product is the column-wise Kronecker product:
/// (A ⊙ B)[:, r] = A[:, r] ⊗ B[:, r]
///
/// # Arguments
///
/// * `a` - First matrix (m × r)
/// * `b` - Second matrix (n × r)
///
/// # Returns
///
/// Khatri-Rao product (m*n × r)
fn khatri_rao<T>(
    a: &scirs2_core::ndarray_ext::ArrayView2<T>,
    b: &scirs2_core::ndarray_ext::ArrayView2<T>,
) -> Result<Array2<T>>
where
    T: Float,
{
    let (m, r_a) = (a.shape()[0], a.shape()[1]);
    let (n, r_b) = (b.shape()[0], b.shape()[1]);

    if r_a != r_b {
        return Err(anyhow!(
            "Khatri-Rao product requires same number of columns: {} vs {}",
            r_a,
            r_b
        ));
    }

    let r = r_a;
    let mut result = Array2::zeros((m * n, r));

    // Compute column by column
    for col in 0..r {
        let a_col = a.slice(s![.., col]);
        let b_col = b.slice(s![.., col]);

        // Kronecker product of columns
        for i in 0..m {
            for j in 0..n {
                result[[i * n + j, col]] = a_col[i] * b_col[j];
            }
        }
    }

    Ok(result)
}

/// Gradient context for Tucker reconstruction
///
/// Computes gradients of the reconstructed tensor w.r.t. Tucker core and factors.
///
/// For reconstruction: `X = G ×₁ U₁ ×₂ U₂ ×₃ ... ×ₙ Uₙ`
/// where G is the core tensor and Uᵢ are factor matrices.
pub struct TuckerReconstructionGrad<T>
where
    T: Float,
{
    /// Tucker core tensor from forward pass
    pub core: DenseND<T>,
    /// Tucker factor matrices from forward pass
    pub factors: Vec<Array2<T>>,
}

impl<T> TuckerReconstructionGrad<T>
where
    T: Float + 'static,
{
    /// Create a new Tucker reconstruction gradient context
    pub fn new(core: DenseND<T>, factors: Vec<Array2<T>>) -> Self {
        Self { core, factors }
    }

    /// Compute gradient w.r.t. the core tensor
    ///
    /// For `X = G ×₁ U₁ ×₂ U₂ ... ×ₙ Uₙ`, the gradient is:
    /// `∂L/∂G = ∂L/∂X ×₁ U₁ᵀ ×₂ U₂ᵀ ... ×ₙ Uₙᵀ`
    pub fn compute_core_gradient(&self, grad_output: &DenseND<T>) -> Result<DenseND<T>> {
        let mut grad_core = grad_output.clone();

        // Contract with transpose of each factor matrix
        for (mode, factor) in self.factors.iter().enumerate() {
            // Mode-n product with Uᵀ
            grad_core = mode_n_product(&grad_core, &factor.t().to_owned(), mode)?;
        }

        Ok(grad_core)
    }

    /// Compute gradients w.r.t. factor matrices
    ///
    /// For factor matrix Uₙ, the gradient is:
    /// `∂L/∂Uₙ = (∂L/∂X)₍ₙ₎ × (G ×₁ U₁ ... ×ₙ₋₁ Uₙ₋₁ ×ₙ₊₁ Uₙ₊₁ ... ×ₘ Uₘ)₍ₙ₎ᵀ`
    pub fn compute_factor_gradients(&self, grad_output: &DenseND<T>) -> Result<Vec<Array2<T>>> {
        let n_modes = self.factors.len();
        let mut factor_grads = Vec::with_capacity(n_modes);

        for mode in 0..n_modes {
            // Unfold gradient tensor along mode n
            let grad_unfolded = grad_output.unfold(mode)?;

            // Contract core with all factors except mode n
            let mut partial_recon = self.core.clone();
            for (m, factor) in self.factors.iter().enumerate() {
                if m != mode {
                    partial_recon = mode_n_product(&partial_recon, factor, m)?;
                }
            }

            // Unfold partial reconstruction
            let partial_unfolded = partial_recon.unfold(mode)?;

            // Gradient: grad_unfolded @ partial_unfolded^T
            let grad_factor: Array2<T> = grad_unfolded.dot(&partial_unfolded.t());

            factor_grads.push(grad_factor);
        }

        Ok(factor_grads)
    }
}

/// Mode-n product of a tensor with a matrix
///
/// Computes `X ×ₙ U` where X is a tensor and U is a matrix.
/// This contracts mode n of X with the rows of U.
fn mode_n_product<T>(tensor: &DenseND<T>, matrix: &Array2<T>, mode: usize) -> Result<DenseND<T>>
where
    T: Float + 'static,
{
    // Unfold tensor along mode n
    let unfolded = tensor.unfold(mode)?;

    // Matrix multiply: matrix @ unfolded
    let result_unfolded: Array2<T> = matrix.dot(&unfolded);

    // Get output shape
    let mut output_shape = tensor.shape().to_vec();
    output_shape[mode] = matrix.shape()[0];

    // Fold back to tensor
    DenseND::fold(&result_unfolded, &output_shape, mode)
}

/// Gradient context for Tensor Train (TT) reconstruction
///
/// TT format: X(i₁, i₂, ..., iₙ) = G₁(i₁) × G₂(i₂) × ... × Gₙ(iₙ)
/// where each Gₖ(iₖ) is a matrix of shape (rₖ₋₁ × rₖ)
pub struct TtReconstructionGrad<T>
where
    T: Float,
{
    /// TT cores from forward pass
    pub cores: Vec<DenseND<T>>,
}

impl<T> TtReconstructionGrad<T>
where
    T: Float,
{
    /// Create a new TT reconstruction gradient context
    pub fn new(cores: Vec<DenseND<T>>) -> Self {
        Self { cores }
    }

    /// Compute gradients w.r.t. TT cores
    ///
    /// This is complex and requires forward/backward passes through the TT chain.
    /// For now, we provide a placeholder that returns an error.
    ///
    /// TODO: Implement full TT gradient computation
    pub fn compute_core_gradients(&self, _grad_output: &DenseND<T>) -> Result<Vec<DenseND<T>>> {
        Err(anyhow!(
            "TT gradient computation not yet implemented. \
             This requires forward/backward passes through the tensor train."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_khatri_rao_basic() {
        use scirs2_core::ndarray_ext::array;

        let a = array![[1.0, 2.0], [3.0, 4.0]]; // 2x2
        let b = array![[5.0, 6.0], [7.0, 8.0]]; // 2x2

        let kr = khatri_rao(&a.view(), &b.view()).unwrap();

        // Result should be 4x2
        assert_eq!(kr.shape(), &[4, 2]);

        // First column: [1*5, 1*7, 3*5, 3*7] = [5, 7, 15, 21]
        assert_eq!(kr[[0, 0]], 5.0);
        assert_eq!(kr[[1, 0]], 7.0);
        assert_eq!(kr[[2, 0]], 15.0);
        assert_eq!(kr[[3, 0]], 21.0);
    }

    #[test]
    fn test_cp_reconstruction_grad_shapes() {
        // Create simple CP decomposition: 3x4x5 tensor, rank 2
        let factors = vec![
            Array2::<f64>::zeros((3, 2)),
            Array2::<f64>::zeros((4, 2)),
            Array2::<f64>::zeros((5, 2)),
        ];

        let grad_ctx = CpReconstructionGrad::new(factors.clone(), None);

        let grad_output = DenseND::<f64>::ones(&[3, 4, 5]);
        let factor_grads = grad_ctx.compute_factor_gradients(&grad_output).unwrap();

        assert_eq!(factor_grads.len(), 3);
        assert_eq!(factor_grads[0].shape(), &[3, 2]);
        assert_eq!(factor_grads[1].shape(), &[4, 2]);
        assert_eq!(factor_grads[2].shape(), &[5, 2]);
    }
}
