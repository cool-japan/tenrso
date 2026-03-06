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
/// For reconstruction: `X = Sigma_r lambda_r (u_1r x u_2r x ... x u_nr)`
///
/// The gradient w.r.t. factor matrix Un involves contracting the gradient tensor
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
    /// * `grad_output` - Gradient w.r.t. the reconstructed tensor (dL/dX)
    ///
    /// # Returns
    ///
    /// Vector of gradients, one for each factor matrix: `[dL/dU1, dL/dU2, ...]`
    ///
    /// # Algorithm
    ///
    /// For each mode n:
    /// 1. Unfold gradient tensor along mode n: `G_(n) in R^(In x prod(Ii, i!=n))`
    /// 2. Compute Khatri-Rao product of all factors except n
    /// 3. Gradient: `dL/dUn = G_(n) x V`
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
    /// For CP gradient computation, we need: V = U1 odot U2 odot ... odot U(n-1) odot U(n+1) odot ... odot Um
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
/// (A odot B)[:, r] = A[:, r] kron B[:, r]
///
/// # Arguments
///
/// * `a` - First matrix (m x r)
/// * `b` - Second matrix (n x r)
///
/// # Returns
///
/// Khatri-Rao product (m*n x r)
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
/// For reconstruction: `X = G x1 U1 x2 U2 x3 ... xn Un`
/// where G is the core tensor and Ui are factor matrices.
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
    /// For `X = G x1 U1 x2 U2 ... xn Un`, the gradient is:
    /// `dL/dG = dL/dX x1 U1^T x2 U2^T ... xn Un^T`
    pub fn compute_core_gradient(&self, grad_output: &DenseND<T>) -> Result<DenseND<T>> {
        let mut grad_core = grad_output.clone();

        // Contract with transpose of each factor matrix
        for (mode, factor) in self.factors.iter().enumerate() {
            // Mode-n product with U^T
            grad_core = mode_n_product(&grad_core, &factor.t().to_owned(), mode)?;
        }

        Ok(grad_core)
    }

    /// Compute gradients w.r.t. factor matrices
    ///
    /// For factor matrix Un, the gradient is:
    /// `dL/dUn = (dL/dX)_(n) x (G x1 U1 ... x(n-1) U(n-1) x(n+1) U(n+1) ... xm Um)_(n)^T`
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
/// Computes `X xn U` where X is a tensor and U is a matrix.
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
/// Computes gradients of the reconstructed tensor w.r.t. TT cores using
/// the chain rule applied through the sequential core contractions.
///
/// # TT Format
///
/// A tensor in TT format is represented as:
/// ```text
/// X(i1, i2, ..., iN) = sum_{r0,...,rN} G1[r0,i1,r1] * G2[r1,i2,r2] * ... * GN[rN-1,iN,rN]
/// ```
/// where each core Gk has shape (r_{k-1}, n_k, r_k) with r0 = rN = 1.
///
/// # Gradient Algorithm
///
/// For each core k, the gradient dL/dGk is computed by contracting dL/dX with
/// left partial products (cores 1..k-1) and right partial products (cores k+1..N).
///
/// Specifically, for core k with shape (r_{k-1}, n_k, r_k):
/// ```text
/// dL/dGk[alpha, ik, beta] = sum_{i1,...,i_{k-1},i_{k+1},...,iN}
///     dL/dX(i1,...,iN) * L_{k-1}[alpha, i1,...,i_{k-1}] * R_{k+1}[beta, i_{k+1},...,iN]
/// ```
/// where L and R are partial products from left and right respectively.
pub struct TtReconstructionGrad<T>
where
    T: Float,
{
    /// TT cores from forward pass, each with shape (r_{k-1}, n_k, r_k)
    pub cores: Vec<DenseND<T>>,
}

impl<T> TtReconstructionGrad<T>
where
    T: Float + 'static,
{
    /// Create a new TT reconstruction gradient context
    pub fn new(cores: Vec<DenseND<T>>) -> Self {
        Self { cores }
    }

    /// Reconstruct the full tensor from TT cores
    ///
    /// This is needed for gradient verification and testing.
    /// Computes X(i1,...,iN) by contracting all cores sequentially.
    pub fn reconstruct(&self) -> Result<DenseND<T>> {
        let n_cores = self.cores.len();
        if n_cores == 0 {
            return Err(anyhow!("No TT cores provided"));
        }

        let output_shape: Vec<usize> = self.cores.iter().map(|core| core.shape()[1]).collect();

        let total_elements: usize = output_shape.iter().product();
        let mut result_data = vec![T::zero(); total_elements];

        // Iterate over all multi-indices of the output tensor
        let mut multi_idx = vec![0usize; n_cores];
        for (flat_idx, result_elem) in result_data.iter_mut().enumerate() {
            // Convert flat index to multi-index
            let mut remaining = flat_idx;
            for d in (0..n_cores).rev() {
                multi_idx[d] = remaining % output_shape[d];
                remaining /= output_shape[d];
            }

            // Compute tensor element: product of matrix slices
            // Start with G1[:,i1,:] which is (r0 x r1) = (1 x r1)
            let first_core = &self.cores[0];
            let r0 = first_core.shape()[0];
            let r1_first = first_core.shape()[2];

            // Extract the slice G1[:,i1,:] as a matrix
            let mut current = Array2::<T>::zeros((r0, r1_first));
            for a in 0..r0 {
                for b in 0..r1_first {
                    current[[a, b]] = *first_core
                        .get(&[a, multi_idx[0], b])
                        .ok_or_else(|| anyhow!("Index error in TT reconstruction"))?;
                }
            }

            // Multiply through remaining cores
            for (k, core_k) in self.cores.iter().enumerate().skip(1) {
                let rk_prev = core_k.shape()[0];
                let rk = core_k.shape()[2];

                // Extract slice Gk[:,ik,:] as (rk_prev x rk) matrix
                let mut slice_k = Array2::<T>::zeros((rk_prev, rk));
                for a in 0..rk_prev {
                    for b in 0..rk {
                        slice_k[[a, b]] = *core_k
                            .get(&[a, multi_idx[k], b])
                            .ok_or_else(|| anyhow!("Index error in TT reconstruction"))?;
                    }
                }

                // current = current @ slice_k
                current = current.dot(&slice_k);
            }

            // Result should be 1x1
            *result_elem = current[[0, 0]];
        }

        DenseND::from_vec(result_data, &output_shape)
    }

    /// Compute gradients w.r.t. TT cores
    ///
    /// Implements gradient computation through the tensor train using
    /// element-wise contraction with left and right partial products.
    ///
    /// # Algorithm
    ///
    /// For each element X(i1,...,iN), the value is computed as:
    /// ```text
    /// X(i1,...,iN) = G1[:,i1,:] * G2[:,i2,:] * ... * GN[:,iN,:]
    /// ```
    ///
    /// The gradient for core k at slice ik is:
    /// ```text
    /// dL/dGk[:,ik,:] += dL/dX(i1,...,iN) * L_{k-1}^T * R_{k+1}
    /// ```
    /// where:
    /// - L\_{k-1\} = G1\[:,i1,:\] * ... * G(k-1)\[:,i(k-1),:\] is a (1 x r\_{k-1\}) row vector
    /// - R\_{k+1\} = G(k+1)\[:,i(k+1),:\] * ... * GN\[:,iN,:\] is a (r_k x 1) column vector
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient w.r.t. reconstructed tensor (dL/dX)
    ///
    /// # Returns
    ///
    /// Vector of gradients for each TT core, preserving original core shapes
    pub fn compute_core_gradients(&self, grad_output: &DenseND<T>) -> Result<Vec<DenseND<T>>> {
        let n_cores = self.cores.len();
        if n_cores == 0 {
            return Err(anyhow!("No TT cores provided"));
        }

        // Verify shapes match
        let output_shape: Vec<usize> = self
            .cores
            .iter()
            .map(|core| core.shape()[1]) // Middle dimension is the mode size
            .collect();

        if grad_output.shape() != output_shape.as_slice() {
            return Err(anyhow!(
                "Shape mismatch: grad_output {:?} vs expected {:?}",
                grad_output.shape(),
                output_shape
            ));
        }

        // Special case: single core
        if n_cores == 1 {
            // For a single core G with shape (1, n, 1), the reconstruction is just
            // X[i] = G[0, i, 0], so dL/dG = dL/dX reshaped to (1, n, 1)
            let core_shape = self.cores[0].shape();
            let n = core_shape[1];
            let mut grad_core = DenseND::zeros(core_shape);
            for i in 0..n {
                let grad_val = *grad_output
                    .get(&[i])
                    .ok_or_else(|| anyhow!("Index error in single core gradient"))?;
                *grad_core
                    .get_mut(&[0, i, 0])
                    .ok_or_else(|| anyhow!("Index error in single core gradient"))? = grad_val;
            }
            return Ok(vec![grad_core]);
        }

        // Initialize gradient accumulators (zeros with same shapes as cores)
        let mut core_grads: Vec<DenseND<T>> = self
            .cores
            .iter()
            .map(|c| DenseND::zeros(c.shape()))
            .collect();

        // Total number of elements in the output tensor
        let total_elements: usize = output_shape.iter().product();

        // Iterate over all multi-indices of the output tensor
        let mut multi_idx = vec![0usize; n_cores];

        for flat_idx in 0..total_elements {
            // Convert flat index to multi-index
            let mut remaining = flat_idx;
            for d in (0..n_cores).rev() {
                multi_idx[d] = remaining % output_shape[d];
                remaining /= output_shape[d];
            }

            // Get the gradient value at this position
            let grad_val = *grad_output
                .get(&multi_idx)
                .ok_or_else(|| anyhow!("Index error accessing grad_output"))?;

            // Skip if gradient is zero (optimization)
            if grad_val == T::zero() {
                continue;
            }

            // Compute left partial products: L[k] = G1[:,i1,:] * ... * Gk[:,ik,:]
            // L[k] has shape (1 x r_k) -- a row vector
            let left_products = self.compute_left_chain(&multi_idx)?;

            // Compute right partial products: R[k] = Gk[:,ik,:] * ... * GN[:,iN,:]
            // R[k] has shape (r_{k-1} x 1) -- a column vector
            let right_products = self.compute_right_chain(&multi_idx)?;

            // Compute gradient contribution for each core
            for k in 0..n_cores {
                let core_shape = self.cores[k].shape();
                let rk_prev = core_shape[0];
                let rk = core_shape[2];
                let ik = multi_idx[k];

                // For core k, the gradient contribution at slice ik is:
                // dGk[:,ik,:] += grad_val * L_{k-1}^T @ R_{k+1}
                //
                // L_{k-1} is (1 x r_{k-1}) => L_{k-1}^T is (r_{k-1} x 1)
                // R_{k+1} is (r_k x 1) => R_{k+1}^T is (1 x r_k)
                //
                // So the outer product L_{k-1}^T * R_{k+1}^T = (r_{k-1} x r_k)

                // Get left context (everything to the left of core k)
                let left_vec = if k == 0 {
                    // No cores to the left, L = identity (1x1 with value 1)
                    Array1::from_vec(vec![T::one()])
                } else {
                    // left_products[k-1] is the row vector after multiplying cores 0..k-1
                    left_products[k - 1].clone()
                };

                // Get right context (everything to the right of core k)
                let right_vec = if k == n_cores - 1 {
                    // No cores to the right, R = identity (1x1 with value 1)
                    Array1::from_vec(vec![T::one()])
                } else {
                    // right_products[k+1] is the column vector after multiplying cores k+1..N-1
                    right_products[k + 1].clone()
                };

                // Accumulate outer product scaled by grad_val into dGk[:,ik,:]
                for alpha in 0..rk_prev {
                    for beta in 0..rk {
                        let contribution = grad_val * left_vec[alpha] * right_vec[beta];
                        let current = *core_grads[k]
                            .get(&[alpha, ik, beta])
                            .ok_or_else(|| anyhow!("Index error in gradient accumulation"))?;
                        *core_grads[k]
                            .get_mut(&[alpha, ik, beta])
                            .ok_or_else(|| anyhow!("Index error in gradient accumulation"))? =
                            current + contribution;
                    }
                }
            }
        }

        Ok(core_grads)
    }

    /// Compute left partial products for a given set of mode indices
    ///
    /// Returns a vector where entry k contains the row vector resulting from
    /// multiplying cores 0 through k with the given indices:
    /// L[k] = G1[:,i1,:] * G2[:,i2,:] * ... * Gk[:,ik,:]
    ///
    /// L[k] is stored as Array1 of length r_{k} (the right TT-rank of core k).
    fn compute_left_chain(&self, indices: &[usize]) -> Result<Vec<Array1<T>>> {
        let n_cores = self.cores.len();
        let mut left_products = Vec::with_capacity(n_cores);

        // Start with core 0: extract slice G0[:,i0,:] which is (r0 x r1) = (1 x r1)
        let first_core = &self.cores[0];
        let r1 = first_core.shape()[2];
        let mut current = Array1::<T>::zeros(r1);
        for b in 0..r1 {
            current[b] = *first_core
                .get(&[0, indices[0], b])
                .ok_or_else(|| anyhow!("Index error in left chain"))?;
        }
        left_products.push(current.clone());

        // Multiply through remaining cores
        for (k, core_k) in self.cores.iter().enumerate().skip(1) {
            let rk = core_k.shape()[2];
            let rk_prev = core_k.shape()[0];
            let ik = indices[k];

            // Extract Gk[:,ik,:] as (rk_prev x rk) and multiply: current (rk_prev,) @ slice
            let mut next = Array1::<T>::zeros(rk);
            for beta in 0..rk {
                let mut sum = T::zero();
                for alpha in 0..rk_prev {
                    let gval = *core_k
                        .get(&[alpha, ik, beta])
                        .ok_or_else(|| anyhow!("Index error in left chain"))?;
                    sum = sum + current[alpha] * gval;
                }
                next[beta] = sum;
            }

            current = next;
            left_products.push(current.clone());
        }

        Ok(left_products)
    }

    /// Compute right partial products for a given set of mode indices
    ///
    /// Returns a vector where entry k contains the column vector resulting from
    /// multiplying cores k through N-1 with the given indices:
    /// R[k] = Gk[:,ik,:] * G(k+1)[:,i(k+1),:] * ... * GN[:,iN,:]
    ///
    /// R[k] is stored as Array1 of length r_{k-1} (the left TT-rank of core k).
    fn compute_right_chain(&self, indices: &[usize]) -> Result<Vec<Array1<T>>> {
        let n_cores = self.cores.len();
        let mut right_products = vec![Array1::<T>::zeros(0); n_cores];

        // Start from the last core: GN[:,iN,:] is (r_{N-1} x 1)
        let last_core = &self.cores[n_cores - 1];
        let rn_prev = last_core.shape()[0];
        let mut current = Array1::<T>::zeros(rn_prev);
        for a in 0..rn_prev {
            current[a] = *last_core
                .get(&[a, indices[n_cores - 1], 0])
                .ok_or_else(|| anyhow!("Index error in right chain"))?;
        }
        right_products[n_cores - 1] = current.clone();

        // Multiply through cores from right to left
        for k in (0..n_cores - 1).rev() {
            let core_k = &self.cores[k];
            let rk_prev = core_k.shape()[0];
            let rk = core_k.shape()[2];
            let ik = indices[k];

            // Extract Gk[:,ik,:] as (rk_prev x rk) and multiply: slice @ current
            let mut next = Array1::<T>::zeros(rk_prev);
            for alpha in 0..rk_prev {
                let mut sum = T::zero();
                for beta in 0..rk {
                    let gval = *core_k
                        .get(&[alpha, ik, beta])
                        .ok_or_else(|| anyhow!("Index error in right chain"))?;
                    sum = sum + gval * current[beta];
                }
                next[alpha] = sum;
            }

            current = next;
            right_products[k] = current.clone();
        }

        Ok(right_products)
    }
}

/// Reconstruct a full tensor from TT cores (convenience function)
///
/// This is useful for testing and verification of TT gradient correctness.
pub fn tt_reconstruct<T: Float + 'static>(cores: &[DenseND<T>]) -> Result<DenseND<T>> {
    let ctx = TtReconstructionGrad::new(cores.to_vec());
    ctx.reconstruct()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_khatri_rao_basic() {
        use scirs2_core::ndarray_ext::array;

        let a = array![[1.0, 2.0], [3.0, 4.0]]; // 2x2
        let b = array![[5.0, 6.0], [7.0, 8.0]]; // 2x2

        let kr = khatri_rao(&a.view(), &b.view()).expect("khatri_rao failed");

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

        let grad_ctx = CpReconstructionGrad::new(factors, None);

        let grad_output = DenseND::<f64>::ones(&[3, 4, 5]);
        let factor_grads = grad_ctx
            .compute_factor_gradients(&grad_output)
            .expect("CP grad failed");

        assert_eq!(factor_grads.len(), 3);
        assert_eq!(factor_grads[0].shape(), &[3, 2]);
        assert_eq!(factor_grads[1].shape(), &[4, 2]);
        assert_eq!(factor_grads[2].shape(), &[5, 2]);
    }

    #[test]
    fn test_tt_reconstruction_basic() {
        // Create a simple 2-core TT: cores (1,3,2) and (2,4,1)
        // This represents a 3x4 matrix
        let mut core0 = DenseND::<f64>::zeros(&[1, 3, 2]);
        let mut core1 = DenseND::<f64>::zeros(&[2, 4, 1]);

        // Set core0: G0[0, i, r] for i in 0..3, r in 0..2
        // G0[0,0,:] = [1, 0]
        // G0[0,1,:] = [0, 1]
        // G0[0,2,:] = [1, 1]
        *core0.get_mut(&[0, 0, 0]).expect("idx") = 1.0;
        *core0.get_mut(&[0, 0, 1]).expect("idx") = 0.0;
        *core0.get_mut(&[0, 1, 0]).expect("idx") = 0.0;
        *core0.get_mut(&[0, 1, 1]).expect("idx") = 1.0;
        *core0.get_mut(&[0, 2, 0]).expect("idx") = 1.0;
        *core0.get_mut(&[0, 2, 1]).expect("idx") = 1.0;

        // Set core1: G1[r, j, 0] for r in 0..2, j in 0..4
        // G1[0,j,0] = [1, 2, 3, 4]
        // G1[1,j,0] = [5, 6, 7, 8]
        for j in 0..4 {
            *core1.get_mut(&[0, j, 0]).expect("idx") = (j + 1) as f64;
            *core1.get_mut(&[1, j, 0]).expect("idx") = (j + 5) as f64;
        }

        let ctx = TtReconstructionGrad::new(vec![core0, core1]);
        let recon = ctx.reconstruct().expect("reconstruction failed");

        // X[i,j] = sum_r G0[0,i,r] * G1[r,j,0]
        // X[0,j] = 1*G1[0,j,0] + 0*G1[1,j,0] = [1, 2, 3, 4]
        // X[1,j] = 0*G1[0,j,0] + 1*G1[1,j,0] = [5, 6, 7, 8]
        // X[2,j] = 1*G1[0,j,0] + 1*G1[1,j,0] = [6, 8, 10, 12]
        assert_eq!(recon.shape(), &[3, 4]);
        assert_eq!(*recon.get(&[0, 0]).expect("idx"), 1.0);
        assert_eq!(*recon.get(&[0, 3]).expect("idx"), 4.0);
        assert_eq!(*recon.get(&[1, 0]).expect("idx"), 5.0);
        assert_eq!(*recon.get(&[2, 0]).expect("idx"), 6.0);
        assert_eq!(*recon.get(&[2, 3]).expect("idx"), 12.0);
    }

    #[test]
    fn test_tt_reconstruction_grad_basic() {
        // Create simple TT decomposition with 3 cores
        // Core shapes: (1, 3, 2), (2, 4, 2), (2, 5, 1)
        let cores: Vec<DenseND<f64>> = vec![
            DenseND::zeros(&[1, 3, 2]),
            DenseND::zeros(&[2, 4, 2]),
            DenseND::zeros(&[2, 5, 1]),
        ];

        let grad_ctx = TtReconstructionGrad::new(cores);

        // Gradient output should have shape [3, 4, 5]
        let grad_output = DenseND::<f64>::ones(&[3, 4, 5]);

        // Compute gradients
        let result = grad_ctx.compute_core_gradients(&grad_output);

        assert!(result.is_ok());
        let grads = result.expect("gradient computation failed");

        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0].shape(), &[1, 3, 2]);
        assert_eq!(grads[1].shape(), &[2, 4, 2]);
        assert_eq!(grads[2].shape(), &[2, 5, 1]);
    }

    #[test]
    fn test_tt_gradient_nonzero() {
        // Create a 2-core TT with non-trivial values and verify non-zero gradients
        let mut core0 = DenseND::<f64>::zeros(&[1, 2, 2]);
        let mut core1 = DenseND::<f64>::zeros(&[2, 3, 1]);

        // G0[0,0,:] = [1, 2], G0[0,1,:] = [3, 4]
        *core0.get_mut(&[0, 0, 0]).expect("idx") = 1.0;
        *core0.get_mut(&[0, 0, 1]).expect("idx") = 2.0;
        *core0.get_mut(&[0, 1, 0]).expect("idx") = 3.0;
        *core0.get_mut(&[0, 1, 1]).expect("idx") = 4.0;

        // G1[0,:,0] = [1, 2, 3], G1[1,:,0] = [4, 5, 6]
        for j in 0..3 {
            *core1.get_mut(&[0, j, 0]).expect("idx") = (j + 1) as f64;
            *core1.get_mut(&[1, j, 0]).expect("idx") = (j + 4) as f64;
        }

        let ctx = TtReconstructionGrad::new(vec![core0, core1]);
        let grad_output = DenseND::<f64>::ones(&[2, 3]);
        let grads = ctx
            .compute_core_gradients(&grad_output)
            .expect("gradient computation failed");

        // Gradients should be non-zero since cores have non-zero values
        let grad0_sum: f64 = grads[0].as_array().iter().map(|x| x.abs()).sum();
        let grad1_sum: f64 = grads[1].as_array().iter().map(|x| x.abs()).sum();
        assert!(grad0_sum > 1e-10, "Gradient for core 0 should be non-zero");
        assert!(grad1_sum > 1e-10, "Gradient for core 1 should be non-zero");
    }

    #[test]
    fn test_tt_gradient_numerical_check() {
        // Verify TT gradient numerically using finite differences
        let mut core0 = DenseND::<f64>::zeros(&[1, 2, 2]);
        let mut core1 = DenseND::<f64>::zeros(&[2, 3, 1]);

        *core0.get_mut(&[0, 0, 0]).expect("idx") = 1.0;
        *core0.get_mut(&[0, 0, 1]).expect("idx") = 2.0;
        *core0.get_mut(&[0, 1, 0]).expect("idx") = 3.0;
        *core0.get_mut(&[0, 1, 1]).expect("idx") = 4.0;

        for j in 0..3 {
            *core1.get_mut(&[0, j, 0]).expect("idx") = (j + 1) as f64;
            *core1.get_mut(&[1, j, 0]).expect("idx") = (j + 4) as f64;
        }

        let cores = vec![core0.clone(), core1.clone()];
        let ctx = TtReconstructionGrad::new(cores);
        let grad_output = DenseND::<f64>::ones(&[2, 3]);
        let analytical_grads = ctx
            .compute_core_gradients(&grad_output)
            .expect("gradient computation failed");

        // Numerical gradient check for core 0
        let epsilon = 1e-7;
        for alpha in 0..1 {
            for i in 0..2 {
                for beta in 0..2 {
                    let mut core0_plus = core0.clone();
                    let mut core0_minus = core0.clone();

                    let val = *core0.get(&[alpha, i, beta]).expect("idx");
                    *core0_plus.get_mut(&[alpha, i, beta]).expect("idx") = val + epsilon;
                    *core0_minus.get_mut(&[alpha, i, beta]).expect("idx") = val - epsilon;

                    let ctx_plus = TtReconstructionGrad::new(vec![core0_plus, core1.clone()]);
                    let ctx_minus = TtReconstructionGrad::new(vec![core0_minus, core1.clone()]);

                    let recon_plus = ctx_plus.reconstruct().expect("recon failed");
                    let recon_minus = ctx_minus.reconstruct().expect("recon failed");

                    // Numerical gradient: sum of grad_output * (recon_plus - recon_minus) / (2*eps)
                    let mut numerical_grad = 0.0;
                    for idx in 0..recon_plus.len() {
                        let mi = vec![idx / 3, idx % 3];
                        let vp = *recon_plus.get(&mi).expect("idx");
                        let vm = *recon_minus.get(&mi).expect("idx");
                        let g = *grad_output.get(&mi).expect("idx");
                        numerical_grad += g * (vp - vm) / (2.0 * epsilon);
                    }

                    let analytical_val = *analytical_grads[0].get(&[alpha, i, beta]).expect("idx");
                    let diff = (analytical_val - numerical_grad).abs();
                    assert!(
                        diff < 1e-4,
                        "Numerical check failed for core0[{},{},{}]: analytical={}, numerical={}, diff={}",
                        alpha, i, beta, analytical_val, numerical_grad, diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_tt_single_core() {
        // Single core case: G has shape (1, n, 1)
        // X[i] = G[0, i, 0]
        let mut core = DenseND::<f64>::zeros(&[1, 5, 1]);
        for i in 0..5 {
            *core.get_mut(&[0, i, 0]).expect("idx") = (i + 1) as f64;
        }

        let grad_ctx = TtReconstructionGrad::new(vec![core]);
        let grad_output =
            DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).expect("from_vec failed");

        let grads = grad_ctx
            .compute_core_gradients(&grad_output)
            .expect("gradient computation failed");

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[1, 5, 1]);

        // dL/dG[0,i,0] = dL/dX[i] since X[i] = G[0,i,0]
        for i in 0..5 {
            let expected = (i + 1) as f64 * 10.0;
            let actual = *grads[0].get(&[0, i, 0]).expect("idx");
            assert_eq!(actual, expected, "Single core grad mismatch at i={}", i);
        }
    }
}
