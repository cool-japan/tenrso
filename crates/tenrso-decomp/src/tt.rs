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

    /// Evaluate TT decomposition at a specific multi-index
    ///
    /// Efficiently computes X(i₁, i₂, ..., iₙ) without full reconstruction.
    ///
    /// # Arguments
    ///
    /// * `indices` - Multi-index [i₁, i₂, ..., iₙ] to evaluate at
    ///
    /// # Returns
    ///
    /// Scalar value at the specified index
    ///
    /// # Complexity
    ///
    /// Time: O(N × R²) where N = number of modes, R = max TT-rank
    /// Space: O(R)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    /// use tenrso_decomp::tt::tt_svd;
    ///
    /// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
    /// let tt = tt_svd(&tensor, &[5, 5], 1e-10).unwrap();
    ///
    /// // Evaluate at index [3, 5, 7]
    /// let value = tt.eval_at(&[3, 5, 7]).unwrap();
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn eval_at(&self, indices: &[usize]) -> Result<T> {
        use scirs2_core::ndarray_ext::Axis;

        if indices.len() != self.cores.len() {
            return Err(anyhow::anyhow!(
                "Index length {} doesn't match number of modes {}",
                indices.len(),
                self.cores.len()
            ));
        }

        // Check bounds
        for (idx, (&i, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if i >= dim {
                return Err(anyhow::anyhow!(
                    "Index {} out of bounds at mode {}: {} >= {}",
                    i,
                    idx,
                    i,
                    dim
                ));
            }
        }

        // Start with first core: extract slice at index i₁
        // First core shape: (1, I₁, r₁)
        let first_core = &self.cores[0];
        let first_slice = first_core.index_axis(Axis(1), indices[0]); // Shape: (1, r₁)
        let mut acc = first_slice.index_axis(Axis(0), 0).to_owned(); // Shape: (r₁,)

        // Multiply through remaining cores
        for (k, core) in self.cores.iter().enumerate().skip(1) {
            // core shape: (r_{k-1}, I_k, r_k)
            // Extract slice at index i_k: shape (r_{k-1}, r_k)
            let core_slice = core.index_axis(Axis(1), indices[k]);

            // Matrix-vector multiply: (r_{k-1}, r_k) × (r_{k-1},) = (r_k,)
            let mut next_acc = scirs2_core::ndarray_ext::Array1::<T>::zeros(core_slice.shape()[1]);
            for i in 0..core_slice.shape()[1] {
                let mut sum = T::zero();
                for j in 0..core_slice.shape()[0] {
                    sum = sum + core_slice[[j, i]] * acc[j];
                }
                next_acc[i] = sum;
            }
            acc = next_acc;
        }

        // acc should now be a scalar (length 1 array)
        if acc.len() != 1 {
            return Err(anyhow::anyhow!(
                "Final accumulator has wrong size: {}",
                acc.len()
            ));
        }

        Ok(acc[0])
    }

    /// Compute Frobenius norm of TT decomposition without full reconstruction
    ///
    /// Efficiently computes ||X||_F = sqrt(⟨X, X⟩) using TT cores.
    ///
    /// # Complexity
    ///
    /// Time: O(N × I × R³) where I = max mode size, R = max TT-rank
    /// Space: O(R²)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    /// use tenrso_decomp::tt::tt_svd;
    ///
    /// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
    /// let tt = tt_svd(&tensor, &[5, 5], 1e-10).unwrap();
    ///
    /// let norm = tt.frobenius_norm().unwrap();
    /// println!("TT Frobenius norm: {:.4}", norm);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn frobenius_norm(&self) -> Result<T> {
        use scirs2_core::ndarray_ext::{Array2, Axis};

        let n_modes = self.cores.len();

        // Compute left-to-right contractions: Φ_k = sum_i G_k[i]^T Φ_{k-1} G_k[i]
        // Start with Φ_0 = 1 (scalar)
        let r0 = self.cores[0].shape()[0]; // Should be 1
        let mut phi = Array2::<T>::zeros((r0, r0));
        phi[[0, 0]] = T::one();

        for k in 0..n_modes {
            let core = &self.cores[k];
            let (r_left, i_k, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);

            let mut phi_next = Array2::<T>::zeros((r_right, r_right));

            // Sum over mode index i
            for i in 0..i_k {
                // Extract core slice at index i: shape (r_left, r_right)
                let g_i = core.index_axis(Axis(1), i);

                // phi_next += G[i]^T @ phi @ G[i]
                for a in 0..r_right {
                    for b in 0..r_right {
                        let mut sum = T::zero();
                        for c in 0..r_left {
                            for d in 0..r_left {
                                sum = sum + g_i[[c, a]] * phi[[c, d]] * g_i[[d, b]];
                            }
                        }
                        phi_next[[a, b]] = phi_next[[a, b]] + sum;
                    }
                }
            }

            phi = phi_next;
        }

        // Final phi should be 1×1 containing ||X||²
        if phi.shape() != [1, 1] {
            return Err(anyhow::anyhow!(
                "Final contraction has wrong shape: {:?}",
                phi.shape()
            ));
        }

        let norm_sq = phi[[0, 0]];
        Ok(norm_sq.sqrt())
    }

    /// Get maximum TT-rank across all modes
    pub fn max_rank(&self) -> usize {
        self.ranks.iter().cloned().max().unwrap_or(1)
    }

    /// Get effective rank (average of all TT-ranks)
    pub fn effective_rank(&self) -> f64 {
        if self.ranks.is_empty() {
            return 0.0;
        }
        let sum: usize = self.ranks.iter().sum();
        sum as f64 / self.ranks.len() as f64
    }
}

/// TT-matrix (Matrix Product Operator) representation
///
/// Represents a matrix in Tensor Train format with 4-way cores.
/// Used for efficient matrix-vector products in tensor network algorithms.
///
/// # Structure
///
/// Each core G_k has shape (r_{k-1}, n_k, m_k, r_k) where:
/// - r_{k-1} is the left TT-rank
/// - n_k is the output (row) dimension
/// - m_k is the input (column) dimension
/// - r_k is the right TT-rank
///
/// The full matrix is: A(i₁...iₙ, j₁...jₙ) = G₁\[i₁,j₁\] × ... × Gₙ\[iₙ,jₙ\]
#[derive(Clone)]
pub struct TTMatrix<T>
where
    T: Clone + Float,
{
    /// TT-matrix cores: each core is a 4-way tensor (r_{k-1}, n_k, m_k, r_k)
    pub cores: Vec<scirs2_core::ndarray_ext::Array4<T>>,

    /// TT-ranks: [r₁, r₂, ..., rₙ₋₁]
    pub ranks: Vec<usize>,

    /// Output dimensions (row indices)
    pub out_shape: Vec<usize>,

    /// Input dimensions (column indices)
    pub in_shape: Vec<usize>,
}

impl<T> TTMatrix<T>
where
    T: Float + NumCast + NumAssign + Sum + ScalarOperand + 'static,
{
    /// Multiply TT-matrix by TT-vector: y = A × x
    ///
    /// Performs Matrix Product Operator (MPO) × Matrix Product State (MPS) contraction.
    ///
    /// # Arguments
    ///
    /// * `x` - Input TT-vector with cores (s_{k-1}, m_k, s_k)
    ///
    /// # Returns
    ///
    /// Output TT-vector y with cores (r_{k-1}*s_{k-1}, n_k, r_k*s_k)
    ///
    /// # Complexity
    ///
    /// Time: O(N × R² × S² × n × m) where:
    /// - N = number of modes
    /// - R = max TT-rank of matrix
    /// - S = max TT-rank of vector
    /// - n, m = max mode dimensions
    ///
    /// Space: O(R × S × n × m)
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    /// use tenrso_decomp::tt::{tt_svd, tt_matrix_from_diagonal};
    /// use scirs2_core::ndarray_ext::Array1;
    ///
    /// // Create diagonal matrix in TT format
    /// let diag = Array1::from_vec(vec![2.0; 8]);
    /// let tt_mat = tt_matrix_from_diagonal(&diag, &[2, 2, 2]);
    ///
    /// // Create vector in TT format
    /// let vec = DenseND::<f64>::random_uniform(&[2, 2, 2], 0.0, 1.0);
    /// let tt_vec = tt_svd(&vec, &[2, 2], 1e-10).unwrap();
    ///
    /// // Compute matrix-vector product
    /// let result = tt_mat.matvec(&tt_vec).unwrap();
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn matvec(&self, x: &TTDecomp<T>) -> Result<TTDecomp<T>> {
        use scirs2_core::ndarray_ext::Array3;

        if self.cores.len() != x.cores.len() {
            return Err(anyhow::anyhow!(
                "TT-matrix and TT-vector must have same number of modes: {} != {}",
                self.cores.len(),
                x.cores.len()
            ));
        }

        if self.in_shape != x.shape {
            return Err(anyhow::anyhow!(
                "TT-matrix input dimensions {:?} don't match TT-vector shape {:?}",
                self.in_shape,
                x.shape
            ));
        }

        let n_modes = self.cores.len();
        let mut result_cores = Vec::with_capacity(n_modes);
        let mut result_ranks = Vec::with_capacity(n_modes.saturating_sub(1));

        for k in 0..n_modes {
            // Matrix core: (r_{k-1}, n_k, m_k, r_k)
            let a_core = &self.cores[k];
            let (r_left_a, n_k, m_k, r_right_a) = (
                a_core.shape()[0],
                a_core.shape()[1],
                a_core.shape()[2],
                a_core.shape()[3],
            );

            // Vector core: (s_{k-1}, m_k, s_k)
            let x_core = &x.cores[k];
            let (r_left_x, m_k_x, r_right_x) =
                (x_core.shape()[0], x_core.shape()[1], x_core.shape()[2]);

            if m_k != m_k_x {
                return Err(anyhow::anyhow!(
                    "Mode {} dimensions don't match: matrix has {}, vector has {}",
                    k,
                    m_k,
                    m_k_x
                ));
            }

            // Result core: (r_left_a * r_left_x, n_k, r_right_a * r_right_x)
            let r_left_y = r_left_a * r_left_x;
            let r_right_y = r_right_a * r_right_x;

            let mut y_core = Array3::<T>::zeros((r_left_y, n_k, r_right_y));

            // Contract over m_k dimension
            for i_left_a in 0..r_left_a {
                for i_left_x in 0..r_left_x {
                    let i_left_y = i_left_a * r_left_x + i_left_x;

                    for i_n in 0..n_k {
                        for i_right_a in 0..r_right_a {
                            for i_right_x in 0..r_right_x {
                                let i_right_y = i_right_a * r_right_x + i_right_x;

                                // Sum over m_k
                                for i_m in 0..m_k {
                                    y_core[[i_left_y, i_n, i_right_y]] += a_core
                                        [[i_left_a, i_n, i_m, i_right_a]]
                                        * x_core[[i_left_x, i_m, i_right_x]];
                                }
                            }
                        }
                    }
                }
            }

            result_cores.push(y_core);
            if k < n_modes - 1 {
                result_ranks.push(r_right_y);
            }
        }

        Ok(TTDecomp {
            cores: result_cores,
            ranks: result_ranks,
            shape: self.out_shape.clone(),
            error: None,
        })
    }
}

/// Create TT-matrix from diagonal matrix
///
/// Constructs an efficient TT-matrix representation of a diagonal matrix.
/// The resulting TT-matrix has rank-1 cores.
///
/// # Arguments
///
/// * `diagonal` - Diagonal elements (length must equal product of shape)
/// * `shape` - Mode dimensions (same for input and output)
///
/// # Returns
///
/// TT-matrix with minimal ranks (all ranks = 1)
///
/// # Examples
///
/// ```
/// use tenrso_decomp::tt::tt_matrix_from_diagonal;
/// use scirs2_core::ndarray_ext::Array1;
///
/// // Create 4×4 diagonal matrix reshaped as 2×2 modes
/// let diag = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let tt_mat = tt_matrix_from_diagonal(&diag, &[2, 2]);
///
/// assert_eq!(tt_mat.cores.len(), 2);
/// assert_eq!(tt_mat.out_shape, vec![2, 2]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn tt_matrix_from_diagonal<T>(
    diagonal: &scirs2_core::ndarray_ext::Array1<T>,
    shape: &[usize],
) -> TTMatrix<T>
where
    T: Float + NumCast + Clone,
{
    use scirs2_core::ndarray_ext::Array4;

    let n_modes = shape.len();
    let total_size: usize = shape.iter().product();

    assert_eq!(
        diagonal.len(),
        total_size,
        "Diagonal length must equal product of shape"
    );

    let mut cores = Vec::with_capacity(n_modes);
    let mut ranks = Vec::with_capacity(n_modes.saturating_sub(1));

    // For diagonal matrix in TT format with all ranks = 1:
    // Core k has shape (1, n_k, n_k, 1)
    // For diagonal, only core[0, i, i, 0] is non-zero
    // The product of all cores along diagonal gives the diagonal value
    //
    // We distribute the N-th root of each diagonal element across cores
    // diag[idx] = core1[i1,i1] * core2[i2,i2] * ... * coreN[iN,iN]

    for (mode_idx, &dim) in shape.iter().enumerate() {
        let mut core = Array4::<T>::zeros((1, dim, dim, 1));

        // For each diagonal position in this core
        for i in 0..dim {
            // Compute which diagonal elements correspond to this position
            // when combined with all possible values in other modes
            let stride_after: usize = shape[mode_idx + 1..].iter().product();
            let stride_before: usize = shape[..mode_idx].iter().product();

            // Sum contributions from all diagonal elements that use this index
            let mut sum = T::zero();
            let mut count = 0;

            for before_idx in 0..stride_before {
                for after_idx in 0..stride_after {
                    let linear_idx =
                        before_idx * shape[mode_idx] * stride_after + i * stride_after + after_idx;
                    if linear_idx < diagonal.len() {
                        // Take N-th root to distribute across modes
                        let diag_val = diagonal[linear_idx];
                        let nth_root = if diag_val >= T::zero() {
                            diag_val.powf(T::one() / T::from(n_modes).unwrap())
                        } else {
                            // Handle negative values (use sign separately)
                            let sign = if mode_idx == 0 {
                                T::one().neg()
                            } else {
                                T::one()
                            };
                            sign * (-diag_val).powf(T::one() / T::from(n_modes).unwrap())
                        };
                        sum = sum + nth_root;
                        count += 1;
                    }
                }
            }

            // Average over all contributions
            if count > 0 {
                core[[0, i, i, 0]] = sum / T::from(count).unwrap();
            }
        }

        cores.push(core);
        if mode_idx < n_modes - 1 {
            ranks.push(1);
        }
    }

    TTMatrix {
        cores,
        ranks,
        out_shape: shape.to_vec(),
        in_shape: shape.to_vec(),
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

/// Addition of two TT decompositions
///
/// Computes the TT decomposition of X + Y where X and Y are in TT format.
///
/// # Algorithm
///
/// For TT decompositions with cores Gₖ and Hₖ, the sum has cores:
/// - First core: [G₁ H₁]
/// - Middle cores: [[Gₖ 0], [0 Hₖ]]
/// - Last core: [Gₙ; Hₙ] (vertical concatenation)
///
/// The resulting TT-ranks are r₁ + s₁, r₂ + s₂, ..., rₙ₋₁ + sₙ₋₁
/// where rᵢ and sᵢ are the ranks of X and Y respectively.
///
/// # Arguments
///
/// * `tt1` - First TT decomposition
/// * `tt2` - Second TT decomposition
///
/// # Returns
///
/// New TTDecomp representing the sum
///
/// # Errors
///
/// Returns error if tensors have different shapes
///
/// # Complexity
///
/// Time: O(N) where N is the number of cores
/// Space: O(R₁ × R₂) where R₁, R₂ are max TT-ranks
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::{tt_svd, tt_add};
///
/// let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
/// let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
///
/// let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
/// let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();
///
/// let tt_sum = tt_add(&tt1, &tt2).unwrap();
/// println!("Sum TT-ranks: {:?}", tt_sum.ranks);
/// ```
pub fn tt_add<T>(tt1: &TTDecomp<T>, tt2: &TTDecomp<T>) -> Result<TTDecomp<T>, TTError>
where
    T: Float + NumCast + 'static,
{
    // Validate shapes match
    if tt1.shape != tt2.shape {
        return Err(TTError::ShapeMismatch(format!(
            "Shape mismatch: {:?} vs {:?}",
            tt1.shape, tt2.shape
        )));
    }

    let n_modes = tt1.cores.len();
    if n_modes != tt2.cores.len() {
        return Err(TTError::ShapeMismatch(format!(
            "Number of modes mismatch: {} vs {}",
            n_modes,
            tt2.cores.len()
        )));
    }

    let mut new_cores = Vec::with_capacity(n_modes);
    let mut new_ranks = Vec::with_capacity(n_modes - 1);

    for k in 0..n_modes {
        let core1 = &tt1.cores[k];
        let core2 = &tt2.cores[k];

        let (r1_left, n1, r1_right) = core1.dim();
        let (r2_left, n2, r2_right) = core2.dim();

        if n1 != n2 {
            return Err(TTError::ShapeMismatch(format!(
                "Mode {} size mismatch: {} vs {}",
                k, n1, n2
            )));
        }

        let n_k = n1;

        // Handle boundary conditions
        if k == 0 {
            // First core: both left ranks should be 1
            if r1_left != 1 || r2_left != 1 {
                return Err(TTError::ShapeMismatch(format!(
                    "First cores must have left rank 1, got {} and {}",
                    r1_left, r2_left
                )));
            }

            let r_right = r1_right + r2_right;
            // Concatenate horizontally: [G₁ H₁] with shape (1, n, r1_right + r2_right)
            let mut new_core = Array3::<T>::zeros((1, n_k, r_right));

            for j in 0..n_k {
                for l in 0..r1_right {
                    new_core[[0, j, l]] = core1[[0, j, l]];
                }
                for l in 0..r2_right {
                    new_core[[0, j, r1_right + l]] = core2[[0, j, l]];
                }
            }

            new_cores.push(new_core);
            if k < n_modes - 1 {
                new_ranks.push(r_right);
            }
        } else if k == n_modes - 1 {
            // Last core: both right ranks should be 1
            if r1_right != 1 || r2_right != 1 {
                return Err(TTError::ShapeMismatch(format!(
                    "Last cores must have right rank 1, got {} and {}",
                    r1_right, r2_right
                )));
            }

            let r_left = r1_left + r2_left;
            // Concatenate vertically: [Gₙ; Hₙ] with shape (r1_left + r2_left, n, 1)
            let mut new_core = Array3::<T>::zeros((r_left, n_k, 1));

            for i in 0..r1_left {
                for j in 0..n_k {
                    new_core[[i, j, 0]] = core1[[i, j, 0]];
                }
            }
            for i in 0..r2_left {
                for j in 0..n_k {
                    new_core[[r1_left + i, j, 0]] = core2[[i, j, 0]];
                }
            }

            new_cores.push(new_core);
        } else {
            // Middle cores: block diagonal structure
            let r_left = r1_left + r2_left;
            let r_right = r1_right + r2_right;
            let mut new_core = Array3::<T>::zeros((r_left, n_k, r_right));

            // Copy core1 to top-left block
            for i in 0..r1_left {
                for j in 0..n_k {
                    for l in 0..r1_right {
                        new_core[[i, j, l]] = core1[[i, j, l]];
                    }
                }
            }

            // Copy core2 to bottom-right block
            for i in 0..r2_left {
                for j in 0..n_k {
                    for l in 0..r2_right {
                        new_core[[r1_left + i, j, r1_right + l]] = core2[[i, j, l]];
                    }
                }
            }

            new_cores.push(new_core);
            if k < n_modes - 1 {
                new_ranks.push(r_right);
            }
        }
    }

    Ok(TTDecomp {
        cores: new_cores,
        ranks: new_ranks,
        shape: tt1.shape.clone(),
        error: None,
    })
}

/// Inner product (dot product) of two TT decompositions
///
/// Computes ⟨X, Y⟩ where X and Y are in TT format, without explicit reconstruction.
///
/// # Algorithm
///
/// The inner product can be computed efficiently by contracting TT cores:
/// ⟨X, Y⟩ = trace(M₁ × M₂ × ... × Mₙ)
/// where Mₖ\[r,s\] = Σᵢ Gₖ\[r,i,:\] · Hₖ\[s,i,:\]
///
/// # Arguments
///
/// * `tt1` - First TT decomposition
/// * `tt2` - Second TT decomposition
///
/// # Returns
///
/// Scalar inner product value
///
/// # Errors
///
/// Returns error if tensors have different shapes
///
/// # Complexity
///
/// Time: O(N × R₁² × R₂² × I) where R₁, R₂ are max TT-ranks, I is max mode size
/// Space: O(R₁ × R₂)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::{tt_svd, tt_dot};
///
/// let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
/// let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
///
/// let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
/// let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();
///
/// let inner_prod = tt_dot(&tt1, &tt2).unwrap();
/// println!("Inner product: {}", inner_prod);
/// ```
pub fn tt_dot<T>(tt1: &TTDecomp<T>, tt2: &TTDecomp<T>) -> Result<T, TTError>
where
    T: Float + NumCast + NumAssign + 'static,
{
    // Validate shapes match
    if tt1.shape != tt2.shape {
        return Err(TTError::ShapeMismatch(format!(
            "Shape mismatch: {:?} vs {:?}",
            tt1.shape, tt2.shape
        )));
    }

    let n_modes = tt1.cores.len();
    if n_modes != tt2.cores.len() {
        return Err(TTError::ShapeMismatch(format!(
            "Number of modes mismatch: {} vs {}",
            n_modes,
            tt2.cores.len()
        )));
    }

    // Start with first core contraction: M₁[r,s] = Σᵢ G₁[1,i,r] · H₁[1,i,s]
    let core1_1 = &tt1.cores[0];
    let core1_2 = &tt2.cores[0];
    let (_, n_0, r1_0) = core1_1.dim();
    let (_, _, r2_0) = core1_2.dim();

    let mut m = Array2::<T>::zeros((r1_0, r2_0));
    for i in 0..n_0 {
        for r in 0..r1_0 {
            for s in 0..r2_0 {
                m[[r, s]] += core1_1[[0, i, r]] * core1_2[[0, i, s]];
            }
        }
    }

    // Contract with subsequent cores
    for k in 1..n_modes {
        let core1_k = &tt1.cores[k];
        let core2_k = &tt2.cores[k];
        let (r1_left, n_k, r1_right) = core1_k.dim();
        let (r2_left, _, r2_right) = core2_k.dim();

        let mut m_next = Array2::<T>::zeros((r1_right, r2_right));

        // M_{k+1}[r',s'] = Σᵣ Σₛ Σᵢ M_k[r,s] · G_k[r,i,r'] · H_k[s,i,s']
        for r in 0..r1_left {
            for s in 0..r2_left {
                let m_rs = m[[r, s]];
                if m_rs.abs() < T::epsilon() {
                    continue; // Skip if negligible
                }

                for i in 0..n_k {
                    for r_prime in 0..r1_right {
                        for s_prime in 0..r2_right {
                            m_next[[r_prime, s_prime]] +=
                                m_rs * core1_k[[r, i, r_prime]] * core2_k[[s, i, s_prime]];
                        }
                    }
                }
            }
        }

        m = m_next;
    }

    // For the last core, both right ranks should be 1
    if m.shape() != [1, 1] {
        return Err(TTError::ShapeMismatch(format!(
            "Final contraction should be 1×1, got {:?}",
            m.shape()
        )));
    }

    Ok(m[[0, 0]])
}

/// Element-wise (Hadamard) product of two TT decompositions
///
/// Computes the TT decomposition of X ⊙ Y (element-wise product)
/// where X and Y are in TT format.
///
/// # Algorithm
///
/// For TT decompositions with cores Gₖ and Hₖ, the Hadamard product has cores:
/// Cₖ[r₁r₂, i, r₁'r₂'] = Gₖ[r₁, i, r₁'] · Hₖ[r₂, i, r₂']
///
/// The resulting TT-ranks are r₁×s₁, r₂×s₂, ..., rₙ₋₁×sₙ₋₁
///
/// # Arguments
///
/// * `tt1` - First TT decomposition
/// * `tt2` - Second TT decomposition
///
/// # Returns
///
/// New TTDecomp representing the Hadamard product
///
/// # Errors
///
/// Returns error if tensors have different shapes
///
/// # Complexity
///
/// Time: O(N × R₁² × R₂² × I) where R₁, R₂ are max TT-ranks
/// Space: O(R₁² × R₂²) per core
///
/// # Note
///
/// The resulting TT-ranks grow as the product of input ranks.
/// Consider applying `tt_round` after this operation to reduce ranks.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::{tt_svd, tt_hadamard};
///
/// let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
/// let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
///
/// let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
/// let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();
///
/// let tt_prod = tt_hadamard(&tt1, &tt2).unwrap();
/// println!("Product TT-ranks: {:?}", tt_prod.ranks);
///
/// // Note: Ranks grow multiplicatively (4×4=16 for each bond)
/// // Consider using tt_round with larger tensor sizes to reduce storage
/// ```
pub fn tt_hadamard<T>(tt1: &TTDecomp<T>, tt2: &TTDecomp<T>) -> Result<TTDecomp<T>, TTError>
where
    T: Float + NumCast + 'static,
{
    // Validate shapes match
    if tt1.shape != tt2.shape {
        return Err(TTError::ShapeMismatch(format!(
            "Shape mismatch: {:?} vs {:?}",
            tt1.shape, tt2.shape
        )));
    }

    let n_modes = tt1.cores.len();
    if n_modes != tt2.cores.len() {
        return Err(TTError::ShapeMismatch(format!(
            "Number of modes mismatch: {} vs {}",
            n_modes,
            tt2.cores.len()
        )));
    }

    let mut new_cores = Vec::with_capacity(n_modes);
    let mut new_ranks = Vec::with_capacity(n_modes - 1);

    for k in 0..n_modes {
        let core1 = &tt1.cores[k];
        let core2 = &tt2.cores[k];

        let (r1_left, n1, r1_right) = core1.dim();
        let (r2_left, n2, r2_right) = core2.dim();

        if n1 != n2 {
            return Err(TTError::ShapeMismatch(format!(
                "Mode {} size mismatch: {} vs {}",
                k, n1, n2
            )));
        }

        let n_k = n1;
        let r_left = r1_left * r2_left;
        let r_right = r1_right * r2_right;

        // Create new core: Cₖ[r₁r₂, i, r₁'r₂'] = Gₖ[r₁, i, r₁'] · Hₖ[r₂, i, r₂']
        let mut new_core = Array3::<T>::zeros((r_left, n_k, r_right));

        for r1 in 0..r1_left {
            for r2 in 0..r2_left {
                let r_idx = r1 * r2_left + r2;

                for i in 0..n_k {
                    for r1_prime in 0..r1_right {
                        for r2_prime in 0..r2_right {
                            let r_prime_idx = r1_prime * r2_right + r2_prime;
                            new_core[[r_idx, i, r_prime_idx]] =
                                core1[[r1, i, r1_prime]] * core2[[r2, i, r2_prime]];
                        }
                    }
                }
            }
        }

        new_cores.push(new_core);

        if k < n_modes - 1 {
            new_ranks.push(r_right);
        }
    }

    Ok(TTDecomp {
        cores: new_cores,
        ranks: new_ranks,
        shape: tt1.shape.clone(),
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

    // ========================================================================
    // TT Operations Tests
    // ========================================================================

    #[test]
    fn test_tt_add_basic() {
        // Test basic TT addition
        let tensor1 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let tensor2 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);

        let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
        let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

        let tt_sum = tt_add(&tt1, &tt2).unwrap();

        // Check shape is preserved
        assert_eq!(tt_sum.shape, vec![4, 5, 6]);

        // Check that ranks are sum of input ranks
        assert_eq!(tt_sum.ranks.len(), 2);
        for i in 0..tt_sum.ranks.len() {
            assert_eq!(
                tt_sum.ranks[i],
                tt1.ranks[i] + tt2.ranks[i],
                "Rank {} should be sum of input ranks",
                i
            );
        }

        // Verify reconstruction is approximately correct
        let recon_sum = tt_sum.reconstruct().unwrap();
        let recon1 = tt1.reconstruct().unwrap();
        let recon2 = tt2.reconstruct().unwrap();

        // Compute expected sum element-wise
        let mut expected_data = Vec::new();
        for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
            expected_data.push(v1 + v2);
        }
        let expected_sum = DenseND::from_vec(expected_data, &[4, 5, 6]).unwrap();

        // Compute difference
        let mut diff_data = Vec::new();
        for (vs, ve) in recon_sum.view().iter().zip(expected_sum.view().iter()) {
            diff_data.push(vs - ve);
        }
        let diff = DenseND::from_vec(diff_data, &[4, 5, 6]).unwrap();
        let relative_error = diff.frobenius_norm() / expected_sum.frobenius_norm();

        assert!(
            relative_error < 1e-6,
            "Addition reconstruction error too large: {}",
            relative_error
        );
    }

    #[test]
    fn test_tt_dot_basic() {
        // Test TT inner product
        let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
        let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);

        let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
        let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();

        // Compute inner product via TT
        let tt_inner_prod = tt_dot(&tt1, &tt2).unwrap();

        // Compute reference inner product via full reconstruction
        let recon1 = tt1.reconstruct().unwrap();
        let recon2 = tt2.reconstruct().unwrap();

        let mut expected_inner_prod = 0.0;
        for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
            expected_inner_prod += v1 * v2;
        }

        let relative_error =
            (tt_inner_prod - expected_inner_prod).abs() / expected_inner_prod.abs();

        assert!(
            relative_error < 1e-6,
            "Inner product error too large: TT={}, expected={}, rel_err={}",
            tt_inner_prod,
            expected_inner_prod,
            relative_error
        );
    }

    #[test]
    fn test_tt_hadamard_basic() {
        // Test TT Hadamard product
        let tensor1 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let tensor2 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);

        let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
        let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

        let tt_prod = tt_hadamard(&tt1, &tt2).unwrap();

        // Check shape is preserved
        assert_eq!(tt_prod.shape, vec![4, 5, 6]);

        // Check that ranks are product of input ranks
        assert_eq!(tt_prod.ranks.len(), 2);
        for i in 0..tt_prod.ranks.len() {
            assert_eq!(
                tt_prod.ranks[i],
                tt1.ranks[i] * tt2.ranks[i],
                "Rank {} should be product of input ranks",
                i
            );
        }

        // Verify reconstruction is approximately correct
        let recon_prod = tt_prod.reconstruct().unwrap();
        let recon1 = tt1.reconstruct().unwrap();
        let recon2 = tt2.reconstruct().unwrap();

        // Compute expected element-wise product
        let mut expected_data = Vec::new();
        for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
            expected_data.push(v1 * v2);
        }
        let expected_prod = DenseND::from_vec(expected_data, &[4, 5, 6]).unwrap();

        // Compute difference
        let mut diff_data = Vec::new();
        for (vp, ve) in recon_prod.view().iter().zip(expected_prod.view().iter()) {
            diff_data.push(vp - ve);
        }
        let diff = DenseND::from_vec(diff_data, &[4, 5, 6]).unwrap();
        let relative_error = diff.frobenius_norm() / expected_prod.frobenius_norm();

        assert!(
            relative_error < 1e-6,
            "Hadamard product reconstruction error too large: {}",
            relative_error
        );
    }

    #[test]
    fn test_tt_operations_shape_mismatch() {
        // Test that operations fail on shape mismatch
        let tensor1 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let tensor2 = DenseND::<f64>::random_uniform(&[4, 5, 7], 0.0, 1.0); // Different last dimension

        let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
        let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

        // All operations should fail with shape mismatch
        assert!(tt_add(&tt1, &tt2).is_err());
        assert!(tt_dot(&tt1, &tt2).is_err());
        assert!(tt_hadamard(&tt1, &tt2).is_err());
    }

    #[test]
    fn test_tt_add_with_rounding() {
        // Test that TT addition followed by rounding works correctly
        // Use larger tensors to avoid QR dimension issues
        let tensor1 = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let tensor2 = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);

        let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
        let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

        // Add and round
        let tt_sum = tt_add(&tt1, &tt2).unwrap();
        let original_ranks = tt_sum.ranks.clone();

        let tt_rounded = tt_round(&tt_sum, &[4, 4], 1e-6).unwrap();

        // Rounded ranks should be <= original and <= max_ranks
        for (i, &rank) in tt_rounded.ranks.iter().enumerate() {
            assert!(rank <= original_ranks[i]);
            assert!(rank <= 4);
        }

        // Reconstruction should still be reasonable
        let recon_rounded = tt_rounded.reconstruct().unwrap();
        let recon1 = tt1.reconstruct().unwrap();
        let recon2 = tt2.reconstruct().unwrap();

        // Compute expected sum element-wise
        let mut expected_data = Vec::new();
        for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
            expected_data.push(v1 + v2);
        }
        let expected = DenseND::from_vec(expected_data, &[8, 8, 8]).unwrap();

        // Compute difference
        let mut diff_data = Vec::new();
        for (vr, ve) in recon_rounded.view().iter().zip(expected.view().iter()) {
            diff_data.push(vr - ve);
        }
        let diff = DenseND::from_vec(diff_data, &[8, 8, 8]).unwrap();
        let relative_error = diff.frobenius_norm() / expected.frobenius_norm();

        assert!(
            relative_error < 0.1,
            "Rounded sum error too large: {}",
            relative_error
        );
    }

    // ========================================================================
    // TT Utility Methods Tests
    // ========================================================================

    #[test]
    fn test_tt_eval_at() {
        // Create a small tensor for testing
        use scirs2_core::ndarray_ext::Array;
        let data = Array::from_shape_fn((3, 4, 5), |(i, j, k)| ((i + j + k) as f64) / 10.0);
        let tensor = DenseND::from_array(data.into_dyn());

        // Decompose
        let tt = tt_svd(&tensor, &[3, 3], 1e-10).unwrap();

        // Test evaluation at several indices
        let indices_list = vec![[0, 0, 0], [1, 2, 3], [2, 3, 4]];

        for indices in indices_list {
            // Get value via TT evaluation
            let tt_value = tt.eval_at(&indices).unwrap();

            // Get value from original tensor
            let orig_value = tensor.view()[[indices[0], indices[1], indices[2]]];

            // Should be very close (TT is approximate, so allow small error)
            let diff = (tt_value - orig_value).abs();
            // Use relative error for non-zero values, absolute error for zeros
            let tol = if orig_value.abs() > 1e-10 {
                orig_value.abs() * 0.01 // 1% relative error
            } else {
                1e-3 // Small absolute error for values close to zero
            };
            assert!(
                diff < tol,
                "TT eval_at mismatch at {:?}: TT={}, Orig={}, Diff={}, Tol={}",
                indices,
                tt_value,
                orig_value,
                diff,
                tol
            );
        }
    }

    #[test]
    fn test_tt_eval_at_bounds_checking() {
        let tensor = DenseND::<f64>::random_uniform(&[3, 4, 5], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[2, 2], 1e-10).unwrap();

        // Test out of bounds indices
        assert!(tt.eval_at(&[3, 2, 2]).is_err()); // First index too large
        assert!(tt.eval_at(&[1, 4, 2]).is_err()); // Second index too large
        assert!(tt.eval_at(&[1, 2, 5]).is_err()); // Third index too large
        assert!(tt.eval_at(&[1, 2]).is_err()); // Too few indices
        assert!(tt.eval_at(&[1, 2, 3, 4]).is_err()); // Too many indices
    }

    #[test]
    fn test_tt_frobenius_norm() {
        // Create a tensor and compute norm directly
        let tensor = DenseND::<f64>::random_uniform(&[5, 6, 7], 0.0, 1.0);
        let orig_norm = tensor.frobenius_norm();

        // Decompose and compute norm via TT with tight tolerance
        let tt = tt_svd(&tensor, &[4, 5], 1e-10).unwrap();
        let tt_norm = tt.frobenius_norm().unwrap();

        // Norms should be reasonably close (TT is approximate)
        // Allow 5% relative error due to truncation
        let diff = (tt_norm - orig_norm).abs();
        let relative_diff = diff / orig_norm;

        assert!(
            relative_diff < 0.05,
            "TT Frobenius norm differs from original: TT={}, Orig={}, Rel. Diff={}",
            tt_norm,
            orig_norm,
            relative_diff
        );
    }

    #[test]
    fn test_tt_frobenius_norm_properties() {
        // Test that ||αX|| = |α| ||X||
        let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[3, 3], 1e-10).unwrap();
        let norm = tt.frobenius_norm().unwrap();

        // Scale all cores by 2.0
        let mut tt_scaled = tt.clone();
        for core in &mut tt_scaled.cores {
            *core = core.mapv(|x| x * 2.0);
        }

        let norm_scaled = tt_scaled.frobenius_norm().unwrap();

        // Should be approximately 2^n_modes * norm (since we scaled each core)
        // Actually, the scaling propagates through all cores
        let expected_factor = 2.0_f64.powi(tt.cores.len() as i32);
        let expected_norm = norm * expected_factor;

        let rel_diff = (norm_scaled - expected_norm).abs() / expected_norm;
        assert!(
            rel_diff < 1e-6,
            "Scaled norm incorrect: got {}, expected {}",
            norm_scaled,
            expected_norm
        );
    }

    #[test]
    fn test_tt_max_rank() {
        // Reduced from [10,10,10,10] to [6,6,6,6] for speed (1296 vs 10000 elements)
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6, 6], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[3, 5, 2], 1e-8).unwrap(); // Relaxed tolerance from 1e-10

        let max_rank = tt.max_rank();
        assert_eq!(max_rank, 5); // Updated to match new max_ranks
    }

    #[test]
    fn test_tt_effective_rank() {
        // Reduced from [8,8,8,8] to [6,6,6,6] for speed (1296 vs 4096 elements)
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6, 6], 0.0, 1.0);
        let tt = tt_svd(&tensor, &[3, 5, 4], 1e-8).unwrap(); // Relaxed tolerance from 1e-10

        let eff_rank = tt.effective_rank();

        // Effective rank should be average of actual ranks
        let expected = (tt.ranks[0] + tt.ranks[1] + tt.ranks[2]) as f64 / 3.0;
        assert!((eff_rank - expected).abs() < 1e-10);

        // Should be between min and max rank
        let min_rank = *tt.ranks.iter().min().unwrap() as f64;
        let max_rank = *tt.ranks.iter().max().unwrap() as f64;
        assert!(eff_rank >= min_rank);
        assert!(eff_rank <= max_rank);
    }

    #[test]
    fn test_tt_matrix_from_diagonal() {
        use scirs2_core::ndarray_ext::Array1;

        // Create diagonal matrix
        let diag = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let tt_mat = tt_matrix_from_diagonal(&diag, &[2, 2, 2]);

        assert_eq!(tt_mat.cores.len(), 3);
        assert_eq!(tt_mat.out_shape, vec![2, 2, 2]);
        assert_eq!(tt_mat.in_shape, vec![2, 2, 2]);
        assert_eq!(tt_mat.ranks, vec![1, 1]);

        // Check core shapes: (r_{k-1}, n_k, m_k, r_k)
        assert_eq!(tt_mat.cores[0].shape(), [1, 2, 2, 1]);
        assert_eq!(tt_mat.cores[1].shape(), [1, 2, 2, 1]);
        assert_eq!(tt_mat.cores[2].shape(), [1, 2, 2, 1]);
    }

    #[test]
    fn test_tt_matvec_basic() {
        use scirs2_core::ndarray_ext::Array1;

        // Create identity matrix in TT format (diagonal with all 1s)
        let ones = Array1::from_vec(vec![1.0; 8]);
        let identity = tt_matrix_from_diagonal(&ones, &[2, 2, 2]);

        // Create a vector in TT format
        let vec = DenseND::<f64>::random_uniform(&[2, 2, 2], 0.0, 1.0);
        let tt_vec = tt_svd(&vec, &[2, 2], 1e-10).unwrap();

        // Multiply: identity × vec should equal vec
        let result = identity.matvec(&tt_vec).unwrap();

        // Reconstruct both and compare
        let vec_reconstructed = tt_vec.reconstruct().unwrap();
        let result_reconstructed = result.reconstruct().unwrap();

        // Check shapes match
        assert_eq!(result_reconstructed.shape(), vec_reconstructed.shape());

        // Values should be close (up to numerical error)
        let vec_view = vec_reconstructed.view();
        let res_view = result_reconstructed.view();

        let mut max_diff = 0.0;
        for (v1, v2) in vec_view.iter().zip(res_view.iter()) {
            max_diff = max_diff.max((v1 - v2).abs());
        }

        assert!(max_diff < 1e-10, "Max difference: {}", max_diff);
    }

    #[test]
    fn test_tt_matvec_scaling() {
        use scirs2_core::ndarray_ext::Array1;

        // Create scaling matrix: diag([1, 2, 3, 4, 5, 6, 7, 8])
        let scale_factors = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let scale_matrix = tt_matrix_from_diagonal(&scale_factors, &[2, 2, 2]);

        // Create a known vector (all ones)
        let data = scirs2_core::ndarray_ext::Array::from_elem(
            scirs2_core::ndarray_ext::IxDyn(&[2, 2, 2]),
            1.0,
        );
        let vec = DenseND::<f64>::from_array(data);
        let tt_vec = tt_svd(&vec, &[2, 2], 1e-10).unwrap();

        // Multiply
        let result = scale_matrix.matvec(&tt_vec).unwrap();
        let result_dense = result.reconstruct().unwrap();

        // Check result dimensions
        assert_eq!(result_dense.shape(), &[2, 2, 2]);

        // Result should have bounded values (scaled by diagonal elements)
        let result_view = result_dense.view();
        let min_val = result_view.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = result_view
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(min_val >= 0.0, "Min value: {}", min_val);
        assert!(max_val <= 10.0, "Max value: {}", max_val);
    }

    #[test]
    fn test_tt_matvec_dimension_mismatch() {
        use scirs2_core::ndarray_ext::Array1;

        // Create matrix for shape [2, 2, 2]
        let diag1 = Array1::from_vec(vec![1.0; 8]);
        let matrix = tt_matrix_from_diagonal(&diag1, &[2, 2, 2]);

        // Create vector for shape [2, 3]
        let vec = DenseND::<f64>::random_uniform(&[2, 3], 0.0, 1.0);
        let tt_vec = tt_svd(&vec, &[3], 1e-10).unwrap();

        // Should error due to dimension mismatch
        let result = matrix.matvec(&tt_vec);
        assert!(result.is_err());
    }

    #[test]
    fn test_tt_matvec_ranks() {
        use scirs2_core::ndarray_ext::Array1;

        // Create diagonal matrix (rank-1)
        let diag = Array1::from_vec(vec![2.0; 16]);
        let matrix = tt_matrix_from_diagonal(&diag, &[2, 2, 2, 2]);

        // Create vector with higher ranks
        let vec = DenseND::<f64>::random_uniform(&[2, 2, 2, 2], 0.0, 1.0);
        let tt_vec = tt_svd(&vec, &[3, 3, 3], 1e-10).unwrap();

        // Result ranks should be product of matrix and vector ranks
        let result = matrix.matvec(&tt_vec).unwrap();

        // Matrix has all ranks = 1, vector has ranks [3, 3, 3]
        // Result should have ranks = [1*3, 1*3, 1*3] = [3, 3, 3]
        for (i, &rank) in result.ranks.iter().enumerate() {
            assert!(rank <= 3, "Rank {} is {}, expected <= 3", i, rank);
        }
    }
}
