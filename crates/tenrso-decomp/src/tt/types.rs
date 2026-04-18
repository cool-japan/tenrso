//! TT-specific data types: `TTError`, `TTDecomp`, `TTMatrix`.
//!
//! These types form the shared vocabulary used by the TT-SVD / TT-rounding
//! algorithms and by the TT arithmetic operations. Keeping them together
//! mirrors the original `tt.rs` layout while allowing the algorithm/operation
//! code paths to live in dedicated sub-modules.
//!
//! All array operations use `scirs2_core::ndarray_ext`; direct use of
//! `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array3, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, NumCast, ToPrimitive};
use std::iter::Sum;
use tenrso_core::DenseND;
use thiserror::Error;

/// Infallibly cast a literal / well-bounded numeric value to `T`.
///
/// SAFETY: all call sites use numeric literals or small non-negative
/// `usize` values (`n_modes = shape.len()`, `count` loop counter) that
/// are representable in any supported `T: Float` (f32, f64). Falling
/// back to `T::zero()` in the unreachable failure path preserves
/// numerical safety without a panic, respecting the no-unwrap policy.
#[inline]
fn cast_lit<T: NumCast, V: ToPrimitive>(v: V) -> T {
    T::from(v).unwrap_or_else(|| {
        T::from(0u8).unwrap_or_else(|| {
            unreachable!("NumCast::from(0u8) must succeed for primitive numeric T")
        })
    })
}

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
                            diag_val.powf(T::one() / cast_lit::<T, _>(n_modes))
                        } else {
                            // Handle negative values (use sign separately)
                            let sign = if mode_idx == 0 {
                                T::one().neg()
                            } else {
                                T::one()
                            };
                            sign * (-diag_val).powf(T::one() / cast_lit::<T, _>(n_modes))
                        };
                        sum = sum + nth_root;
                        count += 1;
                    }
                }
            }

            // Average over all contributions
            if count > 0 {
                core[[0, i, i, 0]] = sum / cast_lit::<T, _>(count);
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
