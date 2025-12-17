//! Advanced linear algebra operations using scirs2-linalg
//!
//! This module provides wrappers for advanced matrix factorizations and decompositions
//! from scirs2-linalg, including:
//! - QR decomposition
//! - SVD (Singular Value Decomposition)
//! - Eigendecomposition (general and Hermitian)
//! - Cholesky decomposition
//!
//! These operations are critical for tensor decompositions (Tucker-HOSVD, TT-SVD)
//! and numerical linear algebra applications.

use crate::DenseND;
use anyhow::{bail, Result};
use scirs2_core::ndarray_ext::ScalarOperand;
use scirs2_core::num_complex::Complex;
use scirs2_core::num_traits::{Float, FromPrimitive, NumAssign, NumCast};
use std::iter::Sum;

#[cfg(feature = "linalg")]
use scirs2_core::ndarray_ext::{Array2, Ix2};

#[cfg(feature = "linalg")]
use scirs2_linalg;

/// Type alias for eigendecomposition result (eigenvalues, eigenvectors)
pub type EigResult<T> = (DenseND<Complex<T>>, DenseND<Complex<T>>);

impl<T> DenseND<T>
where
    T: Float + FromPrimitive + NumCast + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    /// Compute QR decomposition: A = QR
    ///
    /// Decomposes a matrix into an orthogonal matrix Q and an upper triangular matrix R.
    ///
    /// # Arguments
    ///
    /// * `self` - Input matrix (must be 2D)
    ///
    /// # Returns
    ///
    /// * `Ok((Q, R))` - Tuple of orthogonal matrix Q and upper triangular matrix R
    /// * `Err` - If matrix is not 2D or if scirs2-linalg feature is not enabled
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// let a: DenseND<f64> = DenseND::from_vec(
    ///     vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    ///     &[3, 3]
    /// ).unwrap();
    ///
    /// let (q, r) = a.qr().unwrap();
    /// assert_eq!(q.shape(), &[3, 3]);
    /// assert_eq!(r.shape(), &[3, 3]);
    ///
    /// // Q should be orthogonal: Q^T Q = I
    /// let identity_check = q.transpose().unwrap().matmul(&q).unwrap();
    /// for i in 0..3 {
    ///     for j in 0..3 {
    ///         let expected: f64 = if i == j { 1.0 } else { 0.0 };
    ///         let actual = identity_check[&[i, j]];
    ///         assert!((actual - expected).abs() < 1e-10);
    ///     }
    /// }
    /// # }
    /// ```
    #[cfg(feature = "linalg")]
    pub fn qr(&self) -> Result<(DenseND<T>, DenseND<T>)> {
        if self.rank() != 2 {
            bail!(
                "QR decomposition requires 2D matrix, got {}D tensor",
                self.rank()
            );
        }

        // Convert to ndarray Array2
        let matrix: Array2<T> = self
            .data
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| anyhow::anyhow!("Failed to convert to 2D array: {}", e))?;

        // Call scirs2-linalg QR decomposition
        let (q_array, r_array) = scirs2_linalg::qr(&matrix.view(), None)
            .map_err(|e| anyhow::anyhow!("QR decomposition failed: {}", e))?;

        // Convert back to DenseND
        let q = DenseND::from_array(q_array.into_dyn());
        let r = DenseND::from_array(r_array.into_dyn());

        Ok((q, r))
    }

    /// Compute Singular Value Decomposition: A = U Σ V^T
    ///
    /// Decomposes a matrix into three matrices: left singular vectors (U),
    /// singular values (Σ), and right singular vectors (V).
    ///
    /// # Arguments
    ///
    /// * `self` - Input matrix (must be 2D)
    /// * `compute_uv` - If true, compute U and V^T; if false, only compute singular values
    ///
    /// # Returns
    ///
    /// * `Ok((U, S, Vt))` - Tuple of left singular vectors, singular values, and right singular vectors (transposed)
    /// * `Err` - If matrix is not 2D or if scirs2-linalg feature is not enabled
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// let a = DenseND::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let (u, s, vt) = a.svd(true).unwrap();
    /// assert_eq!(u.shape(), &[2, 2]);
    /// assert_eq!(s.shape(), &[2]);
    /// assert_eq!(vt.shape(), &[3, 3]);
    ///
    /// // Verify that singular values are sorted in descending order
    /// assert!(s[&[0]] >= s[&[1]]);
    /// # }
    /// ```
    #[cfg(feature = "linalg")]
    pub fn svd(&self, compute_uv: bool) -> Result<(DenseND<T>, DenseND<T>, DenseND<T>)> {
        if self.rank() != 2 {
            bail!("SVD requires 2D matrix, got {}D tensor", self.rank());
        }

        // Convert to ndarray Array2
        let matrix: Array2<T> = self
            .data
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| anyhow::anyhow!("Failed to convert to 2D array: {}", e))?;

        // Call scirs2-linalg SVD
        let (u_array, s_array, vt_array) = scirs2_linalg::svd(&matrix.view(), compute_uv, None)
            .map_err(|e| anyhow::anyhow!("SVD failed: {}", e))?;

        // Convert back to DenseND
        let u = DenseND::from_array(u_array.into_dyn());
        let s = DenseND::from_array(s_array.into_dyn());
        let vt = DenseND::from_array(vt_array.into_dyn());

        Ok((u, s, vt))
    }

    /// Compute eigenvalues and eigenvectors of a general matrix
    ///
    /// Solves the eigenvalue problem: A v = λ v
    /// Note: For general matrices, eigenvalues and eigenvectors can be complex.
    /// For real symmetric matrices, use `eigh()` instead.
    ///
    /// # Arguments
    ///
    /// * `self` - Input matrix (must be square 2D)
    ///
    /// # Returns
    ///
    /// * `Ok((eigenvalues, eigenvectors))` - Complex eigenvalues (1D) and eigenvectors (2D)
    /// * `Err` - If matrix is not square or if scirs2-linalg feature is not enabled
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// // Example: general matrix (may have complex eigenvalues)
    /// let a = DenseND::from_vec(
    ///     vec![0.0, -1.0, 1.0, 0.0],
    ///     &[2, 2]
    /// ).unwrap();
    ///
    /// let (eigenvalues, eigenvectors) = a.eig().unwrap();
    /// assert_eq!(eigenvalues.shape(), &[2]);
    /// assert_eq!(eigenvectors.shape(), &[2, 2]);
    /// # }
    /// ```
    #[cfg(feature = "linalg")]
    pub fn eig(&self) -> Result<EigResult<T>> {
        if self.rank() != 2 {
            bail!(
                "Eigendecomposition requires 2D matrix, got {}D tensor",
                self.rank()
            );
        }

        if !self.is_square() {
            bail!("Eigendecomposition requires square matrix");
        }

        // Convert to ndarray Array2
        let matrix: Array2<T> = self
            .data
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| anyhow::anyhow!("Failed to convert to 2D array: {}", e))?;

        // Call scirs2-linalg eigendecomposition
        let (eigenvalues_array, eigenvectors_array) = scirs2_linalg::eig(&matrix.view(), None)
            .map_err(|e| anyhow::anyhow!("Eigendecomposition failed: {}", e))?;

        // Convert back to DenseND
        let eigenvalues = DenseND::from_array(eigenvalues_array.into_dyn());
        let eigenvectors = DenseND::from_array(eigenvectors_array.into_dyn());

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute eigenvalues and eigenvectors of a Hermitian (symmetric for real matrices) matrix
    ///
    /// Solves the symmetric eigenvalue problem: A v = λ v
    /// For Hermitian/symmetric matrices, all eigenvalues are real and eigenvectors are orthogonal.
    ///
    /// # Arguments
    ///
    /// * `self` - Input Hermitian/symmetric matrix (must be square 2D)
    ///
    /// # Returns
    ///
    /// * `Ok((eigenvalues, eigenvectors))` - Real eigenvalues (1D) and orthogonal eigenvectors (2D)
    /// * `Err` - If matrix is not square or if scirs2-linalg feature is not enabled
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// // Symmetric positive definite matrix
    /// let a = DenseND::from_vec(
    ///     vec![4.0, 2.0, 2.0, 3.0],
    ///     &[2, 2]
    /// ).unwrap();
    ///
    /// let (eigenvalues, eigenvectors) = a.eigh().unwrap();
    /// assert_eq!(eigenvalues.shape(), &[2]);
    /// assert_eq!(eigenvectors.shape(), &[2, 2]);
    ///
    /// // All eigenvalues should be real and positive
    /// assert!(eigenvalues[&[0]] > 0.0);
    /// assert!(eigenvalues[&[1]] > 0.0);
    /// # }
    /// ```
    #[cfg(feature = "linalg")]
    pub fn eigh(&self) -> Result<(DenseND<T>, DenseND<T>)> {
        if self.rank() != 2 {
            bail!(
                "Hermitian eigendecomposition requires 2D matrix, got {}D tensor",
                self.rank()
            );
        }

        if !self.is_square() {
            bail!("Hermitian eigendecomposition requires square matrix");
        }

        // Convert to ndarray Array2
        let matrix: Array2<T> = self
            .data
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| anyhow::anyhow!("Failed to convert to 2D array: {}", e))?;

        // Call scirs2-linalg Hermitian eigendecomposition
        let (eigenvalues_array, eigenvectors_array) = scirs2_linalg::eigh(&matrix.view(), None)
            .map_err(|e| anyhow::anyhow!("Hermitian eigendecomposition failed: {}", e))?;

        // Convert back to DenseND
        let eigenvalues = DenseND::from_array(eigenvalues_array.into_dyn());
        let eigenvectors = DenseND::from_array(eigenvectors_array.into_dyn());

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute Cholesky decomposition: A = L L^T
    ///
    /// Decomposes a symmetric positive definite matrix into a lower triangular matrix L.
    ///
    /// # Arguments
    ///
    /// * `self` - Input symmetric positive definite matrix (must be square 2D)
    ///
    /// # Returns
    ///
    /// * `Ok(L)` - Lower triangular matrix L such that A = L L^T
    /// * `Err` - If matrix is not square, not positive definite, or if scirs2-linalg feature is not enabled
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// // Symmetric positive definite matrix
    /// let a: DenseND<f64> = DenseND::from_vec(
    ///     vec![4.0, 2.0, 2.0, 3.0],
    ///     &[2, 2]
    /// ).unwrap();
    ///
    /// let l = a.cholesky().unwrap();
    /// assert_eq!(l.shape(), &[2, 2]);
    ///
    /// // Verify: A = L L^T
    /// let reconstructed = l.matmul(&l.transpose().unwrap()).unwrap();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         let expected: f64 = a[&[i, j]];
    ///         let actual: f64 = reconstructed[&[i, j]];
    ///         assert!((actual - expected).abs() < 1e-10);
    ///     }
    /// }
    /// # }
    /// ```
    #[cfg(feature = "linalg")]
    pub fn cholesky(&self) -> Result<DenseND<T>> {
        if self.rank() != 2 {
            bail!(
                "Cholesky decomposition requires 2D matrix, got {}D tensor",
                self.rank()
            );
        }

        if !self.is_square() {
            bail!("Cholesky decomposition requires square matrix");
        }

        // Convert to ndarray Array2
        let matrix: Array2<T> = self
            .data
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| anyhow::anyhow!("Failed to convert to 2D array: {}", e))?;

        // Call scirs2-linalg Cholesky decomposition
        let l_array = scirs2_linalg::cholesky(&matrix.view(), None).map_err(|e| {
            anyhow::anyhow!(
                "Cholesky decomposition failed (matrix may not be positive definite): {}",
                e
            )
        })?;

        // Convert back to DenseND
        let l = DenseND::from_array(l_array.into_dyn());

        Ok(l)
    }
}

// Stub implementations when linalg feature is not enabled
#[cfg(not(feature = "linalg"))]
impl<T> DenseND<T>
where
    T: Float + FromPrimitive + NumCast + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    /// QR decomposition (requires `linalg` feature)
    pub fn qr(&self) -> Result<(DenseND<T>, DenseND<T>)> {
        bail!("QR decomposition requires the 'linalg' feature to be enabled")
    }

    /// SVD (requires `linalg` feature)
    pub fn svd(&self, _compute_uv: bool) -> Result<(DenseND<T>, DenseND<T>, DenseND<T>)> {
        bail!("SVD requires the 'linalg' feature to be enabled")
    }

    /// Eigendecomposition (requires `linalg` feature)
    pub fn eig(&self) -> Result<EigResult<T>> {
        bail!("Eigendecomposition requires the 'linalg' feature to be enabled")
    }

    /// Hermitian eigendecomposition (requires `linalg` feature)
    pub fn eigh(&self) -> Result<(DenseND<T>, DenseND<T>)> {
        bail!("Hermitian eigendecomposition requires the 'linalg' feature to be enabled")
    }

    /// Cholesky decomposition (requires `linalg` feature)
    pub fn cholesky(&self) -> Result<DenseND<T>> {
        bail!("Cholesky decomposition requires the 'linalg' feature to be enabled")
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "linalg")]
    use super::DenseND;

    #[test]
    #[cfg(feature = "linalg")]
    fn test_qr_decomposition() {
        let a = DenseND::from_vec(
            vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            &[3, 3],
        )
        .unwrap();

        let (q, r) = a.qr().unwrap();
        assert_eq!(q.shape(), &[3, 3]);
        assert_eq!(r.shape(), &[3, 3]);

        // Q should be orthogonal: Q^T Q ≈ I
        let identity_check = q.transpose().unwrap().matmul(&q).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                let actual: f64 = identity_check[&[i, j]];
                assert!((actual - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_svd() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let (u, s, vt) = a.svd(true).unwrap();
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[3, 3]);

        // Singular values should be sorted in descending order
        assert!(s[&[0]] >= s[&[1]]);
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_eigh_symmetric() {
        // Symmetric matrix
        let a = DenseND::from_vec(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]).unwrap();

        let (eigenvalues, eigenvectors) = a.eigh().unwrap();
        assert_eq!(eigenvalues.shape(), &[2]);
        assert_eq!(eigenvectors.shape(), &[2, 2]);

        // For symmetric positive definite, all eigenvalues should be positive
        assert!(eigenvalues[&[0]] > 0.0);
        assert!(eigenvalues[&[1]] > 0.0);
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_cholesky() {
        // Symmetric positive definite matrix
        let a = DenseND::from_vec(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]).unwrap();

        let l = a.cholesky().unwrap();
        assert_eq!(l.shape(), &[2, 2]);

        // Verify: A = L L^T
        let reconstructed = l.matmul(&l.transpose().unwrap()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected: f64 = a[&[i, j]];
                let actual: f64 = reconstructed[&[i, j]];
                assert!((actual - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_qr_requires_2d() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(a.qr().is_err());
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_svd_requires_2d() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(a.svd(true).is_err());
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_eig_requires_square() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert!(a.eig().is_err());
    }
}
