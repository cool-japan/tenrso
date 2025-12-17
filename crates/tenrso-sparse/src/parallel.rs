//! Parallel sparse tensor format conversions and operations
//!
//! This module provides parallel implementations of format conversions
//! and operations for improved performance on large sparse tensors.
//!
//! # Features
//!
//! - Parallel COO → CSR/CSC conversion with parallel sorting and counting
//! - Parallel Dense → COO conversion with parallel threshold filtering
//! - Parallel element-wise operations
//! - Automatic fallback to sequential when `parallel` feature is disabled
//!
//! # Performance
//!
//! Parallel versions typically show speedup for:
//! - Tensors with nnz > 100,000
//! - Dense-to-sparse conversions with large arrays
//! - Multi-threaded environments
//!
//! For small tensors (nnz < 10,000), sequential versions may be faster
//! due to threading overhead.
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{CooTensor, CsrMatrix};
//! use tenrso_sparse::parallel::par_coo_to_csr;
//!
//! // Create a large COO tensor
//! let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
//! let values = vec![1.0, 2.0, 3.0];
//! let shape = vec![1000, 1000];
//! let coo = CooTensor::new(indices, values, shape).unwrap();
//!
//! // Convert to CSR in parallel
//! let csr = par_coo_to_csr(&coo).unwrap();
//! ```

#![allow(unused_imports)]

use crate::{CooTensor, CscError, CscMatrix, CsrError, CsrMatrix};
use scirs2_core::numeric::Float;
use tenrso_core::DenseND;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

// Import Dimension trait for as_array_view()
use scirs2_core::ndarray_ext::Dimension;

/// Parallel COO → CSR conversion
///
/// Converts a COO tensor to CSR format using parallel operations for:
/// - Sorting coordinates
/// - Counting elements per row
/// - Building CSR structure
///
/// # Complexity
///
/// O(nnz × log(nnz)) for sorting + O(nnz) for construction
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CooTensor, parallel::par_coo_to_csr};
///
/// let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
/// let values = vec![1.0, 2.0, 3.0];
/// let shape = vec![3, 4];
/// let coo = CooTensor::new(indices, values, shape).unwrap();
///
/// let csr = par_coo_to_csr(&coo).unwrap();
/// assert_eq!(csr.nnz(), 3);
/// ```
#[cfg(feature = "parallel")]
pub fn par_coo_to_csr<T>(coo: &CooTensor<T>) -> Result<CsrMatrix<T>, CsrError>
where
    T: Clone + Send + Sync + Float,
{
    if coo.rank() != 2 {
        return Err(CsrError::InvalidShape(format!(
            "COO tensor must be 2D, got {}D",
            coo.rank()
        )));
    }

    let nrows = coo.shape()[0];
    let ncols = coo.shape()[1];
    let nnz = coo.nnz();

    // Parallel sort COO by row then column
    let mut perm: Vec<usize> = (0..nnz).collect();
    perm.par_sort_by(|&i, &j| {
        coo.indices()[i][0]
            .cmp(&coo.indices()[j][0])
            .then_with(|| coo.indices()[i][1].cmp(&coo.indices()[j][1]))
    });

    // Build CSR structure
    let mut row_ptr = vec![0; nrows + 1];

    // Parallel count elements per row
    let row_counts: Vec<usize> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            perm.iter()
                .filter(|&&idx| coo.indices()[idx][0] == row)
                .count()
        })
        .collect();

    // Sequential cumulative sum for row_ptr (must be sequential)
    for i in 0..nrows {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
    }

    // Parallel fill col_indices and values using collect
    let (col_indices, values): (Vec<_>, Vec<_>) = (0..nnz)
        .into_par_iter()
        .map(|i| {
            let idx = perm[i];
            (coo.indices()[idx][1], coo.values()[idx])
        })
        .unzip();

    CsrMatrix::new(row_ptr, col_indices, values, (nrows, ncols))
}

/// Sequential fallback for par_coo_to_csr when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub fn par_coo_to_csr<T>(coo: &CooTensor<T>) -> Result<CsrMatrix<T>, CsrError>
where
    T: Clone + Float,
{
    CsrMatrix::from_coo(coo)
}

/// Parallel COO → CSC conversion
///
/// Converts a COO tensor to CSC format using parallel operations.
///
/// # Complexity
///
/// O(nnz × log(nnz)) for sorting + O(nnz) for construction
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CooTensor, parallel::par_coo_to_csc};
///
/// let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
/// let values = vec![1.0, 2.0, 3.0];
/// let shape = vec![3, 4];
/// let coo = CooTensor::new(indices, values, shape).unwrap();
///
/// let csc = par_coo_to_csc(&coo).unwrap();
/// assert_eq!(csc.nnz(), 3);
/// ```
#[cfg(feature = "parallel")]
pub fn par_coo_to_csc<T>(coo: &CooTensor<T>) -> Result<CscMatrix<T>, CscError>
where
    T: Clone + Send + Sync + Float,
{
    if coo.rank() != 2 {
        return Err(CscError::InvalidShape(format!(
            "COO tensor must be 2D, got {}D",
            coo.rank()
        )));
    }

    let nrows = coo.shape()[0];
    let ncols = coo.shape()[1];
    let nnz = coo.nnz();

    // Parallel sort COO by column then row
    let mut perm: Vec<usize> = (0..nnz).collect();
    perm.par_sort_by(|&i, &j| {
        coo.indices()[i][1]
            .cmp(&coo.indices()[j][1])
            .then_with(|| coo.indices()[i][0].cmp(&coo.indices()[j][0]))
    });

    // Build CSC structure
    let mut col_ptr = vec![0; ncols + 1];

    // Parallel count elements per column
    let col_counts: Vec<usize> = (0..ncols)
        .into_par_iter()
        .map(|col| {
            perm.iter()
                .filter(|&&idx| coo.indices()[idx][1] == col)
                .count()
        })
        .collect();

    // Sequential cumulative sum for col_ptr
    for i in 0..ncols {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i];
    }

    // Parallel fill row_indices and values using collect
    let (row_indices, values): (Vec<_>, Vec<_>) = (0..nnz)
        .into_par_iter()
        .map(|i| {
            let idx = perm[i];
            (coo.indices()[idx][0], coo.values()[idx])
        })
        .unzip();

    CscMatrix::new(col_ptr, row_indices, values, (nrows, ncols))
}

/// Sequential fallback for par_coo_to_csc when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub fn par_coo_to_csc<T>(coo: &CooTensor<T>) -> Result<CscMatrix<T>, CscError>
where
    T: Clone + Float,
{
    CscMatrix::from_coo(coo)
}

/// Parallel Dense → COO conversion
///
/// Converts a dense tensor to COO format by filtering elements with
/// absolute value greater than threshold.
///
/// Uses parallel iteration for large tensors to find non-zero elements.
///
/// # Complexity
///
/// O(numel) where numel = product of shape dimensions
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::parallel::par_dense_to_coo;
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
/// let dense = DenseND::from_array(data.into_dyn());
///
/// let coo = par_dense_to_coo(&dense, 0.5);
/// assert_eq!(coo.nnz(), 3); // Three elements > 0.5
/// ```
#[cfg(feature = "parallel")]
pub fn par_dense_to_coo<T>(dense: &DenseND<T>, threshold: T) -> CooTensor<T>
where
    T: Clone + Send + Sync + Float,
{
    use scirs2_core::ndarray_ext::{Axis, IxDyn};

    let shape = dense.shape().to_vec();
    let data = dense.as_array();

    // Parallel collect non-zero elements
    let non_zeros: Vec<(Vec<usize>, T)> = data
        .indexed_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter_map(|(idx, &value)| {
            if value.abs() > threshold {
                let indices = idx.as_array_view().iter().copied().collect();
                Some((indices, value))
            } else {
                None
            }
        })
        .collect();

    let (indices, values): (Vec<_>, Vec<_>) = non_zeros.into_iter().unzip();

    CooTensor::new(indices, values, shape).expect("Valid COO construction from dense")
}

/// Sequential fallback for par_dense_to_coo when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub fn par_dense_to_coo<T>(dense: &DenseND<T>, threshold: T) -> CooTensor<T>
where
    T: Clone + Float,
{
    CooTensor::from_dense(dense, threshold)
}

/// Parallel CSR transpose (CSR → CSC)
///
/// More efficient than sequential version for large sparse matrices.
///
/// # Complexity
///
/// O(nnz) with parallel column counting
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, parallel::par_csr_transpose};
///
/// let row_ptr = vec![0, 2, 3, 3];
/// let col_indices = vec![0, 2, 1];
/// let values = vec![1.0, 2.0, 3.0];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 4)).unwrap();
///
/// let csc = par_csr_transpose(&csr);
/// assert_eq!(csc.nnz(), 3);
/// ```
#[cfg(feature = "parallel")]
pub fn par_csr_transpose<T>(csr: &CsrMatrix<T>) -> CscMatrix<T>
where
    T: Clone + Send + Sync + Float,
{
    let nrows = csr.nrows();
    let ncols = csr.ncols();
    let nnz = csr.nnz();

    let mut col_ptr = vec![0; ncols + 1];
    let mut row_indices = vec![0; nnz];
    let mut values = vec![T::zero(); nnz];

    // Parallel count elements per column
    let col_counts: Vec<usize> = (0..ncols)
        .into_par_iter()
        .map(|col| csr.col_indices().iter().filter(|&&c| c == col).count())
        .collect();

    // Sequential cumulative sum
    for i in 0..ncols {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i];
    }

    // Build position tracker for each column
    let mut col_pos = col_ptr[..ncols].to_vec();

    // Fill CSC structure (must be sequential for correctness)
    for row in 0..nrows {
        let start = csr.row_ptr()[row];
        let end = csr.row_ptr()[row + 1];

        for idx in start..end {
            let col = csr.col_indices()[idx];
            let pos = col_pos[col];

            row_indices[pos] = row;
            values[pos] = csr.values()[idx];
            col_pos[col] += 1;
        }
    }

    CscMatrix::new(col_ptr, row_indices, values, (nrows, ncols))
        .expect("Valid CSC construction from CSR")
}

/// Sequential fallback for par_csr_transpose when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub fn par_csr_transpose<T>(csr: &CsrMatrix<T>) -> CscMatrix<T>
where
    T: Clone + Float,
{
    csr.to_csc()
}

/// Parallel sort COO tensor in-place
///
/// Sorts the COO tensor by coordinates in row-major (lexicographic) order
/// using parallel sorting.
///
/// # Complexity
///
/// O(nnz × log(nnz)) with parallel sorting
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CooTensor, parallel::par_sort_coo};
///
/// let indices = vec![vec![2, 0], vec![0, 1], vec![1, 2]];
/// let values = vec![3.0, 1.0, 2.0];
/// let shape = vec![3, 4];
/// let mut coo = CooTensor::new(indices, values, shape).unwrap();
///
/// par_sort_coo(&mut coo);
/// // Now sorted: (0,1), (1,2), (2,0)
/// ```
#[cfg(feature = "parallel")]
pub fn par_sort_coo<T>(coo: &mut CooTensor<T>)
where
    T: Clone + Send + Sync,
{
    let nnz = coo.nnz();

    // Create index permutation
    let mut perm: Vec<usize> = (0..nnz).collect();

    // Parallel sort by indices in row-major order
    perm.par_sort_by(|&i, &j| coo.indices()[i].cmp(&coo.indices()[j]));

    // Apply permutation using parallel map and collect
    let old_indices = coo.indices().to_vec();
    let old_values = coo.values().to_vec();

    let (new_indices, new_values): (Vec<_>, Vec<_>) = perm
        .par_iter()
        .map(|&old_idx| (old_indices[old_idx].clone(), old_values[old_idx].clone()))
        .unzip();

    // Now update coo with the sorted data
    // Update indices first
    {
        let indices_mut = coo.indices_mut();
        for (dst, src) in indices_mut.iter_mut().zip(new_indices.iter()) {
            *dst = src.clone();
        }
    }
    // Then update values
    {
        let values_mut = coo.values_mut();
        for (dst, src) in values_mut.iter_mut().zip(new_values.iter()) {
            *dst = src.clone();
        }
    }
}

/// Sequential fallback for par_sort_coo when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub fn par_sort_coo<T>(coo: &mut CooTensor<T>)
where
    T: Clone,
{
    coo.sort();
}

/// Parallel Sparse Matrix-Vector multiplication (SpMV)
///
/// Computes y = A * x where A is a sparse CSR matrix and x is a dense vector.
/// Uses parallel iteration over rows for improved performance on large matrices.
///
/// # Complexity
///
/// O(nnz) with parallel row processing
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, parallel::par_spmv};
/// use scirs2_core::ndarray_ext::array;
///
/// let row_ptr = vec![0, 2, 3];
/// let col_indices = vec![0, 2, 1];
/// let values = vec![1.0, 2.0, 3.0];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();
///
/// let x = array![1.0, 2.0, 3.0];
/// let y = par_spmv(&csr, &x.view()).unwrap();
/// assert_eq!(y.len(), 2);
/// ```
#[cfg(feature = "parallel")]
pub fn par_spmv<T>(
    csr: &CsrMatrix<T>,
    x: &scirs2_core::ndarray_ext::ArrayView1<T>,
) -> Result<scirs2_core::ndarray_ext::Array1<T>, CsrError>
where
    T: Clone + Send + Sync + Float,
{
    use scirs2_core::ndarray_ext::Array1;

    // Validate dimensions
    if x.len() != csr.ncols() {
        return Err(CsrError::ShapeMismatch {
            nrows: csr.nrows(),
            ncols: csr.ncols(),
            vec_len: x.len(),
        });
    }

    let nrows = csr.nrows();

    // Parallel compute each row of the result
    let result: Vec<T> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            let start = csr.row_ptr()[row];
            let end = csr.row_ptr()[row + 1];

            let mut sum = T::zero();
            for idx in start..end {
                let col = csr.col_indices()[idx];
                let val = csr.values()[idx];
                sum = sum + val * x[col];
            }
            sum
        })
        .collect();

    Ok(Array1::from(result))
}

/// Sequential fallback for par_spmv when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub fn par_spmv<T>(
    csr: &CsrMatrix<T>,
    x: &scirs2_core::ndarray_ext::ArrayView1<T>,
) -> Result<scirs2_core::ndarray_ext::Array1<T>, CsrError>
where
    T: Clone + Float,
{
    csr.spmv(x)
}

/// Parallel Sparse Matrix-Matrix multiplication (SpMM)
///
/// Computes C = A * B where A is a sparse CSR matrix and B is a dense matrix.
/// Uses parallel iteration over rows of A for improved performance.
///
/// # Complexity
///
/// O(nnz * k) where k is the number of columns in B, with parallel row processing
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, parallel::par_spmm};
/// use scirs2_core::ndarray_ext::array;
///
/// let row_ptr = vec![0, 2, 3];
/// let col_indices = vec![0, 2, 1];
/// let values = vec![1.0, 2.0, 3.0];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();
///
/// let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let c = par_spmm(&csr, &b.view()).unwrap();
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
#[cfg(feature = "parallel")]
pub fn par_spmm<T>(
    csr: &CsrMatrix<T>,
    b: &scirs2_core::ndarray_ext::ArrayView2<T>,
) -> Result<scirs2_core::ndarray_ext::Array2<T>, CsrError>
where
    T: Clone + Send + Sync + Float,
{
    use scirs2_core::ndarray_ext::Array2;

    let (b_rows, b_cols) = (b.nrows(), b.ncols());

    // Validate dimensions: A is (m, n), B is (n, k) -> C is (m, k)
    if csr.ncols() != b_rows {
        return Err(CsrError::MatrixShapeMismatch {
            m1: csr.nrows(),
            n1: csr.ncols(),
            m2: b_rows,
            n2: b_cols,
        });
    }

    let nrows = csr.nrows();

    // Parallel compute each row of the result
    let result_data: Vec<T> = (0..nrows)
        .into_par_iter()
        .flat_map(|row| {
            let start = csr.row_ptr()[row];
            let end = csr.row_ptr()[row + 1];

            // Compute row of C
            let mut c_row = vec![T::zero(); b_cols];
            for idx in start..end {
                let col = csr.col_indices()[idx];
                let a_val = csr.values()[idx];

                // C[row, k] += A[row, col] * B[col, k]
                for k in 0..b_cols {
                    c_row[k] = c_row[k] + a_val * b[[col, k]];
                }
            }
            c_row
        })
        .collect();

    Array2::from_shape_vec((nrows, b_cols), result_data)
        .map_err(|e| CsrError::InvalidShape(e.to_string()))
}

/// Sequential fallback for par_spmm when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub fn par_spmm<T>(
    csr: &CsrMatrix<T>,
    b: &scirs2_core::ndarray_ext::ArrayView2<T>,
) -> Result<scirs2_core::ndarray_ext::Array2<T>, CsrError>
where
    T: Clone + Float,
{
    csr.spmm(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_par_coo_to_csr() {
        let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0], vec![0, 2]];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![3, 4];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let csr = par_coo_to_csr(&coo).unwrap();
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 4);

        // Verify structure
        assert_eq!(csr.row_ptr()[0], 0);
        assert_eq!(csr.row_ptr()[1], 2); // Row 0 has 2 elements
        assert_eq!(csr.row_ptr()[2], 3); // Row 1 has 1 element
        assert_eq!(csr.row_ptr()[3], 4); // Row 2 has 1 element
    }

    #[test]
    fn test_par_coo_to_csc() {
        let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0], vec![0, 2]];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![3, 4];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let csc = par_coo_to_csc(&coo).unwrap();
        assert_eq!(csc.nnz(), 4);
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 4);
    }

    #[test]
    fn test_par_dense_to_coo() {
        let data = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let dense = DenseND::from_array(data.into_dyn());

        let coo = par_dense_to_coo(&dense, 0.5);
        assert_eq!(coo.nnz(), 3);
    }

    #[test]
    fn test_par_csr_transpose() {
        let row_ptr = vec![0, 2, 3, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 4)).unwrap();

        let csc = par_csr_transpose(&csr);
        assert_eq!(csc.nnz(), 3);
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 4);
    }

    #[test]
    fn test_par_sort_coo() {
        let indices = vec![vec![2, 0], vec![0, 1], vec![1, 2]];
        let values = vec![3.0, 1.0, 2.0];
        let shape = vec![3, 4];
        let mut coo = CooTensor::new(indices, values, shape).unwrap();

        par_sort_coo(&mut coo);

        // Check sorted order
        assert_eq!(coo.indices()[0], vec![0, 1]);
        assert_eq!(coo.indices()[1], vec![1, 2]);
        assert_eq!(coo.indices()[2], vec![2, 0]);
    }

    #[test]
    fn test_par_coo_to_csr_large() {
        // Test with a larger matrix to benefit from parallelization
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..1000 {
            for j in 0..100 {
                if (i + j) % 10 == 0 {
                    indices.push(vec![i, j]);
                    values.push((i + j) as f64);
                }
            }
        }

        let shape = vec![1000, 100];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let csr = par_coo_to_csr(&coo).unwrap();
        assert_eq!(csr.nrows(), 1000);
        assert_eq!(csr.ncols(), 100);
        assert!(csr.nnz() > 0);
    }

    #[test]
    fn test_par_conversions_correctness() {
        // Verify parallel conversions match sequential results
        let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 4];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let csr_par = par_coo_to_csr(&coo).unwrap();
        let csr_seq = CsrMatrix::from_coo(&coo).unwrap();

        assert_eq!(csr_par.nnz(), csr_seq.nnz());
        assert_eq!(csr_par.row_ptr(), csr_seq.row_ptr());
        assert_eq!(csr_par.col_indices(), csr_seq.col_indices());
    }

    #[test]
    fn test_par_spmv_basic() {
        // Create a simple CSR matrix
        // [1  0  2]
        // [0  3  0]
        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = par_spmv(&csr, &x.view()).unwrap();

        // Expected: [1*1 + 2*3, 3*2] = [7.0, 6.0]
        assert_eq!(y.len(), 2);
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_par_spmv_empty_row() {
        // Matrix with an empty row
        let row_ptr = vec![0, 2, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = par_spmv(&csr, &x.view()).unwrap();

        assert_eq!(y.len(), 3);
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 0.0).abs() < 1e-10); // Empty row
        assert!((y[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_par_spmv_shape_mismatch() {
        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        let x = array![1.0, 2.0]; // Wrong size
        let result = par_spmv(&csr, &x.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_par_spmv_correctness() {
        // Verify parallel SpMV matches sequential
        let row_ptr = vec![0, 2, 4, 5];
        let col_indices = vec![0, 2, 1, 2, 0];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let x = array![1.0, 2.0, 3.0];

        let y_par = par_spmv(&csr, &x.view()).unwrap();
        let y_seq = csr.spmv(&x.view()).unwrap();

        assert_eq!(y_par.len(), y_seq.len());
        for i in 0..y_par.len() {
            assert!((y_par[i] - y_seq[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_par_spmm_basic() {
        // Create a simple CSR matrix
        // [1  0  2]
        // [0  3  0]
        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = par_spmm(&csr, &b.view()).unwrap();

        // Expected:
        // Row 0: [1*1 + 2*5, 1*2 + 2*6] = [11, 14]
        // Row 1: [3*3, 3*4] = [9, 12]
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c[[0, 0]] - 11.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 14.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 9.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_par_spmm_single_column() {
        // SpMM with single column should match SpMV
        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        let b = array![[1.0], [2.0], [3.0]];
        let c = par_spmm(&csr, &b.view()).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = par_spmv(&csr, &x.view()).unwrap();

        assert_eq!(c.shape(), &[2, 1]);
        for i in 0..2 {
            assert!((c[[i, 0]] - y[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_par_spmm_shape_mismatch() {
        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        let b = array![[1.0, 2.0], [3.0, 4.0]]; // Wrong size (2x2 instead of 3xk)
        let result = par_spmm(&csr, &b.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_par_spmm_correctness() {
        // Verify parallel SpMM matches sequential
        let row_ptr = vec![0, 2, 4, 5];
        let col_indices = vec![0, 2, 1, 2, 0];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let c_par = par_spmm(&csr, &b.view()).unwrap();
        let c_seq = csr.spmm(&b.view()).unwrap();

        assert_eq!(c_par.shape(), c_seq.shape());
        for i in 0..c_par.nrows() {
            for j in 0..c_par.ncols() {
                assert!((c_par[[i, j]] - c_seq[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_par_spmm_large() {
        // Test with a larger matrix to benefit from parallelization
        let mut row_ptr = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..100 {
            for j in 0..10 {
                if (i + j) % 3 == 0 {
                    col_indices.push(j);
                    values.push((i + j) as f64);
                }
            }
            row_ptr.push(col_indices.len());
        }

        let csr = CsrMatrix::new(row_ptr, col_indices, values, (100, 10)).unwrap();
        let b = scirs2_core::ndarray_ext::Array2::<f64>::zeros((10, 5));

        let c = par_spmm(&csr, &b.view()).unwrap();
        assert_eq!(c.shape(), &[100, 5]);
    }
}
