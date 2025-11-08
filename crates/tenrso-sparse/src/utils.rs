//! Utility functions for sparse tensor operations
//!
//! This module provides helper functions for:
//! - Sparsity statistics and analysis
//! - Format selection heuristics
//! - Memory footprint calculations
//! - Conversion utilities
//! - Performance estimation
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::utils::{recommend_format, SparsityStats, FormatRecommendation};
//!
//! let stats = SparsityStats {
//!     nnz: 1000,
//!     total_elements: 1_000_000,
//!     shape: vec![1000, 1000],
//!     density: 0.001,
//! };
//!
//! let rec = recommend_format(&stats);
//! match rec {
//!     FormatRecommendation::COO => println!("Use COO format"),
//!     FormatRecommendation::CSR => println!("Use CSR format"),
//!     _ => {}
//! }
//! ```

use scirs2_core::numeric::Float;

/// Sparsity statistics for a tensor
#[derive(Debug, Clone)]
pub struct SparsityStats {
    /// Number of nonzero elements
    pub nnz: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Density (nnz / total_elements)
    pub density: f64,
}

impl SparsityStats {
    /// Compute sparsity statistics from shape and nnz
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::utils::SparsityStats;
    ///
    /// let stats = SparsityStats::from_shape_nnz(vec![100, 100], 500);
    /// assert_eq!(stats.nnz, 500);
    /// assert_eq!(stats.total_elements, 10000);
    /// assert!((stats.density - 0.05).abs() < 1e-10);
    /// ```
    pub fn from_shape_nnz(shape: Vec<usize>, nnz: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let density = if total_elements > 0 {
            nnz as f64 / total_elements as f64
        } else {
            0.0
        };

        Self {
            nnz,
            total_elements,
            shape,
            density,
        }
    }

    /// Returns true if the tensor is extremely sparse (< 0.01% density)
    pub fn is_extremely_sparse(&self) -> bool {
        self.density < 0.0001
    }

    /// Returns true if the tensor is very sparse (< 1% density)
    pub fn is_very_sparse(&self) -> bool {
        self.density < 0.01
    }

    /// Returns true if the tensor is moderately sparse (1% - 10% density)
    pub fn is_moderately_sparse(&self) -> bool {
        self.density >= 0.01 && self.density < 0.1
    }

    /// Returns true if the tensor should be considered dense (>= 10% density)
    pub fn should_use_dense(&self) -> bool {
        self.density >= 0.1
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Check if tensor has block structure (useful for BCSR)
    ///
    /// Returns (has_blocks, suggested_block_size)
    pub fn has_block_structure(&self, min_block_size: usize) -> (bool, Vec<usize>) {
        // Heuristic: if shape dimensions are divisible by min_block_size
        // and density suggests clustered nonzeros
        if self.ndim() != 2 {
            return (false, vec![]);
        }

        let block_sizes: Vec<usize> = self
            .shape
            .iter()
            .map(|&dim| {
                for bs in (min_block_size..=dim.min(64)).rev() {
                    if dim % bs == 0 {
                        return bs;
                    }
                }
                1
            })
            .collect();

        let has_blocks = block_sizes.iter().all(|&bs| bs > 1) && self.density > 0.001;
        (has_blocks, block_sizes)
    }
}

/// Recommended sparse format based on tensor properties
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatRecommendation {
    /// Use dense format (>= 10% density)
    Dense,
    /// Use COO format (very flexible, good for construction)
    COO,
    /// Use CSR format (good for row-wise operations, 2D)
    CSR,
    /// Use CSC format (good for column-wise operations, 2D)
    CSC,
    /// Use BCSR format (block-structured sparsity, 2D)
    BCSR { block_shape: Vec<usize> },
    /// Use CSF format (N-D hierarchical, very sparse)
    CSF { mode_order: Vec<usize> },
    /// Use HiCOO format (N-D blocked, cache-efficient)
    HiCOO { block_shape: Vec<usize> },
}

/// Recommend sparse format based on tensor statistics
///
/// # Arguments
///
/// - `stats`: Sparsity statistics of the tensor
///
/// # Returns
///
/// Recommended format for optimal performance
///
/// # Examples
///
/// ```
/// use tenrso_sparse::utils::{recommend_format, SparsityStats, FormatRecommendation};
///
/// // Very sparse 2D matrix - recommend CSR
/// let stats = SparsityStats::from_shape_nnz(vec![1000, 1000], 500);
/// let rec = recommend_format(&stats);
/// assert_eq!(rec, FormatRecommendation::CSR);
///
/// // Dense-ish matrix - recommend Dense
/// let stats = SparsityStats::from_shape_nnz(vec![100, 100], 1500);
/// let rec = recommend_format(&stats);
/// assert_eq!(rec, FormatRecommendation::Dense);
/// ```
pub fn recommend_format(stats: &SparsityStats) -> FormatRecommendation {
    // Dense threshold
    if stats.should_use_dense() {
        return FormatRecommendation::Dense;
    }

    let ndim = stats.ndim();

    match ndim {
        0 | 1 => FormatRecommendation::Dense, // Scalars and vectors are better dense
        2 => {
            // 2D matrices
            // Check for block structure
            let (has_blocks, block_shape) = stats.has_block_structure(4);
            if has_blocks {
                return FormatRecommendation::BCSR { block_shape };
            }

            // Default to CSR for 2D sparse matrices
            FormatRecommendation::CSR
        }
        _ => {
            // N-dimensional tensors (N >= 3)
            if stats.is_extremely_sparse() {
                // Very sparse: CSF with natural mode order
                FormatRecommendation::CSF {
                    mode_order: (0..ndim).collect(),
                }
            } else if stats.is_very_sparse() {
                // Moderately sparse: HiCOO with reasonable block size
                let block_size = 4; // Default block size
                let block_shape = vec![block_size; ndim];
                FormatRecommendation::HiCOO { block_shape }
            } else {
                // Less sparse but still sparse: COO for flexibility
                FormatRecommendation::COO
            }
        }
    }
}

/// Memory footprint estimation for different formats
#[derive(Debug, Clone)]
pub struct MemoryFootprint {
    /// Format name
    pub format: String,
    /// Total bytes required
    pub bytes: usize,
    /// Breakdown by component
    pub breakdown: Vec<(String, usize)>,
}

impl MemoryFootprint {
    /// Estimate memory for COO format
    ///
    /// Memory = (ndim * nnz * sizeof(usize)) + (nnz * sizeof(T)) + (ndim * sizeof(usize))
    pub fn coo<T>(shape: &[usize], nnz: usize) -> Self {
        let ndim = shape.len();
        let index_bytes = ndim * nnz * std::mem::size_of::<usize>();
        let value_bytes = nnz * std::mem::size_of::<T>();
        let shape_bytes = std::mem::size_of_val(shape);

        Self {
            format: "COO".to_string(),
            bytes: index_bytes + value_bytes + shape_bytes,
            breakdown: vec![
                ("indices".to_string(), index_bytes),
                ("values".to_string(), value_bytes),
                ("shape".to_string(), shape_bytes),
            ],
        }
    }

    /// Estimate memory for CSR format
    ///
    /// Memory = ((m+1) * sizeof(usize)) + (nnz * sizeof(usize)) + (nnz * sizeof(T))
    pub fn csr<T>(nrows: usize, nnz: usize) -> Self {
        let row_ptr_bytes = (nrows + 1) * std::mem::size_of::<usize>();
        let col_idx_bytes = nnz * std::mem::size_of::<usize>();
        let value_bytes = nnz * std::mem::size_of::<T>();

        Self {
            format: "CSR".to_string(),
            bytes: row_ptr_bytes + col_idx_bytes + value_bytes,
            breakdown: vec![
                ("row_ptr".to_string(), row_ptr_bytes),
                ("col_indices".to_string(), col_idx_bytes),
                ("values".to_string(), value_bytes),
            ],
        }
    }

    /// Estimate memory for CSC format
    ///
    /// Memory = ((n+1) * sizeof(usize)) + (nnz * sizeof(usize)) + (nnz * sizeof(T))
    pub fn csc<T>(ncols: usize, nnz: usize) -> Self {
        let col_ptr_bytes = (ncols + 1) * std::mem::size_of::<usize>();
        let row_idx_bytes = nnz * std::mem::size_of::<usize>();
        let value_bytes = nnz * std::mem::size_of::<T>();

        Self {
            format: "CSC".to_string(),
            bytes: col_ptr_bytes + row_idx_bytes + value_bytes,
            breakdown: vec![
                ("col_ptr".to_string(), col_ptr_bytes),
                ("row_indices".to_string(), row_idx_bytes),
                ("values".to_string(), value_bytes),
            ],
        }
    }

    /// Estimate memory for dense format
    ///
    /// Memory = (total_elements * sizeof(T))
    pub fn dense<T>(shape: &[usize]) -> Self {
        let total_elements: usize = shape.iter().product();
        let bytes = total_elements * std::mem::size_of::<T>();

        Self {
            format: "Dense".to_string(),
            bytes,
            breakdown: vec![("array".to_string(), bytes)],
        }
    }

    /// Compression ratio compared to dense storage
    pub fn compression_ratio<T>(&self, shape: &[usize]) -> f64 {
        let dense_bytes = Self::dense::<T>(shape).bytes;
        if dense_bytes == 0 {
            return 1.0;
        }
        dense_bytes as f64 / self.bytes as f64
    }
}

/// Estimate FLOPs for sparse matrix-vector multiplication (SpMV)
///
/// # Complexity
///
/// O(2 * nnz) FLOPs (one multiply + one add per nonzero)
pub fn estimate_spmv_flops(nnz: usize) -> usize {
    2 * nnz
}

/// Estimate FLOPs for sparse matrix-matrix multiplication (SpMM)
///
/// # Complexity
///
/// O(2 * nnz * k) FLOPs where k is the number of columns in the dense matrix
pub fn estimate_spmm_flops(nnz: usize, k: usize) -> usize {
    2 * nnz * k
}

/// Estimate FLOPs for sparse-sparse matrix multiplication (SpSpMM)
///
/// # Complexity
///
/// O(m * nnz_per_row_A * nnz_per_row_B) FLOPs (worst case)
/// In practice, depends heavily on sparsity pattern
pub fn estimate_spspmm_flops(m: usize, avg_nnz_a: f64, avg_nnz_b: f64) -> usize {
    (m as f64 * avg_nnz_a * avg_nnz_b * 2.0) as usize
}

/// Check if indices are sorted in lexicographic order
pub fn is_sorted_lex(indices: &[Vec<usize>]) -> bool {
    if indices.len() <= 1 {
        return true;
    }

    for i in 1..indices.len() {
        if indices[i] < indices[i - 1] {
            return false;
        }
    }
    true
}

/// Sort indices in lexicographic order along with values
///
/// # Examples
///
/// ```
/// use tenrso_sparse::utils::sort_coo_inplace;
///
/// let mut indices = vec![vec![1, 0], vec![0, 1], vec![0, 0]];
/// let mut values = vec![3.0, 2.0, 1.0];
///
/// sort_coo_inplace(&mut indices, &mut values);
///
/// assert_eq!(indices, vec![vec![0, 0], vec![0, 1], vec![1, 0]]);
/// assert_eq!(values, vec![1.0, 2.0, 3.0]);
/// ```
pub fn sort_coo_inplace<T: Clone>(indices: &mut [Vec<usize>], values: &mut [T]) {
    assert_eq!(
        indices.len(),
        values.len(),
        "Indices and values length mismatch"
    );

    if indices.is_empty() {
        return;
    }

    // Create index array for sorting
    let mut idx: Vec<usize> = (0..indices.len()).collect();

    // Sort indices by comparing index vectors
    idx.sort_by(|&a, &b| indices[a].cmp(&indices[b]));

    // Apply permutation to both indices and values
    let indices_sorted: Vec<_> = idx.iter().map(|&i| indices[i].clone()).collect();
    let values_sorted: Vec<_> = idx.iter().map(|&i| values[i].clone()).collect();

    indices.clone_from_slice(&indices_sorted);
    values.clone_from_slice(&values_sorted);
}

/// Deduplicate sorted COO indices by summing values
///
/// # Examples
///
/// ```
/// use tenrso_sparse::utils::deduplicate_coo;
///
/// let indices = vec![vec![0, 0], vec![0, 0], vec![1, 1]];
/// let values = vec![1.0, 2.0, 3.0];
///
/// let (dedup_indices, dedup_values) = deduplicate_coo(&indices, &values);
///
/// assert_eq!(dedup_indices, vec![vec![0, 0], vec![1, 1]]);
/// assert_eq!(dedup_values, vec![3.0, 3.0]);
/// ```
pub fn deduplicate_coo<T: Float>(
    indices: &[Vec<usize>],
    values: &[T],
) -> (Vec<Vec<usize>>, Vec<T>) {
    if indices.is_empty() {
        return (vec![], vec![]);
    }

    let mut dedup_indices = Vec::new();
    let mut dedup_values = Vec::new();

    let mut current_idx = indices[0].clone();
    let mut current_val = values[0];

    for i in 1..indices.len() {
        if indices[i] == current_idx {
            // Same index, accumulate value
            current_val = current_val + values[i];
        } else {
            // New index, push previous
            dedup_indices.push(current_idx.clone());
            dedup_values.push(current_val);

            current_idx = indices[i].clone();
            current_val = values[i];
        }
    }

    // Push last element
    dedup_indices.push(current_idx);
    dedup_values.push(current_val);

    (dedup_indices, dedup_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparsity_stats() {
        let stats = SparsityStats::from_shape_nnz(vec![1000, 1000], 100);
        assert_eq!(stats.nnz, 100);
        assert_eq!(stats.total_elements, 1_000_000);
        assert!((stats.density - 0.0001).abs() < 1e-10);
        assert!(stats.is_very_sparse()); // 100/1M = 0.0001 < 0.01
    }

    #[test]
    fn test_recommend_format_dense() {
        let stats = SparsityStats::from_shape_nnz(vec![100, 100], 1500);
        assert_eq!(recommend_format(&stats), FormatRecommendation::Dense);
    }

    #[test]
    fn test_recommend_format_csr() {
        let stats = SparsityStats::from_shape_nnz(vec![1000, 1000], 500);
        assert_eq!(recommend_format(&stats), FormatRecommendation::CSR);
    }

    #[test]
    fn test_recommend_format_3d_sparse() {
        let stats = SparsityStats::from_shape_nnz(vec![100, 100, 100], 50);
        let rec = recommend_format(&stats);
        assert!(matches!(rec, FormatRecommendation::CSF { .. }));
    }

    #[test]
    fn test_memory_footprint_coo() {
        let mem = MemoryFootprint::coo::<f64>(&[100, 100, 100], 500);
        assert_eq!(mem.format, "COO");
        // 3 * 500 * 8 (indices) + 500 * 8 (values) + 3 * 8 (shape)
        assert_eq!(mem.bytes, 3 * 500 * 8 + 500 * 8 + 3 * 8);
    }

    #[test]
    fn test_memory_footprint_csr() {
        let mem = MemoryFootprint::csr::<f64>(100, 500);
        assert_eq!(mem.format, "CSR");
        // (100+1) * 8 (row_ptr) + 500 * 8 (col_indices) + 500 * 8 (values)
        assert_eq!(mem.bytes, 101 * 8 + 500 * 8 + 500 * 8);
    }

    #[test]
    fn test_compression_ratio() {
        let sparse = MemoryFootprint::csr::<f64>(1000, 500);
        let ratio = sparse.compression_ratio::<f64>(&[1000, 1000]);
        // Dense: 1000*1000*8 = 8MB
        // Sparse: ~12KB
        assert!(ratio > 100.0); // Much better compression
    }

    #[test]
    fn test_is_sorted_lex() {
        assert!(is_sorted_lex(&[vec![0, 0], vec![0, 1], vec![1, 0]]));
        assert!(!is_sorted_lex(&[vec![1, 0], vec![0, 1]]));
        assert!(is_sorted_lex(&[]));
        assert!(is_sorted_lex(&[vec![0]]));
    }

    #[test]
    fn test_sort_coo_inplace() {
        let mut indices = vec![vec![1, 0], vec![0, 1], vec![0, 0]];
        let mut values = vec![3.0, 2.0, 1.0];

        sort_coo_inplace(&mut indices, &mut values);

        assert_eq!(indices, vec![vec![0, 0], vec![0, 1], vec![1, 0]]);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_deduplicate_coo() {
        let indices = vec![vec![0, 0], vec![0, 0], vec![1, 1], vec![1, 1]];
        let values = vec![1.0, 2.0, 3.0, 4.0];

        let (dedup_idx, dedup_val) = deduplicate_coo(&indices, &values);

        assert_eq!(dedup_idx, vec![vec![0, 0], vec![1, 1]]);
        assert_eq!(dedup_val, vec![3.0, 7.0]);
    }

    #[test]
    fn test_estimate_flops() {
        assert_eq!(estimate_spmv_flops(1000), 2000);
        assert_eq!(estimate_spmm_flops(1000, 10), 20000);
        assert_eq!(estimate_spspmm_flops(100, 10.0, 10.0), 20000);
    }

    #[test]
    fn test_block_structure_detection() {
        let stats = SparsityStats::from_shape_nnz(vec![64, 64], 500);
        let (has_blocks, block_shape) = stats.has_block_structure(4);
        assert!(has_blocks);
        assert!(block_shape.iter().all(|&bs| bs >= 4));
    }
}
