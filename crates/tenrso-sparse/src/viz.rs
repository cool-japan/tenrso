//! Sparsity pattern visualization utilities
//!
//! This module provides tools for visualizing sparse matrix structures:
//! - ASCII art representation
//! - Spy plot data generation
//! - Pattern statistics
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{CsrMatrix, viz};
//!
//! let row_ptr = vec![0, 2, 4, 6];
//! let col_indices = vec![0, 2, 0, 1, 1, 2];
//! let values = vec![1.0; 6];
//! let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
//!
//! // Print ASCII representation
//! println!("{}", viz::ascii_pattern(&csr, 10, 10));
//!
//! // Get spy plot coordinates
//! let coords = viz::spy_coords(&csr);
//! assert_eq!(coords.len(), 6);
//! ```

use crate::CsrMatrix;
use scirs2_core::numeric::Float;

/// Generate ASCII art representation of sparsity pattern
///
/// Uses '█' for nonzero entries and '·' for zeros.
/// Automatically scales if matrix is larger than max dimensions.
///
/// # Arguments
///
/// - `matrix`: The sparse matrix to visualize
/// - `max_rows`: Maximum rows to display (will downsample if needed)
/// - `max_cols`: Maximum columns to display (will downsample if needed)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, viz::ascii_pattern};
///
/// let row_ptr = vec![0, 1, 3, 4];
/// let col_indices = vec![0, 1, 2, 2];
/// let values = vec![1.0; 4];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let pattern = ascii_pattern(&csr, 10, 10);
/// assert!(pattern.contains('█'));
/// assert!(pattern.contains('·'));
/// ```
pub fn ascii_pattern<T: Float>(matrix: &CsrMatrix<T>, max_rows: usize, max_cols: usize) -> String {
    let (nrows, ncols) = matrix.shape();

    // Calculate scaling factors
    let row_scale = if nrows > max_rows {
        nrows as f64 / max_rows as f64
    } else {
        1.0
    };
    let col_scale = if ncols > max_cols {
        ncols as f64 / max_cols as f64
    } else {
        1.0
    };

    let display_rows = if nrows > max_rows { max_rows } else { nrows };
    let display_cols = if ncols > max_cols { max_cols } else { ncols };

    // Build pattern
    let mut result = String::new();

    // Header
    result.push_str(&format!(
        "Sparsity Pattern ({} x {}, {:.2}% nonzero)\n",
        nrows,
        ncols,
        matrix.nnz() as f64 / (nrows * ncols) as f64 * 100.0
    ));
    result.push_str(&"─".repeat(display_cols + 2));
    result.push('\n');

    for display_row in 0..display_rows {
        result.push('│');

        for display_col in 0..display_cols {
            // Map display coordinates to matrix coordinates
            let row_start = (display_row as f64 * row_scale) as usize;
            let row_end = ((display_row + 1) as f64 * row_scale).ceil() as usize;
            let col_start = (display_col as f64 * col_scale) as usize;
            let col_end = ((display_col + 1) as f64 * col_scale).ceil() as usize;

            // Check if any nonzero in this block
            let mut has_nonzero = false;
            for r in row_start..row_end.min(nrows) {
                let start_idx = matrix.row_ptr()[r];
                let end_idx = matrix.row_ptr()[r + 1];

                for idx in start_idx..end_idx {
                    let c = matrix.col_indices()[idx];
                    if c >= col_start && c < col_end {
                        has_nonzero = true;
                        break;
                    }
                }

                if has_nonzero {
                    break;
                }
            }

            result.push(if has_nonzero { '█' } else { '·' });
        }

        result.push('│');
        result.push('\n');
    }

    result.push_str(&"─".repeat(display_cols + 2));
    result.push('\n');

    result
}

/// Generate coordinates for spy plot
///
/// Returns `(row, col)` pairs for all nonzero entries,
/// suitable for plotting with external tools.
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, viz::spy_coords};
///
/// let row_ptr = vec![0, 1, 2, 3];
/// let col_indices = vec![0, 1, 2];
/// let values = vec![1.0; 3];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let coords = spy_coords(&csr);
/// assert_eq!(coords, vec![(0, 0), (1, 1), (2, 2)]);
/// ```
pub fn spy_coords<T: Float>(matrix: &CsrMatrix<T>) -> Vec<(usize, usize)> {
    let (nrows, _) = matrix.shape();
    let mut coords = Vec::with_capacity(matrix.nnz());

    for row in 0..nrows {
        let start = matrix.row_ptr()[row];
        let end = matrix.row_ptr()[row + 1];

        for idx in start..end {
            let col = matrix.col_indices()[idx];
            coords.push((row, col));
        }
    }

    coords
}

/// Analyze block structure of sparsity pattern
///
/// Divides matrix into blocks and reports density per block.
///
/// # Arguments
///
/// - `matrix`: The sparse matrix to analyze
/// - `block_rows`: Number of block rows
/// - `block_cols`: Number of block columns
///
/// Returns a 2D vector where `result[i][j]` is the density of block `(i, j)`.
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, viz::block_density};
///
/// let row_ptr = vec![0, 2, 4, 6, 8];
/// let col_indices = vec![0, 1, 2, 3, 0, 1, 2, 3];
/// let values = vec![1.0; 8];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();
///
/// let densities = block_density(&csr, 2, 2);
/// assert_eq!(densities.len(), 2); // 2 block rows
/// assert_eq!(densities[0].len(), 2); // 2 block columns
/// ```
pub fn block_density<T: Float>(
    matrix: &CsrMatrix<T>,
    block_rows: usize,
    block_cols: usize,
) -> Vec<Vec<f64>> {
    let (nrows, ncols) = matrix.shape();

    let row_block_size = nrows.div_ceil(block_rows);
    let col_block_size = ncols.div_ceil(block_cols);

    let mut densities = vec![vec![0.0; block_cols]; block_rows];
    let mut block_nnz = vec![vec![0; block_cols]; block_rows];
    let mut block_sizes = vec![vec![0; block_cols]; block_rows];

    // Count nonzeros per block
    for row in 0..nrows {
        let block_row = row / row_block_size;
        let start = matrix.row_ptr()[row];
        let end = matrix.row_ptr()[row + 1];

        for idx in start..end {
            let col = matrix.col_indices()[idx];
            let block_col = col / col_block_size;

            block_nnz[block_row][block_col] += 1;
        }
    }

    // Calculate densities
    for br in 0..block_rows {
        for bc in 0..block_cols {
            let row_start = br * row_block_size;
            let row_end = ((br + 1) * row_block_size).min(nrows);
            let col_start = bc * col_block_size;
            let col_end = ((bc + 1) * col_block_size).min(ncols);

            let block_size = (row_end - row_start) * (col_end - col_start);
            block_sizes[br][bc] = block_size;

            densities[br][bc] = if block_size > 0 {
                block_nnz[br][bc] as f64 / block_size as f64
            } else {
                0.0
            };
        }
    }

    densities
}

/// Generate a heatmap-style ASCII representation of block densities
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, viz::block_density_heatmap};
///
/// let row_ptr = vec![0, 3, 6, 9, 12];
/// let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
/// let values = vec![1.0; 12];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (4, 3)).unwrap();
///
/// let heatmap = block_density_heatmap(&csr, 2, 2);
/// assert!(heatmap.contains("Block Density"));
/// ```
pub fn block_density_heatmap<T: Float>(
    matrix: &CsrMatrix<T>,
    block_rows: usize,
    block_cols: usize,
) -> String {
    let densities = block_density(matrix, block_rows, block_cols);
    let mut result = String::new();

    result.push_str(&format!(
        "Block Density Heatmap ({} x {} blocks)\n",
        block_rows, block_cols
    ));

    for row in &densities {
        for &density in row {
            let char = if density > 0.75 {
                '█'
            } else if density > 0.5 {
                '▓'
            } else if density > 0.25 {
                '▒'
            } else if density > 0.0 {
                '░'
            } else {
                '·'
            };
            result.push(char);
            result.push(' ');
        }
        result.push('\n');
    }

    result
}

/// Compute row and column bandwidth visualization
///
/// Returns strings showing nnz distribution across rows and columns.
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, viz::bandwidth_profile};
///
/// let row_ptr = vec![0, 3, 5, 6];
/// let col_indices = vec![0, 1, 2, 1, 2, 0];
/// let values = vec![1.0; 6];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let (row_profile, _col_profile) = bandwidth_profile(&csr);
/// assert!(!row_profile.is_empty());
/// ```
pub fn bandwidth_profile<T: Float>(matrix: &CsrMatrix<T>) -> (String, String) {
    let (nrows, ncols) = matrix.shape();

    // Row profile
    let mut row_nnz = vec![0; nrows];
    for (row, nnz) in row_nnz.iter_mut().enumerate().take(nrows) {
        *nnz = matrix.row_ptr()[row + 1] - matrix.row_ptr()[row];
    }

    let max_row_nnz = *row_nnz.iter().max().unwrap_or(&0);

    let mut row_profile = String::from("Row Profile:\n");
    for (i, &nnz) in row_nnz.iter().enumerate() {
        let bar_len = if max_row_nnz > 0 {
            (nnz * 40 / max_row_nnz).max(1)
        } else {
            0
        };
        row_profile.push_str(&format!(
            "{:4} | {}{} ({})\n",
            i,
            "█".repeat(bar_len),
            " ".repeat(40 - bar_len),
            nnz
        ));
    }

    // Column profile
    let mut col_nnz = vec![0; ncols];
    for row in 0..nrows {
        for idx in matrix.row_ptr()[row]..matrix.row_ptr()[row + 1] {
            col_nnz[matrix.col_indices()[idx]] += 1;
        }
    }

    let max_col_nnz = *col_nnz.iter().max().unwrap_or(&0);

    let mut col_profile = String::from("Column Profile:\n");
    for (j, &nnz) in col_nnz.iter().enumerate() {
        let bar_len = if max_col_nnz > 0 {
            (nnz * 40 / max_col_nnz).max(1)
        } else {
            0
        };
        col_profile.push_str(&format!(
            "{:4} | {}{} ({})\n",
            j,
            "█".repeat(bar_len),
            " ".repeat(40 - bar_len),
            nnz
        ));
    }

    (row_profile, col_profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_pattern_small() {
        let row_ptr = vec![0, 2, 3, 4];
        let col_indices = vec![0, 2, 1, 0];
        let values = vec![1.0; 4];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let pattern = ascii_pattern(&csr, 10, 10);
        assert!(pattern.contains('█'));
        assert!(pattern.contains('·'));
        assert!(pattern.contains("Sparsity Pattern"));
    }

    #[test]
    fn test_ascii_pattern_scaling() {
        // Large matrix should be downsampled
        let mut row_ptr = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..100 {
            col_indices.push(i);
            values.push(1.0);
            row_ptr.push(col_indices.len());
        }

        let csr = CsrMatrix::new(row_ptr, col_indices, values, (100, 100)).unwrap();
        let pattern = ascii_pattern(&csr, 10, 10);

        // Should be scaled down to fit
        let lines: Vec<&str> = pattern.lines().collect();
        assert!(lines.len() < 20); // Header + 10 rows + footer
    }

    #[test]
    fn test_spy_coords() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0; 3];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let coords = spy_coords(&csr);
        assert_eq!(coords, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_block_density() {
        // 4x4 matrix with entries in first 2x2 block
        let row_ptr = vec![0, 2, 4, 4, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![1.0; 4];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let densities = block_density(&csr, 2, 2);
        assert_eq!(densities.len(), 2);
        assert_eq!(densities[0].len(), 2);

        // First block should be fully dense
        assert_eq!(densities[0][0], 1.0);

        // Other blocks should be empty
        assert_eq!(densities[0][1], 0.0);
        assert_eq!(densities[1][0], 0.0);
        assert_eq!(densities[1][1], 0.0);
    }

    #[test]
    fn test_block_density_heatmap() {
        let row_ptr = vec![0, 3, 6, 9];
        let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![1.0; 9];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let heatmap = block_density_heatmap(&csr, 2, 2);
        assert!(heatmap.contains("Block Density"));
        assert!(heatmap.contains('█')); // Should have dense blocks
    }

    #[test]
    fn test_bandwidth_profile() {
        let row_ptr = vec![0, 3, 5, 6];
        let col_indices = vec![0, 1, 2, 1, 2, 0];
        let values = vec![1.0; 6];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let (row_profile, col_profile) = bandwidth_profile(&csr);

        assert!(row_profile.contains("Row Profile"));
        assert!(col_profile.contains("Column Profile"));
        assert!(row_profile.contains('█'));
        assert!(col_profile.contains('█'));
    }

    #[test]
    fn test_spy_coords_empty() {
        let row_ptr = vec![0, 0, 0];
        let col_indices: Vec<usize> = vec![];
        let values: Vec<f64> = vec![];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let coords = spy_coords(&csr);
        assert!(coords.is_empty());
    }
}
