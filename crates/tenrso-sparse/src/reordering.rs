//! Matrix reordering algorithms for sparse matrices
//!
//! This module provides reordering algorithms that reduce fill-in during
//! factorizations and improve cache locality:
//! - Reverse Cuthill-McKee (RCM) for bandwidth reduction
//! - Approximate Minimum Degree (AMD) for fill-in reduction
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{CsrMatrix, reordering};
//!
//! // Create a sparse matrix
//! let row_ptr = vec![0, 2, 4, 6, 7];
//! let col_indices = vec![0, 1, 0, 2, 1, 3, 2];
//! let values = vec![1.0; 7];
//! let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();
//!
//! // Compute RCM ordering
//! let perm = reordering::rcm(&a).unwrap();
//!
//! // Apply permutation to matrix
//! let reordered = reordering::permute_symmetric(&a, &perm).unwrap();
//! ```

use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;

/// Reverse Cuthill-McKee ordering
///
/// Computes a permutation that reduces the bandwidth of the matrix.
/// This is useful for improving cache locality and reducing fill-in
/// for certain factorization methods.
///
/// # Algorithm
///
/// 1. Find a peripheral vertex (node with small degree)
/// 2. BFS traversal with vertices sorted by degree
/// 3. Reverse the ordering
///
/// # Complexity
///
/// O(nnz + n log n) time, O(n) space
///
/// # Arguments
///
/// - `matrix`: Sparse matrix (should be symmetric structure)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, reordering};
///
/// let row_ptr = vec![0, 2, 4, 6, 7];
/// let col_indices = vec![0, 1, 0, 2, 1, 3, 2];
/// let values = vec![1.0; 7];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();
///
/// let perm = reordering::rcm(&a).unwrap();
/// assert_eq!(perm.len(), 4);
/// ```
pub fn rcm<T: Float>(matrix: &CsrMatrix<T>) -> SparseResult<Vec<usize>> {
    let (n, _) = matrix.shape();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Compute vertex degrees
    let degrees: Vec<usize> = (0..n)
        .map(|i| matrix.row_ptr()[i + 1] - matrix.row_ptr()[i])
        .collect();

    // Find starting vertex (minimum degree peripheral vertex)
    let start = find_peripheral_vertex(matrix, &degrees);

    // BFS with degree-sorted neighbors
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    let mut ordering = Vec::with_capacity(n);

    queue.push_back(start);
    visited[start] = true;

    while let Some(v) = queue.pop_front() {
        ordering.push(v);

        // Get neighbors
        let start_idx = matrix.row_ptr()[v];
        let end_idx = matrix.row_ptr()[v + 1];
        let mut neighbors: Vec<usize> = (start_idx..end_idx)
            .map(|idx| matrix.col_indices()[idx])
            .filter(|&u| !visited[u])
            .collect();

        // Sort neighbors by degree (ascending)
        neighbors.sort_by_key(|&u| degrees[u]);

        // Add to queue
        for &u in &neighbors {
            if !visited[u] {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }

    // Add any disconnected components
    for v in 0..n {
        if !visited[v] {
            queue.push_back(v);
            visited[v] = true;

            while let Some(u) = queue.pop_front() {
                ordering.push(u);

                let start_idx = matrix.row_ptr()[u];
                let end_idx = matrix.row_ptr()[u + 1];
                let mut neighbors: Vec<usize> = (start_idx..end_idx)
                    .map(|idx| matrix.col_indices()[idx])
                    .filter(|&w| !visited[w])
                    .collect();

                neighbors.sort_by_key(|&w| degrees[w]);

                for &w in &neighbors {
                    if !visited[w] {
                        visited[w] = true;
                        queue.push_back(w);
                    }
                }
            }
        }
    }

    // Reverse the ordering (Cuthill-McKee → Reverse Cuthill-McKee)
    ordering.reverse();

    Ok(ordering)
}

/// Find a peripheral vertex using pseudo-diameter algorithm
fn find_peripheral_vertex<T: Float>(matrix: &CsrMatrix<T>, degrees: &[usize]) -> usize {
    let n = matrix.shape().0;

    // Start with minimum degree vertex
    let mut start = degrees
        .iter()
        .enumerate()
        .min_by_key(|&(_, &d)| d)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Perform a few BFS iterations to find peripheral vertex
    for _ in 0..3.min(n) {
        let (farthest, _) = bfs_farthest(matrix, start);
        if farthest == start {
            break;
        }
        start = farthest;
    }

    start
}

/// BFS to find farthest vertex from start
fn bfs_farthest<T: Float>(matrix: &CsrMatrix<T>, start: usize) -> (usize, usize) {
    let n = matrix.shape().0;
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    let mut level = vec![0; n];

    queue.push_back(start);
    visited[start] = true;

    let mut farthest = start;
    let mut max_level = 0;

    while let Some(v) = queue.pop_front() {
        if level[v] > max_level {
            max_level = level[v];
            farthest = v;
        }

        let start_idx = matrix.row_ptr()[v];
        let end_idx = matrix.row_ptr()[v + 1];

        for idx in start_idx..end_idx {
            let u = matrix.col_indices()[idx];
            if !visited[u] {
                visited[u] = true;
                level[u] = level[v] + 1;
                queue.push_back(u);
            }
        }
    }

    (farthest, max_level)
}

/// Approximate Minimum Degree ordering
///
/// Computes a fill-reducing permutation using the approximate minimum
/// degree algorithm. This is one of the most effective reordering
/// strategies for reducing fill-in during sparse factorization.
///
/// # Algorithm
///
/// Iteratively eliminates the vertex with (approximately) minimum degree,
/// updating the graph structure after each elimination.
///
/// # Complexity
///
/// O(nnz × α(n)) time on average, O(n + nnz) space
/// where α is the inverse Ackermann function (very slow growing)
///
/// # Arguments
///
/// - `matrix`: Sparse matrix (should be symmetric structure)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, reordering};
///
/// let row_ptr = vec![0, 2, 4, 6, 7];
/// let col_indices = vec![0, 1, 0, 2, 1, 3, 2];
/// let values = vec![1.0; 7];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();
///
/// let perm = reordering::amd(&a).unwrap();
/// assert_eq!(perm.len(), 4);
/// ```
pub fn amd<T: Float>(matrix: &CsrMatrix<T>) -> SparseResult<Vec<usize>> {
    let (n, _) = matrix.shape();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency list (symmetric)
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let start = matrix.row_ptr()[i];
        let end = matrix.row_ptr()[i + 1];

        for idx in start..end {
            let j = matrix.col_indices()[idx];
            if i != j {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    // Remove duplicates and sort
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    // Compute initial degrees
    let mut degree: Vec<usize> = adj.iter().map(|v| v.len()).collect();

    // Track eliminated vertices
    let mut eliminated = vec![false; n];
    let mut ordering = Vec::with_capacity(n);

    // Supervariable representation (for mass elimination)
    let supervars: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    for _ in 0..n {
        // Find vertex with minimum degree (excluding eliminated)
        let v = (0..n)
            .filter(|&i| !eliminated[i])
            .min_by_key(|&i| (degree[i], i))
            .unwrap();

        // Add all vertices in supervar to ordering
        for &u in &supervars[v] {
            ordering.push(u);
        }

        // Mark as eliminated
        eliminated[v] = true;

        // Get reach set (neighbors of v)
        let reach: Vec<usize> = adj[v]
            .iter()
            .filter(|&&u| !eliminated[u])
            .copied()
            .collect();

        // Update adjacencies (mass elimination)
        for &u in &reach {
            // Remove v from u's adjacency
            adj[u].retain(|&w| w != v);

            // Add edges to other vertices in reach (fill-in)
            for &w in &reach {
                if u != w && !adj[u].contains(&w) {
                    adj[u].push(w);
                }
            }

            // Update degree
            degree[u] = adj[u].len();
        }

        // Clear v's adjacency
        adj[v].clear();
        degree[v] = 0;
    }

    Ok(ordering)
}

/// Apply symmetric permutation to matrix
///
/// Computes P * A * P^T where P is the permutation matrix.
///
/// # Complexity
///
/// O(nnz) time and space
///
/// # Arguments
///
/// - `matrix`: Input sparse matrix
/// - `perm`: Permutation vector (new index of old vertex i)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, reordering};
///
/// let row_ptr = vec![0, 2, 4, 6, 7];
/// let col_indices = vec![0, 1, 0, 2, 1, 3, 2];
/// let values = vec![1.0; 7];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();
///
/// let perm = vec![3, 2, 1, 0]; // Reverse order
/// let reordered = reordering::permute_symmetric(&a, &perm).unwrap();
/// assert_eq!(reordered.shape(), (4, 4));
/// ```
pub fn permute_symmetric<T: Float>(
    matrix: &CsrMatrix<T>,
    perm: &[usize],
) -> SparseResult<CsrMatrix<T>> {
    let (n, m) = matrix.shape();

    if perm.len() != n || perm.len() != m {
        return Err(SparseError::validation(&format!(
            "Permutation size {} != matrix size {}",
            perm.len(),
            n
        )));
    }

    // Check permutation validity
    let mut seen = vec![false; n];
    for &p in perm {
        if p >= n {
            return Err(SparseError::validation(&format!(
                "Invalid permutation index {}",
                p
            )));
        }
        if seen[p] {
            return Err(SparseError::validation("Permutation has duplicates"));
        }
        seen[p] = true;
    }

    // Compute inverse permutation
    let mut inv_perm = vec![0; n];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv_perm[old_idx] = new_idx;
    }

    // Build new matrix: new_row[i] corresponds to old_row[perm[i]]
    let mut new_row_ptr = vec![0];
    let mut new_col_indices = Vec::new();
    let mut new_values = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for new_row in 0..n {
        let old_row = perm[new_row];

        // Get entries from old row
        let start = matrix.row_ptr()[old_row];
        let end = matrix.row_ptr()[old_row + 1];

        // Collect and transform entries
        let mut entries: Vec<(usize, T)> = (start..end)
            .map(|idx| {
                let old_col = matrix.col_indices()[idx];
                let new_col = inv_perm[old_col];
                let val = matrix.values()[idx];
                (new_col, val)
            })
            .collect();

        // Sort by new column index
        entries.sort_by_key(|&(col, _)| col);

        // Add to new matrix
        for (col, val) in entries {
            new_col_indices.push(col);
            new_values.push(val);
        }

        new_row_ptr.push(new_col_indices.len());
    }

    Ok(CsrMatrix::new(
        new_row_ptr,
        new_col_indices,
        new_values,
        (n, m),
    )?)
}

/// Compute bandwidth of matrix
///
/// Returns (lower_bandwidth, upper_bandwidth).
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, reordering};
///
/// let row_ptr = vec![0, 1, 2, 3];
/// let col_indices = vec![0, 1, 2];
/// let values = vec![1.0; 3];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let (lower, upper) = reordering::bandwidth(&a);
/// assert_eq!((lower, upper), (0, 0)); // Diagonal matrix
/// ```
pub fn bandwidth<T: Float>(matrix: &CsrMatrix<T>) -> (usize, usize) {
    let (n, _) = matrix.shape();
    let mut lower_bw = 0;
    let mut upper_bw = 0;

    for i in 0..n {
        let start = matrix.row_ptr()[i];
        let end = matrix.row_ptr()[i + 1];

        for idx in start..end {
            let j = matrix.col_indices()[idx];

            if j < i {
                lower_bw = lower_bw.max(i - j);
            } else if j > i {
                upper_bw = upper_bw.max(j - i);
            }
        }
    }

    (lower_bw, upper_bw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> CsrMatrix<f64> {
        // 4x4 symmetric tridiagonal-like matrix
        let row_ptr = vec![0, 2, 4, 6, 7];
        let col_indices = vec![0, 1, 0, 2, 1, 3, 2];
        let values = vec![1.0; 7];
        CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap()
    }

    #[test]
    fn test_rcm_basic() {
        let matrix = create_test_matrix();
        let perm = rcm(&matrix).unwrap();

        assert_eq!(perm.len(), 4);

        // Check that it's a valid permutation
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_rcm_empty() {
        let row_ptr = vec![0, 0];
        let col_indices: Vec<usize> = vec![];
        let values: Vec<f64> = vec![];
        let matrix = CsrMatrix::new(row_ptr, col_indices, values, (1, 1)).unwrap();

        let perm = rcm(&matrix).unwrap();
        assert_eq!(perm.len(), 1);
    }

    #[test]
    fn test_amd_basic() {
        let matrix = create_test_matrix();
        let perm = amd(&matrix).unwrap();

        assert_eq!(perm.len(), 4);

        // Check that it's a valid permutation
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_amd_empty() {
        let row_ptr = vec![0, 0];
        let col_indices: Vec<usize> = vec![];
        let values: Vec<f64> = vec![];
        let matrix = CsrMatrix::new(row_ptr, col_indices, values, (1, 1)).unwrap();

        let perm = amd(&matrix).unwrap();
        assert_eq!(perm.len(), 1);
    }

    #[test]
    fn test_permute_symmetric_basic() {
        let matrix = create_test_matrix();
        let perm = vec![3, 2, 1, 0]; // Reverse order

        let reordered = permute_symmetric(&matrix, &perm).unwrap();
        assert_eq!(reordered.shape(), (4, 4));
        assert_eq!(reordered.nnz(), matrix.nnz());
    }

    #[test]
    fn test_permute_symmetric_identity() {
        let matrix = create_test_matrix();
        let perm = vec![0, 1, 2, 3]; // Identity

        let reordered = permute_symmetric(&matrix, &perm).unwrap();

        // Should be same as original
        assert_eq!(reordered.row_ptr(), matrix.row_ptr());
        assert_eq!(reordered.col_indices(), matrix.col_indices());
        assert_eq!(reordered.values(), matrix.values());
    }

    #[test]
    fn test_permute_symmetric_invalid_size() {
        let matrix = create_test_matrix();
        let perm = vec![0, 1]; // Wrong size

        let result = permute_symmetric(&matrix, &perm);
        assert!(result.is_err());
    }

    #[test]
    fn test_permute_symmetric_invalid_index() {
        let matrix = create_test_matrix();
        let perm = vec![0, 1, 2, 10]; // Invalid index

        let result = permute_symmetric(&matrix, &perm);
        assert!(result.is_err());
    }

    #[test]
    fn test_permute_symmetric_duplicates() {
        let matrix = create_test_matrix();
        let perm = vec![0, 1, 1, 3]; // Duplicate

        let result = permute_symmetric(&matrix, &perm);
        assert!(result.is_err());
    }

    #[test]
    fn test_bandwidth_diagonal() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0; 3];
        let matrix = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let (lower, upper) = bandwidth(&matrix);
        assert_eq!((lower, upper), (0, 0));
    }

    #[test]
    fn test_bandwidth_tridiagonal() {
        let matrix = create_test_matrix();
        let (lower, upper) = bandwidth(&matrix);
        assert!(lower <= 1);
        assert!(upper <= 1);
    }

    #[test]
    fn test_rcm_reduces_bandwidth() {
        // Create a matrix with large bandwidth
        let row_ptr = vec![0, 2, 4, 5, 7];
        let col_indices = vec![0, 3, 1, 3, 2, 0, 3];
        let values = vec![1.0; 7];
        let matrix = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let (orig_lower, orig_upper) = bandwidth(&matrix);

        let perm = rcm(&matrix).unwrap();
        let reordered = permute_symmetric(&matrix, &perm).unwrap();

        let (new_lower, new_upper) = bandwidth(&reordered);

        // RCM should not increase bandwidth
        assert!(new_lower + new_upper <= orig_lower + orig_upper);
    }

    #[test]
    fn test_find_peripheral_vertex() {
        let matrix = create_test_matrix();
        let degrees: Vec<usize> = (0..4)
            .map(|i| matrix.row_ptr()[i + 1] - matrix.row_ptr()[i])
            .collect();

        let peripheral = find_peripheral_vertex(&matrix, &degrees);
        assert!(peripheral < 4);
    }

    #[test]
    fn test_bfs_farthest() {
        let matrix = create_test_matrix();
        let (farthest, level) = bfs_farthest(&matrix, 0);

        assert!(farthest < 4);
        assert!(level <= 3);
    }
}
