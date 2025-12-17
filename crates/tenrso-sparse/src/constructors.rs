//! Special matrix constructors for common sparse matrix patterns
//!
//! This module provides convenient constructors for commonly used sparse matrices:
//! - Graph Laplacian matrices
//! - Adjacency matrices
//! - Stiffness matrices from finite element discretizations
//! - Toeplitz/circulant matrices
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{constructors, CsrMatrix};
//!
//! // Create a graph Laplacian matrix for a 3-node graph
//! let edges = vec![(0, 1), (1, 2), (0, 2)];
//! let laplacian: CsrMatrix<f64> = constructors::graph_laplacian(3, &edges, None).unwrap();
//!
//! // Create a 2D Poisson matrix (5-point stencil)
//! let poisson: CsrMatrix<f64> = constructors::poisson_2d(10, 10).unwrap();
//! ```

use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Construct a graph Laplacian matrix
///
/// The graph Laplacian is defined as L = D - A, where:
/// - D is the degree matrix (diagonal matrix with vertex degrees)
/// - A is the adjacency matrix
///
/// # Arguments
///
/// - `num_vertices`: Number of vertices in the graph
/// - `edges`: List of edges as (i, j) pairs
/// - `weights`: Optional edge weights (defaults to 1.0 if None)
///
/// # Complexity
///
/// O(num_edges) time and space
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{constructors, CsrMatrix};
///
/// // Triangle graph: 0-1-2-0
/// let edges = vec![(0, 1), (1, 2), (0, 2)];
/// let laplacian: CsrMatrix<f64> = constructors::graph_laplacian(3, &edges, None).unwrap();
///
/// // Weighted graph
/// let weights = vec![1.0, 2.0, 1.5];
/// let weighted: CsrMatrix<f64> = constructors::graph_laplacian(3, &edges, Some(&weights)).unwrap();
/// ```
pub fn graph_laplacian<T: Float>(
    num_vertices: usize,
    edges: &[(usize, usize)],
    weights: Option<&[T]>,
) -> SparseResult<CsrMatrix<T>> {
    if num_vertices == 0 {
        return Err(SparseError::validation("Number of vertices must be > 0"));
    }

    // Default weights to 1.0 if not provided
    let default_weights: Vec<T>;
    let edge_weights = if let Some(w) = weights {
        if w.len() != edges.len() {
            return Err(SparseError::validation(
                "Weights length must match number of edges",
            ));
        }
        w
    } else {
        default_weights = vec![T::one(); edges.len()];
        &default_weights
    };

    // Compute degree for each vertex
    let mut degrees = vec![T::zero(); num_vertices];
    let mut adj_map: HashMap<(usize, usize), T> = HashMap::new();

    for (idx, &(i, j)) in edges.iter().enumerate() {
        let w = edge_weights[idx];

        if i >= num_vertices || j >= num_vertices {
            return Err(SparseError::validation("Edge vertex out of bounds"));
        }

        if i == j {
            // Self-loop contributes twice to degree
            degrees[i] = degrees[i] + w + w;
        } else {
            // Add both directions for undirected graph
            degrees[i] = degrees[i] + w;
            degrees[j] = degrees[j] + w;

            // Store adjacency (both directions for symmetric matrix)
            *adj_map.entry((i, j)).or_insert(T::zero()) = w;
            *adj_map.entry((j, i)).or_insert(T::zero()) = w;
        }
    }

    // Build CSR Laplacian: L = D - A
    let mut row_ptr = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for (i, &_degree) in degrees.iter().enumerate().take(num_vertices) {
        // Collect all neighbors
        let mut neighbors: Vec<_> = adj_map
            .iter()
            .filter(|&(&(row, _col), _val)| row == i)
            .map(|(&(_row, col), &val)| (col, val))
            .collect();

        // Always add diagonal even if isolated vertex
        let has_diagonal = neighbors.iter().any(|(col, _)| *col == i);
        if !has_diagonal {
            neighbors.push((i, T::zero()));
        }

        // Sort by column index
        neighbors.sort_by_key(|&(col, _val)| col);

        // Add entries for this row
        for (j, adj_val) in neighbors {
            if i == j {
                // Diagonal: degree
                col_indices.push(i);
                values.push(degrees[i]);
            } else {
                // Off-diagonal: -adjacency
                col_indices.push(j);
                values.push(-adj_val);
            }
        }

        row_ptr.push(col_indices.len());
    }

    CsrMatrix::new(row_ptr, col_indices, values, (num_vertices, num_vertices))
        .map_err(|e| SparseError::Other(e.to_string()))
}

/// Construct an adjacency matrix from a list of edges
///
/// # Arguments
///
/// - `num_vertices`: Number of vertices in the graph
/// - `edges`: List of edges as (i, j) pairs
/// - `weights`: Optional edge weights (defaults to 1.0 if None)
/// - `symmetric`: Whether to create a symmetric matrix (undirected graph)
///
/// # Complexity
///
/// O(num_edges) time and space
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{constructors, CsrMatrix};
///
/// // Directed graph
/// let edges = vec![(0, 1), (1, 2), (2, 0)];
/// let adj: CsrMatrix<f64> = constructors::adjacency_matrix(3, &edges, None, false).unwrap();
///
/// // Undirected graph (symmetric)
/// let adj_sym: CsrMatrix<f64> = constructors::adjacency_matrix(3, &edges, None, true).unwrap();
/// ```
pub fn adjacency_matrix<T: Float>(
    num_vertices: usize,
    edges: &[(usize, usize)],
    weights: Option<&[T]>,
    symmetric: bool,
) -> SparseResult<CsrMatrix<T>> {
    if num_vertices == 0 {
        return Err(SparseError::validation("Number of vertices must be > 0"));
    }

    let default_weights: Vec<T>;
    let edge_weights = if let Some(w) = weights {
        if w.len() != edges.len() {
            return Err(SparseError::validation(
                "Weights length must match number of edges",
            ));
        }
        w
    } else {
        default_weights = vec![T::one(); edges.len()];
        &default_weights
    };

    let mut adj_map: HashMap<(usize, usize), T> = HashMap::new();

    for (idx, &(i, j)) in edges.iter().enumerate() {
        let w = edge_weights[idx];

        if i >= num_vertices || j >= num_vertices {
            return Err(SparseError::validation("Edge vertex out of bounds"));
        }

        *adj_map.entry((i, j)).or_insert(T::zero()) = w;

        if symmetric && i != j {
            *adj_map.entry((j, i)).or_insert(T::zero()) = w;
        }
    }

    // Build CSR matrix
    let mut row_ptr = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..num_vertices {
        let mut neighbors: Vec<_> = adj_map
            .iter()
            .filter(|&(&(row, _col), _val)| row == i)
            .map(|(&(_row, col), &val)| (col, val))
            .collect();

        neighbors.sort_by_key(|&(col, _val)| col);

        for (j, val) in neighbors {
            col_indices.push(j);
            values.push(val);
        }

        row_ptr.push(col_indices.len());
    }

    CsrMatrix::new(row_ptr, col_indices, values, (num_vertices, num_vertices))
        .map_err(|e| SparseError::Other(e.to_string()))
}

/// Construct a 2D Poisson matrix using 5-point stencil
///
/// Creates the discrete Laplacian for a 2D rectangular grid using
/// finite differences with the standard 5-point stencil:
/// ```text
///        -1
///    -1   4  -1
///        -1
/// ```
///
/// # Arguments
///
/// - `nx`: Number of grid points in x direction
/// - `ny`: Number of grid points in y direction
///
/// # Complexity
///
/// O(nx × ny) time and space
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{constructors, CsrMatrix};
///
/// // 10×10 grid Poisson matrix
/// let poisson: CsrMatrix<f64> = constructors::poisson_2d(10, 10).unwrap();
/// assert_eq!(poisson.nrows(), 100);
/// ```
pub fn poisson_2d<T: Float>(nx: usize, ny: usize) -> SparseResult<CsrMatrix<T>> {
    if nx == 0 || ny == 0 {
        return Err(SparseError::validation("Grid dimensions must be > 0"));
    }

    let n = nx * ny;
    let mut row_ptr = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    // 5-point stencil
    let four = T::from(4.0).unwrap();
    let neg_one = -T::one();

    for iy in 0..ny {
        for ix in 0..nx {
            let i = iy * nx + ix;

            // West neighbor (i-1)
            if ix > 0 {
                col_indices.push(i - 1);
                values.push(neg_one);
            }

            // South neighbor (i-nx)
            if iy > 0 {
                col_indices.push(i - nx);
                values.push(neg_one);
            }

            // Center
            col_indices.push(i);
            values.push(four);

            // North neighbor (i+nx)
            if iy < ny - 1 {
                col_indices.push(i + nx);
                values.push(neg_one);
            }

            // East neighbor (i+1)
            if ix < nx - 1 {
                col_indices.push(i + 1);
                values.push(neg_one);
            }

            row_ptr.push(col_indices.len());
        }
    }

    CsrMatrix::new(row_ptr, col_indices, values, (n, n))
        .map_err(|e| SparseError::Other(e.to_string()))
}

/// Construct a tridiagonal matrix
///
/// Creates a tridiagonal matrix with specified diagonals.
///
/// # Arguments
///
/// - `n`: Matrix dimension
/// - `lower`: Value for lower diagonal
/// - `diag`: Value for main diagonal
/// - `upper`: Value for upper diagonal
///
/// # Complexity
///
/// O(n) time and space
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{constructors, CsrMatrix};
///
/// // Standard finite difference matrix: [-1, 2, -1]
/// let fd: CsrMatrix<f64> = constructors::tridiagonal(100, -1.0, 2.0, -1.0).unwrap();
/// ```
pub fn tridiagonal<T: Float>(n: usize, lower: T, diag: T, upper: T) -> SparseResult<CsrMatrix<T>> {
    if n == 0 {
        return Err(SparseError::validation("Matrix dimension must be > 0"));
    }

    let mut row_ptr = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        // Lower diagonal
        if i > 0 {
            col_indices.push(i - 1);
            values.push(lower);
        }

        // Main diagonal
        col_indices.push(i);
        values.push(diag);

        // Upper diagonal
        if i < n - 1 {
            col_indices.push(i + 1);
            values.push(upper);
        }

        row_ptr.push(col_indices.len());
    }

    CsrMatrix::new(row_ptr, col_indices, values, (n, n))
        .map_err(|e| SparseError::Other(e.to_string()))
}

/// Construct an identity matrix
///
/// # Arguments
///
/// - `n`: Matrix dimension
///
/// # Complexity
///
/// O(n) time and space
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::constructors;
///
/// let eye = constructors::identity::<f64>(100).unwrap();
/// ```
pub fn identity<T: Float>(n: usize) -> SparseResult<CsrMatrix<T>> {
    if n == 0 {
        return Err(SparseError::validation("Matrix dimension must be > 0"));
    }

    let row_ptr: Vec<usize> = (0..=n).collect();
    let col_indices: Vec<usize> = (0..n).collect();
    let values = vec![T::one(); n];

    CsrMatrix::new(row_ptr, col_indices, values, (n, n))
        .map_err(|e| SparseError::Other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_laplacian_triangle() {
        // Triangle: 0-1-2-0
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let l = graph_laplacian::<f64>(3, &edges, None).unwrap();

        assert_eq!(l.nrows(), 3);
        assert_eq!(l.ncols(), 3);

        // Each vertex has degree 2
        // L = D - A where D = diag(2, 2, 2)
        // Check diagonal is 2
        for i in 0..3 {
            let row_start = l.row_ptr()[i];
            let row_end = l.row_ptr()[i + 1];
            let mut found_diag = false;
            for idx in row_start..row_end {
                if l.col_indices()[idx] == i {
                    assert_eq!(l.values()[idx], 2.0);
                    found_diag = true;
                }
            }
            assert!(found_diag);
        }
    }

    #[test]
    fn test_adjacency_matrix_directed() {
        let edges = vec![(0, 1), (1, 2)];
        let adj = adjacency_matrix::<f64>(3, &edges, None, false).unwrap();

        assert_eq!(adj.nrows(), 3);
        assert_eq!(adj.nnz(), 2);
    }

    #[test]
    fn test_adjacency_matrix_undirected() {
        let edges = vec![(0, 1), (1, 2)];
        let adj = adjacency_matrix::<f64>(3, &edges, None, true).unwrap();

        assert_eq!(adj.nrows(), 3);
        assert_eq!(adj.nnz(), 4); // Both directions
    }

    #[test]
    fn test_poisson_2d_small() {
        let poisson = poisson_2d::<f64>(3, 3).unwrap();

        assert_eq!(poisson.nrows(), 9);
        assert_eq!(poisson.ncols(), 9);

        // Corner vertices have 2 neighbors + center = 3 nonzeros
        // Edge vertices have 3 neighbors + center = 4 nonzeros
        // Center vertex has 4 neighbors + center = 5 nonzeros
        // Total: 4 corners × 3 + 4 edges × 4 + 1 center × 5 = 12 + 16 + 5 = 33
        assert_eq!(poisson.nnz(), 33);
    }

    #[test]
    fn test_tridiagonal() {
        let tri = tridiagonal(5, -1.0, 2.0, -1.0).unwrap();

        assert_eq!(tri.nrows(), 5);
        assert_eq!(tri.nnz(), 13); // 1+2+3+3+3+1 = 13
    }

    #[test]
    fn test_identity() {
        let eye = identity::<f64>(10).unwrap();

        assert_eq!(eye.nrows(), 10);
        assert_eq!(eye.nnz(), 10);

        // Check all diagonal elements are 1
        for i in 0..10 {
            let row_start = eye.row_ptr()[i];
            let row_end = eye.row_ptr()[i + 1];
            assert_eq!(row_end - row_start, 1);
            assert_eq!(eye.col_indices()[row_start], i);
            assert_eq!(eye.values()[row_start], 1.0);
        }
    }
}
