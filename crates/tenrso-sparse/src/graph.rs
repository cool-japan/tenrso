//! Graph algorithms on sparse matrices
//!
//! This module treats sparse matrices as adjacency matrices of graphs
//! and provides common graph algorithms:
//! - Breadth-First Search (BFS)
//! - Depth-First Search (DFS)
//! - Connected components
//! - Shortest paths
//! - Strongly connected components (Tarjan's algorithm)
//! - Topological sort
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{CsrMatrix, graph};
//!
//! // Create adjacency matrix for a graph with 4 nodes
//! let row_ptr = vec![0, 2, 4, 6, 7];
//! let col_indices = vec![1, 2, 0, 3, 0, 3, 2];
//! let values = vec![1.0; 7]; // Weights
//! let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();
//!
//! // Find connected components
//! let components = graph::connected_components(&graph);
//! assert_eq!(components.num_components, 1); // Fully connected
//! ```

use crate::CsrMatrix;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;

/// Result of connected components analysis
#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    /// Number of connected components
    pub num_components: usize,
    /// Component label for each vertex
    pub labels: Vec<usize>,
    /// Size of each component
    pub sizes: Vec<usize>,
}

/// Breadth-First Search from a starting vertex
///
/// Returns the vertices in the order they were visited.
///
/// # Complexity
///
/// O(V + E) where V is number of vertices, E is number of edges
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::bfs};
///
/// let row_ptr = vec![0, 2, 3, 4];
/// let col_indices = vec![1, 2, 2, 0];
/// let values = vec![1.0; 4];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let visited = bfs(&graph, 0);
/// assert_eq!(visited.len(), 3);
/// ```
pub fn bfs<T: Float>(graph: &CsrMatrix<T>, start: usize) -> Vec<usize> {
    let (n, _) = graph.shape();
    if start >= n {
        return Vec::new();
    }

    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited[start] = true;

    while let Some(u) = queue.pop_front() {
        order.push(u);

        // Visit all neighbors
        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];

        for idx in row_start..row_end {
            let v = graph.col_indices()[idx];
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }

    order
}

/// Depth-First Search from a starting vertex
///
/// Returns the vertices in the order they were visited.
///
/// # Complexity
///
/// O(V + E) where V is number of vertices, E is number of edges
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::dfs};
///
/// let row_ptr = vec![0, 2, 3, 4];
/// let col_indices = vec![1, 2, 2, 0];
/// let values = vec![1.0; 4];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let visited = dfs(&graph, 0);
/// assert_eq!(visited.len(), 3);
/// ```
pub fn dfs<T: Float>(graph: &CsrMatrix<T>, start: usize) -> Vec<usize> {
    let (n, _) = graph.shape();
    if start >= n {
        return Vec::new();
    }

    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut stack = vec![start];

    while let Some(u) = stack.pop() {
        if visited[u] {
            continue;
        }

        visited[u] = true;
        order.push(u);

        // Visit all neighbors (in reverse to maintain order)
        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];

        for idx in (row_start..row_end).rev() {
            let v = graph.col_indices()[idx];
            if !visited[v] {
                stack.push(v);
            }
        }
    }

    order
}

/// Find all connected components in an undirected graph
///
/// Returns information about connected components including
/// the number of components, labels for each vertex, and component sizes.
///
/// # Complexity
///
/// O(V + E) where V is number of vertices, E is number of edges
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::connected_components};
///
/// // Two disconnected triangles
/// let row_ptr = vec![0, 2, 4, 6, 8, 10, 12];
/// let col_indices = vec![1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4];
/// let values = vec![1.0; 12];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (6, 6)).unwrap();
///
/// let components = connected_components(&graph);
/// assert_eq!(components.num_components, 2);
/// assert_eq!(components.sizes, vec![3, 3]);
/// ```
pub fn connected_components<T: Float>(graph: &CsrMatrix<T>) -> ConnectedComponents {
    let (n, _) = graph.shape();
    let mut labels = vec![usize::MAX; n];
    let mut component_id = 0;

    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }

        // BFS from this vertex
        let mut queue = VecDeque::new();
        queue.push_back(start);
        labels[start] = component_id;

        while let Some(u) = queue.pop_front() {
            let row_start = graph.row_ptr()[u];
            let row_end = graph.row_ptr()[u + 1];

            for idx in row_start..row_end {
                let v = graph.col_indices()[idx];
                if labels[v] == usize::MAX {
                    labels[v] = component_id;
                    queue.push_back(v);
                }
            }
        }

        component_id += 1;
    }

    // Compute component sizes
    let mut sizes = vec![0; component_id];
    for &label in &labels {
        if label != usize::MAX {
            sizes[label] += 1;
        }
    }

    ConnectedComponents {
        num_components: component_id,
        labels,
        sizes,
    }
}

/// Dijkstra's shortest path algorithm
///
/// Computes shortest distances from a source vertex to all other vertices.
/// Returns `None` for unreachable vertices.
///
/// # Requirements
///
/// - Graph weights must be non-negative
///
/// # Complexity
///
/// O((V + E) log V) with binary heap
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::dijkstra};
///
/// let row_ptr = vec![0, 2, 3, 4];
/// let col_indices = vec![1, 2, 2, 0];
/// let values = vec![1.0, 4.0, 2.0, 1.0];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let distances = dijkstra(&graph, 0);
/// assert_eq!(distances[0], Some(0.0));
/// assert_eq!(distances[1], Some(1.0));
/// assert_eq!(distances[2], Some(3.0)); // via vertex 1
/// ```
pub fn dijkstra<T: Float + std::cmp::PartialOrd>(
    graph: &CsrMatrix<T>,
    source: usize,
) -> Vec<Option<T>> {
    let (n, _) = graph.shape();
    if source >= n {
        return vec![None; n];
    }

    let mut dist = vec![None; n];
    let mut visited = vec![false; n];

    // Use Vec as simple priority queue (not optimal but simple)
    let mut pq = Vec::new();
    pq.push((T::zero(), source));
    dist[source] = Some(T::zero());

    while !pq.is_empty() {
        // Find minimum distance vertex
        let mut min_idx = 0;
        let mut min_dist = pq[0].0;
        for (i, &(dist, _)) in pq.iter().enumerate().skip(1) {
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        let (d, u) = pq.swap_remove(min_idx);
        if visited[u] {
            continue;
        }
        visited[u] = true;

        // Relax edges
        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];

        for idx in row_start..row_end {
            let v = graph.col_indices()[idx];
            let weight = graph.values()[idx];

            if weight < T::zero() {
                continue; // Skip negative weights
            }

            let new_dist = d + weight;
            if dist[v].is_none() || new_dist < dist[v].unwrap() {
                dist[v] = Some(new_dist);
                pq.push((new_dist, v));
            }
        }
    }

    dist
}

/// Compute the degree of each vertex
///
/// Returns two vectors:
/// - `out_degree`: Number of outgoing edges for each vertex
/// - `in_degree`: Number of incoming edges for each vertex
///
/// For undirected graphs, out_degree == in_degree.
///
/// # Complexity
///
/// O(V + E)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::vertex_degrees};
///
/// let row_ptr = vec![0, 2, 3, 4];
/// let col_indices = vec![1, 2, 2, 0];
/// let values = vec![1.0; 4];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let (out_deg, in_deg) = vertex_degrees(&graph);
/// assert_eq!(out_deg, vec![2, 1, 1]);
/// ```
pub fn vertex_degrees<T: Float>(graph: &CsrMatrix<T>) -> (Vec<usize>, Vec<usize>) {
    let (n, _) = graph.shape();
    let mut out_degree = vec![0; n];
    let mut in_degree = vec![0; n];

    for (u, deg) in out_degree.iter_mut().enumerate().take(n) {
        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];
        *deg = row_end - row_start;

        for idx in row_start..row_end {
            let v = graph.col_indices()[idx];
            in_degree[v] += 1;
        }
    }

    (out_degree, in_degree)
}

/// Detect if graph has cycles using DFS
///
/// Returns `true` if the directed graph contains at least one cycle.
///
/// # Complexity
///
/// O(V + E)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::has_cycle};
///
/// // Acyclic: 0 -> 1 -> 2
/// let row_ptr = vec![0, 1, 2, 2];
/// let col_indices = vec![1, 2];
/// let values = vec![1.0; 2];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
/// assert!(!has_cycle(&graph));
///
/// // Cyclic: 0 -> 1 -> 2 -> 0
/// let row_ptr = vec![0, 1, 2, 3];
/// let col_indices = vec![1, 2, 0];
/// let values = vec![1.0; 3];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
/// assert!(has_cycle(&graph));
/// ```
pub fn has_cycle<T: Float>(graph: &CsrMatrix<T>) -> bool {
    let (n, _) = graph.shape();
    let mut visited = vec![false; n];
    let mut rec_stack = vec![false; n];

    fn dfs_cycle<T: Float>(
        u: usize,
        graph: &CsrMatrix<T>,
        visited: &mut [bool],
        rec_stack: &mut [bool],
    ) -> bool {
        visited[u] = true;
        rec_stack[u] = true;

        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];

        for idx in row_start..row_end {
            let v = graph.col_indices()[idx];

            if !visited[v] {
                if dfs_cycle(v, graph, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack[v] {
                return true; // Back edge found
            }
        }

        rec_stack[u] = false;
        false
    }

    for u in 0..n {
        if !visited[u] && dfs_cycle(u, graph, &mut visited, &mut rec_stack) {
            return true;
        }
    }

    false
}

/// Strongly connected components using Tarjan's algorithm
///
/// Finds all strongly connected components in a directed graph.
/// A strongly connected component is a maximal set of vertices where
/// every vertex is reachable from every other vertex.
///
/// # Complexity
///
/// O(V + E) time, O(V) space
///
/// # Returns
///
/// Vector of SCCs, where each SCC is a vector of vertex indices.
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::strongly_connected_components};
///
/// // Graph with one SCC: 0 -> 1 -> 2 -> 0
/// let row_ptr = vec![0, 1, 2, 3];
/// let col_indices = vec![1, 2, 0];
/// let values = vec![1.0; 3];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let sccs = strongly_connected_components(&graph);
/// assert_eq!(sccs.len(), 1);
/// assert_eq!(sccs[0].len(), 3);
/// ```
pub fn strongly_connected_components<T: Float>(graph: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let (n, _) = graph.shape();

    let mut index = 0;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![None; n];
    let mut lowlinks = vec![0; n];
    let mut sccs = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn strongconnect<T: Float>(
        v: usize,
        graph: &CsrMatrix<T>,
        index: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut [bool],
        indices: &mut [Option<usize>],
        lowlinks: &mut [usize],
        sccs: &mut Vec<Vec<usize>>,
    ) {
        // Set the depth index for v
        indices[v] = Some(*index);
        lowlinks[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        // Consider successors of v
        let row_start = graph.row_ptr()[v];
        let row_end = graph.row_ptr()[v + 1];

        for idx in row_start..row_end {
            let w = graph.col_indices()[idx];

            if indices[w].is_none() {
                // Successor w has not been visited; recurse
                strongconnect(w, graph, index, stack, on_stack, indices, lowlinks, sccs);
                lowlinks[v] = lowlinks[v].min(lowlinks[w]);
            } else if on_stack[w] {
                // Successor w is in stack and hence in current SCC
                lowlinks[v] = lowlinks[v].min(indices[w].unwrap());
            }
        }

        // If v is a root node, pop the stack to create an SCC
        if lowlinks[v] == indices[v].unwrap() {
            let mut scc = Vec::new();

            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                scc.push(w);

                if w == v {
                    break;
                }
            }

            sccs.push(scc);
        }
    }

    for v in 0..n {
        if indices[v].is_none() {
            strongconnect(
                v,
                graph,
                &mut index,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut lowlinks,
                &mut sccs,
            );
        }
    }

    sccs
}

/// Topological sort of a directed acyclic graph (DAG)
///
/// Returns vertices in topological order (dependencies before dependents).
/// Returns None if the graph contains a cycle.
///
/// # Complexity
///
/// O(V + E) time, O(V) space
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::topological_sort};
///
/// // DAG: 0 -> 1, 0 -> 2, 1 -> 2
/// let row_ptr = vec![0, 2, 3, 3];
/// let col_indices = vec![1, 2, 2];
/// let values = vec![1.0; 3];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let sorted = topological_sort(&graph).unwrap();
/// assert_eq!(sorted[0], 0); // 0 must come first
/// ```
pub fn topological_sort<T: Float>(graph: &CsrMatrix<T>) -> Option<Vec<usize>> {
    let (n, _) = graph.shape();

    // Compute in-degrees
    let mut in_degree = vec![0; n];
    for u in 0..n {
        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];

        for idx in row_start..row_end {
            let v = graph.col_indices()[idx];
            in_degree[v] += 1;
        }
    }

    // Initialize queue with vertices having in-degree 0
    let mut queue = VecDeque::new();
    #[allow(clippy::needless_range_loop)]
    for v in 0..n {
        if in_degree[v] == 0 {
            queue.push_back(v);
        }
    }

    let mut sorted = Vec::new();

    while let Some(u) = queue.pop_front() {
        sorted.push(u);

        // Decrease in-degree of neighbors
        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];

        for idx in row_start..row_end {
            let v = graph.col_indices()[idx];
            in_degree[v] -= 1;

            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    // Check if all vertices were processed (no cycle)
    if sorted.len() == n {
        Some(sorted)
    } else {
        None // Graph has a cycle
    }
}

/// Check if a graph is bipartite
///
/// A graph is bipartite if its vertices can be divided into two disjoint sets
/// such that every edge connects vertices from different sets.
///
/// # Complexity
///
/// O(V + E) time, O(V) space
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::is_bipartite};
///
/// // Bipartite graph: 0 - 1 - 2
/// let row_ptr = vec![0, 1, 3, 4];
/// let col_indices = vec![1, 0, 2, 1];
/// let values = vec![1.0; 4];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// assert!(is_bipartite(&graph));
/// ```
pub fn is_bipartite<T: Float>(graph: &CsrMatrix<T>) -> bool {
    let (n, _) = graph.shape();

    // Color: None = unvisited, Some(0) or Some(1) = colors
    let mut color = vec![None; n];

    // Try to 2-color the graph using BFS
    for start in 0..n {
        if color[start].is_some() {
            continue;
        }

        let mut queue = VecDeque::new();
        queue.push_back(start);
        color[start] = Some(0);

        while let Some(u) = queue.pop_front() {
            let u_color = color[u].unwrap();

            let row_start = graph.row_ptr()[u];
            let row_end = graph.row_ptr()[u + 1];

            for idx in row_start..row_end {
                let v = graph.col_indices()[idx];

                match color[v] {
                    None => {
                        // Assign opposite color
                        color[v] = Some(1 - u_color);
                        queue.push_back(v);
                    }
                    Some(v_color) if v_color == u_color => {
                        // Same color as parent - not bipartite
                        return false;
                    }
                    _ => {
                        // Different color - OK
                    }
                }
            }
        }
    }

    true
}

/// PageRank algorithm for ranking vertices by importance
///
/// Computes the PageRank score for each vertex in a directed graph.
/// PageRank is widely used in search engines, recommendation systems,
/// and social network analysis.
///
/// # Algorithm
///
/// Uses the power iteration method to solve:
/// PR(v) = (1-d)/N + d * Σ(PR(u) / outdegree(u))
/// where d is the damping factor (typically 0.85).
///
/// # Complexity
///
/// O((V + E) × iterations) where iterations is typically 20-50
///
/// # Arguments
///
/// - `graph`: Directed graph as CSR sparse matrix (edge from i to j means A\[i,j\] != 0)
/// - `damping`: Damping factor (typically 0.85), must be in (0, 1)
/// - `max_iter`: Maximum number of iterations
/// - `tol`: Convergence tolerance
///
/// # Returns
///
/// Vector of PageRank scores (sums to 1.0)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::pagerank};
///
/// // Simple graph: 0 -> 1 -> 2 -> 0 (cycle)
/// let row_ptr = vec![0, 1, 2, 3];
/// let col_indices = vec![1, 2, 0];
/// let values = vec![1.0; 3];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();
/// // All nodes should have equal rank (approximately 1/3 each)
/// for &rank in &ranks {
///     assert!((rank - 1.0/3.0).abs() < 1e-4);
/// }
/// ```
pub fn pagerank<T: Float>(
    graph: &CsrMatrix<T>,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    let (n, m) = graph.shape();
    if n != m {
        return Err(format!("Graph must be square, got {}x{}", n, m));
    }
    if !(0.0 < damping && damping < 1.0) {
        return Err(format!("Damping factor must be in (0,1), got {}", damping));
    }

    // Compute out-degrees
    let mut out_degree = vec![0.0; n];
    for (i, out_deg) in out_degree.iter_mut().enumerate().take(n) {
        let row_start = graph.row_ptr()[i];
        let row_end = graph.row_ptr()[i + 1];
        *out_deg = (row_end - row_start) as f64;
    }

    // Initialize PageRank: uniform distribution
    let mut rank = vec![1.0 / n as f64; n];
    let mut rank_new = vec![0.0; n];

    let base_rank = (1.0 - damping) / n as f64;

    for _ in 0..max_iter {
        // Compute new ranks
        rank_new.iter_mut().for_each(|r| *r = base_rank);

        // Add contributions from incoming edges
        for i in 0..n {
            if out_degree[i] == 0.0 {
                // Dangling node: distribute equally
                let contribution = damping * rank[i] / n as f64;
                rank_new.iter_mut().for_each(|r| *r += contribution);
            } else {
                let row_start = graph.row_ptr()[i];
                let row_end = graph.row_ptr()[i + 1];
                let contribution = damping * rank[i] / out_degree[i];

                for idx in row_start..row_end {
                    let j = graph.col_indices()[idx];
                    rank_new[j] += contribution;
                }
            }
        }

        // Check convergence (L1 norm of difference)
        let mut diff = 0.0;
        for i in 0..n {
            diff += (rank_new[i] - rank[i]).abs();
        }

        // Swap ranks
        std::mem::swap(&mut rank, &mut rank_new);

        if diff < tol {
            return Ok(rank);
        }
    }

    Ok(rank)
}

/// Minimum Spanning Tree using Kruskal's algorithm
///
/// Finds the minimum spanning tree (MST) of an undirected weighted graph.
/// The MST is a subset of edges that connects all vertices with minimum total weight.
///
/// # Algorithm
///
/// Uses Kruskal's algorithm with Union-Find data structure:
/// 1. Sort all edges by weight
/// 2. Iterate through edges in ascending weight order
/// 3. Add edge if it doesn't create a cycle (using Union-Find)
///
/// # Complexity
///
/// O(E log E) for sorting edges, O(E α(V)) for Union-Find operations,
/// where α is the inverse Ackermann function (effectively constant)
///
/// # Arguments
///
/// - `graph`: Undirected weighted graph as CSR matrix
///
/// # Returns
///
/// Vector of (u, v, weight) tuples representing MST edges, or error if graph is disconnected
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::minimum_spanning_tree};
///
/// // Triangle graph with weights
/// let row_ptr = vec![0, 2, 4, 6];
/// let col_indices = vec![1, 2, 0, 2, 0, 1];
/// let values = vec![1.0, 3.0, 1.0, 2.0, 3.0, 2.0];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let mst = minimum_spanning_tree(&graph).unwrap();
/// // MST should have 2 edges (n-1 for n vertices)
/// assert_eq!(mst.len(), 2);
/// // Total weight should be 3.0 (edges 0-1 and 1-2)
/// let total_weight: f64 = mst.iter().map(|(_, _, w)| w).sum();
/// assert!((total_weight - 3.0).abs() < 1e-6);
/// ```
pub fn minimum_spanning_tree<T: Float>(
    graph: &CsrMatrix<T>,
) -> Result<Vec<(usize, usize, f64)>, String> {
    let (n, m) = graph.shape();
    if n != m {
        return Err(format!("Graph must be square, got {}x{}", n, m));
    }

    // Extract all edges with weights
    let mut edges = Vec::new();
    for i in 0..n {
        let row_start = graph.row_ptr()[i];
        let row_end = graph.row_ptr()[i + 1];

        for idx in row_start..row_end {
            let j = graph.col_indices()[idx];
            // For undirected graph, only consider i < j to avoid duplicates
            if i < j {
                let weight = graph.values()[idx].to_f64().unwrap_or(0.0);
                edges.push((i, j, weight));
            }
        }
    }

    // Sort edges by weight (ascending)
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find data structure
    let mut parent = (0..n).collect::<Vec<_>>();
    let mut rank = vec![0; n];

    // Find with path compression
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            let next = parent[x];
            parent[x] = parent[parent[x]]; // Path compression
            x = next;
        }
        x
    }

    // Union by rank
    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) -> bool {
        let root_x = find(parent, x);
        let root_y = find(parent, y);

        if root_x == root_y {
            return false; // Already in same set
        }

        // Union by rank
        if rank[root_x] < rank[root_y] {
            parent[root_x] = root_y;
        } else if rank[root_x] > rank[root_y] {
            parent[root_y] = root_x;
        } else {
            parent[root_y] = root_x;
            rank[root_x] += 1;
        }
        true
    }

    // Kruskal's algorithm
    let mut mst = Vec::new();
    for (u, v, weight) in edges {
        if union(&mut parent, &mut rank, u, v) {
            mst.push((u, v, weight));
            if mst.len() == n - 1 {
                break; // MST complete
            }
        }
    }

    // Check if graph is connected
    if mst.len() != n - 1 {
        return Err(format!(
            "Graph is disconnected: MST has {} edges, expected {}",
            mst.len(),
            n - 1
        ));
    }

    Ok(mst)
}

/// Graph coloring using greedy algorithm
///
/// Assigns colors to vertices such that no two adjacent vertices share the same color.
/// Uses a greedy approach with degree-based vertex ordering for better results.
///
/// # Algorithm
///
/// 1. Order vertices by degree (descending) for better coloring
/// 2. For each vertex, assign the smallest available color not used by neighbors
/// 3. Returns color assignment for each vertex
///
/// # Complexity
///
/// O(V + E) time for greedy coloring
///
/// # Arguments
///
/// - `graph`: Undirected graph as CSR sparse matrix
///
/// # Returns
///
/// Vector of colors for each vertex (0-indexed) and the chromatic number (total colors used)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::graph_coloring};
///
/// // Triangle graph (needs 3 colors)
/// let row_ptr = vec![0, 2, 4, 6];
/// let col_indices = vec![1, 2, 0, 2, 0, 1];
/// let values = vec![1.0; 6];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let (colors, num_colors) = graph_coloring(&graph).unwrap();
/// assert_eq!(num_colors, 3); // Triangle needs 3 colors
/// // Adjacent vertices should have different colors
/// assert_ne!(colors[0], colors[1]);
/// assert_ne!(colors[1], colors[2]);
/// assert_ne!(colors[0], colors[2]);
/// ```
pub fn graph_coloring<T: Float>(graph: &CsrMatrix<T>) -> Result<(Vec<usize>, usize), String> {
    let (n, m) = graph.shape();
    if n != m {
        return Err(format!("Graph must be square, got {}x{}", n, m));
    }

    // Compute degrees for vertex ordering
    let mut degrees: Vec<(usize, usize)> = (0..n)
        .map(|i| {
            let row_start = graph.row_ptr()[i];
            let row_end = graph.row_ptr()[i + 1];
            (i, row_end - row_start)
        })
        .collect();

    // Sort vertices by degree (descending) - higher degree vertices colored first
    degrees.sort_by(|a, b| b.1.cmp(&a.1));

    // Initialize colors (uncolored = usize::MAX)
    let mut colors = vec![usize::MAX; n];

    // Color each vertex
    for &(vertex, _) in &degrees {
        // Find colors used by neighbors
        let mut used_colors = vec![false; n]; // Worst case: n colors

        let row_start = graph.row_ptr()[vertex];
        let row_end = graph.row_ptr()[vertex + 1];

        for idx in row_start..row_end {
            let neighbor = graph.col_indices()[idx];
            if neighbor != vertex && colors[neighbor] != usize::MAX {
                used_colors[colors[neighbor]] = true;
            }
        }

        // Assign smallest available color
        for (color, &used) in used_colors.iter().enumerate() {
            if !used {
                colors[vertex] = color;
                break;
            }
        }
    }

    // Find chromatic number (maximum color + 1)
    let chromatic_number = colors.iter().max().map(|&c| c + 1).unwrap_or(0);

    Ok((colors, chromatic_number))
}

/// Bellman-Ford algorithm for shortest paths with negative edge weights
///
/// Computes shortest paths from a source vertex to all other vertices,
/// handling graphs with negative edge weights. Can also detect negative cycles.
///
/// # Algorithm
///
/// Iteratively relaxes all edges V-1 times. If a shorter path is found
/// in the Vth iteration, a negative cycle exists.
///
/// # Complexity
///
/// O(V × E) time, O(V) space
///
/// # Arguments
///
/// - `graph`: Directed weighted graph as CSR matrix
/// - `source`: Source vertex
///
/// # Returns
///
/// `Ok(distances)` where distances\[i\] is shortest path length to vertex i,
/// or `Err` if negative cycle is detected
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, graph::bellman_ford};
///
/// // Simple graph: 0 -> 1 (weight 5), 1 -> 2 (weight -3)
/// let row_ptr = vec![0, 1, 2, 2];
/// let col_indices = vec![1, 2];
/// let values = vec![5.0, -3.0];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let distances = bellman_ford(&graph, 0).unwrap();
/// assert_eq!(distances[0], 0.0); // Source
/// assert_eq!(distances[1], 5.0); // 0->1
/// assert_eq!(distances[2], 2.0); // 0->1->2 = 5 + (-3) = 2
/// ```
pub fn bellman_ford<T: Float>(graph: &CsrMatrix<T>, source: usize) -> Result<Vec<f64>, String> {
    let (n, m) = graph.shape();
    if n != m {
        return Err(format!("Graph must be square, got {}x{}", n, m));
    }
    if source >= n {
        return Err(format!("Source {} out of bounds (n={})", source, n));
    }

    // Initialize distances
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;

    // Relax edges V-1 times
    for _ in 0..(n - 1) {
        let mut updated = false;

        for u in 0..n {
            if dist[u].is_infinite() {
                continue;
            }

            let row_start = graph.row_ptr()[u];
            let row_end = graph.row_ptr()[u + 1];

            for idx in row_start..row_end {
                let v = graph.col_indices()[idx];
                let weight = graph.values()[idx].to_f64().unwrap_or(0.0);

                let new_dist = dist[u] + weight;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    updated = true;
                }
            }
        }

        // Early termination if no updates
        if !updated {
            break;
        }
    }

    // Check for negative cycles
    for u in 0..n {
        if dist[u].is_infinite() {
            continue;
        }

        let row_start = graph.row_ptr()[u];
        let row_end = graph.row_ptr()[u + 1];

        for idx in row_start..row_end {
            let v = graph.col_indices()[idx];
            let weight = graph.values()[idx].to_f64().unwrap_or(0.0);

            if dist[u] + weight < dist[v] {
                return Err("Negative cycle detected".to_string());
            }
        }
    }

    Ok(dist)
}

/// Finds a maximal independent set using a greedy algorithm.
///
/// An independent set is a set of vertices with no edges between them.
/// A maximal independent set cannot be extended by adding more vertices.
///
/// # Arguments
///
/// * `graph` - Sparse matrix representing an undirected graph (adjacency matrix)
///
/// # Returns
///
/// A vector of vertex indices forming a maximal independent set.
///
/// # Complexity
///
/// O(V + E) time, O(V) space where V is vertices, E is edges
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, graph};
///
/// // Triangle graph: all vertices connected
/// let row_ptr = vec![0, 2, 4, 6];
/// let col_indices = vec![1, 2, 0, 2, 0, 1];
/// let values = vec![1.0; 6];
/// let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let mis = graph::maximal_independent_set(&graph).unwrap();
/// assert_eq!(mis.len(), 1); // Can only pick one vertex from a triangle
/// ```
pub fn maximal_independent_set<T: Float>(graph: &CsrMatrix<T>) -> Result<Vec<usize>, String> {
    let n = graph.nrows();
    if n != graph.ncols() {
        return Err("Graph must be square".to_string());
    }

    let mut independent_set = Vec::new();
    let mut excluded = vec![false; n];

    // Greedy algorithm: pick vertices with minimum degree first
    let mut degrees: Vec<_> = (0..n)
        .map(|i| {
            let deg = graph.row_ptr()[i + 1] - graph.row_ptr()[i];
            (deg, i)
        })
        .collect();
    degrees.sort_unstable();

    for (_deg, v) in degrees {
        if !excluded[v] {
            // Add v to independent set
            independent_set.push(v);

            // Exclude v and all its neighbors
            excluded[v] = true;
            let start = graph.row_ptr()[v];
            let end = graph.row_ptr()[v + 1];
            for idx in start..end {
                let neighbor = graph.col_indices()[idx];
                excluded[neighbor] = true;
            }
        }
    }

    Ok(independent_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_simple() {
        // Linear graph: 0 -> 1 -> 2
        let row_ptr = vec![0, 1, 2, 2];
        let col_indices = vec![1, 2];
        let values = vec![1.0; 2];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let visited = bfs(&graph, 0);
        assert_eq!(visited, vec![0, 1, 2]);
    }

    #[test]
    fn test_bfs_disconnected() {
        // 0 -> 1, 2 alone
        let row_ptr = vec![0, 1, 1, 1];
        let col_indices = vec![1];
        let values = vec![1.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let visited = bfs(&graph, 0);
        assert_eq!(visited, vec![0, 1]); // Can't reach 2
    }

    #[test]
    fn test_dfs_simple() {
        let row_ptr = vec![0, 2, 3, 3];
        let col_indices = vec![1, 2, 2];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let visited = dfs(&graph, 0);
        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], 0);
    }

    #[test]
    fn test_connected_components_single() {
        // Fully connected triangle
        let row_ptr = vec![0, 2, 4, 6];
        let col_indices = vec![1, 2, 0, 2, 0, 1];
        let values = vec![1.0; 6];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.num_components, 1);
        assert_eq!(components.sizes, vec![3]);
    }

    #[test]
    fn test_connected_components_multiple() {
        // Two separate edges: 0-1, 2-3
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_indices = vec![1, 0, 3, 2];
        let values = vec![1.0; 4];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.num_components, 2);
        assert_eq!(components.sizes, vec![2, 2]);
    }

    #[test]
    fn test_dijkstra_simple() {
        // 0 -1-> 1 -2-> 2
        let row_ptr = vec![0, 1, 2, 2];
        let col_indices = vec![1, 2];
        let values = vec![1.0, 2.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let distances = dijkstra(&graph, 0);
        assert_eq!(distances[0], Some(0.0));
        assert_eq!(distances[1], Some(1.0));
        assert_eq!(distances[2], Some(3.0));
    }

    #[test]
    fn test_dijkstra_unreachable() {
        // 0 -> 1, 2 isolated
        let row_ptr = vec![0, 1, 1, 1];
        let col_indices = vec![1];
        let values = vec![1.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let distances = dijkstra(&graph, 0);
        assert_eq!(distances[2], None);
    }

    #[test]
    fn test_vertex_degrees() {
        let row_ptr = vec![0, 2, 3, 4];
        let col_indices = vec![1, 2, 2, 0];
        let values = vec![1.0; 4];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let (out_deg, in_deg) = vertex_degrees(&graph);
        assert_eq!(out_deg, vec![2, 1, 1]);
        assert_eq!(in_deg, vec![1, 1, 2]);
    }

    #[test]
    fn test_has_cycle_acyclic() {
        // DAG: 0 -> 1 -> 2
        let row_ptr = vec![0, 1, 2, 2];
        let col_indices = vec![1, 2];
        let values = vec![1.0; 2];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        assert!(!has_cycle(&graph));
    }

    #[test]
    fn test_has_cycle_cyclic() {
        // Cycle: 0 -> 1 -> 2 -> 0
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        assert!(has_cycle(&graph));
    }

    #[test]
    fn test_has_cycle_self_loop() {
        // Self loop: 0 -> 0
        let row_ptr = vec![0, 1];
        let col_indices = vec![0];
        let values = vec![1.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (1, 1)).unwrap();

        assert!(has_cycle(&graph));
    }

    #[test]
    fn test_strongly_connected_components_single() {
        // One SCC: 0 -> 1 -> 2 -> 0
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let sccs = strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_strongly_connected_components_multiple() {
        // Multiple SCCs: 0 -> 1 (two separate SCCs), 2 -> 3 -> 2 (one SCC)
        let row_ptr = vec![0, 1, 1, 2, 3];
        let col_indices = vec![1, 3, 2];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let sccs = strongly_connected_components(&graph);
        // 0 (alone), 1 (alone), 2-3 cycle (together) = 3 SCCs
        assert_eq!(sccs.len(), 3);
    }

    #[test]
    fn test_strongly_connected_components_dag() {
        // DAG: 0 -> 1 -> 2 (no cycles, each vertex is its own SCC)
        let row_ptr = vec![0, 1, 2, 2];
        let col_indices = vec![1, 2];
        let values = vec![1.0; 2];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let sccs = strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 3);
        // Each vertex is its own SCC
        for scc in sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    #[test]
    fn test_topological_sort_dag() {
        // DAG: 0 -> 1, 0 -> 2, 1 -> 2
        let row_ptr = vec![0, 2, 3, 3];
        let col_indices = vec![1, 2, 2];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let sorted = topological_sort(&graph).unwrap();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0], 0); // 0 must come first

        // Verify topological property: for each edge u -> v, u appears before v
        let pos: Vec<usize> = (0..3)
            .map(|i| sorted.iter().position(|&x| x == i).unwrap())
            .collect();
        assert!(pos[0] < pos[1]); // 0 before 1
        assert!(pos[0] < pos[2]); // 0 before 2
        assert!(pos[1] < pos[2]); // 1 before 2
    }

    #[test]
    fn test_topological_sort_cyclic() {
        // Cycle: 0 -> 1 -> 2 -> 0
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let sorted = topological_sort(&graph);
        assert!(sorted.is_none()); // Should fail due to cycle
    }

    #[test]
    fn test_topological_sort_empty() {
        let row_ptr = vec![0, 0, 0];
        let col_indices: Vec<usize> = vec![];
        let values: Vec<f64> = vec![];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let sorted = topological_sort(&graph).unwrap();
        assert_eq!(sorted.len(), 2);
    }

    #[test]
    fn test_is_bipartite_true() {
        // Bipartite: 0 - 1 - 2 (undirected edges represented as bidirectional)
        let row_ptr = vec![0, 1, 3, 4];
        let col_indices = vec![1, 0, 2, 1];
        let values = vec![1.0; 4];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        assert!(is_bipartite(&graph));
    }

    #[test]
    fn test_is_bipartite_false() {
        // Triangle: 0 - 1 - 2 - 0 (odd cycle, not bipartite)
        let row_ptr = vec![0, 2, 4, 6];
        let col_indices = vec![1, 2, 0, 2, 0, 1];
        let values = vec![1.0; 6];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        assert!(!is_bipartite(&graph));
    }

    #[test]
    fn test_is_bipartite_disconnected() {
        // Two separate edges: 0-1, 2-3 (both bipartite)
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_indices = vec![1, 0, 3, 2];
        let values = vec![1.0; 4];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        assert!(is_bipartite(&graph));
    }

    #[test]
    fn test_pagerank_cycle() {
        // Simple cycle: 0 -> 1 -> 2 -> 0
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();

        // All nodes should have equal rank (1/3 each)
        for &rank in &ranks {
            assert!((rank - 1.0 / 3.0).abs() < 1e-4);
        }

        // Ranks should sum to 1.0
        let sum: f64 = ranks.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_star() {
        // Star graph: 1,2,3 all point to 0
        let row_ptr = vec![0, 0, 1, 2, 3];
        let col_indices = vec![0, 0, 0];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();

        // Node 0 should have highest rank
        assert!(ranks[0] > ranks[1]);
        assert!(ranks[0] > ranks[2]);
        assert!(ranks[0] > ranks[3]);

        // Peripheral nodes should have equal rank
        assert!((ranks[1] - ranks[2]).abs() < 1e-6);
        assert!((ranks[2] - ranks[3]).abs() < 1e-6);

        // Ranks should sum to 1.0
        let sum: f64 = ranks.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_chain() {
        // Chain: 0 -> 1 -> 2 -> 3
        let row_ptr = vec![0, 1, 2, 3, 3];
        let col_indices = vec![1, 2, 3];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();

        // Node 3 (sink) should have highest rank
        assert!(ranks[3] > ranks[0]);
        assert!(ranks[3] > ranks[1]);
        assert!(ranks[3] > ranks[2]);

        // Ranks should sum to 1.0
        let sum: f64 = ranks.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_invalid_damping() {
        let row_ptr = vec![0, 1];
        let col_indices = vec![0];
        let values = vec![1.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (1, 1)).unwrap();

        // Damping = 0 (invalid)
        assert!(pagerank(&graph, 0.0, 100, 1e-6).is_err());

        // Damping = 1.0 (invalid)
        assert!(pagerank(&graph, 1.0, 100, 1e-6).is_err());

        // Damping = 1.5 (invalid)
        assert!(pagerank(&graph, 1.5, 100, 1e-6).is_err());

        // Damping = 0.85 (valid)
        assert!(pagerank(&graph, 0.85, 100, 1e-6).is_ok());
    }

    #[test]
    fn test_pagerank_non_square() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 1, 2];
        let values = vec![1.0; 4];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        assert!(pagerank(&graph, 0.85, 100, 1e-6).is_err());
    }

    #[test]
    fn test_mst_triangle() {
        // Triangle: 0-1 (weight 1), 1-2 (weight 2), 0-2 (weight 3)
        let row_ptr = vec![0, 2, 4, 6];
        let col_indices = vec![1, 2, 0, 2, 0, 1];
        let values = vec![1.0, 3.0, 1.0, 2.0, 3.0, 2.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let mst = minimum_spanning_tree(&graph).unwrap();

        // MST should have 2 edges (n-1)
        assert_eq!(mst.len(), 2);

        // Total weight should be 3.0 (edges 0-1=1.0 and 1-2=2.0)
        let total_weight: f64 = mst.iter().map(|(_, _, w)| w).sum();
        assert!((total_weight - 3.0).abs() < 1e-6);

        // Check that we have edges 0-1 and 1-2
        let edges: Vec<(usize, usize)> = mst.iter().map(|(u, v, _)| (*u, *v)).collect();
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
    }

    #[test]
    fn test_mst_line() {
        // Line: 0-1 (1), 1-2 (2), 2-3 (3)
        let row_ptr = vec![0, 1, 3, 5, 6];
        let col_indices = vec![1, 0, 2, 1, 3, 2];
        let values = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let mst = minimum_spanning_tree(&graph).unwrap();

        // MST should have 3 edges
        assert_eq!(mst.len(), 3);

        // Total weight should be 6.0 (1+2+3)
        let total_weight: f64 = mst.iter().map(|(_, _, w)| w).sum();
        assert!((total_weight - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_mst_complete_graph() {
        // Complete graph K4 with specific weights
        let row_ptr = vec![0, 3, 6, 9, 12];
        let col_indices = vec![
            1, 2, 3, // from 0
            0, 2, 3, // from 1
            0, 1, 3, // from 2
            0, 1, 2, // from 3
        ];
        let values = vec![
            1.0, 5.0, 4.0, // from 0
            1.0, 2.0, 6.0, // from 1
            5.0, 2.0, 3.0, // from 2
            4.0, 6.0, 3.0, // from 3
        ];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let mst = minimum_spanning_tree(&graph).unwrap();

        // MST should have 3 edges
        assert_eq!(mst.len(), 3);

        // Total weight should be 6.0 (edges 0-1=1, 1-2=2, 2-3=3)
        let total_weight: f64 = mst.iter().map(|(_, _, w)| w).sum();
        assert!((total_weight - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_mst_single_vertex() {
        // Single vertex (no edges needed)
        let row_ptr = vec![0, 0];
        let col_indices: Vec<usize> = vec![];
        let values: Vec<f64> = vec![];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (1, 1)).unwrap();

        let mst = minimum_spanning_tree(&graph).unwrap();

        // MST should have 0 edges (n-1 = 0)
        assert_eq!(mst.len(), 0);
    }

    #[test]
    fn test_mst_disconnected() {
        // Two separate edges: 0-1, 2-3 (disconnected)
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_indices = vec![1, 0, 3, 2];
        let values = vec![1.0, 1.0, 2.0, 2.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        // Should return error for disconnected graph
        assert!(minimum_spanning_tree(&graph).is_err());
    }

    #[test]
    fn test_mst_non_square() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 1, 2];
        let values = vec![1.0; 4];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        assert!(minimum_spanning_tree(&graph).is_err());
    }

    #[test]
    fn test_graph_coloring_triangle() {
        // Triangle: needs 3 colors
        let row_ptr = vec![0, 2, 4, 6];
        let col_indices = vec![1, 2, 0, 2, 0, 1];
        let values = vec![1.0; 6];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let (colors, num_colors) = graph_coloring(&graph).unwrap();

        // Triangle needs 3 colors
        assert_eq!(num_colors, 3);

        // Adjacent vertices should have different colors
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[0], colors[2]);
    }

    #[test]
    fn test_graph_coloring_bipartite() {
        // Bipartite graph: needs 2 colors
        // 0-1, 0-3, 2-1, 2-3
        let row_ptr = vec![0, 2, 4, 6, 8];
        let col_indices = vec![1, 3, 0, 2, 1, 3, 0, 2];
        let values = vec![1.0; 8];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let (colors, num_colors) = graph_coloring(&graph).unwrap();

        // Bipartite graph needs only 2 colors
        assert!(num_colors <= 2);

        // Check all edges have different colors
        for i in 0..4 {
            let row_start = graph.row_ptr()[i];
            let row_end = graph.row_ptr()[i + 1];
            for idx in row_start..row_end {
                let j = graph.col_indices()[idx];
                if i != j {
                    assert_ne!(colors[i], colors[j]);
                }
            }
        }
    }

    #[test]
    fn test_graph_coloring_line() {
        // Line graph: 0-1-2-3 (needs 2 colors)
        let row_ptr = vec![0, 1, 3, 5, 6];
        let col_indices = vec![1, 0, 2, 1, 3, 2];
        let values = vec![1.0; 6];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let (colors, num_colors) = graph_coloring(&graph).unwrap();

        // Line graph needs only 2 colors (alternating)
        assert!(num_colors <= 2);

        // Adjacent vertices have different colors
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[2], colors[3]);
    }

    #[test]
    fn test_graph_coloring_complete() {
        // Complete graph K4: needs 4 colors
        let row_ptr = vec![0, 3, 6, 9, 12];
        let col_indices = vec![
            1, 2, 3, // from 0
            0, 2, 3, // from 1
            0, 1, 3, // from 2
            0, 1, 2, // from 3
        ];
        let values = vec![1.0; 12];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let (colors, num_colors) = graph_coloring(&graph).unwrap();

        // Complete graph K4 needs 4 colors
        assert_eq!(num_colors, 4);

        // All vertices have different colors
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(colors[i], colors[j]);
            }
        }
    }

    #[test]
    fn test_bellman_ford_simple() {
        // Simple path: 0 -> 1 -> 2
        let row_ptr = vec![0, 1, 2, 2];
        let col_indices = vec![1, 2];
        let values = vec![5.0, 3.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let dist = bellman_ford(&graph, 0).unwrap();

        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[1], 5.0);
        assert_eq!(dist[2], 8.0);
    }

    #[test]
    fn test_bellman_ford_negative_weights() {
        // Graph with negative edges: 0 -> 1 (5), 1 -> 2 (-3)
        let row_ptr = vec![0, 1, 2, 2];
        let col_indices = vec![1, 2];
        let values = vec![5.0, -3.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let dist = bellman_ford(&graph, 0).unwrap();

        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[1], 5.0);
        assert_eq!(dist[2], 2.0); // 5 + (-3) = 2
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        // Negative cycle: 0 -> 1 (1), 1 -> 2 (-3), 2 -> 0 (1)
        // Total cycle weight: 1 + (-3) + 1 = -1 (negative!)
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0, -3.0, 1.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let result = bellman_ford(&graph, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Negative cycle"));
    }

    #[test]
    fn test_bellman_ford_disconnected() {
        // Disconnected graph: 0 -> 1, 2 isolated
        let row_ptr = vec![0, 1, 1, 1];
        let col_indices = vec![1];
        let values = vec![5.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let dist = bellman_ford(&graph, 0).unwrap();

        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[1], 5.0);
        assert!(dist[2].is_infinite()); // Unreachable
    }

    #[test]
    fn test_bellman_ford_multiple_paths() {
        // Multiple paths: 0 -> 1 (4), 0 -> 2 (2), 1 -> 2 (1), 2 -> 3 (3)
        // Shortest to 2: 0 -> 2 = 2 (not 0 -> 1 -> 2 = 5)
        let row_ptr = vec![0, 2, 3, 4, 4];
        let col_indices = vec![1, 2, 2, 3];
        let values = vec![4.0, 2.0, 1.0, 3.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let dist = bellman_ford(&graph, 0).unwrap();

        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[1], 4.0);
        assert_eq!(dist[2], 2.0); // Direct path is shorter
        assert_eq!(dist[3], 5.0); // 0 -> 2 -> 3
    }

    #[test]
    fn test_bellman_ford_invalid_source() {
        let row_ptr = vec![0, 1];
        let col_indices = vec![0];
        let values = vec![1.0];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (1, 1)).unwrap();

        assert!(bellman_ford(&graph, 5).is_err());
    }

    #[test]
    fn test_scc_additional_cycle() {
        // Additional test: Single cycle 0 -> 1 -> 2 -> 0
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0; 3];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let sccs = strongly_connected_components(&graph);

        // All vertices should be in one SCC
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_scc_additional_dag() {
        // Additional test: DAG (no cycles): 0 -> 1 -> 2
        let row_ptr = vec![0, 1, 2, 2];
        let col_indices = vec![1, 2];
        let values = vec![1.0; 2];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let sccs = strongly_connected_components(&graph);

        // Each vertex should be its own SCC
        assert_eq!(sccs.len(), 3);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    #[test]
    fn test_mis_triangle() {
        // Triangle: all vertices connected
        let row_ptr = vec![0, 2, 4, 6];
        let col_indices = vec![1, 2, 0, 2, 0, 1];
        let values = vec![1.0; 6];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let mis = maximal_independent_set(&graph).unwrap();

        // Can only pick one vertex from a triangle
        assert_eq!(mis.len(), 1);
    }

    #[test]
    fn test_mis_path() {
        // Path: 0 - 1 - 2 - 3
        let row_ptr = vec![0, 1, 3, 5, 6];
        let col_indices = vec![1, 0, 2, 1, 3, 2];
        let values = vec![1.0; 6];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let mis = maximal_independent_set(&graph).unwrap();

        // Can pick 0 and 2, or 1 and 3 (2 vertices)
        assert_eq!(mis.len(), 2);

        // Verify it's actually independent
        for i in 0..mis.len() {
            for j in (i + 1)..mis.len() {
                let v1 = mis[i];
                let v2 = mis[j];
                // Check no edge between v1 and v2
                let start = graph.row_ptr()[v1];
                let end = graph.row_ptr()[v1 + 1];
                let neighbors: Vec<_> = graph.col_indices()[start..end].to_vec();
                assert!(!neighbors.contains(&v2));
            }
        }
    }

    #[test]
    fn test_mis_independent_vertices() {
        // No edges: all vertices are independent
        let row_ptr = vec![0, 0, 0, 0];
        let col_indices = vec![];
        let values: Vec<f64> = vec![];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let mis = maximal_independent_set(&graph).unwrap();

        // All vertices should be in MIS
        assert_eq!(mis.len(), 3);
    }

    #[test]
    fn test_mis_complete_graph() {
        // Complete graph K4: all pairs connected
        let row_ptr = vec![0, 3, 6, 9, 12];
        let col_indices = vec![
            1, 2, 3, // 0 -> 1,2,3
            0, 2, 3, // 1 -> 0,2,3
            0, 1, 3, // 2 -> 0,1,3
            0, 1, 2, // 3 -> 0,1,2
        ];
        let values = vec![1.0; 12];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let mis = maximal_independent_set(&graph).unwrap();

        // Can only pick one vertex from complete graph
        assert_eq!(mis.len(), 1);
    }

    #[test]
    fn test_mis_bipartite() {
        // Bipartite graph K_{2,2}: {0,1} connected to {2,3}
        let row_ptr = vec![0, 2, 4, 6, 8];
        let col_indices = vec![
            2, 3, // 0 -> 2,3
            2, 3, // 1 -> 2,3
            0, 1, // 2 -> 0,1
            0, 1, // 3 -> 0,1
        ];
        let values = vec![1.0; 8];
        let graph = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

        let mis = maximal_independent_set(&graph).unwrap();

        // Can pick {0,1} or {2,3} (2 vertices)
        assert_eq!(mis.len(), 2);
    }
}
