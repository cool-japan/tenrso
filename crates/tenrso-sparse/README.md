# tenrso-sparse

Sparse tensor formats, operations, linear solvers, graph algorithms, and more for TenRSo.

**Version:** 0.1.0-rc.1 | **Status:** RC.1 — 426 tests passing (3 ignored), 100% pass rate | **Last Updated:** 2026-03-06

## Overview

`tenrso-sparse` provides efficient sparse tensor storage formats, operations, iterative linear solvers, eigensolvers, and graph algorithms:

### Sparse Formats
- **COO** - Coordinate format (triplets); for construction and format conversion
- **CSR** - Compressed Sparse Row; optimized for row-wise access and SpMV
- **CSC** - Compressed Sparse Column; optimized for column-wise access
- **BCSR** - Block Compressed Sparse Row; for block-structured sparsity
- **ELL** - ELLPACK format; suited for SIMD/GPU execution
- **DIA** - Diagonal format; for banded/diagonal-dominant matrices
- **CSF** - Compressed Sparse Fiber; true N-D sparsity (feature-gated: `csf`)
- **HiCOO** - Hierarchical COO; cache-friendly N-D format (feature-gated: `csf`)

### Operations (36 total)
- Element-wise: add, sub, mul, div (sparse-sparse, sparse-scalar)
- Transcendentals: exp, ln, sqrt, abs, sign, sin, cos, pow
- Binary: min, max, gt, lt, ge, le, eq, ne
- Reductions: sum, min_val, max_val, frobenius norm, trace
- Matrix products: SpMV, SpMM, SpSpMM, spmv_transpose
- Masked operations: einsum with sparse masks

### Iterative Linear Solvers (6)
- **CG** - Conjugate Gradient; for symmetric positive definite systems
- **BiCGSTAB** - Biconjugate Gradient Stabilized; for nonsymmetric systems
- **GMRES** - Generalized Minimal Residual; with configurable restart
- **MINRES** - Minimum Residual; for symmetric indefinite systems
- **CGNE** - CG on Normal Equations; for overdetermined least-squares (m > n)
- **CGNR** - CG on Normal Residual; for underdetermined systems (m < n)

### Preconditioners (4)
- **Identity** - No preconditioning (baseline)
- **ILU(0)** - Incomplete LU factorization
- **Jacobi** - Diagonal (simplest, O(nnz) construction)
- **SSOR** - Symmetric Successive Over-Relaxation (configurable omega)

### Eigensolvers (3)
- **Power Iteration** - Dominant eigenvalue/eigenvector
- **Inverse Power Iteration** - Smallest eigenvalue (via ILU)
- **Lanczos Algorithm** - Multiple eigenvalues for symmetric matrices

### Graph Algorithms (14)
- BFS, DFS - Breadth-first and depth-first traversal
- Dijkstra - Single-source shortest paths (non-negative weights)
- Bellman-Ford - Shortest paths with negative weights, negative cycle detection
- SCC - Strongly Connected Components (Tarjan/Kosaraju)
- PageRank - Vertex importance ranking (power iteration, configurable damping)
- MST - Minimum Spanning Tree (Kruskal's + Union-Find with path compression)
- Graph Coloring - Greedy coloring with degree-based vertex ordering
- MIS - Maximal Independent Set (greedy, degree-based)
- Bipartite detection, Topological Sort, Has Cycle, Vertex Degrees, Connected Components

### Matrix Utilities
- **Constructors** (5): Graph Laplacian, Adjacency, 2D Poisson (5-point stencil), Tridiagonal, Identity
- **Factorizations** (2): ILU(0), IC(0) with forward/backward substitution
- **Reordering** (2): Reverse Cuthill-McKee (RCM), Approximate Minimum Degree (AMD)
- **I/O**: Matrix Market format (read/write)

## Features

- 8 sparse formats optimized for different access patterns
- Efficient conversion between all formats
- 36 sparse matrix operations with full type genericity
- 6 iterative linear solvers with preconditioning support
- 3 eigensolvers for spectral analysis
- 14 graph algorithms
- Sparsity statistics (nnz, density, structural analysis)
- Generic over scalar types (f32, f64, complex types)
- Parallel operations via Rayon

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tenrso-sparse = "0.1"

# Enable CSF and HiCOO formats
tenrso-sparse = { version = "0.1", features = ["csf"] }
```

### COO Format

```rust
use tenrso_sparse::Coo;

// Create sparse tensor from triplets
let indices = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
let values = vec![1.0f64, 2.0, 3.0];
let shape = vec![10, 10, 10];

let coo = Coo::from_triplets(indices, values, shape)?;
println!("NNZ: {}", coo.nnz());
println!("Density: {:.2}%", coo.density() * 100.0);
```

### CSR Operations

```rust
use tenrso_sparse::{Coo, Csr};

let csr: Csr<f64> = coo.to_csr()?;
let x = vec![1.0f64; csr.ncols()];
let y = csr.spmv(&x)?;
```

### Masked Einsum

```rust
use tenrso_sparse::MaskPack;
use tenrso_exec::einsum_ex;

// Create mask for selective computation
let mask = MaskPack::from_indices(sparse_indices);

// Compute only masked elements
let result = einsum_ex::<f32>("ij,jk->ik")
    .inputs(&[a, b])
    .hints(&ExecHints {
        mask: Some(mask),
        prefer_sparse: true,
        ..Default::default()
    })
    .run()?;
```

### Iterative Linear Solvers

```rust
use tenrso_sparse::solvers::{ConjugateGradient, BiCgStab, Gmres, SolverConfig};

let config = SolverConfig { max_iters: 1000, tol: 1e-10, ..Default::default() };

// CG for SPD systems
let (x, info) = ConjugateGradient::solve(&a, &b, &config)?;

// BiCGSTAB for nonsymmetric systems
let (x, info) = BiCgStab::solve(&a, &b, &config)?;

// GMRES with restart
let (x, info) = Gmres::solve_with_restart(&a, &b, 30, &config)?;

println!("Converged in {} iterations, residual: {:.2e}", info.iters, info.residual);
```

### Least-Squares Solvers

```rust
use tenrso_sparse::solvers::{Cgne, Cgnr};

// Overdetermined system (m > n): min ||Ax - b||
let (x, info) = Cgne::solve(&a, &b, &config)?;

// Underdetermined system (m < n): min-norm solution
let (x, info) = Cgnr::solve(&a, &b, &config)?;
```

### Preconditioned Solving

```rust
use tenrso_sparse::solvers::{ConjugateGradient, IluPreconditioner, JacobiPreconditioner};

// ILU(0) preconditioning
let precond = IluPreconditioner::new(&a)?;
let (x, info) = ConjugateGradient::solve_preconditioned(&a, &b, &precond, &config)?;

// Jacobi preconditioning
let precond = JacobiPreconditioner::new(&a);
let (x, info) = ConjugateGradient::solve_preconditioned(&a, &b, &precond, &config)?;
```

### Eigensolvers

```rust
use tenrso_sparse::eigensolvers::{power_iteration, inverse_power_iteration, lanczos};

// Dominant eigenvalue
let (eigenval, eigenvec, info) = power_iteration(&a, 1000, 1e-10)?;

// Smallest eigenvalue
let (eigenval, eigenvec, info) = inverse_power_iteration(&a, 1000, 1e-10)?;

// Multiple eigenvalues via Lanczos
let (eigenvals, eigenvecs, info) = lanczos(&a, 10, 1000, 1e-10)?;
```

### Graph Algorithms

```rust
use tenrso_sparse::graph::{bfs, dijkstra, page_rank, minimum_spanning_tree, graph_coloring};

// BFS traversal
let order = bfs(&adjacency, start_vertex)?;

// Shortest paths
let distances = dijkstra(&weighted_adj, source)?;

// PageRank (damping = 0.85)
let ranks = page_rank(&adj, 0.85, 1000, 1e-10)?;

// Minimum Spanning Tree (Kruskal's)
let mst_edges = minimum_spanning_tree(&weighted_adj)?;

// Graph Coloring
let (coloring, chromatic_number) = graph_coloring(&adj)?;
```

### Matrix Construction

```rust
use tenrso_sparse::constructors::{graph_laplacian, poisson_2d, tridiagonal};

// 2D Poisson matrix (finite difference)
let p = poisson_2d::<f64>(50, 50)?;  // 50x50 grid

// Tridiagonal matrix
let a = tridiagonal::<f64>(n, -1.0, 2.0, -1.0)?;

// Graph Laplacian from edge list
let l = graph_laplacian::<f64>(&edges, num_vertices)?;
```

### Matrix Market I/O

```rust
use tenrso_sparse::io::{read_matrix_market, write_matrix_market};

let csr = read_matrix_market::<f64>("matrix.mtx")?;
write_matrix_market(&csr, "output.mtx")?;
```

## Sparse Formats

### COO (Coordinate)
- **Use case:** Construction, format conversion, random access by index
- **Access:** O(nnz) scan
- **Memory:** ndim x nnz (indices) + nnz (values)

### CSR/CSC (Compressed Sparse Row/Column)
- **Use case:** Matrix-vector products, row/col access patterns
- **Access:** O(1) row/col start, O(nnz) iteration
- **Memory:** 2 x nnz + nrows/ncols

### BCSR (Block CSR)
- **Use case:** Block-structured sparsity (FEM, graph convolutions)
- **Access:** Block-level operations
- **Memory:** Efficient for dense blocks

### ELL (ELLPACK)
- **Use case:** Regular sparsity patterns, SIMD/GPU execution
- **Access:** Regular stride (SIMD-friendly)
- **Memory:** max_nnz_per_row x nrows

### DIA (Diagonal)
- **Use case:** Banded matrices, finite difference stencils
- **Access:** O(1) diagonal access
- **Memory:** n_diags x n (dense diagonals)

### CSF (Compressed Sparse Fiber)
- **Use case:** True N-dimensional sparsity, tensor operations
- **Access:** Fiber-based iteration
- **Memory:** Hierarchical compression
- **Feature:** Requires `csf` feature flag

### HiCOO (Hierarchical COO)
- **Use case:** Large sparse tensors with cache locality
- **Access:** Block-hierarchical iteration
- **Memory:** Better cache utilization than flat COO
- **Feature:** Requires `csf` feature flag

## Linear Solvers

| Solver | System Type | Complexity | Notes |
|--------|-------------|------------|-------|
| CG | SPD | O(nnz x sqrt(kappa)) | Optimal for SPD |
| BiCGSTAB | Nonsymmetric | O(nnz x iters) | Stable variant of BiCG |
| GMRES | General | O(nnz x m x iters) | m = restart parameter |
| MINRES | Symmetric indefinite | O(nnz x iters) | Lanczos + Givens rotations |
| CGNE | Overdetermined (m > n) | O((nnz_A + nnz_At) x iters) | Solves normal equations |
| CGNR | Underdetermined (m < n) | O((nnz_A + nnz_At) x iters) | Minimum-norm solution |

## Graph Algorithms

| Algorithm | Complexity | Applications |
|-----------|------------|--------------|
| BFS | O(V + E) | Level-set traversal, shortest hops |
| DFS | O(V + E) | Cycle detection, topological order |
| Dijkstra | O((V + E) log V) | Routing, shortest paths |
| Bellman-Ford | O(V x E) | Negative weights, arbitrage detection |
| SCC | O(V + E) | Strongly connected components |
| PageRank | O((V + E) x iters) | Search ranking, influence analysis |
| MST (Kruskal) | O(E log E + E x alpha(V)) | Network design, clustering |
| Graph Coloring | O(V + E) | Register allocation, scheduling |
| MIS | O(V + E) | Vertex cover, parallel processing |
| Bipartite | O(V + E) | Matching, two-colorability |
| Topological Sort | O(V + E) | Dependency ordering |
| Has Cycle | O(V + E) | DAG verification |
| Vertex Degrees | O(V + E) | Degree distribution |
| Connected Components | O(V + E) | Cluster identification |

## Performance

- **SpMV**: > 70% of dense GEMM throughput at 10% density
- **Masked einsum**: >= 5x speedup vs dense at 90% sparsity
- **Format conversion**: Parallel sorting for COO-to-CSR

## Examples

See `examples/` directory:
- `coo_basics.rs` - COO format usage and conversion
- `csr_operations.rs` - CSR SpMV, SpMM operations
- `masked_einsum.rs` - Selective computation with masks
- `format_conversion.rs` - Converting between formats
- `iterative_solvers.rs` - CG, BiCGSTAB, GMRES with preconditioning
- `reordering.rs` - RCM/AMD reordering for bandwidth reduction

## Feature Flags

- `default` = `["parallel"]` - Parallel operations enabled
- `parallel` - Multi-threaded format conversion and operations
- `csf` - Enable CSF and HiCOO formats for N-D sparse tensors

## Testing

**RC.1 test coverage:** 426 passing (3 ignored), 100% pass rate
- Property tests: 22 (proptest framework)
- Benchmark groups: 21
- Zero `todo!()` / `unimplemented!()` macros
- Zero clippy warnings
- Zero documentation warnings

```bash
# Run all tests
cargo test

# Run with nextest
cargo nextest run

# Run benchmarks
cargo bench

# Property tests
cargo test --test properties
```

## Dependencies

- **tenrso-core** - Tensor types
- **scirs2-core** - Array operations
- **indexmap** - Efficient hash maps for sparse construction
- **rayon** (optional) - Parallel operations

## License

Apache-2.0
