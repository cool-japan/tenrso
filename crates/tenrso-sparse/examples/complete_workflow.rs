//! Complete workflow example: Sparse linear system solution
//!
//! This example demonstrates a complete workflow for solving
//! a sparse linear system using multiple features:
//! 1. Load or create a sparse matrix
//! 2. Analyze sparsity pattern
//! 3. Apply matrix reordering for optimization
//! 4. Compute ILU preconditioner
//! 5. Solve with iterative solver
//! 6. Visualize results

use tenrso_sparse::{
    graph, io, patterns, reordering,
    solvers::{self, IluPreconditioner},
    viz, CooTensor, CsrMatrix,
};

fn main() {
    println!("=== Complete Sparse Linear System Workflow ===\n");

    // Workflow 1: Solve a Poisson equation (5-point stencil)
    solve_poisson_equation();

    // Workflow 2: Graph-based analysis
    graph_analysis_workflow();

    // Workflow 3: I/O and visualization
    io_visualization_workflow();
}

/// Solve a 2D Poisson equation using complete workflow
fn solve_poisson_equation() {
    println!("1. Solving 2D Poisson Equation (-∇²u = f)");
    println!("   Grid size: 5×5 (25 unknowns)");
    println!("   5-point stencil discretization\n");

    // Step 1: Create the sparse matrix (5-point stencil)
    let n = 5; // Grid size
    let size = n * n;

    let mut row_ptr = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;

            // Diagonal element
            col_indices.push(idx);
            values.push(4.0);

            // Left neighbor
            if j > 0 {
                col_indices.push(idx - 1);
                values.push(-1.0);
            }

            // Right neighbor
            if j < n - 1 {
                col_indices.push(idx + 1);
                values.push(-1.0);
            }

            // Top neighbor
            if i > 0 {
                col_indices.push(idx - n);
                values.push(-1.0);
            }

            // Bottom neighbor
            if i < n - 1 {
                col_indices.push(idx + n);
                values.push(-1.0);
            }

            row_ptr.push(col_indices.len());
        }
    }

    let a = CsrMatrix::new(row_ptr, col_indices, values, (size, size)).unwrap();

    println!("   Matrix created:");
    println!("     Size: {}×{}", size, size);
    println!("     NNZ: {}", a.nnz());
    println!("     Density: {:.2}%", a.density() * 100.0);

    // Step 2: Analyze sparsity pattern
    println!("\n   Sparsity analysis:");
    let (lower_bw, upper_bw) = patterns::bandwidth(&a);
    println!(
        "     Bandwidth: {} (lower) + {} (upper)",
        lower_bw, upper_bw
    );

    let is_sym = patterns::is_structurally_symmetric(&a);
    println!("     Structurally symmetric: {}", is_sym);

    let is_diag_dom = patterns::is_row_diagonally_dominant(&a);
    println!("     Diagonally dominant: {}", is_diag_dom);

    // Step 3: Apply RCM reordering
    println!("\n   Applying RCM reordering...");
    let perm = reordering::rcm(&a).unwrap();
    let a_reordered = reordering::permute_symmetric(&a, &perm).unwrap();

    let (lower_bw_new, upper_bw_new) = patterns::bandwidth(&a_reordered);
    println!(
        "     New bandwidth: {} (lower) + {} (upper)",
        lower_bw_new, upper_bw_new
    );
    let bw_reduction = ((lower_bw + upper_bw) - (lower_bw_new + upper_bw_new)) as f64
        / (lower_bw + upper_bw) as f64
        * 100.0;
    println!("     Bandwidth reduction: {:.1}%", bw_reduction.max(0.0));

    // Step 4: Create right-hand side (f = 1 everywhere)
    let b: Vec<f64> = vec![1.0; size];
    let b_reordered: Vec<f64> = (0..size).map(|i| b[perm[i]]).collect();

    // Step 5: Solve without preconditioning
    println!("\n   Solving with CG (no preconditioning)...");
    let (x_no_precond, info_no_precond) = solvers::cg::<f64, solvers::IdentityPreconditioner>(
        &a_reordered,
        &b_reordered,
        1000,
        1e-8,
        None,
    )
    .unwrap();
    println!("     {}", info_no_precond);

    // Step 6: Compute ILU(0) preconditioner
    println!("\n   Computing ILU(0) preconditioner...");
    match IluPreconditioner::from_matrix(&a_reordered) {
        Ok(precond) => {
            // Step 7: Solve with ILU preconditioning
            println!("   Solving with preconditioned CG...");
            match solvers::cg(&a_reordered, &b_reordered, 1000, 1e-8, Some(&precond)) {
                Ok((x_precond, info_precond)) => {
                    println!("     {}", info_precond);

                    // Step 8: Compare results
                    let iter_reduction = (info_no_precond.iterations as f64
                        - info_precond.iterations as f64)
                        / info_no_precond.iterations as f64
                        * 100.0;
                    println!(
                        "\n   Iteration reduction: {:.1}%",
                        iter_reduction.clamp(0.0, 100.0)
                    );

                    // Verify solutions are similar
                    let sol_diff: f64 = x_no_precond
                        .iter()
                        .zip(x_precond.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    println!("   Solution difference: {:.2e}", sol_diff);

                    // Show sample solution values
                    println!("\n   Sample solution values (first 5):");
                    for (i, &val) in x_precond.iter().enumerate().take(5) {
                        println!("     x[{}] = {:.6}", i, val);
                    }
                }
                Err(e) => {
                    println!("     Note: Preconditioned solver failed: {}", e);
                    println!("     This can happen with certain matrix structures.");
                    println!("\n   Solution without preconditioning (first 5):");
                    for (i, &val) in x_no_precond.iter().enumerate().take(5) {
                        println!("     x[{}] = {:.6}", i, val);
                    }
                }
            }
        }
        Err(e) => {
            println!("     ILU factorization failed: {}", e);
            println!("     Continuing with solution from unpreconditioned solver.");
            println!("\n   Solution (first 5):");
            for (i, &val) in x_no_precond.iter().enumerate().take(5) {
                println!("     x[{}] = {:.6}", i, val);
            }
        }
    }

    println!();
}

/// Graph analysis workflow
fn graph_analysis_workflow() {
    println!("2. Graph Analysis Workflow");

    // Create a directed graph
    let row_ptr = vec![0, 2, 4, 6, 8, 9];
    let col_indices = vec![
        1, 2, // vertex 0 -> {1, 2}
        2, 3, // vertex 1 -> {2, 3}
        3, 4, // vertex 2 -> {3, 4}
        0, 4, // vertex 3 -> {0, 4}
        3, // vertex 4 -> {3}
    ];
    let values = vec![1.0; col_indices.len()];
    let graph_matrix = CsrMatrix::new(row_ptr, col_indices, values, (5, 5)).unwrap();

    println!("   Directed graph with 5 vertices");
    println!("     Edges: {}", graph_matrix.nnz());

    // Compute vertex degrees
    let (in_degrees, out_degrees) = graph::vertex_degrees(&graph_matrix);
    println!("\n   Vertex degrees:");
    for i in 0..5 {
        println!(
            "     Vertex {}: in={}, out={}",
            i, in_degrees[i], out_degrees[i]
        );
    }

    // Check for cycles
    let has_cycle = graph::has_cycle(&graph_matrix);
    println!("\n   Has cycle: {}", has_cycle);

    // Try topological sort
    match graph::topological_sort(&graph_matrix) {
        Some(order) => println!("   Topological order: {:?}", order),
        None => println!("   Cannot compute topological order (graph has cycles)"),
    }

    // Find strongly connected components
    let sccs = graph::strongly_connected_components(&graph_matrix);
    println!("\n   Strongly connected components:");
    for (i, scc) in sccs.iter().enumerate() {
        println!("     Component {}: {:?}", i, scc);
    }

    // Check if bipartite (convert to undirected for this test)
    let is_bipartite = graph::is_bipartite(&graph_matrix);
    println!("\n   Is bipartite: {}", is_bipartite);

    println!();
}

/// I/O and visualization workflow
fn io_visualization_workflow() {
    println!("3. I/O and Visualization Workflow");

    // Create a small sparse matrix
    let indices = vec![
        vec![0, 0],
        vec![0, 2],
        vec![1, 1],
        vec![2, 0],
        vec![2, 2],
        vec![3, 3],
    ];
    let values = vec![4.0, -1.0, 4.0, -1.0, 4.0, 4.0];
    let shape = vec![4, 4];
    let coo = CooTensor::new(indices, values, shape).unwrap();

    println!("   Created 4×4 sparse matrix");
    println!("     NNZ: {}", coo.nnz());

    // Write to Matrix Market format (in-memory)
    let mut mtx_buffer = Vec::new();
    io::write_matrix_market(&coo, &mut mtx_buffer).unwrap();
    println!("\n   Matrix Market format (first 200 chars):");
    let mtx_str = String::from_utf8_lossy(&mtx_buffer);
    println!("{}", mtx_str.chars().take(200).collect::<String>());

    // Convert to CSR for visualization
    let csr = CsrMatrix::from_coo(&coo).unwrap();

    // ASCII visualization
    println!("\n   ASCII sparsity pattern:");
    let ascii_art = viz::ascii_pattern(&csr, 10, 10);
    println!("{}", ascii_art);

    // Bandwidth profile
    println!("   Bandwidth profile:");
    let (row_profile, col_profile) = viz::bandwidth_profile(&csr);
    println!("     Row profile:\n{}", row_profile);
    println!("     Column profile:\n{}", col_profile);

    println!();
}
