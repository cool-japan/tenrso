//! Comprehensive examples of iterative linear solvers
//!
//! This example demonstrates:
//! - Conjugate Gradient (CG) for SPD systems
//! - BiCGSTAB for nonsymmetric systems
//! - GMRES for general systems
//! - ILU preconditioning
//! - Convergence analysis

use tenrso_sparse::{
    solvers::{self, IdentityPreconditioner, IluPreconditioner},
    CsrMatrix,
};

fn main() {
    println!("=== Iterative Linear Solvers Examples ===\n");

    example_cg_simple();
    example_bicgstab_nonsymmetric();
    example_gmres_with_restart();
    example_ilu_preconditioning();
    example_convergence_comparison();
}

/// Example 1: Conjugate Gradient for a simple SPD system
fn example_cg_simple() {
    println!("1. Conjugate Gradient (CG) - Simple SPD System");
    println!("   System: [[2, -1], [-1, 2]] * x = [1, 1]");

    let row_ptr = vec![0, 2, 4];
    let col_indices = vec![0, 1, 0, 1];
    let values = vec![2.0, -1.0, -1.0, 2.0];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

    let b = vec![1.0, 1.0];

    let (x, info) = solvers::cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

    println!("   Solution: x = [{:.6}, {:.6}]", x[0], x[1]);
    println!("   {}", info);
    println!("   Expected: x ≈ [1.0, 1.0]\n");
}

/// Example 2: BiCGSTAB for a nonsymmetric system
fn example_bicgstab_nonsymmetric() {
    println!("2. BiCGSTAB - Nonsymmetric System");
    println!("   System: [[3, -1], [-2, 2]] * x = [2, 1]");

    let row_ptr = vec![0, 2, 4];
    let col_indices = vec![0, 1, 0, 1];
    let values = vec![3.0, -1.0, -2.0, 2.0];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

    let b = vec![2.0, 1.0];

    let (x, info) =
        solvers::bicgstab::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

    println!("   Solution: x = [{:.6}, {:.6}]", x[0], x[1]);
    println!("   {}", info);

    // Verify solution
    let residual_0 = (a.row_ptr()[0]..a.row_ptr()[1])
        .map(|i| a.values()[i] * x[a.col_indices()[i]])
        .sum::<f64>()
        - b[0];
    let residual_1 = (a.row_ptr()[1]..a.row_ptr()[2])
        .map(|i| a.values()[i] * x[a.col_indices()[i]])
        .sum::<f64>()
        - b[1];
    println!("   Residual: [{:.2e}, {:.2e}]\n", residual_0, residual_1);
}

/// Example 3: GMRES with restart parameter
fn example_gmres_with_restart() {
    println!("3. GMRES - With Restart");
    println!("   System: [[4, -1, 0], [-1, 4, -1], [0, -1, 4]] * x = [1, 2, 3]");

    let row_ptr = vec![0, 2, 5, 7];
    let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
    let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

    let b = vec![1.0, 2.0, 3.0];

    // GMRES with restart = 10
    let (x, info) =
        solvers::gmres::<f64, IdentityPreconditioner>(&a, &b, 100, 10, 1e-6, None).unwrap();

    println!("   Solution: x = [{:.6}, {:.6}, {:.6}]", x[0], x[1], x[2]);
    println!("   {}", info);
    println!("   Restart parameter: 10\n");
}

/// Example 4: ILU Preconditioning
fn example_ilu_preconditioning() {
    println!("4. ILU(0) Preconditioning");
    println!("   Solving larger SPD system with and without preconditioning");

    // Create a 5x5 tridiagonal-like SPD matrix
    let row_ptr = vec![0, 2, 5, 8, 11, 13];
    let col_indices = vec![
        0, 1, // row 0
        0, 1, 2, // row 1
        1, 2, 3, // row 2
        2, 3, 4, // row 3
        3, 4, // row 4
    ];
    let values = vec![
        4.0, -1.0, // row 0
        -1.0, 4.0, -1.0, // row 1
        -1.0, 4.0, -1.0, // row 2
        -1.0, 4.0, -1.0, // row 3
        -1.0, 4.0, // row 4
    ];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (5, 5)).unwrap();

    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Without preconditioning
    let (x_no_precond, info_no_precond) =
        solvers::cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-8, None).unwrap();

    println!("   Without preconditioning:");
    println!("     {}", info_no_precond);

    // With ILU(0) preconditioning
    let precond = IluPreconditioner::from_matrix(&a).unwrap();
    let (x_precond, info_precond) = solvers::cg(&a, &b, 100, 1e-8, Some(&precond)).unwrap();

    println!("   With ILU(0) preconditioning:");
    println!("     {}", info_precond);

    // Compare iteration counts
    let reduction = (info_no_precond.iterations as f64 - info_precond.iterations as f64)
        / info_no_precond.iterations as f64
        * 100.0;
    println!(
        "   Iteration reduction: {:.1}%",
        reduction.clamp(0.0, 100.0)
    );

    // Verify both solutions are similar
    let diff: f64 = x_no_precond
        .iter()
        .zip(x_precond.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("   Solution difference: {:.2e}\n", diff);
}

/// Example 5: Convergence comparison across solvers
fn example_convergence_comparison() {
    println!("5. Convergence Comparison");
    println!("   Comparing CG, BiCGSTAB, and GMRES on the same SPD system");

    // 4x4 SPD matrix
    let row_ptr = vec![0, 2, 5, 8, 10];
    let col_indices = vec![
        0, 1, // row 0
        0, 1, 2, // row 1
        1, 2, 3, // row 2
        2, 3, // row 3
    ];
    let values = vec![
        5.0, -1.0, // row 0
        -1.0, 5.0, -1.0, // row 1
        -1.0, 5.0, -1.0, // row 2
        -1.0, 5.0, // row 3
    ];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

    let b = vec![1.0, 2.0, 3.0, 4.0];

    // CG (optimal for SPD)
    let (_, info_cg) = solvers::cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-8, None).unwrap();
    println!("   CG:       {}", info_cg);

    // BiCGSTAB
    let (_, info_bicg) =
        solvers::bicgstab::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-8, None).unwrap();
    println!("   BiCGSTAB: {}", info_bicg);

    // GMRES
    let (_, info_gmres) =
        solvers::gmres::<f64, IdentityPreconditioner>(&a, &b, 100, 10, 1e-8, None).unwrap();
    println!("   GMRES:    {}", info_gmres);

    println!("\n   Note: CG is optimal for SPD matrices!");
    println!("   BiCGSTAB and GMRES are more general but may take more iterations.\n");
}
