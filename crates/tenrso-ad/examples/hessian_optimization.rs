//! Hessian-Based Optimization Example
//!
//! Demonstrates second-order optimization using Hessian information.
//! Hessian-based methods can converge faster than gradient descent
//! by using curvature information.

use anyhow::Result;
use scirs2_core::ndarray_ext::array;
use tenrso_ad::hessian::{
    compute_hessian_diagonal, hessian_vector_product, HessianFn, QuadraticFn,
};

fn main() -> Result<()> {
    println!("=== Hessian-Based Optimization Example ===\n");

    // Example 1: Quadratic optimization with exact Hessian
    println!("Example 1: Quadratic Function Optimization");
    println!("------------------------------------------");

    // Minimize f(x) = (1/2) x^T A x where A = [[2, 0], [0, 4]]
    // Optimal point: x* = [0, 0]
    let a = array![[2.0, 0.0], [0.0, 4.0]];
    let f = QuadraticFn::new(a.clone());

    let mut x = array![10.0, 10.0].into_dyn();
    println!("Initial point: {:?}", x);
    println!("Initial value: {:.6}", f.eval(&x)?);

    // Newton's method using diagonal Hessian (simplified)
    let learning_rate = 1.0;
    let epsilon = 1e-5;

    for iter in 0..5 {
        let grad = f.grad(&x)?;
        let hess_diag = compute_hessian_diagonal(&f, &x, epsilon)?;

        // Newton update: x_{k+1} = x_k - H^{-1} g
        // Using diagonal approximation: x_{k+1} = x_k - g / diag(H)
        let grad_slice = grad.as_slice().unwrap();
        let hess_slice = hess_diag.as_slice().unwrap();
        let x_slice = x.as_slice_mut().unwrap();

        for i in 0..x_slice.len() {
            let hess_val: f64 = hess_slice[i];
            if hess_val.abs() > 1e-10 {
                x_slice[i] -= learning_rate * grad_slice[i] / hess_val;
            }
        }

        println!("Iter {}: x = {:?}, f(x) = {:.6}", iter + 1, x, f.eval(&x)?);
    }

    // Example 2: Hessian-vector products
    println!("\nExample 2: Hessian-Vector Products");
    println!("----------------------------------");

    let a = array![[2.0, 1.0], [1.0, 2.0]];
    let f = QuadraticFn::new(a.clone());

    let x = array![0.0, 0.0].into_dyn();
    let v = array![1.0, 0.0].into_dyn();
    let epsilon = 1e-5;

    let hv = hessian_vector_product(&f, &x, &v, epsilon)?;

    println!("Function: f(x) = (1/2) x^T A x");
    println!("Matrix A:");
    println!("  [{:.1}, {:.1}]", a[[0, 0]], a[[0, 1]]);
    println!("  [{:.1}, {:.1}]", a[[1, 0]], a[[1, 1]]);
    println!("\nHessian H = A (for quadratic functions)");
    println!("Direction vector v = {:?}", v);
    println!("Hessian-vector product H*v ≈ {:?}", hv);
    println!("Expected: [2.0, 1.0]");

    // Example 3: Diagonal Hessian for preconditioning
    println!("\nExample 3: Diagonal Hessian Preconditioning");
    println!("------------------------------------------");

    let a = array![[4.0, 0.0], [0.0, 1.0]];
    let f = QuadraticFn::new(a.clone());

    let x = array![1.0, 1.0].into_dyn();
    let epsilon = 1e-5;

    let diag = compute_hessian_diagonal(&f, &x, epsilon)?;

    println!("Function with ill-conditioned Hessian:");
    println!("Matrix A:");
    println!("  [{:.1}, {:.1}]", a[[0, 0]], a[[0, 1]]);
    println!("  [{:.1}, {:.1}]", a[[1, 0]], a[[1, 1]]);
    println!("\nDiagonal of Hessian: {:?}", diag);
    println!("Condition number (ratio): {:.1}", 4.0 / 1.0);
    println!("\nPreconditioner P = diag(H)^{{-1}} can improve convergence");

    // Example 4: Comparison of optimization methods
    println!("\nExample 4: Gradient Descent vs. Newton's Method");
    println!("----------------------------------------------");

    let a = array![[10.0, 0.0], [0.0, 1.0]];
    let f = QuadraticFn::new(a.clone());

    // Gradient descent
    println!("\nGradient Descent (learning rate = 0.1):");
    let mut x_gd = array![10.0, 10.0].into_dyn();
    let lr = 0.1;

    for iter in 0..5 {
        let grad = f.grad(&x_gd)?;
        let grad_slice = grad.as_slice().unwrap();
        let x_slice = x_gd.as_slice_mut().unwrap();

        for i in 0..x_slice.len() {
            x_slice[i] -= lr * grad_slice[i];
        }

        println!("  Iter {}: f(x) = {:.6}", iter + 1, f.eval(&x_gd)?);
    }

    // Newton's method (diagonal approximation)
    println!("\nNewton's Method (diagonal Hessian):");
    let mut x_newton = array![10.0, 10.0].into_dyn();
    let epsilon = 1e-5;

    for iter in 0..5 {
        let grad = f.grad(&x_newton)?;
        let hess_diag = compute_hessian_diagonal(&f, &x_newton, epsilon)?;

        let grad_slice = grad.as_slice().unwrap();
        let hess_slice = hess_diag.as_slice().unwrap();
        let x_slice = x_newton.as_slice_mut().unwrap();

        for i in 0..x_slice.len() {
            let hess_val: f64 = hess_slice[i];
            if hess_val.abs() > 1e-10 {
                x_slice[i] -= grad_slice[i] / hess_val;
            }
        }

        println!("  Iter {}: f(x) = {:.6}", iter + 1, f.eval(&x_newton)?);
    }

    println!("\nConclusion: Newton's method converges much faster");
    println!("for this ill-conditioned problem (condition number = 10)");

    println!("\n=== Hessian Optimization Complete ===");

    Ok(())
}
