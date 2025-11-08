//! Basic example of automatic differentiation with einsum operations
//!
//! This example demonstrates:
//! - Computing gradients through einsum contractions
//! - Using VJP (Vector-Jacobian Product) for backward passes
//! - Verifying gradients with finite differences
//!
//! Run with: cargo run --example basic_einsum_ad

use anyhow::Result;
use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig};
use tenrso_ad::vjp::{EinsumVjp, VjpOp};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction;
use tenrso_planner::EinsumSpec;

fn main() -> Result<()> {
    println!("=== Basic Einsum Automatic Differentiation ===\n");

    // Example 1: Matrix multiplication gradient
    println!("Example 1: Matrix Multiplication (C = A @ B)");
    println!("---------------------------------------------");

    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

    println!("Matrix A:\n{:?}", a.as_array());
    println!("\nMatrix B:\n{:?}", b.as_array());

    // Forward pass: C = A @ B
    let spec = EinsumSpec::parse("ij,jk->ik")?;
    let c = execute_dense_contraction(&spec, &a, &b)?;

    println!("\nResult C = A @ B:\n{:?}", c.as_array());

    // Backward pass: compute gradients
    let grad_c = DenseND::ones(c.shape()); // Assume dL/dC = 1 everywhere
    let vjp_ctx = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
    let grads = vjp_ctx.vjp(&grad_c)?;

    let grad_a = &grads[0];
    let grad_b = &grads[1];

    println!("\nGradient ∂L/∂A:\n{:?}", grad_a.as_array());
    println!("\nGradient ∂L/∂B:\n{:?}", grad_b.as_array());

    // Verify gradients with finite differences
    println!("\nVerifying gradient ∂L/∂A with finite differences...");
    let f = |x: &DenseND<f64>| execute_dense_contraction(&spec, x, &b);
    let df = |_x: &DenseND<f64>, grad_y: &DenseND<f64>| {
        let vjp = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
        let grads = vjp.vjp(grad_y)?;
        Ok(grads[0].clone())
    };

    let config = GradCheckConfig {
        epsilon: 1e-5,
        rtol: 1e-3,
        atol: 1e-5,
        use_central_diff: true,
        verbose: true,
    };

    let result = check_gradient(f, df, &a, &grad_c, &config)?;
    if result.passed {
        println!("✓ Gradient check passed!");
    } else {
        println!("✗ Gradient check failed!");
    }

    // Example 2: Inner product (dot product)
    println!("\n\nExample 2: Inner Product (Dot Product)");
    println!("---------------------------------------");

    let vec_a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let vec_b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4])?;

    println!("Vector A: {:?}", vec_a.as_array());
    println!("Vector B: {:?}", vec_b.as_array());

    // Inner product: scalar = sum_i a[i] * b[i]
    let inner_spec = EinsumSpec::parse("i,i->")?;
    let inner_result = execute_dense_contraction(&inner_spec, &vec_a, &vec_b)?;

    println!("\nInner product result: {:?}", inner_result.as_array());

    // Compute gradients
    let grad_inner = DenseND::from_elem(&[], 1.0); // Gradient w.r.t. scalar output
    let inner_vjp = EinsumVjp::new(inner_spec, vec_a.clone(), vec_b.clone());
    let inner_grads = inner_vjp.vjp(&grad_inner)?;

    println!("\nGradient ∂L/∂A: {:?}", inner_grads[0].as_array());
    println!("Gradient ∂L/∂B: {:?}", inner_grads[1].as_array());

    // Example 3: Outer product
    println!("\n\nExample 3: Outer Product");
    println!("-------------------------");

    let vec1 = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
    let vec2 = DenseND::from_vec(vec![4.0, 5.0], &[2])?;

    println!("Vector 1: {:?}", vec1.as_array());
    println!("Vector 2: {:?}", vec2.as_array());

    // Outer product: C[i,j] = vec1[i] * vec2[j]
    let outer_spec = EinsumSpec::parse("i,j->ij")?;
    let outer_result = execute_dense_contraction(&outer_spec, &vec1, &vec2)?;

    println!("\nOuter product result:\n{:?}", outer_result.as_array());

    // Compute gradients
    let grad_outer = DenseND::ones(outer_result.shape());
    let outer_vjp = EinsumVjp::new(outer_spec, vec1.clone(), vec2.clone());
    let outer_grads = outer_vjp.vjp(&grad_outer)?;

    println!("\nGradient ∂L/∂vec1: {:?}", outer_grads[0].as_array());
    println!("Gradient ∂L/∂vec2: {:?}", outer_grads[1].as_array());

    println!("\n=== All examples completed successfully! ===");

    Ok(())
}
