//! Example of computing gradients through CP decomposition
//!
//! This example demonstrates:
//! - Computing gradients w.r.t. CP factor matrices
//! - Handling weighted CP decompositions
//! - Using MTTKRP for efficient gradient computation
//!
//! Run with: cargo run --example cp_decomposition_gradients

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2};
use tenrso_ad::grad::CpReconstructionGrad;
use tenrso_core::DenseND;

fn main() -> Result<()> {
    println!("=== CP Decomposition Gradients ===\n");

    // Example 1: Simple 3-mode CP decomposition
    println!("Example 1: 3-Mode Tensor (4×5×6) with Rank 3");
    println!("----------------------------------------------");

    let rank = 3;

    // Create random-like factor matrices
    let factors = vec![
        Array2::from_shape_vec(
            (4, rank),
            vec![
                1.0, 0.5, 0.2, // mode 0, component 0-2
                0.8, 0.6, 0.3, // mode 0, component 0-2
                0.5, 0.9, 0.4, // mode 0, component 0-2
                0.3, 0.7, 0.8, // mode 0, component 0-2
            ],
        )?,
        Array2::from_shape_vec(
            (5, rank),
            vec![
                0.9, 0.4, 0.1, // mode 1, component 0-2
                0.7, 0.5, 0.3, // mode 1, component 0-2
                0.6, 0.8, 0.2, // mode 1, component 0-2
                0.4, 0.6, 0.5, // mode 1, component 0-2
                0.2, 0.3, 0.9, // mode 1, component 0-2
            ],
        )?,
        Array2::from_shape_vec(
            (6, rank),
            vec![
                1.0, 0.0, 0.0, // mode 2, component 0-2
                0.0, 1.0, 0.0, // mode 2, component 0-2
                0.0, 0.0, 1.0, // mode 2, component 0-2
                0.5, 0.5, 0.0, // mode 2, component 0-2
                0.0, 0.5, 0.5, // mode 2, component 0-2
                0.5, 0.0, 0.5, // mode 2, component 0-2
            ],
        )?,
    ];

    println!("Factor shapes:");
    for (i, factor) in factors.iter().enumerate() {
        println!("  Factor {}: {:?}", i, factor.shape());
    }

    // Create gradient context
    let grad_ctx = CpReconstructionGrad::new(factors.clone(), None);

    // Assume we have a gradient w.r.t. the reconstructed tensor
    let grad_output = DenseND::ones(&[4, 5, 6]);
    println!("\nGradient output shape: {:?}", grad_output.shape());

    // Compute gradients w.r.t. factor matrices
    println!("\nComputing gradients via MTTKRP...");
    let factor_grads = grad_ctx.compute_factor_gradients(&grad_output)?;

    println!("Factor gradient shapes:");
    for (i, grad) in factor_grads.iter().enumerate() {
        println!("  ∂L/∂Factor_{}: {:?}", i, grad.shape());
    }

    // Display gradient statistics
    println!("\nGradient statistics:");
    for (i, grad) in factor_grads.iter().enumerate() {
        let sum: f64 = grad.iter().copied().sum();
        let mean = sum / (grad.shape()[0] * grad.shape()[1]) as f64;
        let max = grad.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);

        println!(
            "  Factor {}: sum={:.4}, mean={:.4}, max={:.4}",
            i, sum, mean, max
        );
    }

    // Example 2: Weighted CP decomposition
    println!("\n\nExample 2: Weighted CP Decomposition");
    println!("-------------------------------------");

    let weights = Some(Array1::from_vec(vec![2.0, 1.5, 1.0]));
    println!("Component weights: {:?}", weights.as_ref().unwrap());

    let weighted_grad_ctx = CpReconstructionGrad::new(factors.clone(), weights);
    let weighted_grads = weighted_grad_ctx.compute_factor_gradients(&grad_output)?;

    println!("\nWeighted gradient statistics:");
    for (i, grad) in weighted_grads.iter().enumerate() {
        let sum: f64 = grad.iter().copied().sum();
        let mean = sum / (grad.shape()[0] * grad.shape()[1]) as f64;

        println!("  Factor {}: sum={:.4}, mean={:.4}", i, sum, mean);
    }

    // Example 3: Higher-order tensor (4-mode)
    println!("\n\nExample 3: 4-Mode Tensor (3×4×5×2) with Rank 2");
    println!("-----------------------------------------------");

    let rank4 = 2;
    let factors_4mode = vec![
        Array2::from_shape_vec((3, rank4), vec![1.0, 0.5, 0.8, 0.3, 0.6, 0.9])?,
        Array2::from_shape_vec((4, rank4), vec![0.7, 0.4, 0.5, 0.6, 0.8, 0.3, 0.4, 0.7])?,
        Array2::from_shape_vec(
            (5, rank4),
            vec![0.9, 0.2, 0.6, 0.5, 0.3, 0.8, 0.7, 0.4, 0.5, 0.6],
        )?,
        Array2::from_shape_vec((2, rank4), vec![1.0, 0.0, 0.0, 1.0])?,
    ];

    let grad_ctx_4mode = CpReconstructionGrad::new(factors_4mode.clone(), None);
    let grad_output_4mode = DenseND::ones(&[3, 4, 5, 2]);

    println!("Computing gradients for 4-mode tensor...");
    let factor_grads_4mode = grad_ctx_4mode.compute_factor_gradients(&grad_output_4mode)?;

    println!("\n4-mode factor gradient shapes:");
    for (i, grad) in factor_grads_4mode.iter().enumerate() {
        println!("  ∂L/∂Factor_{}: {:?}", i, grad.shape());
    }

    // Display gradient norms
    println!("\nGradient Frobenius norms:");
    for (i, grad) in factor_grads_4mode.iter().enumerate() {
        let norm_sq: f64 = grad.iter().map(|x: &f64| x * x).sum();
        let norm = norm_sq.sqrt();
        println!("  ||∂L/∂Factor_{}||_F = {:.4}", i, norm);
    }

    println!("\n=== All examples completed successfully! ===");

    Ok(())
}
