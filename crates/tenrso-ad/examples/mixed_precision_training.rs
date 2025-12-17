//! Demonstration of mixed precision training with automatic loss scaling.
//!
//! This example shows how to use GradScaler for training with reduced precision
//! while maintaining numerical stability.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array, Array1, Array2, Ix1};
use tenrso_ad::mixed_precision::{
    analyze_gradients, convert_to_lower_precision, detect_underflow_risk, GradScaler,
    MixedPrecisionConfig, PrecisionType,
};

fn main() -> Result<()> {
    println!("=== Mixed Precision Training Examples ===\n");

    example_1_basic_grad_scaler()?;
    example_2_dynamic_loss_scaling()?;
    example_3_precision_analysis()?;
    example_4_full_training_loop()?;

    Ok(())
}

/// Example 1: Basic gradient scaler usage
fn example_1_basic_grad_scaler() -> Result<()> {
    println!("Example 1: Basic Gradient Scaler");
    println!("---------------------------------");

    // Create a gradient scaler with default configuration
    let config = MixedPrecisionConfig::default();
    let mut scaler = GradScaler::new(config);

    println!("Initial scale: {}", scaler.get_scale());

    // Simulate a loss value
    let loss = 0.5f32;

    // Scale the loss before backward pass
    let scaled_loss = scaler.scale(loss);
    println!("Loss: {}, Scaled loss: {}", loss, scaled_loss);

    // Simulate gradients (in practice, these come from backward pass)
    let gradients = vec![
        Array::from_vec(vec![1000.0f32, 2000.0, 3000.0]).into_dyn(),
        Array::from_vec(vec![500.0f32, 1500.0]).into_dyn(),
    ];

    // Check for overflow
    let has_overflow = scaler.check_overflow(&gradients);
    println!("Overflow detected: {}", has_overflow);

    // Unscale gradients
    let unscaled = scaler.unscale(&gradients);
    println!("Original gradient[0][0]: {}", gradients[0][[0]]);
    println!("Unscaled gradient[0][0]: {}", unscaled[0][[0]]);

    // Update scaler (no overflow)
    scaler.update(false);
    println!("Scale after update: {}\n", scaler.get_scale());

    Ok(())
}

/// Example 2: Dynamic loss scaling in action
fn example_2_dynamic_loss_scaling() -> Result<()> {
    println!("Example 2: Dynamic Loss Scaling");
    println!("--------------------------------");

    let config = MixedPrecisionConfig {
        init_scale: 1024.0,
        growth_interval: 5,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config);

    println!("Training simulation with dynamic scaling:");

    // Simulate 20 training iterations
    for iter in 0..20 {
        // Simulate overflow every 8 iterations
        let has_overflow = iter % 8 == 7;

        // Update scaler
        scaler.update(has_overflow);

        let stats = scaler.stats();
        println!(
            "Iter {:2}: scale={:8.1}, growth_tracker={}, overflow={}",
            iter + 1,
            stats.current_scale,
            stats.growth_tracker,
            has_overflow
        );
    }

    println!("\nFinal scale: {}\n", scaler.get_scale());
    Ok(())
}

/// Example 3: Gradient analysis for precision selection
fn example_3_precision_analysis() -> Result<()> {
    println!("Example 3: Gradient Analysis");
    println!("----------------------------");

    // Create gradients with different characteristics
    let test_cases = vec![
        (
            "Normal gradients",
            vec![
                Array::from_vec(vec![0.01f32, 0.02, 0.03, 0.04]).into_dyn(),
                Array::from_vec(vec![0.001f32, 0.002, 0.003]).into_dyn(),
            ],
        ),
        (
            "Very small gradients (underflow risk)",
            vec![
                Array::from_vec(vec![1e-9f32, 2e-9, 3e-9, 4e-9]).into_dyn(),
                Array::from_vec(vec![1e-10f32, 2e-10, 3e-10]).into_dyn(),
            ],
        ),
        (
            "Large gradients (overflow risk)",
            vec![
                Array::from_vec(vec![1e5f32, 2e5, 3e5]).into_dyn(),
                Array::from_vec(vec![5e4f32, 1e5]).into_dyn(),
            ],
        ),
    ];

    for (name, gradients) in test_cases {
        println!("\n{}:", name);

        // Analyze for FP16
        let analysis_fp16 = analyze_gradients(&gradients, PrecisionType::FP16)?;
        println!("  FP16 Analysis:");
        println!("    Underflow: {:.2}%", analysis_fp16.underflow_percentage);
        println!("    Overflow:  {:.2}%", analysis_fp16.overflow_percentage);
        println!(
            "    Recommended precision: {:?}",
            analysis_fp16.recommended_precision
        );
        println!(
            "    Recommended scale: {:.1}",
            analysis_fp16.recommended_scale
        );

        // Check underflow risk
        let fp16_risk = detect_underflow_risk(&gradients, PrecisionType::FP16);
        let bf16_risk = detect_underflow_risk(&gradients, PrecisionType::BF16);
        println!("    FP16 underflow risk: {}", fp16_risk);
        println!("    BF16 underflow risk: {}", bf16_risk);
    }

    println!();
    Ok(())
}

/// Example 4: Full training loop with mixed precision
fn example_4_full_training_loop() -> Result<()> {
    println!("Example 4: Full Training Loop");
    println!("-----------------------------");

    // Setup
    let config = MixedPrecisionConfig::fp16_optimized();
    let mut scaler = GradScaler::new(config);

    // Simulate model parameters (weights and biases)
    let mut weights = Array2::from_shape_vec((3, 2), vec![0.5, 0.3, 0.2, 0.4, 0.1, 0.6])?;
    let mut bias = Array1::from_vec(vec![0.1, 0.2, 0.3]);

    let learning_rate = 0.01f32;

    println!("Initial weights:\n{:?}\n", weights);
    println!("Training for 10 iterations with FP16 precision:\n");

    for iter in 0..10 {
        // Simulate forward pass
        let loss = 1.0 / (iter as f32 + 1.0); // Decreasing loss

        // Scale loss for backward pass
        let _scaled_loss = scaler.scale(loss);

        // Simulate gradients (in practice, from backward pass)
        // Gradients get larger as loss decreases (simulating gradient explosion)
        let grad_scale = if iter > 5 { 100000.0 } else { 1.0 };
        let grad_weights = Array::from_vec(vec![
            0.1 * grad_scale,
            -0.05 * grad_scale,
            0.08 * grad_scale,
            -0.03 * grad_scale,
            0.06 * grad_scale,
            -0.02 * grad_scale,
        ])
        .into_dyn();
        let grad_bias = Array::from_vec(vec![
            0.01 * grad_scale,
            -0.02 * grad_scale,
            0.015 * grad_scale,
        ])
        .into_dyn();

        let gradients = vec![grad_weights.clone(), grad_bias.clone()];

        // Check for overflow
        let has_overflow = scaler.check_overflow(&gradients);

        if has_overflow {
            println!(
                "Iter {}: OVERFLOW DETECTED! Skipping update, reducing scale",
                iter + 1
            );
            scaler.update(true);
            continue;
        }

        // Unscale gradients
        let unscaled = scaler.unscale(&gradients);

        // Convert to lower precision (simulate mixed precision computation)
        let grad_weights_fp16 = convert_to_lower_precision(&unscaled[0], PrecisionType::FP16)?;
        let grad_bias_fp16 = convert_to_lower_precision(&unscaled[1], PrecisionType::FP16)?;

        // Update parameters (simple SGD)
        let grad_w = grad_weights_fp16
            .into_dimensionality::<Ix1>()?
            .to_shape((3, 2))?
            .to_owned();
        let grad_b = grad_bias_fp16.into_dimensionality::<Ix1>()?;

        weights = &weights - &(&grad_w * learning_rate);
        bias = &bias - &(&grad_b * learning_rate);

        // Update scaler
        scaler.update(false);

        let stats = scaler.stats();
        println!(
            "Iter {}: loss={:.4}, scale={:.0}, grad_max={:.2e}",
            iter + 1,
            loss,
            stats.current_scale,
            unscaled[0]
                .iter()
                .map(|&v: &f32| v.abs())
                .fold(0.0f32, f32::max)
        );
    }

    println!("\nFinal weights:\n{:?}", weights);
    println!("Final bias:\n{:?}", bias);
    println!("\nFinal scaler stats:");
    let final_stats = scaler.stats();
    println!("  Scale: {}", final_stats.current_scale);
    println!("  Growth progress: {:.1}%", final_stats.growth_progress());

    Ok(())
}
