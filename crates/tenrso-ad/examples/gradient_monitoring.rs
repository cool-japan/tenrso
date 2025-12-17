//! Comprehensive demonstration of gradient monitoring and analysis.
//!
//! This example shows how to use the gradient monitoring system to track,
//! analyze, and debug gradient flow in neural network training.

use anyhow::Result;
use scirs2_core::ndarray_ext::Array;
use tenrso_ad::monitoring::{GradientMonitor, HealthStatus, MonitorConfig};

fn main() -> Result<()> {
    println!("=== Gradient Monitoring Examples ===\n");

    example_1_basic_monitoring()?;
    example_2_vanishing_gradients()?;
    example_3_exploding_gradients()?;
    example_4_gradient_flow_analysis()?;
    example_5_training_simulation()?;

    Ok(())
}

/// Example 1: Basic gradient monitoring
fn example_1_basic_monitoring() -> Result<()> {
    println!("Example 1: Basic Gradient Monitoring");
    println!("------------------------------------");

    let config = MonitorConfig::default();
    let mut monitor = GradientMonitor::new(config);

    // Simulate gradients for multiple layers
    let layer1_grad = Array::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]).into_dyn();
    let layer2_grad = Array::from_vec(vec![0.05, 0.15, 0.25, 0.35]).into_dyn();
    let layer3_grad = Array::from_vec(vec![0.01, 0.02, 0.03]).into_dyn();

    // Record gradients
    monitor.record_step("layer1", &layer1_grad);
    monitor.record_step("layer2", &layer2_grad);
    monitor.record_step("layer3", &layer3_grad);

    // Get layer statistics
    println!("\nLayer Statistics:");
    for (name, stats) in monitor.get_current_stats() {
        println!(
            "  {}: L2={:.6}, mean={:.6}, std={:.6}",
            name, stats.l2_norm, stats.mean, stats.std
        );
    }

    // Analyze health
    let health = monitor.analyze_health();
    println!("\nGradient Health:");
    println!("  Status: {:?}", health.status);
    println!("  Severity: {:.2}", health.severity);
    println!("  Healthy: {}", health.is_healthy());

    monitor.step();
    println!();

    Ok(())
}

/// Example 2: Detecting vanishing gradients
fn example_2_vanishing_gradients() -> Result<()> {
    println!("Example 2: Vanishing Gradient Detection");
    println!("----------------------------------------");

    let config = MonitorConfig::default();
    let mut monitor = GradientMonitor::new(config);

    // Simulate a deep network with vanishing gradients
    println!("\nSimulating 5-layer network with vanishing gradients:");

    for layer_idx in 0..5 {
        // Gradients get exponentially smaller in deeper layers
        let scale = 0.1_f32.powi(layer_idx + 1);
        let gradient = Array::from_vec(vec![scale, scale * 1.5, scale * 2.0]).into_dyn();

        let layer_name = format!("layer{}", layer_idx + 1);
        monitor.record_step(&layer_name, &gradient);

        if let Some(stats) = monitor.get_layer_stats(&layer_name) {
            println!(
                "  {}: L2={:.2e}, vanishing={}",
                layer_name,
                stats.l2_norm,
                stats.has_vanishing(1e-7)
            );
        }
    }

    // Analyze health
    let health = monitor.analyze_health();
    println!("\nHealth Analysis:");
    println!("  Status: {:?}", health.status);
    println!("  Vanishing layers: {:?}", health.vanishing_layers);
    println!("  Severity: {:.2}", health.severity);

    // Get recommendations
    println!("\nRecommendations:");
    for (i, rec) in monitor.get_recommendations().iter().enumerate() {
        println!("  {}. {}", i + 1, rec);
    }

    monitor.step();
    println!();

    Ok(())
}

/// Example 3: Detecting exploding gradients
fn example_3_exploding_gradients() -> Result<()> {
    println!("Example 3: Exploding Gradient Detection");
    println!("----------------------------------------");

    let config = MonitorConfig {
        exploding_threshold: 5.0, // Lower threshold for demonstration
        ..Default::default()
    };
    let mut monitor = GradientMonitor::new(config);

    // Simulate gradients that explode
    println!("\nSimulating unstable training with exploding gradients:");

    for layer_idx in 0..3 {
        // Gradients grow exponentially
        let scale = 2.0_f32.powi(layer_idx + 1);
        let gradient = Array::from_vec(vec![
            scale,
            scale * 1.2,
            scale * 0.8,
            scale * 1.5,
            scale * 0.9,
        ])
        .into_dyn();

        let layer_name = format!("layer{}", layer_idx + 1);
        monitor.record_step(&layer_name, &gradient);

        if let Some(stats) = monitor.get_layer_stats(&layer_name) {
            println!(
                "  {}: L2={:.2e}, exploding={}",
                layer_name,
                stats.l2_norm,
                stats.has_exploding(5.0)
            );
        }
    }

    // Analyze health
    let health = monitor.analyze_health();
    println!("\nHealth Analysis:");
    println!("  Status: {:?}", health.status);
    println!("  Exploding layers: {:?}", health.exploding_layers);

    if health.status == HealthStatus::Critical {
        println!("\n⚠️  CRITICAL: Immediate action required!");
    }

    // Get recommendations
    println!("\nRecommendations:");
    for (i, rec) in monitor.get_recommendations().iter().enumerate() {
        println!("  {}. {}", i + 1, rec);
    }

    monitor.step();
    println!();

    Ok(())
}

/// Example 4: Gradient flow analysis
fn example_4_gradient_flow_analysis() -> Result<()> {
    println!("Example 4: Gradient Flow Analysis");
    println!("----------------------------------");

    let config = MonitorConfig::default();
    let mut monitor = GradientMonitor::new(config);

    // Simulate unbalanced gradient flow
    println!("\nSimulating network with unbalanced gradient flow:");

    let gradients_data = vec![
        ("input_layer", vec![1.0, 1.2, 0.9, 1.1, 1.0]),
        ("hidden1", vec![0.5, 0.6, 0.4, 0.55, 0.5]),
        ("hidden2", vec![0.1, 0.12, 0.08, 0.11, 0.1]), // Bottleneck
        ("hidden3", vec![0.05, 0.06, 0.04, 0.055, 0.05]),
        ("output_layer", vec![0.02, 0.025, 0.015, 0.022, 0.02]),
    ];

    for (layer_name, grad_values) in gradients_data {
        let gradient = Array::from_vec(
            grad_values
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<f32>>(),
        )
        .into_dyn();
        monitor.record_step(layer_name, &gradient);
    }

    // Analyze flow
    let flow = monitor.analyze_flow()?;
    println!("\nGradient Flow Analysis:");
    println!("  Flow ratio (first/last): {:.4}", flow.flow_ratio);
    println!("  Is balanced: {}", flow.is_balanced);

    if !flow.bottlenecks.is_empty() {
        println!("\n  Bottleneck layers detected:");
        for bottleneck in &flow.bottlenecks {
            println!("    - {}", bottleneck);
        }
    }

    println!("\n  Layer-wise magnitudes:");
    for (name, magnitude) in &flow.layer_magnitudes {
        println!("    {}: {:.6}", name, magnitude);
    }

    monitor.step();
    println!();

    Ok(())
}

/// Example 5: Full training simulation with monitoring
fn example_5_training_simulation() -> Result<()> {
    println!("Example 5: Full Training Simulation");
    println!("------------------------------------");

    let config = MonitorConfig {
        window_size: 5,
        ..Default::default()
    };
    let mut monitor = GradientMonitor::new(config);

    println!("\nSimulating 20 training steps with evolving gradients:");
    println!();

    for step in 0..20 {
        // Simulate gradients that change over training
        // Early training: larger gradients
        // Later training: gradients stabilize and shrink

        let base_scale = if step < 5 {
            1.0 // Initial large gradients
        } else if step < 15 {
            0.1 // Stabilizing
        } else {
            0.01 // Converging
        };

        // Add some noise
        let noise = (step as f32 * 0.1).sin() * 0.1;

        let layer1_grad = Array::from_vec(vec![
            base_scale + noise,
            base_scale * 1.2 + noise,
            base_scale * 0.9 + noise,
        ])
        .into_dyn();

        let layer2_grad = Array::from_vec(vec![
            base_scale * 0.8 + noise,
            base_scale * 0.85 + noise,
            base_scale * 0.9 + noise,
        ])
        .into_dyn();

        monitor.record_step("layer1", &layer1_grad);
        monitor.record_step("layer2", &layer2_grad);

        // Print periodic summaries
        if (step + 1) % 5 == 0 {
            println!("Step {}:", step + 1);

            for layer in &["layer1", "layer2"] {
                if let Some(stats) = monitor.get_layer_stats(layer) {
                    let avg = monitor.get_moving_average(layer).unwrap_or(0.0);
                    let trend = monitor.get_layer_trend(layer);

                    println!(
                        "  {}: L2={:.6}, moving_avg={:.6}, trend={:?}",
                        layer, stats.l2_norm, avg, trend
                    );
                }
            }

            let health = monitor.analyze_health();
            println!(
                "  Health: {:?} (severity={:.2})",
                health.status, health.severity
            );
            println!();
        }

        monitor.step();
    }

    // Final summary
    println!("=== Final Training Summary ===");
    println!("{}", monitor.summary());

    Ok(())
}
