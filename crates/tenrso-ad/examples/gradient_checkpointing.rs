//! Gradient Checkpointing Example
//!
//! Demonstrates memory-efficient training using gradient checkpointing.
//! Checkpointing trades computation for memory by recomputing intermediate
//! activations during the backward pass instead of storing them.

use anyhow::Result;
use scirs2_core::ndarray_ext::array;
use std::sync::Arc;
use tenrso_ad::checkpoint::{
    CheckpointConfig, CheckpointStrategy, CheckpointedSequence, ElementwiseOp,
};

fn main() -> Result<()> {
    println!("=== Gradient Checkpointing Example ===\n");

    // Example 1: Simple sequential operations with checkpointing
    println!("Example 1: Sequential Operations");
    println!("---------------------------------");

    let config = CheckpointConfig {
        num_segments: 2,
        strategy: CheckpointStrategy::Uniform,
        verify_recomputation: false,
        verification_tolerance: 1e-6,
    };

    let mut sequence = CheckpointedSequence::new(config.clone());

    // Add a sequence of operations: x -> 2x -> x² -> x+1
    sequence.add_operation(Arc::new(ElementwiseOp::new(
        |x: &f64| x * 2.0, // Forward: double
        |_: &f64| 2.0,     // Derivative: 2
    )));

    sequence.add_operation(Arc::new(ElementwiseOp::new(
        |x: &f64| x * x,   // Forward: square
        |x: &f64| 2.0 * x, // Derivative: 2x
    )));

    sequence.add_operation(Arc::new(ElementwiseOp::new(
        |x: &f64| x + 1.0, // Forward: add 1
        |_: &f64| 1.0,     // Derivative: 1
    )));

    sequence.add_operation(Arc::new(ElementwiseOp::new(
        |x: &f64| x * 0.5, // Forward: halve
        |_: &f64| 0.5,     // Derivative: 0.5
    )));

    // Forward pass
    let input = array![1.0, 2.0, 3.0].into_dyn();
    println!("Input: {:?}", input);

    let output = sequence.forward(&input)?;
    println!("Output after 4 operations: {:?}", output);

    // Memory savings analysis
    println!("\n Example 2: Memory Savings Analysis");
    println!("-----------------------------------");

    let stats = sequence.memory_savings_estimate(&[100, 100]);

    println!("Without checkpointing:");
    println!("  Memory usage: {} bytes", stats.base_memory_bytes);

    println!("\nWith checkpointing ({} segments):", config.num_segments);
    println!("  Memory usage: {} bytes", stats.checkpoint_memory_bytes);
    println!("  Memory saved: {} bytes", stats.savings_bytes);
    println!("  Savings ratio: {:.1}%", stats.savings_ratio * 100.0);

    // Example 3: Different checkpointing strategies
    println!("\nExample 3: Checkpointing Strategies");
    println!("-----------------------------------");

    let strategies = vec![
        (CheckpointStrategy::All, "Store all activations"),
        (CheckpointStrategy::Uniform, "Uniform checkpointing"),
        (CheckpointStrategy::None, "No checkpointing"),
    ];

    for (strategy, description) in strategies {
        let config = CheckpointConfig {
            num_segments: 2,
            strategy,
            ..Default::default()
        };

        let mut seq = CheckpointedSequence::<f64>::new(config);

        // Add 10 operations
        for _ in 0..10 {
            seq.add_operation(Arc::new(ElementwiseOp::new(
                |x: &f64| x * 1.1,
                |_: &f64| 1.1,
            )));
        }

        let stats = seq.memory_savings_estimate(&[1000]);

        println!("\nStrategy: {}", description);
        println!("  Base memory: {} KB", stats.base_memory_bytes / 1024);
        println!(
            "  Checkpoint memory: {} KB",
            stats.checkpoint_memory_bytes / 1024
        );
        println!("  Savings: {:.1}%", stats.savings_ratio * 100.0);
    }

    // Example 4: Practical deep network simulation
    println!("\nExample 4: Deep Network Simulation");
    println!("----------------------------------");

    let layer_count = 50;
    let batch_size = 32;
    let hidden_dim = 512;

    println!("\nSimulated Network:");
    println!("  Layers: {}", layer_count);
    println!("  Batch size: {}", batch_size);
    println!("  Hidden dimension: {}", hidden_dim);

    let mut deep_network = CheckpointedSequence::<f64>::new(CheckpointConfig {
        num_segments: 5, // Checkpoint every 10 layers
        strategy: CheckpointStrategy::Uniform,
        ..Default::default()
    });

    for _ in 0..layer_count {
        deep_network.add_operation(Arc::new(ElementwiseOp::new(
            |x: &f64| x.max(0.0), // ReLU activation
            |x: &f64| if *x > 0.0 { 1.0 } else { 0.0 },
        )));
    }

    let stats = deep_network.memory_savings_estimate(&[batch_size, hidden_dim]);

    println!("\nMemory Analysis:");
    println!(
        "  Without checkpointing: {:.2} MB",
        stats.base_memory_bytes as f64 / 1_048_576.0
    );
    println!(
        "  With checkpointing: {:.2} MB",
        stats.checkpoint_memory_bytes as f64 / 1_048_576.0
    );
    println!(
        "  Memory saved: {:.2} MB",
        stats.savings_bytes as f64 / 1_048_576.0
    );
    println!("  Savings ratio: {:.1}%", stats.savings_ratio * 100.0);

    println!("\n=== Gradient Checkpointing Complete ===");

    Ok(())
}
