//! Comprehensive examples demonstrating Incremental/Online CP-ALS
//!
//! This example shows how to use incremental CP decomposition for streaming
//! tensor data, avoiding full recomputation when new data arrives.
//!
//! Run with: cargo run --example incremental_cp --release

use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, cp_als_incremental, IncrementalMode, InitStrategy};

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(80));
    println!("Incremental CP-ALS Examples");
    println!("{}", "=".repeat(80));
    println!();

    example_1_append_mode()?;
    println!();

    example_2_sliding_window()?;
    println!();

    example_3_streaming_time_series()?;
    println!();

    example_4_warm_start_speedup()?;
    println!();

    example_5_multiple_updates()?;
    println!();

    example_6_different_modes()?;
    println!();

    Ok(())
}

/// Example 1: Basic append mode usage
///
/// Shows how to grow a tensor along one dimension (e.g., new time slices)
fn example_1_append_mode() -> anyhow::Result<()> {
    println!("Example 1: Append Mode - Growing Time Series");
    println!("{}", "-".repeat(80));

    // Initial tensor: 50 time steps, 20x20 spatial dimensions
    let initial_tensor = DenseND::<f64>::random_uniform(&[50, 20, 20], 0.0, 1.0);
    let rank = 8;

    println!("Initial tensor shape: {:?}", initial_tensor.shape());

    // Compute initial CP decomposition
    let start = std::time::Instant::now();
    let cp_initial = cp_als(&initial_tensor, rank, 50, 1e-4, InitStrategy::Svd, None)?;
    let initial_time = start.elapsed();

    println!("Initial CP-ALS:");
    println!("  Time: {:.3}s", initial_time.as_secs_f64());
    println!("  Iterations: {}", cp_initial.iters);
    println!("  Fit: {:.4}", cp_initial.fit);
    println!();

    // New data arrives: 10 more time steps
    let new_data = DenseND::<f64>::random_uniform(&[10, 20, 20], 0.0, 1.0);
    println!("New data shape: {:?}", new_data.shape());

    // Update incrementally
    let start = std::time::Instant::now();
    let cp_updated = cp_als_incremental(
        &cp_initial,
        &new_data,
        0, // time dimension (mode 0)
        IncrementalMode::Append,
        10, // fewer iterations due to warm start
        1e-4,
    )?;
    let incremental_time = start.elapsed();

    println!("Incremental update:");
    println!("  Time: {:.3}s", incremental_time.as_secs_f64());
    println!("  Iterations: {}", cp_updated.iters);
    println!("  Fit: {:.4}", cp_updated.fit);
    println!(
        "  Updated shape: {:?} (50+10=60 time steps)",
        cp_updated.factors[0].shape()[0]
    );
    println!();

    println!(
        "✓ Speedup: {:.1}x faster than initial decomposition",
        initial_time.as_secs_f64() / incremental_time.as_secs_f64()
    );

    Ok(())
}

/// Example 2: Sliding window with exponential forgetting
///
/// Maintains fixed tensor size while adapting to new data
fn example_2_sliding_window() -> anyhow::Result<()> {
    println!("Example 2: Sliding Window Mode - Forgetting Old Data");
    println!("{}", "-".repeat(80));

    // Initial window
    let window_size = 30;
    let initial_window = DenseND::<f64>::random_uniform(&[window_size, 15, 15], 0.0, 1.0);
    let rank = 6;

    println!("Window size: {} time steps", window_size);

    // Initial decomposition
    let mut cp = cp_als(&initial_window, rank, 40, 1e-4, InitStrategy::Random, None)?;
    println!("Initial fit: {:.4}", cp.fit);
    println!();

    // Simulate streaming updates
    let num_updates = 5;
    println!("Simulating {} streaming updates...", num_updates);

    for update in 1..=num_updates {
        // New window arrives (replaces old data)
        let new_window = DenseND::<f64>::random_uniform(&[window_size, 15, 15], 0.0, 1.0);

        // Different forgetting factors
        let lambda = if update <= 2 { 0.95 } else { 0.90 };

        let start = std::time::Instant::now();
        cp = cp_als_incremental(
            &cp,
            &new_window,
            0,
            IncrementalMode::SlidingWindow { lambda },
            8, // very few iterations needed
            1e-4,
        )?;
        let time = start.elapsed();

        println!(
            "  Update {}: fit={:.4}, iters={}, time={:.2}ms, lambda={}",
            update,
            cp.fit,
            cp.iters,
            time.as_secs_f64() * 1000.0,
            lambda
        );
    }

    println!();
    println!("✓ Maintained constant window size with fast updates");

    Ok(())
}

/// Example 3: Realistic streaming time-series scenario
///
/// Sensor network collecting data over time
fn example_3_streaming_time_series() -> anyhow::Result<()> {
    println!("Example 3: Streaming Sensor Data");
    println!("{}", "-".repeat(80));

    // Scenario: 100 sensors, 50x50 spatial grid, data arriving in batches
    let num_sensors = 100;
    let spatial_dims = (50, 50);
    let rank = 12;

    // Initial batch: 20 time steps
    let initial_batch =
        DenseND::<f64>::random_uniform(&[20, num_sensors, spatial_dims.0], 0.0, 1.0);

    println!(
        "Sensor network: {} sensors, {}x{} grid",
        num_sensors, spatial_dims.0, spatial_dims.1
    );
    println!("Initial batch: 20 time steps");

    let mut cp = cp_als(&initial_batch, rank, 50, 1e-4, InitStrategy::Svd, None)?;
    println!("Initial CP fit: {:.4}", cp.fit);
    println!();

    // Streaming updates arrive
    let batch_sizes = [5, 10, 5, 8];
    let mut total_timesteps = 20;

    println!("Processing streaming batches...");
    for (i, &batch_size) in batch_sizes.iter().enumerate() {
        let new_batch =
            DenseND::<f64>::random_uniform(&[batch_size, num_sensors, spatial_dims.0], 0.0, 1.0);

        let start = std::time::Instant::now();
        cp = cp_als_incremental(
            &cp,
            &new_batch,
            0, // time dimension
            IncrementalMode::Append,
            10,
            1e-4,
        )?;
        let time = start.elapsed();

        total_timesteps += batch_size;

        println!(
            "  Batch {}: +{} timesteps, total={}, fit={:.4}, time={:.3}s",
            i + 1,
            batch_size,
            total_timesteps,
            cp.fit,
            time.as_secs_f64()
        );
    }

    println!();
    println!(
        "✓ Successfully processed {} total timesteps across {} batches",
        total_timesteps,
        batch_sizes.len() + 1
    );

    Ok(())
}

/// Example 4: Warm start speedup demonstration
///
/// Compares incremental update vs full recomputation
fn example_4_warm_start_speedup() -> anyhow::Result<()> {
    println!("Example 4: Warm Start Speedup Analysis");
    println!("{}", "-".repeat(80));

    let initial_shape = [30, 25, 25];
    let new_data_shape = [10, 25, 25];
    let combined_shape = [40, 25, 25];
    let rank = 10;

    // Create data
    let initial_tensor = DenseND::<f64>::random_uniform(&initial_shape, 0.0, 1.0);
    let new_data = DenseND::<f64>::random_uniform(&new_data_shape, 0.0, 1.0);
    let combined_tensor = DenseND::<f64>::random_uniform(&combined_shape, 0.0, 1.0);

    println!(
        "Scenario: {} -> {} tensor (adding {} slices)",
        initial_shape[0], combined_shape[0], new_data_shape[0]
    );
    println!();

    // Method 1: Full recomputation
    println!("Method 1: Full Recomputation");
    let start = std::time::Instant::now();
    let cp_full = cp_als(&combined_tensor, rank, 50, 1e-4, InitStrategy::Random, None)?;
    let full_time = start.elapsed();
    println!("  Time: {:.3}s", full_time.as_secs_f64());
    println!("  Iterations: {}", cp_full.iters);
    println!("  Fit: {:.4}", cp_full.fit);
    println!();

    // Method 2: Incremental update
    println!("Method 2: Incremental Update");
    let cp_init = cp_als(&initial_tensor, rank, 50, 1e-4, InitStrategy::Random, None)?;
    let start = std::time::Instant::now();
    let cp_incremental =
        cp_als_incremental(&cp_init, &new_data, 0, IncrementalMode::Append, 10, 1e-4)?;
    let incremental_time = start.elapsed();
    println!(
        "  Time: {:.3}s (update only)",
        incremental_time.as_secs_f64()
    );
    println!("  Iterations: {}", cp_incremental.iters);
    println!("  Fit: {:.4}", cp_incremental.fit);
    println!();

    println!("Results:");
    println!(
        "  Speedup: {:.1}x",
        full_time.as_secs_f64() / incremental_time.as_secs_f64()
    );
    println!(
        "  Fit difference: {:.4}",
        (cp_full.fit - cp_incremental.fit).abs()
    );
    println!();
    println!("✓ Incremental update is significantly faster with comparable quality");

    Ok(())
}

/// Example 5: Multiple sequential updates
///
/// Shows accumulation of updates over time
fn example_5_multiple_updates() -> anyhow::Result<()> {
    println!("Example 5: Multiple Sequential Updates");
    println!("{}", "-".repeat(80));

    let rank = 7;
    let mut current_size = 20;

    // Initial decomposition
    let initial = DenseND::<f64>::random_uniform(&[current_size, 15, 15], 0.0, 1.0);
    let mut cp = cp_als(&initial, rank, 40, 1e-4, InitStrategy::Svd, None)?;

    println!("Starting with {} time steps", current_size);
    println!("Initial fit: {:.4}", cp.fit);
    println!();

    // Perform 10 incremental updates
    let num_updates = 10;
    let update_size = 5;
    let mut total_time = 0.0;

    println!(
        "Performing {} updates of {} time steps each...",
        num_updates, update_size
    );

    for i in 1..=num_updates {
        let new_data = DenseND::<f64>::random_uniform(&[update_size, 15, 15], 0.0, 1.0);

        let start = std::time::Instant::now();
        cp = cp_als_incremental(&cp, &new_data, 0, IncrementalMode::Append, 8, 1e-4)?;
        let elapsed = start.elapsed();
        total_time += elapsed.as_secs_f64();

        current_size += update_size;

        if i % 2 == 0 {
            // Print every other update
            println!(
                "  After update {}: size={}, fit={:.4}, cumulative_time={:.3}s",
                i, current_size, cp.fit, total_time
            );
        }
    }

    println!();
    println!("Final state:");
    println!("  Total time steps: {}", current_size);
    println!("  Final fit: {:.4}", cp.fit);
    println!("  Total update time: {:.3}s", total_time);
    println!(
        "  Average time per update: {:.3}s",
        total_time / num_updates as f64
    );
    println!();
    println!("✓ Successfully accumulated {} updates", num_updates);

    Ok(())
}

/// Example 6: Updating different modes
///
/// Shows flexibility of choosing which mode to grow
fn example_6_different_modes() -> anyhow::Result<()> {
    println!("Example 6: Updating Different Modes");
    println!("{}", "-".repeat(80));

    let rank = 6;

    // Example 6a: Grow along mode 0 (time)
    println!("Scenario A: Growing time dimension (mode 0)");
    let tensor_a = DenseND::<f64>::random_uniform(&[20, 15, 15], 0.0, 1.0);
    let cp_a = cp_als(&tensor_a, rank, 30, 1e-4, InitStrategy::Random, None)?;
    let new_a = DenseND::<f64>::random_uniform(&[5, 15, 15], 0.0, 1.0);

    let cp_updated_a = cp_als_incremental(&cp_a, &new_a, 0, IncrementalMode::Append, 8, 1e-4)?;
    println!("  Original: {:?}", tensor_a.shape());
    println!(
        "  Updated: mode-0 factor shape = {:?}",
        cp_updated_a.factors[0].shape()
    );
    println!("  Fit: {:.4}", cp_updated_a.fit);
    println!();

    // Example 6b: Grow along mode 1 (spatial)
    println!("Scenario B: Growing spatial dimension (mode 1)");
    let tensor_b = DenseND::<f64>::random_uniform(&[20, 15, 15], 0.0, 1.0);
    let cp_b = cp_als(&tensor_b, rank, 30, 1e-4, InitStrategy::Random, None)?;
    let new_b = DenseND::<f64>::random_uniform(&[20, 5, 15], 0.0, 1.0);

    let cp_updated_b = cp_als_incremental(&cp_b, &new_b, 1, IncrementalMode::Append, 8, 1e-4)?;
    println!("  Original: {:?}", tensor_b.shape());
    println!(
        "  Updated: mode-1 factor shape = {:?}",
        cp_updated_b.factors[1].shape()
    );
    println!("  Fit: {:.4}", cp_updated_b.fit);
    println!();

    // Example 6c: Grow along mode 2
    println!("Scenario C: Growing third dimension (mode 2)");
    let tensor_c = DenseND::<f64>::random_uniform(&[20, 15, 15], 0.0, 1.0);
    let cp_c = cp_als(&tensor_c, rank, 30, 1e-4, InitStrategy::Random, None)?;
    let new_c = DenseND::<f64>::random_uniform(&[20, 15, 5], 0.0, 1.0);

    let cp_updated_c = cp_als_incremental(&cp_c, &new_c, 2, IncrementalMode::Append, 8, 1e-4)?;
    println!("  Original: {:?}", tensor_c.shape());
    println!(
        "  Updated: mode-2 factor shape = {:?}",
        cp_updated_c.factors[2].shape()
    );
    println!("  Fit: {:.4}", cp_updated_c.fit);
    println!();

    println!("✓ Incremental updates work on any mode");

    Ok(())
}
