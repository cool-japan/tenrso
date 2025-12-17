//! Example demonstrating plan visualization capabilities
//!
//! This example shows how to:
//! - Use the Display trait to print human-readable plans
//! - Export plans to Graphviz DOT format for visualization
//! - Validate plans for correctness
//!
//! Run with:
//! ```bash
//! cargo run --example plan_visualization
//! ```

use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};

fn main() -> anyhow::Result<()> {
    println!("=== Plan Visualization Example ===\n");

    // Example 1: Matrix multiplication chain
    println!("Example 1: Matrix Multiplication Chain");
    println!("---------------------------------------");

    let spec = EinsumSpec::parse("ab,bc,cd->ad")?;
    let shapes = vec![vec![100, 50], vec![50, 200], vec![200, 30]];
    let hints = PlanHints::default();

    let plan = greedy_planner(&spec, &shapes, &hints)?;

    // Display the plan in human-readable format
    println!("\n{}", plan);

    // Validate the plan
    println!("Validating plan...");
    match plan.validate() {
        Ok(()) => println!("✓ Plan is valid!\n"),
        Err(e) => println!("✗ Plan validation failed: {}\n", e),
    }

    // Export to DOT format
    println!("Graphviz DOT Export:");
    println!("-------------------");
    let dot = plan.to_dot();
    println!("{}", dot);
    println!("\nTo visualize, save the above to 'plan.dot' and run:");
    println!("  dot -Tpng plan.dot -o plan.png\n");

    // Example 2: Tensor contraction with multiple indices
    println!("\n========================================\n");
    println!("Example 2: Complex Tensor Contraction");
    println!("--------------------------------------");

    let spec2 = EinsumSpec::parse("ijk,jkl,klm->ilm")?;
    let shapes2 = vec![vec![10, 20, 30], vec![20, 30, 40], vec![30, 40, 50]];

    let plan2 = greedy_planner(&spec2, &shapes2, &hints)?;

    println!("\n{}", plan2);

    // Analyze the plan
    let analysis = plan2.analyze();
    println!("Plan Analysis:");
    println!("  Number of steps: {}", analysis.num_steps);
    println!("  Total FLOPs: {:.2e}", analysis.total_flops);
    println!(
        "  Peak memory: {} bytes ({:.2} MB)",
        analysis.peak_memory,
        analysis.peak_memory as f64 / 1_048_576.0
    );
    println!(
        "  Average cost per step: {:.2e}",
        analysis.avg_cost_per_step
    );
    println!("  Max cost step: {:.2e}", analysis.max_cost_step);
    println!("  Min cost step: {:.2e}", analysis.min_cost_step);
    println!(
        "  Average memory per step: {} bytes",
        analysis.avg_memory_per_step
    );
    println!("  Max memory step: {} bytes", analysis.max_memory_step);
    println!("  Min memory step: {} bytes\n", analysis.min_memory_step);

    // Example 3: Compare different plans
    println!("\n========================================\n");
    println!("Example 3: Comparing Plans");
    println!("--------------------------");

    // Create two different contraction orders manually for comparison
    let spec3 = EinsumSpec::parse("ab,bc->ac")?;
    let shapes3 = vec![vec![100, 200], vec![200, 300]];

    let plan3a = greedy_planner(&spec3, &shapes3, &hints)?;
    println!("Plan A:");
    println!("{}", plan3a);

    // For demonstration, create a modified version
    // In practice, you'd use different planners (DP, Beam Search, etc.)
    let spec3b = EinsumSpec::parse("ab,bc->ac")?;
    let plan3b = greedy_planner(&spec3b, &shapes3, &hints)?;

    // Compare plans
    let comparison = plan3a.compare(&plan3b);
    println!("Comparison (Plan B vs Plan A):");
    println!("  FLOPs ratio: {:.2}", comparison.flops_ratio);
    println!("  FLOPs difference: {:.2e}", comparison.flops_difference);
    println!(
        "  FLOPs improvement: {:.2}%",
        comparison.flops_improvement_percent()
    );
    println!("  Memory ratio: {:.2}", comparison.memory_ratio);
    println!(
        "  Memory difference: {} bytes",
        comparison.memory_difference
    );
    println!("  Steps difference: {}\n", comparison.steps_difference);

    // Check which plan is better
    use tenrso_planner::PlanMetric;
    if plan3a.is_better_than(&plan3b, PlanMetric::FLOPs) {
        println!("Plan A is better (lower FLOPs)");
    } else {
        println!("Plan B is better (lower FLOPs)");
    }

    println!("\n=== Example Complete ===\n");

    Ok(())
}
