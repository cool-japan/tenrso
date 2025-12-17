//! Advanced features showcase
//!
//! This example demonstrates the new advanced features of tenrso-planner:
//! - Plan caching for improved performance
//! - Execution simulation for cost estimation
//! - Hardware-specific optimization
//! - Plan analysis and comparison

use tenrso_planner::{
    beam_search_planner, dp_planner, greedy_planner, simulate_execution,
    simulate_execution_with_hardware, EinsumSpec, HardwareModel, PlanCache, PlanHints,
};

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(80));
    println!("TenRSo-Planner: Advanced Features Showcase");
    println!("{}", "=".repeat(80));
    println!();

    // Example 1: Plan Caching
    demonstration_plan_caching()?;
    println!();

    // Example 2: Execution Simulation
    demonstration_execution_simulation()?;
    println!();

    // Example 3: Hardware-Specific Optimization
    demonstration_hardware_optimization()?;
    println!();

    // Example 4: Plan Analysis and Comparison
    demonstration_plan_analysis()?;

    Ok(())
}

fn demonstration_plan_caching() -> anyhow::Result<()> {
    println!("📦 DEMONSTRATION 1: Plan Caching");
    println!("{}", "-".repeat(80));

    // Create a cache with capacity for 100 plans
    let cache = PlanCache::new(100);

    let spec = EinsumSpec::parse("ij,jk,kl->il")?;
    let shapes = vec![vec![100, 200], vec![200, 300], vec![300, 100]];
    let hints = PlanHints::default();

    println!("Planning a complex contraction: ij,jk,kl->il");
    println!("Tensor shapes: [100x200], [200x300], [300x100]");
    println!();

    // First call: cache miss (plan is computed)
    println!("First call (cache miss):");
    let plan1 = cache.get_or_compute(&spec, &shapes, &hints, || {
        greedy_planner(&spec, &shapes, &hints)
    })?;
    println!("  ✓ Plan computed and cached");
    println!("  Estimated FLOPs: {:.2e}", plan1.estimated_flops);

    // Second call: cache hit (plan is retrieved)
    println!("\nSecond call (cache hit):");
    let plan2 = cache.get_or_compute(&spec, &shapes, &hints, || {
        greedy_planner(&spec, &shapes, &hints)
    })?;
    println!("  ✓ Plan retrieved from cache (fast!)");
    println!("  Estimated FLOPs: {:.2e}", plan2.estimated_flops);

    // Cache statistics
    let stats = cache.stats();
    println!("\nCache Statistics:");
    println!("  Hits: {}", stats.hits);
    println!("  Misses: {}", stats.misses);
    println!("  Hit rate: {:.1}%", stats.hit_rate() * 100.0);
    println!("  Cached entries: {}", stats.entries);

    Ok(())
}

fn demonstration_execution_simulation() -> anyhow::Result<()> {
    println!("⚡ DEMONSTRATION 2: Execution Simulation");
    println!("{}", "-".repeat(80));

    let spec = EinsumSpec::parse("ij,jk,kl,lm->im")?;
    let shapes = vec![
        vec![100, 200],
        vec![200, 300],
        vec![300, 400],
        vec![400, 100],
    ];
    let hints = PlanHints::default();

    println!("Planning a 4-tensor contraction chain: ij,jk,kl,lm->im");
    println!("Tensor shapes: [100x200], [200x300], [300x400], [400x100]");
    println!();

    // Plan with different algorithms
    let greedy_plan = greedy_planner(&spec, &shapes, &hints)?;
    let dp_plan = dp_planner(&spec, &shapes, &hints)?;
    let beam_plan = beam_search_planner(&spec, &shapes, &hints, 5)?;

    println!("Planner       | FLOPs        | Memory (MB) | Steps");
    println!("{}", "-".repeat(60));

    for (name, plan) in &[
        ("Greedy", &greedy_plan),
        ("DP (optimal)", &dp_plan),
        ("Beam (k=5)", &beam_plan),
    ] {
        println!(
            "{:<13} | {:<12.2e} | {:<11} | {}",
            name,
            plan.estimated_flops,
            plan.estimated_memory / 1_000_000,
            plan.nodes.len()
        );
    }

    println!("\nSimulating execution of optimal (DP) plan:");
    let simulation = simulate_execution(&dp_plan);

    println!("\nSimulation Results:");
    println!("  Total runtime: {:.2}ms", simulation.total_time_ms);
    println!(
        "  Peak memory: {} MB",
        simulation.peak_memory_bytes / 1_000_000
    );
    println!(
        "  Compute time: {:.2}ms ({:.1}%)",
        simulation.total_compute_time_ms(),
        (simulation.total_compute_time_ms() / simulation.total_time_ms) * 100.0
    );
    println!(
        "  Memory time: {:.2}ms ({:.1}%)",
        simulation.total_memory_time_ms(),
        (simulation.total_memory_time_ms() / simulation.total_time_ms) * 100.0
    );
    println!(
        "  Allocation overhead: {:.2}ms",
        simulation.total_allocation_time_ms()
    );
    println!("  Compute intensity: {:.2}", simulation.compute_intensity);
    println!(
        "  Cache efficiency: {:.1}%",
        simulation.cache_efficiency * 100.0
    );

    if simulation.is_compute_bound() {
        println!("\n  ✓ This operation is compute-bound (good for GPU acceleration)");
    } else {
        println!("\n  ⚠ This operation is memory-bound (consider tiling or compression)");
    }

    // Find critical step
    if let Some(critical) = simulation.critical_step() {
        println!("\nCritical step (bottleneck):");
        println!("  Total time: {:.2}ms", critical.total_time_ms);
        println!("  Compute: {:.2}ms", critical.compute_time_ms);
        println!("  Memory: {:.2}ms", critical.memory_time_ms);
        println!(
            "  Cache-friendly: {}",
            if critical.cache_friendly { "Yes" } else { "No" }
        );
    }

    Ok(())
}

fn demonstration_hardware_optimization() -> anyhow::Result<()> {
    println!("🖥️  DEMONSTRATION 3: Hardware-Specific Optimization");
    println!("{}", "-".repeat(80));

    let spec = EinsumSpec::parse("ij,jk->ik")?;
    let shapes = vec![vec![500, 1000], vec![1000, 500]];
    let hints = PlanHints::default();

    println!("Matrix multiplication: ij,jk->ik");
    println!("Matrix sizes: 500x1000 × 1000x500");
    println!();

    let plan = greedy_planner(&spec, &shapes, &hints)?;

    // Simulate on different hardware
    let hardware_models = vec![
        ("Low-end CPU", HardwareModel::low_end_cpu()),
        ("Standard CPU", HardwareModel::default()),
        ("High-end CPU", HardwareModel::high_end_cpu()),
        ("GPU", HardwareModel::gpu()),
    ];

    println!("Hardware       | Runtime (ms) | Compute (ms) | Memory (ms) | Speedup");
    println!("{}", "-".repeat(75));

    let baseline_time =
        simulate_execution_with_hardware(&plan, &HardwareModel::low_end_cpu()).total_time_ms;

    for (name, model) in &hardware_models {
        let sim = simulate_execution_with_hardware(&plan, model);
        let speedup = baseline_time / sim.total_time_ms;

        println!(
            "{:<14} | {:>12.2} | {:>12.2} | {:>11.2} | {:>7.2}x",
            name,
            sim.total_time_ms,
            sim.total_compute_time_ms(),
            sim.total_memory_time_ms(),
            speedup
        );
    }

    println!("\nRecommendation:");
    let gpu_sim = simulate_execution_with_hardware(&plan, &HardwareModel::gpu());
    let cpu_sim = simulate_execution_with_hardware(&plan, &HardwareModel::high_end_cpu());

    if gpu_sim.total_time_ms < cpu_sim.total_time_ms * 0.5 {
        println!(
            "  ✓ GPU acceleration recommended ({:.1}x faster than high-end CPU)",
            cpu_sim.total_time_ms / gpu_sim.total_time_ms
        );
    } else {
        println!("  ✓ CPU execution is sufficient for this problem size");
    }

    Ok(())
}

fn demonstration_plan_analysis() -> anyhow::Result<()> {
    println!("📊 DEMONSTRATION 4: Plan Analysis and Comparison");
    println!("{}", "-".repeat(80));

    let spec = EinsumSpec::parse("ij,jk,kl->il")?;
    let shapes = vec![vec![50, 100], vec![100, 150], vec![150, 50]];
    let hints = PlanHints::default();

    println!("Comparing planning algorithms on: ij,jk,kl->il");
    println!("Tensor shapes: [50x100], [100x150], [150x50]");
    println!();

    let greedy_plan = greedy_planner(&spec, &shapes, &hints)?;
    let dp_plan = dp_planner(&spec, &shapes, &hints)?;

    // Analyze plans
    let greedy_analysis = greedy_plan.analyze();
    let dp_analysis = dp_plan.analyze();

    println!("Greedy Plan Analysis:");
    println!("  Total FLOPs: {:.2e}", greedy_analysis.total_flops);
    println!("  Steps: {}", greedy_analysis.num_steps);
    println!("  Avg cost/step: {:.2e}", greedy_analysis.avg_cost_per_step);
    println!("  Max cost step: {:.2e}", greedy_analysis.max_cost_step);
    println!("  Min cost step: {:.2e}", greedy_analysis.min_cost_step);

    println!("\nDP (Optimal) Plan Analysis:");
    println!("  Total FLOPs: {:.2e}", dp_analysis.total_flops);
    println!("  Steps: {}", dp_analysis.num_steps);
    println!("  Avg cost/step: {:.2e}", dp_analysis.avg_cost_per_step);
    println!("  Max cost step: {:.2e}", dp_analysis.max_cost_step);
    println!("  Min cost step: {:.2e}", dp_analysis.min_cost_step);

    // Compare plans
    let comparison = greedy_plan.compare(&dp_plan);

    println!("\nComparison (Greedy vs DP):");
    println!("  FLOPs difference: {:.2e}", comparison.flops_difference);
    println!("  FLOPs ratio: {:.2}", comparison.flops_ratio);

    if comparison.flops_ratio < 1.0 {
        let improvement = (1.0 - comparison.flops_ratio) * 100.0;
        println!("  ✓ DP is {:.1}% better than greedy", improvement);
    } else {
        println!("  ✓ Greedy found the optimal plan!");
    }

    // Check if plan is better
    println!("\nIs DP better than greedy?");
    use tenrso_planner::PlanMetric;
    for metric in [PlanMetric::FLOPs, PlanMetric::Memory, PlanMetric::Combined] {
        let is_better = dp_plan.is_better_than(&greedy_plan, metric);
        println!(
            "  {:?}: {}",
            metric,
            if is_better { "Yes ✓" } else { "No ✗" }
        );
    }

    Ok(())
}
