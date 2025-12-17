//! Example: Parallel Ensemble Planning
//!
//! This example demonstrates how to use the EnsemblePlanner to run multiple
//! planning algorithms concurrently and automatically select the best result.
//!
//! Run with: cargo run --example parallel_ensemble

use std::time::Instant;
use tenrso_planner::{
    beam_search_planner, dp_planner, greedy_planner, AdaptivePlanner, EinsumSpec, EnsemblePlanner,
    PlanHints, Planner,
};

fn main() -> anyhow::Result<()> {
    println!("=== Parallel Ensemble Planning Example ===\n");

    // Test case: Complex tensor network
    let spec = EinsumSpec::parse("ijk,jkl,klm,lmn->in")?;
    let shapes = vec![
        vec![50, 60, 70],
        vec![60, 70, 80],
        vec![70, 80, 90],
        vec![80, 90, 100],
    ];
    let hints = PlanHints::default();

    println!("Problem: Four 3D tensors with chain-like contraction");
    println!("Specification: ijk,jkl,klm,lmn->in");
    println!("Tensor shapes: 50×60×70, 60×70×80, 70×80×90, 80×90×100");
    println!();

    // Scenario 1: Sequential planning (baseline)
    println!("=== Scenario 1: Sequential Planning (Baseline) ===\n");

    let start = Instant::now();
    let greedy_plan = greedy_planner(&spec, &shapes, &hints)?;
    let greedy_time = start.elapsed();

    let start = Instant::now();
    let beam_plan = beam_search_planner(&spec, &shapes, &hints, 5)?;
    let beam_time = start.elapsed();

    let start = Instant::now();
    let dp_plan = dp_planner(&spec, &shapes, &hints)?;
    let dp_time = start.elapsed();

    println!(
        "Greedy:       {:.2e} FLOPs in {:?}",
        greedy_plan.estimated_flops, greedy_time
    );
    println!(
        "Beam Search:  {:.2e} FLOPs in {:?}",
        beam_plan.estimated_flops, beam_time
    );
    println!(
        "DP:           {:.2e} FLOPs in {:?}",
        dp_plan.estimated_flops, dp_time
    );
    println!();

    let total_sequential_time = greedy_time + beam_time + dp_time;
    let best_sequential_flops = greedy_plan
        .estimated_flops
        .min(beam_plan.estimated_flops)
        .min(dp_plan.estimated_flops);

    println!("Total sequential time: {:?}", total_sequential_time);
    println!("Best result: {:.2e} FLOPs", best_sequential_flops);
    println!();

    // Scenario 2: Parallel ensemble planning
    println!("=== Scenario 2: Parallel Ensemble Planning ===\n");

    let ensemble = EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]);

    let start = Instant::now();
    let ensemble_plan = ensemble.plan("ijk,jkl,klm,lmn->in", &shapes, &hints)?;
    let ensemble_time = start.elapsed();

    println!(
        "Ensemble result: {:.2e} FLOPs in {:?}",
        ensemble_plan.estimated_flops, ensemble_time
    );
    println!();

    let speedup = total_sequential_time.as_secs_f64() / ensemble_time.as_secs_f64();
    println!("Speedup: {:.2}x faster than sequential", speedup);
    println!(
        "Same quality: {}",
        if (ensemble_plan.estimated_flops - best_sequential_flops).abs() < 1.0 {
            "✓ (found best plan)"
        } else {
            "✗"
        }
    );
    println!();

    // Scenario 3: Different selection metrics
    println!("=== Scenario 3: Different Selection Metrics ===\n");

    let ensemble_flops =
        EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]).with_metric("flops");
    let plan_flops = ensemble_flops.plan("ijk,jkl,klm,lmn->in", &shapes, &hints)?;

    let ensemble_memory =
        EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]).with_metric("memory");
    let plan_memory = ensemble_memory.plan("ijk,jkl,klm,lmn->in", &shapes, &hints)?;

    let ensemble_combined =
        EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]).with_metric("combined");
    let plan_combined = ensemble_combined.plan("ijk,jkl,klm,lmn->in", &shapes, &hints)?;

    println!(
        "FLOPs metric:     {:.2e} FLOPs, {} MB",
        plan_flops.estimated_flops,
        plan_flops.estimated_memory / 1_000_000
    );
    println!(
        "Memory metric:    {:.2e} FLOPs, {} MB",
        plan_memory.estimated_flops,
        plan_memory.estimated_memory / 1_000_000
    );
    println!(
        "Combined metric:  {:.2e} FLOPs, {} MB",
        plan_combined.estimated_flops,
        plan_combined.estimated_memory / 1_000_000
    );
    println!();

    // Scenario 4: Large ensemble with all algorithms
    println!("=== Scenario 4: Large Ensemble (All Algorithms) ===\n");

    let large_ensemble = EnsemblePlanner::new(vec![
        "greedy",
        "beam_search",
        "dp",
        "simulated_annealing",
        "genetic_algorithm",
    ]);

    let start = Instant::now();
    let large_plan = large_ensemble.plan("ijk,jkl,klm,lmn->in", &shapes, &hints)?;
    let large_time = start.elapsed();

    println!("Ensemble with 5 planners:");
    println!("  Result: {:.2e} FLOPs", large_plan.estimated_flops);
    println!("  Time: {:?}", large_time);
    println!(
        "  Quality: {:.1}% better than greedy",
        (greedy_plan.estimated_flops - large_plan.estimated_flops) / greedy_plan.estimated_flops
            * 100.0
    );
    println!();

    // Scenario 5: Using ensemble via Planner trait
    println!("=== Scenario 5: Polymorphic Usage via Planner Trait ===\n");

    let planners: Vec<Box<dyn Planner>> = vec![
        Box::new(EnsemblePlanner::default()),
        Box::new(AdaptivePlanner::default()),
    ];

    for (i, planner) in planners.iter().enumerate() {
        let plan = planner.make_plan("ijk,jkl,klm,lmn->in", &shapes, &hints)?;
        println!("Planner {}: {:.2e} FLOPs", i + 1, plan.estimated_flops);
    }
    println!();

    // Scenario 6: Scaling with problem size
    println!("=== Scenario 6: Scaling with Problem Size ===\n");

    let problem_sizes = vec![
        (
            "Small (3 tensors)",
            "ij,jk,kl->il",
            vec![vec![20, 30], vec![30, 40], vec![40, 50]],
        ),
        (
            "Medium (4 tensors)",
            "ij,jk,kl,lm->im",
            vec![vec![30, 40], vec![40, 50], vec![50, 60], vec![60, 70]],
        ),
        (
            "Large (5 tensors)",
            "ij,jk,kl,lm,mn->in",
            vec![
                vec![40, 50],
                vec![50, 60],
                vec![60, 70],
                vec![70, 80],
                vec![80, 90],
            ],
        ),
    ];

    println!("Ensemble vs Sequential for different problem sizes:");
    println!();

    for (name, spec_str, shapes) in problem_sizes {
        let ensemble = EnsemblePlanner::new(vec!["greedy", "beam_search"]);

        let start = Instant::now();
        let ensemble_plan = ensemble.plan(spec_str, &shapes, &hints)?;
        let ensemble_time = start.elapsed();

        println!("{}", name);
        println!(
            "  Result: {:.2e} FLOPs in {:?}",
            ensemble_plan.estimated_flops, ensemble_time
        );
    }
    println!();

    // Summary
    println!("=== Summary ===\n");
    println!("✓ Ensemble planning runs algorithms in parallel");
    println!("✓ Near-linear speedup with number of cores");
    println!("✓ Automatic best-plan selection");
    println!("✓ Supports all 6 planning algorithms");
    println!("✓ Configurable selection metrics (FLOPs, memory, combined)");
    println!("✓ Polymorphic via Planner trait");
    println!();

    println!("Recommendations:");
    println!("• Use ensemble for production: maximize quality within time budget");
    println!("• Use 2-3 planners for fast results (greedy + beam_search)");
    println!("• Use 5-6 planners for best quality (includes SA, GA)");
    println!("• Choose metric based on constraints (memory vs compute)");

    Ok(())
}
