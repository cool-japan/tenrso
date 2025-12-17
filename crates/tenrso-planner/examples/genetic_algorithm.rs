//! Genetic Algorithm Planner Example
//!
//! This example demonstrates the use of the Genetic Algorithm (GA) planner
//! for large tensor network contraction planning. GA is particularly effective
//! for networks with many tensors (> 20) where dynamic programming becomes
//! computationally infeasible.
//!
//! # What You'll Learn
//!
//! 1. When to use the GA planner
//! 2. How to configure GA parameters
//! 3. Comparing GA with other planners
//! 4. Trade-offs between planning time and solution quality
//!
//! # Run This Example
//!
//! ```bash
//! cargo run --example genetic_algorithm
//! ```

use tenrso_planner::{
    BeamSearchPlanner, GeneticAlgorithmPlanner, GreedyPlanner, PlanHints, Planner,
};

fn main() {
    println!("=== Genetic Algorithm Planner Example ===\n");

    // Example 1: Large tensor chain (where GA shines)
    println!("Example 1: Large Tensor Chain (10 tensors)");
    println!("--------------------------------------------");

    // Create a chain of 10 matrix multiplications
    let spec = "ab,bc,cd,de,ef,fg,gh,hi,ij,jk->ak";
    let shapes: Vec<Vec<usize>> = vec![
        vec![50, 100],
        vec![100, 30],
        vec![30, 200],
        vec![200, 40],
        vec![40, 150],
        vec![150, 60],
        vec![60, 180],
        vec![180, 70],
        vec![70, 120],
        vec![120, 50],
    ];
    let hints = PlanHints::default();

    // Compare different planners
    println!("\n1a. Greedy Planner (fast, decent quality):");
    let greedy_planner = GreedyPlanner::new();
    let greedy_plan = greedy_planner.make_plan(spec, &shapes, &hints).unwrap();
    println!("   FLOPs: {:.2e}", greedy_plan.estimated_flops);
    println!("   Steps: {}", greedy_plan.nodes.len());

    println!("\n1b. Beam Search Planner (medium quality, medium speed):");
    let beam_planner = BeamSearchPlanner::with_beam_width(5);
    let beam_plan = beam_planner.make_plan(spec, &shapes, &hints).unwrap();
    println!("   FLOPs: {:.2e}", beam_plan.estimated_flops);
    println!("   Steps: {}", beam_plan.nodes.len());
    let beam_improvement = (1.0 - beam_plan.estimated_flops / greedy_plan.estimated_flops) * 100.0;
    println!(
        "   Improvement over greedy: {:.1}%",
        beam_improvement.max(0.0)
    );

    println!("\n1c. GA Planner - Fast (quick exploration):");
    let ga_fast = GeneticAlgorithmPlanner::fast();
    let ga_fast_plan = ga_fast.make_plan(spec, &shapes, &hints).unwrap();
    println!("   FLOPs: {:.2e}", ga_fast_plan.estimated_flops);
    println!("   Steps: {}", ga_fast_plan.nodes.len());
    println!("   Config: pop=50, gen=50, mut=0.2");
    let ga_fast_improvement =
        (1.0 - ga_fast_plan.estimated_flops / greedy_plan.estimated_flops) * 100.0;
    println!(
        "   Improvement over greedy: {:.1}%",
        ga_fast_improvement.max(0.0)
    );

    println!("\n1d. GA Planner - High Quality (thorough search):");
    let ga_hq = GeneticAlgorithmPlanner::high_quality();
    let ga_hq_plan = ga_hq.make_plan(spec, &shapes, &hints).unwrap();
    println!("   FLOPs: {:.2e}", ga_hq_plan.estimated_flops);
    println!("   Steps: {}", ga_hq_plan.nodes.len());
    println!("   Config: pop=200, gen=200, mut=0.15");
    let ga_hq_improvement =
        (1.0 - ga_hq_plan.estimated_flops / greedy_plan.estimated_flops) * 100.0;
    println!(
        "   Improvement over greedy: {:.1}%",
        ga_hq_improvement.max(0.0)
    );

    // Example 2: Star topology (center node connected to all others)
    println!("\n\nExample 2: Star Topology (1 hub + 8 spokes)");
    println!("--------------------------------------------");

    let star_spec = "ai,bi,ci,di,ei,fi,gi,hi->abcdefgh";
    let star_shapes: Vec<Vec<usize>> = vec![
        vec![10, 50], // spoke 1
        vec![20, 50], // spoke 2
        vec![15, 50], // spoke 3
        vec![25, 50], // spoke 4
        vec![12, 50], // spoke 5
        vec![18, 50], // spoke 6
        vec![22, 50], // spoke 7
        vec![16, 50], // spoke 8
    ];

    println!("\n2a. Greedy:");
    let star_greedy = greedy_planner
        .make_plan(star_spec, &star_shapes, &hints)
        .unwrap();
    println!("   FLOPs: {:.2e}", star_greedy.estimated_flops);

    println!("\n2b. GA (default params):");
    let ga_default = GeneticAlgorithmPlanner::new();
    let star_ga = ga_default
        .make_plan(star_spec, &star_shapes, &hints)
        .unwrap();
    println!("   FLOPs: {:.2e}", star_ga.estimated_flops);
    let star_improvement = (1.0 - star_ga.estimated_flops / star_greedy.estimated_flops) * 100.0;
    println!(
        "   Improvement over greedy: {:.1}%",
        star_improvement.max(0.0)
    );

    // Example 3: Custom GA parameters
    println!("\n\nExample 3: Custom GA Configuration");
    println!("-----------------------------------");

    let custom_ga = GeneticAlgorithmPlanner::with_params(
        150,  // population_size
        150,  // max_generations
        0.25, // mutation_rate (higher for more exploration)
        8,    // elitism_count
    );

    println!("Custom GA Parameters:");
    println!("  Population: 150");
    println!("  Generations: 150");
    println!("  Mutation Rate: 0.25");
    println!("  Elitism: 8 best individuals");

    let custom_plan = custom_ga.make_plan(spec, &shapes, &hints).unwrap();
    println!("\nResults:");
    println!("   FLOPs: {:.2e}", custom_plan.estimated_flops);
    println!("   Steps: {}", custom_plan.nodes.len());

    // Summary and recommendations
    println!("\n\n=== Summary and Recommendations ===");
    println!("\n1. When to use Genetic Algorithm:");
    println!("   ✓ Large tensor networks (> 20 tensors)");
    println!("   ✓ Complex topologies (not simple chains)");
    println!("   ✓ When DP is too slow (exponential time)");
    println!("   ✓ When you have time budget for planning");
    println!("   ✓ When quality matters more than planning speed");

    println!("\n2. GA Configuration Guide:");
    println!("   Fast GA (pop=50, gen=50):");
    println!("     - Planning time: ~50-100ms");
    println!("     - Quality: Better than greedy, similar to beam search");
    println!("     - Use: Quick experiments, development");
    println!();
    println!("   Default GA (pop=100, gen=100):");
    println!("     - Planning time: ~100-300ms");
    println!("     - Quality: Better than beam search");
    println!("     - Use: General production use");
    println!();
    println!("   High-Quality GA (pop=200, gen=200):");
    println!("     - Planning time: ~500-1000ms");
    println!("     - Quality: Near-optimal for large networks");
    println!("     - Use: Critical paths, offline optimization");

    println!("\n3. Performance Trade-offs:");
    println!("   Greedy     : O(n³)         - Fastest, good quality");
    println!("   Beam(k=5)  : O(5n³)        - 5x slower, better quality");
    println!("   GA(100,100): O(100*100*n³) - 10000x slower, best quality for large n");
    println!("   DP         : O(3ⁿ)         - Optimal but infeasible for n > 20");

    println!("\n4. Adaptive Planner:");
    println!("   For hassle-free optimization, use AdaptivePlanner which");
    println!("   automatically selects GA for large networks with high quality");
    println!("   requirements and sufficient time budget.");

    println!("\n=== Example Complete ===");
}
