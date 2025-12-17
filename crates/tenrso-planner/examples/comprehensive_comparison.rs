//! Comprehensive comparison of all planning algorithms
//!
//! This example demonstrates all 6 planners on various problem types:
//! - Matrix chain multiplication (sequential)
//! - Star topology (one tensor contracts with many)
//! - Tree topology (hierarchical contractions)
//! - Dense mesh (many interconnections)
//!
//! Run with: cargo run --example comprehensive_comparison

use tenrso_planner::{
    AdaptivePlanner, BeamSearchPlanner, DPPlanner, EinsumSpec, GeneticAlgorithmPlanner,
    GreedyPlanner, PlanHints, Planner, SimulatedAnnealingPlanner,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  TenRSo-Planner: Comprehensive Algorithm Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    // Test 1: Matrix Chain Multiplication
    println!("Test 1: Matrix Chain Multiplication (8 tensors)");
    println!("───────────────────────────────────────────────────────────");
    matrix_chain_test()?;

    println!("\n═══════════════════════════════════════════════════════════\n");

    // Test 2: Star Topology
    println!("Test 2: Star Topology (6 tensors)");
    println!("───────────────────────────────────────────────────────────");
    star_topology_test()?;

    println!("\n═══════════════════════════════════════════════════════════\n");

    // Test 3: Small Dense Network (optimal solvable)
    println!("Test 3: Small Dense Network (5 tensors)");
    println!("───────────────────────────────────────────────────────────");
    small_dense_test()?;

    println!("\n═══════════════════════════════════════════════════════════\n");

    // Test 4: Large Network (DP infeasible)
    println!("Test 4: Large Network (25 tensors - DP infeasible)");
    println!("───────────────────────────────────────────────────────────");
    large_network_test()?;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Comparison Complete!");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Key Takeaways:");
    println!("  • Adaptive planner automatically selects best algorithm");
    println!("  • DP finds optimal plans for small networks (n ≤ 20)");
    println!("  • Beam Search offers good quality-speed tradeoff");
    println!("  • SA and GA excel on large, complex networks");
    println!("  • Greedy is fastest but may miss optimal solutions");

    Ok(())
}

fn matrix_chain_test() -> Result<(), Box<dyn std::error::Error>> {
    // Chain: A(10x20) × B(20x30) × C(30x15) × D(15x25) × E(25x10) × F(10x40) × G(40x5) × H(5x30)
    let spec_str = "ab,bc,cd,de,ef,fg,gh,hi->ai";
    let spec = EinsumSpec::parse(spec_str)?;
    let shapes = vec![
        vec![10, 20],
        vec![20, 30],
        vec![30, 15],
        vec![15, 25],
        vec![25, 10],
        vec![10, 40],
        vec![40, 5],
        vec![5, 30],
    ];
    let hints = PlanHints::default();

    println!("Problem: 8 matrix multiplications in a chain");
    println!("Shape: (10×20) × (20×30) × ... × (5×30)");
    println!();

    compare_all_planners(spec_str, &spec, &shapes, &hints, false)?;

    Ok(())
}

fn star_topology_test() -> Result<(), Box<dyn std::error::Error>> {
    // Star: Central tensor 'i' contracts with all others
    let spec_str = "ia,ib,ic,id,ie,if->i";
    let spec = EinsumSpec::parse(spec_str)?;
    let shapes = vec![
        vec![100, 50],
        vec![100, 30],
        vec![100, 40],
        vec![100, 20],
        vec![100, 60],
        vec![100, 25],
    ];
    let hints = PlanHints::default();

    println!("Problem: Star topology (6 tensors, all share index 'i')");
    println!("Central index size: 100");
    println!();

    compare_all_planners(spec_str, &spec, &shapes, &hints, true)?;

    Ok(())
}

fn small_dense_test() -> Result<(), Box<dyn std::error::Error>> {
    // Dense interconnection with 5 tensors
    let spec_str = "ij,jk,kl,li,ik->i";
    let spec = EinsumSpec::parse(spec_str)?;
    let shapes = vec![
        vec![20, 30],
        vec![30, 25],
        vec![25, 20],
        vec![20, 20],
        vec![20, 25],
    ];
    let hints = PlanHints::default();

    println!("Problem: Dense interconnected network (5 tensors)");
    println!("Complex contraction pattern with cycles");
    println!();

    compare_all_planners(spec_str, &spec, &shapes, &hints, true)?;

    Ok(())
}

fn large_network_test() -> Result<(), Box<dyn std::error::Error>> {
    // Large chain where DP becomes infeasible
    let spec_str = "ab,bc,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,rs,st,tu,uv,vw,wx,xy,yz->az";
    let spec = EinsumSpec::parse(spec_str)?;
    let shapes = vec![
        vec![10, 20],
        vec![20, 30],
        vec![30, 15],
        vec![15, 25],
        vec![25, 10],
        vec![10, 40],
        vec![40, 5],
        vec![5, 30],
        vec![30, 12],
        vec![12, 28],
        vec![28, 18],
        vec![18, 22],
        vec![22, 16],
        vec![16, 35],
        vec![35, 8],
        vec![8, 45],
        vec![45, 14],
        vec![14, 32],
        vec![32, 11],
        vec![11, 26],
        vec![26, 19],
        vec![19, 38],
        vec![38, 7],
        vec![7, 42],
        vec![42, 15],
    ];
    let hints = PlanHints::default();

    println!("Problem: Large matrix chain (25 tensors)");
    println!("Note: DP skipped (too expensive for n > 20)");
    println!();

    compare_all_planners(spec_str, &spec, &shapes, &hints, false)?;

    Ok(())
}

fn compare_all_planners(
    spec_str: &str,
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    hints: &PlanHints,
    include_dp: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = spec.num_inputs();

    // 1. Greedy
    let greedy = GreedyPlanner::new();
    let start = std::time::Instant::now();
    let greedy_plan = greedy.make_plan(spec_str, shapes, hints)?;
    let greedy_time = start.elapsed();

    // 2. Beam Search
    let beam = BeamSearchPlanner::with_beam_width(5);
    let start = std::time::Instant::now();
    let beam_plan = beam.make_plan(spec_str, shapes, hints)?;
    let beam_time = start.elapsed();

    // 3. DP (if feasible)
    let (dp_plan, dp_time) = if include_dp && n <= 15 {
        let dp = DPPlanner::new();
        let start = std::time::Instant::now();
        let plan = dp.make_plan(spec_str, shapes, hints)?;
        let time = start.elapsed();
        (Some(plan), Some(time))
    } else {
        (None, None)
    };

    // 4. Simulated Annealing (reduced iterations for speed)
    let sa = SimulatedAnnealingPlanner::with_params(1000.0, 0.95, 200);
    let start = std::time::Instant::now();
    let sa_plan = sa.make_plan(spec_str, shapes, hints)?;
    let sa_time = start.elapsed();

    // 5. Genetic Algorithm (fast preset)
    let ga = GeneticAlgorithmPlanner::fast();
    let start = std::time::Instant::now();
    let ga_plan = ga.make_plan(spec_str, shapes, hints)?;
    let ga_time = start.elapsed();

    // 6. Adaptive
    let adaptive = AdaptivePlanner::new();
    let start = std::time::Instant::now();
    let adaptive_plan = adaptive.make_plan(spec_str, shapes, hints)?;
    let adaptive_time = start.elapsed();

    // Print results table
    println!("┌──────────────────┬────────────────┬─────────────┬───────────┐");
    println!("│ Algorithm        │ FLOPs          │ Time (ms)   │ Quality   │");
    println!("├──────────────────┼────────────────┼─────────────┼───────────┤");

    let baseline_flops = greedy_plan.estimated_flops;

    print_result(
        "Greedy",
        greedy_plan.estimated_flops,
        greedy_time,
        baseline_flops,
    );
    print_result(
        "Beam Search",
        beam_plan.estimated_flops,
        beam_time,
        baseline_flops,
    );

    if let (Some(plan), Some(time)) = (dp_plan, dp_time) {
        print_result("DP (Optimal)", plan.estimated_flops, time, baseline_flops);
    } else {
        println!("│ DP (Optimal)     │ N/A (too large)│ N/A         │ N/A       │");
    }

    print_result(
        "Sim. Annealing",
        sa_plan.estimated_flops,
        sa_time,
        baseline_flops,
    );
    print_result(
        "Genetic Algo",
        ga_plan.estimated_flops,
        ga_time,
        baseline_flops,
    );
    print_result(
        "Adaptive",
        adaptive_plan.estimated_flops,
        adaptive_time,
        baseline_flops,
    );

    println!("└──────────────────┴────────────────┴─────────────┴───────────┘");

    Ok(())
}

fn print_result(name: &str, flops: f64, time: std::time::Duration, baseline: f64) {
    let quality = if (flops - baseline).abs() < 1e-6 {
        "100%".to_string()
    } else {
        format!("{:.1}%", (baseline / flops) * 100.0)
    };

    println!(
        "│ {:<16} │ {:>14.2e} │ {:>11.2} │ {:>9} │",
        name,
        flops,
        time.as_secs_f64() * 1000.0,
        quality
    );
}
