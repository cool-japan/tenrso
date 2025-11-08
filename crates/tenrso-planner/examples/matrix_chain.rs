//! Example: Planning a matrix chain multiplication
//!
//! This example demonstrates how the planner finds an efficient contraction
//! order for a chain of matrix multiplications: A_ij * B_jk * C_kl * D_lm -> E_im
//!
//! The classical matrix chain multiplication problem is a well-known dynamic
//! programming problem. This example shows how the greedy and DP planners
//! find good contraction orders.

use tenrso_planner::{dp_planner, greedy_planner, EinsumSpec, PlanHints};

fn main() {
    // Parse the einsum specification for 4-matrix chain
    let spec = EinsumSpec::parse("ij,jk,kl,lm->im").expect("Valid einsum spec");

    // Define tensor shapes with varying dimensions
    // This creates different costs for different contraction orders
    let shapes = vec![
        vec![10, 100], // A: 10x100
        vec![100, 5],  // B: 100x5
        vec![5, 50],   // C: 5x50
        vec![50, 20],  // D: 50x20
    ];

    let hints = PlanHints::default();

    println!("Matrix Chain Multiplication");
    println!("==========================");
    println!("Specification: ij,jk,kl,lm->im");
    println!("Input shapes: {:?}", shapes);
    println!();

    // Try greedy planner
    println!("GREEDY PLANNER");
    println!("--------------");
    let greedy_plan = greedy_planner(&spec, &shapes, &hints).expect("Planning succeeded");
    println!("  Number of contractions: {}", greedy_plan.nodes.len());
    println!("  Estimated FLOPs: {:.2e}", greedy_plan.estimated_flops);
    println!("  Peak memory (bytes): {}", greedy_plan.estimated_memory);
    println!();

    println!("  Contraction Order:");
    for (i, node) in greedy_plan.nodes.iter().enumerate() {
        println!(
            "    Step {}: {} -> {}",
            i + 1,
            node.output_spec.input_specs.join(","),
            node.output_spec.output_spec
        );
        println!("      Cost: {:.2e} FLOPs", node.cost);
    }
    println!();

    // Try DP planner (optimal for small problems)
    println!("DYNAMIC PROGRAMMING PLANNER (Optimal)");
    println!("-------------------------------------");
    let dp_plan = dp_planner(&spec, &shapes, &hints).expect("Planning succeeded");
    println!("  Number of contractions: {}", dp_plan.nodes.len());
    println!("  Estimated FLOPs: {:.2e}", dp_plan.estimated_flops);
    println!("  Peak memory (bytes): {}", dp_plan.estimated_memory);
    println!();

    println!("  Contraction Order:");
    for (i, node) in dp_plan.nodes.iter().enumerate() {
        println!(
            "    Step {}: {} -> {}",
            i + 1,
            node.output_spec.input_specs.join(","),
            node.output_spec.output_spec
        );
        println!("      Cost: {:.2e} FLOPs", node.cost);
    }
    println!();

    // Compare the two plans
    println!("COMPARISON");
    println!("----------");
    let improvement = (greedy_plan.estimated_flops - dp_plan.estimated_flops)
        / greedy_plan.estimated_flops
        * 100.0;

    if improvement > 0.0 {
        println!(
            "DP planner found a {:.1}% better plan than greedy!",
            improvement
        );
    } else {
        println!("Both planners found equivalent solutions.");
    }
}
