//! Basic example: Planning a matrix multiplication
//!
//! This example demonstrates how to use the planner for a simple matrix
//! multiplication: A_ij * B_jk -> C_ik

use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};

fn main() {
    // Parse the einsum specification
    let spec = EinsumSpec::parse("ij,jk->ik").expect("Valid einsum spec");

    // Define tensor shapes
    let shapes = vec![
        vec![100, 200], // A: 100x200
        vec![200, 300], // B: 200x300
    ];

    // Create planning hints (use defaults for now)
    let hints = PlanHints::default();

    // Generate the plan
    let plan = greedy_planner(&spec, &shapes, &hints).expect("Planning succeeded");

    // Display plan information
    println!("Matrix Multiplication Plan");
    println!("=========================");
    println!("Specification: ij,jk->ik");
    println!("Input shapes: {:?}", shapes);
    println!();

    println!("Plan Details:");
    println!("  Number of contractions: {}", plan.nodes.len());
    println!("  Estimated FLOPs: {:.2e}", plan.estimated_flops);
    println!("  Peak memory (bytes): {}", plan.estimated_memory);
    println!();

    println!("Contraction Steps:");
    for (i, node) in plan.nodes.iter().enumerate() {
        println!("  Step {}: {:?}", i + 1, node.output_spec);
        println!("    Cost (FLOPs): {:.2e}", node.cost);
        println!("    Memory (bytes): {}", node.memory);
        println!("    Representation: {:?}", node.repr);
    }
}
