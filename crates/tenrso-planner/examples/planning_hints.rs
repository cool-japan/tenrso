//! Example: Using planning hints to control optimization
//!
//! This example demonstrates how to use PlanHints to influence the planner's
//! behavior, such as setting memory budgets and providing sparsity information.

use std::collections::HashMap;
use tenrso_planner::{greedy_planner, Device, EinsumSpec, PlanHints};

fn main() {
    // Large tensor contraction that could exceed memory
    let spec = EinsumSpec::parse("ijk,jkl,klm->ilm").expect("Valid einsum spec");

    let shapes = vec![
        vec![1000, 1000, 100], // First tensor
        vec![1000, 100, 500],  // Second tensor
        vec![100, 500, 200],   // Third tensor
    ];

    println!("Planning with Hints Example");
    println!("===========================");
    println!("Specification: ijk,jkl,klm->ilm");
    println!("Input shapes: {:?}", shapes);
    println!();

    // Example 1: Default hints (no constraints)
    println!("1. DEFAULT HINTS (No constraints)");
    println!("----------------------------------");
    let default_hints = PlanHints::default();
    let default_plan = greedy_planner(&spec, &shapes, &default_hints).expect("Planning succeeded");

    println!("  Estimated FLOPs: {:.2e}", default_plan.estimated_flops);
    println!(
        "  Peak memory: {:.2} MB",
        default_plan.estimated_memory as f64 / 1_000_000.0
    );
    println!();

    // Example 2: Memory-constrained planning
    println!("2. MEMORY-CONSTRAINED PLANNING");
    println!("-------------------------------");
    let memory_hints = PlanHints {
        minimize_memory: true,
        memory_budget: Some(500_000_000), // 500 MB budget
        ..Default::default()
    };

    let memory_plan = greedy_planner(&spec, &shapes, &memory_hints).expect("Planning succeeded");

    println!("  Memory budget: 500 MB");
    println!("  Estimated FLOPs: {:.2e}", memory_plan.estimated_flops);
    println!(
        "  Peak memory: {:.2} MB",
        memory_plan.estimated_memory as f64 / 1_000_000.0
    );
    println!();

    // Example 3: Sparse tensor hints
    println!("3. SPARSE TENSOR HINTS");
    println!("----------------------");
    let mut sparse_hints = PlanHints::default();

    // Indicate that input 0 and 1 are sparse (5% density)
    let mut sparsity_map = HashMap::new();
    sparsity_map.insert(0, 0.05);
    sparsity_map.insert(1, 0.05);
    sparse_hints.sparsity_hints = sparsity_map;

    let sparse_plan = greedy_planner(&spec, &shapes, &sparse_hints).expect("Planning succeeded");

    println!("  Sparsity: inputs 0,1 are 5% dense");
    println!("  Estimated FLOPs: {:.2e}", sparse_plan.estimated_flops);
    println!(
        "  Peak memory: {:.2} MB",
        sparse_plan.estimated_memory as f64 / 1_000_000.0
    );
    println!();

    // Example 4: Device targeting
    println!("4. DEVICE TARGETING");
    println!("-------------------");
    let cpu_hints = PlanHints {
        device: Device::Cpu,
        ..Default::default()
    };

    let cpu_plan = greedy_planner(&spec, &shapes, &cpu_hints).expect("Planning succeeded");

    println!("  Target device: CPU");
    println!("  Estimated FLOPs: {:.2e}", cpu_plan.estimated_flops);
    println!(
        "  Peak memory: {:.2} MB",
        cpu_plan.estimated_memory as f64 / 1_000_000.0
    );
    println!();

    // Summary
    println!("SUMMARY");
    println!("-------");
    println!("Hints allow you to:");
    println!("  - Control memory usage with budgets");
    println!("  - Optimize for sparse tensors");
    println!("  - Target specific devices (CPU/GPU)");
    println!("  - Enable out-of-core execution for large tensors");
}
