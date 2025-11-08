//! Chunk graph execution example
//!
//! Demonstrates how to build and execute a chunk graph for deterministic streaming.

use anyhow::Result;
use tenrso_ooc::{ChunkGraph, ChunkNode, ChunkOp};

fn main() -> Result<()> {
    println!("=== Chunk Graph Execution Example ===\n");

    // Build a computation graph
    let mut graph = ChunkGraph::new();

    println!("Building computation graph:");
    println!("  Graph: (A + B) * C\n");

    // Add input nodes
    let a_node = graph.add_node(
        ChunkNode::input("A", vec![0, 0])
            .with_memory(1000)
            .with_compute_cost(0),
    );
    println!("  Added input A (node {})", a_node);

    let b_node = graph.add_node(
        ChunkNode::input("B", vec![0, 0])
            .with_memory(1000)
            .with_compute_cost(0),
    );
    println!("  Added input B (node {})", b_node);

    let c_node = graph.add_node(
        ChunkNode::input("C", vec![0, 0])
            .with_memory(1000)
            .with_compute_cost(0),
    );
    println!("  Added input C (node {})", c_node);

    // Add operation: A + B
    let add_node = graph.add_node(
        ChunkNode::operation(ChunkOp::Add, vec![a_node, b_node])
            .with_memory(1000)
            .with_compute_cost(1000),
    );
    println!("  Added A + B (node {})", add_node);

    // Add operation: (A + B) * C
    let mul_node = graph.add_node(
        ChunkNode::operation(ChunkOp::Multiply, vec![add_node, c_node])
            .with_memory(1000)
            .with_compute_cost(1000),
    );
    println!("  Added (A+B) * C (node {})", mul_node);

    // Compute execution order
    println!("\nComputing execution order...");
    let order = graph.topological_order()?;
    println!("  Execution order: {:?}", order);

    // Memory-constrained schedule
    println!("\nComputing memory-constrained schedule...");
    let memory_limit = 3000; // Can hold 3 chunks at a time
    let (exec_order, keep_in_memory, to_spill) = graph.memory_constrained_schedule(memory_limit)?;

    println!("  Memory limit: {} bytes", memory_limit);
    println!("  Execution order: {:?}", exec_order);
    println!("  Keep in memory: {} chunks", keep_in_memory.len());
    println!("  To spill: {} chunks", to_spill.len());

    // Critical path analysis
    println!("\nCritical path analysis...");
    let critical_path = graph.critical_path()?;
    println!("  Critical path: {:?}", critical_path);
    println!("  Path length: {} nodes", critical_path.len());

    // Graph statistics
    println!("\nGraph statistics:");
    println!("  Total nodes: {}", graph.len());
    println!("  Input nodes: {}", graph.input_nodes().len());
    println!("  Output nodes: {}", graph.output_nodes().len());
    println!("  Total memory: {} bytes", graph.total_memory());
    println!("  Total compute cost: {} FLOPs", graph.total_compute_cost());

    println!("\n✓ Chunk graph execution completed successfully!");

    Ok(())
}
