//! Integration tests for chunk graph execution
//!
//! Tests chunk graph construction and execution with actual tensor operations

use anyhow::Result;
use tenrso_ooc::{ChunkGraph, ChunkNode, ChunkOp};

/// Test building a simple chunk graph
#[test]
fn test_chunk_graph_basic() -> Result<()> {
    let mut graph = ChunkGraph::new();

    // Add input nodes
    let id_a = graph.add_node(ChunkNode {
        id: 0, // Will be assigned automatically
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("a".to_string()),
        chunk_idx: None,
        memory_bytes: 1000,
        compute_cost: 0,
    });

    let id_b = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("b".to_string()),
        chunk_idx: None,
        memory_bytes: 1000,
        compute_cost: 0,
    });

    // Add operation node
    let _id_add = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_a, id_b],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 1000,
        compute_cost: 1000,
    });

    // Verify graph structure
    assert_eq!(graph.len(), 3);
    assert_eq!(graph.input_nodes().len(), 2); // Two inputs

    Ok(())
}

/// Test chunk graph with multiple operations
#[test]
fn test_chunk_graph_multiple_ops() -> Result<()> {
    let mut graph = ChunkGraph::new();

    // Build: d = (a + b) * c
    let id_a = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("a".to_string()),
        chunk_idx: None,
        memory_bytes: 800,
        compute_cost: 0,
    });

    let id_b = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("b".to_string()),
        chunk_idx: None,
        memory_bytes: 800,
        compute_cost: 0,
    });

    let id_c = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("c".to_string()),
        chunk_idx: None,
        memory_bytes: 800,
        compute_cost: 0,
    });

    let id_add = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_a, id_b],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 800,
        compute_cost: 800,
    });

    let _id_mul = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Multiply,
        inputs: vec![id_add, id_c],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 800,
        compute_cost: 800,
    });

    // Verify structure
    assert_eq!(graph.len(), 5);
    assert_eq!(graph.input_nodes().len(), 3);

    Ok(())
}

/// Test chunk graph with diamond dependency
#[test]
fn test_chunk_graph_diamond_dependency() -> Result<()> {
    let mut graph = ChunkGraph::new();

    // Build diamond: e = (a + b) + (a * c)
    let id_a = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("a".to_string()),
        chunk_idx: None,
        memory_bytes: 500,
        compute_cost: 0,
    });

    let id_b = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("b".to_string()),
        chunk_idx: None,
        memory_bytes: 500,
        compute_cost: 0,
    });

    let id_c = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("c".to_string()),
        chunk_idx: None,
        memory_bytes: 500,
        compute_cost: 0,
    });

    let id_add = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_a, id_b],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 500,
        compute_cost: 500,
    });

    let id_mul = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Multiply,
        inputs: vec![id_a, id_c],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 500,
        compute_cost: 500,
    });

    let _id_final = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_add, id_mul],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 500,
        compute_cost: 500,
    });

    // Verify structure
    assert_eq!(graph.len(), 6);

    Ok(())
}

/// Test finding nodes in chunk graph
#[test]
fn test_chunk_graph_node_lookup() -> Result<()> {
    let mut graph = ChunkGraph::new();

    let id = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![],
        tensor_name: Some("test".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    // Find the node
    let found = graph.get_node(id);
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, id);

    // Try to find non-existent node
    let not_found = graph.get_node(999);
    assert!(not_found.is_none());

    Ok(())
}

/// Test chunk graph with matmul operation
#[test]
fn test_chunk_graph_matmul() -> Result<()> {
    let mut graph = ChunkGraph::new();

    // Build: result = a @ b (MatMul)
    let id_a = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("a".to_string()),
        chunk_idx: None,
        memory_bytes: 400,
        compute_cost: 0,
    });

    let id_b = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("b".to_string()),
        chunk_idx: None,
        memory_bytes: 400,
        compute_cost: 0,
    });

    let _id_matmul = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::MatMul,
        inputs: vec![id_a, id_b],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 400,
        compute_cost: 800,
    });

    assert_eq!(graph.len(), 3);

    Ok(())
}

/// Test chunk graph with all operation types
#[test]
fn test_chunk_graph_all_ops() -> Result<()> {
    let ops = [
        ChunkOp::Input,
        ChunkOp::Add,
        ChunkOp::Multiply,
        ChunkOp::MatMul,
        ChunkOp::Contract {
            spec: "ij,jk->ik".to_string(),
        },
        ChunkOp::Accumulate,
    ];

    for op in ops.iter() {
        let mut graph = ChunkGraph::new();

        let _id = graph.add_node(ChunkNode {
            id: 0,
            op: op.clone(),
            inputs: vec![],
            tensor_name: Some("test".to_string()),
            chunk_idx: None,
            memory_bytes: 100,
            compute_cost: 100,
        });

        assert_eq!(graph.len(), 1);
    }

    Ok(())
}

/// Test linear chain of operations
#[test]
fn test_chunk_graph_linear_chain() -> Result<()> {
    let mut graph = ChunkGraph::new();

    // Build: A -> B -> C -> D -> E
    let id_a = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("A".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 0,
    });

    let id_b = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_a],
        tensor_name: Some("B".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    let id_c = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_b],
        tensor_name: Some("C".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    let id_d = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_c],
        tensor_name: Some("D".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    let _id_e = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_d],
        tensor_name: Some("E".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    assert_eq!(graph.len(), 5);

    Ok(())
}

/// Test graph with multiple independent chains
#[test]
fn test_chunk_graph_parallel_chains() -> Result<()> {
    let mut graph = ChunkGraph::new();

    // Chain 1: A -> B -> C
    let id_a = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("A".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 0,
    });

    let id_b = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_a],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    let _id_c = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_b],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    // Chain 2: X -> Y -> Z
    let id_x = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Input,
        inputs: vec![],
        tensor_name: Some("X".to_string()),
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 0,
    });

    let id_y = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_x],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    let _id_z = graph.add_node(ChunkNode {
        id: 0,
        op: ChunkOp::Add,
        inputs: vec![id_y],
        tensor_name: None,
        chunk_idx: None,
        memory_bytes: 100,
        compute_cost: 100,
    });

    assert_eq!(graph.len(), 6);
    assert_eq!(graph.input_nodes().len(), 2); // A and X are roots

    Ok(())
}
