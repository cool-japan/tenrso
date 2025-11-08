//! Deterministic chunk graphs for streaming execution
//!
//! This module provides a graph-based representation of tensor chunk operations,
//! enabling deterministic, memory-efficient execution of large tensor contractions.
//!
//! # Features
//!
//! - Deterministic chunk execution order
//! - Dependency tracking between chunks
//! - Memory requirement estimation
//! - Automatic scheduling with memory constraints
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::chunk_graph::{ChunkGraph, ChunkNode, ChunkOp};
//!
//! let mut graph = ChunkGraph::new();
//!
//! // Add nodes for input chunks
//! let n0 = graph.add_node(ChunkNode::input("A", vec![0, 0]));
//! let n1 = graph.add_node(ChunkNode::input("B", vec![0, 0]));
//!
//! // Add operation node
//! let n2 = graph.add_node(ChunkNode::operation(ChunkOp::MatMul, vec![n0, n1]));
//!
//! // Get execution order
//! let order = graph.topological_order()?;
//! ```

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::chunking::ChunkIndex;

/// Types of chunk operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChunkOp {
    /// Input chunk (no dependencies)
    Input,
    /// Matrix multiplication
    MatMul,
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Multiply,
    /// Contraction (general einsum)
    Contract { spec: String },
    /// Accumulation (reduce across chunks)
    Accumulate,
}

/// Node in the chunk graph
#[derive(Debug, Clone)]
pub struct ChunkNode {
    /// Node ID
    pub id: usize,
    /// Operation type
    pub op: ChunkOp,
    /// Input node IDs (dependencies)
    pub inputs: Vec<usize>,
    /// Tensor name (for input nodes)
    pub tensor_name: Option<String>,
    /// Chunk index (for input nodes)
    pub chunk_idx: Option<ChunkIndex>,
    /// Estimated memory requirement in bytes
    pub memory_bytes: usize,
    /// Estimated computation cost (flops)
    pub compute_cost: u64,
}

impl ChunkNode {
    /// Create an input chunk node
    pub fn input(tensor_name: &str, chunk_coords: Vec<usize>) -> Self {
        Self {
            id: 0, // Will be set by ChunkGraph
            op: ChunkOp::Input,
            inputs: vec![],
            tensor_name: Some(tensor_name.to_string()),
            chunk_idx: Some(ChunkIndex::new(chunk_coords)),
            memory_bytes: 0,
            compute_cost: 0,
        }
    }

    /// Create an operation node
    pub fn operation(op: ChunkOp, inputs: Vec<usize>) -> Self {
        Self {
            id: 0, // Will be set by ChunkGraph
            op,
            inputs,
            tensor_name: None,
            chunk_idx: None,
            memory_bytes: 0,
            compute_cost: 0,
        }
    }

    /// Set memory requirement
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Set computation cost
    pub fn with_compute_cost(mut self, flops: u64) -> Self {
        self.compute_cost = flops;
        self
    }
}

/// Deterministic chunk execution graph
#[derive(Debug, Clone)]
pub struct ChunkGraph {
    /// Nodes in the graph
    nodes: Vec<ChunkNode>,
    /// Adjacency list (node_id -> dependent_node_ids)
    dependents: HashMap<usize, Vec<usize>>,
    /// Next node ID
    next_id: usize,
}

impl ChunkGraph {
    /// Create a new empty chunk graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dependents: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph
    ///
    /// # Arguments
    ///
    /// * `mut node` - The node to add (id will be assigned automatically)
    ///
    /// # Returns
    ///
    /// The assigned node ID
    pub fn add_node(&mut self, mut node: ChunkNode) -> usize {
        let id = self.next_id;
        node.id = id;
        self.next_id += 1;

        // Build dependency edges
        for &input_id in &node.inputs {
            self.dependents.entry(input_id).or_default().push(id);
        }

        self.nodes.push(node);
        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: usize) -> Option<&ChunkNode> {
        self.nodes.get(id)
    }

    /// Get number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all input nodes
    pub fn input_nodes(&self) -> Vec<&ChunkNode> {
        self.nodes
            .iter()
            .filter(|n| n.op == ChunkOp::Input)
            .collect()
    }

    /// Get all output nodes (nodes with no dependents)
    pub fn output_nodes(&self) -> Vec<&ChunkNode> {
        self.nodes
            .iter()
            .filter(|n| !self.dependents.contains_key(&n.id))
            .collect()
    }

    /// Compute topological order for execution
    ///
    /// Uses Kahn's algorithm for deterministic ordering.
    ///
    /// # Returns
    ///
    /// Ordered list of node IDs for execution
    ///
    /// # Errors
    ///
    /// Returns an error if the graph has cycles
    pub fn topological_order(&self) -> Result<Vec<usize>> {
        // Build in-degree map
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.entry(node.id).or_insert(0);
            for &input_id in &node.inputs {
                let _ = in_degree.entry(input_id).or_insert(0);
            }
        }

        for node in &self.nodes {
            for &_input_id in &node.inputs {
                *in_degree.get_mut(&node.id).unwrap() += 1;
            }
        }

        // Initialize queue with nodes having in-degree 0
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Sort queue for determinism
        let mut queue_vec: Vec<usize> = queue.into_iter().collect();
        queue_vec.sort_unstable();
        queue = queue_vec.into();

        let mut order = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            order.push(node_id);

            // Decrease in-degree of dependents
            if let Some(deps) = self.dependents.get(&node_id) {
                let mut new_ready = Vec::new();
                for &dep_id in deps {
                    let deg = in_degree.get_mut(&dep_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        new_ready.push(dep_id);
                    }
                }
                // Sort for determinism
                new_ready.sort_unstable();
                queue.extend(new_ready);
            }
        }

        // Check for cycles
        if order.len() != self.nodes.len() {
            return Err(anyhow!(
                "Graph has cycles: {} nodes, {} in order",
                self.nodes.len(),
                order.len()
            ));
        }

        Ok(order)
    }

    /// Compute memory-constrained execution schedule
    ///
    /// Returns a schedule that respects the memory limit by determining
    /// which chunks should be kept in memory vs spilled to disk.
    ///
    /// # Arguments
    ///
    /// * `memory_limit_bytes` - Maximum memory available
    ///
    /// # Returns
    ///
    /// Tuple of (execution_order, nodes_to_keep_in_memory, nodes_to_spill)
    pub fn memory_constrained_schedule(
        &self,
        memory_limit_bytes: usize,
    ) -> Result<(Vec<usize>, HashSet<usize>, HashSet<usize>)> {
        let order = self.topological_order()?;

        let mut keep_in_memory = HashSet::new();
        let mut to_spill = HashSet::new();
        let mut current_memory: usize = 0;

        // Track which nodes are still needed
        let mut ref_count: HashMap<usize, usize> = HashMap::new();
        for node in &self.nodes {
            ref_count.insert(node.id, 0);
            for &input_id in &node.inputs {
                *ref_count.get_mut(&input_id).unwrap() += 1;
            }
        }

        for &node_id in &order {
            let node = &self.nodes[node_id];

            // Add current node's memory requirement
            current_memory += node.memory_bytes;

            // Check if we exceed memory limit
            if current_memory > memory_limit_bytes {
                // Find nodes to spill (least recently used with no future refs)
                let mut candidates: Vec<usize> = keep_in_memory
                    .iter()
                    .filter(|&&id| ref_count.get(&id).is_some_and(|&rc| rc == 0))
                    .copied()
                    .collect();

                candidates.sort_by_key(|&id| {
                    // Prefer spilling nodes with larger memory
                    std::cmp::Reverse(self.nodes[id].memory_bytes)
                });

                // Spill nodes until we're under the limit
                for candidate_id in candidates {
                    if current_memory <= memory_limit_bytes {
                        break;
                    }

                    let candidate_node = &self.nodes[candidate_id];
                    current_memory = current_memory.saturating_sub(candidate_node.memory_bytes);
                    keep_in_memory.remove(&candidate_id);
                    to_spill.insert(candidate_id);
                }
            }

            keep_in_memory.insert(node_id);

            // Decrement reference counts for inputs
            for &input_id in &node.inputs {
                if let Some(count) = ref_count.get_mut(&input_id) {
                    *count = count.saturating_sub(1);
                }
            }
        }

        Ok((order, keep_in_memory, to_spill))
    }

    /// Estimate total memory requirement
    pub fn total_memory(&self) -> usize {
        self.nodes.iter().map(|n| n.memory_bytes).sum()
    }

    /// Estimate total computation cost
    pub fn total_compute_cost(&self) -> u64 {
        self.nodes.iter().map(|n| n.compute_cost).sum()
    }

    /// Get critical path (longest path in terms of compute cost)
    pub fn critical_path(&self) -> Result<Vec<usize>> {
        let order = self.topological_order()?;

        // Compute longest path using dynamic programming
        let mut longest_path: HashMap<usize, u64> = HashMap::new();
        let mut predecessors: HashMap<usize, Option<usize>> = HashMap::new();

        for &node_id in &order {
            let node = &self.nodes[node_id];

            // Find max path length from inputs
            let max_input_path = node
                .inputs
                .iter()
                .filter_map(|&id| longest_path.get(&id))
                .max()
                .copied()
                .unwrap_or(0);

            let path_length = max_input_path + node.compute_cost;
            longest_path.insert(node_id, path_length);

            // Track predecessor
            let pred = node
                .inputs
                .iter()
                .max_by_key(|&&id| longest_path.get(&id).unwrap_or(&0))
                .copied();
            predecessors.insert(node_id, pred);
        }

        // Find node with maximum path length
        let (&end_node, _) = longest_path
            .iter()
            .max_by_key(|(_, &len)| len)
            .ok_or_else(|| anyhow!("Empty graph"))?;

        // Backtrack to construct path
        let mut path = Vec::new();
        let mut current = Some(end_node);

        while let Some(node_id) = current {
            path.push(node_id);
            current = predecessors.get(&node_id).and_then(|&p| p);
        }

        path.reverse();
        Ok(path)
    }
}

impl Default for ChunkGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_graph_creation() {
        let graph = ChunkGraph::new();
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = ChunkGraph::new();

        let n0 = graph.add_node(ChunkNode::input("A", vec![0, 0]));
        let n1 = graph.add_node(ChunkNode::input("B", vec![0, 0]));
        let n2 = graph.add_node(ChunkNode::operation(ChunkOp::MatMul, vec![n0, n1]));

        assert_eq!(graph.len(), 3);
        assert_eq!(n0, 0);
        assert_eq!(n1, 1);
        assert_eq!(n2, 2);
    }

    #[test]
    fn test_input_output_nodes() {
        let mut graph = ChunkGraph::new();

        let n0 = graph.add_node(ChunkNode::input("A", vec![0, 0]));
        let n1 = graph.add_node(ChunkNode::input("B", vec![0, 0]));
        let n2 = graph.add_node(ChunkNode::operation(ChunkOp::MatMul, vec![n0, n1]));

        let inputs = graph.input_nodes();
        assert_eq!(inputs.len(), 2);

        let outputs = graph.output_nodes();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].id, n2);
    }

    #[test]
    fn test_topological_order_simple() {
        let mut graph = ChunkGraph::new();

        let n0 = graph.add_node(ChunkNode::input("A", vec![0, 0]));
        let n1 = graph.add_node(ChunkNode::input("B", vec![0, 0]));
        let n2 = graph.add_node(ChunkNode::operation(ChunkOp::MatMul, vec![n0, n1]));

        let order = graph.topological_order().unwrap();

        assert_eq!(order.len(), 3);
        // Inputs should come before MatMul
        assert!(
            order.iter().position(|&x| x == n0).unwrap()
                < order.iter().position(|&x| x == n2).unwrap()
        );
        assert!(
            order.iter().position(|&x| x == n1).unwrap()
                < order.iter().position(|&x| x == n2).unwrap()
        );
    }

    #[test]
    fn test_topological_order_complex() {
        let mut graph = ChunkGraph::new();

        // Build a DAG: a0, a1 -> add -> mul <- b0
        let a0 = graph.add_node(ChunkNode::input("A", vec![0]));
        let a1 = graph.add_node(ChunkNode::input("A", vec![1]));
        let b0 = graph.add_node(ChunkNode::input("B", vec![0]));

        let add_node = graph.add_node(ChunkNode::operation(ChunkOp::Add, vec![a0, a1]));
        let mul_node = graph.add_node(ChunkNode::operation(ChunkOp::Multiply, vec![add_node, b0]));

        let order = graph.topological_order().unwrap();

        assert_eq!(order.len(), 5);

        // add_node must come after a0 and a1
        let add_pos = order.iter().position(|&x| x == add_node).unwrap();
        let a0_pos = order.iter().position(|&x| x == a0).unwrap();
        let a1_pos = order.iter().position(|&x| x == a1).unwrap();
        assert!(a0_pos < add_pos);
        assert!(a1_pos < add_pos);

        // mul_node must come after add_node and b0
        let mul_pos = order.iter().position(|&x| x == mul_node).unwrap();
        let b0_pos = order.iter().position(|&x| x == b0).unwrap();
        assert!(add_pos < mul_pos);
        assert!(b0_pos < mul_pos);
    }

    #[test]
    fn test_memory_constrained_schedule() {
        let mut graph = ChunkGraph::new();

        let n0 = graph.add_node(ChunkNode::input("A", vec![0, 0]).with_memory(1000));
        let n1 = graph.add_node(ChunkNode::input("B", vec![0, 0]).with_memory(1000));
        let _n2 =
            graph.add_node(ChunkNode::operation(ChunkOp::MatMul, vec![n0, n1]).with_memory(500));

        // Tight memory limit: can only fit 2 nodes at a time
        let (order, _keep, _spill) = graph.memory_constrained_schedule(2000).unwrap();

        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_total_memory() {
        let mut graph = ChunkGraph::new();

        graph.add_node(ChunkNode::input("A", vec![0]).with_memory(1000));
        graph.add_node(ChunkNode::input("B", vec![0]).with_memory(2000));
        graph.add_node(ChunkNode::input("C", vec![0]).with_memory(3000));

        assert_eq!(graph.total_memory(), 6000);
    }

    #[test]
    fn test_total_compute_cost() {
        let mut graph = ChunkGraph::new();

        graph.add_node(ChunkNode::input("A", vec![0]).with_compute_cost(100));
        graph.add_node(ChunkNode::input("B", vec![0]).with_compute_cost(200));

        assert_eq!(graph.total_compute_cost(), 300);
    }

    #[test]
    fn test_critical_path() {
        let mut graph = ChunkGraph::new();

        // Build a graph with multiple paths
        let a = graph.add_node(ChunkNode::input("A", vec![0]).with_compute_cost(10));
        let b = graph.add_node(ChunkNode::input("B", vec![0]).with_compute_cost(5));
        let c = graph.add_node(ChunkNode::operation(ChunkOp::Add, vec![a]).with_compute_cost(20));
        let d = graph
            .add_node(ChunkNode::operation(ChunkOp::Multiply, vec![b, c]).with_compute_cost(15));

        let path = graph.critical_path().unwrap();

        // Critical path should be: a -> c -> d (total: 10 + 20 + 15 = 45)
        assert!(path.contains(&a));
        assert!(path.contains(&c));
        assert!(path.contains(&d));
    }

    #[test]
    fn test_deterministic_ordering() {
        // Run topological sort multiple times to ensure determinism
        let mut graph = ChunkGraph::new();

        let n0 = graph.add_node(ChunkNode::input("A", vec![0]));
        let n1 = graph.add_node(ChunkNode::input("B", vec![0]));
        let n2 = graph.add_node(ChunkNode::input("C", vec![0]));
        let n3 = graph.add_node(ChunkNode::operation(ChunkOp::Add, vec![n0, n1]));
        let _n4 = graph.add_node(ChunkNode::operation(ChunkOp::Multiply, vec![n2, n3]));

        let order1 = graph.topological_order().unwrap();
        let order2 = graph.topological_order().unwrap();
        let order3 = graph.topological_order().unwrap();

        assert_eq!(order1, order2);
        assert_eq!(order2, order3);
    }
}
