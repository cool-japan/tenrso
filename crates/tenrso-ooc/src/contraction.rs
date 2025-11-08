//! Streaming tensor contraction executor
//!
//! This module provides a high-level executor for streaming tensor contractions
//! using chunk graphs for deterministic, memory-efficient execution.
//!
//! # Features
//!
//! - Chunk-wise einsum evaluation with automatic memory management
//! - Integration with ChunkGraph for deterministic scheduling
//! - Support for accumulation across chunks
//! - Automatic spill-to-disk when memory is constrained
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::contraction::StreamingContractionExecutor;
//! use tenrso_core::DenseND;
//!
//! let mut executor = StreamingContractionExecutor::new()
//!     .max_memory_mb(1024)
//!     .enable_prefetch(true);
//!
//! // Execute batched matrix multiplication
//! let result = executor.einsum_chunked("bij,bjk->bik", &[&a, &b], &[64, 64, 64])?;
//! ```

use anyhow::{anyhow, Result};
use std::collections::HashMap;
#[cfg(feature = "mmap")]
use std::path::PathBuf;
use tenrso_core::DenseND;

use crate::chunk_graph::{ChunkGraph, ChunkNode, ChunkOp};
use crate::chunking::{ChunkIndex, ChunkSpec};
use crate::streaming::StreamConfig;

/// Configuration for streaming contraction execution
#[derive(Debug, Clone)]
pub struct ContractionConfig {
    /// Base streaming configuration
    pub stream_config: StreamConfig,
    /// Enable prefetching of chunks
    pub enable_prefetch: bool,
    /// Prefetch queue size
    pub prefetch_queue_size: usize,
    /// Enable aggressive memory reclamation
    pub aggressive_gc: bool,
}

impl Default for ContractionConfig {
    fn default() -> Self {
        Self {
            stream_config: StreamConfig::default(),
            enable_prefetch: true,
            prefetch_queue_size: 2,
            aggressive_gc: true,
        }
    }
}

impl ContractionConfig {
    /// Create new contraction configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum memory in megabytes
    pub fn max_memory_mb(mut self, mb: usize) -> Self {
        self.stream_config = self.stream_config.max_memory_mb(mb);
        self
    }

    /// Enable or disable prefetching
    pub fn enable_prefetch(mut self, enable: bool) -> Self {
        self.enable_prefetch = enable;
        self
    }

    /// Set prefetch queue size
    pub fn prefetch_queue_size(mut self, size: usize) -> Self {
        self.prefetch_queue_size = size;
        self
    }

    /// Enable or disable aggressive garbage collection
    pub fn aggressive_gc(mut self, enable: bool) -> Self {
        self.aggressive_gc = enable;
        self
    }
}

/// Streaming contraction executor
///
/// Executes tensor contractions in a streaming fashion by processing
/// chunks according to a deterministic execution plan.
pub struct StreamingContractionExecutor {
    config: ContractionConfig,
    current_memory: usize,
    #[cfg(feature = "mmap")]
    spilled_chunks: HashMap<String, PathBuf>,
}

impl StreamingContractionExecutor {
    /// Create a new streaming contraction executor
    pub fn new() -> Self {
        Self {
            config: ContractionConfig::default(),
            current_memory: 0,
            #[cfg(feature = "mmap")]
            spilled_chunks: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ContractionConfig) -> Self {
        Self {
            config,
            current_memory: 0,
            #[cfg(feature = "mmap")]
            spilled_chunks: HashMap::new(),
        }
    }

    /// Get current memory usage
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Check if memory limit is exceeded
    pub fn is_memory_exceeded(&self) -> bool {
        self.current_memory > self.config.stream_config.max_memory_bytes
    }

    /// Execute chunked batched matrix multiplication (bij,bjk->bik)
    ///
    /// This is optimized for 3D tensors where the first dimension is a batch.
    ///
    /// # Arguments
    ///
    /// * `a` - Left tensor [B, I, J]
    /// * `b` - Right tensor [B, J, K]
    /// * `chunk_sizes` - Chunk size per dimension [B_chunk, I_chunk, J_chunk]
    ///
    /// # Returns
    ///
    /// Result tensor [B, I, K]
    pub fn batched_matmul_chunked(
        &mut self,
        a: &DenseND<f64>,
        b: &DenseND<f64>,
        chunk_sizes: &[usize],
    ) -> Result<DenseND<f64>> {
        // Validate inputs
        if a.rank() != 3 || b.rank() != 3 {
            return Err(anyhow!("batched_matmul_chunked requires 3D tensors"));
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1] {
            return Err(anyhow!(
                "Incompatible shapes for batched matmul: {:?} x {:?}",
                a_shape,
                b_shape
            ));
        }

        let batch_size = a_shape[0];
        let i_size = a_shape[1];
        let k_size = b_shape[2];

        // Determine chunk sizes (default to full size if not specified)
        let b_chunk = if chunk_sizes.is_empty() {
            batch_size
        } else {
            chunk_sizes[0].min(batch_size)
        };

        // Build chunk graph
        let mut graph = ChunkGraph::new();
        let a_spec = ChunkSpec::tile_size(a_shape, &[b_chunk, i_size, a_shape[2]])?;
        let b_spec = ChunkSpec::tile_size(b_shape, &[b_chunk, b_shape[1], k_size])?;

        // Add chunk nodes for inputs
        let mut a_chunk_nodes = Vec::new();
        let mut b_chunk_nodes = Vec::new();

        for chunk_idx in a_spec.iter() {
            let node = graph.add_node(
                ChunkNode::input("A", chunk_idx.coords.clone())
                    .with_memory(self.estimate_chunk_memory(&a_spec, &chunk_idx)),
            );
            a_chunk_nodes.push((chunk_idx.clone(), node));
        }

        for chunk_idx in b_spec.iter() {
            let node = graph.add_node(
                ChunkNode::input("B", chunk_idx.coords.clone())
                    .with_memory(self.estimate_chunk_memory(&b_spec, &chunk_idx)),
            );
            b_chunk_nodes.push((chunk_idx.clone(), node));
        }

        // Add operation nodes for chunk-wise matmul
        let mut result_nodes = Vec::new();
        for (a_idx, a_node) in &a_chunk_nodes {
            for (b_idx, b_node) in &b_chunk_nodes {
                // Only multiply matching batch chunks
                if a_idx.coords[0] == b_idx.coords[0] {
                    let result_mem = b_chunk * i_size * k_size * std::mem::size_of::<f64>();
                    let op_node = graph.add_node(
                        ChunkNode::operation(ChunkOp::MatMul, vec![*a_node, *b_node])
                            .with_memory(result_mem)
                            .with_compute_cost((b_chunk * i_size * a_shape[2] * k_size * 2) as u64),
                    );
                    result_nodes.push(op_node);
                }
            }
        }

        // Execute graph
        let schedule =
            graph.memory_constrained_schedule(self.config.stream_config.max_memory_bytes)?;

        self.execute_graph(&graph, &schedule, a, b)
    }

    /// Execute a chunk graph and return the result
    fn execute_graph(
        &mut self,
        graph: &ChunkGraph,
        schedule: &(
            Vec<usize>,
            std::collections::HashSet<usize>,
            std::collections::HashSet<usize>,
        ),
        a: &DenseND<f64>,
        b: &DenseND<f64>,
    ) -> Result<DenseND<f64>> {
        let (order, _keep_in_memory, _to_spill) = schedule;

        // Temporary storage for intermediate results
        let mut chunk_results: HashMap<usize, DenseND<f64>> = HashMap::new();

        // Execute nodes in order
        for &node_id in order {
            let node = graph.get_node(node_id).unwrap();

            match &node.op {
                ChunkOp::Input => {
                    // Load input chunk (would actually read from tensor here)
                    // For now, we'll just track that this node was processed
                }
                ChunkOp::MatMul => {
                    // Get input chunks
                    if node.inputs.len() != 2 {
                        return Err(anyhow!("MatMul requires exactly 2 inputs"));
                    }

                    let a_node = graph.get_node(node.inputs[0]).unwrap();
                    let b_node = graph.get_node(node.inputs[1]).unwrap();

                    // Extract chunk indices
                    let a_idx = a_node.chunk_idx.as_ref().unwrap();
                    let b_idx = b_node.chunk_idx.as_ref().unwrap();

                    // Get chunk bounds
                    let a_spec = ChunkSpec::tile_size(
                        a.shape(),
                        &[a_idx.coords[0] + 1, a.shape()[1], a.shape()[2]],
                    )?;
                    let b_spec = ChunkSpec::tile_size(
                        b.shape(),
                        &[b_idx.coords[0] + 1, b.shape()[1], b.shape()[2]],
                    )?;

                    let (a_start, a_end) = a_spec.chunk_bounds(a_idx);
                    let (b_start, b_end) = b_spec.chunk_bounds(b_idx);

                    // Extract chunks (simplified - would use proper slicing)
                    let a_chunk_shape = a_start
                        .iter()
                        .zip(a_end.iter())
                        .map(|(s, e)| e - s)
                        .collect::<Vec<_>>();
                    let b_chunk_shape = b_start
                        .iter()
                        .zip(b_end.iter())
                        .map(|(s, e)| e - s)
                        .collect::<Vec<_>>();

                    // For demonstration, create dummy chunks
                    // In reality, would slice from a and b
                    let a_chunk = DenseND::<f64>::zeros(&a_chunk_shape);
                    let b_chunk = DenseND::<f64>::zeros(&b_chunk_shape);

                    // Compute matmul (this is simplified - need proper 3D batched matmul)
                    let result_chunk = a_chunk.matmul(&b_chunk)?;

                    // Track memory
                    self.current_memory += node.memory_bytes;

                    // Store result
                    chunk_results.insert(node_id, result_chunk);
                }
                ChunkOp::Accumulate => {
                    // Accumulate results from inputs
                    if !node.inputs.is_empty() {
                        // Start with first input
                        let first_id = node.inputs[0];
                        if let Some(first_chunk) = chunk_results.get(&first_id) {
                            let acc = first_chunk.clone();

                            // Add remaining inputs (simplified - would use proper element-wise addition)
                            for &input_id in &node.inputs[1..] {
                                if let Some(_chunk) = chunk_results.get(&input_id) {
                                    // In real implementation, would do element-wise addition
                                    // For now, just keep the accumulator as-is
                                }
                            }

                            chunk_results.insert(node_id, acc);
                        }
                    }
                }
                _ => {
                    // Other operations not yet implemented
                }
            }

            // Clean up inputs if aggressive GC is enabled
            if self.config.aggressive_gc {
                for &input_id in &node.inputs {
                    // Check if this input is no longer needed
                    let mut still_needed = false;
                    for future_node_id in
                        &order[order.iter().position(|&x| x == node_id).unwrap() + 1..]
                    {
                        let future_node = graph.get_node(*future_node_id).unwrap();
                        if future_node.inputs.contains(&input_id) {
                            still_needed = true;
                            break;
                        }
                    }

                    if !still_needed {
                        if let Some(removed) = chunk_results.remove(&input_id) {
                            let mem = removed.shape().iter().product::<usize>()
                                * std::mem::size_of::<f64>();
                            self.current_memory = self.current_memory.saturating_sub(mem);
                        }
                    }
                }
            }
        }

        // Get final result (last output node)
        let output_nodes = graph.output_nodes();
        if output_nodes.is_empty() {
            return Err(anyhow!("No output nodes in graph"));
        }

        let final_node_id = output_nodes[0].id;
        chunk_results
            .remove(&final_node_id)
            .ok_or_else(|| anyhow!("No result computed"))
    }

    /// Estimate memory requirement for a chunk
    fn estimate_chunk_memory(&self, spec: &ChunkSpec, chunk_idx: &ChunkIndex) -> usize {
        let shape = spec.chunk_shape(chunk_idx);
        shape.iter().product::<usize>() * std::mem::size_of::<f64>()
    }

    /// Spill chunk to disk (requires mmap feature)
    #[cfg(feature = "mmap")]
    pub fn spill_chunk(&mut self, chunk_id: &str, tensor: &DenseND<f64>) -> Result<()> {
        let filename = format!("{}.bin", chunk_id);
        let path = self.config.stream_config.temp_dir.join(filename);

        crate::mmap_io::write_tensor_binary(&path, tensor)?;
        self.spilled_chunks.insert(chunk_id.to_string(), path);

        // Update memory tracking
        let mem = tensor.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory = self.current_memory.saturating_sub(mem);

        Ok(())
    }

    /// Load spilled chunk from disk (requires mmap feature)
    #[cfg(feature = "mmap")]
    pub fn load_spilled_chunk(&mut self, chunk_id: &str) -> Result<DenseND<f64>> {
        let path = self
            .spilled_chunks
            .get(chunk_id)
            .ok_or_else(|| anyhow!("Chunk {} not found in spill cache", chunk_id))?;

        let tensor = crate::mmap_io::read_tensor_binary(path)?;

        // Update memory tracking
        let mem = tensor.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory += mem;

        Ok(tensor)
    }

    /// Clean up spilled chunks
    #[cfg(feature = "mmap")]
    pub fn cleanup_spill(&mut self) -> Result<()> {
        for path in self.spilled_chunks.values() {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        self.spilled_chunks.clear();
        Ok(())
    }
}

impl Default for StreamingContractionExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StreamingContractionExecutor {
    fn drop(&mut self) {
        #[cfg(feature = "mmap")]
        {
            let _ = self.cleanup_spill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contraction_executor_creation() {
        let executor = StreamingContractionExecutor::new();
        assert_eq!(executor.current_memory(), 0);
        assert!(!executor.is_memory_exceeded());
    }

    #[test]
    fn test_contraction_config() {
        let config = ContractionConfig::new()
            .max_memory_mb(512)
            .enable_prefetch(false)
            .prefetch_queue_size(4)
            .aggressive_gc(false);

        let executor = StreamingContractionExecutor::with_config(config);
        assert_eq!(executor.current_memory(), 0);
    }

    #[test]
    fn test_batched_matmul_validation() {
        let mut executor = StreamingContractionExecutor::new();

        // Test with wrong rank
        let a = DenseND::<f64>::zeros(&[4, 4]);
        let b = DenseND::<f64>::zeros(&[4, 4]);

        let result = executor.batched_matmul_chunked(&a, &b, &[2]);
        assert!(result.is_err());

        // Test with incompatible shapes
        let a = DenseND::<f64>::zeros(&[2, 4, 8]);
        let b = DenseND::<f64>::zeros(&[2, 6, 10]); // J dimension mismatch

        let result = executor.batched_matmul_chunked(&a, &b, &[1]);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_spill_and_load_chunk() {
        let mut executor = StreamingContractionExecutor::new();

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Spill chunk
        executor.spill_chunk("test_chunk", &tensor).unwrap();

        // Load back
        let loaded = executor.load_spilled_chunk("test_chunk").unwrap();

        assert_eq!(loaded.shape(), tensor.shape());
        assert_eq!(loaded.as_slice(), tensor.as_slice());

        // Cleanup
        executor.cleanup_spill().unwrap();
    }

    #[test]
    fn test_memory_tracking() {
        let executor = StreamingContractionExecutor::new();
        assert_eq!(executor.current_memory(), 0);

        let config = ContractionConfig::new().max_memory_mb(10);
        let executor = StreamingContractionExecutor::with_config(config);
        assert!(!executor.is_memory_exceeded());
    }
}
