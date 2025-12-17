//! Property-based tests for out-of-core tensor operations
//!
//! These tests use proptest to verify algebraic properties and correctness
//! of chunking, streaming execution, and memory management.

use proptest::prelude::*;
use scirs2_core::ndarray_ext::Array;
use tenrso_core::DenseND;
use tenrso_ooc::{
    ChunkGraph, ChunkNode, ChunkOp, ChunkSpec, PrefetchStrategy, Prefetcher, StreamConfig,
    StreamingExecutor,
};

// ============================================================================
// Test Utilities
// ============================================================================

/// Strategy for generating valid tensor shapes (2D for simplicity)
fn shape_strategy() -> impl Strategy<Value = (usize, usize)> {
    (2usize..20, 2usize..20)
}

/// Strategy for generating valid chunk sizes
fn chunk_size_strategy(max_dim: usize) -> impl Strategy<Value = usize> {
    1usize..=max_dim.min(10)
}

// ============================================================================
// Chunk Iteration Properties
// ============================================================================

proptest! {
    /// Property: ChunkIterator covers entire tensor exactly once
    #[test]
    fn prop_chunk_coverage(
        (rows, cols) in shape_strategy(),
        chunk_row in chunk_size_strategy(20),
        chunk_col in chunk_size_strategy(20),
    ) {
        let spec = ChunkSpec::tile_size(&[rows, cols], &[chunk_row, chunk_col]).unwrap();

        // Collect all chunks
        let chunks: Vec<_> = spec.iter().collect();

        // Verify each element is covered exactly once
        let mut covered = vec![vec![0u8; cols]; rows];

        for chunk in chunks {
            let (start, end) = spec.chunk_bounds(&chunk);
            for row in covered.iter_mut().take(end[0]).skip(start[0]) {
                for cell in row.iter_mut().take(end[1]).skip(start[1]) {
                    *cell += 1;
                }
            }
        }

        // Check every element is covered exactly once
        for (i, row) in covered.iter().enumerate() {
            for (j, &count) in row.iter().enumerate() {
                prop_assert_eq!(count, 1, "Element ({}, {}) covered {} times", i, j, count);
            }
        }
    }

    /// Property: Number of chunks matches expected count
    #[test]
    fn prop_chunk_count(
        (rows, cols) in shape_strategy(),
        chunk_row in chunk_size_strategy(20),
        chunk_col in chunk_size_strategy(20),
    ) {
        let spec = ChunkSpec::tile_size(&[rows, cols], &[chunk_row, chunk_col]).unwrap();

        let expected_row_chunks = rows.div_ceil(chunk_row);
        let expected_col_chunks = cols.div_ceil(chunk_col);
        let expected_total = expected_row_chunks * expected_col_chunks;

        let actual_count = spec.iter().count();

        prop_assert_eq!(actual_count, expected_total);
    }

    /// Property: All chunks are within tensor bounds
    #[test]
    fn prop_chunk_bounds(
        (rows, cols) in shape_strategy(),
        chunk_row in chunk_size_strategy(20),
        chunk_col in chunk_size_strategy(20),
    ) {
        let spec = ChunkSpec::tile_size(&[rows, cols], &[chunk_row, chunk_col]).unwrap();

        for chunk in spec.iter() {
            let (start, end) = spec.chunk_bounds(&chunk);
            prop_assert!(start[0] < rows);
            prop_assert!(start[1] < cols);
            prop_assert!(end[0] <= rows);
            prop_assert!(end[1] <= cols);

            // Verify chunk is non-empty
            prop_assert!(start[0] < end[0]);
            prop_assert!(start[1] < end[1]);
        }
    }
}

// ============================================================================
// Streaming Execution Properties
// ============================================================================

proptest! {
    /// Property: Chunked addition matches dense addition
    #[test]
    fn prop_chunked_add_correctness(
        (rows, cols) in shape_strategy(),
        chunk_size in chunk_size_strategy(20),
        seed in any::<u64>(),
    ) {
        use scirs2_core::random::{rngs::StdRng, SeedableRng, RngCore};

        // Create deterministic random tensors
        let mut rng = StdRng::seed_from_u64(seed);
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|_| (rng.next_u64() % 100) as f64)
            .collect();

        let a_arr = Array::from_shape_vec((rows, cols), a_data).unwrap();
        let b_arr = Array::from_shape_vec((rows, cols), b_data).unwrap();

        let a = DenseND::from_array(a_arr.clone().into_dyn());
        let b = DenseND::from_array(b_arr.clone().into_dyn());

        // Dense baseline
        let expected = &a_arr + &b_arr;

        // Chunked execution
        let config = StreamConfig::new()
            .chunk_size(vec![chunk_size])
            .max_memory_mb(16);
        let mut executor = StreamingExecutor::new(config);

        let result = executor.add_chunked(&a, &b).unwrap();
        let result_arr = result.as_array();

        // Compare results
        for (i, (result_row, expected_row)) in result_arr.outer_iter().zip(expected.outer_iter()).enumerate() {
            for (j, (&r, &e)) in result_row.iter().zip(expected_row.iter()).enumerate() {
                let diff = (r - e).abs();
                prop_assert!(diff < 1e-9, "Mismatch at ({}, {}): {} vs {}", i, j, r, e);
            }
        }
    }

    /// Property: Chunked multiplication matches dense multiplication
    #[test]
    fn prop_chunked_multiply_correctness(
        (rows, cols) in shape_strategy(),
        chunk_size in chunk_size_strategy(20),
        seed in any::<u64>(),
    ) {
        use scirs2_core::random::{rngs::StdRng, SeedableRng, RngCore};

        let mut rng = StdRng::seed_from_u64(seed);
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|_| (rng.next_u64() % 10) as f64)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|_| (rng.next_u64() % 10) as f64)
            .collect();

        let a_arr = Array::from_shape_vec((rows, cols), a_data).unwrap();
        let b_arr = Array::from_shape_vec((rows, cols), b_data).unwrap();

        let a = DenseND::from_array(a_arr.clone().into_dyn());
        let b = DenseND::from_array(b_arr.clone().into_dyn());

        // Dense baseline
        let expected = &a_arr * &b_arr;

        // Chunked execution
        let config = StreamConfig::new()
            .chunk_size(vec![chunk_size])
            .max_memory_mb(16);
        let mut executor = StreamingExecutor::new(config);

        let result = executor.multiply_chunked(&a, &b).unwrap();
        let result_arr = result.as_array();

        // Compare results
        for (result_row, expected_row) in result_arr.outer_iter().zip(expected.outer_iter()) {
            for (&r, &e) in result_row.iter().zip(expected_row.iter()) {
                let diff = (r - e).abs();
                prop_assert!(diff < 1e-9);
            }
        }
    }

    /// Property: FMA (fused multiply-add) matches separate ops
    #[test]
    fn prop_fma_correctness(
        (rows, cols) in shape_strategy(),
        chunk_size in chunk_size_strategy(20),
        seed in any::<u64>(),
    ) {
        use scirs2_core::random::{rngs::StdRng, SeedableRng, RngCore};

        let mut rng = StdRng::seed_from_u64(seed);
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|_| (rng.next_u64() % 10) as f64)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|_| (rng.next_u64() % 10) as f64)
            .collect();
        let c_data: Vec<f64> = (0..rows * cols)
            .map(|_| (rng.next_u64() % 10) as f64)
            .collect();

        let a_arr = Array::from_shape_vec((rows, cols), a_data).unwrap();
        let b_arr = Array::from_shape_vec((rows, cols), b_data).unwrap();
        let c_arr = Array::from_shape_vec((rows, cols), c_data).unwrap();

        let a = DenseND::from_array(a_arr.clone().into_dyn());
        let b = DenseND::from_array(b_arr.clone().into_dyn());
        let c = DenseND::from_array(c_arr.clone().into_dyn());

        // Dense baseline: a * b + c
        let expected = &a_arr * &b_arr + &c_arr;

        // FMA execution
        let config = StreamConfig::new()
            .chunk_size(vec![chunk_size])
            .max_memory_mb(16);
        let mut executor = StreamingExecutor::new(config);

        let result = executor.fma_chunked(&a, &b, &c).unwrap();
        let result_arr = result.as_array();

        // Compare results
        for (result_row, expected_row) in result_arr.outer_iter().zip(expected.outer_iter()) {
            for (&r, &e) in result_row.iter().zip(expected_row.iter()) {
                let diff = (r - e).abs();
                prop_assert!(diff < 1e-9);
            }
        }
    }

    /// Property: Matrix multiplication correctness (small sizes)
    #[test]
    fn prop_matmul_correctness(
        m in 2usize..8,
        k in 2usize..8,
        n in 2usize..8,
        chunk_size in 2usize..6,
        seed in any::<u64>(),
    ) {
        use scirs2_core::random::{rngs::StdRng, SeedableRng, RngCore};

        let mut rng = StdRng::seed_from_u64(seed);
        let a_data: Vec<f64> = (0..m * k)
            .map(|_| (rng.next_u64() % 10) as f64)
            .collect();
        let b_data: Vec<f64> = (0..k * n)
            .map(|_| (rng.next_u64() % 10) as f64)
            .collect();

        let a_arr = Array::from_shape_vec((m, k), a_data).unwrap();
        let b_arr = Array::from_shape_vec((k, n), b_data).unwrap();

        let a = DenseND::from_array(a_arr.clone().into_dyn());
        let b = DenseND::from_array(b_arr.clone().into_dyn());

        // Dense baseline using ndarray's dot
        let expected = a_arr.dot(&b_arr);

        // Chunked matmul
        let config = StreamConfig::new()
            .chunk_size(vec![chunk_size])
            .max_memory_mb(16);
        let mut executor = StreamingExecutor::new(config);

        let result = executor.matmul_chunked(&a, &b, Some(chunk_size)).unwrap();
        let result_arr = result.as_array();

        // Compare results (allow small numerical errors)
        for (i, (result_row, expected_row)) in result_arr.outer_iter().zip(expected.outer_iter()).enumerate() {
            for (j, (&r, &e)) in result_row.iter().zip(expected_row.iter()).enumerate() {
                let diff = (r - e).abs();
                prop_assert!(diff < 1e-6, "Matmul mismatch at ({}, {}): {} vs {}", i, j, r, e);
            }
        }
    }
}

// ============================================================================
// Memory Management Properties
// ============================================================================

proptest! {
    /// Property: Memory tracking is non-negative
    #[test]
    fn prop_memory_tracking_nonnegative(
        (rows, cols) in shape_strategy(),
        num_ops in 1usize..5,
    ) {
        let config = StreamConfig::new()
            .chunk_size(vec![4])
            .max_memory_mb(32);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::zeros(&[rows, cols]);
        let b = DenseND::<f64>::zeros(&[rows, cols]);

        for _ in 0..num_ops {
            let _result = executor.add_chunked(&a, &b).unwrap();

            // Memory tracking should be accessible (usize is always non-negative)
            let _current_mem = executor.current_memory();
        }
    }
}

// ============================================================================
// Chunk Graph Properties
// ============================================================================

proptest! {
    /// Property: Topological order respects dependencies
    #[test]
    fn prop_chunk_graph_topological_order(
        num_nodes in 2usize..10,
    ) {
        let mut graph = ChunkGraph::new();

        // Build a linear chain: input -> op1 -> op2 -> ... -> opN
        let input = graph.add_node(ChunkNode::input("A", vec![0]));
        let mut prev = input;

        for _ in 0..num_nodes - 1 {
            let node = graph.add_node(ChunkNode::operation(
                ChunkOp::Add,
                vec![prev, input], // Each op depends on previous
            ));
            prev = node;
        }

        let order = graph.topological_order().unwrap();

        // Total nodes: 1 input + (num_nodes - 1) operations = num_nodes
        prop_assert_eq!(order.len(), num_nodes);

        // Verify input comes first
        prop_assert_eq!(order[0], input);

        // Verify all nodes are present
        let node_set: std::collections::HashSet<_> = order.iter().copied().collect();
        prop_assert_eq!(node_set.len(), order.len(), "No duplicate nodes in topological order");
    }

    /// Property: Chunk graph with no edges has all nodes in topological order
    #[test]
    fn prop_chunk_graph_independent_nodes(
        num_nodes in 1usize..10,
    ) {
        let mut graph = ChunkGraph::new();

        let mut nodes = Vec::new();
        for i in 0..num_nodes {
            let node = graph.add_node(ChunkNode::input(&format!("input_{}", i), vec![i]));
            nodes.push(node);
        }

        let order = graph.topological_order().unwrap();

        // All nodes should be in the order
        prop_assert_eq!(order.len(), num_nodes);

        // All original nodes should be present
        for node in nodes {
            prop_assert!(order.contains(&node));
        }
    }
}

// ============================================================================
// Prefetch Strategy Properties
// ============================================================================

proptest! {
    /// Property: Prefetcher tracks access patterns correctly
    #[test]
    fn prop_prefetch_access_tracking(
        num_accesses in 1usize..20,
        strategy in prop_oneof![
            Just(PrefetchStrategy::None),
            Just(PrefetchStrategy::Sequential),
            Just(PrefetchStrategy::Adaptive),
        ],
    ) {
        let mut prefetcher = Prefetcher::new()
            .strategy(strategy)
            .queue_size(10);

        // Record sequential accesses
        for i in 0..num_accesses {
            let chunk_id = format!("chunk_{}", i);
            prefetcher.record_access(&chunk_id);
        }

        let stats = prefetcher.stats();

        // Verify basic stats structure
        prop_assert_eq!(stats.strategy, strategy);
        prop_assert_eq!(stats.queue_size, 10);
        prop_assert!(stats.queue_len <= stats.queue_size);
        prop_assert!(stats.prefetched_count <= num_accesses);
        prop_assert!(stats.access_history_len <= num_accesses);
    }

    /// Property: Prefetch stats are consistent
    #[test]
    fn prop_prefetch_stats_consistency(
        num_accesses in 10usize..30,
        queue_size in 1usize..10,
    ) {
        // Sequential prefetcher
        let mut prefetcher = Prefetcher::new()
            .strategy(PrefetchStrategy::Sequential)
            .queue_size(queue_size);

        // Record sequential accesses
        for i in 0..num_accesses {
            let chunk_id = format!("chunk_{}", i);
            prefetcher.record_access(&chunk_id);
        }

        let stats = prefetcher.stats();

        // Verify consistency of stats
        prop_assert_eq!(stats.queue_size, queue_size);
        prop_assert!(stats.queue_len <= queue_size);
        prop_assert_eq!(stats.enabled, true);
        prop_assert_eq!(stats.strategy, PrefetchStrategy::Sequential);
    }
}

// ============================================================================
// Streaming Configuration Properties
// ============================================================================

proptest! {
    /// Property: StreamConfig builder methods work correctly
    #[test]
    fn prop_stream_config_builder(
        max_mb in 1usize..100,
        chunk_size in 1usize..512,
        queue_size in 1usize..20,
    ) {
        let config = StreamConfig::new()
            .max_memory_mb(max_mb)
            .chunk_size(vec![chunk_size])
            .prefetch_queue_size(queue_size)
            .enable_profiling(true)
            .enable_prefetching(true);

        prop_assert_eq!(config.max_memory_bytes, max_mb * 1024 * 1024);
        prop_assert_eq!(config.default_chunk_size[0], chunk_size);
        prop_assert_eq!(config.prefetch_queue_size, queue_size);
        prop_assert_eq!(config.enable_profiling, true);
        prop_assert_eq!(config.enable_prefetching, true);
    }
}
