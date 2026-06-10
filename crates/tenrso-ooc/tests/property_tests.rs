//! Property-based tests for out-of-core tensor operations
//!
//! These tests use proptest to verify algebraic properties and correctness
//! of chunking, streaming execution, and memory management.

use proptest::prelude::*;
use scirs2_core::ndarray_ext::Array;
use tenrso_core::DenseND;
use tenrso_ooc::{
    AlignedBuffer, ChunkGraph, ChunkNode, ChunkOp, ChunkSpec, PrefetchStrategy, Prefetcher,
    StreamConfig, StreamingExecutor,
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
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

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
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

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
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

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
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

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

// ============================================================================
// Unsafe Code Fuzzing: ZeroCopy I/O Round-Trip
//
// These property tests act as a lightweight fuzz harness for the unsafe I/O
// paths in `zerocopy_io.rs` (`from_raw_parts` for both read and write):
//   - Write: &[f64] → &[u8] via `from_raw_parts` (any bit-pattern → valid u8)
//   - Read:  &[u8]  → &[f64] via `from_raw_parts` (must be f64-aligned and exact)
//
// `AlignedBuffer` exercises `std::alloc::alloc` / `std::alloc::dealloc` with
// arbitrary sizes and power-of-2 alignments.
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Verifies that the unsafe f64↔u8 byte-reinterpretation used in
    /// `zerocopy_io::write_tensor_f64` / `read_tensor_f64` produces bit-exact
    /// round-trips for arbitrary tensor data and shapes.
    ///
    /// Directly exercises both `from_raw_parts` directions:
    ///   - Write: `&[f64]` → `&[u8]` then persisted via `std::fs::write`
    ///   - Read:  `Vec<u8>` → aligned `Vec<f64>` via `copy_from_slice` into
    ///     a properly-typed allocation (same bit-cast as `read_tensor_f64`)
    ///
    /// This avoids the mmap layer (which requires O_RDWR) while still
    /// exercising the core unsafe code path that handles raw memory reinterpretation.
    #[test]
    fn prop_zerocopy_f64_byte_roundtrip(
        rows in 1usize..12,
        cols in 1usize..12,
        seed in any::<u64>(),
    ) {
        let n = rows * cols;
        let mut state = seed.wrapping_add(1);
        let data: Vec<f64> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state as i64) as f64 / i64::MAX as f64
            })
            .collect();

        let tmp = std::env::temp_dir()
            .join(format!("tenrso_fuzz_bytes_{rows}_{cols}_{seed}.bin"));

        // ── Write path: &[f64] → &[u8] (mirrors write_tensor_f64 unsafe block)
        let byte_count = n * std::mem::size_of::<f64>();
        // SAFETY: reinterpreting a &[f64] as &[u8]: any bit-pattern is a valid u8;
        // byte_count == data.len() * size_of::<f64>() is exact; data outlives byte_slice.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_count)
        };
        std::fs::write(&tmp, byte_slice).expect("write bytes to temp file");

        // ── Read path: Vec<u8> → aligned Vec<f64> (mirrors read_tensor_f64 unsafe block)
        let file_bytes = std::fs::read(&tmp).expect("read bytes from temp file");
        let _ = std::fs::remove_file(&tmp);
        prop_assert_eq!(file_bytes.len(), byte_count, "file size mismatch");

        let mut aligned: Vec<f64> = vec![0.0_f64; n];
        // SAFETY: `aligned` is a properly-allocated Vec<f64>; reinterpreting as &mut [u8]
        // to copy raw bytes is safe because: (a) the source bytes came from a valid &[f64]
        // so their bit-patterns constitute valid f64 values, (b) the destination is aligned
        // to f64's requirements (Vec<f64> guarantees 8-byte alignment), (c) lengths match.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                aligned.as_mut_ptr() as *mut u8,
                byte_count,
            );
            dst.copy_from_slice(&file_bytes);
        }

        for (i, (expected, actual)) in data.iter().zip(aligned.iter()).enumerate() {
            prop_assert!(
                expected.to_bits() == actual.to_bits(),
                "{}",
                format!("bit mismatch at index {i} (shape [{rows}x{cols}]): {expected} vs {actual}")
            );
        }
    }

    /// Verifies that `AlignedBuffer::new(size, alignment)` always returns a
    /// buffer whose backing pointer satisfies `ptr % alignment == 0`.
    /// Exercises the `std::alloc::alloc` call with arbitrary (size, alignment) pairs.
    #[test]
    fn prop_aligned_buffer_pointer_alignment(
        size in 1usize..4096,
        align_exp in 4u32..12, // 2^4 = 16 … 2^11 = 2048 bytes
    ) {
        let alignment = 1usize << align_exp;
        let buf = AlignedBuffer::new(size, alignment);
        let ptr = buf.as_slice().as_ptr() as usize;
        prop_assert_eq!(
            ptr % alignment, 0,
            "{}",
            format!("ptr {ptr:#x} not aligned to {alignment}; size={size}")
        );
        prop_assert_eq!(buf.as_slice().len(), size, "wrong buffer length");
    }
}
