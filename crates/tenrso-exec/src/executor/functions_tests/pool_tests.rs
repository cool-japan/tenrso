//! Tests for memory-pool behaviour and automatic-pooling integration on
//! `CpuExecutor`.

#![allow(clippy::unnecessary_cast)]

use super::super::{
    functions::TenrsoExecutor,
    types::{BinaryOp, CpuExecutor, MemoryPool},
};
use crate::hints::ExecHints;
use tenrso_core::{DenseND, TensorHandle};

#[test]
fn test_memory_pool_stats() {
    let executor = CpuExecutor::new();
    let (hits, misses, hit_rate) = executor.pool_stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 0);
    assert_eq!(hit_rate, 0.0);
}

#[test]
fn test_executor_with_threads() {
    let executor = CpuExecutor::with_threads(4).expect("4-thread pool");
    assert_eq!(executor.num_threads(), 4);
    // The configured count is only worth anything if it is also the *effective*
    // one; `thread_pool_tests` proves the parallel regions really run on it.
    assert_eq!(executor.effective_num_threads(), 4);
}

#[test]
fn test_clear_pool() {
    let mut executor = CpuExecutor::new();
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let _result = executor
        .einsum("ij,jk->ik", &[handle_a, handle_b], &ExecHints::default())
        .unwrap();
    executor.clear_pool();
    let (hits, misses, hit_rate) = executor.pool_stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 0);
    assert_eq!(hit_rate, 0.0);
}

#[test]
fn test_memory_pool_shape_signature() {
    assert_eq!(MemoryPool::<f64>::shape_signature(&[2, 3, 4]), "2x3x4");
    assert_eq!(MemoryPool::<f64>::shape_signature(&[10, 20]), "10x20");
    assert_eq!(MemoryPool::<f64>::shape_signature(&[1]), "1");
}

#[test]
fn test_memory_pool_acquire_release() {
    let mut pool = MemoryPool::<f64>::new();
    let buffer1 = pool.acquire(&[2, 3]);
    assert_eq!(buffer1.len(), 2 * 3);
    let (hits, misses, _) = pool.stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 1);
    pool.release(&[2, 3], buffer1);
    let buffer2 = pool.acquire(&[2, 3]);
    assert_eq!(buffer2.len(), 2 * 3);
    let (hits, misses, hit_rate) = pool.stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);
    assert!((hit_rate - 0.5).abs() < 1e-10);
}

// Phase 1 Memory Pool Tests
#[test]
fn test_pool_detailed_stats() {
    let mut pool = MemoryPool::<f64>::new();

    // Initial state
    let stats = pool.detailed_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.total_allocations, 0);
    assert_eq!(stats.total_releases, 0);
    assert_eq!(stats.hit_rate, 0.0);
    assert_eq!(stats.unique_shapes, 0);
    assert_eq!(stats.total_buffers_pooled, 0);

    // Acquire a buffer (miss)
    let buffer1 = pool.acquire(&[2, 3]);
    let stats = pool.detailed_stats();
    assert_eq!(stats.total_allocations, 1);
    assert_eq!(stats.misses, 1);

    // Release buffer
    pool.release(&[2, 3], buffer1);
    let stats = pool.detailed_stats();
    assert_eq!(stats.total_releases, 1);
    assert_eq!(stats.unique_shapes, 1);
    assert_eq!(stats.total_buffers_pooled, 1);

    // Acquire again (hit)
    let _buffer2 = pool.acquire(&[2, 3]);
    let stats = pool.detailed_stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.total_allocations, 2);
    assert_eq!(stats.hit_rate, 0.5);
}

#[test]
fn test_pool_disabled() {
    let mut pool = MemoryPool::<f64>::disabled();
    assert!(!pool.is_enabled());

    // Acquire should still work but not pool
    let buffer1 = pool.acquire(&[2, 3]);
    assert_eq!(buffer1.len(), 2 * 3);

    // Release should not pool
    pool.release(&[2, 3], buffer1);
    let stats = pool.detailed_stats();
    assert_eq!(stats.total_buffers_pooled, 0);

    // Second acquire should miss
    let _buffer2 = pool.acquire(&[2, 3]);
    let stats = pool.detailed_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 2);
}

#[test]
fn test_pool_enable_disable() {
    let mut pool = MemoryPool::<f64>::new();
    assert!(pool.is_enabled());

    // Add some buffers
    let buffer = pool.acquire(&[2, 3]);
    pool.release(&[2, 3], buffer);
    assert_eq!(pool.num_buffers(), 1);

    // Disable pool (should clear)
    pool.set_enabled(false);
    assert!(!pool.is_enabled());
    assert_eq!(pool.num_buffers(), 0);

    // Re-enable
    pool.set_enabled(true);
    assert!(pool.is_enabled());
}

#[test]
fn test_executor_pool_api() {
    let mut executor = CpuExecutor::new();
    assert!(executor.is_pool_enabled());

    // Get initial stats
    let stats = executor.get_pool_stats();
    assert_eq!(stats.total_allocations, 0);

    // Disable pooling
    executor.set_pool_enabled(false);
    assert!(!executor.is_pool_enabled());

    // Re-enable
    executor.set_pool_enabled(true);
    assert!(executor.is_pool_enabled());

    // Check pool counts
    assert_eq!(executor.pool_num_shapes(), 0);
    assert_eq!(executor.pool_num_buffers(), 0);
}

#[test]
fn test_executor_with_memory_pool() {
    let executor_enabled = CpuExecutor::new().with_memory_pool(true);
    assert!(executor_enabled.is_pool_enabled());

    let executor_disabled = CpuExecutor::new().with_memory_pool(false);
    assert!(!executor_disabled.is_pool_enabled());
}

#[test]
fn test_unoptimized_executor_pool_disabled() {
    let executor = CpuExecutor::unoptimized();
    assert!(!executor.is_pool_enabled());
    assert!(!executor.enable_memory_pool);
}

#[test]
fn test_pool_multiple_shapes() {
    let mut pool = MemoryPool::<f64>::new();

    // Add buffers for different shapes
    let b1 = pool.acquire(&[2, 3]);
    pool.release(&[2, 3], b1);

    let b2 = pool.acquire(&[4, 5]);
    pool.release(&[4, 5], b2);

    let b3 = pool.acquire(&[6, 7, 8]);
    pool.release(&[6, 7, 8], b3);

    assert_eq!(pool.num_shapes(), 3);
    assert_eq!(pool.num_buffers(), 3);

    let stats = pool.detailed_stats();
    assert_eq!(stats.unique_shapes, 3);
    assert_eq!(stats.total_buffers_pooled, 3);
}

// ========================================================================
// Phase 5: Automatic Memory Pool Integration Tests
// ========================================================================

#[test]
fn test_automatic_pooling_binary_op() {
    // Test that binary operations with broadcasting automatically use the memory pool
    let mut executor = CpuExecutor::new();

    // Clear pool to start fresh
    executor.clear_pool();

    // Use broadcasting: [2, 3] + [3] -> requires pooled allocation
    let a = DenseND::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = DenseND::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);

    // First operation - should miss (allocate)
    let result1 = executor
        .binary_op(BinaryOp::Add, &handle_a, &handle_b)
        .unwrap();
    let stats1 = executor.get_pool_stats_f32();
    assert_eq!(stats1.misses, 1, "First operation should allocate (miss)");
    assert_eq!(stats1.total_allocations, 1);

    // Second operation with same output shape - should hit (reuse)
    let c = DenseND::from_vec(vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3]).unwrap();
    let d = DenseND::from_vec(vec![2.0f32, 3.0, 4.0], &[3]).unwrap();
    let handle_c = TensorHandle::from_dense_auto(c);
    let handle_d = TensorHandle::from_dense_auto(d);
    let result2 = executor
        .binary_op(BinaryOp::Mul, &handle_c, &handle_d)
        .unwrap();
    let stats2 = executor.get_pool_stats_f32();
    assert_eq!(
        stats2.hits, 1,
        "Second operation with same output shape should reuse buffer (hit)"
    );
    assert_eq!(
        stats2.total_allocations, 2,
        "Should have 2 total allocations"
    );

    // Verify results are correct (broadcasting works)
    let result1_dense = result1.as_dense().unwrap();
    let result2_dense = result2.as_dense().unwrap();
    assert_eq!(result1_dense.shape(), &[2, 3]);
    assert_eq!(result2_dense.shape(), &[2, 3]);
}

#[test]
fn test_automatic_pooling_conv1d() {
    // Test that conv1d automatically uses the memory pool
    let mut executor = CpuExecutor::new();
    executor.clear_pool();

    let x = DenseND::from_vec(vec![1.0f32; 2 * 3 * 10], &[2, 3, 10]).unwrap();
    let kernel = DenseND::from_vec(vec![1.0f32; 4 * 3 * 3], &[4, 3, 3]).unwrap();
    let x_handle = TensorHandle::from_dense_auto(x.clone());
    let kernel_handle = TensorHandle::from_dense_auto(kernel.clone());

    // First conv - should miss
    let _result1 = executor
        .conv1d(&x_handle, &kernel_handle, None, 1, (0, 0))
        .unwrap();
    let stats1 = executor.get_pool_stats_f32();
    assert!(stats1.misses >= 1, "First conv should allocate");

    // Second conv with same shape - should hit
    let x_handle2 = TensorHandle::from_dense_auto(x);
    let kernel_handle2 = TensorHandle::from_dense_auto(kernel);
    let _result2 = executor
        .conv1d(&x_handle2, &kernel_handle2, None, 1, (0, 0))
        .unwrap();
    let stats2 = executor.get_pool_stats_f32();
    assert!(
        stats2.hits >= 1,
        "Second conv with same shape should reuse buffer"
    );
}

#[test]
fn test_automatic_pooling_hit_rate() {
    // Test high hit rate with repeated operations (using broadcasting to trigger pooling)
    let mut executor = CpuExecutor::new();
    executor.clear_pool();

    let a = DenseND::from_vec(vec![1.0f32; 100], &[10, 10]).unwrap();
    let b = DenseND::from_vec(vec![2.0f32; 10], &[10]).unwrap(); // Broadcasting: [10,10] + [10]

    // Perform 10 operations with same output shape (triggers pooling via broadcasting)
    for _ in 0..10 {
        let handle_a_copy = TensorHandle::from_dense_auto(a.clone());
        let handle_b_copy = TensorHandle::from_dense_auto(b.clone());
        let _ = executor
            .binary_op(BinaryOp::Add, &handle_a_copy, &handle_b_copy)
            .unwrap();
    }

    let stats = executor.get_pool_stats_f32();
    // First operation misses, rest should hit
    assert!(
        stats.hits >= 8,
        "Should have high hit rate (>= 80%), got {} hits out of {} allocations",
        stats.hits,
        stats.total_allocations
    );
    assert!(
        stats.hit_rate >= 0.8,
        "Hit rate should be >= 80%, got {}",
        stats.hit_rate
    );
}

#[test]
fn test_pooling_can_be_disabled() {
    // Test that pooling can be disabled
    let mut executor = CpuExecutor::new();
    executor.set_pool_enabled(false);

    let a = DenseND::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);

    // Operations should work but not use the pool
    let result = executor
        .binary_op(BinaryOp::Add, &handle_a, &handle_b)
        .unwrap();
    let stats = executor.get_pool_stats_f32();

    assert_eq!(stats.hits, 0, "Disabled pool should have no hits");
    assert_eq!(stats.misses, 0, "Disabled pool should have no misses");

    // Verify result is correct
    let result_dense = result.as_dense().unwrap();
    assert_eq!(result_dense.view()[[0, 0]], 6.0);
}

#[test]
fn test_pooling_with_different_shapes() {
    // Test that pool handles different shapes correctly (using broadcasting)
    let mut executor = CpuExecutor::new();
    executor.clear_pool();

    // Operation with output shape [2, 2] (broadcasting: [2, 2] + [2])
    let a = DenseND::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::from_vec(vec![5.0f32, 6.0], &[2]).unwrap();
    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);
    let _ = executor
        .binary_op(BinaryOp::Add, &handle_a, &handle_b)
        .unwrap();

    // Operation with output shape [3, 3] (broadcasting: [3, 3] + [3])
    let c = DenseND::from_vec(vec![1.0f32; 9], &[3, 3]).unwrap();
    let d = DenseND::from_vec(vec![2.0f32; 3], &[3]).unwrap();
    let handle_c = TensorHandle::from_dense_auto(c);
    let handle_d = TensorHandle::from_dense_auto(d);
    let _ = executor
        .binary_op(BinaryOp::Add, &handle_c, &handle_d)
        .unwrap();

    let stats = executor.get_pool_stats_f32();
    assert_eq!(
        stats.unique_shapes, 2,
        "Should have 2 unique shapes in pool"
    );
    assert_eq!(
        stats.misses, 2,
        "Should have 2 misses (one per unique shape)"
    );
}
