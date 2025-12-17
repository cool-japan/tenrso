//! Tests for TenrsoExecutor trait implementations
//!
//! This module contains comprehensive tests for all executor operations.

#[cfg(test)]
#[allow(clippy::unnecessary_cast)]
mod tests {
    use super::super::{
        functions::TenrsoExecutor,
        types::{BinaryOp, CpuExecutor, ElemOp, MemoryPool, ReduceOp, ScatterMode},
    };
    use crate::hints::ExecHints;
    use tenrso_core::{DenseND, TensorHandle};
    #[test]
    fn test_cpu_executor_matmul() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .einsum("ij,jk->ik", &[handle_a, handle_b], &ExecHints::default())
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        let diff: f64 = result_view[[0, 0]] - 58.0;
        assert!(diff.abs() < 1e-10);
    }
    #[test]
    fn test_cpu_executor_input_validation() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.einsum("ij,jk->ik", &[handle_a], &ExecHints::default());
        assert!(result.is_err());
    }
    #[test]
    fn test_cpu_executor_three_tensors() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let c = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let handle_c = TensorHandle::from_dense_auto(c);
        let result = executor
            .einsum(
                "ij,jk,kl->il",
                &[handle_a, handle_b, handle_c],
                &ExecHints::default(),
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        let val: f64 = result_view[[0, 0]];
        assert!(val.abs() > 0.0);
    }
    #[test]
    fn test_cpu_executor_outer_then_contract() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let c = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let handle_c = TensorHandle::from_dense_auto(c);
        let result = executor
            .einsum(
                "i,j,ij->",
                &[handle_a, handle_b, handle_c],
                &ExecHints::default(),
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert!(result_dense.shape().is_empty() || result_dense.shape() == [1]);
        let result_view = result_dense.view();
        let result_val = if result_dense.shape().is_empty() {
            result_view[[]]
        } else {
            result_view[[0]]
        };
        let diff: f64 = result_val - 78.0;
        assert!(diff.abs() < 1e-10, "Expected 78.0, got {}", result_val);
    }
    #[test]
    fn test_elem_op_neg() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Neg, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0, 0]] - -1.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 1]] - 2.0_f64).abs() < 1e-10);
        assert!((result_view[[1, 0]] - -3.0_f64).abs() < 1e-10);
        assert!((result_view[[1, 1]] - 4.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_elem_op_abs() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Abs, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0, 0]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 1]] - 2.0_f64).abs() < 1e-10);
        assert!((result_view[[1, 0]] - 3.0_f64).abs() < 1e-10);
        assert!((result_view[[1, 1]] - 4.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_elem_op_exp() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Exp, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] as f64 - std::f64::consts::E as f64).abs() < 1e-10);
        assert!((result_view[[2]] - std::f64::consts::E.powi(2)).abs() < 1e-9);
    }
    #[test]
    fn test_elem_op_log() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(
            vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)],
            &[3],
        )
        .unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Log, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 2.0_f64).abs() < 1e-9);
    }
    #[test]
    fn test_reduce_sum_single_axis() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 5.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 7.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 9.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_reduce_sum_axis_1() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[1]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2]);
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 6.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 15.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_reduce_max() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Max, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 2.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 8.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 4.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_reduce_min() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Min, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 5.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 3.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_reduce_mean() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Mean, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 2.5_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 3.5_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 4.5_f64).abs() < 1e-10);
    }
    #[test]
    fn test_reduce_multiple_axes() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[0, 2]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert!(result_view[[0]] > 0.0);
    }
    #[test]
    fn test_reduce_invalid_axis() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[2]);
        assert!(result.is_err());
    }
    #[test]
    fn test_reduce_no_axes() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[]);
        assert!(result.is_err());
    }
    #[test]
    fn test_elem_op_sin() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(
            vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI],
            &[3],
        )
        .unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Sin, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 0.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_elem_op_cos() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(
            vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI],
            &[3],
        )
        .unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Cos, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - -1.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_elem_op_sqrt() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Sqrt, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 2.0_f64).abs() < 1e-10);
        assert!((result_view[[3]] - 3.0_f64).abs() < 1e-10);
        assert!((result_view[[4]] - 4.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_elem_op_sqr() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Sqr, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 4.0_f64).abs() < 1e-10);
        assert!((result_view[[3]] - 9.0_f64).abs() < 1e-10);
        assert!((result_view[[4]] - 16.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_elem_op_recip() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 4.0, 10.0], &[4]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Recip, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 0.5_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 0.25_f64).abs() < 1e-10);
        assert!((result_view[[3]] - 0.1_f64).abs() < 1e-10);
    }
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
        let executor = CpuExecutor::with_threads(4);
        assert_eq!(executor.num_threads, 4);
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
    #[test]
    fn test_elem_op_tanh() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Tanh, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[2]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[0]] + result_view[[4]]).abs() < 1e-10);
        assert!((result_view[[1]] + result_view[[3]]).abs() < 1e-10);
        for i in 0..5 {
            assert!((result_view[[i]] as f64).abs() < 1.0);
        }
    }
    #[test]
    fn test_elem_op_sigmoid() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-10.0, 0.0, 10.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Sigmoid, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[1]] - 0.5_f64).abs() < 1e-10);
        assert!(result_view[[0]] < 0.001);
        assert!(result_view[[2]] > 0.999);
        for i in 0..3 {
            assert!(result_view[[i]] > 0.0 && result_view[[i]] < 1.0);
        }
    }
    #[test]
    fn test_elem_op_relu() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::ReLU, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 0.0);
        assert_eq!(result_view[[1]], 0.0);
        assert_eq!(result_view[[2]], 0.0);
        assert_eq!(result_view[[3]], 1.0);
        assert_eq!(result_view[[4]], 2.0);
    }
    #[test]
    fn test_elem_op_gelu() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Gelu, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[1]] as f64).abs() < 0.01);
        assert!(result_view[[2]] > 0.8);
        assert!((result_view[[0]] as f64).abs() < 0.2);
    }
    #[test]
    fn test_elem_op_elu() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Elu, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[2]], 1.0);
        assert_eq!(result_view[[3]], 2.0);
        assert_eq!(result_view[[1]], 0.0);
        assert!((result_view[[0]] - (std::f64::consts::E.recip() - 1.0)).abs() < 1e-9);
    }
    #[test]
    fn test_elem_op_selu() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Selu, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        let scale = 1.050_700_987_355_480_5;
        let alpha = 1.673_263_242_354_377_2;
        assert!((result_view[[2]] as f64 - scale as f64).abs() < 1e-9);
        assert_eq!(result_view[[1]], 0.0);
        let expected_neg = scale * alpha * ((-1.0_f64).exp() - 1.0);
        assert!((result_view[[0]] as f64 - expected_neg as f64).abs() < 1e-9);
    }
    #[test]
    fn test_elem_op_softplus() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-10.0, 0.0, 10.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Softplus, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[1]] - 2.0_f64.ln()).abs() < 1e-9);
        assert!(result_view[[0]] < 0.001);
        assert!((result_view[[2]] - 10.0_f64).abs() < 0.001);
    }
    #[test]
    fn test_elem_op_sign() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-2.5, -1.0, 0.0, 1.0, 3.5], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.elem_op(ElemOp::Sign, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], -1.0);
        assert_eq!(result_view[[1]], -1.0);
        assert_eq!(result_view[[2]], 0.0);
        assert_eq!(result_view[[3]], 1.0);
        assert_eq!(result_view[[4]], 1.0);
    }
    #[test]
    fn test_binary_op_add() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Add, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 6.0);
        assert_eq!(result_view[[0, 1]], 8.0);
        assert_eq!(result_view[[1, 0]], 10.0);
        assert_eq!(result_view[[1, 1]], 12.0);
    }
    #[test]
    fn test_binary_op_sub() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Sub, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 9.0);
        assert_eq!(result_view[[0, 1]], 18.0);
        assert_eq!(result_view[[1, 0]], 27.0);
        assert_eq!(result_view[[1, 1]], 36.0);
    }
    #[test]
    fn test_binary_op_mul() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Mul, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 2.0);
        assert_eq!(result_view[[0, 1]], 6.0);
        assert_eq!(result_view[[1, 0]], 12.0);
        assert_eq!(result_view[[1, 1]], 20.0);
    }
    #[test]
    fn test_binary_op_div() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 4.0, 5.0, 8.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Div, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 5.0);
        assert_eq!(result_view[[0, 1]], 5.0);
        assert_eq!(result_view[[1, 0]], 6.0);
        assert_eq!(result_view[[1, 1]], 5.0);
    }
    #[test]
    fn test_binary_op_pow() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 2.0, 3.0, 2.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Pow, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 4.0);
        assert_eq!(result_view[[0, 1]], 9.0);
        assert_eq!(result_view[[1, 0]], 64.0);
        assert_eq!(result_view[[1, 1]], 25.0);
    }
    #[test]
    fn test_binary_op_maximum() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 7.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 4.0, 6.0, 1.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Maximum, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 2.0);
        assert_eq!(result_view[[0, 1]], 5.0);
        assert_eq!(result_view[[1, 0]], 6.0);
        assert_eq!(result_view[[1, 1]], 7.0);
    }
    #[test]
    fn test_binary_op_minimum() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 7.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 4.0, 6.0, 1.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Minimum, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 1.0);
        assert_eq!(result_view[[0, 1]], 4.0);
        assert_eq!(result_view[[1, 0]], 3.0);
        assert_eq!(result_view[[1, 1]], 1.0);
    }
    #[test]
    fn test_binary_op_broadcast_scalar() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![5.0], &[1]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Add, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 6.0);
        assert_eq!(result_view[[0, 1]], 7.0);
        assert_eq!(result_view[[1, 0]], 8.0);
        assert_eq!(result_view[[1, 1]], 9.0);
    }
    #[test]
    fn test_binary_op_broadcast_scalar_mul() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![2.0], &[1]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Mul, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 2.0);
        assert_eq!(result_view[[0, 1]], 4.0);
        assert_eq!(result_view[[1, 0]], 6.0);
        assert_eq!(result_view[[1, 1]], 8.0);
    }
    #[test]
    fn test_clip_basic() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![-5.0, -1.0, 0.0, 3.0, 10.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.clip(&handle_a, -2.0, 5.0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], -2.0);
        assert_eq!(result_view[[1]], -1.0);
        assert_eq!(result_view[[2]], 0.0);
        assert_eq!(result_view[[3]], 3.0);
        assert_eq!(result_view[[4]], 5.0);
    }
    #[test]
    fn test_clip_no_change() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.clip(&handle_a, 0.0, 5.0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 1.0);
        assert_eq!(result_view[[1]], 2.0);
        assert_eq!(result_view[[2]], 3.0);
    }
    #[test]
    fn test_clip_invalid_bounds() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.clip(&handle_a, 5.0, 1.0);
        assert!(result.is_err());
    }
    #[test]
    fn test_softmax_axis0() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.softmax(&handle_a, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        let col0_sum: f64 = result_view[[0, 0]] + result_view[[1, 0]];
        let col1_sum: f64 = result_view[[0, 1]] + result_view[[1, 1]];
        let col2_sum: f64 = result_view[[0, 2]] + result_view[[1, 2]];
        assert!((col0_sum - 1.0).abs() < 1e-10);
        assert!((col1_sum - 1.0).abs() < 1e-10);
        assert!((col2_sum - 1.0).abs() < 1e-10);
        for i in 0..2 {
            for j in 0..3 {
                assert!(result_view[[i, j]] > 0.0);
                assert!(result_view[[i, j]] < 1.0);
            }
        }
    }
    #[test]
    fn test_softmax_axis1() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.softmax(&handle_a, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        let row0_sum: f64 = result_view[[0, 0]] + result_view[[0, 1]] + result_view[[0, 2]];
        let row1_sum: f64 = result_view[[1, 0]] + result_view[[1, 1]] + result_view[[1, 2]];
        assert!((row0_sum - 1.0).abs() < 1e-10);
        assert!((row1_sum - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_softmax_numerical_stability() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1000.0, 1001.0, 1002.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.softmax(&handle_a, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        let sum: f64 = result_view[[0]] + result_view[[1]] + result_view[[2]];
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((result_view[[0]] as f64).is_finite());
        assert!((result_view[[1]] as f64).is_finite());
        assert!((result_view[[2]] as f64).is_finite());
    }
    #[test]
    fn test_softmax_invalid_axis() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.softmax(&handle_a, 2);
        assert!(result.is_err());
    }
    #[test]
    fn test_log_softmax_axis1() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.log_softmax(&handle_a, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        let row0_sum = (result_view[[0, 0]] as f64).exp()
            + (result_view[[0, 1]] as f64).exp()
            + (result_view[[0, 2]] as f64).exp();
        let row1_sum = (result_view[[1, 0]] as f64).exp()
            + (result_view[[1, 1]] as f64).exp()
            + (result_view[[1, 2]] as f64).exp();
        assert!((row0_sum - 1.0).abs() < 1e-10);
        assert!((row1_sum - 1.0).abs() < 1e-10);
        for i in 0..2 {
            for j in 0..3 {
                assert!(result_view[[i, j]] < 0.0 || (result_view[[i, j]] as f64).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn test_log_softmax_numerical_stability() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1000.0, 1001.0, 1002.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.log_softmax(&handle_a, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[0]] as f64).is_finite());
        assert!((result_view[[1]] as f64).is_finite());
        assert!((result_view[[2]] as f64).is_finite());
        let sum: f64 = (result_view[[0]] as f64).exp()
            + (result_view[[1]] as f64).exp()
            + (result_view[[2]] as f64).exp();
        assert!((sum - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_log_softmax_equivalence() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let softmax_result = executor.softmax(&handle_a, 1).unwrap();
        let log_of_softmax = executor.elem_op(ElemOp::Log, &softmax_result).unwrap();
        let los_dense = log_of_softmax.as_dense().unwrap();
        let log_softmax_result = executor.log_softmax(&handle_a, 1).unwrap();
        let ls_dense = log_softmax_result.as_dense().unwrap();
        let los_view = los_dense.view();
        let ls_view = ls_dense.view();
        for i in 0..2 {
            for j in 0..2 {
                let diff: f64 = los_view[[i, j]] - ls_view[[i, j]];
                assert!(diff.abs() < 1e-9);
            }
        }
    }
    #[test]
    fn test_transpose_2d() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.transpose(&handle_a, &[1, 0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 1.0);
        assert_eq!(result_view[[0, 1]], 4.0);
        assert_eq!(result_view[[1, 0]], 2.0);
        assert_eq!(result_view[[1, 1]], 5.0);
        assert_eq!(result_view[[2, 0]], 3.0);
        assert_eq!(result_view[[2, 1]], 6.0);
    }
    #[test]
    fn test_transpose_3d() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let a = DenseND::from_vec(data, &[2, 3, 4]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.transpose(&handle_a, &[2, 0, 1]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4, 2, 3]);
    }
    #[test]
    fn test_transpose_invalid_axes() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.transpose(&handle_a, &[0]);
        assert!(result.is_err());
        let result = executor.transpose(&handle_a, &[0, 0]);
        assert!(result.is_err());
        let result = executor.transpose(&handle_a, &[0, 2]);
        assert!(result.is_err());
    }
    #[test]
    fn test_reshape_2d_to_1d() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reshape(&handle_a, &[6]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[6]);
        let result_view = result_dense.view();
        for (i, &expected) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
            assert_eq!(result_view[[i]], expected);
        }
    }
    #[test]
    fn test_reshape_1d_to_3d() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let a = DenseND::from_vec(data, &[24]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reshape(&handle_a, &[2, 3, 4]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 3, 4]);
    }
    #[test]
    fn test_reshape_invalid_size() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.reshape(&handle_a, &[3, 3]);
        assert!(result.is_err());
    }
    #[test]
    fn test_concatenate_axis0() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor.concatenate(&[handle_a, handle_b], 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4, 2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 1.0);
        assert_eq!(result_view[[1, 1]], 4.0);
        assert_eq!(result_view[[2, 0]], 5.0);
        assert_eq!(result_view[[3, 1]], 8.0);
    }
    #[test]
    fn test_concatenate_axis1() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor.concatenate(&[handle_a, handle_b], 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 4]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 1.0);
        assert_eq!(result_view[[0, 1]], 2.0);
        assert_eq!(result_view[[0, 2]], 5.0);
        assert_eq!(result_view[[0, 3]], 6.0);
    }
    #[test]
    fn test_concatenate_multiple() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = DenseND::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let c = DenseND::from_vec(vec![5.0, 6.0], &[2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let handle_c = TensorHandle::from_dense_auto(c);
        let result = executor
            .concatenate(&[handle_a, handle_b, handle_c], 0)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[6]);
        let result_view = result_dense.view();
        for (i, &expected) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
            assert_eq!(result_view[[i]], expected);
        }
    }
    #[test]
    fn test_split_axis0() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let a = DenseND::from_vec(data, &[4, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let results = executor.split(&handle_a, 2, 0).unwrap();
        assert_eq!(results.len(), 2);
        let result0 = results[0].as_dense().unwrap();
        let result1 = results[1].as_dense().unwrap();
        assert_eq!(result0.shape(), &[2, 3]);
        assert_eq!(result1.shape(), &[2, 3]);
        let view0 = result0.view();
        let view1 = result1.view();
        assert_eq!(view0[[0, 0]], 0.0);
        assert_eq!(view0[[1, 2]], 5.0);
        assert_eq!(view1[[0, 0]], 6.0);
        assert_eq!(view1[[1, 2]], 11.0);
    }
    #[test]
    fn test_split_axis1() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let a = DenseND::from_vec(data, &[3, 4]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let results = executor.split(&handle_a, 2, 1).unwrap();
        assert_eq!(results.len(), 2);
        let result0 = results[0].as_dense().unwrap();
        let result1 = results[1].as_dense().unwrap();
        assert_eq!(result0.shape(), &[3, 2]);
        assert_eq!(result1.shape(), &[3, 2]);
    }
    #[test]
    fn test_split_invalid() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let result = executor.split(&handle_a, 2, 0);
        assert!(result.is_err());
    }
    #[test]
    fn test_layer_norm_2d() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let eps = 1e-5;
        let result = executor.layer_norm(&handle_a, eps).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 3]);
        let result_view = result_dense.view();
        let row0_mean: f64 =
            (result_view[[0, 0]] + result_view[[0, 1]] + result_view[[0, 2]]) / 3.0;
        assert!(row0_mean.abs() < 1e-6);
        let row1_mean: f64 =
            (result_view[[1, 0]] + result_view[[1, 1]] + result_view[[1, 2]]) / 3.0;
        assert!(row1_mean.abs() < 1e-6);
    }
    #[test]
    fn test_batch_norm_2d() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let eps = 1e-5;
        let result = executor.batch_norm(&handle_a, eps).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4, 2]);
        let result_view = result_dense.view();
        let col0_mean: f64 =
            (result_view[[0, 0]] + result_view[[1, 0]] + result_view[[2, 0]] + result_view[[3, 0]])
                / 4.0;
        assert!(col0_mean.abs() < 1e-6);
        let col1_mean: f64 =
            (result_view[[0, 1]] + result_view[[1, 1]] + result_view[[2, 1]] + result_view[[3, 1]])
                / 4.0;
        assert!(col1_mean.abs() < 1e-6);
    }
    #[test]
    fn test_binary_op_general_broadcast() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let b = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let result = executor
            .binary_op(BinaryOp::Add, &handle_a, &handle_b)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 11.0);
        assert_eq!(result_view[[0, 1]], 22.0);
        assert_eq!(result_view[[0, 2]], 33.0);
        assert_eq!(result_view[[1, 0]], 41.0);
        assert_eq!(result_view[[1, 1]], 52.0);
        assert_eq!(result_view[[1, 2]], 63.0);
    }
    #[test]
    fn test_where_op_basic() {
        let mut executor = CpuExecutor::new();
        let condition = DenseND::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
        let x = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let y = DenseND::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[2, 2]).unwrap();
        let cond_handle = TensorHandle::from_dense_auto(condition);
        let x_handle = TensorHandle::from_dense_auto(x);
        let y_handle = TensorHandle::from_dense_auto(y);
        let result = executor
            .where_op(&cond_handle, &x_handle, &y_handle)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 10.0);
        assert_eq!(result_view[[0, 1]], 200.0);
        assert_eq!(result_view[[1, 0]], 30.0);
        assert_eq!(result_view[[1, 1]], 400.0);
    }
    #[test]
    fn test_where_op_shape_mismatch() {
        let mut executor = CpuExecutor::new();
        let condition = DenseND::from_vec(vec![1.0, 0.0], &[2]).unwrap();
        let x = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let y = DenseND::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[2, 2]).unwrap();
        let cond_handle = TensorHandle::from_dense_auto(condition);
        let x_handle = TensorHandle::from_dense_auto(x);
        let y_handle = TensorHandle::from_dense_auto(y);
        let result = executor.where_op(&cond_handle, &x_handle, &y_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_masked_select_basic() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();
        let mask = DenseND::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();
        let data_handle = TensorHandle::from_dense_auto(data);
        let mask_handle = TensorHandle::from_dense_auto(mask);
        let result = executor.masked_select(&data_handle, &mask_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 10.0);
        assert_eq!(result_view[[1]], 30.0);
        assert_eq!(result_view[[2]], 50.0);
    }
    #[test]
    fn test_masked_select_no_match() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let mask = DenseND::from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();
        let data_handle = TensorHandle::from_dense_auto(data);
        let mask_handle = TensorHandle::from_dense_auto(mask);
        let result = executor.masked_select(&data_handle, &mask_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[0]);
    }
    #[test]
    fn test_modulo_basic() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![10.0, 11.0, 12.0, 13.0], &[2, 2]).unwrap();
        let handle = TensorHandle::from_dense_auto(data);
        let result = executor.modulo(&handle, 3.0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 1]] - 2.0_f64).abs() < 1e-10);
        assert!((result_view[[1, 0]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[1, 1]] - 1.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_modulo_zero_divisor() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![10.0, 20.0], &[2]).unwrap();
        let handle = TensorHandle::from_dense_auto(data);
        let result = executor.modulo(&handle, 0.0);
        assert!(result.is_err());
    }
    #[test]
    fn test_remainder_basic() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![7.0, 8.0, 9.0], &[3]).unwrap();
        let handle = TensorHandle::from_dense_auto(data);
        let result = executor.remainder(&handle, 4.0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 3.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 0.0_f64).abs() < 1e-10);
        assert!((result_view[[2]] - 1.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_max_pool_1d_basic() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0];
        let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[6]).unwrap());
        let result = executor.max_pool_1d(&handle, 2, 2).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 3.0);
        assert_eq!(result_view[[1]], 4.0);
        assert_eq!(result_view[[2]], 5.0);
    }
    #[test]
    fn test_max_pool_1d_overlapping() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[5]).unwrap());
        let result = executor.max_pool_1d(&handle, 3, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 5.0);
        assert_eq!(result_view[[1]], 8.0);
        assert_eq!(result_view[[2]], 8.0);
    }
    #[test]
    fn test_avg_pool_1d_basic() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 6.0, 2.0];
        let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[6]).unwrap());
        let result = executor.avg_pool_1d(&handle, 2, 2).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 2.0);
        assert_eq!(result_view[[1]], 3.0);
        assert_eq!(result_view[[2]], 4.0);
    }
    #[test]
    fn test_max_pool_2d_basic() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[4, 4]).unwrap());
        let result = executor.max_pool_2d(&handle, (2, 2), (2, 2)).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 6.0);
        assert_eq!(result_view[[0, 1]], 8.0);
        assert_eq!(result_view[[1, 0]], 14.0);
        assert_eq!(result_view[[1, 1]], 16.0);
    }
    #[test]
    fn test_avg_pool_2d_basic() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[4, 4]).unwrap());
        let result = executor.avg_pool_2d(&handle, (2, 2), (2, 2)).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 3.5);
        assert_eq!(result_view[[0, 1]], 5.5);
        assert_eq!(result_view[[1, 0]], 11.5);
        assert_eq!(result_view[[1, 1]], 13.5);
    }
    #[test]
    fn test_max_pool_2d_non_square() {
        let mut executor = CpuExecutor::new();
        let data: Vec<f64> = (1..=18).map(|x| x as f64).collect();
        let handle = TensorHandle::from_dense_auto(DenseND::from_vec(data, &[3, 6]).unwrap());
        let result = executor.max_pool_2d(&handle, (3, 2), (3, 2)).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 3]);
    }
    #[test]
    fn test_pooling_invalid_params() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let handle = TensorHandle::from_dense_auto(data);
        assert!(executor.max_pool_1d(&handle, 0, 1).is_err());
        assert!(executor.avg_pool_1d(&handle, 2, 0).is_err());
        assert!(executor.max_pool_1d(&handle, 10, 1).is_err());
    }
    #[test]
    fn test_conv1d_basic() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0, 0.0, -1.0], &[1, 1, 3]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv1d(&input_handle, &kernel_handle, None, 1, (0, 0))
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 3]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0, 0]] - -2.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 1]] - -2.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 2]] - -2.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_conv1d_with_bias() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0, 1.0], &[1, 1, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let bias = DenseND::from_vec(vec![10.0], &[1]).unwrap();
        let bias_handle = TensorHandle::from_dense_auto(bias);
        let result = executor
            .conv1d(&input_handle, &kernel_handle, Some(&bias_handle), 1, (0, 0))
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 3]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0, 0]] - 13.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 1]] - 15.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 2]] - 17.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_conv1d_with_padding() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0, 1.0, 1.0], &[1, 1, 3]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv1d(&input_handle, &kernel_handle, None, 1, (1, 1))
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 3]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0, 0]] - 3.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 1]] - 6.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 2]] - 5.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_conv1d_multi_channel() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv1d(&input_handle, &kernel_handle, None, 1, (0, 0))
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 2]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0, 0]] - 6.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 1]] - 8.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_conv2d_basic() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        )
        .unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![0.25, 0.25, 0.25, 0.25], &[1, 1, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv2d(&input_handle, &kernel_handle, None, (1, 1), (0, 0, 0, 0))
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 2, 2]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0, 0, 0]] - 3.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 0, 1]] - 4.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 1, 0]] - 6.0_f64).abs() < 1e-10);
        assert!((result_view[[0, 0, 1, 1]] - 7.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_conv2d_with_bias() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let bias = DenseND::from_vec(vec![100.0], &[1]).unwrap();
        let bias_handle = TensorHandle::from_dense_auto(bias);
        let result = executor
            .conv2d(
                &input_handle,
                &kernel_handle,
                Some(&bias_handle),
                (1, 1),
                (0, 0, 0, 0),
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 1, 1]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0, 0, 0]] - 105.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_conv2d_stride() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec((0..16).map(|x| x as f64).collect(), &[1, 1, 4, 4]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv2d(&input_handle, &kernel_handle, None, (2, 2), (0, 0, 0, 0))
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 2, 2]);
    }
    #[test]
    fn test_conv2d_invalid_channels() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec((0..18).map(|x| x as f64).collect(), &[1, 2, 3, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor.conv2d(&input_handle, &kernel_handle, None, (1, 1), (0, 0, 0, 0));
        assert!(result.is_err());
    }
    #[test]
    fn test_gather_basic() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4, 3],
        )
        .unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let indices = DenseND::from_vec(vec![0.0, 2.0], &[2]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);
        let result = executor.gather(&input_handle, 0, &indices_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 1.0);
        assert_eq!(result_view[[0, 1]], 2.0);
        assert_eq!(result_view[[0, 2]], 3.0);
        assert_eq!(result_view[[1, 0]], 7.0);
        assert_eq!(result_view[[1, 1]], 8.0);
        assert_eq!(result_view[[1, 2]], 9.0);
    }
    #[test]
    fn test_gather_out_of_bounds() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let indices = DenseND::from_vec(vec![0.0, 5.0], &[2]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);
        let result = executor.gather(&input_handle, 0, &indices_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_gather_duplicates() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3, 1]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let indices = DenseND::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);
        let result = executor.gather(&input_handle, 0, &indices_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 1]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 20.0);
        assert_eq!(result_view[[1, 0]], 20.0);
        assert_eq!(result_view[[2, 0]], 20.0);
    }
    #[test]
    fn test_scatter_basic() {
        let mut executor = CpuExecutor::new();
        let indices = DenseND::from_vec(vec![1.0, 3.0], &[2]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);
        let values = DenseND::from_vec(vec![10.0, 11.0, 20.0, 21.0], &[2, 2]).unwrap();
        let values_handle = TensorHandle::from_dense_auto(values);
        let result = executor
            .scatter(&[5, 2], 0, &indices_handle, &values_handle)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[5, 2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 0.0);
        assert_eq!(result_view[[0, 1]], 0.0);
        assert_eq!(result_view[[1, 0]], 10.0);
        assert_eq!(result_view[[1, 1]], 11.0);
        assert_eq!(result_view[[2, 0]], 0.0);
        assert_eq!(result_view[[2, 1]], 0.0);
        assert_eq!(result_view[[3, 0]], 20.0);
        assert_eq!(result_view[[3, 1]], 21.0);
        assert_eq!(result_view[[4, 0]], 0.0);
        assert_eq!(result_view[[4, 1]], 0.0);
    }
    #[test]
    fn test_scatter_out_of_bounds() {
        let mut executor = CpuExecutor::new();
        let indices = DenseND::from_vec(vec![0.0, 10.0], &[2]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);
        let values = DenseND::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let values_handle = TensorHandle::from_dense_auto(values);
        let result = executor.scatter(&[5, 1], 0, &indices_handle, &values_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_scatter_shape_mismatch() {
        let mut executor = CpuExecutor::new();
        let indices = DenseND::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);
        let values = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let values_handle = TensorHandle::from_dense_auto(values);
        let result = executor.scatter(&[5, 2], 0, &indices_handle, &values_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_conv3d_basic() {
        let mut executor = CpuExecutor::new();
        let input_data: Vec<f64> = (1..=27).map(|x| x as f64).collect();
        let input = DenseND::from_vec(input_data, &[1, 1, 3, 3, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel_data = vec![1.0; 8];
        let kernel = DenseND::from_vec(kernel_data, &[1, 1, 2, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv3d(
                &input_handle,
                &kernel_handle,
                None,
                (1, 1, 1),
                (0, 0, 0, 0, 0, 0),
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 2, 2, 2]);
        let result_view = result_dense.view();
        assert!(result_view[[0, 0, 0, 0, 0]] > 0.0);
    }
    #[test]
    fn test_conv3d_with_bias() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let bias = DenseND::from_vec(vec![100.0], &[1]).unwrap();
        let bias_handle = TensorHandle::from_dense_auto(bias);
        let result = executor
            .conv3d(
                &input_handle,
                &kernel_handle,
                Some(&bias_handle),
                (1, 1, 1),
                (0, 0, 0, 0, 0, 0),
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 1, 1, 1]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0, 0, 0, 0]] - 108.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_conv3d_with_stride() {
        let mut executor = CpuExecutor::new();
        let input_data: Vec<f64> = (0..64).map(|x| x as f64).collect();
        let input = DenseND::from_vec(input_data, &[1, 1, 4, 4, 4]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv3d(
                &input_handle,
                &kernel_handle,
                None,
                (2, 2, 2),
                (0, 0, 0, 0, 0, 0),
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 2, 2, 2]);
    }
    #[test]
    fn test_conv3d_with_padding() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor
            .conv3d(
                &input_handle,
                &kernel_handle,
                None,
                (1, 1, 1),
                (1, 1, 1, 1, 1, 1),
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 1, 3, 3, 3]);
    }
    #[test]
    fn test_conv3d_invalid_dimensions() {
        let mut executor = CpuExecutor::new();
        let input_data: Vec<f64> = (0..54).map(|x| x as f64).collect();
        let input = DenseND::from_vec(input_data, &[1, 2, 3, 3, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let kernel = DenseND::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).unwrap();
        let kernel_handle = TensorHandle::from_dense_auto(kernel);
        let result = executor.conv3d(
            &input_handle,
            &kernel_handle,
            None,
            (1, 1, 1),
            (0, 0, 0, 0, 0, 0),
        );
        assert!(result.is_err());
    }
    #[test]
    fn test_determinant_2x2() {
        let mut executor = CpuExecutor::new();
        let matrix = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.determinant(&matrix_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape().len(), 0);
        let result_view = result_dense.view();
        assert!((result_view[[]] - (-2.0_f64)).abs() < 1e-10);
    }
    #[test]
    fn test_determinant_3x3() {
        let mut executor = CpuExecutor::new();
        let matrix =
            DenseND::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0], &[3, 3]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.determinant(&matrix_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert!((result_view[[]] - 24.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_determinant_singular() {
        let mut executor = CpuExecutor::new();
        let matrix = DenseND::from_vec(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.determinant(&matrix_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        let val: f64 = result_view[[]];
        assert!(val.abs() < 1e-10);
    }
    #[test]
    fn test_determinant_batched() {
        let mut executor = CpuExecutor::new();
        let matrices =
            DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0], &[2, 2, 2]).unwrap();
        let matrices_handle = TensorHandle::from_dense_auto(matrices);
        let result = executor.determinant(&matrices_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2]);
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0_f64).abs() < 1e-10);
        assert!((result_view[[1]] - 6.0_f64).abs() < 1e-10);
    }
    #[test]
    fn test_determinant_invalid_shape() {
        let mut executor = CpuExecutor::new();
        let matrix = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.determinant(&matrix_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_matrix_inverse_2x2() {
        let mut executor = CpuExecutor::new();
        let matrix = DenseND::from_vec(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.matrix_inverse(&matrix_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        assert!((result_view[[0, 0]] - 0.6_f64).abs() < 1e-10);
        assert!((result_view[[0, 1]] - (-0.7_f64)).abs() < 1e-10);
        assert!((result_view[[1, 0]] - (-0.2_f64)).abs() < 1e-10);
        assert!((result_view[[1, 1]] - 0.4_f64).abs() < 1e-10);
    }
    #[test]
    fn test_matrix_inverse_identity() {
        let mut executor = CpuExecutor::new();
        let matrix =
            DenseND::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.matrix_inverse(&matrix_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 3]);
        let result_view = result_dense.view();
        for i in 0..3 {
            for j in 0..3 {
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                assert!((result_view[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn test_matrix_inverse_singular() {
        let mut executor = CpuExecutor::new();
        let matrix = DenseND::from_vec(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.matrix_inverse(&matrix_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_matrix_inverse_batched() {
        let mut executor = CpuExecutor::new();
        let matrices =
            DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]).unwrap();
        let matrices_handle = TensorHandle::from_dense_auto(matrices);
        let result = executor.matrix_inverse(&matrices_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2, 2]);
        let result_view = result_dense.view();
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    let expected: f64 = if i == j { 1.0 } else { 0.0 };
                    assert!((result_view[[b, i, j]] - expected).abs() < 1e-10);
                }
            }
        }
    }
    #[test]
    fn test_matrix_inverse_non_square() {
        let mut executor = CpuExecutor::new();
        let matrix = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let matrix_handle = TensorHandle::from_dense_auto(matrix);
        let result = executor.matrix_inverse(&matrix_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_solve_2x2_simple() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]).unwrap();
        let a_handle = TensorHandle::from_dense_auto(a);
        let b = DenseND::from_vec(vec![5.0, 5.0], &[2]).unwrap();
        let b_handle = TensorHandle::from_dense_auto(b);
        let result = executor.solve(&a_handle, &b_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2]);
        let result_view = result_dense.view();
        assert!((result_view[[0]] - 2.0_f64).abs() < 1e-9);
        assert!((result_view[[1]] - 1.0_f64).abs() < 1e-9);
    }
    #[test]
    fn test_solve_3x3() {
        let mut executor = CpuExecutor::new();
        let a =
            DenseND::from_vec(vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0], &[3, 3]).unwrap();
        let a_handle = TensorHandle::from_dense_auto(a);
        let b = DenseND::from_vec(vec![6.0, 9.0, 12.0], &[3]).unwrap();
        let b_handle = TensorHandle::from_dense_auto(b);
        let result = executor.solve(&a_handle, &b_handle).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        for i in 0..3 {
            assert!((result_view[[i]] - 3.0_f64).abs() < 1e-9);
        }
    }
    #[test]
    fn test_solve_singular() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        let a_handle = TensorHandle::from_dense_auto(a);
        let b = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b_handle = TensorHandle::from_dense_auto(b);
        let result = executor.solve(&a_handle, &b_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_solve_dimension_mismatch() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let a_handle = TensorHandle::from_dense_auto(a);
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b_handle = TensorHandle::from_dense_auto(b);
        let result = executor.solve(&a_handle, &b_handle);
        assert!(result.is_err());
    }
    #[test]
    fn test_solve_non_square_matrix() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let a_handle = TensorHandle::from_dense_auto(a);
        let b = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b_handle = TensorHandle::from_dense_auto(b);
        let result = executor.solve(&a_handle, &b_handle);
        assert!(result.is_err());
    }

    // Tests for advanced indexing operations
    #[test]
    fn test_advanced_gather_basic() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let indices = DenseND::from_vec(vec![0.0, 2.0, 4.0, 1.0], &[4]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);

        let result = executor
            .advanced_gather(&input_handle, 0, &indices_handle, false)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 10.0);
        assert_eq!(result_view[[1]], 30.0);
        assert_eq!(result_view[[2]], 50.0);
        assert_eq!(result_view[[3]], 20.0);
    }

    #[test]
    fn test_advanced_gather_negative_indices() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let indices = DenseND::from_vec(vec![-1.0, -2.0], &[2]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);

        let result = executor
            .advanced_gather(&input_handle, 0, &indices_handle, true)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 50.0); // -1 -> index 4
        assert_eq!(result_view[[1]], 40.0); // -2 -> index 3
    }

    #[test]
    fn test_advanced_scatter_replace() {
        let mut executor = CpuExecutor::new();
        let shape = vec![5];
        let indices = DenseND::from_vec(vec![0.0, 2.0, 4.0], &[3]).unwrap();
        let indices_handle = TensorHandle::from_dense_auto(indices);
        let values = DenseND::from_vec(vec![10.0, 30.0, 50.0], &[3]).unwrap();
        let values_handle = TensorHandle::from_dense_auto(values);

        let result = executor
            .advanced_scatter(
                &shape,
                0,
                &indices_handle,
                &values_handle,
                ScatterMode::Replace,
            )
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[5]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 10.0);
        assert_eq!(result_view[[1]], 0.0);
        assert_eq!(result_view[[2]], 30.0);
        assert_eq!(result_view[[3]], 0.0);
        assert_eq!(result_view[[4]], 50.0);
    }

    #[test]
    fn test_fancy_index_mask_basic() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let mask = DenseND::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0], &[5]).unwrap();
        let mask_handle = TensorHandle::from_dense_auto(mask);

        let result = executor
            .fancy_index_mask(&input_handle, &mask_handle)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 10.0);
        assert_eq!(result_view[[1]], 30.0);
        assert_eq!(result_view[[2]], 50.0);
    }

    #[test]
    fn test_fancy_index_mask_all_false() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);
        let mask = DenseND::from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();
        let mask_handle = TensorHandle::from_dense_auto(mask);

        let result = executor
            .fancy_index_mask(&input_handle, &mask_handle)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[0]);
    }

    // Tests for tensor manipulation operations
    #[test]
    fn test_tile_1d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.tile(&input_handle, &[3]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[9]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 1.0);
        assert_eq!(result_view[[1]], 2.0);
        assert_eq!(result_view[[2]], 3.0);
        assert_eq!(result_view[[3]], 1.0);
        assert_eq!(result_view[[4]], 2.0);
        assert_eq!(result_view[[5]], 3.0);
    }

    #[test]
    fn test_tile_2d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.tile(&input_handle, &[2, 3]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4, 6]);
        let result_view = result_dense.view();
        // Check first row of tiles
        assert_eq!(result_view[[0, 0]], 1.0);
        assert_eq!(result_view[[0, 1]], 2.0);
        assert_eq!(result_view[[0, 2]], 1.0);
        assert_eq!(result_view[[0, 3]], 2.0);
    }

    #[test]
    fn test_tile_invalid_reps() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.tile(&input_handle, &[2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_1d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.pad(&input_handle, &[(1, 2)], 0.0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[6]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 0.0);
        assert_eq!(result_view[[1]], 1.0);
        assert_eq!(result_view[[2]], 2.0);
        assert_eq!(result_view[[3]], 3.0);
        assert_eq!(result_view[[4]], 0.0);
        assert_eq!(result_view[[5]], 0.0);
    }

    #[test]
    fn test_pad_2d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.pad(&input_handle, &[(1, 1), (1, 1)], 0.0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4, 4]);
        let result_view = result_dense.view();
        // Check corners are padded
        assert_eq!(result_view[[0, 0]], 0.0);
        assert_eq!(result_view[[3, 3]], 0.0);
        // Check original values
        assert_eq!(result_view[[1, 1]], 1.0);
        assert_eq!(result_view[[1, 2]], 2.0);
        assert_eq!(result_view[[2, 1]], 3.0);
        assert_eq!(result_view[[2, 2]], 4.0);
    }

    #[test]
    fn test_pad_invalid_width() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.pad(&input_handle, &[(1, 1), (1, 1)], 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_flip_1d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.flip(&input_handle, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[5]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 5.0);
        assert_eq!(result_view[[1]], 4.0);
        assert_eq!(result_view[[2]], 3.0);
        assert_eq!(result_view[[3]], 2.0);
        assert_eq!(result_view[[4]], 1.0);
    }

    #[test]
    fn test_flip_2d_horizontal() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.flip(&input_handle, &[1]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 3]);
        let result_view = result_dense.view();
        // First row: [1, 2, 3] -> [3, 2, 1]
        assert_eq!(result_view[[0, 0]], 3.0);
        assert_eq!(result_view[[0, 1]], 2.0);
        assert_eq!(result_view[[0, 2]], 1.0);
        // Second row: [4, 5, 6] -> [6, 5, 4]
        assert_eq!(result_view[[1, 0]], 6.0);
        assert_eq!(result_view[[1, 1]], 5.0);
        assert_eq!(result_view[[1, 2]], 4.0);
    }

    #[test]
    fn test_flip_2d_vertical() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.flip(&input_handle, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 3]);
        let result_view = result_dense.view();
        // Rows are flipped
        assert_eq!(result_view[[0, 0]], 4.0);
        assert_eq!(result_view[[0, 1]], 5.0);
        assert_eq!(result_view[[0, 2]], 6.0);
        assert_eq!(result_view[[1, 0]], 1.0);
        assert_eq!(result_view[[1, 1]], 2.0);
        assert_eq!(result_view[[1, 2]], 3.0);
    }

    #[test]
    fn test_flip_2d_both_axes() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.flip(&input_handle, &[0, 1]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        let result_view = result_dense.view();
        // Original: [[1, 2], [3, 4]]
        // Flipped:  [[4, 3], [2, 1]]
        assert_eq!(result_view[[0, 0]], 4.0);
        assert_eq!(result_view[[0, 1]], 3.0);
        assert_eq!(result_view[[1, 0]], 2.0);
        assert_eq!(result_view[[1, 1]], 1.0);
    }

    #[test]
    fn test_flip_invalid_axis() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.flip(&input_handle, &[5]);
        assert!(result.is_err());
    }

    // Tests for squeeze operation
    #[test]
    fn test_squeeze_all() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.squeeze(&input_handle, None).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);
    }

    #[test]
    fn test_squeeze_specific_axis() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.squeeze(&input_handle, Some(&[0])).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
    }

    #[test]
    fn test_squeeze_invalid_axis() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.squeeze(&input_handle, Some(&[0]));
        assert!(result.is_err());
    }

    // Tests for unsqueeze operation
    #[test]
    fn test_unsqueeze_front() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.unsqueeze(&input_handle, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[1, 3]);
    }

    #[test]
    fn test_unsqueeze_end() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.unsqueeze(&input_handle, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 1]);
    }

    #[test]
    fn test_unsqueeze_invalid_axis() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.unsqueeze(&input_handle, 5);
        assert!(result.is_err());
    }

    // Tests for stack operation
    #[test]
    fn test_stack_1d() {
        let mut executor = CpuExecutor::new();
        let t1 = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let t2 = DenseND::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let h1 = TensorHandle::from_dense_auto(t1);
        let h2 = TensorHandle::from_dense_auto(t2);

        let result = executor.stack(&[h1, h2], 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 3]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0, 0]], 1.0);
        assert_eq!(result_view[[1, 0]], 4.0);
    }

    #[test]
    fn test_stack_2d() {
        let mut executor = CpuExecutor::new();
        let t1 = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t2 = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let h1 = TensorHandle::from_dense_auto(t1);
        let h2 = TensorHandle::from_dense_auto(t2);

        let result = executor.stack(&[h1, h2], 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_stack_shape_mismatch() {
        let mut executor = CpuExecutor::new();
        let t1 = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let t2 = DenseND::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
        let h1 = TensorHandle::from_dense_auto(t1);
        let h2 = TensorHandle::from_dense_auto(t2);

        let result = executor.stack(&[h1, h2], 0);
        assert!(result.is_err());
    }

    // Tests for repeat operation
    #[test]
    fn test_repeat_1d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.repeat(&input_handle, 2, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[6]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 1.0);
        assert_eq!(result_view[[1]], 1.0);
        assert_eq!(result_view[[2]], 2.0);
        assert_eq!(result_view[[3]], 2.0);
    }

    #[test]
    fn test_repeat_2d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.repeat(&input_handle, 3, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 6]);
    }

    // Tests for roll operation
    #[test]
    fn test_roll_positive() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.roll(&input_handle, 2, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 4.0);
        assert_eq!(result_view[[1]], 5.0);
        assert_eq!(result_view[[2]], 1.0);
        assert_eq!(result_view[[3]], 2.0);
        assert_eq!(result_view[[4]], 3.0);
    }

    #[test]
    fn test_roll_negative() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.roll(&input_handle, -2, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]], 3.0);
        assert_eq!(result_view[[1]], 4.0);
        assert_eq!(result_view[[2]], 5.0);
        assert_eq!(result_view[[3]], 1.0);
        assert_eq!(result_view[[4]], 2.0);
    }

    #[test]
    fn test_roll_2d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.roll(&input_handle, 1, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        // First row [1, 2, 3] -> [3, 1, 2]
        assert_eq!(result_view[[0, 0]], 3.0);
        assert_eq!(result_view[[0, 1]], 1.0);
        assert_eq!(result_view[[0, 2]], 2.0);
    }

    // Tests for argmax operation
    #[test]
    fn test_argmax_1d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.argmax(&input_handle, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape().len(), 0); // Scalar result
        let result_view = result_dense.view();
        assert_eq!(result_view[[]] as usize, 1); // Index of max value (5.0)
    }

    #[test]
    fn test_argmax_2d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 3.0, 2.0, 2.0, 4.0, 1.0], &[2, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.argmax(&input_handle, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]] as usize, 1); // Max in first row [1.0, 3.0, 2.0] is at index 1
        assert_eq!(result_view[[1]] as usize, 1); // Max in second row [2.0, 4.0, 1.0] is at index 1
    }

    // Tests for argmin operation
    #[test]
    fn test_argmin_1d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![5.0, 1.0, 3.0, 2.0, 4.0], &[5]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.argmin(&input_handle, 0).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();
        assert_eq!(result_view[[]] as usize, 1); // Index of min value (1.0)
    }

    #[test]
    fn test_argmin_2d() {
        let mut executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![3.0, 1.0, 2.0, 4.0, 2.0, 5.0], &[2, 3]).unwrap();
        let input_handle = TensorHandle::from_dense_auto(input);

        let result = executor.argmin(&input_handle, 1).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2]);
        let result_view = result_dense.view();
        assert_eq!(result_view[[0]] as usize, 1); // Min in first row is at index 1
        assert_eq!(result_view[[1]] as usize, 1); // Min in second row is at index 1
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
}
