//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::{StreamConfig, StreamingExecutor};
    use crate::prefetch::PrefetchStrategy;
    use tenrso_core::DenseND;
    #[test]
    fn test_streaming_executor_creation() {
        let config = StreamConfig::new().max_memory_mb(512).chunk_size(vec![128]);
        let executor = StreamingExecutor::new(config);
        assert_eq!(executor.current_memory(), 0);
        assert!(!executor.is_memory_exceeded());
    }
    #[test]
    fn test_matmul_chunked_small() {
        let config = StreamConfig::new().max_memory_mb(512).chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let result = executor.matmul_chunked(&a, &b, Some(1)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 19.0).abs() < 1e-10);
        assert!((data[1] - 22.0).abs() < 1e-10);
        assert!((data[2] - 43.0).abs() < 1e-10);
        assert!((data[3] - 50.0).abs() < 1e-10);
    }
    #[test]
    fn test_matmul_chunked_larger() {
        let config = StreamConfig::new().chunk_size(vec![32]);
        let mut executor = StreamingExecutor::new(config);
        let m = 100;
        let n = 80;
        let p = 60;
        let a_data: Vec<f64> = (0..m * n).map(|i| i as f64).collect();
        let b_data: Vec<f64> = (0..n * p).map(|i| (i % 10) as f64).collect();
        let a = DenseND::<f64>::from_vec(a_data, &[m, n]).unwrap();
        let b = DenseND::<f64>::from_vec(b_data, &[n, p]).unwrap();
        let result = executor.matmul_chunked(&a, &b, Some(32)).unwrap();
        assert_eq!(result.shape(), &[m, p]);
        let expected = a.matmul(&b).unwrap();
        let result_data = result.as_slice();
        let expected_data = expected.as_slice();
        for i in 0..result_data.len() {
            assert!((result_data[i] - expected_data[i]).abs() < 1e-8);
        }
    }
    #[test]
    fn test_add_chunked() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let result = executor.add_chunked(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 6.0).abs() < 1e-10);
        assert!((data[1] - 8.0).abs() < 1e-10);
        assert!((data[2] - 10.0).abs() < 1e-10);
        assert!((data[3] - 12.0).abs() < 1e-10);
    }
    #[test]
    fn test_multiply_chunked() {
        let config = StreamConfig::new().chunk_size(vec![3]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let result = executor.multiply_chunked(&a, &b).unwrap();
        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 2.0).abs() < 1e-10);
        assert!((data[1] - 6.0).abs() < 1e-10);
        assert!((data[2] - 12.0).abs() < 1e-10);
        assert!((data[3] - 20.0).abs() < 1e-10);
    }
    #[test]
    fn test_subtract_chunked() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![10.0, 9.0, 8.0, 7.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = executor.subtract_chunked(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 9.0).abs() < 1e-10);
        assert!((data[1] - 7.0).abs() < 1e-10);
        assert!((data[2] - 5.0).abs() < 1e-10);
        assert!((data[3] - 3.0).abs() < 1e-10);
    }
    #[test]
    fn test_divide_chunked() {
        let config = StreamConfig::new().chunk_size(vec![3]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 4.0, 5.0, 8.0], &[4]).unwrap();
        let result = executor.divide_chunked(&a, &b).unwrap();
        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 5.0).abs() < 1e-10);
        assert!((data[1] - 5.0).abs() < 1e-10);
        assert!((data[2] - 6.0).abs() < 1e-10);
        assert!((data[3] - 5.0).abs() < 1e-10);
    }
    #[test]
    fn test_fma_chunked_basic() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let result = executor.fma_chunked(&a, &b, &c).unwrap();
        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 11.0).abs() < 1e-10);
        assert!((data[1] - 20.0).abs() < 1e-10);
        assert!((data[2] - 31.0).abs() < 1e-10);
        assert!((data[3] - 44.0).abs() < 1e-10);
    }
    #[test]
    fn test_fma_chunked_2d() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();
        let result = executor.fma_chunked(&a, &b, &c).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 3.0).abs() < 1e-10);
        assert!((data[1] - 7.0).abs() < 1e-10);
        assert!((data[2] - 13.0).abs() < 1e-10);
        assert!((data[3] - 21.0).abs() < 1e-10);
    }
    #[test]
    fn test_fma_vs_separate_ops_accuracy() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![1e10, 1e-10, 0.1, 0.2], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1e-10, 1e10, 0.3, 0.4], &[4]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![1.0, 1.0, 0.5, 0.6], &[4]).unwrap();
        let fma_result = executor.fma_chunked(&a, &b, &c).unwrap();
        let mul_result = executor.multiply_chunked(&a, &b).unwrap();
        let separate_result = executor.add_chunked(&mul_result, &c).unwrap();
        let fma_data = fma_result.as_slice();
        let separate_data = separate_result.as_slice();
        for i in 0..fma_data.len() {
            let diff = (fma_data[i] - separate_data[i]).abs();
            assert!(
                diff < 1e-8,
                "FMA vs separate difference at index {}: {}",
                i,
                diff
            );
        }
    }
    #[test]
    fn test_fma_shape_mismatch() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![7.0, 8.0], &[2]).unwrap();
        let result = executor.fma_chunked(&a, &b, &c);
        assert!(result.is_err(), "Should fail with shape mismatch");
    }
    #[test]
    fn test_min_chunked_basic() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 3.0, 1.0, 5.0], &[4]).unwrap();
        let result = executor.min_chunked(&a, &b).unwrap();
        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 2.0).abs() < 1e-10);
        assert!((data[1] - 1.0).abs() < 1e-10);
        assert!((data[2] - 1.0).abs() < 1e-10);
        assert!((data[3] - 2.0).abs() < 1e-10);
    }
    #[test]
    fn test_max_chunked_basic() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 3.0, 1.0, 5.0], &[4]).unwrap();
        let result = executor.max_chunked(&a, &b).unwrap();
        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 3.0).abs() < 1e-10);
        assert!((data[1] - 3.0).abs() < 1e-10);
        assert!((data[2] - 4.0).abs() < 1e-10);
        assert!((data[3] - 5.0).abs() < 1e-10);
    }
    #[test]
    fn test_min_max_with_negatives() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![-5.0, -2.0, 0.0, 3.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![-3.0, -4.0, -1.0, 2.0], &[4]).unwrap();
        let min_result = executor.min_chunked(&a, &b).unwrap();
        let max_result = executor.max_chunked(&a, &b).unwrap();
        let min_data = min_result.as_slice();
        let max_data = max_result.as_slice();
        assert!((min_data[0] - (-5.0)).abs() < 1e-10);
        assert!((min_data[1] - (-4.0)).abs() < 1e-10);
        assert!((min_data[2] - (-1.0)).abs() < 1e-10);
        assert!((min_data[3] - 2.0).abs() < 1e-10);
        assert!((max_data[0] - (-3.0)).abs() < 1e-10);
        assert!((max_data[1] - (-2.0)).abs() < 1e-10);
        assert!((max_data[2] - 0.0).abs() < 1e-10);
        assert!((max_data[3] - 3.0).abs() < 1e-10);
    }
    #[test]
    fn test_min_max_2d() {
        let config = StreamConfig::new().chunk_size(vec![1]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![4.0, 2.0, 1.0, 6.0], &[2, 2]).unwrap();
        let min_result = executor.min_chunked(&a, &b).unwrap();
        let max_result = executor.max_chunked(&a, &b).unwrap();
        assert_eq!(min_result.shape(), &[2, 2]);
        assert_eq!(max_result.shape(), &[2, 2]);
        let min_data = min_result.as_slice();
        let max_data = max_result.as_slice();
        assert!((min_data[0] - 1.0).abs() < 1e-10);
        assert!((min_data[1] - 2.0).abs() < 1e-10);
        assert!((min_data[2] - 1.0).abs() < 1e-10);
        assert!((min_data[3] - 2.0).abs() < 1e-10);
        assert!((max_data[0] - 4.0).abs() < 1e-10);
        assert!((max_data[1] - 5.0).abs() < 1e-10);
        assert!((max_data[2] - 3.0).abs() < 1e-10);
        assert!((max_data[3] - 6.0).abs() < 1e-10);
    }
    #[test]
    fn test_min_max_special_values() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::<f64>::from_vec(vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0], &[4])
            .unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 1.0, -0.0, 0.0], &[4]).unwrap();
        let min_result = executor.min_chunked(&a, &b).unwrap();
        let max_result = executor.max_chunked(&a, &b).unwrap();
        let min_data = min_result.as_slice();
        let max_data = max_result.as_slice();
        assert_eq!(min_data[0], 1.0);
        assert_eq!(min_data[1], f64::NEG_INFINITY);
        assert_eq!(max_data[0], f64::INFINITY);
        assert_eq!(max_data[1], 1.0);
    }
    #[test]
    #[cfg(feature = "mmap")]
    fn test_spill_and_load() {
        let config = StreamConfig::new().enable_spill(true);
        let mut executor = StreamingExecutor::new(config);
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let path = executor.spill_to_disk(&tensor).unwrap();
        assert!(path.exists());
        let loaded = executor.load_from_disk(&path).unwrap();
        assert_eq!(loaded.shape(), tensor.shape());
        assert_eq!(loaded.as_slice(), tensor.as_slice());
        let _ = std::fs::remove_file(path);
    }
    #[test]
    #[cfg(feature = "mmap")]
    fn test_spill_disabled() {
        let config = StreamConfig::new().enable_spill(false);
        let mut executor = StreamingExecutor::new(config);
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert!(executor.spill_to_disk(&tensor).is_err());
    }
    #[test]
    fn test_memory_tracking() {
        let config = StreamConfig::new().max_memory_mb(1);
        let mut executor = StreamingExecutor::new(config);
        assert_eq!(executor.current_memory(), 0);
        assert!(!executor.is_memory_exceeded());
        let a = DenseND::<f64>::zeros(&[200, 200]);
        let b = DenseND::<f64>::zeros(&[200, 200]);
        let _ = executor.matmul_chunked(&a, &b, Some(50)).unwrap();
        assert!(executor.current_memory() > 0);
    }
    #[test]
    fn test_compute_optimal_matmul_chunk_size() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);
        let chunk_size = executor
            .compute_optimal_matmul_chunk_size(&[100, 50], &[50, 80], None)
            .unwrap();
        assert!(chunk_size > 0);
        assert!(chunk_size <= 100);
    }
    #[test]
    fn test_compute_optimal_elementwise_chunk_size() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);
        let chunk_size = executor
            .compute_optimal_elementwise_chunk_size(&[1000, 500], None)
            .unwrap();
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000);
    }
    #[test]
    fn test_estimate_throughput_matmul() {
        let config = StreamConfig::new();
        let executor = StreamingExecutor::new(config);
        let (gflops, bandwidth) = executor
            .estimate_throughput("matmul", &[100, 50], Some(&[50, 80]), 32)
            .unwrap();
        assert!(gflops > 0.0);
        assert!(bandwidth > 0.0);
    }
    #[test]
    fn test_estimate_throughput_elementwise() {
        let config = StreamConfig::new();
        let executor = StreamingExecutor::new(config);
        let (gflops, bandwidth) = executor
            .estimate_throughput("elementwise", &[1000, 500], None, 100)
            .unwrap();
        assert!(gflops > 0.0);
        assert!(bandwidth > 0.0);
    }
    #[test]
    fn test_recommend_config_matmul() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);
        let (chunk_size, explanation) = executor
            .recommend_config("matmul", &[&[100, 50], &[50, 80]])
            .unwrap();
        assert!(chunk_size > 0);
        assert!(chunk_size <= 100);
        assert!(explanation.contains("Recommended chunk size"));
        assert!(explanation.contains("GFlops"));
        assert!(explanation.contains("GB/s"));
    }
    #[test]
    fn test_recommend_config_elementwise() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);
        let (chunk_size, explanation) = executor
            .recommend_config("elementwise", &[&[1000, 500]])
            .unwrap();
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000);
        assert!(explanation.contains("Recommended chunk size"));
    }
    #[test]
    fn test_auto_tuning_respects_memory_limit() {
        let config = StreamConfig::new().max_memory_mb(1);
        let executor = StreamingExecutor::new(config);
        let chunk_size = executor
            .compute_optimal_matmul_chunk_size(&[1000, 1000], &[1000, 1000], Some(1))
            .unwrap();
        assert!(chunk_size > 0);
        assert!(chunk_size < 100);
    }
    #[test]
    fn test_profiling_matmul() {
        let config = StreamConfig::new().enable_profiling(true);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[100, 80], 1.0);
        let b = DenseND::from_elem(&[80, 60], 2.0);
        let _ = executor.matmul_chunked(&a, &b, Some(50)).unwrap();
        let stats = executor.profiler().get_operation_stats("matmul_chunked");
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.call_count, 1);
        assert!(stats.total_time.as_nanos() > 0);
        assert!(stats.bytes_read > 0);
        assert!(stats.bytes_written > 0);
    }
    #[test]
    fn test_profiling_elementwise() {
        let config = StreamConfig::new()
            .enable_profiling(true)
            .chunk_size(vec![100]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[200, 100], 1.0);
        let b = DenseND::from_elem(&[200, 100], 2.0);
        let _ = executor.add_chunked(&a, &b).unwrap();
        let _ = executor.multiply_chunked(&a, &b).unwrap();
        let add_stats = executor.profiler().get_operation_stats("add_chunked");
        let mult_stats = executor.profiler().get_operation_stats("multiply_chunked");
        assert!(add_stats.is_some());
        assert!(mult_stats.is_some());
        assert_eq!(add_stats.unwrap().call_count, 1);
        assert_eq!(mult_stats.unwrap().call_count, 1);
    }
    #[test]
    fn test_profiling_summary() {
        let config = StreamConfig::new()
            .enable_profiling(true)
            .chunk_size(vec![30]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[50, 40], 1.0);
        let b = DenseND::from_elem(&[40, 30], 2.0);
        let _ = executor.matmul_chunked(&a, &b, None).unwrap();
        let _ = executor.add_chunked(&a, &a).unwrap();
        let summary = executor.profiling_summary();
        assert!(summary.total_operations >= 2);
        assert!(summary.total_time.as_nanos() > 0);
        assert!(summary.total_io_bytes > 0);
    }
    #[test]
    fn test_profiling_disabled() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[50, 40], 1.0);
        let b = DenseND::from_elem(&[40, 30], 2.0);
        let _ = executor.matmul_chunked(&a, &b, None).unwrap();
        assert!(!executor.profiler().is_enabled());
    }
    #[test]
    fn test_prefetcher_config() {
        let config = StreamConfig::new()
            .enable_prefetching(true)
            .prefetch_strategy(PrefetchStrategy::Sequential)
            .prefetch_queue_size(8);
        let executor = StreamingExecutor::new(config);
        assert!(executor.prefetcher().is_enabled());
    }
    #[test]
    fn test_profiler_accessor() {
        let config = StreamConfig::new().enable_profiling(true);
        let mut executor = StreamingExecutor::new(config);
        executor.profiler_mut().set_enabled(false);
        assert!(!executor.profiler().is_enabled());
        executor.profiler_mut().set_enabled(true);
        assert!(executor.profiler().is_enabled());
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_matmul() {
        let config = StreamConfig::new().enable_parallel(true).num_threads(4);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[400, 300], 2.0);
        let b = DenseND::from_elem(&[300, 200], 3.0);
        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();
        assert_eq!(result.shape(), vec![400, 200]);
        let expected_value = 2.0 * 3.0 * 300.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_elementwise() {
        let config = StreamConfig::new()
            .enable_parallel(true)
            .num_threads(4)
            .chunk_size(vec![100]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[400, 300], 2.0);
        let b = DenseND::from_elem(&[400, 300], 3.0);
        let result_add = executor.add_chunked(&a, &b).unwrap();
        assert_eq!(result_add.shape(), vec![400, 300]);
        let add_slice = result_add.as_slice();
        for &val in add_slice.iter().take(10) {
            assert!((val - 5.0).abs() < 1e-10);
        }
        let result_mul = executor.multiply_chunked(&a, &b).unwrap();
        assert_eq!(result_mul.shape(), vec![400, 300]);
        let mul_slice = result_mul.as_slice();
        for &val in mul_slice.iter().take(10) {
            assert!((val - 6.0).abs() < 1e-10);
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_disabled() {
        let config = StreamConfig::new().enable_parallel(false);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[200, 150], 2.0);
        let b = DenseND::from_elem(&[150, 100], 3.0);
        let result = executor.matmul_chunked(&a, &b, Some(50)).unwrap();
        assert_eq!(result.shape(), vec![200, 100]);
        let expected_value = 2.0 * 3.0 * 150.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_vs_sequential_correctness() {
        let a = DenseND::from_elem(&[200, 150], 2.0);
        let b = DenseND::from_elem(&[150, 100], 3.0);
        let config_parallel = StreamConfig::new().enable_parallel(true).num_threads(4);
        let mut executor_parallel = StreamingExecutor::new(config_parallel);
        let result_parallel = executor_parallel.matmul_chunked(&a, &b, Some(50)).unwrap();
        let config_sequential = StreamConfig::new().enable_parallel(false);
        let mut executor_sequential = StreamingExecutor::new(config_sequential);
        let result_sequential = executor_sequential
            .matmul_chunked(&a, &b, Some(50))
            .unwrap();
        assert_eq!(result_parallel.shape(), result_sequential.shape());
        let parallel_slice = result_parallel.as_slice();
        let sequential_slice = result_sequential.as_slice();
        for (p, s) in parallel_slice.iter().zip(sequential_slice.iter()) {
            assert!(
                (p - s).abs() < 1e-10,
                "Parallel and sequential results differ"
            );
        }
    }
    #[test]
    fn test_parallel_config_default() {
        let config = StreamConfig::new();
        #[cfg(feature = "parallel")]
        assert!(config.enable_parallel);
        #[cfg(not(feature = "parallel"))]
        assert!(config.enable_parallel);
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_small_chunks() {
        let config = StreamConfig::new()
            .enable_parallel(true)
            .enable_profiling(true);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[200, 100], 2.0);
        let b = DenseND::from_elem(&[100, 80], 3.0);
        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();
        assert_eq!(result.shape(), vec![200, 80]);
        let expected_value = 2.0 * 3.0 * 100.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_large_chunks() {
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(4);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[800, 100], 2.0);
        let b = DenseND::from_elem(&[100, 80], 3.0);
        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();
        assert_eq!(result.shape(), vec![800, 80]);
        let expected_value = 2.0 * 3.0 * 100.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_custom() {
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(10);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[800, 100], 2.0);
        let b = DenseND::from_elem(&[100, 80], 3.0);
        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();
        assert_eq!(result.shape(), vec![800, 80]);
        let expected_value = 2.0 * 3.0 * 100.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_disabled() {
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(0);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[100, 80], 2.0);
        let b = DenseND::from_elem(&[80, 60], 3.0);
        let result = executor.matmul_chunked(&a, &b, Some(200)).unwrap();
        assert_eq!(result.shape(), vec![100, 60]);
        let expected_value = 2.0 * 3.0 * 80.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }
    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_elementwise() {
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(4)
            .chunk_size(vec![100]);
        let mut executor = StreamingExecutor::new(config);
        let a = DenseND::from_elem(&[200, 100], 2.0);
        let b = DenseND::from_elem(&[200, 100], 3.0);
        let result = executor.add_chunked(&a, &b).unwrap();
        assert_eq!(result.shape(), vec![200, 100]);
        let add_slice = result.as_slice();
        for &val in add_slice.iter().take(10) {
            assert!((val - 5.0).abs() < 1e-10);
        }
        let a_large = DenseND::from_elem(&[500, 100], 2.0);
        let b_large = DenseND::from_elem(&[500, 100], 3.0);
        let result_large = executor.add_chunked(&a_large, &b_large).unwrap();
        assert_eq!(result_large.shape(), vec![500, 100]);
        let large_slice = result_large.as_slice();
        for &val in large_slice.iter().take(10) {
            assert!((val - 5.0).abs() < 1e-10);
        }
    }
    #[test]
    fn test_adaptive_threshold_config() {
        let config = StreamConfig::new();
        assert_eq!(config.min_chunks_for_parallel, 4);
        let custom_config = StreamConfig::new().min_chunks_for_parallel(8);
        assert_eq!(custom_config.min_chunks_for_parallel, 8);
        let disabled_config = StreamConfig::new().min_chunks_for_parallel(0);
        assert_eq!(disabled_config.min_chunks_for_parallel, 0);
    }
}
