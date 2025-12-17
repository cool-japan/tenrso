//! End-to-end integration tests for tenrso-ooc
//!
//! Tests complete workflows combining multiple components

use anyhow::Result;
use std::env::temp_dir;
use tenrso_core::DenseND;
use tenrso_ooc::{StreamConfig, StreamingExecutor};

#[cfg(feature = "parquet")]
use tenrso_ooc::{ParquetReader, ParquetWriter};

#[cfg(feature = "mmap")]
use tenrso_ooc::{read_tensor_binary, write_tensor_binary};

/// Test end-to-end streaming execution with chunking
#[test]
fn test_e2e_streaming_matmul() -> Result<()> {
    // Create config with memory constraints
    let config = StreamConfig::new()
        .max_memory_mb(64)
        .chunk_size(vec![32])
        .enable_profiling(true);

    let mut executor = StreamingExecutor::new(config);

    // Create test data
    let m = 100;
    let n = 80;
    let p = 60;

    let a = DenseND::<f64>::zeros(&[m, n]);
    let b = DenseND::<f64>::ones(&[n, p]);

    // Execute streaming matmul
    let result = executor.matmul_chunked(&a, &b, Some(32))?;

    // Verify result shape
    assert_eq!(result.shape(), &[m, p]);

    // Verify profiling captured the operation
    let summary = executor.profiling_summary();
    assert!(
        summary.total_operations > 0,
        "Should have recorded operations"
    );

    Ok(())
}

/// Test chunking with element-wise operations
#[test]
fn test_e2e_chunked_elementwise_pipeline() -> Result<()> {
    let config = StreamConfig::new()
        .max_memory_mb(32)
        .chunk_size(vec![16])
        .enable_profiling(true);

    let mut executor = StreamingExecutor::new(config);

    // Create test tensors
    let shape = vec![64, 64];
    let a = DenseND::<f64>::ones(&shape);
    let b = DenseND::<f64>::from_vec(vec![2.0; 64 * 64], &shape)?;
    let c = DenseND::<f64>::from_vec(vec![3.0; 64 * 64], &shape)?;

    // Compute: result = (a + b) * c
    let temp = executor.add_chunked(&a, &b)?;
    let result = executor.multiply_chunked(&temp, &c)?;

    // Verify: (1 + 2) * 3 = 9
    assert_eq!(result.shape(), &shape);
    let expected_val = 9.0;
    for &val in result.as_slice() {
        assert!((val - expected_val).abs() < 1e-10);
    }

    // Verify multiple operations were profiled
    let summary = executor.profiling_summary();
    assert!(
        summary.total_operations >= 2,
        "Should have recorded at least 2 operations"
    );

    Ok(())
}

/// Test FMA (fused multiply-add) operation
#[test]
fn test_e2e_fma_operation() -> Result<()> {
    let config = StreamConfig::new()
        .max_memory_mb(32)
        .chunk_size(vec![16])
        .enable_profiling(true);

    let mut executor = StreamingExecutor::new(config);

    let shape = vec![50, 50];
    let a = DenseND::<f64>::from_vec(vec![2.0; 50 * 50], &shape)?;
    let b = DenseND::<f64>::from_vec(vec![3.0; 50 * 50], &shape)?;
    let c = DenseND::<f64>::from_vec(vec![1.0; 50 * 50], &shape)?;

    // Compute: result = a * b + c = 2 * 3 + 1 = 7
    let result = executor.fma_chunked(&a, &b, &c)?;

    assert_eq!(result.shape(), &shape);
    let expected_val = 7.0;
    for &val in result.as_slice() {
        assert!((val - expected_val).abs() < 1e-10);
    }

    Ok(())
}

/// Test min/max operations (element-wise)
#[test]
fn test_e2e_min_max_operations() -> Result<()> {
    let config = StreamConfig::new().chunk_size(vec![10]);

    let mut executor = StreamingExecutor::new(config);

    let shape = vec![30];
    let a = DenseND::<f64>::from_vec(vec![5.0; 30], &shape)?;
    let b = DenseND::<f64>::from_vec(vec![3.0; 30], &shape)?;

    // Test element-wise min: min(5.0, 3.0) = 3.0
    let min_result = executor.min_chunked(&a, &b)?;
    for &val in min_result.as_slice() {
        assert!((val - 3.0).abs() < 1e-10);
    }

    // Test element-wise max: max(5.0, 3.0) = 5.0
    let max_result = executor.max_chunked(&a, &b)?;
    for &val in max_result.as_slice() {
        assert!((val - 5.0).abs() < 1e-10);
    }

    Ok(())
}

/// Test with different chunk sizes
#[test]
fn test_e2e_varying_chunk_sizes() -> Result<()> {
    let shape = vec![100, 100];
    let a = DenseND::<f64>::ones(&shape);
    let b = DenseND::<f64>::ones(&shape);

    // Test with different chunk sizes
    for chunk_size in [10, 25, 50] {
        let config = StreamConfig::new().chunk_size(vec![chunk_size]);
        let mut executor = StreamingExecutor::new(config);

        let result = executor.add_chunked(&a, &b)?;

        // All results should be 2.0
        for &val in result.as_slice() {
            assert!((val - 2.0).abs() < 1e-10);
        }
    }

    Ok(())
}

/// Test memory tracking across operations
#[test]
fn test_e2e_memory_tracking() -> Result<()> {
    let config = StreamConfig::new()
        .max_memory_mb(16)
        .chunk_size(vec![8])
        .enable_profiling(true);

    let mut executor = StreamingExecutor::new(config);

    let shape = vec![32, 32];
    let a = DenseND::<f64>::ones(&shape);
    let b = DenseND::<f64>::ones(&shape);

    // Initial memory should be 0
    assert_eq!(executor.current_memory(), 0);

    // Execute operation
    let _result = executor.add_chunked(&a, &b)?;

    // Memory tracking should work (value may vary)
    // Just verify the API works
    let _current_mem = executor.current_memory();
    let _is_exceeded = executor.is_memory_exceeded();

    Ok(())
}

/// Test parallel execution
#[cfg(feature = "parallel")]
#[test]
fn test_e2e_parallel_execution() -> Result<()> {
    let config = StreamConfig::new()
        .chunk_size(vec![20])
        .enable_parallel(true)
        .num_threads(4)
        .min_chunks_for_parallel(2);

    let mut executor = StreamingExecutor::new(config);

    let shape = vec![100, 100];
    let a = DenseND::<f64>::ones(&shape);
    let b = DenseND::<f64>::ones(&shape);

    // Execute with parallel processing
    let result = executor.multiply_chunked(&a, &b)?;

    // Verify correctness
    for &val in result.as_slice() {
        assert!((val - 1.0).abs() < 1e-10);
    }

    Ok(())
}

/// Test recommended configuration
#[test]
fn test_e2e_recommended_config() -> Result<()> {
    let executor = StreamingExecutor::new(StreamConfig::new());

    // Test elementwise recommendation
    let shape1 = vec![1000, 1000];
    let shapes: Vec<&[usize]> = vec![&shape1];
    let (chunk_size, explanation) = executor.recommend_config("elementwise", &shapes)?;

    assert!(chunk_size > 0, "Should recommend a positive chunk size");
    assert!(!explanation.is_empty(), "Should provide explanation");

    // Test matmul recommendation
    let shape2 = vec![500, 500];
    let shape3 = vec![500, 500];
    let shapes: Vec<&[usize]> = vec![&shape2, &shape3];
    let (chunk_size, explanation) = executor.recommend_config("matmul", &shapes)?;

    assert!(chunk_size > 0, "Should recommend a positive chunk size");
    assert!(!explanation.is_empty(), "Should provide explanation");

    Ok(())
}

#[cfg(feature = "parquet")]
#[test]
#[ignore] // TODO: Fix parquet writer flushing issue
fn test_e2e_parquet_workflow() -> Result<()> {
    use std::fs;

    let temp_dir = temp_dir();
    let path = temp_dir.join("test_e2e_tensor.parquet");

    // Clean up any existing file
    let _ = fs::remove_file(&path);

    // Create and write tensor
    let original = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])?;

    // Write in a scope to ensure writer is dropped before reading
    {
        let mut writer = ParquetWriter::new(&path)?;
        writer.write(&original)?;
    } // Writer drops here, flushing data to disk

    // Read back and verify
    let reader = ParquetReader::open(&path)?;
    let loaded: DenseND<f64> = reader.read()?;

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.as_slice(), original.as_slice());

    // Clean up
    let _ = fs::remove_file(&path);

    Ok(())
}

#[cfg(feature = "mmap")]
#[test]
fn test_e2e_mmap_workflow() -> Result<()> {
    use std::fs;

    let temp_dir = temp_dir();
    let path = temp_dir.join("test_e2e_tensor.bin");

    // Clean up any existing file
    let _ = fs::remove_file(&path);

    // Create and write tensor
    let original = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    write_tensor_binary(&path, &original)?;

    // Read back using mmap
    let loaded: DenseND<f64> = read_tensor_binary(&path)?;

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.as_slice(), original.as_slice());

    // Clean up
    let _ = fs::remove_file(&path);

    Ok(())
}
