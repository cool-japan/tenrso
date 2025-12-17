//! Integration tests for compression support in memory management

#[cfg(all(feature = "compression", feature = "mmap"))]
mod compression_tests {
    use tenrso_core::DenseND;
    use tenrso_ooc::{AccessPattern, CompressionCodec, MemoryManager, SpillPolicy};

    #[test]
    fn test_compression_with_lz4() {
        #[cfg(feature = "lz4-compression")]
        {
            let temp_dir = std::env::temp_dir();

            // Create memory manager with LZ4 compression
            let mut manager = MemoryManager::new()
                .max_memory_mb(1) // Very small limit to force spilling
                .spill_policy(SpillPolicy::LRU)
                .compression(CompressionCodec::Lz4)
                .temp_dir(&temp_dir);

            // Create a highly compressible tensor (repeated values)
            let data = vec![42.0f64; 10000];
            let tensor = DenseND::from_vec(data.clone(), &[100, 100]).unwrap();

            // Register the chunk
            manager
                .register_chunk("test_chunk", tensor, AccessPattern::ReadOnce)
                .unwrap();

            // Force spill by allocating more chunks
            let small_tensors: Vec<_> = (0..10)
                .map(|i| {
                    let data = vec![i as f64; 5000];
                    DenseND::from_vec(data, &[50, 100]).unwrap()
                })
                .collect();

            for (i, tensor) in small_tensors.into_iter().enumerate() {
                let _ = manager.register_chunk(
                    &format!("chunk_{}", i),
                    tensor,
                    AccessPattern::ReadOnce,
                );
            }

            // Access the original chunk (should load from disk with decompression)
            let loaded = manager.access_chunk("test_chunk").unwrap();
            assert_eq!(loaded.shape(), &[100, 100]);

            // Verify data integrity
            let loaded_slice = loaded.as_slice();
            assert_eq!(loaded_slice.len(), 10000);
            assert!(loaded_slice.iter().all(|&x| x == 42.0));

            // Cleanup
            let _ = manager.cleanup();
        }
    }

    #[test]
    fn test_compression_with_zstd() {
        #[cfg(feature = "zstd-compression")]
        {
            let temp_dir = std::env::temp_dir();

            // Create memory manager with Zstd compression
            let mut manager = MemoryManager::new()
                .max_memory_mb(1)
                .spill_policy(SpillPolicy::LRU)
                .compression(CompressionCodec::Zstd { level: 3 })
                .temp_dir(&temp_dir);

            // Create a tensor with some pattern
            let data: Vec<f64> = (0..10000).map(|i| (i % 100) as f64).collect();
            let tensor = DenseND::from_vec(data.clone(), &[100, 100]).unwrap();

            manager
                .register_chunk("pattern_chunk", tensor, AccessPattern::ReadMany)
                .unwrap();

            // Force spill
            let large_tensor = DenseND::from_vec(vec![1.0; 100000], &[100, 1000]).unwrap();
            let _ = manager.register_chunk("large", large_tensor, AccessPattern::ReadOnce);

            // Access and verify
            let loaded = manager.access_chunk("pattern_chunk").unwrap();
            let loaded_slice = loaded.as_slice();

            for (i, &val) in loaded_slice.iter().enumerate() {
                assert_eq!(val, (i % 100) as f64);
            }

            let _ = manager.cleanup();
        }
    }

    #[test]
    fn test_compression_no_compression() {
        let temp_dir = std::env::temp_dir();

        // Test with no compression
        let mut manager = MemoryManager::new()
            .max_memory_mb(1)
            .compression(CompressionCodec::None)
            .temp_dir(&temp_dir);

        let pi = std::f64::consts::PI;
        let tensor = DenseND::from_vec(vec![pi; 5000], &[50, 100]).unwrap();
        manager
            .register_chunk("uncompressed", tensor, AccessPattern::ReadOnce)
            .unwrap();

        // Force spill
        let _ = manager.register_chunk(
            "large",
            DenseND::from_vec(vec![2.71; 100000], &[100, 1000]).unwrap(),
            AccessPattern::ReadOnce,
        );

        let loaded = manager.access_chunk("uncompressed").unwrap();
        assert!(loaded.as_slice().iter().all(|&x| x == pi));

        let _ = manager.cleanup();
    }

    #[test]
    fn test_compression_multiple_spills_and_loads() {
        #[cfg(feature = "lz4-compression")]
        {
            let temp_dir = std::env::temp_dir();
            let mut manager = MemoryManager::new()
                .max_memory_mb(2)
                .compression(CompressionCodec::Lz4)
                .temp_dir(&temp_dir);

            // Register multiple chunks
            for i in 0..5 {
                let data = vec![i as f64; 10000];
                let tensor = DenseND::from_vec(data, &[100, 100]).unwrap();
                manager
                    .register_chunk(&format!("chunk_{}", i), tensor, AccessPattern::ReadMany)
                    .unwrap();
            }

            // Access chunks in reverse order
            for i in (0..5).rev() {
                let chunk = manager.access_chunk(&format!("chunk_{}", i)).unwrap();
                assert!(chunk.as_slice().iter().all(|&x| x == i as f64));
            }

            let _ = manager.cleanup();
        }
    }

    #[test]
    fn test_compression_with_different_shapes() {
        #[cfg(feature = "lz4-compression")]
        {
            let temp_dir = std::env::temp_dir();
            let mut manager = MemoryManager::new()
                .max_memory_mb(1)
                .compression(CompressionCodec::Lz4)
                .temp_dir(&temp_dir);

            // Test various tensor shapes
            let shapes = [
                vec![1000],
                vec![50, 20],
                vec![10, 10, 10],
                vec![5, 4, 5, 10],
            ];

            for (i, shape) in shapes.iter().enumerate() {
                let size: usize = shape.iter().product();
                let data = vec![i as f64; size];
                let tensor = DenseND::from_vec(data, shape).unwrap();

                manager
                    .register_chunk(&format!("shape_{}", i), tensor, AccessPattern::ReadOnce)
                    .unwrap();
            }

            // Force spill and reload
            let _ = manager.register_chunk(
                "large",
                DenseND::from_vec(vec![999.0; 100000], &[100, 1000]).unwrap(),
                AccessPattern::ReadOnce,
            );

            // Verify shapes are preserved
            for (i, shape) in shapes.iter().enumerate() {
                let loaded = manager.access_chunk(&format!("shape_{}", i)).unwrap();
                assert_eq!(loaded.shape(), shape.as_slice());
            }

            let _ = manager.cleanup();
        }
    }
}
