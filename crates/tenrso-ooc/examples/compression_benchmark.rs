//! Compression benchmark: Compare spill performance with different compression codecs
//!
//! This example demonstrates:
//! - Memory management with compression enabled
//! - Performance comparison: None vs LZ4 vs Zstd
//! - Compression ratio analysis
//! - Throughput measurement for spill/load operations
//!
//! Run with: cargo run --example compression_benchmark --all-features

use std::time::Instant;
use tenrso_core::DenseND;
use tenrso_ooc::{AccessPattern, CompressionCodec, MemoryManager, SpillPolicy};

fn benchmark_codec(codec: CompressionCodec, name: &str) {
    println!("\n=== Benchmarking {} ===", name);

    let temp_dir = std::env::temp_dir().join(format!("tenrso_compression_bench_{}", name));
    std::fs::create_dir_all(&temp_dir).unwrap();

    let mut manager = MemoryManager::new()
        .max_memory_mb(10) // 10MB limit to allow some chunks in memory
        .spill_policy(SpillPolicy::LRU)
        .compression(codec)
        .auto_spill(true)
        .temp_dir(&temp_dir);

    // Create various types of data patterns
    let patterns = [
        ("Constant", vec![42.0f64; 100000]),
        ("Linear", (0..100000).map(|i| i as f64).collect::<Vec<_>>()),
        (
            "Random",
            (0..100000).map(|i| (i * 7919) as f64 % 1000.0).collect(),
        ),
        (
            "Sparse",
            (0..100000)
                .map(|i| if i % 10 == 0 { i as f64 } else { 0.0 })
                .collect(),
        ),
    ];

    let mut total_spill_time = 0u128;
    let mut total_load_time = 0u128;
    let mut total_original_size = 0usize;
    let mut total_disk_size = 0usize;

    for (i, (pattern_name, data)) in patterns.iter().enumerate() {
        let tensor = DenseND::from_vec(data.clone(), &[100, 1000]).unwrap();
        let original_size = data.len() * std::mem::size_of::<f64>();

        // Register chunk
        manager
            .register_chunk(&format!("chunk_{}", i), tensor, AccessPattern::ReadMany)
            .unwrap();

        // Measure spill time by forcing memory pressure
        let start = Instant::now();
        // Allocate multiple large chunks to force spilling of the pattern chunk
        for j in 0..3 {
            let large = DenseND::from_vec(vec![j as f64; 50000], &[50, 1000]).unwrap();
            let _ = manager.register_chunk(&format!("temp_{}", j), large, AccessPattern::ReadOnce);
        }
        let spill_time = start.elapsed();

        // Measure load time
        let start = Instant::now();
        let loaded = manager.access_chunk(&format!("chunk_{}", i)).unwrap();
        let load_time = start.elapsed();

        // Verify correctness
        assert_eq!(loaded.shape(), &[100, 1000]);
        assert_eq!(loaded.as_slice(), data.as_slice());

        // Measure disk size (check if file exists after forcing more spills)
        let large_force = DenseND::from_vec(vec![0.0; 150000], &[150, 1000]).unwrap();
        let _ = manager.register_chunk("large_force", large_force, AccessPattern::ReadOnce);

        let disk_size = if let Ok(metadata) =
            std::fs::metadata(temp_dir.join(format!("tenrso_chunk_chunk_{}.bin", i)))
        {
            metadata.len() as usize
        } else {
            // File might not exist if not spilled yet
            original_size // Assume no compression if not spilled
        };

        total_spill_time += spill_time.as_micros();
        total_load_time += load_time.as_micros();
        total_original_size += original_size;
        total_disk_size += disk_size;

        println!(
            "  {:<10} | Spill: {:>7.2} ms | Load: {:>7.2} ms | Size: {:>8} -> {:>8} bytes ({:>5.2}x)",
            pattern_name,
            spill_time.as_secs_f64() * 1000.0,
            load_time.as_secs_f64() * 1000.0,
            original_size,
            disk_size,
            if disk_size > 0 {
                original_size as f64 / disk_size as f64
            } else {
                0.0
            }
        );

        // Clean up temporary chunks
        for j in 0..3 {
            let _ = manager.decref(&format!("temp_{}", j));
        }
        let _ = manager.decref("large_force");
        let _ = manager.decref(&format!("chunk_{}", i));
    }

    println!("\n  Summary:");
    println!(
        "    Total spill time:  {:>7.2} ms",
        total_spill_time as f64 / 1000.0
    );
    println!(
        "    Total load time:   {:>7.2} ms",
        total_load_time as f64 / 1000.0
    );
    println!(
        "    Avg compression:   {:>7.2}x",
        if total_disk_size > 0 {
            total_original_size as f64 / total_disk_size as f64
        } else {
            0.0
        }
    );
    println!(
        "    Spill throughput:  {:>7.2} MB/s",
        (total_original_size as f64 / (1024.0 * 1024.0)) / (total_spill_time as f64 / 1_000_000.0)
    );
    println!(
        "    Load throughput:   {:>7.2} MB/s",
        (total_original_size as f64 / (1024.0 * 1024.0)) / (total_load_time as f64 / 1_000_000.0)
    );

    // Cleanup
    let _ = manager.cleanup();
    let _ = std::fs::remove_dir_all(&temp_dir);
}

fn main() {
    println!("=============================================================");
    println!("    TenRSo Out-of-Core: Compression Benchmark");
    println!("=============================================================");
    println!("\nTesting spill/load performance with different compression codecs");
    println!("Data patterns: Constant, Linear, Random, Sparse");
    println!("Tensor size: 100 x 1000 (800KB per tensor)");

    // Benchmark no compression
    benchmark_codec(CompressionCodec::None, "No Compression");

    // Benchmark LZ4
    #[cfg(feature = "lz4-compression")]
    benchmark_codec(CompressionCodec::Lz4, "LZ4 (Fast)");

    // Benchmark Zstd at different levels
    #[cfg(feature = "zstd-compression")]
    {
        benchmark_codec(CompressionCodec::Zstd { level: 1 }, "Zstd (Level 1)");
        benchmark_codec(CompressionCodec::Zstd { level: 3 }, "Zstd (Level 3)");
        benchmark_codec(CompressionCodec::Zstd { level: 9 }, "Zstd (Level 9)");
    }

    println!("\n=============================================================");
    println!("Benchmark complete!");
    println!("=============================================================");
}
