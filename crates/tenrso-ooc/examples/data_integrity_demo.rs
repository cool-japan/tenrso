//! Data integrity validation demonstration.
//!
//! This example demonstrates:
//! - Multiple checksum algorithms (CRC32, XXHash64, Blake3)
//! - Creating chunk metadata with integrity information
//! - Validating chunks on load
//! - Different validation policies
//! - Corruption detection

use anyhow::Result;
use tenrso_ooc::data_integrity::{
    ChecksumAlgorithm, ChunkIntegrityMetadata, IntegrityChecker, ValidationPolicy,
};

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   TenRSo Data Integrity Validation Demo                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Create some test data
    let chunk_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let bytes = unsafe {
        std::slice::from_raw_parts(
            chunk_data.as_ptr() as *const u8,
            chunk_data.len() * std::mem::size_of::<f64>(),
        )
    };

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. Checksum Algorithm Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let algorithms = [
        ChecksumAlgorithm::Crc32,
        ChecksumAlgorithm::XxHash64,
        ChecksumAlgorithm::Blake3,
    ];

    for algo in &algorithms {
        let start = std::time::Instant::now();
        let metadata = ChunkIntegrityMetadata::new(
            "chunk_001".to_string(),
            bytes.len(),
            vec![2, 4],
            *algo,
            bytes,
        );
        let duration = start.elapsed();

        println!("Algorithm: {:?}", algo);
        println!("  Checksum: {:#x}", metadata.checksum);
        println!("  Time: {:?}", duration);
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. Validation Policies");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create metadata with XXHash64
    let metadata = ChunkIntegrityMetadata::new(
        "chunk_001".to_string(),
        bytes.len(),
        vec![2, 4],
        ChecksumAlgorithm::XxHash64,
        bytes,
    );

    // Test Strict policy
    println!("Policy: Strict (always validate)");
    let mut checker = IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);

    match checker.validate(&metadata, bytes) {
        Ok(()) => println!("  ✓ Validation successful"),
        Err(e) => println!("  ✗ Validation failed: {}", e),
    }
    println!(
        "  Stats: {} validations, {} successes",
        checker.stats().validations,
        checker.stats().successes
    );
    println!();

    // Test Opportunistic policy
    println!("Policy: Opportunistic (validate if metadata present)");
    let mut checker =
        IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Opportunistic);

    match checker.validate(&metadata, bytes) {
        Ok(()) => println!("  ✓ Validation successful"),
        Err(e) => println!("  ✗ Validation failed: {}", e),
    }
    println!();

    // Test None policy
    println!("Policy: None (skip validation)");
    let mut checker = IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::None);

    match checker.validate(&metadata, bytes) {
        Ok(()) => println!("  ✓ Validation skipped (always passes)"),
        Err(e) => println!("  ✗ Unexpected failure: {}", e),
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. Corruption Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut checker = IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);

    // Simulate corrupted data (change one byte)
    let mut corrupted = bytes.to_vec();
    corrupted[0] ^= 0xFF; // Flip all bits in first byte

    println!("Testing with corrupted data (first byte flipped):");
    match checker.validate(&metadata, &corrupted) {
        Ok(()) => println!("  ✗ Corruption not detected (UNEXPECTED!)"),
        Err(e) => println!("  ✓ Corruption detected: {}", e),
    }
    println!();

    // Test size mismatch
    let truncated = &bytes[..bytes.len() - 8];
    println!("Testing with truncated data (8 bytes removed):");
    match checker.validate(&metadata, truncated) {
        Ok(()) => println!("  ✗ Size mismatch not detected (UNEXPECTED!)"),
        Err(e) => println!("  ✓ Size mismatch detected: {}", e),
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. Batch Validation Workflow");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut checker = IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);

    // Create and validate multiple chunks
    let chunk_count = 100;
    println!("Creating and validating {} chunks...", chunk_count);

    let start = std::time::Instant::now();
    for i in 0..chunk_count {
        let chunk_id = format!("chunk_{:03}", i);
        let data = vec![i as u8; 1024]; // 1KB chunks

        // Create metadata
        let metadata = checker.create_metadata(chunk_id, vec![32, 32], &data);

        // Validate
        checker.validate(&metadata, &data)?;
    }
    let duration = start.elapsed();

    println!("\nBatch validation complete!");
    println!("  Total chunks: {}", chunk_count);
    println!("  Time: {:?}", duration);
    println!("  Average: {:?} per chunk", duration / chunk_count);
    println!("  Success rate: {:.2}%", checker.success_rate() * 100.0);
    println!("\nStatistics:");
    println!("  Validations: {}", checker.stats().validations);
    println!("  Successes: {}", checker.stats().successes);
    println!("  Failures: {}", checker.stats().failures);
    println!(
        "  Bytes validated: {} ({:.2} MB)",
        checker.stats().bytes_validated,
        checker.stats().bytes_validated as f64 / 1_048_576.0
    );
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. Performance Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let test_data = vec![0u8; 1_000_000]; // 1MB test data

    println!("Testing with 1MB of data:\n");

    for algo in &[
        ChecksumAlgorithm::None,
        ChecksumAlgorithm::Crc32,
        ChecksumAlgorithm::XxHash64,
        ChecksumAlgorithm::Blake3,
    ] {
        let iterations = if matches!(algo, ChecksumAlgorithm::None) {
            10000
        } else {
            100
        };

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = ChunkIntegrityMetadata::new(
                "perf_test".to_string(),
                test_data.len(),
                vec![1000, 1000],
                *algo,
                &test_data,
            );
        }
        let duration = start.elapsed();
        let per_iter = duration / iterations;
        let throughput = (test_data.len() as f64 / 1_048_576.0) / per_iter.as_secs_f64();

        println!("{:?}:", algo);
        println!("  Time per iteration: {:?}", per_iter);
        println!("  Throughput: {:.2} MB/s", throughput);
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Takeaways");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("• XXHash64 provides excellent speed/security balance");
    println!("• CRC32 is fastest for basic error detection");
    println!("• Blake3 provides cryptographic security at reasonable speed");
    println!("• Strict policy recommended for production data");
    println!("• Corruption and size mismatches are reliably detected");
    println!("• Overhead is minimal (~1-10 MB/s depending on algorithm)");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}
