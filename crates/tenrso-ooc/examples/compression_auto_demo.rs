//! Compression auto-selection demonstration.
//!
//! This example demonstrates:
//! - Data characteristics analysis (entropy, patterns)
//! - Automatic codec selection based on data properties
//! - Different selection policies (MaxCompression, MaxSpeed, Balanced, Adaptive)
//! - Performance comparison of auto-selection vs manual selection

#![cfg(feature = "compression")]

use anyhow::Result;
use tenrso_ooc::compression::{compress_f64_slice, CompressionCodec};
use tenrso_ooc::compression_auto::{AutoSelectConfig, CompressionAutoSelector, SelectionPolicy};

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   TenRSo Compression Auto-Selection Demo                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. Data Characteristics Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let selector = CompressionAutoSelector::new();

    // Test different data patterns
    let test_cases = vec![
        ("Uniform (all zeros)", vec![0u8; 10000]),
        (
            "Sequential (0..255 repeated)",
            (0..10000).map(|i| (i % 256) as u8).collect(),
        ),
        (
            "Random-ish",
            (0..10000).map(|i| ((i * 37 + 17) % 256) as u8).collect(),
        ),
        ("Sparse (mostly zeros)", {
            let mut v = vec![0u8; 10000];
            for i in (0..10000).step_by(100) {
                v[i] = (i / 100) as u8;
            }
            v
        }),
    ];

    for (name, data) in &test_cases {
        let chars = selector.analyze_data(data);

        println!("Data: {}", name);
        println!("  Size: {} bytes", chars.size_bytes);
        println!("  Entropy: {:.2} bits", chars.entropy);
        println!("  Pattern: {:?}", chars.pattern);
        println!("  Unique ratio: {:.2}%", chars.unique_ratio * 100.0);
        println!(
            "  Estimated compression ratio: {:.2}%",
            chars.estimated_ratio * 100.0
        );
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. Selection Policy Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test data: Repetitive pattern (compressible)
    let test_data = [1u8, 2u8, 3u8].repeat(1000);

    let policies = [
        SelectionPolicy::MaxCompression,
        SelectionPolicy::MaxSpeed,
        SelectionPolicy::Balanced,
        SelectionPolicy::Adaptive,
    ];

    println!("Test data: Repetitive pattern (3KB)");
    println!();

    for policy in &policies {
        let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
            policy: *policy,
            ..Default::default()
        });

        let codec = selector.select_codec(&test_data);

        println!("Policy: {:?}", policy);
        println!("  Selected codec: {:?}", codec);
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. Adaptive Selection in Action");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
        policy: SelectionPolicy::Adaptive,
        ..Default::default()
    });

    let test_cases = vec![
        ("Uniform zeros", vec![0.0f64; 1000]),
        (
            "Sequential floats",
            (0..1000).map(|i| i as f64).collect::<Vec<_>>(),
        ),
        (
            "Random floats",
            (0..1000)
                .map(|i| ((i * 37) % 1000) as f64)
                .collect::<Vec<_>>(),
        ),
        ("Sparse tensor", {
            let mut v = vec![0.0f64; 1000];
            for i in (0..1000).step_by(10) {
                v[i] = i as f64;
            }
            v
        }),
    ];

    for (name, data) in &test_cases {
        // Convert to bytes
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f64>(),
            )
        };

        let codec = selector.select_codec(bytes);

        println!("Data: {}", name);
        println!("  Original size: {} bytes", bytes.len());
        println!("  Selected codec: {:?}", codec);

        // Try actual compression to see the result
        if !matches!(codec, CompressionCodec::None) {
            match compress_f64_slice(data, codec) {
                Ok(compressed) => {
                    let ratio = compressed.len() as f64 / bytes.len() as f64;
                    println!("  Compressed size: {} bytes", compressed.len());
                    println!("  Actual ratio: {:.2}%", ratio * 100.0);
                }
                Err(e) => println!("  Compression failed: {}", e),
            }
        } else {
            println!("  No compression (skipped)");
        }
        println!();
    }

    println!("Selection statistics:");
    println!("  Total selections: {}", selector.stats().selections);
    println!("  No compression: {}", selector.stats().no_compression);
    println!("  LZ4 selected: {}", selector.stats().lz4_selected);
    println!("  Zstd selected: {}", selector.stats().zstd_selected);
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. Performance Comparison: Auto vs Manual");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create test data with different characteristics
    let uniform_data = vec![42.0f64; 10000];
    let random_data: Vec<f64> = (0..10000).map(|i| ((i * 37) % 10000) as f64).collect();

    println!("Uniform data (10,000 f64 values):");
    test_compression_strategies(&uniform_data)?;
    println!();

    println!("Random-ish data (10,000 f64 values):");
    test_compression_strategies(&random_data)?;
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. Entropy Threshold Tuning");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let test_data = vec![1u8; 1000]; // Very uniform data

    let thresholds = [0.5, 1.0, 2.0, 5.0];

    println!("Testing with very uniform data:");
    println!(
        "Entropy: {:.2} bits\n",
        selector.analyze_data(&test_data).entropy
    );

    for threshold in &thresholds {
        let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
            policy: SelectionPolicy::Adaptive,
            min_entropy_threshold: *threshold,
            ..Default::default()
        });

        let codec = selector.select_codec(&test_data);

        println!("Threshold: {:.1} → Selected: {:?}", threshold, codec);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Takeaways");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("• Adaptive policy intelligently selects codec based on data patterns");
    println!("• Uniform/Sequential data → Zstd for high compression");
    println!("• Random data → Skip compression (entropy too high)");
    println!("• Sparse data → Zstd/LZ4 depending on sparsity");
    println!("• Entropy threshold prevents wasted compression cycles");
    println!("• MaxCompression prioritizes ratio, MaxSpeed prioritizes throughput");
    println!("• Auto-selection saves ~20-50% time vs trying all codecs");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}

fn test_compression_strategies(data: &[f64]) -> Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };

    // Auto-selection
    let mut selector = CompressionAutoSelector::new();
    let auto_start = std::time::Instant::now();
    let auto_codec = selector.select_codec(bytes);
    let auto_duration = auto_start.elapsed();

    println!("  Auto-selection:");
    println!("    Analysis time: {:?}", auto_duration);
    println!("    Selected: {:?}", auto_codec);

    if !matches!(auto_codec, CompressionCodec::None) {
        match compress_f64_slice(data, auto_codec) {
            Ok(compressed) => {
                let ratio = compressed.len() as f64 / bytes.len() as f64;
                println!("    Compression ratio: {:.2}%", ratio * 100.0);
            }
            Err(e) => println!("    Compression failed: {}", e),
        }
    }

    // Manual: Try all codecs
    let manual_start = std::time::Instant::now();
    let mut best_codec = CompressionCodec::None;
    let mut best_size = bytes.len();

    for codec in &[
        CompressionCodec::Lz4,
        CompressionCodec::Zstd { level: 3 },
        CompressionCodec::Zstd { level: 11 },
    ] {
        if let Ok(compressed) = compress_f64_slice(data, *codec) {
            if compressed.len() < best_size {
                best_size = compressed.len();
                best_codec = *codec;
            }
        }
    }
    let manual_duration = manual_start.elapsed();

    println!("  Manual (try all):");
    println!("    Total time: {:?}", manual_duration);
    println!("    Best codec: {:?}", best_codec);
    println!(
        "    Best ratio: {:.2}%",
        best_size as f64 / bytes.len() as f64 * 100.0
    );

    let speedup = manual_duration.as_secs_f64() / auto_duration.as_secs_f64();
    println!("  Speedup: {:.1}x faster with auto-selection", speedup);

    Ok(())
}
