//! Gradient compression demonstration for distributed training
//!
//! This example shows how to use gradient compression techniques to reduce
//! communication overhead in distributed training scenarios.

use anyhow::Result;
use tenrso_ad::compression::{
    CompressedGradient, CompressionConfig, CompressionMethod, CompressionStats,
};
use tenrso_core::DenseND;

fn main() -> Result<()> {
    println!("=== Gradient Compression Demonstration ===\n");

    // Create a sample gradient (simulating a model with 10,000 parameters)
    let gradient_data: Vec<f64> = (0..10000).map(|i| (i as f64 * 0.1).sin() * 100.0).collect();
    let gradient = DenseND::from_vec(gradient_data, &[10000])?;

    println!("Original gradient statistics:");
    println!("  Shape: {:?}", gradient.shape());
    println!("  Size: {} parameters", gradient.len());
    println!(
        "  Memory: {} bytes\n",
        gradient.len() * std::mem::size_of::<f64>()
    );

    // Example 1: 8-bit Quantization
    example_8bit_quantization(&gradient)?;

    // Example 2: 16-bit Quantization
    example_16bit_quantization(&gradient)?;

    // Example 3: Top-K Sparsification
    example_topk_sparsification(&gradient)?;

    // Example 4: Random Sparsification
    example_random_sparsification(&gradient)?;

    // Example 5: Comparison of Methods
    comparison_of_methods(&gradient)?;

    Ok(())
}

fn example_8bit_quantization(gradient: &DenseND<f64>) -> Result<()> {
    println!("--- Example 1: 8-bit Quantization ---");

    let config = CompressionConfig {
        method: CompressionMethod::Quantize8Bit,
        error_feedback: false,
        seed: None,
    };

    let compressed = CompressedGradient::compress(gradient, &config)?;
    let stats = CompressionStats::compute(gradient, &compressed)?;

    println!("{}", stats);
    println!("Decompressing...");
    let decompressed = compressed.decompress()?;

    // Verify a few values
    println!("Sample values comparison:");
    for i in [0, 100, 500, 1000, 5000] {
        let original = *gradient.get(&[i]).unwrap();
        let recovered = *decompressed.get(&[i]).unwrap();
        println!(
            "  [{:4}] Original: {:8.4}, Recovered: {:8.4}, Error: {:8.4}",
            i,
            original,
            recovered,
            (original - recovered).abs()
        );
    }
    println!();

    Ok(())
}

fn example_16bit_quantization(gradient: &DenseND<f64>) -> Result<()> {
    println!("--- Example 2: 16-bit Quantization ---");

    let config = CompressionConfig {
        method: CompressionMethod::Quantize16Bit,
        error_feedback: false,
        seed: None,
    };

    let compressed = CompressedGradient::compress(gradient, &config)?;
    let stats = CompressionStats::compute(gradient, &compressed)?;

    println!("{}", stats);
    println!("Note: 16-bit has much lower error than 8-bit\n");

    Ok(())
}

fn example_topk_sparsification(gradient: &DenseND<f64>) -> Result<()> {
    println!("--- Example 3: Top-K Sparsification ---");
    println!("Keeping only the top 1% of gradients by magnitude\n");

    let k = gradient.len() / 100; // Top 1%
    let config = CompressionConfig {
        method: CompressionMethod::TopK { k },
        error_feedback: false,
        seed: None,
    };

    let compressed = CompressedGradient::compress(gradient, &config)?;
    let stats = CompressionStats::compute(gradient, &compressed)?;

    println!("{}", stats);

    if let Some(sparsity) = compressed.sparsity() {
        println!(
            "Top-K keeps {} out of {} values ({:.1}% sparsity)\n",
            k,
            gradient.len(),
            sparsity * 100.0
        );
    }

    Ok(())
}

fn example_random_sparsification(gradient: &DenseND<f64>) -> Result<()> {
    println!("--- Example 4: Random Sparsification ---");
    println!("Randomly keeping 10% of gradients (90% sparsity)\n");

    let config = CompressionConfig {
        method: CompressionMethod::RandomSparsify { p: 0.9 },
        error_feedback: false,
        seed: Some(42),
    };

    let compressed = CompressedGradient::compress(gradient, &config)?;
    let stats = CompressionStats::compute(gradient, &compressed)?;

    println!("{}", stats);
    println!("Random sparsification provides stochastic gradient compression\n");

    Ok(())
}

fn comparison_of_methods(gradient: &DenseND<f64>) -> Result<()> {
    println!("--- Example 5: Comparison of Methods ---\n");

    let methods = vec![
        ("No Compression", CompressionMethod::None),
        ("8-bit Quantization", CompressionMethod::Quantize8Bit),
        ("16-bit Quantization", CompressionMethod::Quantize16Bit),
        (
            "Top-10% Sparsification",
            CompressionMethod::TopK {
                k: gradient.len() / 10,
            },
        ),
        (
            "Top-1% Sparsification",
            CompressionMethod::TopK {
                k: gradient.len() / 100,
            },
        ),
        (
            "Random 50% Sparsity",
            CompressionMethod::RandomSparsify { p: 0.5 },
        ),
        (
            "Random 90% Sparsity",
            CompressionMethod::RandomSparsify { p: 0.9 },
        ),
    ];

    println!(
        "{:<28} {:>12} {:>12} {:>12} {:>10}",
        "Method", "Ratio", "Saved (MB)", "MSE", "Sparsity"
    );
    println!("{}", "-".repeat(76));

    for (name, method) in methods {
        let config = CompressionConfig {
            method,
            error_feedback: false,
            seed: Some(42),
        };

        let compressed = CompressedGradient::compress(gradient, &config)?;
        let stats = CompressionStats::compute(gradient, &compressed)?;

        let saved_mb = (stats.original_bytes - stats.compressed_bytes) as f64 / (1024.0 * 1024.0);
        let mse = stats.mse.unwrap_or(0.0);
        let sparsity_str = if let Some(sp) = compressed.sparsity() {
            format!("{:.1}%", sp * 100.0)
        } else {
            "-".to_string()
        };

        println!(
            "{:<28} {:>12.2}x {:>12.3} {:>12.6e} {:>10}",
            name, stats.ratio, saved_mb, mse, sparsity_str
        );
    }
    println!();

    println!("Key insights:");
    println!("  • Higher compression ratio → More communication savings");
    println!("  • Lower MSE → Better gradient quality");
    println!("  • Sparsity methods reduce both memory and bandwidth");
    println!("  • Quantization trades precision for compression");
    println!("  • Top-K preserves the most important gradients");
    println!("  • Random sparsification provides unbiased estimates\n");

    Ok(())
}
