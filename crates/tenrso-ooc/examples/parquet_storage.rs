//! Parquet storage example
//!
//! Demonstrates how to save and load tensors using Parquet format.

#[cfg(feature = "parquet")]
use anyhow::Result;
#[cfg(feature = "parquet")]
use std::env;
#[cfg(feature = "parquet")]
use tenrso_core::DenseND;
#[cfg(feature = "parquet")]
use tenrso_ooc::{ParquetReader, ParquetWriter};

#[cfg(feature = "parquet")]
fn main() -> Result<()> {
    println!("=== Parquet Storage Example ===\n");

    let temp_dir = env::temp_dir();
    let file_path = temp_dir.join("example_tensor.parquet");

    println!("File path: {:?}\n", file_path);

    // Create a 3D tensor
    println!("Creating tensor:");
    let data: Vec<f64> = (0..120).map(|x| x as f64 / 10.0).collect();
    let tensor = DenseND::<f64>::from_vec(data, &[3, 4, 10])?;
    println!("  Shape: {:?}", tensor.shape());
    println!("  Elements: {}", tensor.as_slice().len());

    // Write to Parquet
    println!("\nWriting to Parquet...");
    let mut writer = ParquetWriter::new(&file_path)?;
    writer.write(&tensor)?;
    writer.finish()?;
    println!("  ✓ Written to disk");

    // Check file size
    let metadata = std::fs::metadata(&file_path)?;
    println!("  File size: {} bytes", metadata.len());

    // Read from Parquet
    println!("\nReading from Parquet...");
    let reader = ParquetReader::open(&file_path)?;
    let loaded = reader.read()?;
    println!("  ✓ Loaded from disk");
    println!("  Shape: {:?}", loaded.shape());

    // Verify data integrity
    println!("\nVerifying data integrity...");
    let original_slice = tensor.as_slice();
    let loaded_slice = loaded.as_slice();

    let mut max_diff = 0.0;
    for (i, (&orig, &load)) in original_slice.iter().zip(loaded_slice.iter()).enumerate() {
        let diff = (orig - load).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-10 {
            println!("  Mismatch at index {}: {} vs {}", i, orig, load);
        }
    }

    println!("  Max difference: {}", max_diff);
    println!("  ✓ Data integrity verified");

    // Cleanup
    std::fs::remove_file(&file_path)?;
    println!("\n✓ Cleanup completed");

    Ok(())
}

#[cfg(not(feature = "parquet"))]
fn main() {
    println!("This example requires the 'parquet' feature to be enabled.");
    println!("Run with: cargo run --example parquet_storage --features parquet");
}
