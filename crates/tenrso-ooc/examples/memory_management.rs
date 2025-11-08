//! Memory management example
//!
//! Demonstrates automatic memory management and spill-to-disk policies.

#[cfg(feature = "mmap")]
use anyhow::Result;
#[cfg(feature = "mmap")]
use tenrso_core::DenseND;
#[cfg(feature = "mmap")]
use tenrso_ooc::{AccessPattern, MemoryManager, SpillPolicy};

#[cfg(feature = "mmap")]
fn main() -> Result<()> {
    println!("=== Memory Management Example ===\n");

    // Create memory manager with tight memory limit
    let mut manager = MemoryManager::new()
        .max_memory_mb(10) // Very tight: 10MB
        .pressure_threshold(0.7) // Trigger spill at 70%
        .spill_policy(SpillPolicy::LRU) // Use LRU spill policy
        .auto_spill(true);

    println!("Configuration:");
    println!("  Max memory: 10 MB");
    println!("  Pressure threshold: 70%");
    println!("  Spill policy: LRU");
    println!("  Auto spill: enabled\n");

    // Register several chunks
    println!("Registering chunks...");

    for i in 0..5 {
        let chunk_id = format!("chunk_{}", i);
        let tensor = DenseND::<f64>::zeros(&[200, 200]);

        println!(
            "  Registering {}: 200x200 ({}KB)",
            chunk_id,
            (200 * 200 * std::mem::size_of::<f64>()) / 1024
        );

        manager.register_chunk(
            &chunk_id,
            tensor,
            if i < 2 {
                AccessPattern::ReadMany
            } else {
                AccessPattern::ReadOnce
            },
        )?;

        let stats = manager.stats();
        println!(
            "    Memory: {:.1}% ({} / {} bytes)",
            stats.memory_ratio * 100.0,
            stats.current_memory_bytes,
            stats.max_memory_bytes
        );
    }

    // Access chunks
    println!("\nAccessing chunks...");

    for i in 0..3 {
        let chunk_id = format!("chunk_{}", i);
        println!("  Accessing {}", chunk_id);

        let chunk = manager.access_chunk(&chunk_id)?;
        println!("    Retrieved tensor: {:?}", chunk.shape());
    }

    // Display final statistics
    let stats = manager.stats();
    println!("\nFinal statistics:");
    println!("  Total chunks: {}", stats.total_chunks);
    println!("  In memory: {}", stats.in_memory_chunks);
    println!("  Spilled: {}", stats.spilled_chunks);
    println!(
        "  Memory usage: {:.1}% ({} bytes)",
        stats.memory_ratio * 100.0,
        stats.current_memory_bytes
    );
    println!("  Under pressure: {}", stats.under_pressure);

    println!("\n✓ Memory management completed successfully!");

    Ok(())
}

#[cfg(not(feature = "mmap"))]
fn main() {
    println!("This example requires the 'mmap' feature to be enabled.");
    println!("Run with: cargo run --example memory_management --features mmap");
}
