//! Advanced memory management example: Tiered memory + Working set prediction
//!
//! This example demonstrates the integration of:
//! - Hierarchical memory tiers (RAM → SSD → Disk)
//! - Working set prediction with adaptive algorithms
//! - Automatic data migration based on access patterns
//!
//! Run with: cargo run --example advanced_memory_management --all-features

use std::time::Instant;
use tenrso_core::DenseND;
use tenrso_ooc::{
    MemoryTier, PredictionMode, TierAccessPattern, TierConfig, TieredMemoryManager,
    WorkingSetPredictor,
};

fn main() -> anyhow::Result<()> {
    println!("=== Advanced Memory Management Demo ===\n");

    // Scenario: Processing a sequence of tensor chunks with varying access patterns
    // We'll simulate a workload with:
    // - Hot chunks (frequently accessed) → should stay in RAM
    // - Warm chunks (moderately accessed) → migrate to SSD
    // - Cold chunks (rarely accessed) → demote to Disk

    // 1. Set up tiered memory manager
    println!("1. Configuring tiered memory manager...");
    let mut tier_manager = TieredMemoryManager::new()
        .tier_config(TierConfig {
            ram_mb: 10,   // 10 MB RAM (limited for demo)
            ssd_mb: 50,   // 50 MB SSD
            disk_mb: 200, // 200 MB Disk
        })
        .promotion_threshold(0.6) // Promote when tier is 60% full
        .demotion_threshold(0.8) // Demote when tier is 80% full
        .auto_migration(true);

    println!("   RAM: 10 MB, SSD: 50 MB, Disk: 200 MB");
    println!("   Auto-migration enabled\n");

    // 2. Set up working set predictor
    println!("2. Configuring working set predictor...");
    let mut ws_predictor = WorkingSetPredictor::new()
        .prediction_mode(PredictionMode::Adaptive)
        .window_size(50);

    println!("   Prediction mode: Adaptive");
    println!("   Window size: 50 accesses\n");

    // 3. Create chunks with different access patterns
    println!("3. Creating tensor chunks...");
    let chunk_size = 1000; // ~8 KB per chunk (1000 f64 values)
    let num_chunks = 20;

    for i in 0..num_chunks {
        let data = vec![i as f64; chunk_size];
        let tensor = DenseND::from_vec(data, &[10, 100])?;
        let chunk_id = format!("chunk_{}", i);

        // Assign access patterns based on chunk index
        let pattern = match i {
            0..=4 => TierAccessPattern::Temporal,   // Hot chunks
            5..=9 => TierAccessPattern::Sequential, // Warm chunks (streaming)
            _ => TierAccessPattern::Random,         // Cold chunks
        };

        tier_manager.register_chunk(&chunk_id, tensor, pattern)?;
    }

    println!("   Created {} chunks (~8 KB each)", num_chunks);
    print_tier_stats(&tier_manager);

    // 4. Simulate workload with different access patterns
    println!("\n4. Simulating workload...");
    let start = Instant::now();

    // Phase 1: Hot chunks accessed frequently (should stay in RAM)
    println!("\n   Phase 1: Accessing hot chunks frequently...");
    for round in 0..10 {
        for i in 0..5 {
            let chunk_id = format!("chunk_{}", i);
            let _ = tier_manager.get_chunk(&chunk_id)?;
            ws_predictor.record_access(&chunk_id, chunk_size * 8);
        }
        if round % 3 == 0 {
            print_tier_stats(&tier_manager);
        }
    }

    // Phase 2: Sequential access pattern (streaming)
    println!("\n   Phase 2: Sequential streaming access...");
    for i in 5..10 {
        let chunk_id = format!("chunk_{}", i);
        let _ = tier_manager.get_chunk(&chunk_id)?;
        ws_predictor.record_access(&chunk_id, chunk_size * 8);
    }
    print_tier_stats(&tier_manager);

    // Phase 3: Random access to cold chunks
    println!("\n   Phase 3: Random access to cold chunks...");
    for i in [15, 12, 18, 11, 19].iter() {
        let chunk_id = format!("chunk_{}", i);
        let _ = tier_manager.get_chunk(&chunk_id)?;
        ws_predictor.record_access(&chunk_id, chunk_size * 8);
    }
    print_tier_stats(&tier_manager);

    let elapsed = start.elapsed();
    println!("\n   Total workload time: {:?}", elapsed);

    // 5. Working set prediction
    println!("\n5. Working set prediction results:");
    let predictions = ws_predictor.predict_working_set(10)?;

    println!("\n   Top 10 predicted chunks:");
    println!(
        "   {:<15} {:<10} {:<12} {:<10}",
        "Chunk ID", "Score", "Size (KB)", "Confidence"
    );
    println!("   {}", "-".repeat(55));

    for pred in predictions.iter() {
        println!(
            "   {:<15} {:<10.4} {:<12.2} {:<10.2}",
            pred.chunk_id,
            pred.score,
            pred.size_bytes as f64 / 1024.0,
            pred.confidence
        );
    }

    // 6. Analyze working set predictor statistics
    println!("\n6. Working set predictor statistics:");
    let ws_stats = ws_predictor.overall_stats();
    println!("   Total chunks tracked: {}", ws_stats.total_chunks);
    println!("   Total accesses: {}", ws_stats.total_accesses);
    println!(
        "   Regular patterns detected: {}",
        ws_stats.regular_patterns
    );
    println!("   Sequential chunks: {}", ws_stats.sequential_chunks);

    // 7. Verify hot chunks are in RAM
    println!("\n7. Verifying tier placement:");
    for i in 0..5 {
        let chunk_id = format!("chunk_{}", i);
        if let Some(stats) = tier_manager.chunk_stats(&chunk_id) {
            println!("   {} - Tier: {:?}", chunk_id, stats.tier);
        }
    }

    // 8. Demonstrate proactive prefetching based on prediction
    println!("\n8. Proactive prefetching simulation:");
    println!("   Based on predictions, we would prefetch:");

    for (idx, pred) in predictions.iter().take(5).enumerate() {
        if pred.confidence > 0.5 {
            println!(
                "   {}. {} (confidence: {:.2})",
                idx + 1,
                pred.chunk_id,
                pred.confidence
            );
        }
    }

    // 9. Final statistics
    println!("\n9. Final system statistics:");
    let overall_stats = tier_manager.overall_stats();
    println!("   Total chunks: {}", overall_stats.total_chunks);
    println!("   RAM chunks: {}", overall_stats.ram_chunks);
    println!("   SSD chunks: {}", overall_stats.ssd_chunks);
    println!("   Disk chunks: {}", overall_stats.disk_chunks);
    println!("   Working set size: {}", overall_stats.working_set_size);

    if let Some(ram_stats) = tier_manager.tier_stats(MemoryTier::Ram) {
        println!("\n   RAM statistics:");
        println!("     Usage: {:.2}%", ram_stats.usage_ratio * 100.0);
        println!("     Hit rate: {:.2}%", ram_stats.hit_rate * 100.0);
        println!("     Chunks: {}", ram_stats.chunk_count);
    }

    if let Some(ssd_stats) = tier_manager.tier_stats(MemoryTier::Ssd) {
        println!("\n   SSD statistics:");
        println!("     Usage: {:.2}%", ssd_stats.usage_ratio * 100.0);
        println!("     Chunks: {}", ssd_stats.chunk_count);
    }

    // 10. Performance insights
    println!("\n10. Performance insights:");
    println!("   • Hot chunks (0-4) should be in RAM with high hit rates");
    println!("   • Sequential chunks (5-9) may be in SSD with good prefetch");
    println!("   • Cold chunks (10+) are likely in Disk with low access frequency");
    println!("\n   Adaptive tier management automatically optimizes placement!");

    println!("\n=== Demo Complete ===");

    Ok(())
}

fn print_tier_stats(manager: &TieredMemoryManager) {
    let overall = manager.overall_stats();
    println!(
        "   Tiers - RAM: {}, SSD: {}, Disk: {}",
        overall.ram_chunks, overall.ssd_chunks, overall.disk_chunks
    );
}
