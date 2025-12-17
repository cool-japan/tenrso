//! # Machine Learning-based Eviction Policy Demo
//!
//! This example demonstrates the ML-based eviction policy that learns from
//! access patterns to make smarter eviction decisions compared to traditional
//! LRU/LFU policies.
//!
//! ## Features Demonstrated
//!
//! - Online learning from access patterns
//! - Feature extraction (time since access, frequency, regularity, etc.)
//! - Linear regression for predicting next access time
//! - Logistic regression for eviction classification
//! - Ensemble model combining multiple predictors
//! - Comparison with simple heuristics

use tenrso_ooc::ml_eviction::{MLConfig, MLEvictionPolicy};

fn main() {
    println!("=== Machine Learning-based Eviction Policy Demo ===\n");

    // Create ML eviction policy with custom configuration
    let config = MLConfig {
        learning_rate: 0.01,
        l2_lambda: 0.001,
        momentum: 0.9,
        num_features: 6,
        min_samples: 10,
        ema_decay: 0.95,
        normalize_features: true,
        use_ensemble: true,
        classification_threshold: 0.5,
    };

    let mut policy = MLEvictionPolicy::new(config);

    println!("Configuration:");
    println!(
        "  Learning rate: {}",
        policy.get_stats().prediction_accuracy
    );
    println!("  Ensemble mode: enabled");
    println!("  Min samples for prediction: 10\n");

    // Simulate workload with different access patterns
    println!("=== Simulating Access Patterns ===\n");

    // Pattern 1: Frequently accessed chunks (hot data)
    println!("1. Hot chunks (frequently accessed):");
    for t in 0..50 {
        let timestamp = t as f64 * 10.0;
        policy.record_access(0, timestamp); // Chunk 0: very frequent
        if t % 2 == 0 {
            policy.record_access(1, timestamp); // Chunk 1: frequent
        }
        if t % 5 == 0 {
            policy.record_access(2, timestamp); // Chunk 2: moderate
        }
    }

    // Pattern 2: Cold chunks (rarely accessed)
    println!("2. Cold chunks (rarely accessed):");
    policy.record_access(3, 10.0); // Chunk 3: accessed once at start
    policy.record_access(4, 20.0); // Chunk 4: accessed once

    // Pattern 3: Sequential access pattern
    println!("3. Sequential chunks:");
    for t in 0..20 {
        let timestamp = 500.0 + t as f64 * 15.0;
        policy.record_access(5, timestamp); // Chunk 5: regular sequential
    }

    // Set metadata for chunks
    policy.set_chunk_metadata(0, 1024 * 1024, 0); // 1MB in RAM
    policy.set_chunk_metadata(1, 512 * 1024, 0); // 512KB in RAM
    policy.set_chunk_metadata(2, 2048 * 1024, 1); // 2MB in SSD
    policy.set_chunk_metadata(3, 4096 * 1024, 2); // 4MB on Disk
    policy.set_chunk_metadata(4, 8192 * 1024, 2); // 8MB on Disk
    policy.set_chunk_metadata(5, 256 * 1024, 0); // 256KB in RAM

    println!();

    // Get eviction candidates at current time
    let current_time = 1000.0;
    println!("=== Eviction Recommendations at t={} ===\n", current_time);

    let chunk_ids = vec![0, 1, 2, 3, 4, 5];
    let candidates = policy.get_eviction_candidates(&chunk_ids, current_time);

    println!("Chunks ranked by eviction score (highest = most suitable for eviction):\n");
    println!(
        "{:>8} {:>20} {:>20} {:>15}",
        "Chunk ID", "Next Access Time", "Evict Probability", "Combined Score"
    );
    println!("{:-<80}", "");

    for candidate in &candidates {
        println!(
            "{:>8} {:>20.2} {:>20.3} {:>15.3}",
            candidate.chunk_id,
            candidate.predicted_next_access_time,
            candidate.eviction_probability,
            candidate.combined_score
        );
    }

    println!();

    // Show which chunks would be evicted
    let num_to_evict = 3;
    println!("=== Eviction Decision ===\n");
    println!(
        "Need to evict {} chunks. Recommended eviction order:",
        num_to_evict
    );

    for (i, candidate) in candidates.iter().take(num_to_evict).enumerate() {
        println!(
            "  {}. Chunk {} (score: {:.3})",
            i + 1,
            candidate.chunk_id,
            candidate.combined_score
        );
    }

    println!();

    // Show statistics
    let stats = policy.get_stats();
    println!("=== ML Policy Statistics ===\n");
    println!("  Training samples: {}", stats.training_samples);
    println!("  Chunks tracked: {}", stats.num_chunks_tracked);
    println!("  Regression updates: {}", stats.regression_updates);
    println!("  Classification updates: {}", stats.classification_updates);
    println!(
        "  Prediction accuracy: {:.2}%",
        stats.prediction_accuracy * 100.0
    );

    println!();

    // Demonstrate online learning
    println!("=== Online Learning Demo ===\n");
    println!("Recording more accesses to demonstrate online learning...");

    // Chunk 3 suddenly becomes hot
    for t in 0..20 {
        let timestamp = 1000.0 + t as f64 * 5.0;
        policy.record_access(3, timestamp);
    }

    println!("Chunk 3 has become frequently accessed. Getting new recommendations...\n");

    let new_candidates = policy.get_eviction_candidates(&chunk_ids, 1200.0);

    println!("Updated rankings:");
    println!(
        "{:>8} {:>20} {:>20} {:>15}",
        "Chunk ID", "Next Access Time", "Evict Probability", "Combined Score"
    );
    println!("{:-<80}", "");

    for candidate in &new_candidates {
        println!(
            "{:>8} {:>20.2} {:>20.3} {:>15.3}",
            candidate.chunk_id,
            candidate.predicted_next_access_time,
            candidate.eviction_probability,
            candidate.combined_score
        );
    }

    println!();

    // Find chunk 3's new ranking
    let chunk_3_rank = new_candidates.iter().position(|c| c.chunk_id == 3).unwrap() + 1;

    println!(
        "Note: Chunk 3's eviction rank improved from {} to {} after becoming hot!",
        candidates.iter().position(|c| c.chunk_id == 3).unwrap() + 1,
        chunk_3_rank
    );

    println!();

    // Comparison with simple heuristics
    println!("=== Comparison with Simple LRU ===\n");
    println!(
        "ML Policy would evict chunks: {:?}",
        candidates
            .iter()
            .take(3)
            .map(|c| c.chunk_id)
            .collect::<Vec<_>>()
    );

    // Simple LRU would just evict based on last access time
    println!("Simple LRU would evict oldest chunks (4, 3, then 2)");
    println!("ML Policy is smarter because it considers:");
    println!("  - Access frequency (how often accessed)");
    println!("  - Access regularity (predictable patterns)");
    println!("  - Sequential access patterns");
    println!("  - Chunk size and memory tier");
    println!("  - Predicted future access probability");

    println!();
    println!("=== Demo Complete ===");
}
