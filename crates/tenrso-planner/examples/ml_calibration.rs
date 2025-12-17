//! Example: Machine Learning-Based Cost Calibration
//!
//! This example demonstrates how to use ML-based cost model refinement to improve
//! planning accuracy based on execution history.
//!
//! Run with: cargo run --example ml_calibration

use std::time::SystemTime;
use tenrso_planner::{
    greedy_planner, EinsumSpec, ExecutionHistory, ExecutionRecord, MLCostModel, PlanHints,
};

fn main() -> anyhow::Result<()> {
    println!("=== ML-Based Cost Calibration Example ===\n");

    // Scenario: We have a planner that consistently overestimates FLOPs by 20%
    // The ML model will learn this bias and correct it

    println!("Step 1: Create execution history with biased predictions\n");

    let mut history = ExecutionHistory::with_max_size(100);

    // Simulate 10 executions where predictions are 20% too high
    for i in 1..=10 {
        let actual_flops = (i * 1_000_000) as f64;
        let predicted_flops = actual_flops * 1.2; // 20% overestimate

        let record = ExecutionRecord {
            id: format!("matmul_{}", i),
            predicted_flops,
            actual_flops,
            predicted_time_ms: actual_flops / 100_000.0,
            actual_time_ms: actual_flops / 100_000.0,
            predicted_memory: 1_000_000,
            actual_memory: 1_000_000,
            timestamp: SystemTime::now(),
            planner: "greedy".to_string(),
        };

        history.record(record);
    }

    println!(
        "Recorded {} execution records with systematic 20% overestimation",
        history.len()
    );
    println!();

    // Analyze the quality metrics before calibration
    let metrics_before = history.compute_metrics();
    println!("Quality Metrics (Before Calibration):");
    println!(
        "  Average FLOPs error: {:.1}%",
        metrics_before.avg_flops_error * 100.0
    );
    println!(
        "  Maximum FLOPs error: {:.1}%",
        metrics_before.max_flops_error * 100.0
    );
    println!(
        "  Accuracy (within 10%): {:.1}%",
        metrics_before.accuracy_10pct * 100.0
    );
    println!();

    // Step 2: Train ML cost model
    println!("Step 2: Train ML cost model from execution history\n");

    let mut ml_model = MLCostModel::new();
    ml_model.train(&history);

    println!("Model trained successfully!");
    println!("  FLOPs model R²: {:.4}", ml_model.flops_r_squared());
    println!("  Time model R²: {:.4}", ml_model.time_r_squared());
    println!("  Memory model R²: {:.4}", ml_model.memory_r_squared());
    println!();

    // Step 3: Use calibrated predictions
    println!("Step 3: Compare predictions before and after calibration\n");

    // Create a test plan
    let spec = EinsumSpec::parse("ij,jk,kl->il")?;
    let shapes = vec![vec![100, 200], vec![200, 300], vec![300, 400]];
    let hints = PlanHints::default();

    let plan = greedy_planner(&spec, &shapes, &hints)?;

    let original_flops = plan.estimated_flops;
    let calibrated_flops = ml_model.calibrate_flops(original_flops);

    println!("Test Case: Matrix chain multiplication (100×200 × 200×300 × 300×400)");
    println!();
    println!("Original prediction:    {:.2e} FLOPs", original_flops);
    println!("Calibrated prediction:  {:.2e} FLOPs", calibrated_flops);
    println!(
        "Correction factor:      {:.3}x",
        calibrated_flops / original_flops
    );
    println!(
        "Expected correction:    {:.3}x (compensating for 20% overestimation)",
        1.0 / 1.2
    );
    println!();

    // Step 4: Show planner-specific calibration
    println!("Step 4: Planner-specific calibration\n");

    // Add records for a different planner with different bias (10% underestimate)
    for i in 1..=5 {
        let actual = (i * 1_000_000) as f64;
        history.record(ExecutionRecord {
            id: format!("dp_{}", i),
            predicted_flops: actual * 0.9, // 10% underestimate
            actual_flops: actual,
            predicted_time_ms: actual / 100_000.0,
            actual_time_ms: actual / 100_000.0,
            predicted_memory: 1_000_000,
            actual_memory: 1_000_000,
            timestamp: SystemTime::now(),
            planner: "dp".to_string(),
        });
    }

    // Retrain with both planners
    ml_model.train(&history);

    println!("Trained with 2 planners (greedy: 20% over, dp: 10% under)");
    println!(
        "Number of planner-specific models: {}",
        ml_model.num_planner_models()
    );
    println!();

    let test_flops = 5_000_000.0;
    let cal_greedy = ml_model.calibrate_flops_for_planner(test_flops, "greedy");
    let cal_dp = ml_model.calibrate_flops_for_planner(test_flops, "dp");
    let cal_unknown = ml_model.calibrate_flops_for_planner(test_flops, "unknown");

    println!("Calibration for {:.2e} FLOPs:", test_flops);
    println!(
        "  Greedy planner:  {:.2e} ({:.1}% correction)",
        cal_greedy,
        (cal_greedy / test_flops - 1.0) * 100.0
    );
    println!(
        "  DP planner:      {:.2e} ({:.1}% correction)",
        cal_dp,
        (cal_dp / test_flops - 1.0) * 100.0
    );
    println!(
        "  Unknown planner: {:.2e} (uses general model)",
        cal_unknown
    );
    println!();

    // Step 5: Show improvement over time
    println!("Step 5: Prediction accuracy improvement\n");

    // Create a small baseline history (no calibration)
    let mut baseline_history = ExecutionHistory::with_max_size(100);
    for i in 1..=5 {
        let actual = (i * 1_000_000) as f64;
        baseline_history.record(ExecutionRecord {
            id: format!("baseline_{}", i),
            predicted_flops: actual * 1.2,
            actual_flops: actual,
            predicted_time_ms: actual / 100_000.0,
            actual_time_ms: actual / 100_000.0,
            predicted_memory: 1_000_000,
            actual_memory: 1_000_000,
            timestamp: SystemTime::now(),
            planner: "test".to_string(),
        });
    }

    let baseline_metrics = baseline_history.compute_metrics();
    println!("Without ML calibration:");
    println!(
        "  Average error: {:.1}%",
        baseline_metrics.avg_flops_error * 100.0
    );
    println!();

    // Simulate using calibrated predictions
    let mut calibrated_history = ExecutionHistory::with_max_size(100);
    for i in 1..=5 {
        let actual = (i * 1_000_000) as f64;
        let original_pred = actual * 1.2;
        let calibrated_pred = ml_model.calibrate_flops(original_pred);

        calibrated_history.record(ExecutionRecord {
            id: format!("calibrated_{}", i),
            predicted_flops: calibrated_pred,
            actual_flops: actual,
            predicted_time_ms: actual / 100_000.0,
            actual_time_ms: actual / 100_000.0,
            predicted_memory: 1_000_000,
            actual_memory: 1_000_000,
            timestamp: SystemTime::now(),
            planner: "test".to_string(),
        });
    }

    let calibrated_metrics = calibrated_history.compute_metrics();
    println!("With ML calibration:");
    println!(
        "  Average error: {:.1}%",
        calibrated_metrics.avg_flops_error * 100.0
    );
    println!();

    let improvement = (baseline_metrics.avg_flops_error - calibrated_metrics.avg_flops_error)
        / baseline_metrics.avg_flops_error
        * 100.0;
    println!(
        "Improvement: {:.1}% reduction in prediction error",
        improvement
    );
    println!();

    println!("=== Summary ===\n");
    println!("✓ ML cost models learn from execution history");
    println!("✓ Automatic correction of systematic biases");
    println!("✓ Planner-specific calibration for different algorithms");
    println!("✓ Improved prediction accuracy over time");
    println!("✓ Production-ready with R² quality metrics");

    Ok(())
}
