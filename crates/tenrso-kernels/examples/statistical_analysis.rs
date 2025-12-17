//! Statistical Analysis Example for TenRSo Kernels
//!
//! This example demonstrates the comprehensive statistical analysis capabilities
//! of tenrso-kernels, including:
//! - Basic statistics (mean, variance, std)
//! - Distribution analysis (median, percentile, skewness, kurtosis)
//! - Multivariate analysis (covariance, correlation)
//!
//! Run with:
//! ```bash
//! cargo run --example statistical_analysis
//! ```

use scirs2_core::ndarray_ext::{s, Array};
use tenrso_kernels::*;

fn main() {
    println!("=== TenRSo Statistical Analysis Demo ===\n");

    // ========================================================================
    // 1. Basic Statistics
    // ========================================================================
    println!("1. BASIC STATISTICS");
    println!("{}", "=".repeat(60));

    // Create a 3D tensor representing sensor readings
    // Shape: [3 sensors, 24 hours, 7 days]
    let sensor_data = Array::from_shape_fn(vec![3, 24, 7], |idx| {
        let sensor = idx[0];
        let hour = idx[1];
        let day = idx[2];

        // Simulate temperature readings with daily and hourly patterns
        20.0 + (sensor as f64) * 2.0              // Base temp by sensor
            + 5.0 * (hour as f64 / 24.0).sin()    // Diurnal pattern
            + 2.0 * (day as f64 / 7.0).cos()      // Weekly pattern
            + (sensor * hour + day) as f64 * 0.1 // Some variation
    });

    println!("Sensor data shape: {:?}", sensor_data.shape());
    println!("Total measurements: {}\n", sensor_data.len());

    // Compute statistics along the time dimensions (hours and days)
    println!("Computing statistics along time (modes 1 and 2):");

    let mean = mean_along_modes(&sensor_data.view(), &[1, 2]).unwrap();
    println!("  Mean temperature by sensor: {:?}", mean.shape());
    for (i, &temp) in mean.iter().enumerate() {
        println!("    Sensor {}: {:.2}°C", i, temp);
    }

    let std = std_along_modes(&sensor_data.view(), &[1, 2], 1).unwrap();
    println!("\n  Standard deviation by sensor:");
    for (i, &s) in std.iter().enumerate() {
        println!("    Sensor {}: {:.2}°C", i, s);
    }

    // ========================================================================
    // 2. Distribution Analysis
    // ========================================================================
    println!("\n2. DISTRIBUTION ANALYSIS");
    println!("{}", "=".repeat(60));

    // Analyze hourly temperature distribution for sensor 0
    let sensor_0 = Array::from_shape_vec(
        vec![1, 24, 7],
        sensor_data
            .slice(s![0..1, .., ..])
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();

    println!("Analyzing hourly patterns for Sensor 0:");

    let median = median_along_modes(&sensor_0.view(), &[1, 2]).unwrap();
    println!("  Median: {:.2}°C", median[[0]]);

    let q25 = percentile_along_modes(&sensor_0.view(), &[1, 2], 25.0).unwrap();
    let q75 = percentile_along_modes(&sensor_0.view(), &[1, 2], 75.0).unwrap();
    println!("  25th percentile: {:.2}°C", q25[[0]]);
    println!("  75th percentile: {:.2}°C", q75[[0]]);
    println!("  Interquartile range (IQR): {:.2}°C", q75[[0]] - q25[[0]]);

    let skew = skewness_along_modes(&sensor_0.view(), &[1, 2], true).unwrap();
    println!("\n  Skewness: {:.4}", skew[[0]]);
    if skew[[0]].abs() < 0.5 {
        println!("    → Distribution is approximately symmetric");
    } else if skew[[0]] > 0.0 {
        println!("    → Distribution has a right tail (positively skewed)");
    } else {
        println!("    → Distribution has a left tail (negatively skewed)");
    }

    let kurt = kurtosis_along_modes(&sensor_0.view(), &[1, 2], true, true).unwrap();
    println!("\n  Excess Kurtosis: {:.4}", kurt[[0]]);
    if kurt[[0]].abs() < 0.5 {
        println!("    → Distribution is approximately normal (mesokurtic)");
    } else if kurt[[0]] > 0.0 {
        println!("    → Distribution has heavy tails (leptokurtic)");
    } else {
        println!("    → Distribution has light tails (platykurtic)");
    }

    // ========================================================================
    // 3. Multivariate Analysis - Covariance
    // ========================================================================
    println!("\n3. MULTIVARIATE ANALYSIS - COVARIANCE");
    println!("{}", "=".repeat(60));

    // Compare two sensors over time
    let sensor_0_data = Array::from_shape_vec(
        vec![1, 24, 7],
        sensor_data
            .slice(s![0..1, .., ..])
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();

    let sensor_1_data = Array::from_shape_vec(
        vec![1, 24, 7],
        sensor_data
            .slice(s![1..2, .., ..])
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();

    println!("Analyzing relationship between Sensor 0 and Sensor 1:");

    let cov =
        covariance_along_modes(&sensor_0_data.view(), &sensor_1_data.view(), &[1, 2], 1).unwrap();
    println!("  Covariance: {:.4}", cov[[0]]);

    if cov[[0]] > 0.0 {
        println!("    → Positive covariance: sensors tend to vary together");
    } else if cov[[0]] < 0.0 {
        println!("    → Negative covariance: sensors vary inversely");
    } else {
        println!("    → Zero covariance: sensors vary independently");
    }

    // ========================================================================
    // 4. Multivariate Analysis - Correlation
    // ========================================================================
    println!("\n4. MULTIVARIATE ANALYSIS - CORRELATION");
    println!("{}", "=".repeat(60));

    let corr =
        correlation_along_modes(&sensor_0_data.view(), &sensor_1_data.view(), &[1, 2]).unwrap();
    println!("  Pearson correlation: {:.4}", corr[[0]]);

    let corr_val = corr[[0]];
    if corr_val.abs() > 0.9 {
        println!(
            "    → Very strong {} correlation",
            if corr_val > 0.0 {
                "positive"
            } else {
                "negative"
            }
        );
    } else if corr_val.abs() > 0.7 {
        println!(
            "    → Strong {} correlation",
            if corr_val > 0.0 {
                "positive"
            } else {
                "negative"
            }
        );
    } else if corr_val.abs() > 0.5 {
        println!(
            "    → Moderate {} correlation",
            if corr_val > 0.0 {
                "positive"
            } else {
                "negative"
            }
        );
    } else if corr_val.abs() > 0.3 {
        println!(
            "    → Weak {} correlation",
            if corr_val > 0.0 {
                "positive"
            } else {
                "negative"
            }
        );
    } else {
        println!("    → Very weak or no linear correlation");
    }

    // Correlation matrix for all sensors
    println!("\n  Correlation matrix (all sensors):");
    println!("  {:>10} {:>10} {:>10}", "Sensor 0", "Sensor 1", "Sensor 2");

    for i in 0..3 {
        print!("  Sensor {}: ", i);
        for j in 0..3 {
            let s_i = Array::from_shape_vec(
                vec![1, 24, 7],
                sensor_data
                    .slice(s![i..i + 1, .., ..])
                    .iter()
                    .copied()
                    .collect(),
            )
            .unwrap();
            let s_j = Array::from_shape_vec(
                vec![1, 24, 7],
                sensor_data
                    .slice(s![j..j + 1, .., ..])
                    .iter()
                    .copied()
                    .collect(),
            )
            .unwrap();

            let r = correlation_along_modes(&s_i.view(), &s_j.view(), &[1, 2]).unwrap();
            print!("{:10.4} ", r[[0]]);
        }
        println!();
    }

    // ========================================================================
    // 5. Practical Application: Anomaly Detection
    // ========================================================================
    println!("\n5. PRACTICAL APPLICATION: ANOMALY DETECTION");
    println!("{}", "=".repeat(60));

    println!("Detecting outliers using statistical thresholds:");

    // For each sensor, find hours where temp is > 2 std deviations from mean
    for sensor_id in 0..3 {
        let sensor = Array::from_shape_vec(
            vec![1, 24, 7],
            sensor_data
                .slice(s![sensor_id..sensor_id + 1, .., ..])
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();

        let sensor_mean = mean_along_modes(&sensor.view(), &[1, 2]).unwrap();
        let sensor_std = std_along_modes(&sensor.view(), &[1, 2], 1).unwrap();

        let threshold = 2.0 * sensor_std[[0]];

        println!(
            "\n  Sensor {} (μ={:.2}°C, σ={:.2}°C):",
            sensor_id,
            sensor_mean[[0]],
            sensor_std[[0]]
        );
        println!("    Outlier threshold: ±{:.2}°C from mean", threshold);

        // Count potential outliers
        let mut outlier_count = 0;
        for val in sensor.iter() {
            if (*val - sensor_mean[[0]]).abs() > threshold {
                outlier_count += 1;
            }
        }

        let outlier_pct = (outlier_count as f64 / sensor.len() as f64) * 100.0;
        println!(
            "    Outliers detected: {}/{} ({:.1}%)",
            outlier_count,
            sensor.len(),
            outlier_pct
        );
    }

    // ========================================================================
    // 6. Summary Statistics Report
    // ========================================================================
    println!("\n6. SUMMARY STATISTICS REPORT");
    println!("{}", "=".repeat(60));

    let overall_mean = mean_along_modes(&sensor_data.view(), &[0, 1, 2]).unwrap();
    let overall_std = std_along_modes(&sensor_data.view(), &[0, 1, 2], 1).unwrap();
    let overall_min = min_along_modes(&sensor_data.view(), &[0, 1, 2]).unwrap();
    let overall_max = max_along_modes(&sensor_data.view(), &[0, 1, 2]).unwrap();

    println!("Overall Statistics (all sensors, all time):");
    println!("  Mean:     {:.2}°C", overall_mean[[0]]);
    println!("  Std Dev:  {:.2}°C", overall_std[[0]]);
    println!("  Min:      {:.2}°C", overall_min[[0]]);
    println!("  Max:      {:.2}°C", overall_max[[0]]);
    println!("  Range:    {:.2}°C", overall_max[[0]] - overall_min[[0]]);

    println!("\n=== Analysis Complete ===");
    println!("\nThis example demonstrated:");
    println!("  ✓ Basic statistics (mean, std, min, max)");
    println!("  ✓ Distribution analysis (median, percentiles, skewness, kurtosis)");
    println!("  ✓ Multivariate analysis (covariance, correlation)");
    println!("  ✓ Practical application (anomaly detection)");
    println!("\nAll operations performed efficiently on multi-dimensional tensors!");
}
