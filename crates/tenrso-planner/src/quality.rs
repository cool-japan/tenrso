//! Plan quality tracking and adaptive tuning
//!
//! This module provides tools to track execution performance, compare predicted
//! vs actual costs, and adaptively tune planner parameters based on historical data.
//!
//! # Use Cases
//!
//! - **Quality Assessment**: Compare predicted vs actual execution metrics
//! - **Planner Calibration**: Adjust cost models based on real-world performance
//! - **Adaptive Selection**: Choose the best planner based on historical accuracy
//! - **Performance Monitoring**: Track plan quality over time
//!
//! # Examples
//!
//! ```
//! use tenrso_planner::{ExecutionHistory, ExecutionRecord, PlanQualityMetrics};
//! use std::time::SystemTime;
//!
//! let mut history = ExecutionHistory::new();
//!
//! // After executing a plan, record the actual performance
//! let record = ExecutionRecord {
//!     id: "matmul_1000x2000x3000".to_string(),
//!     predicted_flops: 100_000_000.0,
//!     actual_flops: 105_000_000.0,
//!     predicted_time_ms: 50.0,
//!     actual_time_ms: 52.5,
//!     predicted_memory: 1_000_000,
//!     actual_memory: 1_050_000,
//!     timestamp: SystemTime::now(),
//!     planner: "greedy".to_string(),
//! };
//! history.record(record);
//!
//! // Get quality metrics
//! let metrics = history.compute_metrics();
//! println!("Average prediction error: {:.1}%", metrics.avg_flops_error * 100.0);
//! ```

use std::collections::HashMap;

/// Record of a single plan execution
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Unique identifier for this execution (e.g., problem descriptor)
    pub id: String,
    /// Predicted FLOPs
    pub predicted_flops: f64,
    /// Actual FLOPs (measured)
    pub actual_flops: f64,
    /// Predicted execution time (ms)
    pub predicted_time_ms: f64,
    /// Actual execution time (ms)
    pub actual_time_ms: f64,
    /// Predicted peak memory (bytes)
    pub predicted_memory: usize,
    /// Actual peak memory (bytes)
    pub actual_memory: usize,
    /// Timestamp of execution
    pub timestamp: std::time::SystemTime,
    /// Planner used (e.g., "greedy", "dp", "beam_search")
    pub planner: String,
}

impl ExecutionRecord {
    /// Calculate FLOPs prediction error (relative)
    pub fn flops_error(&self) -> f64 {
        if self.actual_flops == 0.0 {
            return 0.0;
        }
        (self.predicted_flops - self.actual_flops).abs() / self.actual_flops
    }

    /// Calculate time prediction error (relative)
    pub fn time_error(&self) -> f64 {
        if self.actual_time_ms == 0.0 {
            return 0.0;
        }
        (self.predicted_time_ms - self.actual_time_ms).abs() / self.actual_time_ms
    }

    /// Calculate memory prediction error (relative)
    pub fn memory_error(&self) -> f64 {
        if self.actual_memory == 0 {
            return 0.0;
        }
        let predicted = self.predicted_memory as f64;
        let actual = self.actual_memory as f64;
        (predicted - actual).abs() / actual
    }

    /// Check if prediction was accurate (within tolerance)
    pub fn is_accurate(&self, tolerance: f64) -> bool {
        self.flops_error() <= tolerance
            && self.time_error() <= tolerance
            && self.memory_error() <= tolerance
    }
}

/// Quality metrics computed from execution history
#[derive(Debug, Clone)]
pub struct PlanQualityMetrics {
    /// Number of executions recorded
    pub num_executions: usize,
    /// Average FLOPs prediction error (0.0 = perfect, 1.0 = 100% error)
    pub avg_flops_error: f64,
    /// Average time prediction error
    pub avg_time_error: f64,
    /// Average memory prediction error
    pub avg_memory_error: f64,
    /// Maximum FLOPs error observed
    pub max_flops_error: f64,
    /// Maximum time error observed
    pub max_time_error: f64,
    /// Maximum memory error observed
    pub max_memory_error: f64,
    /// Percentage of predictions within 10% tolerance
    pub accuracy_10pct: f64,
    /// Percentage of predictions within 20% tolerance
    pub accuracy_20pct: f64,
    /// Per-planner metrics
    pub per_planner: HashMap<String, PlannerMetrics>,
}

/// Metrics for a specific planner
#[derive(Debug, Clone)]
pub struct PlannerMetrics {
    /// Number of executions for this planner
    pub count: usize,
    /// Average FLOPs error for this planner
    pub avg_flops_error: f64,
    /// Average time error for this planner
    pub avg_time_error: f64,
    /// Average memory error for this planner
    pub avg_memory_error: f64,
}

/// Execution history tracker
///
/// Maintains a history of plan executions with predicted and actual performance,
/// enabling quality assessment and adaptive tuning.
///
/// # Thread Safety
///
/// `ExecutionHistory` is not thread-safe by default. Wrap in `Arc<Mutex<...>>`
/// for concurrent access.
#[derive(Debug, Clone)]
pub struct ExecutionHistory {
    /// All execution records
    records: Vec<ExecutionRecord>,
    /// Maximum number of records to keep (0 = unlimited)
    max_records: usize,
}

impl ExecutionHistory {
    /// Create a new execution history tracker
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::ExecutionHistory;
    ///
    /// let history = ExecutionHistory::new();
    /// assert_eq!(history.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            max_records: 0, // unlimited
        }
    }

    /// Create a new execution history with a maximum size
    ///
    /// When the maximum is reached, oldest records are evicted (FIFO).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::ExecutionHistory;
    ///
    /// let history = ExecutionHistory::with_max_size(1000);
    /// ```
    pub fn with_max_size(max_records: usize) -> Self {
        Self {
            records: Vec::new(),
            max_records,
        }
    }

    /// Record a plan execution
    ///
    /// # Arguments
    ///
    /// * `record` - ExecutionRecord with predicted and actual metrics
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{ExecutionHistory, ExecutionRecord};
    /// use std::time::SystemTime;
    ///
    /// let mut history = ExecutionHistory::new();
    /// let record = ExecutionRecord {
    ///     id: "matmul_100x200x300".to_string(),
    ///     predicted_flops: 12_000_000.0,
    ///     actual_flops: 12_500_000.0,
    ///     predicted_time_ms: 10.0,
    ///     actual_time_ms: 10.5,
    ///     predicted_memory: 100_000,
    ///     actual_memory: 105_000,
    ///     timestamp: SystemTime::now(),
    ///     planner: "greedy".to_string(),
    /// };
    /// history.record(record);
    /// assert_eq!(history.len(), 1);
    /// ```
    pub fn record(&mut self, record: ExecutionRecord) {
        self.records.push(record);

        // Enforce max size
        if self.max_records > 0 && self.records.len() > self.max_records {
            self.records.remove(0); // Remove oldest
        }
    }

    /// Get the number of recorded executions
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if the history is empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Clear all execution records
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Get all execution records
    pub fn records(&self) -> &[ExecutionRecord] {
        &self.records
    }

    /// Compute quality metrics from all recorded executions
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{ExecutionHistory, ExecutionRecord};
    /// use std::time::SystemTime;
    ///
    /// let mut history = ExecutionHistory::new();
    /// history.record(ExecutionRecord {
    ///     id: "test1".to_string(),
    ///     predicted_flops: 100.0,
    ///     actual_flops: 105.0,
    ///     predicted_time_ms: 10.0,
    ///     actual_time_ms: 10.5,
    ///     predicted_memory: 1000,
    ///     actual_memory: 1050,
    ///     timestamp: SystemTime::now(),
    ///     planner: "greedy".to_string(),
    /// });
    ///
    /// let metrics = history.compute_metrics();
    /// assert!(metrics.avg_flops_error < 0.1); // Less than 10% error
    /// ```
    pub fn compute_metrics(&self) -> PlanQualityMetrics {
        if self.records.is_empty() {
            return PlanQualityMetrics {
                num_executions: 0,
                avg_flops_error: 0.0,
                avg_time_error: 0.0,
                avg_memory_error: 0.0,
                max_flops_error: 0.0,
                max_time_error: 0.0,
                max_memory_error: 0.0,
                accuracy_10pct: 0.0,
                accuracy_20pct: 0.0,
                per_planner: HashMap::new(),
            };
        }

        let num_executions = self.records.len();
        let mut total_flops_error = 0.0;
        let mut total_time_error = 0.0;
        let mut total_memory_error = 0.0;
        let mut max_flops_error: f64 = 0.0;
        let mut max_time_error: f64 = 0.0;
        let mut max_memory_error: f64 = 0.0;
        let mut accurate_10pct = 0;
        let mut accurate_20pct = 0;

        // Per-planner tracking
        let mut planner_stats: HashMap<String, (usize, f64, f64, f64)> = HashMap::new();

        for record in &self.records {
            let flops_err = record.flops_error();
            let time_err = record.time_error();
            let mem_err = record.memory_error();

            total_flops_error += flops_err;
            total_time_error += time_err;
            total_memory_error += mem_err;

            max_flops_error = max_flops_error.max(flops_err);
            max_time_error = max_time_error.max(time_err);
            max_memory_error = max_memory_error.max(mem_err);

            if record.is_accurate(0.1) {
                accurate_10pct += 1;
            }
            if record.is_accurate(0.2) {
                accurate_20pct += 1;
            }

            // Update per-planner stats
            let entry = planner_stats
                .entry(record.planner.clone())
                .or_insert((0, 0.0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += flops_err;
            entry.2 += time_err;
            entry.3 += mem_err;
        }

        // Compute per-planner metrics
        let per_planner = planner_stats
            .into_iter()
            .map(|(planner, (count, flops_sum, time_sum, mem_sum))| {
                let metrics = PlannerMetrics {
                    count,
                    avg_flops_error: flops_sum / count as f64,
                    avg_time_error: time_sum / count as f64,
                    avg_memory_error: mem_sum / count as f64,
                };
                (planner, metrics)
            })
            .collect();

        PlanQualityMetrics {
            num_executions,
            avg_flops_error: total_flops_error / num_executions as f64,
            avg_time_error: total_time_error / num_executions as f64,
            avg_memory_error: total_memory_error / num_executions as f64,
            max_flops_error,
            max_time_error,
            max_memory_error,
            accuracy_10pct: accurate_10pct as f64 / num_executions as f64,
            accuracy_20pct: accurate_20pct as f64 / num_executions as f64,
            per_planner,
        }
    }

    /// Get the most accurate planner based on historical performance
    ///
    /// Returns the planner name with the lowest average error across all metrics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{ExecutionHistory, ExecutionRecord};
    /// use std::time::SystemTime;
    ///
    /// let mut history = ExecutionHistory::new();
    /// history.record(ExecutionRecord {
    ///     id: "test1".to_string(),
    ///     predicted_flops: 100.0,
    ///     actual_flops: 102.0,
    ///     predicted_time_ms: 10.0,
    ///     actual_time_ms: 10.2,
    ///     predicted_memory: 1000,
    ///     actual_memory: 1020,
    ///     timestamp: SystemTime::now(),
    ///     planner: "greedy".to_string(),
    /// });
    ///
    /// if let Some(best) = history.best_planner() {
    ///     println!("Best planner: {}", best);
    /// }
    /// ```
    pub fn best_planner(&self) -> Option<String> {
        let metrics = self.compute_metrics();

        if metrics.per_planner.is_empty() {
            return None;
        }

        let best = metrics
            .per_planner
            .iter()
            .min_by(|(_, a), (_, b)| {
                let avg_a = (a.avg_flops_error + a.avg_time_error + a.avg_memory_error) / 3.0;
                let avg_b = (b.avg_flops_error + b.avg_time_error + b.avg_memory_error) / 3.0;
                avg_a.partial_cmp(&avg_b).unwrap()
            })
            .map(|(name, _)| name.clone());

        best
    }

    /// Get a calibration factor for cost model adjustment
    ///
    /// Returns the average ratio of actual to predicted FLOPs, which can be used
    /// to calibrate future predictions.
    ///
    /// # Returns
    ///
    /// Calibration factor (e.g., 1.05 means predictions are 5% too low)
    pub fn get_calibration_factor(&self) -> f64 {
        if self.records.is_empty() {
            return 1.0;
        }

        let total_ratio: f64 = self
            .records
            .iter()
            .map(|r| {
                if r.predicted_flops > 0.0 {
                    r.actual_flops / r.predicted_flops
                } else {
                    1.0
                }
            })
            .sum();

        total_ratio / self.records.len() as f64
    }
}

impl Default for ExecutionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function for creating test records
    #[allow(clippy::too_many_arguments)]
    fn make_record(
        id: &str,
        predicted_flops: f64,
        actual_flops: f64,
        predicted_time_ms: f64,
        actual_time_ms: f64,
        predicted_memory: usize,
        actual_memory: usize,
        planner: &str,
    ) -> ExecutionRecord {
        ExecutionRecord {
            id: id.to_string(),
            predicted_flops,
            actual_flops,
            predicted_time_ms,
            actual_time_ms,
            predicted_memory,
            actual_memory,
            timestamp: std::time::SystemTime::now(),
            planner: planner.to_string(),
        }
    }

    #[test]
    fn test_execution_record_errors() {
        let record = ExecutionRecord {
            id: "test".to_string(),
            predicted_flops: 100.0,
            actual_flops: 110.0,
            predicted_time_ms: 10.0,
            actual_time_ms: 11.0,
            predicted_memory: 1000,
            actual_memory: 1100,
            timestamp: std::time::SystemTime::now(),
            planner: "greedy".to_string(),
        };

        assert!((record.flops_error() - 0.0909).abs() < 0.001); // ~9.09% error
        assert!((record.time_error() - 0.0909).abs() < 0.001);
        assert!((record.memory_error() - 0.0909).abs() < 0.001);
    }

    #[test]
    fn test_execution_record_accuracy() {
        let accurate = ExecutionRecord {
            id: "test".to_string(),
            predicted_flops: 100.0,
            actual_flops: 105.0,
            predicted_time_ms: 10.0,
            actual_time_ms: 10.3,
            predicted_memory: 1000,
            actual_memory: 1040,
            timestamp: std::time::SystemTime::now(),
            planner: "greedy".to_string(),
        };

        assert!(accurate.is_accurate(0.1)); // Within 10%
        assert!(!accurate.is_accurate(0.01)); // Not within 1%
    }

    #[test]
    fn test_execution_history_basic() {
        let mut history = ExecutionHistory::new();
        assert_eq!(history.len(), 0);
        assert!(history.is_empty());

        history.record(make_record(
            "test1", 100.0, 105.0, 10.0, 10.5, 1000, 1050, "greedy",
        ));
        assert_eq!(history.len(), 1);
        assert!(!history.is_empty());

        history.clear();
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_execution_history_max_size() {
        let mut history = ExecutionHistory::with_max_size(2);

        history.record(make_record(
            "test1", 100.0, 105.0, 10.0, 10.5, 1000, 1050, "greedy",
        ));
        history.record(make_record(
            "test2", 200.0, 210.0, 20.0, 21.0, 2000, 2100, "greedy",
        ));
        assert_eq!(history.len(), 2);

        // Adding a third should evict the first
        history.record(make_record(
            "test3", 300.0, 315.0, 30.0, 31.5, 3000, 3150, "greedy",
        ));
        assert_eq!(history.len(), 2);
        assert_eq!(history.records()[0].id, "test2");
        assert_eq!(history.records()[1].id, "test3");
    }

    #[test]
    fn test_compute_metrics() {
        let mut history = ExecutionHistory::new();

        // Add some records with known errors
        history.record(make_record(
            "test1", 100.0, 110.0, 10.0, 11.0, 1000, 1100, "greedy",
        )); // 10% error
        history.record(make_record(
            "test2", 200.0, 210.0, 20.0, 21.0, 2000, 2100, "greedy",
        )); // 5% error
        history.record(make_record(
            "test3", 300.0, 330.0, 30.0, 33.0, 3000, 3300, "greedy",
        )); // 10% error

        let metrics = history.compute_metrics();
        assert_eq!(metrics.num_executions, 3);
        assert!(metrics.avg_flops_error > 0.0);
        assert!(metrics.avg_time_error > 0.0);
        assert!(metrics.avg_memory_error > 0.0);
    }

    #[test]
    fn test_per_planner_metrics() {
        let mut history = ExecutionHistory::new();

        history.record(make_record(
            "test1", 100.0, 105.0, 10.0, 10.5, 1000, 1050, "greedy",
        ));
        history.record(make_record(
            "test2", 100.0, 120.0, 10.0, 12.0, 1000, 1200, "dp",
        ));
        history.record(make_record(
            "test3", 100.0, 103.0, 10.0, 10.3, 1000, 1030, "greedy",
        ));

        let metrics = history.compute_metrics();
        assert_eq!(metrics.per_planner.len(), 2);
        assert_eq!(metrics.per_planner["greedy"].count, 2);
        assert_eq!(metrics.per_planner["dp"].count, 1);
    }

    #[test]
    fn test_best_planner() {
        let mut history = ExecutionHistory::new();

        // Greedy is more accurate
        history.record(make_record(
            "test1", 100.0, 102.0, 10.0, 10.2, 1000, 1020, "greedy",
        ));
        history.record(make_record(
            "test2", 100.0, 103.0, 10.0, 10.3, 1000, 1030, "greedy",
        ));

        // DP is less accurate
        history.record(make_record(
            "test3", 100.0, 115.0, 10.0, 11.5, 1000, 1150, "dp",
        ));

        let best = history.best_planner();
        assert_eq!(best, Some("greedy".to_string()));
    }

    #[test]
    fn test_calibration_factor() {
        let mut history = ExecutionHistory::new();

        // Predictions are consistently 10% too low
        history.record(make_record(
            "test1", 100.0, 110.0, 10.0, 11.0, 1000, 1100, "greedy",
        ));
        history.record(make_record(
            "test2", 200.0, 220.0, 20.0, 22.0, 2000, 2200, "greedy",
        ));

        let calibration = history.get_calibration_factor();
        assert!((calibration - 1.1).abs() < 0.01); // Should be ~1.1
    }

    #[test]
    fn test_accuracy_percentages() {
        let mut history = ExecutionHistory::new();

        // 2 within 10%, 1 not
        history.record(make_record(
            "test1", 100.0, 105.0, 10.0, 10.5, 1000, 1050, "greedy",
        )); // 5% error
        history.record(make_record(
            "test2", 100.0, 108.0, 10.0, 10.8, 1000, 1080, "greedy",
        )); // 8% error
        history.record(make_record(
            "test3", 100.0, 125.0, 10.0, 12.5, 1000, 1250, "greedy",
        )); // 25% error

        let metrics = history.compute_metrics();
        assert!((metrics.accuracy_10pct - 0.666).abs() < 0.01); // 2/3
        assert_eq!(metrics.accuracy_20pct, 1.0); // 3/3
    }

    #[test]
    fn test_empty_metrics() {
        let history = ExecutionHistory::new();
        let metrics = history.compute_metrics();

        assert_eq!(metrics.num_executions, 0);
        assert_eq!(metrics.avg_flops_error, 0.0);
        assert!(metrics.per_planner.is_empty());
    }
}
