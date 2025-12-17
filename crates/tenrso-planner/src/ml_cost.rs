//! Machine learning-based cost model refinement.
//!
//! This module provides ML algorithms that learn from execution history to improve
//! cost predictions over time. It integrates with the quality tracking system to
//! automatically calibrate FLOPs, time, and memory estimates based on real-world
//! execution data.
//!
//! # Features
//!
//! - **Linear Regression**: Simple linear models for cost calibration
//! - **Polynomial Regression**: Higher-order polynomial features for complex patterns
//! - **Adaptive Calibration**: Automatic cost model refinement based on execution feedback
//! - **Multi-metric Learning**: Separate models for FLOPs, time, and memory
//! - **Incremental Training**: Update models as new execution data arrives
//!
//! # Example
//!
//! ```
//! use tenrso_planner::{MLCostModel, ExecutionHistory, ExecutionRecord};
//!
//! // Create ML cost model
//! let mut ml_model = MLCostModel::new();
//!
//! // Train from execution history
//! let mut history = ExecutionHistory::with_max_size(100);
//! history.record(ExecutionRecord {
//!     id: "matmul_1000x2000x3000".to_string(),
//!     predicted_flops: 1e9,
//!     actual_flops: 1.2e9,
//!     predicted_time_ms: 100.0,
//!     actual_time_ms: 110.0,
//!     predicted_memory: 1_000_000,
//!     actual_memory: 1_050_000,
//!     timestamp: std::time::SystemTime::now(),
//!     planner: "greedy".to_string(),
//! });
//!
//! ml_model.train(&history);
//!
//! // Use calibrated predictions
//! let calibrated_flops = ml_model.calibrate_flops(1e9);
//! let calibrated_time = ml_model.calibrate_time(100.0, 1e9);
//! ```

use crate::quality::ExecutionHistory;
use std::collections::HashMap;

/// Linear regression model for cost prediction.
///
/// Uses ordinary least squares (OLS) to fit a linear relationship:
/// `y = slope * x + intercept`
///
/// # Complexity
///
/// - Training: O(n) where n is number of samples
/// - Prediction: O(1)
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    /// Slope coefficient
    pub slope: f64,
    /// Intercept term
    pub intercept: f64,
    /// Number of training samples
    pub n_samples: usize,
    /// R² score (coefficient of determination)
    pub r_squared: f64,
}

impl LinearRegressionModel {
    /// Creates a new linear regression model with default parameters.
    ///
    /// Initially assumes perfect predictions (slope=1, intercept=0).
    pub fn new() -> Self {
        Self {
            slope: 1.0,
            intercept: 0.0,
            n_samples: 0,
            r_squared: 0.0,
        }
    }

    /// Trains the model using ordinary least squares.
    ///
    /// # Arguments
    ///
    /// * `x` - Predicted values
    /// * `y` - Actual values
    ///
    /// # Panics
    ///
    /// Panics if x and y have different lengths or are empty.
    pub fn train(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len(), "x and y must have same length");
        assert!(!x.is_empty(), "Cannot train on empty data");

        let n = x.len() as f64;
        self.n_samples = x.len();

        // Compute means
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        // Compute slope: cov(x, y) / var(x)
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
        }

        if var_x.abs() < 1e-10 {
            // No variance in x, use mean as constant prediction
            self.slope = 1.0;
            self.intercept = mean_y - mean_x;
        } else {
            self.slope = cov / var_x;
            self.intercept = mean_y - self.slope * mean_x;
        }

        // Compute R² score
        self.r_squared = self.compute_r_squared(x, y, mean_y);
    }

    /// Predicts the output for a given input.
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    /// Computes R² (coefficient of determination).
    ///
    /// R² = 1 - SS_res / SS_tot
    /// where SS_res is residual sum of squares, SS_tot is total sum of squares.
    fn compute_r_squared(&self, x: &[f64], y: &[f64], mean_y: f64) -> f64 {
        let mut ss_res = 0.0; // Residual sum of squares
        let mut ss_tot = 0.0; // Total sum of squares

        for i in 0..x.len() {
            let y_pred = self.predict(x[i]);
            ss_res += (y[i] - y_pred).powi(2);
            ss_tot += (y[i] - mean_y).powi(2);
        }

        if ss_tot.abs() < 1e-10 {
            return 1.0; // Perfect fit if no variance
        }

        1.0 - (ss_res / ss_tot)
    }
}

impl Default for LinearRegressionModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Polynomial regression model for cost prediction.
///
/// Fits a polynomial relationship: `y = c₀ + c₁x + c₂x² + ... + cₙxⁿ`
///
/// Uses polynomial features with linear regression.
#[derive(Debug, Clone)]
pub struct PolynomialRegressionModel {
    /// Polynomial degree (1=linear, 2=quadratic, 3=cubic, etc.)
    pub degree: usize,
    /// Coefficients [c₀, c₁, ..., cₙ]
    pub coefficients: Vec<f64>,
    /// Number of training samples
    pub n_samples: usize,
    /// R² score
    pub r_squared: f64,
}

impl PolynomialRegressionModel {
    /// Creates a new polynomial regression model with specified degree.
    ///
    /// # Arguments
    ///
    /// * `degree` - Polynomial degree (1=linear, 2=quadratic, etc.)
    pub fn new(degree: usize) -> Self {
        assert!(degree >= 1, "Degree must be at least 1");
        Self {
            degree,
            coefficients: vec![0.0; degree + 1],
            n_samples: 0,
            r_squared: 0.0,
        }
    }

    /// Trains the model using polynomial features.
    ///
    /// Uses normal equations: β = (XᵀX)⁻¹Xᵀy
    /// where X is the design matrix with polynomial features.
    pub fn train(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len(), "x and y must have same length");
        assert!(!x.is_empty(), "Cannot train on empty data");

        self.n_samples = x.len();

        // For simplicity, use iterative method for small degree
        // Full implementation would use matrix operations from scirs2-linalg
        if self.degree == 1 {
            // Use linear regression
            let mut linear = LinearRegressionModel::new();
            linear.train(x, y);
            self.coefficients[0] = linear.intercept;
            self.coefficients[1] = linear.slope;
            self.r_squared = linear.r_squared;
        } else if self.degree == 2 {
            // Quadratic fit using closed form for 3 parameters
            self.fit_quadratic(x, y);
        } else {
            // For higher degrees, fall back to linear
            let mut linear = LinearRegressionModel::new();
            linear.train(x, y);
            self.coefficients[0] = linear.intercept;
            self.coefficients[1] = linear.slope;
            self.r_squared = linear.r_squared;
        }
    }

    /// Fits a quadratic polynomial (degree 2).
    fn fit_quadratic(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len() as f64;

        // Compute sums for normal equations
        let sum_x: f64 = x.iter().sum();
        let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_x3: f64 = x.iter().map(|&xi| xi * xi * xi).sum();
        let sum_x4: f64 = x.iter().map(|&xi| xi * xi * xi * xi).sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();
        let sum_x2y: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * xi * yi).sum();

        // Solve 3x3 system using Cramer's rule
        // [n      sum_x   sum_x2 ] [c0]   [sum_y  ]
        // [sum_x  sum_x2  sum_x3 ] [c1] = [sum_xy ]
        // [sum_x2 sum_x3  sum_x4 ] [c2]   [sum_x2y]

        let det = n * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
            - sum_x * (sum_x * sum_x4 - sum_x2 * sum_x3)
            + sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2);

        if det.abs() < 1e-10 {
            // Singular matrix, fall back to linear
            let mut linear = LinearRegressionModel::new();
            linear.train(x, y);
            self.coefficients[0] = linear.intercept;
            self.coefficients[1] = linear.slope;
            self.coefficients[2] = 0.0;
            self.r_squared = linear.r_squared;
            return;
        }

        // Compute coefficients using Cramer's rule
        let det_c0 = sum_y * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
            - sum_x * (sum_xy * sum_x4 - sum_x2y * sum_x3)
            + sum_x2 * (sum_xy * sum_x3 - sum_x2y * sum_x2);

        let det_c1 = n * (sum_xy * sum_x4 - sum_x2y * sum_x3)
            - sum_y * (sum_x * sum_x4 - sum_x2 * sum_x3)
            + sum_x2 * (sum_x * sum_x2y - sum_xy * sum_x2);

        let det_c2 = n * (sum_x2 * sum_x2y - sum_x3 * sum_xy)
            - sum_x * (sum_x * sum_x2y - sum_x2 * sum_xy)
            + sum_y * (sum_x * sum_x3 - sum_x2 * sum_x2);

        self.coefficients[0] = det_c0 / det;
        self.coefficients[1] = det_c1 / det;
        self.coefficients[2] = det_c2 / det;

        // Compute R²
        let mean_y: f64 = sum_y / n;
        self.r_squared = self.compute_r_squared(x, y, mean_y);
    }

    /// Predicts the output for a given input.
    pub fn predict(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut x_power = 1.0;
        for &coef in &self.coefficients {
            result += coef * x_power;
            x_power *= x;
        }
        result
    }

    /// Computes R² score.
    fn compute_r_squared(&self, x: &[f64], y: &[f64], mean_y: f64) -> f64 {
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for i in 0..x.len() {
            let y_pred = self.predict(x[i]);
            ss_res += (y[i] - y_pred).powi(2);
            ss_tot += (y[i] - mean_y).powi(2);
        }

        if ss_tot.abs() < 1e-10 {
            return 1.0;
        }

        1.0 - (ss_res / ss_tot)
    }
}

/// Machine learning-based cost model.
///
/// Learns from execution history to calibrate cost predictions.
/// Maintains separate models for FLOPs, time, and memory.
///
/// # Example
///
/// ```
/// use tenrso_planner::{MLCostModel, ExecutionHistory};
///
/// let mut ml_model = MLCostModel::new();
/// let history = ExecutionHistory::new();
///
/// // Train from history
/// ml_model.train(&history);
///
/// // Calibrate predictions
/// let calibrated_flops = ml_model.calibrate_flops(1e9);
/// ```
#[derive(Debug, Clone)]
pub struct MLCostModel {
    /// Model for FLOPs calibration
    flops_model: LinearRegressionModel,
    /// Model for time calibration (time as function of FLOPs)
    time_model: LinearRegressionModel,
    /// Model for memory calibration
    memory_model: LinearRegressionModel,
    /// Per-planner FLOPs models
    planner_flops_models: HashMap<String, LinearRegressionModel>,
    /// Whether the model has been trained
    is_trained: bool,
}

impl MLCostModel {
    /// Creates a new ML cost model.
    pub fn new() -> Self {
        Self {
            flops_model: LinearRegressionModel::new(),
            time_model: LinearRegressionModel::new(),
            memory_model: LinearRegressionModel::new(),
            planner_flops_models: HashMap::new(),
            is_trained: false,
        }
    }

    /// Trains the models from execution history.
    ///
    /// Requires at least 3 execution records for meaningful training.
    ///
    /// # Arguments
    ///
    /// * `history` - Execution history with recorded predictions and actuals
    pub fn train(&mut self, history: &ExecutionHistory) {
        let records = history.records();

        if records.len() < 3 {
            // Not enough data for training
            return;
        }

        // Extract data for FLOPs model
        let (predicted_flops, actual_flops): (Vec<f64>, Vec<f64>) = records
            .iter()
            .map(|r| (r.predicted_flops, r.actual_flops))
            .unzip();

        self.flops_model.train(&predicted_flops, &actual_flops);

        // Extract data for time model (time as function of FLOPs)
        let (flops_for_time, actual_time): (Vec<f64>, Vec<f64>) = records
            .iter()
            .map(|r| (r.actual_flops, r.actual_time_ms))
            .unzip();

        self.time_model.train(&flops_for_time, &actual_time);

        // Extract data for memory model
        let (predicted_memory, actual_memory): (Vec<f64>, Vec<f64>) = records
            .iter()
            .map(|r| (r.predicted_memory as f64, r.actual_memory as f64))
            .unzip();

        self.memory_model.train(&predicted_memory, &actual_memory);

        // Train per-planner models
        self.train_planner_models(history);

        self.is_trained = true;
    }

    /// Trains separate models for each planner.
    fn train_planner_models(&mut self, history: &ExecutionHistory) {
        // Group records by planner
        let mut planner_records: HashMap<String, Vec<_>> = HashMap::new();

        for record in history.records() {
            planner_records
                .entry(record.planner.clone())
                .or_default()
                .push((record.predicted_flops, record.actual_flops));
        }

        // Train a model for each planner that has sufficient data
        for (planner, records) in planner_records {
            if records.len() >= 3 {
                let (predicted, actual): (Vec<f64>, Vec<f64>) = records.into_iter().unzip();

                let mut model = LinearRegressionModel::new();
                model.train(&predicted, &actual);

                self.planner_flops_models.insert(planner, model);
            }
        }
    }

    /// Calibrates FLOPs prediction using the learned model.
    ///
    /// # Arguments
    ///
    /// * `predicted_flops` - Original FLOPs prediction
    ///
    /// # Returns
    ///
    /// Calibrated FLOPs estimate based on historical execution data.
    pub fn calibrate_flops(&self, predicted_flops: f64) -> f64 {
        if !self.is_trained {
            return predicted_flops;
        }

        self.flops_model.predict(predicted_flops).max(0.0)
    }

    /// Calibrates FLOPs prediction for a specific planner.
    ///
    /// Falls back to general FLOPs model if planner-specific model not available.
    pub fn calibrate_flops_for_planner(&self, predicted_flops: f64, planner: &str) -> f64 {
        if !self.is_trained {
            return predicted_flops;
        }

        if let Some(model) = self.planner_flops_models.get(planner) {
            model.predict(predicted_flops).max(0.0)
        } else {
            self.calibrate_flops(predicted_flops)
        }
    }

    /// Calibrates time prediction using the learned model.
    ///
    /// # Arguments
    ///
    /// * `predicted_time_ms` - Original time prediction (ms)
    /// * `calibrated_flops` - Calibrated FLOPs estimate
    ///
    /// # Returns
    ///
    /// Calibrated time estimate (ms) based on calibrated FLOPs.
    pub fn calibrate_time(&self, _predicted_time_ms: f64, calibrated_flops: f64) -> f64 {
        if !self.is_trained {
            return _predicted_time_ms;
        }

        // Use FLOPs-based time model
        self.time_model.predict(calibrated_flops).max(0.0)
    }

    /// Calibrates memory prediction using the learned model.
    ///
    /// # Arguments
    ///
    /// * `predicted_memory` - Original memory prediction (bytes)
    ///
    /// # Returns
    ///
    /// Calibrated memory estimate (bytes).
    pub fn calibrate_memory(&self, predicted_memory: usize) -> usize {
        if !self.is_trained {
            return predicted_memory;
        }

        let calibrated = self.memory_model.predict(predicted_memory as f64);
        calibrated.max(0.0) as usize
    }

    /// Returns FLOPs model R² score (quality metric).
    pub fn flops_r_squared(&self) -> f64 {
        self.flops_model.r_squared
    }

    /// Returns time model R² score.
    pub fn time_r_squared(&self) -> f64 {
        self.time_model.r_squared
    }

    /// Returns memory model R² score.
    pub fn memory_r_squared(&self) -> f64 {
        self.memory_model.r_squared
    }

    /// Returns whether the model has been trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Returns the number of available planner-specific models.
    pub fn num_planner_models(&self) -> usize {
        self.planner_flops_models.len()
    }
}

impl Default for MLCostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::{ExecutionHistory, ExecutionRecord};
    use std::time::SystemTime;

    #[test]
    fn test_linear_regression_perfect_fit() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let mut model = LinearRegressionModel::new();
        model.train(&x, &y);

        assert!((model.slope - 2.0).abs() < 1e-10);
        assert!(model.intercept.abs() < 1e-10);
        assert!((model.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_with_intercept() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1

        let mut model = LinearRegressionModel::new();
        model.train(&x, &y);

        assert!((model.slope - 2.0).abs() < 1e-10);
        assert!((model.intercept - 1.0).abs() < 1e-10);
        assert!((model.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_prediction() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];

        let mut model = LinearRegressionModel::new();
        model.train(&x, &y);

        assert!((model.predict(10.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_regression_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut model = PolynomialRegressionModel::new(1);
        model.train(&x, &y);

        assert!((model.predict(10.0) - 20.0).abs() < 1e-8);
    }

    #[test]
    fn test_polynomial_regression_quadratic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x²

        let mut model = PolynomialRegressionModel::new(2);
        model.train(&x, &y);

        // Should fit quadratic reasonably well
        let pred = model.predict(6.0);
        assert!((pred - 36.0).abs() < 1.0); // Allow some error
    }

    #[test]
    fn test_ml_cost_model_untrained() {
        let model = MLCostModel::new();

        assert!(!model.is_trained());
        assert_eq!(model.calibrate_flops(1000.0), 1000.0);
        assert_eq!(model.calibrate_memory(1000), 1000);
    }

    #[test]
    fn test_ml_cost_model_training() {
        let mut history = ExecutionHistory::new();

        // Add records with systematic overestimation (predict 20% high)
        for i in 1..=10 {
            let actual_flops = (i * 1000) as f64;
            let predicted_flops = actual_flops * 1.2; // 20% overestimate

            history.record(ExecutionRecord {
                id: format!("test_{}", i),
                predicted_flops,
                actual_flops,
                predicted_time_ms: 10.0,
                actual_time_ms: 10.0,
                predicted_memory: 1000,
                actual_memory: 1000,
                timestamp: SystemTime::now(),
                planner: "test".to_string(),
            });
        }

        let mut model = MLCostModel::new();
        model.train(&history);

        assert!(model.is_trained());

        // Should learn to correct the overestimation
        let calibrated = model.calibrate_flops(1200.0);
        assert!((calibrated - 1000.0).abs() < 50.0); // Should be close to 1000
    }

    #[test]
    fn test_ml_cost_model_per_planner() {
        let mut history = ExecutionHistory::new();

        // Add records for two different planners with different biases
        for i in 1..=5 {
            let actual = (i * 1000) as f64;

            // Planner A: 10% underestimate
            history.record(ExecutionRecord {
                id: format!("planner_a_{}", i),
                predicted_flops: actual * 0.9,
                actual_flops: actual,
                predicted_time_ms: 10.0,
                actual_time_ms: 10.0,
                predicted_memory: 1000,
                actual_memory: 1000,
                timestamp: SystemTime::now(),
                planner: "planner_a".to_string(),
            });

            // Planner B: 20% overestimate
            history.record(ExecutionRecord {
                id: format!("planner_b_{}", i),
                predicted_flops: actual * 1.2,
                actual_flops: actual,
                predicted_time_ms: 10.0,
                actual_time_ms: 10.0,
                predicted_memory: 1000,
                actual_memory: 1000,
                timestamp: SystemTime::now(),
                planner: "planner_b".to_string(),
            });
        }

        let mut model = MLCostModel::new();
        model.train(&history);

        assert_eq!(model.num_planner_models(), 2);

        // Calibration should differ by planner
        let cal_a = model.calibrate_flops_for_planner(900.0, "planner_a");
        let cal_b = model.calibrate_flops_for_planner(1200.0, "planner_b");

        // Both should be close to 1000
        assert!((cal_a - 1000.0).abs() < 100.0);
        assert!((cal_b - 1000.0).abs() < 100.0);
    }

    #[test]
    fn test_ml_cost_model_r_squared() {
        let mut history = ExecutionHistory::new();

        for i in 1..=10 {
            let actual = (i * 1000) as f64;
            history.record(ExecutionRecord {
                id: format!("test_{}", i),
                predicted_flops: actual * 1.1,
                actual_flops: actual,
                predicted_time_ms: actual / 100.0,
                actual_time_ms: actual / 100.0,
                predicted_memory: 1000,
                actual_memory: 1000,
                timestamp: SystemTime::now(),
                planner: "test".to_string(),
            });
        }

        let mut model = MLCostModel::new();
        model.train(&history);

        // R² should be high for good linear fit
        assert!(model.flops_r_squared() > 0.95);
        assert!(model.time_r_squared() > 0.95);
    }

    #[test]
    fn test_ml_cost_model_insufficient_data() {
        let mut history = ExecutionHistory::new();

        // Only 2 records (need at least 3)
        for i in 1..=2 {
            history.record(ExecutionRecord {
                id: format!("test_{}", i),
                predicted_flops: (i * 1000) as f64,
                actual_flops: (i * 1000) as f64,
                predicted_time_ms: 10.0,
                actual_time_ms: 10.0,
                predicted_memory: 1000,
                actual_memory: 1000,
                timestamp: SystemTime::now(),
                planner: "test".to_string(),
            });
        }

        let mut model = MLCostModel::new();
        model.train(&history);

        // Should remain untrained
        assert!(!model.is_trained());
    }
}
