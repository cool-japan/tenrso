//! Gradient monitoring and analysis for debugging and optimizing training.
//!
//! This module provides comprehensive gradient flow tracking, health metrics,
//! and debugging capabilities for production ML training.
//!
//! # Features
//!
//! - **Gradient Flow Tracking**: Monitor gradient magnitudes across layers
//! - **Health Metrics**: Detect vanishing/exploding gradients
//! - **Historical Analysis**: Track gradient statistics over time
//! - **Layer-wise Statistics**: Detailed per-layer gradient information
//! - **Automated Recommendations**: Suggest learning rate adjustments
//!
//! # Examples
//!
//! ```rust,ignore
//! use tenrso_ad::monitoring::{GradientMonitor, MonitorConfig};
//! use scirs2_core::ndarray_ext::Array;
//!
//! // Create a gradient monitor
//! let config = MonitorConfig::default();
//! let mut monitor = GradientMonitor::new(config);
//!
//! // Record gradients for multiple layers
//! monitor.record_step("layer1", &gradient1);
//! monitor.record_step("layer2", &gradient2);
//! monitor.record_step("layer3", &gradient3);
//!
//! // Analyze gradient health
//! let health = monitor.analyze_health();
//! if health.has_vanishing_gradients() {
//!     println!("Warning: Vanishing gradients detected!");
//! }
//!
//! // Get recommendations
//! let recommendations = monitor.get_recommendations();
//! for rec in recommendations {
//!     println!("{}", rec);
//! }
//! ```

use anyhow::Result;
use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, IxDyn};
use std::collections::{HashMap, VecDeque};

/// Configuration for gradient monitoring.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Maximum history length to keep (default: 1000)
    pub max_history: usize,

    /// Threshold for detecting vanishing gradients (default: 1e-7)
    pub vanishing_threshold: f64,

    /// Threshold for detecting exploding gradients (default: 10.0)
    pub exploding_threshold: f64,

    /// Enable detailed layer-wise statistics (default: true)
    pub detailed_stats: bool,

    /// Enable gradient flow visualization data (default: true)
    pub enable_flow_tracking: bool,

    /// Window size for moving averages (default: 100)
    pub window_size: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            vanishing_threshold: 1e-7,
            exploding_threshold: 10.0,
            detailed_stats: true,
            enable_flow_tracking: true,
            window_size: 100,
        }
    }
}

impl MonitorConfig {
    /// Create configuration for aggressive vanishing gradient detection.
    pub fn aggressive_vanishing_detection() -> Self {
        Self {
            vanishing_threshold: 1e-5,
            ..Default::default()
        }
    }

    /// Create configuration for aggressive exploding gradient detection.
    pub fn aggressive_exploding_detection() -> Self {
        Self {
            exploding_threshold: 5.0,
            ..Default::default()
        }
    }

    /// Create configuration with minimal memory usage.
    pub fn minimal_memory() -> Self {
        Self {
            max_history: 100,
            detailed_stats: false,
            enable_flow_tracking: false,
            window_size: 20,
            ..Default::default()
        }
    }
}

/// Statistics for a single layer's gradients.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Layer name/identifier
    pub name: String,

    /// Mean gradient magnitude
    pub mean: f64,

    /// Standard deviation of gradients
    pub std: f64,

    /// Minimum gradient value
    pub min: f64,

    /// Maximum gradient value
    pub max: f64,

    /// L2 norm of gradients
    pub l2_norm: f64,

    /// Percentage of zero gradients
    pub zero_percentage: f64,

    /// Number of parameters
    pub num_params: usize,
}

impl LayerStats {
    /// Compute statistics from gradient array.
    pub fn from_gradient<T>(name: &str, gradient: &Array<T, IxDyn>) -> Self
    where
        T: Float,
    {
        let values: Vec<f64> = gradient
            .iter()
            .map(|&v| v.to_f64().unwrap_or(0.0))
            .collect();

        let num_params = values.len();
        let mean = values.iter().sum::<f64>() / num_params as f64;

        let variance = values
            .iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / num_params as f64;
        let std = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let l2_norm = values.iter().map(|&v| v * v).sum::<f64>().sqrt();

        let zero_count = values.iter().filter(|&&v| v.abs() < 1e-10).count();
        let zero_percentage = (zero_count as f64 / num_params as f64) * 100.0;

        Self {
            name: name.to_string(),
            mean,
            std,
            min,
            max,
            l2_norm,
            zero_percentage,
            num_params,
        }
    }

    /// Check if this layer has vanishing gradients.
    pub fn has_vanishing(&self, threshold: f64) -> bool {
        self.l2_norm < threshold
    }

    /// Check if this layer has exploding gradients.
    pub fn has_exploding(&self, threshold: f64) -> bool {
        self.l2_norm > threshold
    }
}

/// Historical gradient data for a single layer.
#[derive(Debug, Clone)]
pub struct LayerHistory {
    /// L2 norms over time
    pub l2_norms: VecDeque<f64>,

    /// Mean values over time
    pub means: VecDeque<f64>,

    /// Std deviations over time
    pub stds: VecDeque<f64>,
}

impl LayerHistory {
    fn new(max_size: usize) -> Self {
        Self {
            l2_norms: VecDeque::with_capacity(max_size),
            means: VecDeque::with_capacity(max_size),
            stds: VecDeque::with_capacity(max_size),
        }
    }

    fn add(&mut self, stats: &LayerStats, max_size: usize) {
        if self.l2_norms.len() >= max_size {
            self.l2_norms.pop_front();
            self.means.pop_front();
            self.stds.pop_front();
        }

        self.l2_norms.push_back(stats.l2_norm);
        self.means.push_back(stats.mean);
        self.stds.push_back(stats.std);
    }

    fn moving_average(&self, window: usize) -> f64 {
        let n = self.l2_norms.len().min(window);
        if n == 0 {
            return 0.0;
        }

        self.l2_norms.iter().rev().take(n).sum::<f64>() / n as f64
    }

    fn trend(&self) -> GradientTrend {
        if self.l2_norms.len() < 3 {
            return GradientTrend::Stable;
        }

        let recent: Vec<f64> = self.l2_norms.iter().rev().take(10).cloned().collect();
        let mid = recent.len() / 2;

        let first_half_avg = recent[mid..].iter().sum::<f64>() / (recent.len() - mid) as f64;
        let second_half_avg = recent[..mid].iter().sum::<f64>() / mid as f64;

        let change_ratio = (second_half_avg - first_half_avg) / first_half_avg.max(1e-10);

        if change_ratio > 0.1 {
            GradientTrend::Increasing
        } else if change_ratio < -0.1 {
            GradientTrend::Decreasing
        } else {
            GradientTrend::Stable
        }
    }
}

/// Trend in gradient magnitudes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientTrend {
    /// Gradients are increasing over time
    Increasing,
    /// Gradients are decreasing over time
    Decreasing,
    /// Gradients are relatively stable
    Stable,
}

/// Gradient health assessment.
#[derive(Debug, Clone)]
pub struct GradientHealth {
    /// Layers with vanishing gradients
    pub vanishing_layers: Vec<String>,

    /// Layers with exploding gradients
    pub exploding_layers: Vec<String>,

    /// Overall health status
    pub status: HealthStatus,

    /// Severity score (0.0 = healthy, 1.0 = critical)
    pub severity: f64,

    /// Recommendations for fixing issues
    pub recommendations: Vec<String>,
}

impl GradientHealth {
    /// Check if there are vanishing gradients.
    pub fn has_vanishing_gradients(&self) -> bool {
        !self.vanishing_layers.is_empty()
    }

    /// Check if there are exploding gradients.
    pub fn has_exploding_gradients(&self) -> bool {
        !self.exploding_layers.is_empty()
    }

    /// Check if gradients are healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, HealthStatus::Healthy)
    }
}

/// Overall gradient health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All gradients are healthy
    Healthy,
    /// Minor issues detected
    Warning,
    /// Serious issues detected
    Critical,
}

/// Gradient flow analysis across layers.
#[derive(Debug, Clone)]
pub struct FlowAnalysis {
    /// Layer-wise gradient magnitudes
    pub layer_magnitudes: Vec<(String, f64)>,

    /// Ratio of first to last layer gradient magnitude
    pub flow_ratio: f64,

    /// Indicates if flow is balanced
    pub is_balanced: bool,

    /// Bottleneck layers (if any)
    pub bottlenecks: Vec<String>,
}

impl FlowAnalysis {
    /// Check if gradient flow is healthy.
    pub fn is_healthy(&self) -> bool {
        self.is_balanced && self.bottlenecks.is_empty()
    }
}

/// Main gradient monitor for tracking and analyzing gradients.
pub struct GradientMonitor {
    config: MonitorConfig,
    layer_histories: HashMap<String, LayerHistory>,
    step_count: usize,
    current_step_stats: HashMap<String, LayerStats>,
}

impl GradientMonitor {
    /// Create a new gradient monitor with the given configuration.
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            layer_histories: HashMap::new(),
            step_count: 0,
            current_step_stats: HashMap::new(),
        }
    }

    /// Record gradients for a layer in the current step.
    ///
    /// # Arguments
    /// * `layer_name` - Name/identifier for the layer
    /// * `gradient` - Gradient tensor for the layer
    pub fn record_step<T>(&mut self, layer_name: &str, gradient: &Array<T, IxDyn>)
    where
        T: Float,
    {
        let stats = LayerStats::from_gradient(layer_name, gradient);

        // Store current step stats
        self.current_step_stats
            .insert(layer_name.to_string(), stats.clone());

        // Update history
        if self.config.enable_flow_tracking {
            let history = self
                .layer_histories
                .entry(layer_name.to_string())
                .or_insert_with(|| LayerHistory::new(self.config.max_history));

            history.add(&stats, self.config.max_history);
        }
    }

    /// Complete the current step and prepare for the next.
    pub fn step(&mut self) {
        self.step_count += 1;
        self.current_step_stats.clear();
    }

    /// Analyze gradient health across all layers.
    pub fn analyze_health(&self) -> GradientHealth {
        let mut vanishing_layers = Vec::new();
        let mut exploding_layers = Vec::new();

        for (name, stats) in &self.current_step_stats {
            if stats.has_vanishing(self.config.vanishing_threshold) {
                vanishing_layers.push(name.clone());
            }
            if stats.has_exploding(self.config.exploding_threshold) {
                exploding_layers.push(name.clone());
            }
        }

        let severity = if !vanishing_layers.is_empty() || !exploding_layers.is_empty() {
            let total_layers = self.current_step_stats.len();
            let problem_layers = vanishing_layers.len() + exploding_layers.len();
            problem_layers as f64 / total_layers.max(1) as f64
        } else {
            0.0
        };

        let status = if severity > 0.5 {
            HealthStatus::Critical
        } else if severity > 0.2 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        let mut recommendations = Vec::new();

        if !vanishing_layers.is_empty() {
            recommendations.push(format!(
                "Vanishing gradients detected in {} layers. Consider: \
                 (1) Increase learning rate, (2) Use skip connections, \
                 (3) Change activation functions, (4) Reduce network depth",
                vanishing_layers.len()
            ));
        }

        if !exploding_layers.is_empty() {
            recommendations.push(format!(
                "Exploding gradients detected in {} layers. Consider: \
                 (1) Reduce learning rate, (2) Apply gradient clipping, \
                 (3) Check weight initialization, (4) Use batch normalization",
                exploding_layers.len()
            ));
        }

        GradientHealth {
            vanishing_layers,
            exploding_layers,
            status,
            severity,
            recommendations,
        }
    }

    /// Analyze gradient flow across layers.
    pub fn analyze_flow(&self) -> Result<FlowAnalysis> {
        let mut layer_magnitudes: Vec<(String, f64)> = self
            .current_step_stats
            .iter()
            .map(|(name, stats)| (name.clone(), stats.l2_norm))
            .collect();

        layer_magnitudes.sort_by(|a, b| a.0.cmp(&b.0));

        let flow_ratio = if layer_magnitudes.len() >= 2 {
            let first = layer_magnitudes.first().unwrap().1;
            let last = layer_magnitudes.last().unwrap().1;
            if last > 1e-10 {
                first / last
            } else {
                0.0
            }
        } else {
            1.0
        };

        let is_balanced = flow_ratio > 0.1 && flow_ratio < 10.0;

        // Detect bottlenecks (layers with significantly lower gradients)
        let avg_magnitude =
            layer_magnitudes.iter().map(|(_, m)| m).sum::<f64>() / layer_magnitudes.len() as f64;

        let bottlenecks: Vec<String> = layer_magnitudes
            .iter()
            .filter(|(_, m)| *m < avg_magnitude * 0.1)
            .map(|(name, _)| name.clone())
            .collect();

        Ok(FlowAnalysis {
            layer_magnitudes,
            flow_ratio,
            is_balanced,
            bottlenecks,
        })
    }

    /// Get current statistics for all layers.
    pub fn get_current_stats(&self) -> &HashMap<String, LayerStats> {
        &self.current_step_stats
    }

    /// Get statistics for a specific layer.
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerStats> {
        self.current_step_stats.get(layer_name)
    }

    /// Get historical data for a layer.
    pub fn get_layer_history(&self, layer_name: &str) -> Option<&LayerHistory> {
        self.layer_histories.get(layer_name)
    }

    /// Get gradient trend for a layer.
    pub fn get_layer_trend(&self, layer_name: &str) -> Option<GradientTrend> {
        self.layer_histories.get(layer_name).map(|h| h.trend())
    }

    /// Get moving average of gradient norm for a layer.
    pub fn get_moving_average(&self, layer_name: &str) -> Option<f64> {
        self.layer_histories
            .get(layer_name)
            .map(|h| h.moving_average(self.config.window_size))
    }

    /// Get automated recommendations for training adjustments.
    pub fn get_recommendations(&self) -> Vec<String> {
        let health = self.analyze_health();
        let mut recommendations = health.recommendations;

        // Add trend-based recommendations
        for (name, history) in &self.layer_histories {
            match history.trend() {
                GradientTrend::Decreasing => {
                    recommendations.push(format!(
                        "Layer '{}' shows decreasing gradient trend. \
                         Consider checking if this layer is learning effectively.",
                        name
                    ));
                }
                GradientTrend::Increasing => {
                    if let Some(stats) = self.current_step_stats.get(name) {
                        if stats.l2_norm > self.config.exploding_threshold * 0.5 {
                            recommendations.push(format!(
                                "Layer '{}' shows increasing gradient trend approaching explosion. \
                                 Consider gradient clipping or reducing learning rate.",
                                name
                            ));
                        }
                    }
                }
                GradientTrend::Stable => {}
            }
        }

        // Flow-based recommendations
        if let Ok(flow) = self.analyze_flow() {
            if !flow.is_balanced {
                recommendations.push(
                    "Gradient flow is imbalanced. Consider using residual connections \
                     or normalizing layers."
                        .to_string(),
                );
            }

            if !flow.bottlenecks.is_empty() {
                recommendations.push(format!(
                    "Bottleneck layers detected: {}. These layers may need attention.",
                    flow.bottlenecks.join(", ")
                ));
            }
        }

        recommendations
    }

    /// Get total number of steps recorded.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Clear all historical data.
    pub fn clear_history(&mut self) {
        self.layer_histories.clear();
        self.current_step_stats.clear();
        self.step_count = 0;
    }

    /// Export statistics summary as a formatted string.
    pub fn summary(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "=== Gradient Monitor Summary (Step {}) ===\n\n",
            self.step_count
        ));

        // Current stats
        output.push_str("Current Layer Statistics:\n");
        let mut layers: Vec<_> = self.current_step_stats.keys().collect();
        layers.sort();

        for layer_name in layers {
            if let Some(stats) = self.current_step_stats.get(layer_name) {
                output.push_str(&format!(
                    "  {}: L2={:.6}, mean={:.6}, std={:.6}, zeros={:.1}%\n",
                    layer_name, stats.l2_norm, stats.mean, stats.std, stats.zero_percentage
                ));
            }
        }

        // Health analysis
        output.push_str("\nGradient Health:\n");
        let health = self.analyze_health();
        output.push_str(&format!("  Status: {:?}\n", health.status));
        output.push_str(&format!("  Severity: {:.2}\n", health.severity));

        if !health.vanishing_layers.is_empty() {
            output.push_str(&format!(
                "  Vanishing: {}\n",
                health.vanishing_layers.join(", ")
            ));
        }

        if !health.exploding_layers.is_empty() {
            output.push_str(&format!(
                "  Exploding: {}\n",
                health.exploding_layers.join(", ")
            ));
        }

        // Recommendations
        let recommendations = self.get_recommendations();
        if !recommendations.is_empty() {
            output.push_str("\nRecommendations:\n");
            for (i, rec) in recommendations.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, rec));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_config_default() {
        let config = MonitorConfig::default();
        assert_eq!(config.max_history, 1000);
        assert_eq!(config.vanishing_threshold, 1e-7);
        assert_eq!(config.exploding_threshold, 10.0);
        assert!(config.detailed_stats);
        assert!(config.enable_flow_tracking);
    }

    #[test]
    fn test_layer_stats_computation() {
        let gradient = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).into_dyn();
        let stats = LayerStats::from_gradient("test_layer", &gradient);

        assert_eq!(stats.name, "test_layer");
        assert_eq!(stats.num_params, 5);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!(stats.l2_norm > 0.0);
    }

    #[test]
    fn test_layer_stats_vanishing_detection() {
        let gradient = Array::from_vec(vec![1e-10f32, 1e-10, 1e-10]).into_dyn();
        let stats = LayerStats::from_gradient("vanishing", &gradient);

        assert!(stats.has_vanishing(1e-7));
        assert!(!stats.has_exploding(10.0));
    }

    #[test]
    fn test_layer_stats_exploding_detection() {
        let gradient = Array::from_vec(vec![100.0f32, 200.0, 300.0]).into_dyn();
        let stats = LayerStats::from_gradient("exploding", &gradient);

        assert!(!stats.has_vanishing(1e-7));
        assert!(stats.has_exploding(10.0));
    }

    #[test]
    fn test_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = GradientMonitor::new(config);

        assert_eq!(monitor.step_count(), 0);
        assert!(monitor.get_current_stats().is_empty());
    }

    #[test]
    fn test_monitor_record_step() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad1 = Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        let grad2 = Array::from_vec(vec![4.0f32, 5.0, 6.0]).into_dyn();

        monitor.record_step("layer1", &grad1);
        monitor.record_step("layer2", &grad2);

        assert_eq!(monitor.get_current_stats().len(), 2);
        assert!(monitor.get_layer_stats("layer1").is_some());
        assert!(monitor.get_layer_stats("layer2").is_some());
    }

    #[test]
    fn test_monitor_step_progression() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad = Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();

        monitor.record_step("layer1", &grad);
        assert_eq!(monitor.step_count(), 0);

        monitor.step();
        assert_eq!(monitor.step_count(), 1);
        assert!(monitor.get_current_stats().is_empty());
    }

    #[test]
    fn test_health_analysis_healthy() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad = Array::from_vec(vec![0.1f32, 0.2, 0.3]).into_dyn();
        monitor.record_step("layer1", &grad);

        let health = monitor.analyze_health();
        assert!(health.is_healthy());
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.severity, 0.0);
    }

    #[test]
    fn test_health_analysis_vanishing() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad = Array::from_vec(vec![1e-10f32, 1e-10, 1e-10]).into_dyn();
        monitor.record_step("layer1", &grad);

        let health = monitor.analyze_health();
        assert!(health.has_vanishing_gradients());
        assert!(!health.is_healthy());
        assert!(health.severity > 0.0);
    }

    #[test]
    fn test_health_analysis_exploding() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad = Array::from_vec(vec![100.0f32, 200.0, 300.0]).into_dyn();
        monitor.record_step("layer1", &grad);

        let health = monitor.analyze_health();
        assert!(health.has_exploding_gradients());
        assert!(!health.is_healthy());
    }

    #[test]
    fn test_flow_analysis() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad1 = Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        let grad2 = Array::from_vec(vec![1.5f32, 2.5, 3.5]).into_dyn();

        monitor.record_step("layer1", &grad1);
        monitor.record_step("layer2", &grad2);

        let flow = monitor.analyze_flow().unwrap();
        assert_eq!(flow.layer_magnitudes.len(), 2);
        assert!(flow.flow_ratio > 0.0);
    }

    #[test]
    fn test_recommendations() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        // Add vanishing gradient
        let grad = Array::from_vec(vec![1e-10f32, 1e-10]).into_dyn();
        monitor.record_step("layer1", &grad);

        let recommendations = monitor.get_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_moving_average() {
        let config = MonitorConfig {
            window_size: 3,
            ..Default::default()
        };
        let mut monitor = GradientMonitor::new(config);

        // Add multiple steps
        for i in 0..5 {
            let val = (i + 1) as f32;
            let grad = Array::from_vec(vec![val, val, val]).into_dyn();
            monitor.record_step("layer1", &grad);
            monitor.step();
        }

        // Should average last 3 values
        let avg = monitor.get_moving_average("layer1");
        assert!(avg.is_some());
    }

    #[test]
    fn test_clear_history() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad = Array::from_vec(vec![1.0f32, 2.0]).into_dyn();
        monitor.record_step("layer1", &grad);
        monitor.step();

        assert_eq!(monitor.step_count(), 1);

        monitor.clear_history();
        assert_eq!(monitor.step_count(), 0);
        assert!(monitor.get_current_stats().is_empty());
    }

    #[test]
    fn test_summary_output() {
        let config = MonitorConfig::default();
        let mut monitor = GradientMonitor::new(config);

        let grad = Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        monitor.record_step("layer1", &grad);

        let summary = monitor.summary();
        assert!(summary.contains("Gradient Monitor Summary"));
        assert!(summary.contains("layer1"));
    }
}
