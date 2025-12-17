//! Profiling and instrumentation for plan operations
//!
//! This module provides hooks and utilities for monitoring planning operations
//! in production environments. Use these tools to:
//! - Track planning time and performance
//! - Monitor algorithm selection decisions
//! - Collect metrics for optimization
//! - Debug planning issues in production
//!
//! # Examples
//!
//! ```
//! use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints, PlanProfiler};
//!
//! let mut profiler = PlanProfiler::new();
//!
//! let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
//! let shapes = vec![vec![100, 200], vec![200, 300]];
//! let hints = PlanHints::default();
//!
//! // Profile a planning operation
//! let result = profiler.profile("matmul_100x200x300", || {
//!     greedy_planner(&spec, &shapes, &hints)
//! });
//!
//! // Check metrics
//! let metrics = profiler.metrics();
//! println!("Total plans: {}", metrics.total_plans);
//! println!("Average time: {:.2}ms", metrics.avg_planning_time_ms);
//! ```

use crate::api::Plan;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A single profiling event for a planning operation
#[derive(Debug, Clone)]
pub struct PlanEvent {
    /// Unique identifier for this event
    pub id: String,
    /// Algorithm or operation name
    pub operation: String,
    /// Planning time in milliseconds
    pub duration_ms: f64,
    /// Number of input tensors
    pub num_inputs: usize,
    /// Estimated FLOPs of the result
    pub result_flops: Option<f64>,
    /// Estimated memory of the result
    pub result_memory: Option<usize>,
    /// Timestamp when the event occurred
    pub timestamp: Instant,
    /// Whether the operation succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Aggregated metrics for profiling data
#[derive(Debug, Clone, Default)]
pub struct ProfilingMetrics {
    /// Total number of plans created
    pub total_plans: usize,
    /// Number of successful plans
    pub successful_plans: usize,
    /// Number of failed plans
    pub failed_plans: usize,
    /// Total planning time (ms)
    pub total_time_ms: f64,
    /// Average planning time (ms)
    pub avg_planning_time_ms: f64,
    /// Minimum planning time (ms)
    pub min_time_ms: f64,
    /// Maximum planning time (ms)
    pub max_time_ms: f64,
    /// Median planning time (ms)
    pub median_time_ms: f64,
    /// 95th percentile planning time (ms)
    pub p95_time_ms: f64,
    /// 99th percentile planning time (ms)
    pub p99_time_ms: f64,
    /// Breakdown by operation type
    pub by_operation: HashMap<String, OperationMetrics>,
}

/// Metrics for a specific operation type
#[derive(Debug, Clone, Default)]
pub struct OperationMetrics {
    /// Number of invocations
    pub count: usize,
    /// Total time (ms)
    pub total_time_ms: f64,
    /// Average time (ms)
    pub avg_time_ms: f64,
    /// Minimum time (ms)
    pub min_time_ms: f64,
    /// Maximum time (ms)
    pub max_time_ms: f64,
}

/// A profiler for tracking planning operations
///
/// The profiler collects timing and performance data for all planning operations.
/// It is thread-safe and can be shared across multiple threads.
///
/// # Examples
///
/// ```
/// use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints, PlanProfiler};
///
/// let mut profiler = PlanProfiler::new();
///
/// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
/// let shapes = vec![vec![10, 20], vec![20, 30]];
/// let hints = PlanHints::default();
///
/// // Profile multiple operations
/// for i in 0..10 {
///     profiler.profile(&format!("plan_{}", i), || {
///         greedy_planner(&spec, &shapes, &hints)
///     }).unwrap();
/// }
///
/// // Get aggregated metrics
/// let metrics = profiler.metrics();
/// assert_eq!(metrics.total_plans, 10);
/// ```
#[derive(Clone)]
pub struct PlanProfiler {
    events: Arc<Mutex<Vec<PlanEvent>>>,
    enabled: Arc<Mutex<bool>>,
}

impl PlanProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            enabled: Arc::new(Mutex::new(true)),
        }
    }

    /// Enable profiling
    pub fn enable(&self) {
        let mut enabled = self.enabled.lock().unwrap();
        *enabled = true;
    }

    /// Disable profiling (no-op for profile calls)
    pub fn disable(&self) {
        let mut enabled = self.enabled.lock().unwrap();
        *enabled = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        let enabled = self.enabled.lock().unwrap();
        *enabled
    }

    /// Profile a planning operation
    ///
    /// This method times the execution of the provided closure and records
    /// profiling data if successful.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this operation
    /// * `f` - The planning function to profile
    ///
    /// # Returns
    ///
    /// The result of the planning operation
    pub fn profile<F>(&self, id: &str, f: F) -> Result<Plan>
    where
        F: FnOnce() -> Result<Plan>,
    {
        if !self.is_enabled() {
            return f();
        }

        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        let event = match &result {
            Ok(plan) => PlanEvent {
                id: id.to_string(),
                operation: "plan".to_string(),
                duration_ms: duration.as_secs_f64() * 1000.0,
                num_inputs: plan.nodes.len() + 1, // Approximate
                result_flops: Some(plan.estimated_flops),
                result_memory: Some(plan.estimated_memory),
                timestamp: start,
                success: true,
                error: None,
            },
            Err(e) => PlanEvent {
                id: id.to_string(),
                operation: "plan".to_string(),
                duration_ms: duration.as_secs_f64() * 1000.0,
                num_inputs: 0,
                result_flops: None,
                result_memory: None,
                timestamp: start,
                success: false,
                error: Some(e.to_string()),
            },
        };

        let mut events = self.events.lock().unwrap();
        events.push(event);

        result
    }

    /// Profile a named operation with explicit metadata
    pub fn profile_with_metadata<F>(
        &self,
        id: &str,
        operation: &str,
        num_inputs: usize,
        f: F,
    ) -> Result<Plan>
    where
        F: FnOnce() -> Result<Plan>,
    {
        if !self.is_enabled() {
            return f();
        }

        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        let event = match &result {
            Ok(plan) => PlanEvent {
                id: id.to_string(),
                operation: operation.to_string(),
                duration_ms: duration.as_secs_f64() * 1000.0,
                num_inputs,
                result_flops: Some(plan.estimated_flops),
                result_memory: Some(plan.estimated_memory),
                timestamp: start,
                success: true,
                error: None,
            },
            Err(e) => PlanEvent {
                id: id.to_string(),
                operation: operation.to_string(),
                duration_ms: duration.as_secs_f64() * 1000.0,
                num_inputs,
                result_flops: None,
                result_memory: None,
                timestamp: start,
                success: false,
                error: Some(e.to_string()),
            },
        };

        let mut events = self.events.lock().unwrap();
        events.push(event);

        result
    }

    /// Record a timing event without executing a function
    pub fn record_event(&self, event: PlanEvent) {
        if !self.is_enabled() {
            return;
        }

        let mut events = self.events.lock().unwrap();
        events.push(event);
    }

    /// Get all recorded events
    pub fn events(&self) -> Vec<PlanEvent> {
        let events = self.events.lock().unwrap();
        events.clone()
    }

    /// Get aggregated profiling metrics
    pub fn metrics(&self) -> ProfilingMetrics {
        let events = self.events.lock().unwrap();

        if events.is_empty() {
            return ProfilingMetrics::default();
        }

        let total_plans = events.len();
        let successful_plans = events.iter().filter(|e| e.success).count();
        let failed_plans = total_plans - successful_plans;

        let mut times: Vec<f64> = events.iter().map(|e| e.duration_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let total_time_ms: f64 = times.iter().sum();
        let avg_planning_time_ms = total_time_ms / total_plans as f64;
        let min_time_ms = times.first().copied().unwrap_or(0.0);
        let max_time_ms = times.last().copied().unwrap_or(0.0);

        // Calculate percentiles
        let median_time_ms = percentile(&times, 0.5);
        let p95_time_ms = percentile(&times, 0.95);
        let p99_time_ms = percentile(&times, 0.99);

        // Group by operation
        let mut by_operation: HashMap<String, Vec<f64>> = HashMap::new();
        for event in events.iter() {
            by_operation
                .entry(event.operation.clone())
                .or_default()
                .push(event.duration_ms);
        }

        let by_operation = by_operation
            .into_iter()
            .map(|(op, times)| {
                let count = times.len();
                let total = times.iter().sum::<f64>();
                let avg = total / count as f64;
                let min = times.iter().copied().fold(f64::INFINITY, f64::min);
                let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                (
                    op,
                    OperationMetrics {
                        count,
                        total_time_ms: total,
                        avg_time_ms: avg,
                        min_time_ms: min,
                        max_time_ms: max,
                    },
                )
            })
            .collect();

        ProfilingMetrics {
            total_plans,
            successful_plans,
            failed_plans,
            total_time_ms,
            avg_planning_time_ms,
            min_time_ms,
            max_time_ms,
            median_time_ms,
            p95_time_ms,
            p99_time_ms,
            by_operation,
        }
    }

    /// Clear all recorded events
    pub fn clear(&self) {
        let mut events = self.events.lock().unwrap();
        events.clear();
    }

    /// Get the number of recorded events
    pub fn len(&self) -> usize {
        let events = self.events.lock().unwrap();
        events.len()
    }

    /// Check if no events have been recorded
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Export events to JSON string (requires serde feature)
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        use serde::Serialize;

        #[derive(Serialize)]
        struct EventExport {
            id: String,
            operation: String,
            duration_ms: f64,
            num_inputs: usize,
            result_flops: Option<f64>,
            result_memory: Option<usize>,
            success: bool,
            error: Option<String>,
        }

        let events = self.events.lock().unwrap();
        let exports: Vec<EventExport> = events
            .iter()
            .map(|e| EventExport {
                id: e.id.clone(),
                operation: e.operation.clone(),
                duration_ms: e.duration_ms,
                num_inputs: e.num_inputs,
                result_flops: e.result_flops,
                result_memory: e.result_memory,
                success: e.success,
                error: e.error.clone(),
            })
            .collect();

        Ok(serde_json::to_string_pretty(&exports)?)
    }
}

impl Default for PlanProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let idx = (sorted_data.len() as f64 * p) as usize;
    let idx = idx.min(sorted_data.len() - 1);
    sorted_data[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{greedy_planner, EinsumSpec, PlanHints};

    #[test]
    fn test_profiler_basic() {
        let profiler = PlanProfiler::new();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        // Profile a planning operation
        let result = profiler.profile("test_plan", || greedy_planner(&spec, &shapes, &hints));

        assert!(result.is_ok());
        assert_eq!(profiler.len(), 1);

        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "test_plan");
        assert!(events[0].success);
        assert!(events[0].duration_ms > 0.0);
    }

    #[test]
    fn test_profiler_multiple_operations() {
        let profiler = PlanProfiler::new();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        // Profile 10 operations
        for i in 0..10 {
            profiler
                .profile(&format!("plan_{}", i), || {
                    greedy_planner(&spec, &shapes, &hints)
                })
                .unwrap();
        }

        assert_eq!(profiler.len(), 10);

        let metrics = profiler.metrics();
        assert_eq!(metrics.total_plans, 10);
        assert_eq!(metrics.successful_plans, 10);
        assert_eq!(metrics.failed_plans, 0);
        assert!(metrics.avg_planning_time_ms > 0.0);
    }

    #[test]
    fn test_profiler_disable() {
        let profiler = PlanProfiler::new();
        profiler.disable();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        // Should not record when disabled
        profiler
            .profile("test_plan", || greedy_planner(&spec, &shapes, &hints))
            .unwrap();

        assert_eq!(profiler.len(), 0);
    }

    #[test]
    fn test_profiler_clear() {
        let profiler = PlanProfiler::new();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        profiler
            .profile("test_plan", || greedy_planner(&spec, &shapes, &hints))
            .unwrap();

        assert_eq!(profiler.len(), 1);

        profiler.clear();
        assert_eq!(profiler.len(), 0);
        assert!(profiler.is_empty());
    }

    #[test]
    fn test_profiler_metrics() {
        let profiler = PlanProfiler::new();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        // Profile multiple operations
        for i in 0..5 {
            profiler
                .profile_with_metadata(&format!("plan_{}", i), "greedy", 2, || {
                    greedy_planner(&spec, &shapes, &hints)
                })
                .unwrap();
        }

        let metrics = profiler.metrics();
        assert_eq!(metrics.total_plans, 5);
        assert!(metrics.avg_planning_time_ms > 0.0);
        assert!(metrics.min_time_ms > 0.0);
        assert!(metrics.max_time_ms >= metrics.min_time_ms);
        assert!(metrics.by_operation.contains_key("greedy"));
    }

    #[test]
    fn test_profiler_percentiles() {
        let profiler = PlanProfiler::new();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        // Profile 100 operations to get meaningful percentiles
        for i in 0..100 {
            profiler
                .profile(&format!("plan_{}", i), || {
                    greedy_planner(&spec, &shapes, &hints)
                })
                .unwrap();
        }

        let metrics = profiler.metrics();
        assert!(metrics.median_time_ms > 0.0);
        assert!(metrics.p95_time_ms >= metrics.median_time_ms);
        assert!(metrics.p99_time_ms >= metrics.p95_time_ms);
    }

    #[test]
    fn test_profiler_thread_safety() {
        use std::thread;

        let profiler = PlanProfiler::new();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let profiler = profiler.clone();
                thread::spawn(move || {
                    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
                    let shapes = vec![vec![10, 20], vec![20, 30]];
                    let hints = PlanHints::default();

                    for j in 0..10 {
                        profiler
                            .profile(&format!("thread_{}_plan_{}", i, j), || {
                                greedy_planner(&spec, &shapes, &hints)
                            })
                            .unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(profiler.len(), 100);
    }
}
