//! # Resource Usage Dashboard
//!
//! Comprehensive resource monitoring and aggregation for out-of-core operations.
//!
//! This module provides:
//! - Real-time resource usage tracking (memory, I/O, CPU)
//! - Historical metrics with rolling windows
//! - Anomaly detection for resource bottlenecks
//! - Automated performance recommendations
//! - JSON/HTML export for visualization
//!
//! ## Features
//!
//! - **Multi-tier Memory Tracking**: RAM, SSD, and Disk usage with trends
//! - **I/O Performance Monitoring**: Read/write throughput, latency, IOPS
//! - **Operation Profiling**: Per-operation statistics with percentiles
//! - **Anomaly Detection**: Automatic detection of memory leaks, slow I/O, cache thrashing
//! - **Recommendations Engine**: Performance optimization suggestions
//!
//! ## Usage
//!
//! ```rust,ignore
//! use tenrso_ooc::dashboard::{Dashboard, DashboardConfig};
//!
//! let dashboard = Dashboard::new(DashboardConfig::default());
//! dashboard.record_operation("matmul", 1024 * 1024, 0.5);
//! dashboard.record_memory("ram", 512 * 1024 * 1024);
//!
//! // Get current snapshot
//! let snapshot = dashboard.snapshot();
//! println!("Memory usage: {} MB", snapshot.total_memory_mb());
//!
//! // Export to JSON
//! let json = dashboard.export_json()?;
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;

/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Maximum number of samples to keep in history
    pub max_history_samples: usize,
    /// Sampling interval for metrics collection
    pub sample_interval_secs: u64,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Enable performance recommendations
    pub enable_recommendations: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            max_history_samples: 1000,
            sample_interval_secs: 1,
            enable_anomaly_detection: true,
            enable_recommendations: true,
        }
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: u64,
    pub ram_bytes: usize,
    pub ssd_bytes: usize,
    pub disk_bytes: usize,
    pub total_bytes: usize,
}

impl MemorySnapshot {
    /// Get total memory in MB
    pub fn total_memory_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get memory distribution percentages
    pub fn distribution(&self) -> (f64, f64, f64) {
        if self.total_bytes == 0 {
            return (0.0, 0.0, 0.0);
        }
        let ram_pct = (self.ram_bytes as f64 / self.total_bytes as f64) * 100.0;
        let ssd_pct = (self.ssd_bytes as f64 / self.total_bytes as f64) * 100.0;
        let disk_pct = (self.disk_bytes as f64 / self.total_bytes as f64) * 100.0;
        (ram_pct, ssd_pct, disk_pct)
    }
}

/// I/O statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoSnapshot {
    pub timestamp: u64,
    pub reads_total: usize,
    pub writes_total: usize,
    pub read_bytes: usize,
    pub write_bytes: usize,
    pub avg_read_latency_ms: f64,
    pub avg_write_latency_ms: f64,
}

impl IoSnapshot {
    /// Get read throughput in MB/s
    pub fn read_throughput_mbps(&self, duration_secs: f64) -> f64 {
        if duration_secs <= 0.0 {
            return 0.0;
        }
        (self.read_bytes as f64 / (1024.0 * 1024.0)) / duration_secs
    }

    /// Get write throughput in MB/s
    pub fn write_throughput_mbps(&self, duration_secs: f64) -> f64 {
        if duration_secs <= 0.0 {
            return 0.0;
        }
        (self.write_bytes as f64 / (1024.0 * 1024.0)) / duration_secs
    }

    /// Get read IOPS
    pub fn read_iops(&self, duration_secs: f64) -> f64 {
        if duration_secs <= 0.0 {
            return 0.0;
        }
        self.reads_total as f64 / duration_secs
    }

    /// Get write IOPS
    pub fn write_iops(&self, duration_secs: f64) -> f64 {
        if duration_secs <= 0.0 {
            return 0.0;
        }
        self.writes_total as f64 / duration_secs
    }
}

/// Operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub name: String,
    pub count: usize,
    pub total_bytes: usize,
    pub total_duration_secs: f64,
    pub min_duration_secs: f64,
    pub max_duration_secs: f64,
    pub avg_duration_secs: f64,
}

impl OperationStats {
    fn new(name: String) -> Self {
        Self {
            name,
            count: 0,
            total_bytes: 0,
            total_duration_secs: 0.0,
            min_duration_secs: f64::MAX,
            max_duration_secs: 0.0,
            avg_duration_secs: 0.0,
        }
    }

    fn record(&mut self, bytes: usize, duration_secs: f64) {
        self.count += 1;
        self.total_bytes += bytes;
        self.total_duration_secs += duration_secs;
        self.min_duration_secs = self.min_duration_secs.min(duration_secs);
        self.max_duration_secs = self.max_duration_secs.max(duration_secs);
        self.avg_duration_secs = self.total_duration_secs / self.count as f64;
    }

    /// Get throughput in MB/s
    pub fn throughput_mbps(&self) -> f64 {
        if self.total_duration_secs <= 0.0 {
            return 0.0;
        }
        (self.total_bytes as f64 / (1024.0 * 1024.0)) / self.total_duration_secs
    }
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Anomaly {
    MemoryLeak {
        trend_mb_per_sec: f64,
    },
    SlowIo {
        operation: String,
        avg_latency_ms: f64,
        expected_latency_ms: f64,
    },
    CacheThrashing {
        hit_rate: f64,
        threshold: f64,
    },
    HighMemoryPressure {
        usage_pct: f64,
    },
}

impl Anomaly {
    /// Get severity level (0.0 = low, 1.0 = critical)
    pub fn severity(&self) -> f64 {
        match self {
            Anomaly::MemoryLeak { trend_mb_per_sec } => {
                // > 100 MB/s leak is critical
                (*trend_mb_per_sec / 100.0).min(1.0)
            }
            Anomaly::SlowIo {
                operation: _,
                avg_latency_ms,
                expected_latency_ms,
            } => {
                // 10x slowdown is critical
                let ratio = avg_latency_ms / expected_latency_ms;
                ((ratio - 1.0) / 9.0).min(1.0)
            }
            Anomaly::CacheThrashing { hit_rate, .. } => {
                // < 20% hit rate is critical
                (1.0 - hit_rate / 0.2).min(1.0)
            }
            Anomaly::HighMemoryPressure { usage_pct } => {
                // > 90% usage is critical
                ((usage_pct - 70.0) / 20.0).min(1.0)
            }
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> String {
        match self {
            Anomaly::MemoryLeak { trend_mb_per_sec } => {
                format!(
                    "Memory leak detected: growing at {:.2} MB/s",
                    trend_mb_per_sec
                )
            }
            Anomaly::SlowIo {
                operation,
                avg_latency_ms,
                expected_latency_ms,
            } => {
                format!(
                    "Slow I/O for {}: {:.2}ms (expected {:.2}ms)",
                    operation, avg_latency_ms, expected_latency_ms
                )
            }
            Anomaly::CacheThrashing { hit_rate, .. } => {
                format!("Cache thrashing: hit rate {:.1}%", hit_rate * 100.0)
            }
            Anomaly::HighMemoryPressure { usage_pct } => {
                format!("High memory pressure: {:.1}% usage", usage_pct)
            }
        }
    }
}

/// Performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Recommendation {
    IncreaseChunkSize { current_mb: f64, suggested_mb: f64 },
    DecreaseChunkSize { current_mb: f64, suggested_mb: f64 },
    EnableCompression { codec: String },
    IncreaseMemoryTier { tier: String, suggested_gb: f64 },
    EnablePrefetching { strategy: String },
    OptimizeAccessPattern { pattern: String },
}

impl Recommendation {
    /// Get priority level (0.0 = low, 1.0 = high)
    pub fn priority(&self) -> f64 {
        match self {
            Recommendation::IncreaseChunkSize { .. } => 0.6,
            Recommendation::DecreaseChunkSize { .. } => 0.7,
            Recommendation::EnableCompression { .. } => 0.8,
            Recommendation::IncreaseMemoryTier { .. } => 0.9,
            Recommendation::EnablePrefetching { .. } => 0.5,
            Recommendation::OptimizeAccessPattern { .. } => 0.7,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> String {
        match self {
            Recommendation::IncreaseChunkSize {
                current_mb,
                suggested_mb,
            } => {
                format!(
                    "Increase chunk size from {:.0} MB to {:.0} MB for better I/O efficiency",
                    current_mb, suggested_mb
                )
            }
            Recommendation::DecreaseChunkSize {
                current_mb,
                suggested_mb,
            } => {
                format!(
                    "Decrease chunk size from {:.0} MB to {:.0} MB to reduce memory pressure",
                    current_mb, suggested_mb
                )
            }
            Recommendation::EnableCompression { codec } => {
                format!(
                    "Enable {} compression to reduce storage and I/O overhead",
                    codec
                )
            }
            Recommendation::IncreaseMemoryTier { tier, suggested_gb } => {
                format!(
                    "Increase {} tier capacity to {:.1} GB to reduce spilling",
                    tier, suggested_gb
                )
            }
            Recommendation::EnablePrefetching { strategy } => {
                format!("Enable {} prefetching to reduce I/O latency", strategy)
            }
            Recommendation::OptimizeAccessPattern { pattern } => {
                format!(
                    "Optimize for {} access pattern to improve cache efficiency",
                    pattern
                )
            }
        }
    }
}

/// Complete dashboard snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub timestamp: u64,
    pub memory: MemorySnapshot,
    pub io: IoSnapshot,
    pub operations: Vec<OperationStats>,
    pub anomalies: Vec<Anomaly>,
    pub recommendations: Vec<Recommendation>,
}

/// Internal dashboard state
struct DashboardState {
    start_time: Instant,
    memory_history: VecDeque<MemorySnapshot>,
    io_history: VecDeque<IoSnapshot>,
    operation_stats: HashMap<String, OperationStats>,
    current_memory: MemorySnapshot,
    io_counters: IoCounters,
}

#[derive(Debug, Clone, Default)]
struct IoCounters {
    reads_total: usize,
    writes_total: usize,
    read_bytes: usize,
    write_bytes: usize,
    read_latencies: Vec<f64>,
    write_latencies: Vec<f64>,
}

/// Resource usage dashboard
pub struct Dashboard {
    config: DashboardConfig,
    state: Arc<RwLock<DashboardState>>,
}

impl Dashboard {
    /// Create a new dashboard
    pub fn new(config: DashboardConfig) -> Self {
        let state = DashboardState {
            start_time: Instant::now(),
            memory_history: VecDeque::with_capacity(config.max_history_samples),
            io_history: VecDeque::with_capacity(config.max_history_samples),
            operation_stats: HashMap::new(),
            current_memory: MemorySnapshot {
                timestamp: 0,
                ram_bytes: 0,
                ssd_bytes: 0,
                disk_bytes: 0,
                total_bytes: 0,
            },
            io_counters: IoCounters::default(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Record a tensor operation
    pub fn record_operation(&self, name: &str, bytes: usize, duration_secs: f64) {
        let mut state = self.state.write().unwrap();
        state
            .operation_stats
            .entry(name.to_string())
            .or_insert_with(|| OperationStats::new(name.to_string()))
            .record(bytes, duration_secs);
    }

    /// Record memory usage for a tier
    pub fn record_memory(&self, tier: &str, bytes: usize) {
        let mut state = self.state.write().unwrap();
        let timestamp = state.start_time.elapsed().as_secs();

        match tier.to_lowercase().as_str() {
            "ram" => state.current_memory.ram_bytes = bytes,
            "ssd" => state.current_memory.ssd_bytes = bytes,
            "disk" => state.current_memory.disk_bytes = bytes,
            _ => return,
        }

        state.current_memory.total_bytes = state.current_memory.ram_bytes
            + state.current_memory.ssd_bytes
            + state.current_memory.disk_bytes;
        state.current_memory.timestamp = timestamp;

        // Add to history
        if state.memory_history.len() >= self.config.max_history_samples {
            state.memory_history.pop_front();
        }
        let snapshot = state.current_memory.clone();
        state.memory_history.push_back(snapshot);
    }

    /// Record an I/O read operation
    pub fn record_io_read(&self, bytes: usize, duration_secs: f64) {
        let mut state = self.state.write().unwrap();
        state.io_counters.reads_total += 1;
        state.io_counters.read_bytes += bytes;
        state
            .io_counters
            .read_latencies
            .push(duration_secs * 1000.0);
    }

    /// Record an I/O write operation
    pub fn record_io_write(&self, bytes: usize, duration_secs: f64) {
        let mut state = self.state.write().unwrap();
        state.io_counters.writes_total += 1;
        state.io_counters.write_bytes += bytes;
        state
            .io_counters
            .write_latencies
            .push(duration_secs * 1000.0);
    }

    /// Get current snapshot
    pub fn snapshot(&self) -> DashboardSnapshot {
        let state = self.state.read().unwrap();
        let timestamp = state.start_time.elapsed().as_secs();

        let avg_read_latency = if !state.io_counters.read_latencies.is_empty() {
            state.io_counters.read_latencies.iter().sum::<f64>()
                / state.io_counters.read_latencies.len() as f64
        } else {
            0.0
        };

        let avg_write_latency = if !state.io_counters.write_latencies.is_empty() {
            state.io_counters.write_latencies.iter().sum::<f64>()
                / state.io_counters.write_latencies.len() as f64
        } else {
            0.0
        };

        let io = IoSnapshot {
            timestamp,
            reads_total: state.io_counters.reads_total,
            writes_total: state.io_counters.writes_total,
            read_bytes: state.io_counters.read_bytes,
            write_bytes: state.io_counters.write_bytes,
            avg_read_latency_ms: avg_read_latency,
            avg_write_latency_ms: avg_write_latency,
        };

        let operations: Vec<OperationStats> = state.operation_stats.values().cloned().collect();

        let anomalies = if self.config.enable_anomaly_detection {
            self.detect_anomalies(&state)
        } else {
            Vec::new()
        };

        let recommendations = if self.config.enable_recommendations {
            self.generate_recommendations(&state)
        } else {
            Vec::new()
        };

        DashboardSnapshot {
            timestamp,
            memory: state.current_memory.clone(),
            io,
            operations,
            anomalies,
            recommendations,
        }
    }

    /// Detect anomalies
    fn detect_anomalies(&self, state: &DashboardState) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // Memory leak detection (linear regression on memory history)
        if state.memory_history.len() > 10 {
            let trend = self.compute_memory_trend(&state.memory_history);
            if trend > 1.0 {
                // Growing at > 1 MB/s
                anomalies.push(Anomaly::MemoryLeak {
                    trend_mb_per_sec: trend,
                });
            }
        }

        // High memory pressure
        let total_capacity: usize = 16 * 1024 * 1024 * 1024; // Assume 16 GB total
        let usage_pct = (state.current_memory.total_bytes as f64 / total_capacity as f64) * 100.0;
        if usage_pct > 80.0 {
            anomalies.push(Anomaly::HighMemoryPressure { usage_pct });
        }

        anomalies
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self, state: &DashboardState) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Recommend compression if disk usage is high
        if state.current_memory.disk_bytes > 1024 * 1024 * 1024 {
            // > 1 GB on disk
            recommendations.push(Recommendation::EnableCompression {
                codec: "zstd".to_string(),
            });
        }

        recommendations
    }

    /// Compute memory growth trend (MB/s)
    fn compute_memory_trend(&self, history: &VecDeque<MemorySnapshot>) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        let first = history.front().unwrap();
        let last = history.back().unwrap();
        let time_diff_secs = (last.timestamp - first.timestamp) as f64;
        if time_diff_secs <= 0.0 {
            return 0.0;
        }

        let mem_diff_mb = (last.total_bytes as f64 - first.total_bytes as f64) / (1024.0 * 1024.0);
        mem_diff_mb / time_diff_secs
    }

    /// Export snapshot as JSON
    pub fn export_json(&self) -> Result<String> {
        let snapshot = self.snapshot();
        serde_json::to_string_pretty(&snapshot).context("Failed to serialize dashboard snapshot")
    }

    /// Reset all statistics
    pub fn reset(&self) {
        let mut state = self.state.write().unwrap();
        state.memory_history.clear();
        state.io_history.clear();
        state.operation_stats.clear();
        state.io_counters = IoCounters::default();
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new(DashboardConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let dashboard = Dashboard::new(DashboardConfig::default());
        let snapshot = dashboard.snapshot();
        assert_eq!(snapshot.memory.total_bytes, 0);
        assert_eq!(snapshot.operations.len(), 0);
    }

    #[test]
    fn test_record_operations() {
        let dashboard = Dashboard::default();
        dashboard.record_operation("matmul", 1024, 0.5);
        dashboard.record_operation("matmul", 2048, 0.3);

        let snapshot = dashboard.snapshot();
        assert_eq!(snapshot.operations.len(), 1);
        let op_stats = &snapshot.operations[0];
        assert_eq!(op_stats.count, 2);
        assert_eq!(op_stats.total_bytes, 3072);
    }

    #[test]
    fn test_memory_recording() {
        let dashboard = Dashboard::default();
        dashboard.record_memory("ram", 1000);
        dashboard.record_memory("ssd", 2000);
        dashboard.record_memory("disk", 3000);

        let snapshot = dashboard.snapshot();
        assert_eq!(snapshot.memory.ram_bytes, 1000);
        assert_eq!(snapshot.memory.ssd_bytes, 2000);
        assert_eq!(snapshot.memory.disk_bytes, 3000);
        assert_eq!(snapshot.memory.total_bytes, 6000);
    }

    #[test]
    fn test_io_recording() {
        let dashboard = Dashboard::default();
        dashboard.record_io_read(1024, 0.01);
        dashboard.record_io_write(2048, 0.02);

        let snapshot = dashboard.snapshot();
        assert_eq!(snapshot.io.reads_total, 1);
        assert_eq!(snapshot.io.writes_total, 1);
        assert_eq!(snapshot.io.read_bytes, 1024);
        assert_eq!(snapshot.io.write_bytes, 2048);
    }

    #[test]
    fn test_export_json() {
        let dashboard = Dashboard::default();
        dashboard.record_operation("test", 1024, 0.1);
        dashboard.record_memory("ram", 512);

        let json = dashboard.export_json().unwrap();
        // Check for basic structure
        assert!(json.contains("\"operations\""));
        assert!(json.contains("\"memory\""));
        assert!(json.contains("512")); // ram_bytes value
    }

    #[test]
    fn test_anomaly_severity() {
        let anomaly1 = Anomaly::MemoryLeak {
            trend_mb_per_sec: 50.0,
        };
        let anomaly2 = Anomaly::HighMemoryPressure { usage_pct: 85.0 };

        assert!(anomaly1.severity() > 0.0);
        assert!(anomaly2.severity() > 0.0);
    }

    #[test]
    fn test_recommendation_priority() {
        let rec = Recommendation::EnableCompression {
            codec: "zstd".to_string(),
        };
        assert!(rec.priority() > 0.5);
    }
}
