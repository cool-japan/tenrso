//! Performance profiling and monitoring for out-of-core operations
//!
//! This module provides tools for monitoring and profiling out-of-core tensor operations,
//! including I/O timing, memory usage tracking, and operation statistics.
//!
//! # Features
//!
//! - Operation timing and profiling
//! - I/O bandwidth monitoring
//! - Memory usage tracking
//! - Cache hit/miss statistics
//! - Performance metrics aggregation
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::profiling::{Profiler, ProfileScope};
//!
//! let mut profiler = Profiler::new();
//!
//! // Profile an operation
//! {
//!     let _scope = profiler.begin_scope("matmul");
//!     // ... perform operation
//! }
//!
//! // Get statistics
//! let stats = profiler.get_stats();
//! println!("Total time: {} ms", stats.total_time_ms);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance statistics for a profiled operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Number of times the operation was called
    pub call_count: usize,
    /// Total time spent in the operation
    pub total_time: Duration,
    /// Minimum time for a single call
    pub min_time: Duration,
    /// Maximum time for a single call
    pub max_time: Duration,
    /// Average time per call
    pub avg_time: Duration,
    /// Total bytes read (for I/O operations)
    pub bytes_read: u64,
    /// Total bytes written (for I/O operations)
    pub bytes_written: u64,
}

impl OperationStats {
    fn new() -> Self {
        Self {
            call_count: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            bytes_read: 0,
            bytes_written: 0,
        }
    }

    fn update(&mut self, duration: Duration) {
        self.call_count += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        self.avg_time = self.total_time / self.call_count as u32;
    }

    fn add_io(&mut self, bytes_read: u64, bytes_written: u64) {
        self.bytes_read += bytes_read;
        self.bytes_written += bytes_written;
    }
}

/// RAII scope guard for automatic profiling
pub struct ProfileScope<'a> {
    profiler: &'a mut Profiler,
    operation: String,
    start: Instant,
    bytes_read: u64,
    bytes_written: u64,
}

impl<'a> ProfileScope<'a> {
    fn new(profiler: &'a mut Profiler, operation: String) -> Self {
        Self {
            profiler,
            operation,
            start: Instant::now(),
            bytes_read: 0,
            bytes_written: 0,
        }
    }

    /// Record bytes read in this scope
    pub fn add_bytes_read(&mut self, bytes: u64) {
        self.bytes_read += bytes;
    }

    /// Record bytes written in this scope
    pub fn add_bytes_written(&mut self, bytes: u64) {
        self.bytes_written += bytes;
    }
}

impl<'a> Drop for ProfileScope<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.record_operation(
            &self.operation,
            duration,
            self.bytes_read,
            self.bytes_written,
        );
    }
}

/// Performance profiler for out-of-core operations
pub struct Profiler {
    /// Statistics per operation
    stats: HashMap<String, OperationStats>,
    /// Whether profiling is enabled
    enabled: bool,
    /// Global start time
    start_time: Instant,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            enabled: true,
            start_time: Instant::now(),
        }
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Begin a profiled scope
    ///
    /// Returns a RAII guard that automatically records timing when dropped.
    pub fn begin_scope(&mut self, operation: &str) -> ProfileScope<'_> {
        ProfileScope::new(self, operation.to_string())
    }

    /// Manually record an operation
    pub fn record_operation(
        &mut self,
        operation: &str,
        duration: Duration,
        bytes_read: u64,
        bytes_written: u64,
    ) {
        if !self.enabled {
            return;
        }

        let stats = self
            .stats
            .entry(operation.to_string())
            .or_insert_with(OperationStats::new);

        stats.update(duration);
        stats.add_io(bytes_read, bytes_written);
    }

    /// Get statistics for a specific operation
    pub fn get_operation_stats(&self, operation: &str) -> Option<&OperationStats> {
        self.stats.get(operation)
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> &HashMap<String, OperationStats> {
        &self.stats
    }

    /// Get total time since profiler creation
    pub fn total_elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get total I/O bandwidth (bytes/sec)
    pub fn io_bandwidth(&self) -> f64 {
        let total_bytes: u64 = self
            .stats
            .values()
            .map(|s| s.bytes_read + s.bytes_written)
            .sum();

        let elapsed_secs = self.total_elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            total_bytes as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    /// Get operation with most time
    pub fn slowest_operation(&self) -> Option<(&str, &OperationStats)> {
        self.stats
            .iter()
            .max_by_key(|(_, stats)| stats.total_time)
            .map(|(name, stats)| (name.as_str(), stats))
    }

    /// Get operation with most calls
    pub fn most_called_operation(&self) -> Option<(&str, &OperationStats)> {
        self.stats
            .iter()
            .max_by_key(|(_, stats)| stats.call_count)
            .map(|(name, stats)| (name.as_str(), stats))
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.stats.clear();
        self.start_time = Instant::now();
    }

    /// Generate a summary report
    pub fn summary(&self) -> ProfileSummary {
        let total_operations: usize = self.stats.values().map(|s| s.call_count).sum();
        let total_time: Duration = self.stats.values().map(|s| s.total_time).sum();
        let total_io: u64 = self
            .stats
            .values()
            .map(|s| s.bytes_read + s.bytes_written)
            .sum();

        ProfileSummary {
            total_operations,
            total_time,
            total_io_bytes: total_io,
            io_bandwidth_bps: self.io_bandwidth(),
            num_operation_types: self.stats.len(),
            elapsed_time: self.total_elapsed(),
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of all profiling data
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    /// Total number of operations
    pub total_operations: usize,
    /// Total time across all operations
    pub total_time: Duration,
    /// Total bytes transferred
    pub total_io_bytes: u64,
    /// Average I/O bandwidth (bytes/sec)
    pub io_bandwidth_bps: f64,
    /// Number of different operation types
    pub num_operation_types: usize,
    /// Total elapsed time since profiler creation
    pub elapsed_time: Duration,
}

impl ProfileSummary {
    /// Get I/O bandwidth in MB/s
    pub fn io_bandwidth_mbps(&self) -> f64 {
        self.io_bandwidth_bps / (1024.0 * 1024.0)
    }

    /// Get average operations per second
    pub fn ops_per_second(&self) -> f64 {
        let elapsed_secs = self.elapsed_time.as_secs_f64();
        if elapsed_secs > 0.0 {
            self.total_operations as f64 / elapsed_secs
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(profiler.is_enabled());
        assert_eq!(profiler.get_all_stats().len(), 0);
    }

    #[test]
    fn test_scope_profiling() {
        let mut profiler = Profiler::new();

        {
            let _scope = profiler.begin_scope("test_op");
            thread::sleep(Duration::from_millis(10));
        }

        let stats = profiler.get_operation_stats("test_op").unwrap();
        assert_eq!(stats.call_count, 1);
        assert!(stats.total_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_multiple_operations() {
        let mut profiler = Profiler::new();

        for _ in 0..5 {
            let _scope = profiler.begin_scope("op1");
            thread::sleep(Duration::from_millis(1));
        }

        for _ in 0..3 {
            let _scope = profiler.begin_scope("op2");
            thread::sleep(Duration::from_millis(1));
        }

        assert_eq!(profiler.get_operation_stats("op1").unwrap().call_count, 5);
        assert_eq!(profiler.get_operation_stats("op2").unwrap().call_count, 3);
    }

    #[test]
    fn test_io_tracking() {
        let mut profiler = Profiler::new();

        {
            let mut scope = profiler.begin_scope("io_op");
            scope.add_bytes_read(1024);
            scope.add_bytes_written(512);
        }

        let stats = profiler.get_operation_stats("io_op").unwrap();
        assert_eq!(stats.bytes_read, 1024);
        assert_eq!(stats.bytes_written, 512);
    }

    #[test]
    fn test_summary() {
        let mut profiler = Profiler::new();

        {
            let _scope = profiler.begin_scope("op1");
        }
        {
            let _scope = profiler.begin_scope("op2");
        }

        let summary = profiler.summary();
        assert_eq!(summary.total_operations, 2);
        assert_eq!(summary.num_operation_types, 2);
    }

    #[test]
    fn test_reset() {
        let mut profiler = Profiler::new();

        {
            let _scope = profiler.begin_scope("test");
        }

        assert_eq!(profiler.get_all_stats().len(), 1);

        profiler.reset();
        assert_eq!(profiler.get_all_stats().len(), 0);
    }

    #[test]
    fn test_disabled_profiler() {
        let mut profiler = Profiler::new();
        profiler.set_enabled(false);

        {
            let _scope = profiler.begin_scope("test");
        }

        assert_eq!(profiler.get_all_stats().len(), 0);
    }

    #[test]
    fn test_slowest_operation() {
        let mut profiler = Profiler::new();

        {
            let _scope = profiler.begin_scope("fast");
            thread::sleep(Duration::from_millis(1));
        }
        {
            let _scope = profiler.begin_scope("slow");
            thread::sleep(Duration::from_millis(10));
        }

        let (name, _) = profiler.slowest_operation().unwrap();
        assert_eq!(name, "slow");
    }

    #[test]
    fn test_most_called_operation() {
        let mut profiler = Profiler::new();

        for _ in 0..10 {
            let _scope = profiler.begin_scope("frequent");
        }

        for _ in 0..2 {
            let _scope = profiler.begin_scope("rare");
        }

        let (name, stats) = profiler.most_called_operation().unwrap();
        assert_eq!(name, "frequent");
        assert_eq!(stats.call_count, 10);
    }
}
