//! Smart pooling heuristics for automatic buffer pool management
//!
//! This module provides intelligent heuristics for when and how to use
//! memory pooling to maximize performance while minimizing memory overhead.
//!
//! # Heuristics
//!
//! - **Size-based**: Pool buffers above a minimum size threshold
//! - **Frequency-based**: Pool shapes accessed frequently
//! - **Memory-aware**: Adjust pooling based on available memory
//! - **Operation-aware**: Consider operation characteristics
//!
//! # Example
//!
//! ```
//! use tenrso_exec::executor::pool_heuristics::PoolingPolicy;
//!
//! let policy = PoolingPolicy::default();
//!
//! // Should we pool a 1000-element f64 buffer?
//! if policy.should_pool(&[1000], std::mem::size_of::<f64>()) {
//!     println!("Pooling recommended");
//! }
//! ```

use std::collections::HashMap;

/// Minimum buffer size (in bytes) to consider for pooling
/// Buffers smaller than this are cheap to allocate/deallocate
pub const DEFAULT_MIN_POOL_SIZE_BYTES: usize = 4096; // 4KB

/// Maximum buffer size (in bytes) to pool
/// Very large buffers can cause memory pressure
pub const DEFAULT_MAX_POOL_SIZE_BYTES: usize = 64 * 1024 * 1024; // 64MB

/// Minimum access frequency to warrant pooling
/// Shape must be accessed at least this many times
pub const DEFAULT_MIN_ACCESS_FREQUENCY: usize = 2;

/// Policy for determining when to use buffer pooling
#[derive(Debug, Clone)]
pub struct PoolingPolicy {
    /// Minimum buffer size in bytes to pool
    pub min_size_bytes: usize,
    /// Maximum buffer size in bytes to pool
    pub max_size_bytes: usize,
    /// Minimum access frequency for a shape
    pub min_frequency: usize,
    /// Available memory threshold (0.0-1.0)
    /// Don't pool aggressively if available memory is low
    pub memory_pressure_threshold: f64,
    /// Enable adaptive thresholds based on runtime behavior
    pub adaptive: bool,
}

impl PoolingPolicy {
    /// Create a new pooling policy with default settings
    pub fn new() -> Self {
        Self {
            min_size_bytes: DEFAULT_MIN_POOL_SIZE_BYTES,
            max_size_bytes: DEFAULT_MAX_POOL_SIZE_BYTES,
            min_frequency: DEFAULT_MIN_ACCESS_FREQUENCY,
            memory_pressure_threshold: 0.2, // Pool less when <20% memory free
            adaptive: true,
        }
    }

    /// Create a conservative policy (pool only large, frequently accessed buffers)
    pub fn conservative() -> Self {
        Self {
            min_size_bytes: 16384,            // 16KB minimum
            max_size_bytes: 32 * 1024 * 1024, // 32MB max
            min_frequency: 5,
            memory_pressure_threshold: 0.3,
            adaptive: false,
        }
    }

    /// Create an aggressive policy (pool most buffers)
    pub fn aggressive() -> Self {
        Self {
            min_size_bytes: 1024,              // 1KB minimum
            max_size_bytes: 128 * 1024 * 1024, // 128MB max
            min_frequency: 1,
            memory_pressure_threshold: 0.1,
            adaptive: true,
        }
    }

    /// Create a memory-constrained policy (minimize memory usage)
    pub fn memory_constrained() -> Self {
        Self {
            min_size_bytes: 8192,             // 8KB minimum
            max_size_bytes: 16 * 1024 * 1024, // 16MB max
            min_frequency: 3,
            memory_pressure_threshold: 0.4,
            adaptive: false,
        }
    }

    /// Determine if a buffer should be pooled based on its characteristics
    ///
    /// # Arguments
    ///
    /// * `shape` - Buffer shape
    /// * `elem_size` - Size of each element in bytes
    ///
    /// # Returns
    ///
    /// `true` if the buffer should be pooled, `false` otherwise
    pub fn should_pool(&self, shape: &[usize], elem_size: usize) -> bool {
        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * elem_size;

        // Check size bounds
        if total_bytes < self.min_size_bytes {
            return false; // Too small - allocation overhead is minimal
        }

        if total_bytes > self.max_size_bytes {
            return false; // Too large - can cause memory pressure
        }

        true
    }

    /// Check if we should pool based on access frequency
    ///
    /// This should be called after tracking access patterns.
    pub fn should_pool_with_frequency(
        &self,
        shape: &[usize],
        elem_size: usize,
        frequency: usize,
    ) -> bool {
        if !self.should_pool(shape, elem_size) {
            return false;
        }

        frequency >= self.min_frequency
    }

    /// Adjust thresholds based on memory pressure
    ///
    /// # Arguments
    ///
    /// * `available_memory_ratio` - Fraction of memory available (0.0-1.0)
    ///
    /// # Returns
    ///
    /// Adjusted policy with modified thresholds
    pub fn with_memory_pressure(&self, available_memory_ratio: f64) -> Self {
        if !self.adaptive {
            return self.clone();
        }

        let mut adjusted = self.clone();

        if available_memory_ratio < self.memory_pressure_threshold {
            // High memory pressure - be more conservative
            adjusted.min_size_bytes *= 2;
            adjusted.max_size_bytes /= 2;
            adjusted.min_frequency += 2;
        } else if available_memory_ratio > 0.5 {
            // Low memory pressure - can be more aggressive
            adjusted.min_size_bytes = adjusted.min_size_bytes.saturating_sub(1024);
            adjusted.max_size_bytes = adjusted.max_size_bytes.saturating_mul(2);
        }

        adjusted
    }
}

impl Default for PoolingPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Access pattern tracker for shape-based pooling decisions
///
/// Tracks how frequently different shapes are accessed to make
/// intelligent pooling decisions.
#[derive(Debug, Clone)]
pub struct AccessPatternTracker {
    /// Shape signature -> access count
    access_counts: HashMap<String, usize>,
    /// Total number of allocations tracked
    total_accesses: usize,
}

impl AccessPatternTracker {
    /// Create a new access pattern tracker
    pub fn new() -> Self {
        Self {
            access_counts: HashMap::new(),
            total_accesses: 0,
        }
    }

    fn shape_signature(shape: &[usize]) -> String {
        shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }

    /// Record an access to a shape
    pub fn record_access(&mut self, shape: &[usize]) {
        let sig = Self::shape_signature(shape);
        *self.access_counts.entry(sig).or_insert(0) += 1;
        self.total_accesses += 1;
    }

    /// Get access frequency for a shape
    pub fn get_frequency(&self, shape: &[usize]) -> usize {
        let sig = Self::shape_signature(shape);
        *self.access_counts.get(&sig).unwrap_or(&0)
    }

    /// Get top N most frequently accessed shapes
    pub fn top_shapes(&self, n: usize) -> Vec<(String, usize)> {
        let mut sorted: Vec<_> = self
            .access_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }

    /// Clear all recorded access patterns
    pub fn clear(&mut self) {
        self.access_counts.clear();
        self.total_accesses = 0;
    }

    /// Get total number of unique shapes accessed
    pub fn num_unique_shapes(&self) -> usize {
        self.access_counts.len()
    }

    /// Get total number of accesses
    pub fn total_accesses(&self) -> usize {
        self.total_accesses
    }

    /// Get access distribution (shape -> frequency ratio)
    pub fn access_distribution(&self) -> HashMap<String, f64> {
        if self.total_accesses == 0 {
            return HashMap::new();
        }

        self.access_counts
            .iter()
            .map(|(k, &v)| {
                let ratio = v as f64 / self.total_accesses as f64;
                (k.clone(), ratio)
            })
            .collect()
    }
}

impl Default for AccessPatternTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Automatic pooling recommendation engine
///
/// Analyzes access patterns and provides recommendations for
/// optimal pooling configuration.
pub struct PoolingRecommender {
    policy: PoolingPolicy,
    tracker: AccessPatternTracker,
}

impl PoolingRecommender {
    /// Create a new recommender with default policy
    pub fn new() -> Self {
        Self {
            policy: PoolingPolicy::default(),
            tracker: AccessPatternTracker::new(),
        }
    }

    /// Create a recommender with custom policy
    pub fn with_policy(policy: PoolingPolicy) -> Self {
        Self {
            policy,
            tracker: AccessPatternTracker::new(),
        }
    }

    /// Record a buffer allocation
    pub fn record_allocation(&mut self, shape: &[usize]) {
        self.tracker.record_access(shape);
    }

    /// Get shapes that should be pooled based on recorded patterns
    pub fn recommend_shapes(&self, elem_size: usize) -> Vec<String> {
        let mut recommendations = Vec::new();

        for (shape_sig, &frequency) in &self.tracker.access_counts {
            // Parse shape signature back to shape
            let shape: Vec<usize> = shape_sig
                .split('x')
                .filter_map(|s| s.parse().ok())
                .collect();

            if self
                .policy
                .should_pool_with_frequency(&shape, elem_size, frequency)
            {
                recommendations.push(shape_sig.clone());
            }
        }

        recommendations
    }

    /// Generate a report with pooling recommendations
    pub fn generate_report(&self, elem_size: usize) -> PoolingReport {
        let recommended_shapes = self.recommend_shapes(elem_size);
        let top_shapes = self.tracker.top_shapes(10);

        let total_poolable_accesses: usize = recommended_shapes
            .iter()
            .filter_map(|sig| self.tracker.access_counts.get(sig))
            .sum();

        let potential_hit_rate = if self.tracker.total_accesses > 0 {
            total_poolable_accesses as f64 / self.tracker.total_accesses as f64
        } else {
            0.0
        };

        PoolingReport {
            total_shapes_accessed: self.tracker.num_unique_shapes(),
            total_accesses: self.tracker.total_accesses(),
            recommended_shapes_count: recommended_shapes.len(),
            recommended_shapes,
            top_10_shapes: top_shapes,
            potential_hit_rate,
            policy: self.policy.clone(),
        }
    }

    /// Clear all tracking data
    pub fn clear(&mut self) {
        self.tracker.clear();
    }
}

impl Default for PoolingRecommender {
    fn default() -> Self {
        Self::new()
    }
}

/// Report with pooling recommendations
#[derive(Debug, Clone)]
pub struct PoolingReport {
    /// Total number of unique shapes accessed
    pub total_shapes_accessed: usize,
    /// Total number of allocation requests
    pub total_accesses: usize,
    /// Number of shapes recommended for pooling
    pub recommended_shapes_count: usize,
    /// List of recommended shape signatures
    pub recommended_shapes: Vec<String>,
    /// Top 10 most frequently accessed shapes
    pub top_10_shapes: Vec<(String, usize)>,
    /// Potential hit rate if pooling is enabled for recommended shapes
    pub potential_hit_rate: f64,
    /// Policy used for recommendations
    pub policy: PoolingPolicy,
}

impl PoolingReport {
    /// Print a formatted report to console
    pub fn print(&self) {
        println!("=== Pooling Recommendation Report ===");
        println!("Total shapes accessed: {}", self.total_shapes_accessed);
        println!("Total allocations: {}", self.total_accesses);
        println!(
            "Recommended for pooling: {} shapes",
            self.recommended_shapes_count
        );
        println!(
            "Potential hit rate: {:.1}%",
            self.potential_hit_rate * 100.0
        );
        println!("\nTop 10 most accessed shapes:");
        for (i, (shape, count)) in self.top_10_shapes.iter().enumerate() {
            let is_recommended = self.recommended_shapes.contains(shape);
            let marker = if is_recommended { "✓" } else { " " };
            println!("  {}. [{}] {} - {} accesses", i + 1, marker, shape, count);
        }
        println!("\nPolicy settings:");
        println!("  Min size: {} bytes", self.policy.min_size_bytes);
        println!("  Max size: {} bytes", self.policy.max_size_bytes);
        println!("  Min frequency: {}", self.policy.min_frequency);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pooling_policy_default() {
        let policy = PoolingPolicy::default();

        // Small buffer - should not pool
        assert!(!policy.should_pool(&[10], 8)); // 80 bytes

        // Medium buffer - should pool
        assert!(policy.should_pool(&[1000], 8)); // 8KB

        // Very large buffer - should not pool (exceeds max)
        assert!(!policy.should_pool(&[10_000_000], 8)); // 80MB
    }

    #[test]
    fn test_pooling_policy_conservative() {
        let policy = PoolingPolicy::conservative();

        // More restrictive than default
        assert!(!policy.should_pool(&[1000], 8)); // 8KB - below 16KB min
        assert!(policy.should_pool(&[5000], 8)); // 40KB - within range
    }

    #[test]
    fn test_pooling_policy_aggressive() {
        let policy = PoolingPolicy::aggressive();

        // Pool almost everything
        assert!(policy.should_pool(&[200], 8)); // 1.6KB
        assert!(policy.should_pool(&[10_000], 8)); // 80KB
    }

    #[test]
    fn test_pooling_policy_with_frequency() {
        let policy = PoolingPolicy::default();

        // Good size but low frequency
        assert!(!policy.should_pool_with_frequency(&[1000], 8, 1));

        // Good size and good frequency
        assert!(policy.should_pool_with_frequency(&[1000], 8, 5));
    }

    #[test]
    fn test_pooling_policy_memory_pressure() {
        let policy = PoolingPolicy::default();

        // Low memory - adjust to be more conservative
        let adjusted_low = policy.with_memory_pressure(0.1); // 10% memory free
        assert!(adjusted_low.min_size_bytes > policy.min_size_bytes);
        assert!(adjusted_low.max_size_bytes < policy.max_size_bytes);

        // High memory - can be more aggressive
        let adjusted_high = policy.with_memory_pressure(0.8); // 80% memory free
        assert!(adjusted_high.min_size_bytes <= policy.min_size_bytes);
        assert!(adjusted_high.max_size_bytes >= policy.max_size_bytes);
    }

    #[test]
    fn test_access_pattern_tracker() {
        let mut tracker = AccessPatternTracker::new();

        tracker.record_access(&[100]);
        tracker.record_access(&[100]);
        tracker.record_access(&[200]);

        assert_eq!(tracker.get_frequency(&[100]), 2);
        assert_eq!(tracker.get_frequency(&[200]), 1);
        assert_eq!(tracker.get_frequency(&[300]), 0);
        assert_eq!(tracker.total_accesses(), 3);
        assert_eq!(tracker.num_unique_shapes(), 2);
    }

    #[test]
    fn test_access_pattern_top_shapes() {
        let mut tracker = AccessPatternTracker::new();

        for _ in 0..10 {
            tracker.record_access(&[100]);
        }
        for _ in 0..5 {
            tracker.record_access(&[200]);
        }
        for _ in 0..3 {
            tracker.record_access(&[300]);
        }

        let top = tracker.top_shapes(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].1, 10); // [100] accessed 10 times
        assert_eq!(top[1].1, 5); // [200] accessed 5 times
    }

    #[test]
    fn test_access_pattern_clear() {
        let mut tracker = AccessPatternTracker::new();

        tracker.record_access(&[100]);
        tracker.clear();

        assert_eq!(tracker.total_accesses(), 0);
        assert_eq!(tracker.num_unique_shapes(), 0);
    }

    #[test]
    fn test_pooling_recommender_basic() {
        let mut recommender = PoolingRecommender::new();

        // Record accesses
        for _ in 0..10 {
            recommender.record_allocation(&[1000]); // 8KB buffer
        }
        for _ in 0..2 {
            recommender.record_allocation(&[100]); // 800B buffer (too small)
        }

        let recommendations = recommender.recommend_shapes(8);

        // Should recommend [1000] but not [100]
        assert!(recommendations.contains(&"1000".to_string()));
        assert!(!recommendations.contains(&"100".to_string()));
    }

    #[test]
    fn test_pooling_recommender_report() {
        let mut recommender = PoolingRecommender::new();

        for _ in 0..20 {
            recommender.record_allocation(&[1000]);
        }
        for _ in 0..10 {
            recommender.record_allocation(&[2000]);
        }

        let report = recommender.generate_report(8);

        assert_eq!(report.total_accesses, 30);
        assert_eq!(report.total_shapes_accessed, 2);
        assert!(report.recommended_shapes_count > 0);
        assert!(report.potential_hit_rate > 0.0);
    }

    #[test]
    fn test_pooling_recommender_conservative_vs_aggressive() {
        let mut recommender_conservative =
            PoolingRecommender::with_policy(PoolingPolicy::conservative());
        let mut recommender_aggressive =
            PoolingRecommender::with_policy(PoolingPolicy::aggressive());

        // Small buffer accessed frequently
        for _ in 0..10 {
            recommender_conservative.record_allocation(&[500]); // 4KB
            recommender_aggressive.record_allocation(&[500]);
        }

        let rec_conservative = recommender_conservative.recommend_shapes(8);
        let rec_aggressive = recommender_aggressive.recommend_shapes(8);

        // Aggressive should recommend small buffers, conservative should not
        assert!(rec_aggressive.len() >= rec_conservative.len());
    }

    #[test]
    fn test_access_distribution() {
        let mut tracker = AccessPatternTracker::new();

        for _ in 0..50 {
            tracker.record_access(&[100]);
        }
        for _ in 0..30 {
            tracker.record_access(&[200]);
        }
        for _ in 0..20 {
            tracker.record_access(&[300]);
        }

        let dist = tracker.access_distribution();

        assert_eq!(dist.len(), 3);
        assert!((dist["100"] - 0.5).abs() < 0.01); // 50/100 = 0.5
        assert!((dist["200"] - 0.3).abs() < 0.01); // 30/100 = 0.3
        assert!((dist["300"] - 0.2).abs() < 0.01); // 20/100 = 0.2
    }
}
