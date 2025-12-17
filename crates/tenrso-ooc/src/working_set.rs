//! Working set prediction for memory management
//!
//! This module implements algorithms to predict which data chunks will be accessed
//! in the near future, enabling proactive memory management and prefetching.
//!
//! # Features
//!
//! - **Sliding window analysis**: Track recent access patterns
//! - **Access frequency prediction**: Identify hot chunks
//! - **Sequential pattern detection**: Recognize streaming patterns
//! - **Temporal locality prediction**: Predict re-access timing
//! - **Machine learning-ready features**: Export features for ML models
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::working_set::{WorkingSetPredictor, PredictionMode};
//!
//! let mut predictor = WorkingSetPredictor::new()
//!     .window_size(100)
//!     .prediction_mode(PredictionMode::Adaptive);
//!
//! // Record accesses
//! predictor.record_access("chunk_0", 1024);
//! predictor.record_access("chunk_1", 2048);
//!
//! // Predict working set
//! let predicted = predictor.predict_working_set(10)?;
//! ```

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Prediction mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMode {
    /// Frequency-based prediction (MFU - Most Frequently Used)
    Frequency,
    /// Recency-based prediction (MRU - Most Recently Used)
    Recency,
    /// Hybrid approach combining frequency and recency
    Hybrid,
    /// Adaptive prediction based on access patterns
    Adaptive,
    /// Sequential pattern detection
    Sequential,
}

/// Access record
#[derive(Debug, Clone)]
struct AccessRecord {
    chunk_id: String,
    #[allow(dead_code)]
    timestamp: SystemTime,
    #[allow(dead_code)]
    size_bytes: usize,
}

/// Chunk access statistics
#[derive(Debug, Clone)]
struct ChunkStats {
    chunk_id: String,
    access_count: usize,
    last_access: SystemTime,
    first_access: SystemTime,
    total_size: usize,
    access_intervals: VecDeque<Duration>,
    predicted_next_access: Option<SystemTime>,
}

impl ChunkStats {
    fn new(chunk_id: String, timestamp: SystemTime, size_bytes: usize) -> Self {
        Self {
            chunk_id,
            access_count: 1,
            last_access: timestamp,
            first_access: timestamp,
            total_size: size_bytes,
            access_intervals: VecDeque::new(),
            predicted_next_access: None,
        }
    }

    /// Update with new access
    fn update(&mut self, timestamp: SystemTime, size_bytes: usize) {
        if let Ok(interval) = timestamp.duration_since(self.last_access) {
            self.access_intervals.push_back(interval);
            if self.access_intervals.len() > 10 {
                self.access_intervals.pop_front();
            }
        }

        self.access_count += 1;
        self.last_access = timestamp;
        self.total_size = size_bytes;

        // Predict next access based on interval pattern
        self.update_prediction();
    }

    /// Update prediction of next access time
    fn update_prediction(&mut self) {
        if self.access_intervals.is_empty() {
            return;
        }

        // Calculate average interval
        let sum_secs: f64 = self.access_intervals.iter().map(|d| d.as_secs_f64()).sum();
        let avg_interval = sum_secs / self.access_intervals.len() as f64;

        // Predict next access
        self.predicted_next_access = self
            .last_access
            .checked_add(Duration::from_secs_f64(avg_interval));
    }

    /// Calculate frequency score (accesses per time)
    fn frequency_score(&self) -> f64 {
        let lifetime = SystemTime::now()
            .duration_since(self.first_access)
            .unwrap_or(Duration::from_secs(1))
            .as_secs_f64();

        self.access_count as f64 / lifetime.max(1.0)
    }

    /// Calculate recency score (inverse of time since last access)
    fn recency_score(&self) -> f64 {
        let time_since_access = SystemTime::now()
            .duration_since(self.last_access)
            .unwrap_or(Duration::from_secs(0))
            .as_secs_f64();

        1.0 / (1.0 + time_since_access)
    }

    /// Calculate hybrid score
    fn hybrid_score(&self) -> f64 {
        let freq = self.frequency_score();
        let rec = self.recency_score();

        // Weighted combination (favor recency slightly)
        0.4 * freq + 0.6 * rec
    }

    /// Calculate adaptive score based on access pattern
    fn adaptive_score(&self) -> f64 {
        let freq = self.frequency_score();
        let rec = self.recency_score();

        // Check if access pattern is regular
        let is_regular = self.is_regular_pattern();

        if is_regular {
            // Regular pattern: weight frequency higher
            0.7 * freq + 0.3 * rec
        } else {
            // Irregular pattern: weight recency higher
            0.3 * freq + 0.7 * rec
        }
    }

    /// Check if access pattern is regular
    fn is_regular_pattern(&self) -> bool {
        if self.access_intervals.len() < 3 {
            return false;
        }

        // Calculate coefficient of variation
        let intervals: Vec<f64> = self
            .access_intervals
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance =
            intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / intervals.len() as f64;
        let std_dev = variance.sqrt();

        let cv = std_dev / mean;

        // Regular if coefficient of variation is low
        cv < 0.5
    }
}

/// Working set predictor
pub struct WorkingSetPredictor {
    mode: PredictionMode,
    window_size: usize,
    access_history: VecDeque<AccessRecord>,
    chunk_stats: HashMap<String, ChunkStats>,
    sequential_threshold: f64,
}

impl WorkingSetPredictor {
    /// Create a new working set predictor
    pub fn new() -> Self {
        Self {
            mode: PredictionMode::Adaptive,
            window_size: 100,
            access_history: VecDeque::new(),
            chunk_stats: HashMap::new(),
            sequential_threshold: 0.8,
        }
    }

    /// Set prediction mode
    pub fn prediction_mode(mut self, mode: PredictionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set window size
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set sequential threshold
    pub fn sequential_threshold(mut self, threshold: f64) -> Self {
        self.sequential_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Record a chunk access
    pub fn record_access(&mut self, chunk_id: &str, size_bytes: usize) {
        let timestamp = SystemTime::now();

        // Add to history
        self.access_history.push_back(AccessRecord {
            chunk_id: chunk_id.to_string(),
            timestamp,
            size_bytes,
        });

        // Trim history to window size
        if self.access_history.len() > self.window_size {
            self.access_history.pop_front();
        }

        // Update or create chunk stats
        self.chunk_stats
            .entry(chunk_id.to_string())
            .and_modify(|stats| stats.update(timestamp, size_bytes))
            .or_insert_with(|| ChunkStats::new(chunk_id.to_string(), timestamp, size_bytes));
    }

    /// Predict working set (top N chunks likely to be accessed)
    pub fn predict_working_set(&self, n: usize) -> Result<Vec<WorkingSetPrediction>> {
        let mut predictions: Vec<(String, f64, usize)> = self
            .chunk_stats
            .values()
            .map(|stats| {
                let score = self.calculate_score(stats);
                (stats.chunk_id.clone(), score, stats.total_size)
            })
            .collect();

        // Sort by score (descending)
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top N
        let result = predictions
            .into_iter()
            .take(n)
            .map(|(chunk_id, score, size_bytes)| {
                let confidence = self.calculate_confidence(&chunk_id);
                WorkingSetPrediction {
                    chunk_id,
                    score,
                    size_bytes,
                    confidence,
                }
            })
            .collect();

        Ok(result)
    }

    /// Calculate score based on prediction mode
    fn calculate_score(&self, stats: &ChunkStats) -> f64 {
        match self.mode {
            PredictionMode::Frequency => stats.frequency_score(),
            PredictionMode::Recency => stats.recency_score(),
            PredictionMode::Hybrid => stats.hybrid_score(),
            PredictionMode::Adaptive => stats.adaptive_score(),
            PredictionMode::Sequential => self.sequential_score(stats),
        }
    }

    /// Calculate sequential score
    fn sequential_score(&self, stats: &ChunkStats) -> f64 {
        // Check if this chunk is part of a sequential pattern
        let is_sequential = self.is_sequential_access(&stats.chunk_id);

        if is_sequential {
            // High score for sequential accesses
            stats.recency_score() * 2.0
        } else {
            // Normal score
            stats.recency_score()
        }
    }

    /// Check if chunk is part of sequential access pattern
    fn is_sequential_access(&self, chunk_id: &str) -> bool {
        // Look for numeric suffix pattern (e.g., chunk_0, chunk_1, chunk_2)
        if let Some(last_underscore) = chunk_id.rfind('_') {
            if let Ok(_num) = chunk_id[last_underscore + 1..].parse::<usize>() {
                let prefix = &chunk_id[..last_underscore + 1];

                // Check recent history for sequential pattern
                let recent_chunks: Vec<String> = if self.access_history.len() >= 5 {
                    self.access_history
                        .iter()
                        .skip(self.access_history.len() - 5)
                        .map(|r| r.chunk_id.clone())
                        .collect()
                } else {
                    self.access_history
                        .iter()
                        .map(|r| r.chunk_id.clone())
                        .collect()
                };

                let sequential_count = recent_chunks
                    .windows(2)
                    .filter(|w| {
                        if let (Some(u1), Some(u2)) = (
                            w[0].rfind('_')
                                .and_then(|i| w[0][i + 1..].parse::<usize>().ok()),
                            w[1].rfind('_')
                                .and_then(|i| w[1][i + 1..].parse::<usize>().ok()),
                        ) {
                            w[0].starts_with(prefix) && w[1].starts_with(prefix) && u2 == u1 + 1
                        } else {
                            false
                        }
                    })
                    .count();

                let total_pairs = recent_chunks.len().saturating_sub(1).max(1);
                let sequential_ratio = sequential_count as f64 / total_pairs as f64;

                return sequential_ratio >= self.sequential_threshold;
            }
        }

        false
    }

    /// Calculate confidence for a prediction
    fn calculate_confidence(&self, chunk_id: &str) -> f64 {
        if let Some(stats) = self.chunk_stats.get(chunk_id) {
            // Confidence based on:
            // 1. Number of observations
            // 2. Regularity of pattern
            // 3. Recency

            let observation_confidence = (stats.access_count as f64 / 10.0).min(1.0);
            let pattern_confidence = if stats.is_regular_pattern() { 1.0 } else { 0.5 };
            let recency_confidence = stats.recency_score();

            (observation_confidence + pattern_confidence + recency_confidence) / 3.0
        } else {
            0.0
        }
    }

    /// Get statistics for a chunk
    pub fn chunk_stats(&self, chunk_id: &str) -> Option<ChunkStatistics> {
        self.chunk_stats.get(chunk_id).map(|stats| ChunkStatistics {
            chunk_id: stats.chunk_id.clone(),
            access_count: stats.access_count,
            last_access: stats.last_access,
            predicted_next_access: stats.predicted_next_access,
            is_regular: stats.is_regular_pattern(),
            frequency_score: stats.frequency_score(),
            recency_score: stats.recency_score(),
        })
    }

    /// Get overall statistics
    pub fn overall_stats(&self) -> PredictorStatistics {
        let total_chunks = self.chunk_stats.len();
        let total_accesses = self.access_history.len();

        let regular_patterns = self
            .chunk_stats
            .values()
            .filter(|s| s.is_regular_pattern())
            .count();

        let sequential_chunks = self
            .chunk_stats
            .keys()
            .filter(|id| self.is_sequential_access(id))
            .count();

        PredictorStatistics {
            total_chunks,
            total_accesses,
            regular_patterns,
            sequential_chunks,
            window_size: self.window_size,
            mode: self.mode,
        }
    }

    /// Clear all history (useful for testing or reset)
    pub fn clear(&mut self) {
        self.access_history.clear();
        self.chunk_stats.clear();
    }
}

impl Default for WorkingSetPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Working set prediction result
#[derive(Debug, Clone)]
pub struct WorkingSetPrediction {
    pub chunk_id: String,
    pub score: f64,
    pub size_bytes: usize,
    pub confidence: f64,
}

/// Chunk statistics
#[derive(Debug, Clone)]
pub struct ChunkStatistics {
    pub chunk_id: String,
    pub access_count: usize,
    pub last_access: SystemTime,
    pub predicted_next_access: Option<SystemTime>,
    pub is_regular: bool,
    pub frequency_score: f64,
    pub recency_score: f64,
}

/// Predictor statistics
#[derive(Debug, Clone)]
pub struct PredictorStatistics {
    pub total_chunks: usize,
    pub total_accesses: usize,
    pub regular_patterns: usize,
    pub sequential_chunks: usize,
    pub window_size: usize,
    pub mode: PredictionMode,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_predictor_creation() {
        let predictor = WorkingSetPredictor::new();
        assert_eq!(predictor.mode, PredictionMode::Adaptive);
        assert_eq!(predictor.window_size, 100);
    }

    #[test]
    fn test_record_access() {
        let mut predictor = WorkingSetPredictor::new();
        predictor.record_access("chunk_0", 1024);
        predictor.record_access("chunk_1", 2048);

        assert_eq!(predictor.chunk_stats.len(), 2);
        assert_eq!(predictor.access_history.len(), 2);
    }

    #[test]
    fn test_predict_working_set() {
        let mut predictor = WorkingSetPredictor::new();

        // Record multiple accesses
        for _ in 0..5 {
            predictor.record_access("chunk_0", 1024);
        }
        predictor.record_access("chunk_1", 2048);

        let predictions = predictor.predict_working_set(2).unwrap();
        assert_eq!(predictions.len(), 2);

        // chunk_0 should score higher (more accesses)
        assert_eq!(predictions[0].chunk_id, "chunk_0");
    }

    #[test]
    fn test_frequency_mode() {
        let mut predictor = WorkingSetPredictor::new().prediction_mode(PredictionMode::Frequency);

        // chunk_0 accessed more frequently
        for _ in 0..10 {
            predictor.record_access("chunk_0", 1024);
        }
        predictor.record_access("chunk_1", 2048);

        let predictions = predictor.predict_working_set(2).unwrap();
        assert_eq!(predictions[0].chunk_id, "chunk_0");
    }

    #[test]
    fn test_recency_mode() {
        let mut predictor = WorkingSetPredictor::new().prediction_mode(PredictionMode::Recency);

        predictor.record_access("chunk_0", 1024);
        thread::sleep(Duration::from_millis(10));
        predictor.record_access("chunk_1", 2048);

        let predictions = predictor.predict_working_set(2).unwrap();
        // chunk_1 is more recent
        assert_eq!(predictions[0].chunk_id, "chunk_1");
    }

    #[test]
    fn test_sequential_pattern_detection() {
        let mut predictor = WorkingSetPredictor::new().prediction_mode(PredictionMode::Sequential);

        // Create sequential pattern
        for i in 0..5 {
            predictor.record_access(&format!("chunk_{}", i), 1024);
        }

        // Check if sequential pattern is detected
        assert!(predictor.is_sequential_access("chunk_4"));
    }

    #[test]
    fn test_regular_pattern_detection() {
        let mut predictor = WorkingSetPredictor::new();

        // Create regular access pattern with fixed interval
        for _ in 0..5 {
            predictor.record_access("chunk_0", 1024);
            thread::sleep(Duration::from_millis(10));
        }

        let stats = predictor.chunk_stats.get("chunk_0").unwrap();
        assert!(stats.is_regular_pattern());
    }

    #[test]
    fn test_chunk_statistics() {
        let mut predictor = WorkingSetPredictor::new();
        predictor.record_access("chunk_0", 1024);
        predictor.record_access("chunk_0", 1024);

        let stats = predictor.chunk_stats("chunk_0").unwrap();
        assert_eq!(stats.chunk_id, "chunk_0");
        assert_eq!(stats.access_count, 2);
    }

    #[test]
    fn test_overall_statistics() {
        let mut predictor = WorkingSetPredictor::new();
        predictor.record_access("chunk_0", 1024);
        predictor.record_access("chunk_1", 2048);

        let stats = predictor.overall_stats();
        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.total_accesses, 2);
    }

    #[test]
    fn test_window_size_limit() {
        let mut predictor = WorkingSetPredictor::new().window_size(5);

        // Record more accesses than window size
        for i in 0..10 {
            predictor.record_access(&format!("chunk_{}", i), 1024);
        }

        // History should be limited to window size
        assert_eq!(predictor.access_history.len(), 5);
    }

    #[test]
    fn test_confidence_calculation() {
        let mut predictor = WorkingSetPredictor::new();

        // Single access - low confidence
        predictor.record_access("chunk_0", 1024);
        let conf1 = predictor.calculate_confidence("chunk_0");

        // Multiple accesses - higher confidence
        for _ in 0..10 {
            predictor.record_access("chunk_0", 1024);
        }
        let conf2 = predictor.calculate_confidence("chunk_0");

        assert!(conf2 > conf1);
    }
}
