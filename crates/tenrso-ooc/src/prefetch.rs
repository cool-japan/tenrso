//! Prefetch strategies for improved streaming performance
//!
//! This module provides prefetching strategies to reduce I/O latency
//! by loading chunks before they are needed.
//!
//! # Features
//!
//! - Predictive prefetching based on access patterns
//! - Sequential prefetch for streaming operations
//! - Parallel prefetch with configurable threads
//! - Prefetch queue management
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::prefetch::{Prefetcher, PrefetchStrategy};
//!
//! let mut prefetcher = Prefetcher::new()
//!     .strategy(PrefetchStrategy::Sequential)
//!     .queue_size(4)
//!     .num_threads(2);
//!
//! // Register chunks to prefetch
//! prefetcher.schedule_prefetch(vec!["chunk_0", "chunk_1", "chunk_2"])?;
//!
//! // Retrieve prefetched chunk
//! let chunk = prefetcher.get("chunk_0")?;
//! ```

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use tenrso_core::DenseND;

/// Prefetch strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Sequential prefetch (fetch next N chunks in order)
    Sequential,
    /// Adaptive prefetch based on access patterns
    Adaptive,
    /// All chunks loaded at once (if memory permits)
    Aggressive,
}

/// Prefetch queue entry
#[derive(Debug, Clone)]
struct PrefetchEntry {
    /// Chunk identifier
    #[allow(dead_code)]
    chunk_id: String,
    /// Priority (higher = more important)
    #[allow(dead_code)]
    priority: usize,
    /// Source path (if loading from disk)
    #[allow(dead_code)]
    source_path: Option<PathBuf>,
}

/// Prefetcher for out-of-core chunks
///
/// Manages a queue of chunks to prefetch and provides async loading.
pub struct Prefetcher {
    /// Prefetch strategy
    strategy: PrefetchStrategy,
    /// Maximum queue size
    queue_size: usize,
    /// Number of prefetch threads (for parallel loading)
    num_threads: usize,
    /// Prefetch queue
    queue: VecDeque<PrefetchEntry>,
    /// Prefetched chunks (chunk_id -> tensor)
    prefetched: HashMap<String, DenseND<f64>>,
    /// Access history for adaptive prefetching
    access_history: VecDeque<String>,
    /// History window size
    history_window: usize,
    /// Enable prefetching
    enabled: bool,
}

impl Prefetcher {
    /// Create a new prefetcher
    pub fn new() -> Self {
        Self {
            strategy: PrefetchStrategy::Sequential,
            queue_size: 4,
            num_threads: 1,
            queue: VecDeque::new(),
            prefetched: HashMap::new(),
            access_history: VecDeque::new(),
            history_window: 10,
            enabled: true,
        }
    }

    /// Set prefetch strategy
    pub fn strategy(mut self, strategy: PrefetchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set prefetch queue size
    pub fn queue_size(mut self, size: usize) -> Self {
        self.queue_size = size;
        self
    }

    /// Set number of prefetch threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set history window size
    pub fn history_window(mut self, size: usize) -> Self {
        self.history_window = size;
        self
    }

    /// Enable or disable prefetching
    pub fn enabled(mut self, enable: bool) -> Self {
        self.enabled = enable;
        self
    }

    /// Check if prefetching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.strategy != PrefetchStrategy::None
    }

    /// Schedule chunks for prefetching
    ///
    /// # Arguments
    ///
    /// * `chunk_ids` - List of chunk identifiers to prefetch
    ///
    /// # Returns
    ///
    /// Number of chunks scheduled
    pub fn schedule_prefetch(&mut self, chunk_ids: Vec<&str>) -> Result<usize> {
        if !self.is_enabled() {
            return Ok(0);
        }

        let mut scheduled = 0;

        for (idx, chunk_id) in chunk_ids.iter().enumerate() {
            // Skip if already prefetched
            if self.prefetched.contains_key(*chunk_id) {
                continue;
            }

            // Skip if queue is full
            if self.queue.len() >= self.queue_size {
                break;
            }

            let entry = PrefetchEntry {
                chunk_id: chunk_id.to_string(),
                priority: chunk_ids.len() - idx, // Earlier chunks have higher priority
                source_path: None,
            };

            self.queue.push_back(entry);
            scheduled += 1;
        }

        Ok(scheduled)
    }

    /// Record access to a chunk (for adaptive prefetching)
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier that was accessed
    pub fn record_access(&mut self, chunk_id: &str) {
        self.access_history.push_back(chunk_id.to_string());

        // Maintain window size
        while self.access_history.len() > self.history_window {
            self.access_history.pop_front();
        }

        // If adaptive strategy, predict next chunks
        if self.strategy == PrefetchStrategy::Adaptive && self.enabled {
            let _ = self.predict_next_chunks();
        }
    }

    /// Predict next chunks to prefetch based on access history
    ///
    /// Uses simple pattern matching: if we see patterns like chunk_0, chunk_1, chunk_2,
    /// we'll predict chunk_3, chunk_4, etc.
    ///
    /// # Returns
    ///
    /// Number of chunks predicted and scheduled
    fn predict_next_chunks(&mut self) -> Result<usize> {
        if self.access_history.len() < 2 {
            return Ok(0);
        }

        // Simple sequential pattern detection
        // Check if last N chunks follow a pattern
        let recent: Vec<String> = self
            .access_history
            .iter()
            .rev()
            .take(3)
            .rev()
            .cloned()
            .collect();

        // Try to parse chunk IDs as "chunk_N" format
        let mut numbers = Vec::new();
        for chunk_id in &recent {
            if let Some(num_str) = chunk_id.strip_prefix("chunk_") {
                if let Ok(num) = num_str.parse::<usize>() {
                    numbers.push(num);
                }
            }
        }

        // If we found a sequential pattern, prefetch next chunks
        if numbers.len() >= 2 {
            let is_sequential = numbers.windows(2).all(|w| w[1] == w[0] + 1);

            if is_sequential {
                let next_num = numbers.last().unwrap() + 1;
                let predictions: Vec<String> = (0..self.queue_size)
                    .map(|i| format!("chunk_{}", next_num + i))
                    .collect();

                let chunk_refs: Vec<&str> = predictions.iter().map(|s| s.as_str()).collect();
                return self.schedule_prefetch(chunk_refs);
            }
        }

        Ok(0)
    }

    /// Get a chunk from the prefetch cache
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier
    ///
    /// # Returns
    ///
    /// The prefetched tensor if available
    pub fn get(&mut self, chunk_id: &str) -> Option<DenseND<f64>> {
        let result = self.prefetched.remove(chunk_id);

        // Record access for adaptive prefetching
        self.record_access(chunk_id);

        result
    }

    /// Check if a chunk is prefetched
    pub fn is_prefetched(&self, chunk_id: &str) -> bool {
        self.prefetched.contains_key(chunk_id)
    }

    /// Manually add a prefetched chunk
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier
    /// * `tensor` - The tensor data
    pub fn add_prefetched(&mut self, chunk_id: &str, tensor: DenseND<f64>) {
        self.prefetched.insert(chunk_id.to_string(), tensor);
    }

    /// Clear all prefetched chunks
    pub fn clear(&mut self) {
        self.queue.clear();
        self.prefetched.clear();
    }

    /// Get number of prefetched chunks
    pub fn prefetch_count(&self) -> usize {
        self.prefetched.len()
    }

    /// Get queue size
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Get statistics
    pub fn stats(&self) -> PrefetchStats {
        PrefetchStats {
            strategy: self.strategy,
            queue_size: self.queue_size,
            queue_len: self.queue.len(),
            prefetched_count: self.prefetched.len(),
            access_history_len: self.access_history.len(),
            enabled: self.enabled,
        }
    }

    /// Load chunks from disk (requires mmap feature)
    ///
    /// This method processes the prefetch queue and loads chunks from disk.
    ///
    /// # Arguments
    ///
    /// * `chunk_paths` - Map of chunk_id to file path
    ///
    /// # Returns
    ///
    /// Number of chunks loaded
    #[cfg(feature = "mmap")]
    pub fn load_from_disk(&mut self, chunk_paths: &HashMap<String, PathBuf>) -> Result<usize> {
        if !self.is_enabled() {
            return Ok(0);
        }

        let mut loaded = 0;

        // Process queue
        while let Some(entry) = self.queue.pop_front() {
            if let Some(path) = chunk_paths.get(&entry.chunk_id) {
                let tensor = crate::mmap_io::read_tensor_binary(path)?;
                self.prefetched.insert(entry.chunk_id.clone(), tensor);
                loaded += 1;

                // Don't load more than queue_size chunks at once
                if loaded >= self.queue_size {
                    break;
                }
            }
        }

        Ok(loaded)
    }
}

impl Default for Prefetcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Prefetch statistics
#[derive(Debug, Clone)]
pub struct PrefetchStats {
    /// Current strategy
    pub strategy: PrefetchStrategy,
    /// Maximum queue size
    pub queue_size: usize,
    /// Current queue length
    pub queue_len: usize,
    /// Number of prefetched chunks
    pub prefetched_count: usize,
    /// Access history length
    pub access_history_len: usize,
    /// Whether prefetching is enabled
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetcher_creation() {
        let prefetcher = Prefetcher::new()
            .strategy(PrefetchStrategy::Sequential)
            .queue_size(8)
            .num_threads(2);

        assert_eq!(prefetcher.queue_size, 8);
        assert_eq!(prefetcher.num_threads, 2);
        assert!(prefetcher.is_enabled());
    }

    #[test]
    fn test_schedule_prefetch() {
        let mut prefetcher = Prefetcher::new().queue_size(3);

        let chunks = vec!["chunk_0", "chunk_1", "chunk_2", "chunk_3"];
        let scheduled = prefetcher.schedule_prefetch(chunks).unwrap();

        // Should only schedule queue_size chunks
        assert_eq!(scheduled, 3);
        assert_eq!(prefetcher.queue_len(), 3);
    }

    #[test]
    fn test_record_access() {
        let mut prefetcher = Prefetcher::new().history_window(3);

        prefetcher.record_access("chunk_0");
        prefetcher.record_access("chunk_1");
        prefetcher.record_access("chunk_2");

        assert_eq!(prefetcher.access_history.len(), 3);

        // Adding one more should evict oldest
        prefetcher.record_access("chunk_3");
        assert_eq!(prefetcher.access_history.len(), 3);
        assert_eq!(prefetcher.access_history.front().unwrap(), "chunk_1");
    }

    #[test]
    fn test_add_and_get_prefetched() {
        let mut prefetcher = Prefetcher::new();

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        prefetcher.add_prefetched("chunk_0", tensor.clone());

        assert!(prefetcher.is_prefetched("chunk_0"));
        assert_eq!(prefetcher.prefetch_count(), 1);

        let retrieved = prefetcher.get("chunk_0").unwrap();
        assert_eq!(retrieved.shape(), tensor.shape());

        // Should be removed after get
        assert!(!prefetcher.is_prefetched("chunk_0"));
        assert_eq!(prefetcher.prefetch_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut prefetcher = Prefetcher::new();

        prefetcher
            .schedule_prefetch(vec!["chunk_0", "chunk_1"])
            .unwrap();
        let tensor = DenseND::<f64>::zeros(&[2, 2]);
        prefetcher.add_prefetched("chunk_0", tensor);

        assert_eq!(prefetcher.queue_len(), 2);
        assert_eq!(prefetcher.prefetch_count(), 1);

        prefetcher.clear();

        assert_eq!(prefetcher.queue_len(), 0);
        assert_eq!(prefetcher.prefetch_count(), 0);
    }

    #[test]
    fn test_disabled_prefetcher() {
        let mut prefetcher = Prefetcher::new().enabled(false);

        let scheduled = prefetcher
            .schedule_prefetch(vec!["chunk_0", "chunk_1"])
            .unwrap();

        assert_eq!(scheduled, 0);
        assert_eq!(prefetcher.queue_len(), 0);
    }

    #[test]
    fn test_none_strategy() {
        let mut prefetcher = Prefetcher::new().strategy(PrefetchStrategy::None);

        assert!(!prefetcher.is_enabled());

        let scheduled = prefetcher.schedule_prefetch(vec!["chunk_0"]).unwrap();
        assert_eq!(scheduled, 0);
    }

    #[test]
    fn test_adaptive_prediction() {
        let mut prefetcher = Prefetcher::new()
            .strategy(PrefetchStrategy::Adaptive)
            .queue_size(3);

        // Simulate sequential access pattern
        prefetcher.record_access("chunk_0");
        prefetcher.record_access("chunk_1");
        prefetcher.record_access("chunk_2");

        // Adaptive strategy should predict chunk_3, chunk_4, chunk_5
        // (triggered on record_access)
        assert!(prefetcher.queue_len() > 0);
    }

    #[test]
    fn test_stats() {
        let mut prefetcher = Prefetcher::new()
            .strategy(PrefetchStrategy::Sequential)
            .queue_size(5);

        prefetcher
            .schedule_prefetch(vec!["chunk_0", "chunk_1"])
            .unwrap();

        let stats = prefetcher.stats();
        assert_eq!(stats.strategy, PrefetchStrategy::Sequential);
        assert_eq!(stats.queue_size, 5);
        assert_eq!(stats.queue_len, 2);
        assert!(stats.enabled);
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_load_from_disk() {
        use std::env;

        let mut prefetcher = Prefetcher::new().queue_size(2);

        // Create test files
        let temp_dir = env::temp_dir();
        let path1 = temp_dir.join("test_prefetch_1.bin");
        let path2 = temp_dir.join("test_prefetch_2.bin");

        let tensor1 = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let tensor2 = DenseND::<f64>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

        crate::mmap_io::write_tensor_binary(&path1, &tensor1).unwrap();
        crate::mmap_io::write_tensor_binary(&path2, &tensor2).unwrap();

        // Schedule prefetch
        prefetcher
            .schedule_prefetch(vec!["chunk_0", "chunk_1"])
            .unwrap();

        // Create path map
        let mut chunk_paths = HashMap::new();
        chunk_paths.insert("chunk_0".to_string(), path1.clone());
        chunk_paths.insert("chunk_1".to_string(), path2.clone());

        // Load from disk
        let loaded = prefetcher.load_from_disk(&chunk_paths).unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(prefetcher.prefetch_count(), 2);

        // Cleanup
        std::fs::remove_file(path1).ok();
        std::fs::remove_file(path2).ok();
    }
}
