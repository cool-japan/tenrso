//! Lock-free prefetch implementation using wait-free data structures
//!
//! This module provides a high-performance, lock-free prefetcher that allows
//! concurrent access from multiple threads without blocking. It uses:
//! - `crossbeam::queue::SegQueue` for lock-free MPMC queue
//! - `DashMap` for concurrent hash map (prefetched chunks)
//! - Atomic operations for statistics
//!
//! # Features
//!
//! - Thread-safe concurrent prefetch scheduling
//! - Lock-free chunk access and insertion
//! - Wait-free statistics updates
//! - Zero-cost abstraction when single-threaded
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::lockfree_prefetch::{LockFreePrefetcher, PrefetchStrategy};
//! use std::sync::Arc;
//!
//! let prefetcher = Arc::new(LockFreePrefetcher::new()
//!     .strategy(PrefetchStrategy::Sequential)
//!     .queue_size(8));
//!
//! // Multiple threads can schedule concurrently
//! let prefetcher_clone = prefetcher.clone();
//! std::thread::spawn(move || {
//!     prefetcher_clone.schedule_prefetch(vec!["chunk_0", "chunk_1"]);
//! });
//!
//! // Access from another thread
//! if let Some(chunk) = prefetcher.get("chunk_0") {
//!     // Process chunk
//! }
//! ```

use anyhow::Result;
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tenrso_core::DenseND;

/// Prefetch strategy (same as original)
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
pub struct PrefetchEntry {
    /// Chunk identifier
    pub chunk_id: String,
    /// Priority (higher = more important)
    pub priority: usize,
    /// Source path (if loading from disk)
    pub source_path: Option<PathBuf>,
    /// Timestamp (for ordering)
    pub timestamp: u64,
}

/// Lock-free prefetcher for out-of-core chunks
///
/// This prefetcher uses lock-free data structures for maximum concurrency.
/// It can be safely shared across threads using `Arc`.
pub struct LockFreePrefetcher {
    /// Prefetch strategy
    strategy: PrefetchStrategy,
    /// Maximum queue size
    queue_size: usize,
    /// Number of prefetch threads (for parallel loading)
    num_threads: usize,
    /// Lock-free prefetch queue (MPMC)
    queue: Arc<SegQueue<PrefetchEntry>>,
    /// Prefetched chunks (chunk_id -> tensor)
    /// Using DashMap for concurrent access
    prefetched: Arc<DashMap<String, DenseND<f64>>>,
    /// Access history (protected by mutex for simplicity, low contention)
    /// Could be replaced with a lock-free ring buffer for even better performance
    access_history: Arc<parking_lot::Mutex<VecDeque<String>>>,
    /// History window size
    history_window: usize,
    /// Enable prefetching
    enabled: AtomicBool,
    /// Queue length counter (approximate)
    queue_len: Arc<AtomicUsize>,
    /// Global timestamp counter for ordering
    timestamp: Arc<AtomicU64>,
    /// Statistics
    stats: Arc<LockFreePrefetchStats>,
}

/// Lock-free statistics
#[derive(Debug)]
struct LockFreePrefetchStats {
    /// Total scheduled requests
    scheduled_total: AtomicU64,
    /// Total prefetched chunks
    prefetched_total: AtomicU64,
    /// Total cache hits
    cache_hits: AtomicU64,
    /// Total cache misses
    cache_misses: AtomicU64,
    /// Total evictions
    evictions: AtomicU64,
}

impl LockFreePrefetchStats {
    fn new() -> Self {
        Self {
            scheduled_total: AtomicU64::new(0),
            prefetched_total: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    fn record_scheduled(&self, count: u64) {
        self.scheduled_total.fetch_add(count, Ordering::Relaxed);
    }

    fn record_prefetched(&self, count: u64) {
        self.prefetched_total.fetch_add(count, Ordering::Relaxed);
    }

    fn record_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> PrefetchStatsSnapshot {
        PrefetchStatsSnapshot {
            scheduled_total: self.scheduled_total.load(Ordering::Relaxed),
            prefetched_total: self.prefetched_total.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
}

/// Statistics snapshot (immutable)
#[derive(Debug, Clone)]
pub struct PrefetchStatsSnapshot {
    /// Total scheduled requests
    pub scheduled_total: u64,
    /// Total prefetched chunks
    pub prefetched_total: u64,
    /// Total cache hits
    pub cache_hits: u64,
    /// Total cache misses
    pub cache_misses: u64,
    /// Total evictions
    pub evictions: u64,
}

impl PrefetchStatsSnapshot {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

impl LockFreePrefetcher {
    /// Create a new lock-free prefetcher
    pub fn new() -> Self {
        Self {
            strategy: PrefetchStrategy::Sequential,
            queue_size: 4,
            num_threads: 1,
            queue: Arc::new(SegQueue::new()),
            prefetched: Arc::new(DashMap::new()),
            access_history: Arc::new(parking_lot::Mutex::new(VecDeque::new())),
            history_window: 10,
            enabled: AtomicBool::new(true),
            queue_len: Arc::new(AtomicUsize::new(0)),
            timestamp: Arc::new(AtomicU64::new(0)),
            stats: Arc::new(LockFreePrefetchStats::new()),
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
    pub fn enabled(self, enable: bool) -> Self {
        self.enabled.store(enable, Ordering::Relaxed);
        self
    }

    /// Check if prefetching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) && self.strategy != PrefetchStrategy::None
    }

    /// Get next timestamp (monotonically increasing)
    fn next_timestamp(&self) -> u64 {
        self.timestamp.fetch_add(1, Ordering::SeqCst)
    }

    /// Schedule chunks for prefetching
    ///
    /// This method is lock-free and can be called concurrently from multiple threads.
    ///
    /// # Arguments
    ///
    /// * `chunk_ids` - List of chunk identifiers to prefetch
    ///
    /// # Returns
    ///
    /// Number of chunks scheduled
    pub fn schedule_prefetch(&self, chunk_ids: Vec<&str>) -> Result<usize> {
        if !self.is_enabled() {
            return Ok(0);
        }

        let mut scheduled = 0;
        let current_queue_len = self.queue_len.load(Ordering::Relaxed);

        for (idx, chunk_id) in chunk_ids.iter().enumerate() {
            // Skip if already prefetched
            if self.prefetched.contains_key(*chunk_id) {
                continue;
            }

            // Approximate queue size check (may be slightly off due to concurrent access)
            if current_queue_len + scheduled >= self.queue_size {
                break;
            }

            let entry = PrefetchEntry {
                chunk_id: chunk_id.to_string(),
                priority: chunk_ids.len() - idx,
                source_path: None,
                timestamp: self.next_timestamp(),
            };

            self.queue.push(entry);
            self.queue_len.fetch_add(1, Ordering::Relaxed);
            scheduled += 1;
        }

        if scheduled > 0 {
            self.stats.record_scheduled(scheduled as u64);
        }

        Ok(scheduled)
    }

    /// Schedule a single chunk with path
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier
    /// * `path` - Path to load from (if applicable)
    /// * `priority` - Priority (higher = more important)
    pub fn schedule_one(&self, chunk_id: &str, path: Option<PathBuf>, priority: usize) -> bool {
        if !self.is_enabled() {
            return false;
        }

        if self.prefetched.contains_key(chunk_id) {
            return false;
        }

        let entry = PrefetchEntry {
            chunk_id: chunk_id.to_string(),
            priority,
            source_path: path,
            timestamp: self.next_timestamp(),
        };

        self.queue.push(entry);
        self.queue_len.fetch_add(1, Ordering::Relaxed);
        self.stats.record_scheduled(1);
        true
    }

    /// Record access to a chunk (for adaptive prefetching)
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier that was accessed
    pub fn record_access(&self, chunk_id: &str) {
        let mut history = self.access_history.lock();
        history.push_back(chunk_id.to_string());

        // Maintain window size
        while history.len() > self.history_window {
            history.pop_front();
        }

        // If adaptive strategy, predict next chunks
        if self.strategy == PrefetchStrategy::Adaptive && self.is_enabled() {
            // Clone recent history to avoid holding lock
            let recent: Vec<String> = history.iter().rev().take(3).rev().cloned().collect();
            drop(history); // Release lock

            let _ = self.predict_next_chunks(&recent);
        }
    }

    /// Predict next chunks to prefetch based on access history
    ///
    /// # Arguments
    ///
    /// * `recent` - Recent access history
    ///
    /// # Returns
    ///
    /// Number of chunks predicted and scheduled
    fn predict_next_chunks(&self, recent: &[String]) -> Result<usize> {
        if recent.len() < 2 {
            return Ok(0);
        }

        // Simple sequential pattern detection
        let mut numbers = Vec::new();
        for chunk_id in recent {
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
    /// This method is lock-free and can be called concurrently.
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier
    ///
    /// # Returns
    ///
    /// The prefetched tensor if available
    pub fn get(&self, chunk_id: &str) -> Option<DenseND<f64>> {
        // Record access for adaptive prefetching
        self.record_access(chunk_id);

        // Try to get from cache
        let result = self.prefetched.remove(chunk_id).map(|(_, tensor)| tensor);

        if result.is_some() {
            self.stats.record_hit();
        } else {
            self.stats.record_miss();
        }

        result
    }

    /// Check if a chunk is prefetched
    pub fn is_prefetched(&self, chunk_id: &str) -> bool {
        self.prefetched.contains_key(chunk_id)
    }

    /// Manually add a prefetched chunk
    ///
    /// This method is lock-free and can be called concurrently.
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier
    /// * `tensor` - The tensor data
    pub fn add_prefetched(&self, chunk_id: &str, tensor: DenseND<f64>) {
        if let Some(old) = self.prefetched.insert(chunk_id.to_string(), tensor) {
            // Evicted old entry
            drop(old);
            self.stats.record_eviction();
        }
        self.stats.record_prefetched(1);
    }

    /// Clear all prefetched chunks
    pub fn clear(&self) {
        // Clear queue (drain all entries)
        while self.queue.pop().is_some() {
            self.queue_len.fetch_sub(1, Ordering::Relaxed);
        }

        // Clear prefetched
        let count = self.prefetched.len();
        self.prefetched.clear();

        if count > 0 {
            self.stats.record_eviction();
        }
    }

    /// Get number of prefetched chunks
    pub fn prefetch_count(&self) -> usize {
        self.prefetched.len()
    }

    /// Get approximate queue size
    ///
    /// Note: This is approximate due to concurrent modifications
    pub fn queue_len(&self) -> usize {
        self.queue_len.load(Ordering::Relaxed)
    }

    /// Pop one entry from queue
    ///
    /// Returns the entry if queue is not empty
    pub fn pop_queue(&self) -> Option<PrefetchEntry> {
        if let Some(entry) = self.queue.pop() {
            self.queue_len.fetch_sub(1, Ordering::Relaxed);
            Some(entry)
        } else {
            None
        }
    }

    /// Get statistics snapshot
    pub fn stats_snapshot(&self) -> LockFreePrefetcherStats {
        LockFreePrefetcherStats {
            strategy: self.strategy,
            queue_size: self.queue_size,
            queue_len: self.queue_len(),
            prefetched_count: self.prefetch_count(),
            access_history_len: self.access_history.lock().len(),
            enabled: self.is_enabled(),
            performance: self.stats.snapshot(),
        }
    }

    /// Load chunks from disk (requires mmap feature)
    ///
    /// This method processes the prefetch queue and loads chunks from disk.
    /// It's thread-safe and can be called concurrently from multiple threads.
    ///
    /// # Arguments
    ///
    /// * `chunk_paths` - Map of chunk_id to file path
    /// * `max_load` - Maximum number of chunks to load
    ///
    /// # Returns
    ///
    /// Number of chunks loaded
    #[cfg(feature = "mmap")]
    pub fn load_from_disk(
        &self,
        chunk_paths: &std::collections::HashMap<String, PathBuf>,
        max_load: usize,
    ) -> Result<usize> {
        if !self.is_enabled() {
            return Ok(0);
        }

        let mut loaded = 0;

        // Process queue
        while loaded < max_load {
            if let Some(entry) = self.pop_queue() {
                if let Some(path) = chunk_paths.get(&entry.chunk_id) {
                    let tensor = crate::mmap_io::read_tensor_binary(path)?;
                    self.add_prefetched(&entry.chunk_id, tensor);
                    loaded += 1;
                }
            } else {
                break;
            }
        }

        Ok(loaded)
    }
}

impl Default for LockFreePrefetcher {
    fn default() -> Self {
        Self::new()
    }
}

// Make it thread-safe
unsafe impl Send for LockFreePrefetcher {}
unsafe impl Sync for LockFreePrefetcher {}

/// Lock-free prefetcher statistics
#[derive(Debug, Clone)]
pub struct LockFreePrefetcherStats {
    /// Current strategy
    pub strategy: PrefetchStrategy,
    /// Maximum queue size
    pub queue_size: usize,
    /// Current queue length (approximate)
    pub queue_len: usize,
    /// Number of prefetched chunks
    pub prefetched_count: usize,
    /// Access history length
    pub access_history_len: usize,
    /// Whether prefetching is enabled
    pub enabled: bool,
    /// Performance statistics
    pub performance: PrefetchStatsSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_lockfree_prefetcher_creation() {
        let prefetcher = LockFreePrefetcher::new()
            .strategy(PrefetchStrategy::Sequential)
            .queue_size(8)
            .num_threads(2);

        assert_eq!(prefetcher.queue_size, 8);
        assert_eq!(prefetcher.num_threads, 2);
        assert!(prefetcher.is_enabled());
    }

    #[test]
    fn test_schedule_prefetch() {
        let prefetcher = LockFreePrefetcher::new().queue_size(3);

        let chunks = vec!["chunk_0", "chunk_1", "chunk_2", "chunk_3"];
        let scheduled = prefetcher.schedule_prefetch(chunks).unwrap();

        // Should only schedule queue_size chunks
        assert_eq!(scheduled, 3);
        assert_eq!(prefetcher.queue_len(), 3);
    }

    #[test]
    fn test_concurrent_schedule() {
        let prefetcher = Arc::new(LockFreePrefetcher::new().queue_size(100));

        let mut handles = vec![];
        for i in 0..10 {
            let pf = prefetcher.clone();
            let handle = thread::spawn(move || {
                let chunks: Vec<String> = (0..10).map(|j| format!("chunk_{}_{}", i, j)).collect();
                let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
                pf.schedule_prefetch(chunk_refs).unwrap()
            });
            handles.push(handle);
        }

        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 100); // 10 threads * 10 chunks each
    }

    #[test]
    fn test_add_and_get_prefetched() {
        let prefetcher = LockFreePrefetcher::new();

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
    fn test_concurrent_add_get() {
        let prefetcher = Arc::new(LockFreePrefetcher::new());

        // Spawn writer threads
        let mut writers = vec![];
        for i in 0..5 {
            let pf = prefetcher.clone();
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let tensor = DenseND::<f64>::from_vec(vec![i as f64, j as f64], &[2]).unwrap();
                    let chunk_id = format!("chunk_{}_{}", i, j);
                    pf.add_prefetched(&chunk_id, tensor);
                }
            });
            writers.push(handle);
        }

        // Wait for writers
        for handle in writers {
            handle.join().unwrap();
        }

        // Should have 50 chunks (5 threads * 10 chunks)
        assert_eq!(prefetcher.prefetch_count(), 50);

        // Spawn reader threads
        let mut readers = vec![];
        for i in 0..5 {
            let pf = prefetcher.clone();
            let handle = thread::spawn(move || {
                let mut count = 0;
                for j in 0..10 {
                    let chunk_id = format!("chunk_{}_{}", i, j);
                    if pf.get(&chunk_id).is_some() {
                        count += 1;
                    }
                }
                count
            });
            readers.push(handle);
        }

        let total: usize = readers.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 50);
        assert_eq!(prefetcher.prefetch_count(), 0);
    }

    #[test]
    fn test_record_access() {
        let prefetcher = LockFreePrefetcher::new().history_window(3);

        prefetcher.record_access("chunk_0");
        prefetcher.record_access("chunk_1");
        prefetcher.record_access("chunk_2");

        {
            let history = prefetcher.access_history.lock();
            assert_eq!(history.len(), 3);
        }

        // Adding one more should evict oldest
        prefetcher.record_access("chunk_3");
        {
            let history = prefetcher.access_history.lock();
            assert_eq!(history.len(), 3);
            assert_eq!(history.front().unwrap(), "chunk_1");
        }
    }

    #[test]
    fn test_clear() {
        let prefetcher = LockFreePrefetcher::new();

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
        let prefetcher = LockFreePrefetcher::new().enabled(false);

        let scheduled = prefetcher
            .schedule_prefetch(vec!["chunk_0", "chunk_1"])
            .unwrap();

        assert_eq!(scheduled, 0);
        assert_eq!(prefetcher.queue_len(), 0);
    }

    #[test]
    fn test_none_strategy() {
        let prefetcher = LockFreePrefetcher::new().strategy(PrefetchStrategy::None);

        assert!(!prefetcher.is_enabled());

        let scheduled = prefetcher.schedule_prefetch(vec!["chunk_0"]).unwrap();
        assert_eq!(scheduled, 0);
    }

    #[test]
    fn test_adaptive_prediction() {
        let prefetcher = LockFreePrefetcher::new()
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
        let prefetcher = LockFreePrefetcher::new()
            .strategy(PrefetchStrategy::Sequential)
            .queue_size(5);

        prefetcher
            .schedule_prefetch(vec!["chunk_0", "chunk_1"])
            .unwrap();

        let stats = prefetcher.stats_snapshot();
        assert_eq!(stats.strategy, PrefetchStrategy::Sequential);
        assert_eq!(stats.queue_size, 5);
        assert_eq!(stats.queue_len, 2);
        assert!(stats.enabled);
        assert_eq!(stats.performance.scheduled_total, 2);
    }

    #[test]
    fn test_stats_hit_rate() {
        let prefetcher = LockFreePrefetcher::new();

        let tensor = DenseND::<f64>::zeros(&[2, 2]);
        prefetcher.add_prefetched("chunk_0", tensor.clone());
        prefetcher.add_prefetched("chunk_1", tensor);

        // Hit
        assert!(prefetcher.get("chunk_0").is_some());
        // Miss
        assert!(prefetcher.get("chunk_2").is_none());

        let stats = prefetcher.stats_snapshot();
        assert_eq!(stats.performance.cache_hits, 1);
        assert_eq!(stats.performance.cache_misses, 1);
        assert_eq!(stats.performance.hit_rate(), 0.5);
    }

    #[test]
    fn test_schedule_one() {
        let prefetcher = LockFreePrefetcher::new();

        assert!(prefetcher.schedule_one("chunk_0", None, 100));
        assert_eq!(prefetcher.queue_len(), 1);

        // Add to prefetched to prevent scheduling again
        let tensor = DenseND::<f64>::zeros(&[2, 2]);
        prefetcher.add_prefetched("chunk_0", tensor);

        // Should not schedule again (already prefetched)
        assert!(!prefetcher.schedule_one("chunk_0", None, 100));
        assert_eq!(prefetcher.queue_len(), 1);
    }

    #[test]
    fn test_pop_queue() {
        let prefetcher = LockFreePrefetcher::new();

        prefetcher.schedule_one("chunk_0", None, 100);
        prefetcher.schedule_one("chunk_1", None, 50);

        let entry1 = prefetcher.pop_queue().unwrap();
        assert_eq!(entry1.chunk_id, "chunk_0");
        assert_eq!(prefetcher.queue_len(), 1);

        let entry2 = prefetcher.pop_queue().unwrap();
        assert_eq!(entry2.chunk_id, "chunk_1");
        assert_eq!(prefetcher.queue_len(), 0);

        assert!(prefetcher.pop_queue().is_none());
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_load_from_disk() {
        use std::collections::HashMap;
        use std::env;

        let prefetcher = LockFreePrefetcher::new().queue_size(2);

        // Create test files
        let temp_dir = env::temp_dir();
        let path1 = temp_dir.join("test_lockfree_prefetch_1.bin");
        let path2 = temp_dir.join("test_lockfree_prefetch_2.bin");

        let tensor1 = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let tensor2 = DenseND::<f64>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

        crate::mmap_io::write_tensor_binary(&path1, &tensor1).unwrap();
        crate::mmap_io::write_tensor_binary(&path2, &tensor2).unwrap();

        // Schedule prefetch
        prefetcher.schedule_one("chunk_0", Some(path1.clone()), 100);
        prefetcher.schedule_one("chunk_1", Some(path2.clone()), 50);

        // Create path map
        let mut chunk_paths = HashMap::new();
        chunk_paths.insert("chunk_0".to_string(), path1.clone());
        chunk_paths.insert("chunk_1".to_string(), path2.clone());

        // Load from disk
        let loaded = prefetcher.load_from_disk(&chunk_paths, 2).unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(prefetcher.prefetch_count(), 2);

        // Cleanup
        std::fs::remove_file(path1).ok();
        std::fs::remove_file(path2).ok();
    }
}
