//! Plan caching for memoization and performance optimization
//!
//! This module provides thread-safe caches with multiple eviction policies for storing
//! and retrieving previously computed contraction plans. Caching is particularly useful when:
//! - The same einsum patterns are used repeatedly
//! - Shapes are similar or identical across multiple executions
//! - Planning overhead is significant (e.g., with DP or GA planners)
//!
//! # Eviction Policies
//!
//! - **LRU (Least Recently Used)**: Evicts entries based on access recency
//! - **LFU (Least Frequently Used)**: Evicts entries based on access frequency
//! - **ARC (Adaptive Replacement Cache)**: Adapts between recency and frequency
//!
//! # Examples
//!
//! ```
//! use tenrso_planner::{PlanCache, greedy_planner, EinsumSpec, PlanHints};
//!
//! // Default: LRU policy
//! let mut cache = PlanCache::new_lru(100);
//!
//! let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
//! let shapes = vec![vec![100, 200], vec![200, 300]];
//! let hints = PlanHints::default();
//!
//! // First call: plan is computed
//! let plan1 = cache.get_or_compute(&spec, &shapes, &hints, || {
//!     greedy_planner(&spec, &shapes, &hints)
//! }).unwrap();
//!
//! // Second call: plan is retrieved from cache (fast!)
//! let plan2 = cache.get_or_compute(&spec, &shapes, &hints, || {
//!     greedy_planner(&spec, &shapes, &hints)
//! }).unwrap();
//!
//! assert_eq!(plan1.estimated_flops, plan2.estimated_flops);
//! println!("Cache hit rate: {:.1}%", cache.hit_rate() * 100.0);
//! ```
//!
//! # Choosing an Eviction Policy
//!
//! ```
//! use tenrso_planner::PlanCache;
//!
//! // LRU: Good for temporal locality (recent patterns are reused)
//! let lru_cache = PlanCache::new_lru(100);
//!
//! // LFU: Good for frequency-based workloads (some patterns used often)
//! let lfu_cache = PlanCache::new_lfu(100);
//!
//! // ARC: Adaptive, balances recency and frequency automatically
//! let arc_cache = PlanCache::new_arc(100);
//! ```

use crate::api::{Plan, PlanHints};
use crate::parser::EinsumSpec;
use anyhow::Result;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used - evicts based on access recency
    LRU,
    /// Least Frequently Used - evicts based on access frequency
    LFU,
    /// Adaptive Replacement Cache - adapts between recency and frequency
    ARC,
}

/// A cache key that uniquely identifies a planning problem
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    /// The einsum specification as a string
    spec: String,
    /// Tensor shapes
    shapes: Vec<Vec<usize>>,
    /// Relevant hint values that affect planning
    memory_budget: usize,
    minimize_memory: bool,
    allow_ooc: bool,
}

impl CacheKey {
    /// Create a new cache key from planning inputs
    fn new(spec: &EinsumSpec, shapes: &[Vec<usize>], hints: &PlanHints) -> Self {
        // Create a unique string representation of the einsum spec
        let spec_str = format!("{}->{}", spec.inputs.join(","), spec.output);

        Self {
            spec: spec_str,
            shapes: shapes.to_vec(),
            memory_budget: hints.memory_budget.unwrap_or(0),
            minimize_memory: hints.minimize_memory,
            allow_ooc: hints.allow_ooc,
        }
    }
}

/// Statistics for cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of cache evictions
    pub evictions: usize,
    /// Current number of cached plans
    pub entries: usize,
}

impl CacheStats {
    /// Calculate the cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// An entry in the LRU cache
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached plan
    plan: Plan,
    /// Access counter for LRU eviction
    access_count: usize,
    /// Last access timestamp (logical)
    last_access: usize,
}

/// Thread-safe LRU cache for contraction plans
///
/// # Design
///
/// - **Key**: `(einsum_spec, shapes, hints)` - uniquely identifies a planning problem
/// - **Value**: Computed `Plan` with metadata
/// - **Eviction**: LRU (Least Recently Used) when capacity is exceeded
/// - **Thread Safety**: Uses `Arc<Mutex<...>>` for safe concurrent access
///
/// # Performance
///
/// - **Get**: O(1) average case (HashMap lookup)
/// - **Insert**: O(1) amortized (may trigger eviction)
/// - **Eviction**: O(n) worst case (scans all entries for LRU)
///
/// # Examples
///
/// ```
/// use tenrso_planner::{PlanCache, greedy_planner, EinsumSpec, PlanHints};
///
/// let cache = PlanCache::new(50);
///
/// let spec = EinsumSpec::parse("ab,bc->ac").unwrap();
/// let shapes = vec![vec![10, 20], vec![20, 30]];
/// let hints = PlanHints::default();
///
/// // Cache miss: compute plan
/// let plan = cache.get_or_compute(&spec, &shapes, &hints, || {
///     greedy_planner(&spec, &shapes, &hints)
/// }).unwrap();
///
/// // Cache hit: retrieve plan
/// let cached = cache.get_or_compute(&spec, &shapes, &hints, || {
///     panic!("Should not be called!")
/// }).unwrap();
/// ```
#[derive(Clone)]
pub struct PlanCache {
    /// The actual cache storage
    inner: Arc<Mutex<PlanCacheInner>>,
}

struct PlanCacheInner {
    /// Maximum number of cached plans
    capacity: usize,
    /// The cache storage
    cache: HashMap<CacheKey, CacheEntry>,
    /// Current logical timestamp for LRU/ARC
    current_time: usize,
    /// Statistics
    stats: CacheStats,
    /// Eviction policy
    policy: EvictionPolicy,
    /// ARC-specific: target size for T1 (recently used once)
    arc_p: usize,
    /// ARC-specific: ghost entries for recently evicted from T1
    arc_b1: HashMap<CacheKey, usize>,
    /// ARC-specific: ghost entries for recently evicted from T2
    arc_b2: HashMap<CacheKey, usize>,
}

impl PlanCache {
    /// Create a new plan cache with the given capacity (defaults to LRU policy)
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of plans to cache (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::PlanCache;
    ///
    /// let cache = PlanCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self::new_lru(capacity)
    }

    /// Create a new plan cache with LRU eviction policy
    ///
    /// LRU (Least Recently Used) is good for workloads with temporal locality,
    /// where recently accessed items are likely to be accessed again.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::PlanCache;
    ///
    /// let cache = PlanCache::new_lru(100);
    /// ```
    pub fn new_lru(capacity: usize) -> Self {
        assert!(capacity > 0, "Cache capacity must be greater than 0");
        Self {
            inner: Arc::new(Mutex::new(PlanCacheInner {
                capacity,
                cache: HashMap::new(),
                current_time: 0,
                stats: CacheStats::default(),
                policy: EvictionPolicy::LRU,
                arc_p: 0,
                arc_b1: HashMap::new(),
                arc_b2: HashMap::new(),
            })),
        }
    }

    /// Create a new plan cache with LFU eviction policy
    ///
    /// LFU (Least Frequently Used) is good for workloads where some patterns
    /// are accessed much more frequently than others.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::PlanCache;
    ///
    /// let cache = PlanCache::new_lfu(100);
    /// ```
    pub fn new_lfu(capacity: usize) -> Self {
        assert!(capacity > 0, "Cache capacity must be greater than 0");
        Self {
            inner: Arc::new(Mutex::new(PlanCacheInner {
                capacity,
                cache: HashMap::new(),
                current_time: 0,
                stats: CacheStats::default(),
                policy: EvictionPolicy::LFU,
                arc_p: 0,
                arc_b1: HashMap::new(),
                arc_b2: HashMap::new(),
            })),
        }
    }

    /// Create a new plan cache with ARC eviction policy
    ///
    /// ARC (Adaptive Replacement Cache) adapts between recency and frequency,
    /// providing robust performance across different workload patterns.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::PlanCache;
    ///
    /// let cache = PlanCache::new_arc(100);
    /// ```
    pub fn new_arc(capacity: usize) -> Self {
        assert!(capacity > 0, "Cache capacity must be greater than 0");
        Self {
            inner: Arc::new(Mutex::new(PlanCacheInner {
                capacity,
                cache: HashMap::new(),
                current_time: 0,
                stats: CacheStats::default(),
                policy: EvictionPolicy::ARC,
                arc_p: 0,
                arc_b1: HashMap::new(),
                arc_b2: HashMap::new(),
            })),
        }
    }

    /// Get the eviction policy used by this cache
    pub fn policy(&self) -> EvictionPolicy {
        let inner = self.inner.lock().unwrap();
        inner.policy
    }

    /// Get a cached plan or compute it using the provided function
    ///
    /// This is the main entry point for using the cache. It will:
    /// 1. Check if a plan exists for the given inputs
    /// 2. If yes, return the cached plan (cache hit)
    /// 3. If no, call `compute_fn` to create a plan, cache it, and return it (cache miss)
    ///
    /// # Arguments
    ///
    /// * `spec` - The einsum specification
    /// * `shapes` - Tensor shapes
    /// * `hints` - Planning hints
    /// * `compute_fn` - Function to compute the plan if not cached
    ///
    /// # Returns
    ///
    /// The plan (either cached or freshly computed)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{PlanCache, greedy_planner, EinsumSpec, PlanHints};
    ///
    /// let cache = PlanCache::new(10);
    /// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    /// let shapes = vec![vec![5, 10], vec![10, 15]];
    /// let hints = PlanHints::default();
    ///
    /// let plan = cache.get_or_compute(&spec, &shapes, &hints, || {
    ///     greedy_planner(&spec, &shapes, &hints)
    /// }).unwrap();
    /// ```
    pub fn get_or_compute<F>(
        &self,
        spec: &EinsumSpec,
        shapes: &[Vec<usize>],
        hints: &PlanHints,
        compute_fn: F,
    ) -> Result<Plan>
    where
        F: FnOnce() -> Result<Plan>,
    {
        let key = CacheKey::new(spec, shapes, hints);

        // Try to get from cache
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(entry) = inner.cache.get(&key) {
                // Cache hit - clone plan and capture current time
                let plan = entry.plan.clone();
                let current_time = inner.current_time;

                // Now update the entry (requires get_mut, but entry is dropped)
                if let Some(entry) = inner.cache.get_mut(&key) {
                    entry.access_count += 1;
                    entry.last_access = current_time;
                }

                // Update stats
                inner.stats.hits += 1;
                inner.current_time += 1;

                return Ok(plan);
            }
            inner.stats.misses += 1;
        }

        // Cache miss: compute the plan
        let plan = compute_fn()?;

        // Insert into cache
        {
            let mut inner = self.inner.lock().unwrap();

            // For ARC, adapt based on ghost list hits
            if inner.policy == EvictionPolicy::ARC {
                inner.adapt_arc_on_hit(&key);
            }

            // Evict if at capacity
            if inner.cache.len() >= inner.capacity {
                inner.evict();
            }

            let entry = CacheEntry {
                plan: plan.clone(),
                access_count: 1,
                last_access: inner.current_time,
            };

            inner.cache.insert(key, entry);
            inner.current_time += 1;
            inner.stats.entries = inner.cache.len();
        }

        Ok(plan)
    }

    /// Get a cached plan without computing it
    ///
    /// Returns `Some(plan)` if the plan is cached, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{PlanCache, greedy_planner, EinsumSpec, PlanHints};
    ///
    /// let cache = PlanCache::new(10);
    /// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    /// let shapes = vec![vec![5, 10], vec![10, 15]];
    /// let hints = PlanHints::default();
    ///
    /// assert!(cache.get(&spec, &shapes, &hints).is_none());
    ///
    /// // Compute and cache
    /// cache.get_or_compute(&spec, &shapes, &hints, || {
    ///     greedy_planner(&spec, &shapes, &hints)
    /// }).unwrap();
    ///
    /// assert!(cache.get(&spec, &shapes, &hints).is_some());
    /// ```
    pub fn get(&self, spec: &EinsumSpec, shapes: &[Vec<usize>], hints: &PlanHints) -> Option<Plan> {
        let key = CacheKey::new(spec, shapes, hints);
        let mut inner = self.inner.lock().unwrap();

        if let Some(entry) = inner.cache.get(&key) {
            // Clone plan and capture current time
            let plan = entry.plan.clone();
            let current_time = inner.current_time;

            // Now update the entry (requires get_mut, but entry is dropped)
            if let Some(entry) = inner.cache.get_mut(&key) {
                entry.access_count += 1;
                entry.last_access = current_time;
            }

            // Update stats
            inner.stats.hits += 1;
            inner.current_time += 1;

            Some(plan)
        } else {
            None
        }
    }

    /// Manually insert a plan into the cache
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{PlanCache, greedy_planner, EinsumSpec, PlanHints};
    ///
    /// let cache = PlanCache::new(10);
    /// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    /// let shapes = vec![vec![5, 10], vec![10, 15]];
    /// let hints = PlanHints::default();
    ///
    /// let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
    /// cache.insert(&spec, &shapes, &hints, plan);
    ///
    /// assert!(cache.get(&spec, &shapes, &hints).is_some());
    /// ```
    pub fn insert(&self, spec: &EinsumSpec, shapes: &[Vec<usize>], hints: &PlanHints, plan: Plan) {
        let key = CacheKey::new(spec, shapes, hints);
        let mut inner = self.inner.lock().unwrap();

        // Evict if at capacity
        if inner.cache.len() >= inner.capacity {
            inner.evict();
        }

        let entry = CacheEntry {
            plan,
            access_count: 1,
            last_access: inner.current_time,
        };

        inner.cache.insert(key, entry);
        inner.current_time += 1;
        inner.stats.entries = inner.cache.len();
    }

    /// Clear all cached plans
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::PlanCache;
    ///
    /// let cache = PlanCache::new(10);
    /// // ... use cache ...
    /// cache.clear();
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.cache.clear();
        inner.stats.entries = 0;
    }

    /// Get the cache capacity
    pub fn capacity(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.capacity
    }

    /// Get the current number of cached plans
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get cache statistics
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{PlanCache, greedy_planner, EinsumSpec, PlanHints};
    ///
    /// let cache = PlanCache::new(10);
    /// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    /// let shapes = vec![vec![5, 10], vec![10, 15]];
    /// let hints = PlanHints::default();
    ///
    /// cache.get_or_compute(&spec, &shapes, &hints, || {
    ///     greedy_planner(&spec, &shapes, &hints)
    /// }).unwrap();
    ///
    /// let stats = cache.stats();
    /// assert_eq!(stats.misses, 1);
    /// assert_eq!(stats.entries, 1);
    /// ```
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().unwrap();
        inner.stats.clone()
    }

    /// Get the cache hit rate (0.0 to 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::{PlanCache, greedy_planner, EinsumSpec, PlanHints};
    ///
    /// let cache = PlanCache::new(10);
    /// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    /// let shapes = vec![vec![5, 10], vec![10, 15]];
    /// let hints = PlanHints::default();
    ///
    /// // First access: miss
    /// cache.get_or_compute(&spec, &shapes, &hints, || {
    ///     greedy_planner(&spec, &shapes, &hints)
    /// }).unwrap();
    ///
    /// // Second access: hit
    /// cache.get_or_compute(&spec, &shapes, &hints, || {
    ///     greedy_planner(&spec, &shapes, &hints)
    /// }).unwrap();
    ///
    /// assert_eq!(cache.hit_rate(), 0.5); // 1 hit out of 2 accesses
    /// ```
    pub fn hit_rate(&self) -> f64 {
        let inner = self.inner.lock().unwrap();
        inner.stats.hit_rate()
    }

    /// Reset cache statistics
    pub fn reset_stats(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.stats = CacheStats {
            entries: inner.cache.len(),
            ..Default::default()
        };
    }
}

impl PlanCacheInner {
    /// Evict an entry according to the current eviction policy
    fn evict(&mut self) {
        match self.policy {
            EvictionPolicy::LRU => self.evict_lru(),
            EvictionPolicy::LFU => self.evict_lfu(),
            EvictionPolicy::ARC => self.evict_arc(),
        }
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        // Find the LRU entry (smallest last_access time)
        let mut lru_key: Option<CacheKey> = None;
        let mut lru_time = usize::MAX;

        for (key, entry) in &self.cache {
            if entry.last_access < lru_time {
                lru_time = entry.last_access;
                lru_key = Some(key.clone());
            }
        }

        if let Some(key) = lru_key {
            self.cache.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Evict the least frequently used entry
    fn evict_lfu(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        // Find the LFU entry (smallest access_count)
        // Break ties with LRU (smallest last_access)
        let mut lfu_key: Option<CacheKey> = None;
        let mut lfu_count = usize::MAX;
        let mut lfu_time = usize::MAX;

        for (key, entry) in &self.cache {
            if entry.access_count < lfu_count
                || (entry.access_count == lfu_count && entry.last_access < lfu_time)
            {
                lfu_count = entry.access_count;
                lfu_time = entry.last_access;
                lfu_key = Some(key.clone());
            }
        }

        if let Some(key) = lfu_key {
            self.cache.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Evict an entry using ARC (Adaptive Replacement Cache) policy
    ///
    /// ARC maintains four lists:
    /// - T1: recently seen once (size ≤ p)
    /// - T2: recently seen at least twice (size ≤ c-p)
    /// - B1: ghost entries recently evicted from T1
    /// - B2: ghost entries recently evicted from T2
    ///
    /// The parameter p is adapted based on the workload.
    fn evict_arc(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        // Classify entries into T1 (frequency = 1) and T2 (frequency > 1)
        let mut t1_keys: Vec<(CacheKey, usize)> = Vec::new();
        let mut t2_keys: Vec<(CacheKey, usize)> = Vec::new();

        for (key, entry) in &self.cache {
            if entry.access_count == 1 {
                t1_keys.push((key.clone(), entry.last_access));
            } else {
                t2_keys.push((key.clone(), entry.last_access));
            }
        }

        // Sort by last_access (oldest first)
        t1_keys.sort_by_key(|(_, time)| *time);
        t2_keys.sort_by_key(|(_, time)| *time);

        // Evict from T1 if |T1| > p, otherwise from T2
        let evict_from_t1 = t1_keys.len() > self.arc_p;

        let evicted_key = if evict_from_t1 && !t1_keys.is_empty() {
            t1_keys[0].0.clone()
        } else if !t2_keys.is_empty() {
            t2_keys[0].0.clone()
        } else if !t1_keys.is_empty() {
            t1_keys[0].0.clone()
        } else {
            return;
        };

        // Add to appropriate ghost list
        if evict_from_t1 {
            self.arc_b1.insert(evicted_key.clone(), self.current_time);
            // Limit ghost list size
            if self.arc_b1.len() > self.capacity {
                // Remove oldest from B1
                if let Some(oldest) = self
                    .arc_b1
                    .iter()
                    .min_by_key(|(_, time)| *time)
                    .map(|(k, _)| k.clone())
                {
                    self.arc_b1.remove(&oldest);
                }
            }
        } else {
            self.arc_b2.insert(evicted_key.clone(), self.current_time);
            // Limit ghost list size
            if self.arc_b2.len() > self.capacity {
                // Remove oldest from B2
                if let Some(oldest) = self
                    .arc_b2
                    .iter()
                    .min_by_key(|(_, time)| *time)
                    .map(|(k, _)| k.clone())
                {
                    self.arc_b2.remove(&oldest);
                }
            }
        }

        self.cache.remove(&evicted_key);
        self.stats.evictions += 1;
    }

    /// Adapt ARC parameter p based on cache hits in ghost lists
    fn adapt_arc_on_hit(&mut self, key: &CacheKey) {
        // If hit in B1 (was evicted from T1), increase p (favor T1)
        if self.arc_b1.contains_key(key) {
            self.arc_p = std::cmp::min(self.arc_p + 1, self.capacity);
            self.arc_b1.remove(key);
        }
        // If hit in B2 (was evicted from T2), decrease p (favor T2)
        else if self.arc_b2.contains_key(key) {
            self.arc_p = self.arc_p.saturating_sub(1);
            self.arc_b2.remove(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{greedy_planner, PlanHints};

    #[test]
    fn test_cache_basic() {
        let cache = PlanCache::new(10);
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];
        let hints = PlanHints::default();

        // First call: miss
        let plan1 = cache
            .get_or_compute(&spec, &shapes, &hints, || {
                greedy_planner(&spec, &shapes, &hints)
            })
            .unwrap();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Second call: hit
        let plan2 = cache
            .get_or_compute(&spec, &shapes, &hints, || {
                panic!("Should not compute again!");
            })
            .unwrap();

        assert_eq!(plan1.estimated_flops, plan2.estimated_flops);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_capacity() {
        let cache = PlanCache::new(2);
        let hints = PlanHints::default();

        // Insert 3 different plans (should evict the first one)
        let specs = [
            ("ij,jk->ik", vec![vec![5, 10], vec![10, 15]]),
            ("ab,bc->ac", vec![vec![5, 10], vec![10, 15]]),
            ("xy,yz->xz", vec![vec![5, 10], vec![10, 15]]),
        ];

        for (spec_str, shapes) in &specs {
            let spec = EinsumSpec::parse(spec_str).unwrap();
            cache
                .get_or_compute(&spec, shapes, &hints, || {
                    greedy_planner(&spec, shapes, &hints)
                })
                .unwrap();
        }

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = PlanCache::new(2);
        let hints = PlanHints::default();

        let spec1 = EinsumSpec::parse("ij,jk->ik").unwrap();
        let spec2 = EinsumSpec::parse("ab,bc->ac").unwrap();
        let spec3 = EinsumSpec::parse("xy,yz->xz").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        // Insert plan1 and plan2
        cache
            .get_or_compute(&spec1, &shapes, &hints, || {
                greedy_planner(&spec1, &shapes, &hints)
            })
            .unwrap();
        cache
            .get_or_compute(&spec2, &shapes, &hints, || {
                greedy_planner(&spec2, &shapes, &hints)
            })
            .unwrap();

        // Access plan1 again (refreshes LRU)
        cache
            .get_or_compute(&spec1, &shapes, &hints, || {
                panic!("Should be cached!");
            })
            .unwrap();

        // Insert plan3 (should evict plan2, not plan1)
        cache
            .get_or_compute(&spec3, &shapes, &hints, || {
                greedy_planner(&spec3, &shapes, &hints)
            })
            .unwrap();

        // plan1 should still be cached
        assert!(cache.get(&spec1, &shapes, &hints).is_some());
        // plan2 should be evicted
        assert!(cache.get(&spec2, &shapes, &hints).is_none());
        // plan3 should be cached
        assert!(cache.get(&spec3, &shapes, &hints).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let cache = PlanCache::new(10);
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];
        let hints = PlanHints::default();

        cache
            .get_or_compute(&spec, &shapes, &hints, || {
                greedy_planner(&spec, &shapes, &hints)
            })
            .unwrap();

        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit_rate() {
        let cache = PlanCache::new(10);
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];
        let hints = PlanHints::default();

        // 1 miss
        cache
            .get_or_compute(&spec, &shapes, &hints, || {
                greedy_planner(&spec, &shapes, &hints)
            })
            .unwrap();

        // 3 hits
        for _ in 0..3 {
            cache
                .get_or_compute(&spec, &shapes, &hints, || {
                    panic!("Should be cached!");
                })
                .unwrap();
        }

        assert_eq!(cache.hit_rate(), 0.75); // 3 hits out of 4 accesses
    }

    #[test]
    fn test_cache_insert() {
        let cache = PlanCache::new(10);
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        cache.insert(&spec, &shapes, &hints, plan.clone());

        let cached = cache.get(&spec, &shapes, &hints).unwrap();
        assert_eq!(cached.estimated_flops, plan.estimated_flops);
    }

    #[test]
    fn test_cache_different_shapes() {
        let cache = PlanCache::new(10);
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let hints = PlanHints::default();

        let shapes1 = vec![vec![5, 10], vec![10, 15]];
        let shapes2 = vec![vec![10, 20], vec![20, 30]];

        // Both should be cached separately
        cache
            .get_or_compute(&spec, &shapes1, &hints, || {
                greedy_planner(&spec, &shapes1, &hints)
            })
            .unwrap();
        cache
            .get_or_compute(&spec, &shapes2, &hints, || {
                greedy_planner(&spec, &shapes2, &hints)
            })
            .unwrap();

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_different_hints() {
        let cache = PlanCache::new(10);
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        let hints1 = PlanHints {
            minimize_memory: true,
            ..Default::default()
        };
        let hints2 = PlanHints {
            minimize_memory: false,
            ..Default::default()
        };

        // Both should be cached separately (different hints)
        cache
            .get_or_compute(&spec, &shapes, &hints1, || {
                greedy_planner(&spec, &shapes, &hints1)
            })
            .unwrap();
        cache
            .get_or_compute(&spec, &shapes, &hints2, || {
                greedy_planner(&spec, &shapes, &hints2)
            })
            .unwrap();

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_thread_safety() {
        use std::thread;

        let cache = PlanCache::new(100);
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];
        let hints = PlanHints::default();

        // Spawn multiple threads accessing the cache
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let cache = cache.clone();
                let spec = spec.clone();
                let shapes = shapes.clone();
                let hints = hints.clone();

                thread::spawn(move || {
                    for _ in 0..10 {
                        cache
                            .get_or_compute(&spec, &shapes, &hints, || {
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

        // Should have high hit rate due to concurrent access
        assert!(cache.hit_rate() > 0.9);
    }

    // ========== LFU Policy Tests ==========

    #[test]
    fn test_lfu_eviction_policy() {
        let cache = PlanCache::new_lfu(2);
        assert_eq!(cache.policy(), EvictionPolicy::LFU);

        let hints = PlanHints::default();
        let spec1 = EinsumSpec::parse("ij,jk->ik").unwrap();
        let spec2 = EinsumSpec::parse("ab,bc->ac").unwrap();
        let spec3 = EinsumSpec::parse("xy,yz->xz").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        // Insert spec1
        cache
            .get_or_compute(&spec1, &shapes, &hints, || {
                greedy_planner(&spec1, &shapes, &hints)
            })
            .unwrap();

        // Insert spec2
        cache
            .get_or_compute(&spec2, &shapes, &hints, || {
                greedy_planner(&spec2, &shapes, &hints)
            })
            .unwrap();

        // Access spec1 multiple times (increase frequency)
        for _ in 0..3 {
            cache.get(&spec1, &shapes, &hints).unwrap();
        }

        // Insert spec3 (should evict spec2, which has lower frequency)
        cache
            .get_or_compute(&spec3, &shapes, &hints, || {
                greedy_planner(&spec3, &shapes, &hints)
            })
            .unwrap();

        // spec1 should still be cached (higher frequency)
        assert!(cache.get(&spec1, &shapes, &hints).is_some());
        // spec2 should be evicted (lower frequency)
        assert!(cache.get(&spec2, &shapes, &hints).is_none());
        // spec3 should be cached
        assert!(cache.get(&spec3, &shapes, &hints).is_some());
    }

    #[test]
    fn test_lfu_tie_breaking() {
        let cache = PlanCache::new_lfu(2);
        let hints = PlanHints::default();

        let spec1 = EinsumSpec::parse("ij,jk->ik").unwrap();
        let spec2 = EinsumSpec::parse("ab,bc->ac").unwrap();
        let spec3 = EinsumSpec::parse("xy,yz->xz").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        // Insert spec1 and spec2 (both with frequency 1)
        cache
            .get_or_compute(&spec1, &shapes, &hints, || {
                greedy_planner(&spec1, &shapes, &hints)
            })
            .unwrap();

        cache
            .get_or_compute(&spec2, &shapes, &hints, || {
                greedy_planner(&spec2, &shapes, &hints)
            })
            .unwrap();

        // Insert spec3 (should evict spec1, which is older)
        cache
            .get_or_compute(&spec3, &shapes, &hints, || {
                greedy_planner(&spec3, &shapes, &hints)
            })
            .unwrap();

        // spec1 should be evicted (LRU tie-breaking)
        assert!(cache.get(&spec1, &shapes, &hints).is_none());
        // spec2 and spec3 should be cached
        assert!(cache.get(&spec2, &shapes, &hints).is_some());
        assert!(cache.get(&spec3, &shapes, &hints).is_some());
    }

    // ========== ARC Policy Tests ==========

    #[test]
    fn test_arc_eviction_policy() {
        let cache = PlanCache::new_arc(3);
        assert_eq!(cache.policy(), EvictionPolicy::ARC);

        let hints = PlanHints::default();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        // Insert 4 different plans to test eviction
        let specs = ["ij,jk->ik", "ab,bc->ac", "xy,yz->xz", "pq,qr->pr"];

        for spec_str in &specs {
            let spec = EinsumSpec::parse(spec_str).unwrap();
            cache
                .get_or_compute(&spec, &shapes, &hints, || {
                    greedy_planner(&spec, &shapes, &hints)
                })
                .unwrap();
        }

        // Should have evicted one entry
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_arc_adaptation() {
        let cache = PlanCache::new_arc(2);
        let hints = PlanHints::default();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        let spec1 = EinsumSpec::parse("ij,jk->ik").unwrap();
        let spec2 = EinsumSpec::parse("ab,bc->ac").unwrap();
        let spec3 = EinsumSpec::parse("xy,yz->xz").unwrap();

        // Insert spec1 (frequency 1)
        cache
            .get_or_compute(&spec1, &shapes, &hints, || {
                greedy_planner(&spec1, &shapes, &hints)
            })
            .unwrap();

        // Insert spec2 (frequency 1)
        cache
            .get_or_compute(&spec2, &shapes, &hints, || {
                greedy_planner(&spec2, &shapes, &hints)
            })
            .unwrap();

        // Access spec1 again (frequency 2)
        cache.get(&spec1, &shapes, &hints).unwrap();

        // Insert spec3 (should evict spec2, which has frequency 1)
        cache
            .get_or_compute(&spec3, &shapes, &hints, || {
                greedy_planner(&spec3, &shapes, &hints)
            })
            .unwrap();

        // spec1 should still be cached (frequency 2)
        assert!(cache.get(&spec1, &shapes, &hints).is_some());
        // spec2 should be evicted
        assert!(cache.get(&spec2, &shapes, &hints).is_none());
    }

    #[test]
    fn test_arc_ghost_lists() {
        let cache = PlanCache::new_arc(2);
        let hints = PlanHints::default();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        let spec1 = EinsumSpec::parse("ij,jk->ik").unwrap();
        let spec2 = EinsumSpec::parse("ab,bc->ac").unwrap();
        let spec3 = EinsumSpec::parse("xy,yz->xz").unwrap();

        // Fill cache
        cache
            .get_or_compute(&spec1, &shapes, &hints, || {
                greedy_planner(&spec1, &shapes, &hints)
            })
            .unwrap();
        cache
            .get_or_compute(&spec2, &shapes, &hints, || {
                greedy_planner(&spec2, &shapes, &hints)
            })
            .unwrap();

        // Insert spec3 (evicts one of the previous)
        cache
            .get_or_compute(&spec3, &shapes, &hints, || {
                greedy_planner(&spec3, &shapes, &hints)
            })
            .unwrap();

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
    }

    // ========== Policy Comparison Tests ==========

    #[test]
    fn test_lru_vs_lfu_behavior() {
        let lru_cache = PlanCache::new_lru(2);
        let lfu_cache = PlanCache::new_lfu(2);
        let hints = PlanHints::default();
        let shapes = vec![vec![5, 10], vec![10, 15]];

        let spec1 = EinsumSpec::parse("ij,jk->ik").unwrap();
        let spec2 = EinsumSpec::parse("ab,bc->ac").unwrap();
        let spec3 = EinsumSpec::parse("xy,yz->xz").unwrap();

        // Setup: Insert spec1 and spec2, access spec1 multiple times
        for cache in [&lru_cache, &lfu_cache] {
            cache
                .get_or_compute(&spec1, &shapes, &hints, || {
                    greedy_planner(&spec1, &shapes, &hints)
                })
                .unwrap();
            cache
                .get_or_compute(&spec2, &shapes, &hints, || {
                    greedy_planner(&spec2, &shapes, &hints)
                })
                .unwrap();

            // Access spec1 multiple times (higher frequency)
            for _ in 0..3 {
                cache.get(&spec1, &shapes, &hints).unwrap();
            }
        }

        // Access spec2 once more (making it most recent for LRU)
        lru_cache.get(&spec2, &shapes, &hints).unwrap();
        lfu_cache.get(&spec2, &shapes, &hints).unwrap();

        // Insert spec3
        lru_cache
            .get_or_compute(&spec3, &shapes, &hints, || {
                greedy_planner(&spec3, &shapes, &hints)
            })
            .unwrap();
        lfu_cache
            .get_or_compute(&spec3, &shapes, &hints, || {
                greedy_planner(&spec3, &shapes, &hints)
            })
            .unwrap();

        // LRU: evicts spec1 (least recent)
        assert!(lru_cache.get(&spec1, &shapes, &hints).is_none());
        assert!(lru_cache.get(&spec2, &shapes, &hints).is_some());

        // LFU: evicts spec2 (least frequent)
        assert!(lfu_cache.get(&spec1, &shapes, &hints).is_some());
        assert!(lfu_cache.get(&spec2, &shapes, &hints).is_none());
    }

    #[test]
    fn test_policy_constructors() {
        let lru = PlanCache::new_lru(10);
        let lfu = PlanCache::new_lfu(10);
        let arc = PlanCache::new_arc(10);

        assert_eq!(lru.policy(), EvictionPolicy::LRU);
        assert_eq!(lfu.policy(), EvictionPolicy::LFU);
        assert_eq!(arc.policy(), EvictionPolicy::ARC);
    }

    #[test]
    fn test_default_policy_is_lru() {
        let cache = PlanCache::new(10);
        assert_eq!(cache.policy(), EvictionPolicy::LRU);
    }
}
