//! Hierarchical memory tiers for advanced out-of-core processing
//!
//! This module provides multi-tier memory management where data can be moved between:
//! - Tier 0 (RAM): Fast, limited capacity
//! - Tier 1 (SSD/Fast disk): Medium speed, medium capacity
//! - Tier 2 (HDD/Slow disk): Slow speed, large capacity
//!
//! # Features
//!
//! - Automatic data migration between tiers based on access patterns
//! - Cost-aware promotion/demotion decisions
//! - Per-tier capacity management
//! - Working set estimation
//! - Adaptive threshold tuning
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::memory_tiers::{TieredMemoryManager, TierConfig};
//!
//! let mut manager = TieredMemoryManager::new()
//!     .tier_config(TierConfig {
//!         ram_mb: 1024,        // 1GB RAM
//!         ssd_mb: 10240,       // 10GB SSD
//!         disk_mb: 102400,     // 100GB Disk
//!     })
//!     .promotion_threshold(0.7)
//!     .demotion_threshold(0.9);
//!
//! // Manager automatically moves data between tiers
//! manager.register_chunk("chunk_0", tensor, access_pattern)?;
//! let chunk = manager.get_chunk("chunk_0")?;
//! ```

use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tenrso_core::DenseND;

#[cfg(feature = "compression")]
use crate::compression::{compress_f64_slice, decompress_to_f64_vec, CompressionCodec};

/// Memory tier level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryTier {
    /// Tier 0: Main memory (RAM)
    Ram = 0,
    /// Tier 1: Fast storage (SSD)
    Ssd = 1,
    /// Tier 2: Slow storage (HDD)
    Disk = 2,
}

impl MemoryTier {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            MemoryTier::Ram => "RAM",
            MemoryTier::Ssd => "SSD",
            MemoryTier::Disk => "Disk",
        }
    }

    /// Get typical access latency in microseconds
    pub fn latency_us(&self) -> u64 {
        match self {
            MemoryTier::Ram => 1,      // ~1 µs
            MemoryTier::Ssd => 100,    // ~100 µs
            MemoryTier::Disk => 10000, // ~10 ms
        }
    }

    /// Get typical bandwidth in MB/s
    pub fn bandwidth_mbps(&self) -> u64 {
        match self {
            MemoryTier::Ram => 100_000, // ~100 GB/s
            MemoryTier::Ssd => 3_000,   // ~3 GB/s
            MemoryTier::Disk => 200,    // ~200 MB/s
        }
    }
}

/// Configuration for memory tiers
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// RAM capacity in MB
    pub ram_mb: usize,
    /// SSD capacity in MB (0 to disable)
    pub ssd_mb: usize,
    /// Disk capacity in MB (0 to disable)
    pub disk_mb: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            ram_mb: 1024,    // 1 GB
            ssd_mb: 10240,   // 10 GB
            disk_mb: 102400, // 100 GB
        }
    }
}

/// Access pattern for working set prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierAccessPattern {
    /// Sequential access (streaming)
    Sequential,
    /// Random access
    Random,
    /// Strided access
    Strided,
    /// Temporal locality (recently accessed)
    Temporal,
}

/// Chunk metadata with tier information
#[derive(Debug, Clone)]
struct ChunkMetadata {
    id: String,
    size_bytes: usize,
    shape: Vec<usize>,
    current_tier: MemoryTier,
    access_count: usize,
    last_access: SystemTime,
    #[allow(dead_code)]
    access_interval_avg: Duration,
    access_pattern: TierAccessPattern,
    promotion_score: f64,
}

impl ChunkMetadata {
    /// Calculate promotion score (higher = more likely to promote)
    fn calculate_promotion_score(&self, now: SystemTime) -> f64 {
        let time_since_access = now
            .duration_since(self.last_access)
            .unwrap_or(Duration::from_secs(0))
            .as_secs_f64();

        // Factors:
        // 1. Recency: Recent accesses score higher
        // 2. Frequency: More accesses score higher
        // 3. Access pattern: Sequential/Temporal patterns score higher

        let recency_score = 1.0 / (1.0 + time_since_access);
        let frequency_score = (self.access_count as f64).ln().max(0.0);
        let pattern_score = match self.access_pattern {
            TierAccessPattern::Temporal => 2.0,
            TierAccessPattern::Sequential => 1.5,
            TierAccessPattern::Strided => 1.0,
            TierAccessPattern::Random => 0.5,
        };

        recency_score * frequency_score * pattern_score
    }

    /// Calculate demotion score (higher = more likely to demote)
    fn calculate_demotion_score(&self, now: SystemTime) -> f64 {
        let time_since_access = now
            .duration_since(self.last_access)
            .unwrap_or(Duration::from_secs(0))
            .as_secs_f64();

        // Inverse of promotion score, weighted by size
        let age_score = time_since_access / 3600.0; // Age in hours
        let size_score = (self.size_bytes as f64).ln() / 20.0; // Favor demoting large chunks
        let infrequent_score = 1.0 / (1.0 + self.access_count as f64);

        age_score * (1.0 + size_score) * infrequent_score
    }
}

/// Per-tier statistics
#[derive(Debug, Clone)]
struct TierStats {
    capacity_bytes: usize,
    used_bytes: usize,
    chunk_count: usize,
    hit_count: usize,
    miss_count: usize,
}

impl TierStats {
    fn new(capacity_mb: usize) -> Self {
        Self {
            capacity_bytes: capacity_mb * 1024 * 1024,
            used_bytes: 0,
            chunk_count: 0,
            hit_count: 0,
            miss_count: 0,
        }
    }

    fn usage_ratio(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            return 0.0;
        }
        self.hit_count as f64 / total as f64
    }
}

/// Tiered memory manager
pub struct TieredMemoryManager {
    config: TierConfig,
    chunks: HashMap<String, ChunkMetadata>,
    ram_data: HashMap<String, DenseND<f64>>,
    tier_stats: HashMap<MemoryTier, TierStats>,
    tier_paths: HashMap<MemoryTier, PathBuf>,
    promotion_threshold: f64,
    demotion_threshold: f64,
    working_set_window: usize,
    recent_accesses: VecDeque<(String, SystemTime)>,
    enable_auto_migration: bool,

    #[cfg(feature = "compression")]
    compression: CompressionCodec,
}

impl TieredMemoryManager {
    /// Create a new tiered memory manager
    pub fn new() -> Self {
        let config = TierConfig::default();
        let mut tier_stats = HashMap::new();
        tier_stats.insert(MemoryTier::Ram, TierStats::new(config.ram_mb));
        tier_stats.insert(MemoryTier::Ssd, TierStats::new(config.ssd_mb));
        tier_stats.insert(MemoryTier::Disk, TierStats::new(config.disk_mb));

        let mut tier_paths = HashMap::new();
        let temp_dir = std::env::temp_dir();
        tier_paths.insert(MemoryTier::Ram, temp_dir.clone());
        tier_paths.insert(MemoryTier::Ssd, temp_dir.join("tenrso_ssd"));
        tier_paths.insert(MemoryTier::Disk, temp_dir.join("tenrso_disk"));

        // Create tier directories
        for path in tier_paths.values() {
            let _ = std::fs::create_dir_all(path);
        }

        Self {
            config,
            chunks: HashMap::new(),
            ram_data: HashMap::new(),
            tier_stats,
            tier_paths,
            promotion_threshold: 0.7,
            demotion_threshold: 0.9,
            working_set_window: 100,
            recent_accesses: VecDeque::new(),
            enable_auto_migration: true,

            #[cfg(feature = "compression")]
            compression: CompressionCodec::default(),
        }
    }

    /// Set tier configuration
    pub fn tier_config(mut self, config: TierConfig) -> Self {
        let ram_mb = config.ram_mb;
        let ssd_mb = config.ssd_mb;
        let disk_mb = config.disk_mb;

        self.config = config;
        self.tier_stats
            .insert(MemoryTier::Ram, TierStats::new(ram_mb));
        self.tier_stats
            .insert(MemoryTier::Ssd, TierStats::new(ssd_mb));
        self.tier_stats
            .insert(MemoryTier::Disk, TierStats::new(disk_mb));
        self
    }

    /// Set promotion threshold (0.0-1.0, tier usage ratio to trigger promotion)
    pub fn promotion_threshold(mut self, threshold: f64) -> Self {
        self.promotion_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set demotion threshold (0.0-1.0, tier usage ratio to trigger demotion)
    pub fn demotion_threshold(mut self, threshold: f64) -> Self {
        self.demotion_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set working set window size
    pub fn working_set_window(mut self, window: usize) -> Self {
        self.working_set_window = window;
        self
    }

    /// Enable or disable automatic migration
    pub fn auto_migration(mut self, enable: bool) -> Self {
        self.enable_auto_migration = enable;
        self
    }

    /// Set SSD tier path
    pub fn ssd_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let _ = std::fs::create_dir_all(&path);
        self.tier_paths.insert(MemoryTier::Ssd, path);
        self
    }

    /// Set disk tier path
    pub fn disk_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let _ = std::fs::create_dir_all(&path);
        self.tier_paths.insert(MemoryTier::Disk, path);
        self
    }

    /// Set compression codec
    #[cfg(feature = "compression")]
    pub fn compression(mut self, codec: CompressionCodec) -> Self {
        self.compression = codec;
        self
    }

    /// Register a new chunk
    pub fn register_chunk(
        &mut self,
        chunk_id: &str,
        tensor: DenseND<f64>,
        pattern: TierAccessPattern,
    ) -> Result<()> {
        let size_bytes = tensor.len() * std::mem::size_of::<f64>();
        let shape = tensor.shape().to_vec();

        // Start in RAM
        self.ram_data.insert(chunk_id.to_string(), tensor);

        let metadata = ChunkMetadata {
            id: chunk_id.to_string(),
            size_bytes,
            shape,
            current_tier: MemoryTier::Ram,
            access_count: 0,
            last_access: SystemTime::now(),
            access_interval_avg: Duration::from_secs(0),
            access_pattern: pattern,
            promotion_score: 0.0,
        };

        self.chunks.insert(chunk_id.to_string(), metadata);

        // Update tier stats
        if let Some(stats) = self.tier_stats.get_mut(&MemoryTier::Ram) {
            stats.used_bytes += size_bytes;
            stats.chunk_count += 1;
        }

        // Check if we need to demote
        if self.enable_auto_migration {
            self.check_and_migrate()?;
        }

        Ok(())
    }

    /// Get a chunk (with automatic promotion if needed)
    pub fn get_chunk(&mut self, chunk_id: &str) -> Result<DenseND<f64>> {
        let metadata = self
            .chunks
            .get_mut(chunk_id)
            .ok_or_else(|| anyhow!("Chunk {} not found", chunk_id))?;

        let now = SystemTime::now();
        let tier = metadata.current_tier;

        // Update access statistics
        metadata.access_count += 1;
        metadata.last_access = now;
        metadata.promotion_score = metadata.calculate_promotion_score(now);

        // Track recent access
        self.recent_accesses.push_back((chunk_id.to_string(), now));
        if self.recent_accesses.len() > self.working_set_window {
            self.recent_accesses.pop_front();
        }

        // Get the data
        let tensor = if tier == MemoryTier::Ram {
            // Hit in RAM
            if let Some(stats) = self.tier_stats.get_mut(&MemoryTier::Ram) {
                stats.hit_count += 1;
            }
            self.ram_data
                .get(chunk_id)
                .ok_or_else(|| anyhow!("Chunk {} not in RAM", chunk_id))?
                .clone()
        } else {
            // Miss in RAM, need to promote
            if let Some(stats) = self.tier_stats.get_mut(&MemoryTier::Ram) {
                stats.miss_count += 1;
            }
            self.promote_chunk(chunk_id)?
        };

        // Check if auto-migration is needed
        if self.enable_auto_migration {
            self.check_and_migrate()?;
        }

        Ok(tensor)
    }

    /// Promote a chunk to a higher tier
    fn promote_chunk(&mut self, chunk_id: &str) -> Result<DenseND<f64>> {
        let metadata = self
            .chunks
            .get(chunk_id)
            .ok_or_else(|| anyhow!("Chunk {} not found", chunk_id))?;

        let current_tier = metadata.current_tier;
        if current_tier == MemoryTier::Ram {
            return Err(anyhow!("Chunk {} already in RAM", chunk_id));
        }

        // Load from disk
        let tensor = self.load_from_tier(chunk_id, current_tier)?;

        // Update tier statistics
        let size_bytes = metadata.size_bytes;
        if let Some(stats) = self.tier_stats.get_mut(&current_tier) {
            stats.used_bytes = stats.used_bytes.saturating_sub(size_bytes);
            stats.chunk_count = stats.chunk_count.saturating_sub(1);
        }

        // Move to RAM
        self.ram_data.insert(chunk_id.to_string(), tensor.clone());
        if let Some(stats) = self.tier_stats.get_mut(&MemoryTier::Ram) {
            stats.used_bytes += size_bytes;
            stats.chunk_count += 1;
        }

        // Update metadata
        if let Some(meta) = self.chunks.get_mut(chunk_id) {
            meta.current_tier = MemoryTier::Ram;
        }

        Ok(tensor)
    }

    /// Demote a chunk to a lower tier
    fn demote_chunk(&mut self, chunk_id: &str, target_tier: MemoryTier) -> Result<()> {
        let metadata = self
            .chunks
            .get(chunk_id)
            .ok_or_else(|| anyhow!("Chunk {} not found", chunk_id))?;

        let current_tier = metadata.current_tier;
        if current_tier >= target_tier {
            return Err(anyhow!(
                "Cannot demote chunk {} from {:?} to {:?}",
                chunk_id,
                current_tier,
                target_tier
            ));
        }

        // Get tensor from current tier
        let tensor = if current_tier == MemoryTier::Ram {
            self.ram_data
                .remove(chunk_id)
                .ok_or_else(|| anyhow!("Chunk {} not in RAM", chunk_id))?
        } else {
            self.load_from_tier(chunk_id, current_tier)?
        };

        // Save to target tier
        self.save_to_tier(chunk_id, &tensor, target_tier)?;

        // Update tier statistics
        let size_bytes = metadata.size_bytes;
        if let Some(stats) = self.tier_stats.get_mut(&current_tier) {
            stats.used_bytes = stats.used_bytes.saturating_sub(size_bytes);
            stats.chunk_count = stats.chunk_count.saturating_sub(1);
        }
        if let Some(stats) = self.tier_stats.get_mut(&target_tier) {
            stats.used_bytes += size_bytes;
            stats.chunk_count += 1;
        }

        // Update metadata
        if let Some(meta) = self.chunks.get_mut(chunk_id) {
            meta.current_tier = target_tier;
        }

        Ok(())
    }

    /// Check and perform automatic migration
    fn check_and_migrate(&mut self) -> Result<()> {
        let now = SystemTime::now();

        // Check RAM usage
        if let Some(ram_stats) = self.tier_stats.get(&MemoryTier::Ram) {
            if ram_stats.usage_ratio() > self.demotion_threshold {
                // Need to demote from RAM
                self.demote_from_tier(MemoryTier::Ram, now)?;
            }
        }

        // Check SSD usage
        if let Some(ssd_stats) = self.tier_stats.get(&MemoryTier::Ssd) {
            if ssd_stats.usage_ratio() > self.demotion_threshold {
                // Need to demote from SSD to Disk
                self.demote_from_tier(MemoryTier::Ssd, now)?;
            }
        }

        // Check for promotion opportunities
        if let Some(ram_stats) = self.tier_stats.get(&MemoryTier::Ram) {
            if ram_stats.usage_ratio() < self.promotion_threshold {
                // Room in RAM, consider promotions
                self.promote_to_tier(MemoryTier::Ram, now)?;
            }
        }

        Ok(())
    }

    /// Demote chunks from a tier
    fn demote_from_tier(&mut self, tier: MemoryTier, now: SystemTime) -> Result<()> {
        let target_tier = match tier {
            MemoryTier::Ram => MemoryTier::Ssd,
            MemoryTier::Ssd => MemoryTier::Disk,
            MemoryTier::Disk => return Ok(()), // Cannot demote from lowest tier
        };

        // Find candidates for demotion
        let mut candidates: Vec<(String, f64)> = self
            .chunks
            .values()
            .filter(|m| m.current_tier == tier)
            .map(|m| (m.id.clone(), m.calculate_demotion_score(now)))
            .collect();

        // Sort by demotion score (highest first)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Calculate target usage
        let (capacity_bytes, promotion_threshold) = {
            let target_stats = self.tier_stats.get(&tier).unwrap();
            (target_stats.capacity_bytes, self.promotion_threshold)
        };
        let target_usage = (capacity_bytes as f64 * promotion_threshold) as usize;

        // Demote chunks until we're below threshold
        for (chunk_id, _score) in candidates {
            let current_usage = self.tier_stats.get(&tier).unwrap().used_bytes;
            if current_usage <= target_usage {
                break;
            }
            self.demote_chunk(&chunk_id, target_tier)?;
        }

        Ok(())
    }

    /// Promote chunks to a tier
    fn promote_to_tier(&mut self, tier: MemoryTier, now: SystemTime) -> Result<()> {
        if tier == MemoryTier::Disk {
            return Ok(()); // Cannot promote to lowest tier
        }

        let source_tier = match tier {
            MemoryTier::Ram => MemoryTier::Ssd,
            MemoryTier::Ssd => MemoryTier::Disk,
            MemoryTier::Disk => return Ok(()),
        };

        // Find candidates for promotion
        let mut candidates: Vec<(String, f64, usize)> = self
            .chunks
            .values()
            .filter(|m| m.current_tier == source_tier)
            .map(|m| (m.id.clone(), m.calculate_promotion_score(now), m.size_bytes))
            .collect();

        // Sort by promotion score (highest first)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Calculate available space
        let available_space = {
            let target_stats = self.tier_stats.get(&tier).unwrap();
            target_stats
                .capacity_bytes
                .saturating_sub(target_stats.used_bytes)
        };

        let mut promoted_bytes = 0;
        for (chunk_id, _score, size_bytes) in candidates {
            if promoted_bytes + size_bytes > available_space {
                break;
            }

            // For promotion to RAM, need special handling
            if tier == MemoryTier::Ram {
                let _ = self.promote_chunk(&chunk_id)?;
            }

            promoted_bytes += size_bytes;
        }

        Ok(())
    }

    /// Load tensor from a tier
    #[cfg(feature = "mmap")]
    fn load_from_tier(&self, chunk_id: &str, tier: MemoryTier) -> Result<DenseND<f64>> {
        let path = self.tier_paths.get(&tier).unwrap();
        let filename = format!("chunk_{}.bin", chunk_id);
        let full_path = path.join(filename);

        #[cfg(feature = "compression")]
        {
            let compressed = std::fs::read(&full_path)?;
            let decompressed = decompress_to_f64_vec(&compressed)?;
            let shape = self.chunks.get(chunk_id).unwrap().shape.clone();
            DenseND::from_vec(decompressed, &shape)
        }

        #[cfg(not(feature = "compression"))]
        {
            crate::mmap_io::read_tensor_binary(&full_path)
        }
    }

    /// Save tensor to a tier
    #[cfg(feature = "mmap")]
    fn save_to_tier(&self, chunk_id: &str, tensor: &DenseND<f64>, tier: MemoryTier) -> Result<()> {
        let path = self.tier_paths.get(&tier).unwrap();
        let filename = format!("chunk_{}.bin", chunk_id);
        let full_path = path.join(filename);

        #[cfg(feature = "compression")]
        {
            let data = tensor.as_slice();
            let compressed = compress_f64_slice(data, self.compression)?;
            std::fs::write(&full_path, compressed)?;
            Ok(())
        }

        #[cfg(not(feature = "compression"))]
        {
            crate::mmap_io::write_tensor_binary(&full_path, tensor)
        }
    }

    /// Placeholder when mmap is disabled
    #[cfg(not(feature = "mmap"))]
    fn load_from_tier(&self, _chunk_id: &str, _tier: MemoryTier) -> Result<DenseND<f64>> {
        Err(anyhow!("Tier operations require mmap feature"))
    }

    /// Placeholder when mmap is disabled
    #[cfg(not(feature = "mmap"))]
    fn save_to_tier(
        &self,
        _chunk_id: &str,
        _tensor: &DenseND<f64>,
        _tier: MemoryTier,
    ) -> Result<()> {
        Err(anyhow!("Tier operations require mmap feature"))
    }

    /// Get statistics for a specific tier
    pub fn tier_stats(&self, tier: MemoryTier) -> Option<TierStatistics> {
        self.tier_stats.get(&tier).map(|stats| TierStatistics {
            tier,
            capacity_bytes: stats.capacity_bytes,
            used_bytes: stats.used_bytes,
            chunk_count: stats.chunk_count,
            usage_ratio: stats.usage_ratio(),
            hit_count: stats.hit_count,
            miss_count: stats.miss_count,
            hit_rate: stats.hit_rate(),
        })
    }

    /// Get statistics for a specific chunk
    pub fn chunk_stats(&self, chunk_id: &str) -> Option<ChunkTierStats> {
        self.chunks.get(chunk_id).map(|meta| ChunkTierStats {
            chunk_id: chunk_id.to_string(),
            tier: meta.current_tier,
            size_bytes: meta.size_bytes,
            access_count: meta.access_count,
            last_access: meta.last_access,
        })
    }

    /// Get overall statistics
    pub fn overall_stats(&self) -> OverallStatistics {
        let total_chunks = self.chunks.len();
        let ram_chunks = self
            .chunks
            .values()
            .filter(|m| m.current_tier == MemoryTier::Ram)
            .count();
        let ssd_chunks = self
            .chunks
            .values()
            .filter(|m| m.current_tier == MemoryTier::Ssd)
            .count();
        let disk_chunks = self
            .chunks
            .values()
            .filter(|m| m.current_tier == MemoryTier::Disk)
            .count();

        OverallStatistics {
            total_chunks,
            ram_chunks,
            ssd_chunks,
            disk_chunks,
            working_set_size: self.estimate_working_set(),
        }
    }

    /// Estimate working set size
    fn estimate_working_set(&self) -> usize {
        self.recent_accesses.len()
    }

    /// Clean up all tier files
    pub fn cleanup(&mut self) -> Result<()> {
        for (tier, path) in &self.tier_paths {
            if *tier != MemoryTier::Ram && path.exists() {
                std::fs::remove_dir_all(path)?;
                std::fs::create_dir_all(path)?;
            }
        }
        Ok(())
    }
}

impl Default for TieredMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TieredMemoryManager {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// Statistics for a specific tier
#[derive(Debug, Clone)]
pub struct TierStatistics {
    pub tier: MemoryTier,
    pub capacity_bytes: usize,
    pub used_bytes: usize,
    pub chunk_count: usize,
    pub usage_ratio: f64,
    pub hit_count: usize,
    pub miss_count: usize,
    pub hit_rate: f64,
}

/// Overall statistics across all tiers
#[derive(Debug, Clone)]
pub struct OverallStatistics {
    pub total_chunks: usize,
    pub ram_chunks: usize,
    pub ssd_chunks: usize,
    pub disk_chunks: usize,
    pub working_set_size: usize,
}

/// Statistics for a specific chunk
#[derive(Debug, Clone)]
pub struct ChunkTierStats {
    pub chunk_id: String,
    pub tier: MemoryTier,
    pub size_bytes: usize,
    pub access_count: usize,
    pub last_access: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiered_manager_creation() {
        let manager = TieredMemoryManager::new();
        assert!(manager.enable_auto_migration);
    }

    #[test]
    fn test_tier_config() {
        let config = TierConfig {
            ram_mb: 512,
            ssd_mb: 2048,
            disk_mb: 10240,
        };
        let manager = TieredMemoryManager::new().tier_config(config.clone());
        assert_eq!(manager.config.ram_mb, 512);
    }

    #[test]
    fn test_register_chunk() {
        let mut manager = TieredMemoryManager::new();
        let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = manager.register_chunk("test_chunk", tensor, TierAccessPattern::Sequential);
        assert!(result.is_ok());
        assert_eq!(manager.chunks.len(), 1);
    }

    #[test]
    fn test_get_chunk() {
        let mut manager = TieredMemoryManager::new().auto_migration(false);
        let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        manager
            .register_chunk("test_chunk", tensor.clone(), TierAccessPattern::Sequential)
            .unwrap();
        let retrieved = manager.get_chunk("test_chunk").unwrap();

        assert_eq!(retrieved.as_slice(), tensor.as_slice());
    }

    #[test]
    fn test_tier_statistics() {
        let manager = TieredMemoryManager::new();
        let stats = manager.tier_stats(MemoryTier::Ram);
        assert!(stats.is_some());

        let ram_stats = stats.unwrap();
        assert_eq!(ram_stats.tier, MemoryTier::Ram);
        assert_eq!(ram_stats.usage_ratio, 0.0);
    }

    #[test]
    fn test_overall_statistics() {
        let mut manager = TieredMemoryManager::new();
        let tensor = DenseND::from_vec(vec![1.0; 100], &[10, 10]).unwrap();
        manager
            .register_chunk("chunk1", tensor.clone(), TierAccessPattern::Random)
            .unwrap();

        let stats = manager.overall_stats();
        assert_eq!(stats.total_chunks, 1);
        assert_eq!(stats.ram_chunks, 1);
    }

    #[test]
    fn test_memory_tier_ordering() {
        assert!(MemoryTier::Ram < MemoryTier::Ssd);
        assert!(MemoryTier::Ssd < MemoryTier::Disk);
    }

    #[test]
    fn test_promotion_score_calculation() {
        let now = SystemTime::now();
        let meta = ChunkMetadata {
            id: "test".to_string(),
            size_bytes: 1024,
            shape: vec![32, 32],
            current_tier: MemoryTier::Ssd,
            access_count: 10,
            last_access: now,
            access_interval_avg: Duration::from_secs(1),
            access_pattern: TierAccessPattern::Temporal,
            promotion_score: 0.0,
        };

        let score = meta.calculate_promotion_score(now);
        assert!(score > 0.0);
    }
}
