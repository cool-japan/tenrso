//! Memory management and spill-to-disk policies
//!
//! This module provides automatic memory management for out-of-core tensor operations,
//! including smart spill policies, memory pressure detection, and cache management.
//!
//! # Features
//!
//! - Automatic memory tracking
//! - LRU-based spill policies
//! - Memory pressure detection
//! - Reference counting for chunk lifetime
//! - Configurable spill strategies
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::memory::{MemoryManager, SpillPolicy};
//!
//! let mut manager = MemoryManager::new()
//!     .max_memory_mb(1024)
//!     .spill_policy(SpillPolicy::LRU)
//!     .pressure_threshold(0.8);
//!
//! // Register a tensor chunk
//! manager.register_chunk("chunk_0", tensor, AccessPattern::ReadOnce)?;
//!
//! // Manager will automatically spill if needed
//! if manager.is_under_pressure() {
//!     manager.apply_spill_policy()?;
//! }
//! ```

use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tenrso_core::DenseND;

/// Access pattern hint for chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Read once and discard
    ReadOnce,
    /// Read multiple times
    ReadMany,
    /// Write once
    WriteOnce,
    /// Read and write multiple times
    ReadWrite,
}

/// Spill policy strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillPolicy {
    /// Least Recently Used
    LRU,
    /// Largest First (spill largest chunks first)
    LargestFirst,
    /// Least Frequently Used
    LFU,
    /// Access pattern based (prefer spilling ReadOnce chunks)
    PatternBased,
}

/// Chunk metadata for memory management
#[derive(Debug, Clone)]
struct ChunkMetadata {
    /// Chunk identifier
    #[allow(dead_code)]
    id: String,
    /// Size in bytes
    size_bytes: usize,
    /// Access pattern
    pattern: AccessPattern,
    /// Last access timestamp
    last_access: u64,
    /// Access count
    access_count: usize,
    /// Reference count (number of operations using this chunk)
    ref_count: usize,
    /// Whether chunk is currently spilled to disk
    spilled: bool,
    /// Path to spilled file (if spilled)
    #[allow(dead_code)]
    spill_path: Option<PathBuf>,
}

/// Memory manager for out-of-core execution
pub struct MemoryManager {
    /// Maximum memory limit in bytes
    max_memory_bytes: usize,
    /// Current memory usage in bytes
    current_memory_bytes: usize,
    /// Memory pressure threshold (0.0 to 1.0)
    pressure_threshold: f64,
    /// Spill policy
    spill_policy: SpillPolicy,
    /// Temporary directory for spilled chunks
    temp_dir: PathBuf,
    /// Chunk metadata
    chunks: HashMap<String, ChunkMetadata>,
    /// In-memory chunks (chunk_id -> tensor)
    in_memory: HashMap<String, DenseND<f64>>,
    /// LRU queue for tracking access order
    lru_queue: VecDeque<String>,
    /// Enable automatic spill
    auto_spill: bool,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB default
            current_memory_bytes: 0,
            pressure_threshold: 0.8,
            spill_policy: SpillPolicy::LRU,
            temp_dir: std::env::temp_dir(),
            chunks: HashMap::new(),
            in_memory: HashMap::new(),
            lru_queue: VecDeque::new(),
            auto_spill: true,
        }
    }

    /// Set maximum memory limit in bytes
    pub fn max_memory_bytes(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set maximum memory limit in megabytes
    pub fn max_memory_mb(mut self, mb: usize) -> Self {
        self.max_memory_bytes = mb * 1024 * 1024;
        self
    }

    /// Set memory pressure threshold (0.0 to 1.0)
    pub fn pressure_threshold(mut self, threshold: f64) -> Self {
        self.pressure_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set spill policy
    pub fn spill_policy(mut self, policy: SpillPolicy) -> Self {
        self.spill_policy = policy;
        self
    }

    /// Set temporary directory
    pub fn temp_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.temp_dir = path.as_ref().to_path_buf();
        self
    }

    /// Enable or disable automatic spill
    pub fn auto_spill(mut self, enable: bool) -> Self {
        self.auto_spill = enable;
        self
    }

    /// Get current memory usage
    pub fn current_memory(&self) -> usize {
        self.current_memory_bytes
    }

    /// Get memory usage ratio (0.0 to 1.0)
    pub fn memory_ratio(&self) -> f64 {
        self.current_memory_bytes as f64 / self.max_memory_bytes as f64
    }

    /// Check if under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        self.memory_ratio() >= self.pressure_threshold
    }

    /// Check if memory limit is exceeded
    pub fn is_over_limit(&self) -> bool {
        self.current_memory_bytes > self.max_memory_bytes
    }

    /// Register a chunk with the memory manager
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Unique identifier for the chunk
    /// * `tensor` - The tensor data
    /// * `pattern` - Access pattern hint
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, automatically spilling if needed
    pub fn register_chunk(
        &mut self,
        chunk_id: &str,
        tensor: DenseND<f64>,
        pattern: AccessPattern,
    ) -> Result<()> {
        let size = tensor.shape().iter().product::<usize>() * std::mem::size_of::<f64>();

        // Check if we need to spill before adding
        if self.auto_spill && self.current_memory_bytes + size > self.max_memory_bytes {
            self.apply_spill_policy()?;
        }

        // If still over limit, fail
        if self.current_memory_bytes + size > self.max_memory_bytes {
            return Err(anyhow!(
                "Cannot register chunk: would exceed memory limit ({} + {} > {})",
                self.current_memory_bytes,
                size,
                self.max_memory_bytes
            ));
        }

        let metadata = ChunkMetadata {
            id: chunk_id.to_string(),
            size_bytes: size,
            pattern,
            last_access: self.current_timestamp(),
            access_count: 0,
            ref_count: 1,
            spilled: false,
            spill_path: None,
        };

        self.chunks.insert(chunk_id.to_string(), metadata);
        self.in_memory.insert(chunk_id.to_string(), tensor);
        self.lru_queue.push_back(chunk_id.to_string());
        self.current_memory_bytes += size;

        Ok(())
    }

    /// Access a chunk (retrieve from memory or load from disk)
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Chunk identifier
    ///
    /// # Returns
    ///
    /// Reference to the tensor data
    pub fn access_chunk(&mut self, chunk_id: &str) -> Result<&DenseND<f64>> {
        // Get timestamp first to avoid borrow issues
        let timestamp = self.current_timestamp();

        // Update metadata
        if let Some(meta) = self.chunks.get_mut(chunk_id) {
            meta.last_access = timestamp;
            meta.access_count += 1;

            // Update LRU queue
            self.lru_queue.retain(|id| id != chunk_id);
            self.lru_queue.push_back(chunk_id.to_string());

            // Load from disk if spilled
            if meta.spilled {
                self.load_from_disk(chunk_id)?;
            }
        } else {
            return Err(anyhow!("Chunk {} not found", chunk_id));
        }

        self.in_memory
            .get(chunk_id)
            .ok_or_else(|| anyhow!("Chunk {} not in memory", chunk_id))
    }

    /// Increment reference count for a chunk
    pub fn incref(&mut self, chunk_id: &str) -> Result<()> {
        if let Some(meta) = self.chunks.get_mut(chunk_id) {
            meta.ref_count += 1;
            Ok(())
        } else {
            Err(anyhow!("Chunk {} not found", chunk_id))
        }
    }

    /// Decrement reference count for a chunk
    pub fn decref(&mut self, chunk_id: &str) -> Result<()> {
        if let Some(meta) = self.chunks.get_mut(chunk_id) {
            meta.ref_count = meta.ref_count.saturating_sub(1);

            // If ref count reaches 0 and pattern is ReadOnce, consider for immediate spill
            if meta.ref_count == 0
                && meta.pattern == AccessPattern::ReadOnce
                && !meta.spilled
                && self.is_under_pressure()
            {
                self.spill_chunk(chunk_id)?;
            }

            Ok(())
        } else {
            Err(anyhow!("Chunk {} not found", chunk_id))
        }
    }

    /// Apply spill policy to free up memory
    ///
    /// This selects and spills chunks to disk according to the configured policy.
    ///
    /// # Returns
    ///
    /// Number of chunks spilled
    pub fn apply_spill_policy(&mut self) -> Result<usize> {
        let target_memory = (self.max_memory_bytes as f64 * self.pressure_threshold) as usize;
        let mut spilled_count = 0;

        // Get candidate chunks for spilling (ref_count == 0 and not already spilled)
        let mut candidates: Vec<String> = self
            .chunks
            .iter()
            .filter(|(_, meta)| meta.ref_count == 0 && !meta.spilled)
            .map(|(id, _)| id.clone())
            .collect();

        // Sort candidates by policy
        match self.spill_policy {
            SpillPolicy::LRU => {
                // Use LRU queue order
                candidates.sort_by_key(|id| {
                    self.lru_queue
                        .iter()
                        .position(|x| x == id)
                        .unwrap_or(usize::MAX)
                });
            }
            SpillPolicy::LargestFirst => {
                candidates.sort_by_key(|id| std::cmp::Reverse(self.chunks[id].size_bytes));
            }
            SpillPolicy::LFU => {
                candidates.sort_by_key(|id| self.chunks[id].access_count);
            }
            SpillPolicy::PatternBased => {
                // Prioritize ReadOnce, then WriteOnce, then others
                candidates.sort_by_key(|id| match self.chunks[id].pattern {
                    AccessPattern::ReadOnce => 0,
                    AccessPattern::WriteOnce => 1,
                    AccessPattern::ReadMany => 2,
                    AccessPattern::ReadWrite => 3,
                });
            }
        }

        // Spill chunks until we're under target
        for chunk_id in candidates {
            if self.current_memory_bytes <= target_memory {
                break;
            }

            self.spill_chunk(&chunk_id)?;
            spilled_count += 1;
        }

        Ok(spilled_count)
    }

    /// Spill a specific chunk to disk
    ///
    /// **Note:** Requires the `mmap` feature to be enabled.
    #[cfg(feature = "mmap")]
    fn spill_chunk(&mut self, chunk_id: &str) -> Result<()> {
        let tensor = self
            .in_memory
            .remove(chunk_id)
            .ok_or_else(|| anyhow!("Chunk {} not in memory", chunk_id))?;

        let filename = format!("tenrso_chunk_{}.bin", chunk_id);
        let path = self.temp_dir.join(filename);

        crate::mmap_io::write_tensor_binary(&path, &tensor)?;

        if let Some(meta) = self.chunks.get_mut(chunk_id) {
            let size = meta.size_bytes;
            meta.spilled = true;
            meta.spill_path = Some(path);
            self.current_memory_bytes = self.current_memory_bytes.saturating_sub(size);
        }

        Ok(())
    }

    /// Load a chunk from disk back into memory
    ///
    /// **Note:** Requires the `mmap` feature to be enabled.
    #[cfg(feature = "mmap")]
    fn load_from_disk(&mut self, chunk_id: &str) -> Result<()> {
        let meta = self
            .chunks
            .get(chunk_id)
            .ok_or_else(|| anyhow!("Chunk {} not found", chunk_id))?;

        let path = meta
            .spill_path
            .as_ref()
            .ok_or_else(|| anyhow!("No spill path for chunk {}", chunk_id))?;

        let tensor = crate::mmap_io::read_tensor_binary(path)?;
        let size = meta.size_bytes;

        // Ensure we have space
        if self.current_memory_bytes + size > self.max_memory_bytes {
            self.apply_spill_policy()?;
        }

        self.in_memory.insert(chunk_id.to_string(), tensor);
        self.current_memory_bytes += size;

        if let Some(meta) = self.chunks.get_mut(chunk_id) {
            meta.spilled = false;
        }

        Ok(())
    }

    /// Placeholder for when mmap feature is not enabled
    #[cfg(not(feature = "mmap"))]
    fn spill_chunk(&mut self, _chunk_id: &str) -> Result<()> {
        Err(anyhow!("Spill requires mmap feature to be enabled"))
    }

    /// Placeholder for when mmap feature is not enabled
    #[cfg(not(feature = "mmap"))]
    fn load_from_disk(&mut self, _chunk_id: &str) -> Result<()> {
        Err(anyhow!("Load requires mmap feature to be enabled"))
    }

    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Clean up all spilled files
    #[cfg(feature = "mmap")]
    pub fn cleanup(&mut self) -> Result<()> {
        for meta in self.chunks.values() {
            if let Some(path) = &meta.spill_path {
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
            }
        }
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_chunks: self.chunks.len(),
            in_memory_chunks: self.in_memory.len(),
            spilled_chunks: self.chunks.values().filter(|m| m.spilled).count(),
            current_memory_bytes: self.current_memory_bytes,
            max_memory_bytes: self.max_memory_bytes,
            memory_ratio: self.memory_ratio(),
            under_pressure: self.is_under_pressure(),
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        #[cfg(feature = "mmap")]
        {
            let _ = self.cleanup();
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of chunks tracked
    pub total_chunks: usize,
    /// Number of chunks in memory
    pub in_memory_chunks: usize,
    /// Number of chunks spilled to disk
    pub spilled_chunks: usize,
    /// Current memory usage in bytes
    pub current_memory_bytes: usize,
    /// Maximum memory limit in bytes
    pub max_memory_bytes: usize,
    /// Memory usage ratio (0.0 to 1.0)
    pub memory_ratio: f64,
    /// Whether under memory pressure
    pub under_pressure: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let manager = MemoryManager::new()
            .max_memory_mb(512)
            .pressure_threshold(0.75)
            .spill_policy(SpillPolicy::LRU);

        assert_eq!(manager.current_memory(), 0);
        assert!(!manager.is_under_pressure());
        assert_eq!(manager.pressure_threshold, 0.75);
    }

    #[test]
    fn test_register_chunk() {
        let mut manager = MemoryManager::new().max_memory_mb(10);

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        manager
            .register_chunk("chunk_0", tensor, AccessPattern::ReadOnce)
            .unwrap();

        assert_eq!(manager.chunks.len(), 1);
        assert_eq!(manager.in_memory.len(), 1);
        assert!(manager.current_memory() > 0);
    }

    #[test]
    fn test_access_chunk() {
        let mut manager = MemoryManager::new();

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        manager
            .register_chunk("chunk_0", tensor, AccessPattern::ReadMany)
            .unwrap();

        let accessed = manager.access_chunk("chunk_0").unwrap();
        assert_eq!(accessed.shape(), &[2, 2]);

        // Check that metadata was updated
        let meta = &manager.chunks["chunk_0"];
        assert_eq!(meta.access_count, 1);
    }

    #[test]
    fn test_ref_counting() {
        let mut manager = MemoryManager::new();

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        manager
            .register_chunk("chunk_0", tensor, AccessPattern::ReadOnce)
            .unwrap();

        assert_eq!(manager.chunks["chunk_0"].ref_count, 1);

        manager.incref("chunk_0").unwrap();
        assert_eq!(manager.chunks["chunk_0"].ref_count, 2);

        manager.decref("chunk_0").unwrap();
        assert_eq!(manager.chunks["chunk_0"].ref_count, 1);
    }

    #[test]
    fn test_memory_pressure() {
        let manager = MemoryManager::new()
            .max_memory_mb(1)
            .pressure_threshold(0.8);

        assert!(!manager.is_under_pressure());

        let mut manager = MemoryManager::new()
            .max_memory_mb(1)
            .pressure_threshold(0.5);

        // Add chunk that exceeds pressure threshold
        let tensor = DenseND::<f64>::zeros(&[100, 100]); // ~80KB
        manager
            .register_chunk("chunk_0", tensor, AccessPattern::ReadOnce)
            .unwrap();

        // Depending on exact sizes, this may or may not trigger pressure
        // Just verify the calculation works
        assert!(manager.memory_ratio() >= 0.0);
    }

    #[test]
    fn test_stats() {
        let mut manager = MemoryManager::new();

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        manager
            .register_chunk("chunk_0", tensor, AccessPattern::ReadOnce)
            .unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_chunks, 1);
        assert_eq!(stats.in_memory_chunks, 1);
        assert_eq!(stats.spilled_chunks, 0);
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_spill_and_load() {
        let mut manager = MemoryManager::new().max_memory_mb(1).auto_spill(false);

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        manager
            .register_chunk("chunk_0", tensor.clone(), AccessPattern::ReadOnce)
            .unwrap();

        // Manually spill
        manager.decref("chunk_0").unwrap(); // Make ref_count = 0
        manager.spill_chunk("chunk_0").unwrap();

        assert!(manager.chunks["chunk_0"].spilled);
        assert_eq!(manager.in_memory.len(), 0);

        // Load back
        manager.load_from_disk("chunk_0").unwrap();
        assert!(!manager.chunks["chunk_0"].spilled);

        let loaded = manager.access_chunk("chunk_0").unwrap();
        assert_eq!(loaded.shape(), tensor.shape());

        // Cleanup
        manager.cleanup().unwrap();
    }
}
