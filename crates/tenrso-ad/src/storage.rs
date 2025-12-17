//! Optimized tensor storage for VJP contexts
//!
//! This module provides memory-efficient storage strategies for tensors
//! in VJP contexts, using reference counting for large tensors and
//! direct ownership for small ones.

use std::sync::Arc;
use tenrso_core::DenseND;

/// Threshold for using Arc-based storage (in number of elements)
/// Tensors larger than this will use Arc for shared ownership
const ARC_THRESHOLD: usize = 10_000;

/// Smart tensor storage that adapts based on tensor size
///
/// For small tensors (< 10k elements): Uses direct ownership
/// For large tensors (≥ 10k elements): Uses Arc for zero-copy cloning
#[derive(Clone)]
pub enum TensorStorage<T> {
    /// Direct ownership for small tensors
    Owned(DenseND<T>),
    /// Shared ownership for large tensors
    Shared(Arc<DenseND<T>>),
}

impl<T> TensorStorage<T>
where
    T: scirs2_core::Num + Clone,
{
    /// Create storage from a tensor, choosing the appropriate strategy
    pub fn new(tensor: DenseND<T>) -> Self {
        if tensor.len() >= ARC_THRESHOLD {
            Self::Shared(Arc::new(tensor))
        } else {
            Self::Owned(tensor)
        }
    }

    /// Create storage from a tensor with explicit strategy
    pub fn new_with_strategy(tensor: DenseND<T>, use_arc: bool) -> Self {
        if use_arc {
            Self::Shared(Arc::new(tensor))
        } else {
            Self::Owned(tensor)
        }
    }

    /// Get a reference to the tensor
    pub fn get_ref(&self) -> &DenseND<T> {
        match self {
            Self::Owned(tensor) => tensor,
            Self::Shared(arc) => arc.as_ref(),
        }
    }

    /// Try to get a mutable reference (only works for owned storage with no other refs)
    pub fn get_mut(&mut self) -> Option<&mut DenseND<T>> {
        match self {
            Self::Owned(tensor) => Some(tensor),
            Self::Shared(arc) => Arc::get_mut(arc),
        }
    }

    /// Convert to owned tensor (may clone if Arc has multiple references)
    pub fn into_owned(self) -> DenseND<T> {
        match self {
            Self::Owned(tensor) => tensor,
            Self::Shared(arc) => Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()),
        }
    }

    /// Get the number of elements in the tensor
    pub fn len(&self) -> usize {
        self.get_ref().len()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this storage is using Arc
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }

    /// Get the reference count if using Arc
    pub fn ref_count(&self) -> Option<usize> {
        match self {
            Self::Owned(_) => None,
            Self::Shared(arc) => Some(Arc::strong_count(arc)),
        }
    }

    /// Estimate memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        let element_count = self.len();
        let element_size = std::mem::size_of::<T>();
        let data_size = element_count * element_size;

        // Add overhead for the storage structure itself
        let overhead = match self {
            Self::Owned(_) => 0,
            Self::Shared(_) => std::mem::size_of::<Arc<DenseND<T>>>(),
        };

        data_size + overhead
    }
}

/// Memory budget tracker for VJP contexts
///
/// Tracks total memory usage across all VJP contexts and can
/// enforce limits to prevent excessive memory consumption.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum memory allowed (in bytes), None for unlimited
    max_memory: Option<usize>,
    /// Current memory usage (in bytes)
    current_usage: usize,
}

impl MemoryBudget {
    /// Create a new memory budget with the given limit
    pub fn new(max_memory: Option<usize>) -> Self {
        Self {
            max_memory,
            current_usage: 0,
        }
    }

    /// Create an unlimited budget
    pub fn unlimited() -> Self {
        Self::new(None)
    }

    /// Create a budget with a specific limit in bytes
    pub fn with_limit(bytes: usize) -> Self {
        Self::new(Some(bytes))
    }

    /// Allocate memory (returns true if allocation is allowed)
    pub fn allocate(&mut self, bytes: usize) -> bool {
        if let Some(max) = self.max_memory {
            if self.current_usage + bytes > max {
                return false;
            }
        }
        self.current_usage += bytes;
        true
    }

    /// Deallocate memory
    pub fn deallocate(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_sub(bytes);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get remaining memory budget
    pub fn remaining(&self) -> Option<usize> {
        self.max_memory
            .map(|max| max.saturating_sub(self.current_usage))
    }

    /// Check if we're within budget
    pub fn within_budget(&self) -> bool {
        self.max_memory
            .map(|max| self.current_usage <= max)
            .unwrap_or(true)
    }

    /// Get memory usage as a percentage of budget (if limited)
    pub fn usage_percentage(&self) -> Option<f64> {
        self.max_memory
            .map(|max| (self.current_usage as f64 / max as f64) * 100.0)
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Configuration for storage optimization
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Threshold for using Arc (in number of elements)
    pub arc_threshold: usize,
    /// Memory budget for VJP contexts
    pub memory_budget: MemoryBudget,
    /// Whether to aggressively use Arc for all tensors
    pub force_arc: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            arc_threshold: ARC_THRESHOLD,
            memory_budget: MemoryBudget::unlimited(),
            force_arc: false,
        }
    }
}

impl StorageConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Arc threshold
    pub fn with_arc_threshold(mut self, threshold: usize) -> Self {
        self.arc_threshold = threshold;
        self
    }

    /// Set the memory budget
    pub fn with_memory_budget(mut self, budget: MemoryBudget) -> Self {
        self.memory_budget = budget;
        self
    }

    /// Force Arc for all tensors
    pub fn with_force_arc(mut self, force: bool) -> Self {
        self.force_arc = force;
        self
    }

    /// Check if a tensor should use Arc based on this config
    pub fn should_use_arc<T>(&self, tensor: &DenseND<T>) -> bool
    where
        T: scirs2_core::Num + Clone,
    {
        self.force_arc || tensor.len() >= self.arc_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_storage_small() {
        let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let storage = TensorStorage::new(tensor);

        assert!(
            !storage.is_shared(),
            "Small tensor should use owned storage"
        );
        assert_eq!(storage.len(), 4);
    }

    #[test]
    fn test_tensor_storage_large() {
        let large_vec: Vec<f64> = (0..20_000).map(|i| i as f64).collect();
        let tensor = DenseND::from_vec(large_vec, &[100, 200]).unwrap();
        let storage = TensorStorage::new(tensor);

        assert!(storage.is_shared(), "Large tensor should use Arc storage");
        assert_eq!(storage.len(), 20_000);
    }

    #[test]
    fn test_tensor_storage_clone() {
        let large_vec: Vec<f64> = (0..20_000).map(|i| i as f64).collect();
        let tensor = DenseND::from_vec(large_vec, &[100, 200]).unwrap();
        let storage1 = TensorStorage::new(tensor);
        let storage2 = storage1.clone();

        assert_eq!(storage1.ref_count(), Some(2));
        assert_eq!(storage2.ref_count(), Some(2));
    }

    #[test]
    fn test_memory_budget() {
        let mut budget = MemoryBudget::with_limit(1_000_000);

        assert!(budget.allocate(500_000));
        assert_eq!(budget.current_usage(), 500_000);
        assert_eq!(budget.remaining(), Some(500_000));

        assert!(budget.allocate(300_000));
        assert_eq!(budget.current_usage(), 800_000);

        assert!(!budget.allocate(300_000)); // Would exceed budget
        assert_eq!(budget.current_usage(), 800_000); // Unchanged

        budget.deallocate(500_000);
        assert_eq!(budget.current_usage(), 300_000);
        assert!(budget.allocate(500_000)); // Now fits
    }

    #[test]
    fn test_memory_budget_unlimited() {
        let mut budget = MemoryBudget::unlimited();

        assert!(budget.allocate(1_000_000_000));
        assert!(budget.allocate(1_000_000_000));
        assert_eq!(budget.current_usage(), 2_000_000_000);
        assert!(budget.within_budget());
    }

    #[test]
    fn test_storage_config() {
        let config = StorageConfig::new()
            .with_arc_threshold(5_000)
            .with_force_arc(false);

        let small_tensor = DenseND::from_vec(vec![1.0; 100], &[10, 10]).unwrap();
        assert!(!config.should_use_arc(&small_tensor));

        let large_tensor = DenseND::from_vec(vec![1.0; 10_000], &[100, 100]).unwrap();
        assert!(config.should_use_arc(&large_tensor));
    }

    #[test]
    fn test_storage_memory_usage() {
        let tensor = DenseND::from_vec(vec![1.0f64; 1000], &[10, 100]).unwrap();
        let storage = TensorStorage::new(tensor);

        let usage = storage.memory_usage();
        // Should be at least 1000 * 8 bytes (f64 size)
        assert!(usage >= 8_000);
    }
}
