//! NUMA-aware Memory Allocation
//!
//! This module provides NUMA (Non-Uniform Memory Access) aware memory allocation
//! for optimizing large tensor operations on multi-socket systems.
//!
//! # Features
//!
//! - **NUMA Topology Detection**: Automatic detection of NUMA nodes and their characteristics
//! - **Affinity Policies**: Configure thread and memory affinity for optimal performance
//! - **Local Allocation**: Prefer allocation on the same NUMA node as the executing thread
//! - **Interleaved Allocation**: Distribute allocations across NUMA nodes
//! - **Balanced Allocation**: Balance memory usage across NUMA nodes
//! - **Statistics Tracking**: Monitor per-node memory usage and access patterns
//!
//! # Example
//!
//! ```rust
//! use tenrso_ooc::numa::{NumaAllocator, NumaPolicy};
//!
//! // Create NUMA-aware allocator
//! let mut allocator = NumaAllocator::new();
//!
//! // Detect topology
//! let topology = allocator.detect_topology();
//! println!("NUMA nodes: {}", topology.num_nodes);
//!
//! // Allocate with local policy (prefer current node)
//! let policy = NumaPolicy::Local;
//! let buffer = allocator.allocate_aligned(1024 * 1024, 4096, policy);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// NUMA allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumaPolicy {
    /// Allocate on the same NUMA node as the calling thread (default)
    Local,

    /// Interleave allocations across all NUMA nodes
    Interleaved,

    /// Allocate on a specific NUMA node
    Preferred(usize),

    /// Bind to a specific NUMA node (strict policy)
    Bind(usize),

    /// Balance allocations across NUMA nodes based on usage
    Balanced,

    /// No specific NUMA policy (system default)
    Default,
}

impl Default for NumaPolicy {
    fn default() -> Self {
        Self::Local
    }
}

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNode {
    /// Node ID
    pub node_id: usize,

    /// Total memory capacity in bytes
    pub total_memory_bytes: usize,

    /// Free memory in bytes
    pub free_memory_bytes: usize,

    /// Number of CPUs on this node
    pub num_cpus: usize,

    /// CPU IDs on this node
    pub cpu_ids: Vec<usize>,

    /// Distance to other nodes (node_id -> distance)
    pub distances: HashMap<usize, usize>,
}

impl NumaNode {
    /// Create a new NUMA node
    pub fn new(node_id: usize) -> Self {
        Self {
            node_id,
            total_memory_bytes: 0,
            free_memory_bytes: 0,
            num_cpus: 0,
            cpu_ids: Vec::new(),
            distances: HashMap::new(),
        }
    }

    /// Memory usage as a fraction (0.0 to 1.0)
    pub fn memory_usage(&self) -> f64 {
        if self.total_memory_bytes == 0 {
            return 0.0;
        }
        let used = self
            .total_memory_bytes
            .saturating_sub(self.free_memory_bytes);
        used as f64 / self.total_memory_bytes as f64
    }
}

/// NUMA topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,

    /// NUMA nodes
    pub nodes: Vec<NumaNode>,

    /// Whether NUMA is available on this system
    pub numa_available: bool,

    /// Total system memory
    pub total_memory_bytes: usize,

    /// Number of CPUs in the system
    pub num_cpus: usize,
}

impl NumaTopology {
    /// Create a new topology
    pub fn new() -> Self {
        Self {
            num_nodes: 0,
            nodes: Vec::new(),
            numa_available: false,
            total_memory_bytes: 0,
            num_cpus: 0,
        }
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: usize) -> Option<&NumaNode> {
        self.nodes.get(node_id)
    }

    /// Get the node with the most free memory
    pub fn most_free_node(&self) -> Option<usize> {
        self.nodes
            .iter()
            .max_by_key(|n| n.free_memory_bytes)
            .map(|n| n.node_id)
    }

    /// Get the node with the least usage
    pub fn least_used_node(&self) -> Option<usize> {
        self.nodes
            .iter()
            .min_by(|a, b| {
                a.memory_usage()
                    .partial_cmp(&b.memory_usage())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|n| n.node_id)
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// NUMA allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaStats {
    /// Allocations per node
    pub allocations_per_node: HashMap<usize, usize>,

    /// Bytes allocated per node
    pub bytes_per_node: HashMap<usize, usize>,

    /// Remote accesses (accessing memory on a different node)
    pub remote_accesses: usize,

    /// Local accesses (accessing memory on the same node)
    pub local_accesses: usize,

    /// Failed allocations (could not allocate on preferred node)
    pub failed_allocations: usize,
}

impl NumaStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            allocations_per_node: HashMap::new(),
            bytes_per_node: HashMap::new(),
            remote_accesses: 0,
            local_accesses: 0,
            failed_allocations: 0,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, node_id: usize, bytes: usize) {
        *self.allocations_per_node.entry(node_id).or_insert(0) += 1;
        *self.bytes_per_node.entry(node_id).or_insert(0) += bytes;
    }

    /// Record an access
    pub fn record_access(&mut self, is_local: bool) {
        if is_local {
            self.local_accesses += 1;
        } else {
            self.remote_accesses += 1;
        }
    }

    /// Get locality ratio (local_accesses / total_accesses)
    pub fn locality_ratio(&self) -> f64 {
        let total = self.local_accesses + self.remote_accesses;
        if total == 0 {
            return 0.0;
        }
        self.local_accesses as f64 / total as f64
    }

    /// Get total allocations
    pub fn total_allocations(&self) -> usize {
        self.allocations_per_node.values().sum()
    }

    /// Get total bytes allocated
    pub fn total_bytes(&self) -> usize {
        self.bytes_per_node.values().sum()
    }
}

impl Default for NumaStats {
    fn default() -> Self {
        Self::new()
    }
}

/// NUMA-aware allocator
pub struct NumaAllocator {
    /// NUMA topology
    topology: NumaTopology,

    /// Allocation statistics
    stats: NumaStats,

    /// Current policy
    policy: NumaPolicy,

    /// Round-robin counter for interleaved allocations
    interleave_counter: usize,

    /// Allocations tracking (allocation_id -> node_id)
    allocations: HashMap<usize, usize>,

    /// Next allocation ID
    next_allocation_id: usize,
}

impl NumaAllocator {
    /// Create a new NUMA allocator
    pub fn new() -> Self {
        let mut allocator = Self {
            topology: NumaTopology::new(),
            stats: NumaStats::new(),
            policy: NumaPolicy::default(),
            interleave_counter: 0,
            allocations: HashMap::new(),
            next_allocation_id: 0,
        };

        // Detect topology on creation
        allocator.topology = allocator.detect_topology();

        allocator
    }

    /// Detect NUMA topology
    ///
    /// This function attempts to detect the NUMA configuration of the system.
    /// On systems without NUMA or where detection fails, it returns a single-node topology.
    pub fn detect_topology(&self) -> NumaTopology {
        // Try to detect NUMA using system information
        // This is a simplified implementation - real systems would use libnuma or similar

        #[cfg(target_os = "linux")]
        {
            self.detect_topology_linux()
        }

        #[cfg(not(target_os = "linux"))]
        {
            self.detect_topology_fallback()
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_topology_linux(&self) -> NumaTopology {
        use std::fs;
        use std::path::Path;

        let mut topology = NumaTopology::new();

        // Check if /sys/devices/system/node exists
        let node_path = Path::new("/sys/devices/system/node");
        if !node_path.exists() {
            return self.detect_topology_fallback();
        }

        // Count NUMA nodes
        let mut nodes = Vec::new();
        let Ok(read_dir) = fs::read_dir(node_path) else {
            return self.detect_topology_fallback();
        };
        for entry in read_dir.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("node") {
                if let Some(node_id_str) = name_str.strip_prefix("node") {
                    if let Ok(node_id) = node_id_str.parse::<usize>() {
                        let mut node = NumaNode::new(node_id);

                        // Try to read memory info
                        let meminfo_path = entry.path().join("meminfo");
                        if let Ok(meminfo) = fs::read_to_string(&meminfo_path) {
                            for line in meminfo.lines() {
                                if line.contains("MemTotal:") {
                                    if let Some(size_str) = line.split_whitespace().nth(3) {
                                        if let Ok(size_kb) = size_str.parse::<usize>() {
                                            node.total_memory_bytes = size_kb * 1024;
                                        }
                                    }
                                } else if line.contains("MemFree:") {
                                    if let Some(size_str) = line.split_whitespace().nth(3) {
                                        if let Ok(size_kb) = size_str.parse::<usize>() {
                                            node.free_memory_bytes = size_kb * 1024;
                                        }
                                    }
                                }
                            }
                        }

                        // Try to read CPU list
                        let cpulist_path = entry.path().join("cpulist");
                        if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                            node.cpu_ids = parse_cpu_list(cpulist.trim());
                            node.num_cpus = node.cpu_ids.len();
                        }

                        nodes.push(node);
                    }
                }
            }
        }

        if nodes.is_empty() {
            return self.detect_topology_fallback();
        }

        topology.nodes = nodes;
        topology.num_nodes = topology.nodes.len();
        topology.numa_available = topology.num_nodes > 1;
        topology.total_memory_bytes = topology.nodes.iter().map(|n| n.total_memory_bytes).sum();
        topology.num_cpus = topology.nodes.iter().map(|n| n.num_cpus).sum();

        // Set distances (simplified - real implementation would read from sysfs)
        for node in &mut topology.nodes {
            for other_id in 0..topology.num_nodes {
                let distance = if node.node_id == other_id { 10 } else { 20 };
                node.distances.insert(other_id, distance);
            }
        }

        topology
    }

    #[allow(dead_code)]
    fn detect_topology_fallback(&self) -> NumaTopology {
        // Fallback: create a single-node topology
        let mut topology = NumaTopology::new();

        let num_cpus = num_cpus::get();

        let mut node = NumaNode::new(0);
        node.num_cpus = num_cpus;
        node.cpu_ids = (0..num_cpus).collect();

        // Estimate total memory (simplified)
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(size_str) = line.split_whitespace().nth(1) {
                            if let Ok(size_kb) = size_str.parse::<usize>() {
                                node.total_memory_bytes = size_kb * 1024;
                                node.free_memory_bytes = size_kb * 1024 / 2; // Rough estimate
                            }
                        }
                        break;
                    }
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Very rough estimate for non-Linux systems
            node.total_memory_bytes = 8 * 1024 * 1024 * 1024; // 8 GB default
            node.free_memory_bytes = 4 * 1024 * 1024 * 1024;
        }

        node.distances.insert(0, 10);

        let total_mem = node.total_memory_bytes;
        let total_cpus = node.num_cpus;

        topology.nodes.push(node);
        topology.num_nodes = 1;
        topology.numa_available = false;
        topology.total_memory_bytes = total_mem;
        topology.num_cpus = total_cpus;

        topology
    }

    /// Set allocation policy
    pub fn set_policy(&mut self, policy: NumaPolicy) {
        self.policy = policy;
    }

    /// Get current policy
    pub fn get_policy(&self) -> NumaPolicy {
        self.policy
    }

    /// Get topology
    pub fn get_topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Get statistics
    pub fn get_stats(&self) -> &NumaStats {
        &self.stats
    }

    /// Select node based on policy
    fn select_node(&mut self, size_bytes: usize, policy: NumaPolicy) -> Option<usize> {
        if self.topology.nodes.is_empty() {
            return None;
        }

        match policy {
            NumaPolicy::Local => {
                // For now, just use node 0 as "local"
                // Real implementation would use getcpu() or similar
                Some(0)
            }

            NumaPolicy::Interleaved => {
                // Round-robin across nodes
                let node_id = self.interleave_counter % self.topology.num_nodes;
                self.interleave_counter += 1;
                Some(node_id)
            }

            NumaPolicy::Preferred(node_id) => {
                // Try preferred node, fall back if it doesn't have capacity
                if node_id < self.topology.num_nodes {
                    let node = &self.topology.nodes[node_id];
                    if node.free_memory_bytes >= size_bytes {
                        Some(node_id)
                    } else {
                        // Fall back to node with most free memory
                        self.topology.most_free_node()
                    }
                } else {
                    None
                }
            }

            NumaPolicy::Bind(node_id) => {
                // Strict binding
                if node_id < self.topology.num_nodes {
                    Some(node_id)
                } else {
                    None
                }
            }

            NumaPolicy::Balanced => {
                // Select node with least usage
                self.topology.least_used_node()
            }

            NumaPolicy::Default => {
                // System default (node 0)
                Some(0)
            }
        }
    }

    /// Allocate aligned memory with NUMA awareness
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size of allocation in bytes
    /// * `alignment` - Alignment requirement in bytes
    /// * `policy` - NUMA allocation policy
    ///
    /// # Returns
    ///
    /// Allocation ID that can be used to track the allocation
    pub fn allocate_aligned(
        &mut self,
        size_bytes: usize,
        _alignment: usize,
        policy: NumaPolicy,
    ) -> Option<usize> {
        let node_id = self.select_node(size_bytes, policy)?;

        // In a real implementation, we would use numa_alloc_onnode or similar
        // For now, we just track the allocation

        let allocation_id = self.next_allocation_id;
        self.next_allocation_id += 1;

        // Update tracking
        self.allocations.insert(allocation_id, node_id);
        self.stats.record_allocation(node_id, size_bytes);

        // Update node free memory
        if let Some(node) = self.topology.nodes.get_mut(node_id) {
            node.free_memory_bytes = node.free_memory_bytes.saturating_sub(size_bytes);
        }

        Some(allocation_id)
    }

    /// Free an allocation
    pub fn free(&mut self, allocation_id: usize, size_bytes: usize) {
        if let Some(&node_id) = self.allocations.get(&allocation_id) {
            // Update node free memory
            if let Some(node) = self.topology.nodes.get_mut(node_id) {
                node.free_memory_bytes =
                    (node.free_memory_bytes + size_bytes).min(node.total_memory_bytes);
            }

            self.allocations.remove(&allocation_id);
        }
    }

    /// Get the node ID for an allocation
    pub fn get_allocation_node(&self, allocation_id: usize) -> Option<usize> {
        self.allocations.get(&allocation_id).copied()
    }

    /// Record a memory access
    pub fn record_access(&mut self, allocation_id: usize, accessing_node: usize) {
        if let Some(&allocation_node) = self.allocations.get(&allocation_id) {
            let is_local = allocation_node == accessing_node;
            self.stats.record_access(is_local);
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = NumaStats::new();
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse CPU list from sysfs format (e.g., "0-3,8-11")
#[cfg(target_os = "linux")]
fn parse_cpu_list(cpulist: &str) -> Vec<usize> {
    let mut cpus = Vec::new();

    for part in cpulist.split(',') {
        if let Some((start, end)) = part.split_once('-') {
            // Range format
            if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            // Single CPU
            cpus.push(cpu);
        }
    }

    cpus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_allocator_creation() {
        let allocator = NumaAllocator::new();
        assert!(allocator.get_topology().num_nodes > 0);
    }

    #[test]
    fn test_topology_detection() {
        let allocator = NumaAllocator::new();
        let topology = allocator.get_topology();

        assert!(topology.num_nodes > 0);
        assert_eq!(topology.nodes.len(), topology.num_nodes);
        assert!(topology.num_cpus > 0);
    }

    #[test]
    fn test_local_policy() {
        let mut allocator = NumaAllocator::new();
        allocator.set_policy(NumaPolicy::Local);

        let alloc_id = allocator.allocate_aligned(1024, 4096, NumaPolicy::Local);
        assert!(alloc_id.is_some());

        let node_id = allocator.get_allocation_node(alloc_id.unwrap());
        assert!(node_id.is_some());
    }

    #[test]
    fn test_interleaved_policy() {
        let mut allocator = NumaAllocator::new();

        if allocator.get_topology().num_nodes > 1 {
            let alloc1 = allocator.allocate_aligned(1024, 4096, NumaPolicy::Interleaved);
            let alloc2 = allocator.allocate_aligned(1024, 4096, NumaPolicy::Interleaved);

            let node1 = allocator.get_allocation_node(alloc1.unwrap());
            let node2 = allocator.get_allocation_node(alloc2.unwrap());

            // With interleaved, allocations should potentially be on different nodes
            assert!(node1.is_some());
            assert!(node2.is_some());
        }
    }

    #[test]
    fn test_balanced_policy() {
        let mut allocator = NumaAllocator::new();

        let alloc_id = allocator.allocate_aligned(1024, 4096, NumaPolicy::Balanced);
        assert!(alloc_id.is_some());
    }

    #[test]
    fn test_allocation_tracking() {
        let mut allocator = NumaAllocator::new();

        let alloc_id = allocator
            .allocate_aligned(1024, 4096, NumaPolicy::Local)
            .unwrap();

        let stats = allocator.get_stats();
        assert_eq!(stats.total_allocations(), 1);
        assert!(stats.total_bytes() >= 1024);

        allocator.free(alloc_id, 1024);
    }

    #[test]
    fn test_access_tracking() {
        let mut allocator = NumaAllocator::new();

        let alloc_id = allocator
            .allocate_aligned(1024, 4096, NumaPolicy::Local)
            .unwrap();
        let node_id = allocator.get_allocation_node(alloc_id).unwrap();

        // Local access
        allocator.record_access(alloc_id, node_id);

        // Remote access (if multi-node)
        let remote_node = (node_id + 1) % allocator.get_topology().num_nodes.max(2);
        allocator.record_access(alloc_id, remote_node);

        let stats = allocator.get_stats();
        assert!(stats.local_accesses > 0 || stats.remote_accesses > 0);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_parse_cpu_list() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
        assert_eq!(parse_cpu_list("0-1,4-5"), vec![0, 1, 4, 5]);
    }

    #[test]
    fn test_numa_stats() {
        let mut stats = NumaStats::new();

        stats.record_allocation(0, 1024);
        stats.record_allocation(1, 2048);
        stats.record_access(true);
        stats.record_access(false);

        assert_eq!(stats.total_allocations(), 2);
        assert_eq!(stats.total_bytes(), 3072);
        assert_eq!(stats.local_accesses, 1);
        assert_eq!(stats.remote_accesses, 1);
        assert_eq!(stats.locality_ratio(), 0.5);
    }

    #[test]
    fn test_topology_queries() {
        let allocator = NumaAllocator::new();
        let topology = allocator.get_topology();

        if topology.num_nodes > 0 {
            assert!(topology.most_free_node().is_some());
            assert!(topology.least_used_node().is_some());
            assert!(topology.get_node(0).is_some());
        }
    }
}
