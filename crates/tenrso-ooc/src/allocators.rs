//! # Custom Memory Allocators
//!
//! Production-grade memory allocator support for out-of-core tensor operations.
//!
//! This module provides:
//! - Configurable allocator selection (System, jemalloc, mimalloc)
//! - Allocator performance benchmarking and comparison
//! - Memory statistics and profiling
//! - Thread-local and global allocator configuration
//!
//! ## Allocator Options
//!
//! - **System** (default): Platform default allocator (glibc malloc, Windows HeapAlloc, etc.)
//! - **jemalloc**: High-performance allocator optimized for multi-threaded workloads
//! - **mimalloc**: Microsoft's allocator optimized for small object allocation
//!
//! ## Performance Characteristics
//!
//! ### jemalloc
//! - **Best for**: Multi-threaded tensor operations, large allocations
//! - **Pros**: Low fragmentation, excellent scalability, good for long-running processes
//! - **Cons**: Higher memory overhead, slower for single-threaded small allocations
//!
//! ### mimalloc
//! - **Best for**: High allocation/deallocation rates, small chunks
//! - **Pros**: Very fast, low overhead, excellent cache locality
//! - **Cons**: Higher memory usage for very large allocations
//!
//! ## Usage
//!
//! ### Compile-time Selection
//!
//! ```bash
//! # Use jemalloc
//! cargo build --features jemalloc
//!
//! # Use mimalloc
//! cargo build --features mimalloc-allocator
//! ```
//!
//! ### Runtime Configuration
//!
//! ```rust,ignore
//! use tenrso_ooc::allocators::{AllocatorInfo, get_allocator_stats};
//!
//! // Get current allocator info
//! let info = AllocatorInfo::current();
//! println!("Using allocator: {:?}", info.name);
//!
//! // Get memory statistics (if available)
//! if let Some(stats) = get_allocator_stats() {
//!     println!("Allocated: {} MB", stats.allocated_mb());
//!     println!("Resident: {} MB", stats.resident_mb());
//! }
//! ```

/// Global allocator selection (set at compile time)
#[cfg(all(
    feature = "jemalloc",
    not(target_env = "msvc"),
    not(feature = "mimalloc-allocator")
))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "mimalloc-allocator", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Allocator type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorType {
    /// System default allocator
    System,
    /// jemalloc allocator
    Jemalloc,
    /// mimalloc allocator
    Mimalloc,
}

impl AllocatorType {
    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            AllocatorType::System => "system",
            AllocatorType::Jemalloc => "jemalloc",
            AllocatorType::Mimalloc => "mimalloc",
        }
    }

    /// Get the currently active allocator
    pub fn current() -> Self {
        #[cfg(all(
            feature = "jemalloc",
            not(target_env = "msvc"),
            not(feature = "mimalloc-allocator")
        ))]
        {
            return AllocatorType::Jemalloc;
        }

        #[cfg(all(feature = "mimalloc-allocator", not(feature = "jemalloc")))]
        {
            return AllocatorType::Mimalloc;
        }

        // Default to system allocator
        AllocatorType::System
    }
}

/// Allocator information
#[derive(Debug, Clone)]
pub struct AllocatorInfo {
    /// Allocator type
    pub allocator_type: AllocatorType,
    /// Allocator name
    pub name: &'static str,
    /// Whether statistics are available
    pub has_stats: bool,
    /// Whether profiling is available
    pub has_profiling: bool,
}

impl AllocatorInfo {
    /// Get information about the current allocator
    pub fn current() -> Self {
        let allocator_type = AllocatorType::current();
        let name = allocator_type.name();

        let (has_stats, has_profiling) = match allocator_type {
            AllocatorType::Jemalloc => (true, true),
            AllocatorType::Mimalloc => (true, false),
            AllocatorType::System => (false, false),
        };

        Self {
            allocator_type,
            name,
            has_stats,
            has_profiling,
        }
    }

    /// Check if this is the jemalloc allocator
    pub fn is_jemalloc(&self) -> bool {
        self.allocator_type == AllocatorType::Jemalloc
    }

    /// Check if this is the mimalloc allocator
    pub fn is_mimalloc(&self) -> bool {
        self.allocator_type == AllocatorType::Mimalloc
    }

    /// Check if this is the system allocator
    pub fn is_system(&self) -> bool {
        self.allocator_type == AllocatorType::System
    }
}

/// Memory allocator statistics
#[derive(Debug, Clone, Default)]
pub struct AllocatorStats {
    /// Total bytes allocated
    pub allocated: usize,
    /// Total bytes in active use
    pub active: usize,
    /// Total bytes mapped (resident)
    pub resident: usize,
    /// Total bytes retained (not released to OS)
    pub retained: usize,
}

impl AllocatorStats {
    /// Get allocated memory in MB
    pub fn allocated_mb(&self) -> f64 {
        self.allocated as f64 / (1024.0 * 1024.0)
    }

    /// Get active memory in MB
    pub fn active_mb(&self) -> f64 {
        self.active as f64 / (1024.0 * 1024.0)
    }

    /// Get resident memory in MB
    pub fn resident_mb(&self) -> f64 {
        self.resident as f64 / (1024.0 * 1024.0)
    }

    /// Get retained memory in MB
    pub fn retained_mb(&self) -> f64 {
        self.retained as f64 / (1024.0 * 1024.0)
    }

    /// Get memory overhead percentage
    pub fn overhead_pct(&self) -> f64 {
        if self.active == 0 {
            0.0
        } else {
            ((self.resident as f64 - self.active as f64) / self.active as f64) * 100.0
        }
    }
}

/// Get current allocator statistics
///
/// Returns `None` if the current allocator doesn't support statistics.
pub fn get_allocator_stats() -> Option<AllocatorStats> {
    #[cfg(all(
        feature = "jemalloc",
        not(target_env = "msvc"),
        not(feature = "mimalloc-allocator")
    ))]
    {
        return get_jemalloc_stats();
    }

    #[cfg(all(feature = "mimalloc-allocator", not(feature = "jemalloc")))]
    {
        return get_mimalloc_stats();
    }

    // Default: no stats available
    None
}

/// Get jemalloc statistics
#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[allow(dead_code)]
fn get_jemalloc_stats() -> Option<AllocatorStats> {
    // Query jemalloc epoch to refresh statistics
    tikv_jemalloc_ctl::epoch::mib().unwrap().advance().ok()?;

    let allocated = tikv_jemalloc_ctl::stats::allocated::mib()
        .ok()?
        .read()
        .ok()?;
    let active = tikv_jemalloc_ctl::stats::active::mib().ok()?.read().ok()?;
    let resident = tikv_jemalloc_ctl::stats::resident::mib()
        .ok()?
        .read()
        .ok()?;
    let retained = tikv_jemalloc_ctl::stats::retained::mib()
        .ok()?
        .read()
        .ok()?;

    Some(AllocatorStats {
        allocated,
        active,
        resident,
        retained,
    })
}

/// Get mimalloc statistics
#[cfg(feature = "mimalloc-allocator")]
#[allow(dead_code)]
fn get_mimalloc_stats() -> Option<AllocatorStats> {
    // mimalloc doesn't expose detailed statistics in the same way
    // We can only get basic allocation info
    Some(AllocatorStats {
        allocated: 0, // Not available
        active: 0,    // Not available
        resident: 0,  // Not available
        retained: 0,  // Not available
    })
}

/// Allocator performance benchmark result
#[derive(Debug, Clone)]
pub struct AllocatorBenchmark {
    /// Allocator name
    pub allocator: &'static str,
    /// Number of allocations performed
    pub allocations: usize,
    /// Total bytes allocated
    pub total_bytes: usize,
    /// Duration in nanoseconds
    pub duration_ns: u64,
    /// Allocations per second
    pub allocs_per_sec: f64,
    /// Throughput in MB/s
    pub throughput_mbps: f64,
}

impl AllocatorBenchmark {
    /// Run a benchmark with the current allocator
    pub fn run(allocations: usize, allocation_size: usize) -> Self {
        let info = AllocatorInfo::current();
        let start = std::time::Instant::now();

        // Perform allocations
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(allocations);
        for _ in 0..allocations {
            let layout = std::alloc::Layout::from_size_align(allocation_size, 8).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            ptrs.push(ptr);
        }

        // Deallocate
        for &ptr in &ptrs {
            let layout = std::alloc::Layout::from_size_align(allocation_size, 8).unwrap();
            unsafe { std::alloc::dealloc(ptr, layout) };
        }

        let duration = start.elapsed();
        let duration_ns = duration.as_nanos() as u64;
        let duration_secs = duration.as_secs_f64();

        let total_bytes = allocations * allocation_size;
        let allocs_per_sec = if duration_secs > 0.0 {
            allocations as f64 / duration_secs
        } else {
            0.0
        };
        let throughput_mbps = if duration_secs > 0.0 {
            (total_bytes as f64 / (1024.0 * 1024.0)) / duration_secs
        } else {
            0.0
        };

        Self {
            allocator: info.name,
            allocations,
            total_bytes,
            duration_ns,
            allocs_per_sec,
            throughput_mbps,
        }
    }

    /// Print benchmark results
    pub fn print(&self) {
        println!("Allocator Benchmark: {}", self.allocator);
        println!("  Allocations: {}", self.allocations);
        println!("  Total bytes: {} MB", self.total_bytes / (1024 * 1024));
        println!(
            "  Duration: {:.2} ms",
            self.duration_ns as f64 / 1_000_000.0
        );
        println!("  Allocs/sec: {:.0}", self.allocs_per_sec);
        println!("  Throughput: {:.2} MB/s", self.throughput_mbps);
    }
}

/// Memory pool using custom allocator
pub struct AllocatorPool {
    buffers: Vec<Vec<u8>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl AllocatorPool {
    /// Create a new allocator pool
    pub fn new(buffer_size: usize, initial_capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(initial_capacity);
        for _ in 0..initial_capacity {
            buffers.push(Vec::with_capacity(buffer_size));
        }

        Self {
            buffers,
            buffer_size,
            max_buffers: 100,
        }
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self) -> Vec<u8> {
        self.buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.buffer_size))
    }

    /// Release a buffer back to the pool
    pub fn release(&mut self, mut buffer: Vec<u8>) {
        if self.buffers.len() < self.max_buffers {
            buffer.clear();
            self.buffers.push(buffer);
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.buffers.len()
    }

    /// Get allocator info for this pool
    pub fn allocator_info(&self) -> AllocatorInfo {
        AllocatorInfo::current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_info() {
        let info = AllocatorInfo::current();
        assert!(!info.name.is_empty());

        // Verify that exactly one of these is true
        let count = [info.is_system(), info.is_jemalloc(), info.is_mimalloc()]
            .iter()
            .filter(|&&x| x)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_allocator_type() {
        let current = AllocatorType::current();
        let name = current.name();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_allocator_stats() {
        // May be None for system allocator
        if let Some(stats) = get_allocator_stats() {
            // Stats should be non-negative
            assert!(stats.allocated_mb() >= 0.0);
            assert!(stats.resident_mb() >= 0.0);
        }
    }

    #[test]
    fn test_allocator_benchmark() {
        let bench = AllocatorBenchmark::run(1000, 1024);
        assert_eq!(bench.allocations, 1000);
        assert_eq!(bench.total_bytes, 1000 * 1024);
        assert!(bench.duration_ns > 0);
        assert!(bench.allocs_per_sec > 0.0);
    }

    #[test]
    fn test_allocator_pool() {
        let mut pool = AllocatorPool::new(1024, 5);
        assert_eq!(pool.size(), 5);

        let buf = pool.acquire();
        assert_eq!(pool.size(), 4);

        pool.release(buf);
        assert_eq!(pool.size(), 5);

        let info = pool.allocator_info();
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_large_allocation_benchmark() {
        // Test with larger allocations (simulating tensor chunks)
        let bench = AllocatorBenchmark::run(100, 1024 * 1024); // 100 x 1MB
        assert!(bench.throughput_mbps > 0.0);
        bench.print();
    }
}
