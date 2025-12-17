//! # NUMA-aware Memory Allocation Demo
//!
//! This example demonstrates NUMA-aware memory allocation for optimizing
//! tensor operations on multi-socket systems.
//!
//! ## Features Demonstrated
//!
//! - Automatic NUMA topology detection
//! - Different allocation policies (Local, Interleaved, Balanced, etc.)
//! - Per-node memory usage tracking
//! - Locality statistics (local vs remote accesses)
//! - Performance recommendations based on NUMA configuration

use tenrso_ooc::numa::{NumaAllocator, NumaPolicy};

fn main() {
    println!("=== NUMA-aware Memory Allocation Demo ===\n");

    // Create NUMA allocator (automatically detects topology)
    let mut allocator = NumaAllocator::new();

    // Display detected topology
    println!("=== NUMA Topology Detection ===\n");
    let numa_available = allocator.get_topology().numa_available;
    let num_nodes = allocator.get_topology().num_nodes;
    let total_memory = allocator.get_topology().total_memory_bytes;
    let num_cpus = allocator.get_topology().num_cpus;

    println!("NUMA available: {}", numa_available);
    println!("Number of NUMA nodes: {}", num_nodes);
    println!(
        "Total system memory: {} GB",
        total_memory / (1024 * 1024 * 1024)
    );
    println!("Total CPUs: {}\n", num_cpus);

    println!("Per-node information:");
    for i in 0..num_nodes {
        if let Some(node) = allocator.get_topology().get_node(i) {
            println!("  Node {}:", node.node_id);
            println!(
                "    Total memory: {} GB",
                node.total_memory_bytes / (1024 * 1024 * 1024)
            );
            println!(
                "    Free memory: {} GB",
                node.free_memory_bytes / (1024 * 1024 * 1024)
            );
            println!("    CPUs: {} (IDs: {:?})", node.num_cpus, node.cpu_ids);
            println!("    Usage: {:.1}%", node.memory_usage() * 100.0);
        }
    }

    println!();

    // Demonstrate different allocation policies
    println!("=== Allocation Policies ===\n");

    // 1. Local policy (prefer current node)
    println!("1. LOCAL Policy (prefer current NUMA node):");
    allocator.set_policy(NumaPolicy::Local);
    let allocs_local: Vec<_> = (0..5)
        .filter_map(|i| {
            let alloc_id = allocator.allocate_aligned(
                100 * 1024 * 1024, // 100 MB
                4096,
                NumaPolicy::Local,
            )?;
            let node = allocator.get_allocation_node(alloc_id)?;
            println!("   Allocation {}: placed on node {}", i, node);
            Some((alloc_id, 100 * 1024 * 1024))
        })
        .collect();

    println!();

    // 2. Interleaved policy (round-robin across nodes)
    println!("2. INTERLEAVED Policy (round-robin across nodes):");
    let allocs_interleaved: Vec<_> = (0..5)
        .filter_map(|i| {
            let alloc_id =
                allocator.allocate_aligned(100 * 1024 * 1024, 4096, NumaPolicy::Interleaved)?;
            let node = allocator.get_allocation_node(alloc_id)?;
            println!("   Allocation {}: placed on node {}", i, node);
            Some((alloc_id, 100 * 1024 * 1024))
        })
        .collect();

    println!();

    // 3. Balanced policy (least-used node)
    println!("3. BALANCED Policy (select least-used node):");
    let allocs_balanced: Vec<_> = (0..5)
        .filter_map(|i| {
            let alloc_id =
                allocator.allocate_aligned(100 * 1024 * 1024, 4096, NumaPolicy::Balanced)?;
            let node = allocator.get_allocation_node(alloc_id)?;
            println!("   Allocation {}: placed on node {}", i, node);
            Some((alloc_id, 100 * 1024 * 1024))
        })
        .collect();

    println!();

    // 4. Preferred node policy
    if num_nodes > 1 {
        println!("4. PREFERRED Policy (prefer node 1, fallback if full):");
        let allocs_preferred: Vec<_> = (0..3)
            .filter_map(|i| {
                let alloc_id = allocator.allocate_aligned(
                    100 * 1024 * 1024,
                    4096,
                    NumaPolicy::Preferred(1),
                )?;
                let node = allocator.get_allocation_node(alloc_id)?;
                println!("   Allocation {}: placed on node {}", i, node);
                Some((alloc_id, 100 * 1024 * 1024))
            })
            .collect();

        // Free preferred allocations
        for (alloc_id, size) in allocs_preferred {
            allocator.free(alloc_id, size);
        }

        println!();
    }

    // Simulate access patterns
    println!("=== Access Pattern Simulation ===\n");

    // Simulate local accesses
    for (alloc_id, _) in &allocs_local[..3.min(allocs_local.len())] {
        allocator.record_access(*alloc_id, 0); // Access from node 0
    }

    // Simulate remote accesses (if multi-node)
    if num_nodes > 1 {
        for (alloc_id, _) in &allocs_interleaved[..2.min(allocs_interleaved.len())] {
            allocator.record_access(*alloc_id, 1); // Access from node 1
        }
    }

    println!("Simulated memory accesses:");
    println!(
        "  - {} local accesses (same node)",
        3.min(allocs_local.len())
    );
    if num_nodes > 1 {
        println!(
            "  - {} cross-node accesses",
            2.min(allocs_interleaved.len())
        );
    }

    println!();

    // Display statistics
    println!("=== Allocation Statistics ===\n");
    let stats = allocator.get_stats();

    println!("Per-node allocations:");
    for (node_id, count) in &stats.allocations_per_node {
        let bytes = stats.bytes_per_node.get(node_id).unwrap_or(&0);
        println!(
            "  Node {}: {} allocations, {} MB",
            node_id,
            count,
            bytes / (1024 * 1024)
        );
    }

    println!();
    println!("Access patterns:");
    println!("  Local accesses: {}", stats.local_accesses);
    println!("  Remote accesses: {}", stats.remote_accesses);
    println!("  Locality ratio: {:.1}%", stats.locality_ratio() * 100.0);
    println!("  Failed allocations: {}", stats.failed_allocations);

    println!();

    // Recommendations
    println!("=== Performance Recommendations ===\n");

    if numa_available {
        println!("NUMA is available on this system:");
        println!();
        println!("1. Use LOCAL policy for thread-local data");
        println!("   - Minimizes remote memory access latency");
        println!("   - Best for independent parallel computations");
        println!();
        println!("2. Use INTERLEAVED policy for shared read-only data");
        println!("   - Distributes memory bandwidth load");
        println!("   - Good for large tensors accessed by all threads");
        println!();
        println!("3. Use BALANCED policy for dynamic workloads");
        println!("   - Prevents memory hotspots");
        println!("   - Adapts to changing memory usage patterns");
        println!();

        if stats.locality_ratio() < 0.8 {
            println!(
                "WARNING: Low locality ratio ({:.1}%)!",
                stats.locality_ratio() * 100.0
            );
            println!("Consider:");
            println!("  - Using LOCAL policy more often");
            println!("  - Binding threads to specific NUMA nodes");
            println!("  - Reducing cross-node data sharing");
        } else {
            println!(
                "Good locality ratio ({:.1}%) - memory access patterns are NUMA-friendly!",
                stats.locality_ratio() * 100.0
            );
        }
    } else {
        println!("NUMA is not available on this system (single node):");
        println!("  - All allocation policies will behave similarly");
        println!("  - No remote memory access overhead");
        println!("  - Consider parallel algorithms instead of NUMA optimizations");
    }

    println!();

    // Cleanup
    for (alloc_id, size) in allocs_local {
        allocator.free(alloc_id, size);
    }
    for (alloc_id, size) in allocs_interleaved {
        allocator.free(alloc_id, size);
    }
    for (alloc_id, size) in allocs_balanced {
        allocator.free(alloc_id, size);
    }

    println!("=== Demo Complete ===");
}
