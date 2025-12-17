//! Allocator Selection and Performance Comparison
//!
//! This example demonstrates:
//! - Detecting the current allocator
//! - Querying allocator statistics
//! - Running performance benchmarks
//! - Comparing allocator characteristics
//!
//! # Build Instructions
//!
//! ## System Allocator (default)
//! ```bash
//! cargo run --example allocator_selection
//! ```
//!
//! ## jemalloc
//! ```bash
//! cargo run --example allocator_selection --features jemalloc
//! ```
//!
//! ## mimalloc
//! ```bash
//! cargo run --example allocator_selection --features mimalloc-allocator
//! ```

use tenrso_ooc::allocators::{
    get_allocator_stats, AllocatorBenchmark, AllocatorInfo, AllocatorPool,
};

fn main() {
    println!("=== TenRSo Out-of-Core: Allocator Selection Demo ===\n");

    // 1. Detect current allocator
    let info = AllocatorInfo::current();
    println!("📊 Current Allocator");
    println!("  Type: {:?}", info.allocator_type);
    println!("  Name: {}", info.name);
    println!("  Statistics Available: {}", info.has_stats);
    println!("  Profiling Available: {}", info.has_profiling);
    println!();

    // 2. Query allocator statistics (if available)
    if info.has_stats {
        println!("📈 Allocator Statistics");
        if let Some(stats) = get_allocator_stats() {
            println!("  Allocated: {:.2} MB", stats.allocated_mb());
            println!("  Active: {:.2} MB", stats.active_mb());
            println!("  Resident: {:.2} MB", stats.resident_mb());
            println!("  Retained: {:.2} MB", stats.retained_mb());
            println!("  Overhead: {:.1}%", stats.overhead_pct());
        } else {
            println!("  (Statistics query failed)");
        }
        println!();
    }

    // 3. Run performance benchmarks
    println!("🚀 Performance Benchmarks");
    println!();

    println!("Benchmark 1: Small Allocations (1KB × 10,000)");
    let bench1 = AllocatorBenchmark::run(10_000, 1024);
    bench1.print();
    println!();

    println!("Benchmark 2: Medium Allocations (64KB × 1,000)");
    let bench2 = AllocatorBenchmark::run(1_000, 64 * 1024);
    bench2.print();
    println!();

    println!("Benchmark 3: Large Allocations (1MB × 100)");
    let bench3 = AllocatorBenchmark::run(100, 1024 * 1024);
    bench3.print();
    println!();

    println!("Benchmark 4: Tensor Chunks (256×256 f64 × 100)");
    let tensor_size = 256 * 256 * std::mem::size_of::<f64>();
    let bench4 = AllocatorBenchmark::run(100, tensor_size);
    bench4.print();
    println!();

    // 4. Demonstrate allocator pool
    println!("🔄 Allocator Pool Demo");
    let mut pool = AllocatorPool::new(1024 * 1024, 10); // 1MB buffers, 10 initial
    println!("  Pool created with {} buffers", pool.size());

    let buf1 = pool.acquire();
    println!("  Buffer acquired, pool size: {}", pool.size());

    pool.release(buf1);
    println!("  Buffer released, pool size: {}", pool.size());

    let pool_info = pool.allocator_info();
    println!("  Pool using allocator: {}", pool_info.name);
    println!();

    // 5. Memory usage simulation
    println!("🧪 Memory Usage Simulation");
    simulate_tensor_workload(&info);
    println!();

    // 6. Recommendations
    println!("💡 Recommendations");
    print_recommendations(&info);
}

fn simulate_tensor_workload(info: &AllocatorInfo) {
    println!("  Simulating realistic tensor workload...");

    // Simulate chunk-based tensor processing
    let chunk_size = 128 * 128 * std::mem::size_of::<f64>();
    let num_chunks = 50;

    let start = std::time::Instant::now();

    // Allocate chunks
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        chunks.push(vec![0u8; chunk_size]);
    }

    // Process chunks (simulate work)
    for chunk in &mut chunks {
        for byte in chunk.iter_mut() {
            *byte = byte.wrapping_add(1);
        }
    }

    let duration = start.elapsed();
    let total_mb = (num_chunks * chunk_size) as f64 / (1024.0 * 1024.0);

    println!(
        "  Processed {:.1} MB in {:.2} ms",
        total_mb,
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "  Throughput: {:.2} MB/s",
        total_mb / duration.as_secs_f64()
    );

    // Check stats after workload
    if info.has_stats {
        if let Some(stats) = get_allocator_stats() {
            println!("  Post-workload memory:");
            println!("    Active: {:.2} MB", stats.active_mb());
            println!("    Resident: {:.2} MB", stats.resident_mb());
        }
    }
}

fn print_recommendations(info: &AllocatorInfo) {
    match info.allocator_type {
        tenrso_ooc::allocators::AllocatorType::System => {
            println!("  ⚠️  Using system allocator (default)");
            println!("  Consider using jemalloc or mimalloc for:");
            println!("    • Multi-threaded tensor operations");
            println!("    • Long-running processes");
            println!("    • High allocation/deallocation rates");
            println!();
            println!("  To enable jemalloc:");
            println!("    cargo build --features jemalloc");
            println!();
            println!("  To enable mimalloc:");
            println!("    cargo build --features mimalloc-allocator");
        }
        tenrso_ooc::allocators::AllocatorType::Jemalloc => {
            println!("  ✅ Using jemalloc - excellent choice for:");
            println!("    • Multi-threaded workloads");
            println!("    • Large tensor operations");
            println!("    • Long-running processes");
            println!("    • Production deployments");
            println!();
            println!("  Tips:");
            println!("    • Enable profiling with MALLOC_CONF=prof:true");
            println!("    • Monitor statistics for memory optimization");
        }
        tenrso_ooc::allocators::AllocatorType::Mimalloc => {
            println!("  ✅ Using mimalloc - excellent choice for:");
            println!("    • High allocation/deallocation rates");
            println!("    • Small to medium chunks");
            println!("    • Single-threaded workloads");
            println!("    • Low-latency requirements");
            println!();
            println!("  Note:");
            println!("    • Best for workloads with frequent small allocations");
            println!("    • May use more memory for very large tensors");
        }
    }
}
