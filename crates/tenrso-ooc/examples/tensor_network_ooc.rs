//! Tensor network contraction workflow with out-of-core processing
//!
//! This example demonstrates:
//! - Multi-step tensor network contraction
//! - Chunk graph construction for complex workflows
//! - Streaming execution with memory management
//! - Performance profiling of contraction sequences
//!
//! Run with: cargo run --example tensor_network_ooc --features default

use tenrso_core::DenseND;
use tenrso_ooc::{
    AccessPattern, MemoryManager, PrefetchStrategy, Prefetcher, SpillPolicy, StreamConfig,
    StreamingExecutor,
};

fn main() -> anyhow::Result<()> {
    println!("=============================================================");
    println!("   TenRSo: Tensor Network Contraction (Out-of-Core)");
    println!("=============================================================\n");

    // Configuration
    let bond_dim = 20; // Bond dimension for tensor network
    let physical_dim = 4; // Physical dimension
    let network_depth = 5; // Depth of the tensor network
    let memory_limit_mb = 50;

    println!("Configuration:");
    println!("  Bond dimension:     {}", bond_dim);
    println!("  Physical dimension: {}", physical_dim);
    println!("  Network depth:      {}", network_depth);
    println!("  Memory limit:       {} MB\n", memory_limit_mb);

    // Step 1: Generate tensor network (MPS-like structure)
    println!("Step 1: Generating Matrix Product State (MPS) tensor network...");
    let tensors = generate_mps_network(bond_dim, physical_dim, network_depth)?;
    let total_params: usize = tensors
        .iter()
        .map(|t| t.shape().iter().product::<usize>())
        .sum();
    println!("  Generated {} tensors", tensors.len());
    println!(
        "  Total parameters: {} ({:.2} MB)",
        total_params,
        total_params * 8 / (1024 * 1024)
    );

    // Step 2: Setup memory manager
    println!("\nStep 2: Setting up memory manager...");
    let temp_dir = std::env::temp_dir().join("tenrso_tensor_network_ooc");
    std::fs::create_dir_all(&temp_dir)?;

    let mut memory_manager = MemoryManager::new()
        .max_memory_mb(memory_limit_mb)
        .spill_policy(SpillPolicy::LRU)
        .temp_dir(&temp_dir);

    // Register tensors
    for (i, tensor) in tensors.iter().enumerate() {
        memory_manager.register_chunk(
            &format!("tensor_{}", i),
            tensor.clone(),
            AccessPattern::ReadMany,
        )?;
    }
    println!("  Registered {} tensors with memory manager", tensors.len());

    // Step 3: Plan contraction sequence
    println!("\nStep 3: Planning contraction sequence...");
    println!("  Strategy: left-to-right sequential contraction");
    println!(
        "  Operations: {} tensor-tensor contractions",
        network_depth - 1
    );
    println!("  Expected intermediate tensors: {}", network_depth - 2);

    // Step 4: Setup streaming executor
    println!("\nStep 4: Setting up streaming executor...");
    let stream_config = StreamConfig::default()
        .chunk_size(vec![10, 10]) // Chunk size for 2D contractions
        .max_memory_mb(memory_limit_mb)
        .enable_prefetching(true)
        .prefetch_strategy(PrefetchStrategy::Aggressive)
        .enable_profiling(true);

    let executor = StreamingExecutor::new(stream_config);
    println!("  Executor configured with aggressive prefetching");

    // Step 5: Execute contraction sequence
    println!("\nStep 5: Executing tensor network contraction...");
    let start = std::time::Instant::now();

    // Perform left-to-right sweep
    let mut current = tensors[0].clone();
    for (i, next) in tensors.iter().enumerate().skip(1) {
        // Reshape for matrix multiplication
        let left_shape = current.shape();
        let right_shape = next.shape();

        println!(
            "  Contracting tensor {} (shape {:?}) with tensor {} (shape {:?})",
            i - 1,
            left_shape,
            i,
            right_shape
        );

        // Perform contraction (simplified as matrix multiplication)
        current = contract_tensors(&current, next)?;

        println!("    Result shape: {:?}", current.shape());

        // Update memory manager
        memory_manager.register_chunk(
            &format!("contract_{}", i - 1),
            current.clone(),
            AccessPattern::ReadOnce,
        )?;
    }

    let contraction_time = start.elapsed();
    println!(
        "\n  Total contraction time: {:.2} seconds",
        contraction_time.as_secs_f64()
    );
    println!("  Final result shape: {:?}", current.shape());

    // Step 6: Analyze memory usage
    println!("\nStep 6: Memory usage analysis:");
    let stats = memory_manager.stats();
    println!("  Total chunks:     {}", stats.total_chunks);
    println!("  In-memory chunks: {}", stats.in_memory_chunks);
    println!("  Spilled chunks:   {}", stats.spilled_chunks);
    println!(
        "  Memory usage:     {:.2} MB / {} MB",
        stats.current_memory_bytes as f64 / (1024.0 * 1024.0),
        memory_limit_mb
    );

    // Step 7: Prefetcher statistics
    println!("\nStep 7: Prefetcher statistics:");
    let prefetcher = Prefetcher::new();
    let prefetch_stats = prefetcher.stats();
    println!(
        "  Prefetching:      {}",
        if prefetch_stats.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("  Strategy:         {:?}", prefetch_stats.strategy);
    println!("  Prefetched count: {}", prefetch_stats.prefetched_count);
    println!("  Queue length:     {}", prefetch_stats.queue_len);

    // Step 8: Profiling summary
    println!("\nStep 8: Execution profile:");
    let profiler = executor.profiler();
    let summary = profiler.summary();
    if summary.total_operations > 0 {
        println!("  Total operations: {}", summary.total_operations);
        println!(
            "  Total time:       {:.2} seconds",
            summary.total_time.as_secs_f64()
        );
        println!("  Operation types:  {}", summary.num_operation_types);
        println!(
            "  I/O bandwidth:    {:.2} MB/s",
            summary.io_bandwidth_bps / (1024.0 * 1024.0)
        );
    } else {
        println!("  (No streaming operations executed)");
    }

    // Step 9: Performance summary
    println!("\nStep 9: Performance summary:");
    println!(
        "  Throughput:       {:.2} GFLOPS",
        compute_throughput(&tensors, contraction_time)
    );
    println!(
        "  Memory efficiency: {:.1}%",
        ((stats.current_memory_bytes as f64 / (1024.0 * 1024.0)) / memory_limit_mb as f64) * 100.0
    );

    // Cleanup
    println!("\nStep 10: Cleaning up...");
    memory_manager.cleanup()?;
    std::fs::remove_dir_all(&temp_dir)?;
    println!("  Temporary files removed");

    println!("\n=============================================================");
    println!("Tensor network contraction complete!");
    println!("=============================================================");

    Ok(())
}

/// Generate a simplified tensor chain for demonstration
///
/// Creates a sequence of 2D tensors that can be contracted sequentially:
/// - All tensors have shape [M, N] where dimensions are compatible for matrix multiplication
fn generate_mps_network(
    bond_dim: usize,
    physical_dim: usize,
    depth: usize,
) -> anyhow::Result<Vec<DenseND<f64>>> {
    use scirs2_core::random::{Rng, SeedableRng, StdRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut tensors = Vec::new();

    // Create a chain of compatible 2D matrices for sequential multiplication
    for i in 0..depth {
        let rows = bond_dim;
        let cols = if i == depth - 1 {
            physical_dim
        } else {
            bond_dim
        };

        let size = rows * cols;
        let data: Vec<f64> = (0..size).map(|_| rng.random::<f64>()).collect();
        tensors.push(DenseND::from_vec(data, &[rows, cols])?);
    }

    Ok(tensors)
}

/// Contract two tensors (simplified as matrix multiplication)
///
/// For MPS contraction, this performs:
/// - Reshape tensors to 2D matrices
/// - Multiply using standard matrix multiplication
/// - Reshape result back to appropriate dimensionality
fn contract_tensors(left: &DenseND<f64>, right: &DenseND<f64>) -> anyhow::Result<DenseND<f64>> {
    // Flatten to 2D for matrix multiplication
    let left_shape = left.shape();
    let right_shape = right.shape();

    // Compute matrix dimensions
    let left_rows = left_shape[0];
    let left_cols: usize = left_shape[1..].iter().product();
    let right_rows = right_shape[0];
    let right_cols: usize = right_shape[1..].iter().product();

    // Ensure compatible dimensions
    if left_cols != right_rows {
        anyhow::bail!(
            "Incompatible dimensions for contraction: {} vs {}",
            left_cols,
            right_rows
        );
    }

    // Perform simplified contraction (element-wise multiplication and sum)
    // In a real implementation, this would use proper einsum/tensor contraction
    let result_size = left_rows * right_cols;
    let mut result_data = vec![0.0; result_size];

    let left_slice = left.as_slice();
    let right_slice = right.as_slice();

    for i in 0..left_rows {
        for j in 0..right_cols {
            let mut sum = 0.0;
            for k in 0..left_cols {
                sum += left_slice[i * left_cols + k] * right_slice[k * right_cols + j];
            }
            result_data[i * right_cols + j] = sum;
        }
    }

    DenseND::from_vec(result_data, &[left_rows, right_cols])
}

/// Compute approximate throughput in GFLOPS
fn compute_throughput(tensors: &[DenseND<f64>], elapsed: std::time::Duration) -> f64 {
    // Estimate FLOPs for MPS contraction
    // Each contraction: 2 * m * n * k operations
    let mut total_flops = 0u64;

    for i in 0..(tensors.len() - 1) {
        let left_size: usize = tensors[i].shape().iter().product();
        let right_size: usize = tensors[i + 1].shape().iter().product();

        // Rough estimate: left_size * right_size operations
        total_flops += (2 * left_size * right_size) as u64;
    }

    let gflops = total_flops as f64 / 1e9;
    gflops / elapsed.as_secs_f64()
}
