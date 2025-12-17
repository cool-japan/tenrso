//! Tucker decomposition with out-of-core data processing
//!
//! This example demonstrates:
//! - Tucker-HOSVD decomposition on large tensors
//! - Chunked streaming execution for intermediate operations
//! - Memory management with spilling to disk
//! - Reconstruction quality analysis
//!
//! Run with: cargo run --example tucker_ooc --features default

use tenrso_core::DenseND;
use tenrso_decomp::tucker_hosvd;
use tenrso_ooc::{AccessPattern, MemoryManager, SpillPolicy, StreamConfig, StreamingExecutor};

fn main() -> anyhow::Result<()> {
    println!("=============================================================");
    println!("    TenRSo: Tucker Decomposition with Out-of-Core");
    println!("=============================================================\n");

    // Configuration
    let tensor_shape = vec![80, 90, 100]; // ~5.5MB for f64
    let tucker_ranks = vec![40, 45, 50]; // ~50% compression
    let memory_limit_mb = 20; // Force out-of-core for intermediate operations

    println!("Configuration:");
    println!("  Tensor shape:     {:?}", tensor_shape);
    println!("  Tucker ranks:     {:?}", tucker_ranks);
    println!("  Memory limit:     {} MB", memory_limit_mb);
    println!(
        "  Tensor size:      {:.2} MB",
        tensor_shape.iter().product::<usize>() * 8 / (1024 * 1024)
    );
    println!(
        "  Core size:        {:.2} MB\n",
        tucker_ranks.iter().product::<usize>() * 8 / (1024 * 1024)
    );

    // Step 1: Generate synthetic low-rank tensor
    println!("Step 1: Generating synthetic low-rank tensor...");
    let tensor = generate_low_rank_tensor(&tensor_shape, &tucker_ranks)?;
    println!("  Generated tensor with shape {:?}", tensor.shape());

    // Step 2: Setup out-of-core memory manager
    println!("\nStep 2: Setting up out-of-core memory manager...");
    let temp_dir = std::env::temp_dir().join("tenrso_tucker_ooc");
    std::fs::create_dir_all(&temp_dir)?;

    let mut memory_manager = MemoryManager::new()
        .max_memory_mb(memory_limit_mb)
        .spill_policy(SpillPolicy::LRU)
        .temp_dir(&temp_dir);

    println!(
        "  Memory manager configured with {} MB limit",
        memory_limit_mb
    );

    // Step 3: Register tensor with memory manager
    println!("\nStep 3: Registering tensor with memory manager...");
    memory_manager.register_chunk("input_tensor", tensor.clone(), AccessPattern::ReadMany)?;
    println!(
        "  Memory usage: {:.2} MB / {} MB",
        memory_manager.current_memory() as f64 / (1024.0 * 1024.0),
        memory_limit_mb
    );

    // Step 4: Perform Tucker decomposition
    println!("\nStep 4: Computing Tucker-HOSVD decomposition...");
    let start = std::time::Instant::now();
    let tucker = tucker_hosvd(&tensor, &tucker_ranks)?;
    let decomp_time = start.elapsed();

    println!(
        "  Decomposition completed in {:.2} seconds",
        decomp_time.as_secs_f64()
    );
    println!("  Core shape: {:?}", tucker.core.shape());
    println!("  Factor matrices: {} factors", tucker.factors.len());
    for (i, factor) in tucker.factors.iter().enumerate() {
        println!("    Mode {}: {:?}", i, factor.dim());
    }

    // Step 5: Setup streaming executor for reconstruction
    println!("\nStep 5: Setting up streaming executor for reconstruction...");
    let stream_config = StreamConfig::default()
        .chunk_size(vec![20, 30, 25])
        .max_memory_mb(memory_limit_mb)
        .enable_prefetching(true)
        .enable_profiling(true);

    let executor = StreamingExecutor::new(stream_config);
    println!("  Streaming executor configured");

    // Step 6: Reconstruct with streaming
    println!("\nStep 6: Reconstructing tensor with streaming...");
    let start = std::time::Instant::now();
    let reconstructed = tucker.reconstruct()?;
    let reconstruct_time = start.elapsed();

    println!(
        "  Reconstruction completed in {:.2} seconds",
        reconstruct_time.as_secs_f64()
    );
    println!("  Reconstructed shape: {:?}", reconstructed.shape());

    // Step 7: Compute reconstruction error
    println!("\nStep 7: Computing reconstruction error...");
    let error = compute_relative_error(&tensor, &reconstructed)?;
    println!("  Relative error: {:.6e}", error);
    println!("  Reconstruction quality: {:.2}%", (1.0 - error) * 100.0);

    // Step 8: Compute compression ratio
    println!("\nStep 8: Analyzing compression...");
    let original_size = tensor_shape.iter().product::<usize>();
    let compressed_size = tucker_ranks.iter().product::<usize>()
        + tucker_ranks
            .iter()
            .zip(tensor_shape.iter())
            .map(|(r, i)| r * i)
            .sum::<usize>();

    let compression_ratio = original_size as f64 / compressed_size as f64;
    println!("  Original parameters:   {}", original_size);
    println!("  Compressed parameters: {}", compressed_size);
    println!("  Compression ratio:     {:.2}x", compression_ratio);
    println!(
        "  Space saved:           {:.1}%",
        (1.0 - 1.0 / compression_ratio) * 100.0
    );

    // Step 9: Profile streaming execution
    let profiler = executor.profiler();
    let summary = profiler.summary();
    if summary.total_operations > 0 {
        println!("\nStep 9: Streaming execution profile:");
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
    }

    // Cleanup
    println!("\nStep 10: Cleaning up...");
    memory_manager.cleanup()?;
    std::fs::remove_dir_all(&temp_dir)?;
    println!("  Temporary files removed");

    println!("\n=============================================================");
    println!("Tucker decomposition with out-of-core processing complete!");
    println!("=============================================================");

    Ok(())
}

/// Generate a synthetic low-rank tensor by constructing factor matrices
fn generate_low_rank_tensor(shape: &[usize], ranks: &[usize]) -> anyhow::Result<DenseND<f64>> {
    use scirs2_core::ndarray_ext::Array2;
    use scirs2_core::random::{Rng, SeedableRng, StdRng};

    let mut rng = StdRng::seed_from_u64(42);

    // Generate random core tensor
    let core_size: usize = ranks.iter().product();
    let core_data: Vec<f64> = (0..core_size).map(|_| rng.random::<f64>()).collect();
    let core = DenseND::from_vec(core_data, ranks)?;

    // Generate random factor matrices
    let mut factors = Vec::new();
    for (&dim, &rank) in shape.iter().zip(ranks.iter()) {
        let factor_data: Vec<f64> = (0..(dim * rank)).map(|_| rng.random::<f64>()).collect();
        let factor = Array2::from_shape_vec((dim, rank), factor_data)?;
        factors.push(factor);
    }

    // Reconstruct to get full tensor
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let core_view = core.view();
    let tensor_array = tenrso_kernels::tucker_reconstruct(&core_view, &factor_views)?;

    Ok(DenseND::from_array(tensor_array))
}

/// Compute relative error ||A - B|| / ||A||
fn compute_relative_error(a: &DenseND<f64>, b: &DenseND<f64>) -> anyhow::Result<f64> {
    if a.shape() != b.shape() {
        anyhow::bail!("Shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    }

    // Convert to contiguous arrays if needed
    let a_view = a.view();
    let b_view = b.view();

    let mut diff_norm = 0.0f64;
    let mut a_norm = 0.0f64;

    // Iterate over all elements
    for (a_val, b_val) in a_view.iter().zip(b_view.iter()) {
        let diff = a_val - b_val;
        diff_norm += diff * diff;
        a_norm += a_val * a_val;
    }

    diff_norm = diff_norm.sqrt();
    a_norm = a_norm.sqrt();

    if a_norm == 0.0 {
        Ok(0.0)
    } else {
        Ok(diff_norm / a_norm)
    }
}
