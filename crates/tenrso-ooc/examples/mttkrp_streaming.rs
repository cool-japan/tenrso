//! Out-of-core (streaming, chunked) MTTKRP.
//!
//! MTTKRP is the inner kernel of CP-ALS. This example drives the *real* streaming
//! implementation ([`tenrso_ooc::StreamingMttkrp`]) — the tensor is written to a file, and
//! the result is computed by reading it back one chunk at a time under a hard memory
//! budget, accumulating each chunk's `I_n × R` contribution.
//!
//! It demonstrates:
//!
//! 1. Streaming from RAM vs from a file on disk, checked against the in-core kernel.
//! 2. The effect of the chunk grid on speed (and on the last bits of the answer).
//! 3. A **full CP-ALS sweep in one disk pass**: all N modes accumulated together.
//! 4. A tensor computed successfully under a budget **far smaller than the tensor**.
//!
//! ```bash
//! cargo run --release --example mttkrp_streaming
//! ```

use std::time::Instant;

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};
use tenrso_core::DenseND;
use tenrso_kernels::mttkrp;
use tenrso_ooc::chunk_source::{DenseChunkSource, MmapChunkSource};
use tenrso_ooc::mmap_io::write_tensor_binary;
use tenrso_ooc::{ChunkSpec, MttkrpStreamConfig, StreamingMttkrp};

fn random_tensor(shape: &[usize], seed: u64) -> Result<DenseND<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    DenseND::from_vec(data, shape)
}

fn random_factors(shape: &[usize], rank: usize, seed: u64) -> Result<Vec<Array2<f64>>> {
    let mut rng = StdRng::seed_from_u64(seed);
    shape
        .iter()
        .map(|&dim| {
            let data: Vec<f64> = (0..dim * rank)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            Ok(Array2::from_shape_vec((dim, rank), data)?)
        })
        .collect()
}

fn rel_error(got: &Array2<f64>, want: &Array2<f64>) -> f64 {
    let num: f64 = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let den: f64 = want.iter().map(|b| b * b).sum::<f64>().sqrt();
    if den == 0.0 {
        num
    } else {
        num / den
    }
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<()> {
    let shape = vec![96, 80, 64]; // 491_520 elements = 3.75 MiB of f64
    let rank = 16;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Out-of-core MTTKRP");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let tensor = random_tensor(&shape, 42)?;
    let factors = random_factors(&shape, rank, 7)?;
    let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();
    let tensor_bytes = tensor.len() * 8;

    println!("Tensor  : {:?}  ({:.2} MiB, f64)", shape, mib(tensor_bytes));
    println!("CP rank : {}", rank);
    println!();

    // In-core reference for every mode.
    let x = tensor.as_array().view();
    let started = Instant::now();
    let reference: Vec<Array2<f64>> = (0..shape.len())
        .map(|m| mttkrp(&x, &views, m))
        .collect::<Result<_>>()?;
    let in_core_secs = started.elapsed().as_secs_f64();
    println!(
        "In-core reference (N separate mttkrp calls): {:.4}s",
        in_core_secs
    );
    println!();

    // ---------------------------------------------------------------------------------
    // 1. Streaming from RAM, several chunk grids. Isolates streaming overhead from I/O.
    // ---------------------------------------------------------------------------------
    println!("── Streaming from RAM (no disk in the loop) ───────────────");
    println!();
    println!(
        "{:<18} {:>8} {:>8} {:>10} {:>12}",
        "chunk grid", "chunks", "window", "time", "rel err"
    );

    let dense_source = DenseChunkSource::new(tensor.clone())?;
    for chunk in [vec![48, 40, 32], vec![24, 20, 16], vec![13, 11, 7]] {
        let spec = ChunkSpec::tile_size(&shape, &chunk)?;
        let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(32));

        // Warm up (first touch of the accumulators / page cache) so the printed time is
        // steady-state rather than first-run noise.
        let _ = exec.mttkrp_all_modes(&dense_source, &spec, &views)?;

        let started = Instant::now();
        let out = exec.mttkrp_all_modes(&dense_source, &spec, &views)?;
        let secs = started.elapsed().as_secs_f64();

        let stats = exec.stats().expect("a completed run records stats");
        let worst = (0..shape.len())
            .map(|m| rel_error(&out[m], &reference[m]))
            .fold(0.0f64, f64::max);

        println!(
            "{:<18} {:>8} {:>8} {:>9.4}s {:>12.2e}",
            format!("{:?}", chunk),
            stats.chunks_processed,
            stats.plan.window_chunks,
            secs,
            worst
        );
    }
    println!();
    println!("  The last column is the price of regrouping a floating-point sum:");
    println!("  chunking changes the summation order, never the answer.");
    println!();

    // ---------------------------------------------------------------------------------
    // 2. The real out-of-core path: a file on disk, streamed under a tight budget.
    // ---------------------------------------------------------------------------------
    println!("── Streaming from a file on disk ──────────────────────────");
    println!();

    let path = std::env::temp_dir().join("tenrso_example_mttkrp_stream.bin");
    write_tensor_binary(&path, &tensor)?;
    println!("Wrote {} ({:.2} MiB)", path.display(), mib(tensor_bytes));
    println!();

    let source = MmapChunkSource::open(&path)?;
    let spec = ChunkSpec::tile_size(&shape, &[24, 20, 16])?;

    println!(
        "{:<20} {:>12} {:>7} {:>8} {:>9} {:>12} {:>10}",
        "chunk grid", "budget", "window", "windows", "time", "peak w. set", "rel err"
    );

    // A tighter budget needs a finer grid: a chunk must fit inside the budget, so below
    // ~100 KiB the [24, 20, 16] grid becomes infeasible and the run *errors out* (it does
    // not silently over-allocate). The [8, 8, 8] grid keeps going down to ~37 KiB.
    for (chunk, budget) in [
        (vec![24, 20, 16], 4 * 1024 * 1024), // more than the tensor: one big window
        (vec![24, 20, 16], 512 * 1024),
        (vec![24, 20, 16], 128 * 1024),
        (vec![8, 8, 8], 48 * 1024), // 1.25% of the tensor
    ] {
        let spec = ChunkSpec::tile_size(&shape, &chunk)?;
        let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_bytes(budget));

        let _ = exec.mttkrp_all_modes(&source, &spec, &views)?; // warm the page cache

        let started = Instant::now();
        let out = exec.mttkrp_all_modes(&source, &spec, &views)?;
        let secs = started.elapsed().as_secs_f64();

        let stats = exec.stats().expect("a completed run records stats");
        let worst = (0..shape.len())
            .map(|m| rel_error(&out[m], &reference[m]))
            .fold(0.0f64, f64::max);

        // The bound is not a hope: assert it.
        assert!(
            stats.peak_working_set_bytes() <= budget,
            "peak working set {} exceeded the budget {}",
            stats.peak_working_set_bytes(),
            budget
        );

        println!(
            "{:<20} {:>8.0} KiB {:>7} {:>8} {:>8.4}s {:>8.0} KiB {:>10.2e}",
            format!("{:?}", chunk),
            budget as f64 / 1024.0,
            stats.plan.window_chunks,
            stats.plan.num_windows(),
            secs,
            stats.peak_working_set_bytes() as f64 / 1024.0,
            worst
        );
    }
    println!();
    println!(
        "  A {:.2} MiB tensor is fully processed inside a 48 KiB working set.",
        mib(tensor_bytes)
    );
    println!("  The accumulators are I_n x R (30 KiB here) — they do not grow with the");
    println!("  tensor, which is the whole reason this streams. That also sets the floor:");
    println!("  no budget below the size of the answer itself can ever work, and asking");
    println!("  for one is a hard error, never a silent over-allocation.");
    println!();

    // ---------------------------------------------------------------------------------
    // 3. A CP-ALS sweep: N passes vs one all-modes pass.
    // ---------------------------------------------------------------------------------
    println!("── A full CP-ALS sweep (all N modes) ──────────────────────");
    println!();

    let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(8));

    let started = Instant::now();
    for mode in 0..shape.len() {
        let _ = exec.mttkrp(&source, &spec, &views, mode)?;
    }
    let n_pass_secs = started.elapsed().as_secs_f64();
    let n_pass_bytes = shape.len() * tensor_bytes;

    let started = Instant::now();
    let one_pass = exec.mttkrp_all_modes(&source, &spec, &views)?;
    let one_pass_secs = started.elapsed().as_secs_f64();
    let one_pass_bytes = exec
        .stats()
        .expect("a completed run records stats")
        .bytes_read;

    println!(
        "  {} single-mode passes : {:.4}s, {:.2} MiB read",
        shape.len(),
        n_pass_secs,
        mib(n_pass_bytes)
    );
    println!(
        "  one all-modes pass   : {:.4}s, {:.2} MiB read  ({:.2}x less I/O)",
        one_pass_secs,
        mib(one_pass_bytes),
        n_pass_bytes as f64 / one_pass_bytes as f64
    );
    println!();
    println!("  Every element contributes to every mode's accumulator, so one pass over");
    println!("  the data suffices; the dimension-tree kernel additionally shares partial");
    println!("  contractions across the modes, so it is cheaper in FLOPs too.");
    println!();

    let worst = (0..shape.len())
        .map(|m| rel_error(&one_pass[m], &reference[m]))
        .fold(0.0f64, f64::max);
    println!(
        "  worst-mode relative error vs in-core kernel: {:.2e}",
        worst
    );
    println!();

    drop(source);
    std::fs::remove_file(&path)?;

    println!("═══════════════════════════════════════════════════════════");
    Ok(())
}
