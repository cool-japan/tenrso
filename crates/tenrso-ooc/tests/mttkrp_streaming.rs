//! Integration tests for out-of-core (streaming) MTTKRP.
//!
//! The contract under test: for a tensor that fits in memory, the streaming MTTKRP must
//! agree with the in-core `tenrso_kernels::mttkrp` reference for **every mode**, under
//! every chunk grid — including grids that do not evenly divide the dimensions (ragged
//! final chunks), the all-ones grid, and the whole-tensor grid.
//!
//! Tolerance: chunking regroups a floating-point sum, so results differ from the in-core
//! kernel in the last bits. We assert a relative Frobenius error below `1e-13`, which is
//! ~2 orders of magnitude tighter than anything a real bug could sneak under (an index
//! offset error produces O(1) relative error, not O(1e-13)).

// `for mode in 0..N { ... mttkrp(x, views, mode) ... }` uses `mode` as the semantic mode
// argument to the kernel, not merely as an index into the result vector.
#![allow(clippy::needless_range_loop)]

use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};
use tenrso_core::DenseND;
use tenrso_ooc::chunk_source::{ArrowChunkStore, ChunkSource, DenseChunkSource, MmapChunkSource};
use tenrso_ooc::mmap_io::write_tensor_binary;
use tenrso_ooc::{streaming_mttkrp_all_modes, ChunkSpec, MttkrpStreamConfig, StreamingMttkrp};

/// Tightest tolerance that is still robust to fp regrouping.
const TOL: f64 = 1e-13;

fn random_tensor(shape: &[usize], seed: u64) -> DenseND<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    DenseND::from_vec(data, shape).expect("shape and data length agree")
}

fn random_factors(shape: &[usize], rank: usize, seed: u64) -> Vec<Array2<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    shape
        .iter()
        .map(|&dim| {
            let data: Vec<f64> = (0..dim * rank)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            Array2::from_shape_vec((dim, rank), data).expect("shape and data length agree")
        })
        .collect()
}

fn views(factors: &[Array2<f64>]) -> Vec<ArrayView2<'_, f64>> {
    factors.iter().map(|f| f.view()).collect()
}

fn rel_error(got: &Array2<f64>, want: &Array2<f64>) -> f64 {
    assert_eq!(got.dim(), want.dim(), "result shape mismatch");
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

/// Assert that streaming over `chunk_size` reproduces the in-core kernel for every mode,
/// via both the single-mode API and the one-pass all-modes API.
fn assert_matches_in_core(shape: &[usize], chunk_size: &[usize], rank: usize, seed: u64) {
    let tensor = random_tensor(shape, seed);
    let factors = random_factors(shape, rank, seed + 1);
    let fviews = views(&factors);

    let source = DenseChunkSource::new(tensor.clone()).expect("contiguous tensor");
    let spec = ChunkSpec::tile_size(shape, chunk_size).expect("valid chunk spec");
    let config = MttkrpStreamConfig::new().max_memory_mb(64);

    let (all_modes, stats) = streaming_mttkrp_all_modes(&source, &spec, &fviews, config.clone())
        .expect("streaming all-modes MTTKRP");
    assert_eq!(stats.chunks_processed, spec.total_chunks());
    assert_eq!(
        stats.bytes_read,
        tensor.len() * 8,
        "every element read once"
    );

    let x = tensor.as_array().view();
    for mode in 0..shape.len() {
        let reference = tenrso_kernels::mttkrp(&x, &fviews, mode).expect("in-core reference");

        let single = StreamingMttkrp::new(config.clone())
            .mttkrp(&source, &spec, &fviews, mode)
            .expect("streaming single-mode MTTKRP");

        let err_single = rel_error(&single, &reference);
        assert!(
            err_single < TOL,
            "shape {:?} chunk {:?} rank {} mode {}: single-mode streaming rel err {:e}",
            shape,
            chunk_size,
            rank,
            mode,
            err_single
        );

        let err_all = rel_error(&all_modes[mode], &reference);
        assert!(
            err_all < TOL,
            "shape {:?} chunk {:?} rank {} mode {}: all-modes streaming rel err {:e}",
            shape,
            chunk_size,
            rank,
            mode,
            err_all
        );
    }
}

#[test]
fn test_3rd_order_cubic_even_division() {
    assert_matches_in_core(&[8, 8, 8], &[4, 4, 4], 3, 11);
}

#[test]
fn test_3rd_order_asymmetric_shapes() {
    // Asymmetric shapes catch index-offset bugs that cubes hide: a transposed or
    // mode-confused offset cannot produce a shape-valid answer here by accident.
    for (shape, chunk) in [
        (vec![7, 3, 5], vec![3, 2, 2]),
        (vec![3, 11, 2], vec![2, 4, 1]),
        (vec![13, 2, 7], vec![5, 1, 3]),
    ] {
        for rank in [1usize, 2, 5] {
            assert_matches_in_core(&shape, &chunk, rank, 20 + rank as u64);
        }
    }
}

#[test]
fn test_ragged_final_chunks() {
    // Chunk sizes that do NOT evenly divide any dimension: every axis has a short final
    // chunk, so the corner chunk is short in all three axes at once.
    for (shape, chunk) in [
        (vec![10, 7, 5], vec![3, 3, 3]),
        (vec![9, 9, 9], vec![4, 4, 4]),
        (vec![5, 6, 7], vec![4, 5, 6]), // one chunk short by 1 on every axis
        (vec![11, 4, 6], vec![7, 3, 4]),
    ] {
        // Guard the premise of the test: every axis must actually be ragged, otherwise
        // this silently degenerates into the even-division case.
        assert!(
            shape.iter().zip(chunk.iter()).all(|(d, c)| d % c != 0),
            "shape {:?} / chunk {:?} is not ragged on every axis",
            shape,
            chunk
        );
        assert_matches_in_core(&shape, &chunk, 3, 33);
    }
}

#[test]
fn test_chunk_size_one_degenerate() {
    // Every element is its own chunk: maximum number of index offsets, maximum number of
    // accumulations. If any offset is wrong, this cannot possibly pass.
    assert_matches_in_core(&[4, 3, 5], &[1, 1, 1], 2, 44);
    assert_matches_in_core(&[3, 2, 4, 2], &[1, 1, 1, 1], 1, 45);
}

#[test]
fn test_chunk_size_whole_tensor_degenerate() {
    // A single chunk covering the whole tensor: the streaming path degenerates to one
    // in-core kernel call at offset zero.
    assert_matches_in_core(&[6, 5, 4], &[6, 5, 4], 4, 55);
    // Chunk size larger than the tensor is clamped, not padded.
    assert_matches_in_core(&[6, 5, 4], &[100, 100, 100], 4, 56);
}

#[test]
fn test_4th_order_non_cubic() {
    for (shape, chunk) in [
        (vec![4, 5, 3, 2], vec![3, 2, 2, 1]),
        (vec![2, 7, 3, 5], vec![1, 4, 2, 2]),
        (vec![6, 2, 5, 3], vec![4, 1, 3, 2]),
    ] {
        for rank in [1usize, 3] {
            assert_matches_in_core(&shape, &chunk, rank, 60 + rank as u64);
        }
    }
}

#[test]
fn test_rank_one() {
    // R = 1 collapses the Khatri-Rao product to an elementwise product of columns; a
    // rank-indexing bug that survives R > 1 tests often dies here (and vice versa).
    assert_matches_in_core(&[7, 3, 5], &[2, 2, 2], 1, 77);
    assert_matches_in_core(&[3, 4, 2, 5], &[2, 3, 1, 2], 1, 78);
}

#[test]
fn test_high_rank() {
    assert_matches_in_core(&[5, 4, 6], &[2, 3, 4], 12, 88);
}

/// Streaming must be reproducible bit-for-bit: same chunk grid => same bits, no matter how
/// many chunks are in flight, whether chunks are processed in parallel, or in what order
/// their reads complete.
#[test]
fn test_bitwise_determinism_across_schedules() {
    let shape = vec![9, 7, 5];
    let tensor = random_tensor(&shape, 101);
    let factors = random_factors(&shape, 4, 102);
    let fviews = views(&factors);
    let source = DenseChunkSource::new(tensor).expect("contiguous tensor");
    let spec = ChunkSpec::tile_size(&shape, &[4, 3, 2]).expect("valid chunk spec");

    let schedules = [
        MttkrpStreamConfig::new()
            .max_memory_mb(64)
            .max_window_chunks(1)
            .parallel(false),
        MttkrpStreamConfig::new()
            .max_memory_mb(64)
            .max_window_chunks(3)
            .parallel(false),
        MttkrpStreamConfig::new()
            .max_memory_mb(64)
            .max_window_chunks(1024)
            .parallel(true),
        // A budget so tight the window collapses to a single chunk.
        MttkrpStreamConfig::new()
            .max_memory_bytes(8 * 1024)
            .parallel(true),
    ];

    let mut baseline: Option<Vec<Array2<f64>>> = None;
    for config in schedules {
        let out = StreamingMttkrp::new(config)
            .mttkrp_all_modes(&source, &spec, &fviews)
            .expect("streaming MTTKRP");
        match &baseline {
            None => baseline = Some(out),
            Some(base) => {
                for mode in 0..shape.len() {
                    assert_eq!(
                        base[mode], out[mode],
                        "mode {} differs bitwise across execution schedules",
                        mode
                    );
                }
            }
        }
    }
}

/// The memory budget is a hard bound, and it holds for a tensor far larger than the budget.
#[test]
fn test_bounded_memory_tensor_exceeds_budget() {
    let shape = vec![40, 32, 24]; // 30_720 elements = 245_760 bytes
    let tensor = random_tensor(&shape, 200);
    let factors = random_factors(&shape, 6, 201);
    let fviews = views(&factors);
    let tensor_bytes = tensor.len() * 8;

    let source = DenseChunkSource::new(tensor.clone()).expect("contiguous tensor");
    let spec = ChunkSpec::tile_size(&shape, &[7, 6, 5]).expect("valid chunk spec");

    // 32 KiB budget for a 240 KiB tensor: an 7.7x over-subscription.
    let budget = 32 * 1024;
    assert!(
        tensor_bytes > 7 * budget,
        "budget must be a real constraint"
    );

    let (results, stats) = streaming_mttkrp_all_modes(
        &source,
        &spec,
        &fviews,
        MttkrpStreamConfig::new().max_memory_bytes(budget),
    )
    .expect("streaming MTTKRP under a tight budget");

    // The plan's bound, and the *observed* peak, both stay under the budget.
    assert!(
        stats.plan.peak_working_set_bytes() <= budget,
        "planned peak {} exceeds budget {}",
        stats.plan.peak_working_set_bytes(),
        budget
    );
    assert!(
        stats.peak_working_set_bytes() <= budget,
        "observed peak {} exceeds budget {}",
        stats.peak_working_set_bytes(),
        budget
    );
    assert!(stats.peak_chunk_bytes <= stats.plan.chunk_budget_bytes());
    assert_eq!(stats.chunks_processed, spec.total_chunks());
    assert_eq!(stats.bytes_read, tensor_bytes);

    // And it is still the right answer.
    let x = tensor.as_array().view();
    for mode in 0..shape.len() {
        let reference = tenrso_kernels::mttkrp(&x, &fviews, mode).expect("in-core reference");
        assert!(rel_error(&results[mode], &reference) < TOL);
    }
}

/// A budget that cannot hold even one chunk is an error, not a silent over-run.
#[test]
fn test_budget_too_small_is_an_error() {
    let shape = vec![8, 8, 8];
    let tensor = random_tensor(&shape, 300);
    let factors = random_factors(&shape, 4, 301);
    let fviews = views(&factors);
    let source = DenseChunkSource::new(tensor).expect("contiguous tensor");
    let spec = ChunkSpec::tile_size(&shape, &[8, 8, 8]).expect("valid chunk spec");

    let err = streaming_mttkrp_all_modes(
        &source,
        &spec,
        &fviews,
        MttkrpStreamConfig::new().max_memory_bytes(512),
    )
    .expect_err("a 512-byte budget cannot hold a 4096-byte chunk");
    let msg = err.to_string();
    assert!(msg.contains("too small"), "unhelpful error: {}", msg);
    assert!(
        msg.contains("minimum"),
        "error must state the minimum: {}",
        msg
    );
}

/// The real out-of-core path: a memory-mapped file on disk, never fully materialized.
#[test]
fn test_out_of_core_mmap_file() {
    let shape = vec![13, 7, 5]; // deliberately prime-ish and non-cubic
    let tensor = random_tensor(&shape, 400);
    let factors = random_factors(&shape, 3, 401);
    let fviews = views(&factors);

    let path = std::env::temp_dir().join("tenrso_test_mttkrp_stream_mmap.bin");
    write_tensor_binary(&path, &tensor).expect("write tensor file");

    let source = MmapChunkSource::open(&path).expect("mmap the tensor file");
    assert_eq!(source.shape(), shape.as_slice());
    assert_eq!(source.source_name(), "mmap-file");

    // Ragged grid (27 chunks) + a budget small enough that the window cannot hold them
    // all, so the file really is streamed in several passes of the window.
    let spec = ChunkSpec::tile_size(&shape, &[5, 3, 2]).expect("valid chunk spec");
    let (results, stats) = streaming_mttkrp_all_modes(
        &source,
        &spec,
        &fviews,
        MttkrpStreamConfig::new().max_memory_bytes(4 * 1024),
    )
    .expect("streaming MTTKRP from a memory-mapped file");

    assert_eq!(stats.source_name, "mmap-file");
    assert_eq!(stats.chunks_processed, spec.total_chunks());
    assert!(
        stats.plan.window_chunks < spec.total_chunks(),
        "budget should not fit the whole grid in one window"
    );
    assert!(
        stats.plan.num_windows() > 1,
        "budget should force >1 window, got {} window(s) of {} chunk(s)",
        stats.plan.num_windows(),
        stats.plan.window_chunks
    );
    assert!(stats.peak_working_set_bytes() <= 4 * 1024);

    let x = tensor.as_array().view();
    for mode in 0..shape.len() {
        let reference = tenrso_kernels::mttkrp(&x, &fviews, mode).expect("in-core reference");
        let err = rel_error(&results[mode], &reference);
        assert!(err < TOL, "mode {} from mmap file: rel err {:e}", mode, err);
    }

    drop(source);
    std::fs::remove_file(&path).expect("clean up tensor file");
}

/// The real out-of-core path over Arrow IPC: one record batch per chunk, random access via
/// the file footer, only one batch decoded at a time.
#[test]
fn test_out_of_core_arrow_chunk_store() {
    let shape = vec![9, 4, 6];
    let tensor = random_tensor(&shape, 500);
    let factors = random_factors(&shape, 5, 501);
    let fviews = views(&factors);

    let path = std::env::temp_dir().join("tenrso_test_mttkrp_stream_store.arrow");
    let meta_path = std::env::temp_dir().join("tenrso_test_mttkrp_stream_store.arrow.chunks.json");

    // A ragged grid, materialized on disk block by block.
    let spec = ChunkSpec::tile_size(&shape, &[4, 3, 4]).expect("valid chunk spec");
    ArrowChunkStore::create(&path, &tensor, &spec).expect("write arrow chunk store");

    let source = ArrowChunkStore::open(&path).expect("open arrow chunk store");
    assert_eq!(source.shape(), shape.as_slice());
    assert_eq!(source.source_name(), "arrow-chunk-store");
    // The store knows its own grid, recovered from the sidecar.
    assert_eq!(
        source.native_chunk_spec().expect("store has a native grid"),
        &spec
    );

    let (results, stats) = streaming_mttkrp_all_modes(
        &source,
        &spec,
        &fviews,
        MttkrpStreamConfig::new().max_memory_bytes(64 * 1024),
    )
    .expect("streaming MTTKRP from an Arrow chunk store");
    assert_eq!(stats.chunks_processed, spec.total_chunks());
    assert_eq!(stats.bytes_read, tensor.len() * 8);

    let x = tensor.as_array().view();
    for mode in 0..shape.len() {
        let reference = tenrso_kernels::mttkrp(&x, &fviews, mode).expect("in-core reference");
        let err = rel_error(&results[mode], &reference);
        assert!(
            err < TOL,
            "mode {} from arrow store: rel err {:e}",
            mode,
            err
        );
    }

    // Streaming a store with a grid it was not written with must fail loudly rather than
    // silently reading the wrong blocks.
    let other = ChunkSpec::tile_size(&shape, &[3, 2, 2]).expect("valid chunk spec");
    let err = StreamingMttkrp::new(MttkrpStreamConfig::new())
        .mttkrp_all_modes(&source, &other, &fviews)
        .expect_err("mismatched grid must be rejected");
    assert!(err.to_string().contains("fixed chunk grid"), "{}", err);

    drop(source);
    std::fs::remove_file(&path).expect("clean up arrow file");
    std::fs::remove_file(&meta_path).expect("clean up sidecar");
}

/// All three sources must agree with each other exactly (they see the same bytes and use
/// the same grid, so this is a bitwise assertion, not a tolerance one).
#[test]
fn test_sources_agree_bitwise() {
    let shape = vec![6, 5, 7];
    let tensor = random_tensor(&shape, 600);
    let factors = random_factors(&shape, 3, 601);
    let fviews = views(&factors);
    let spec = ChunkSpec::tile_size(&shape, &[4, 3, 3]).expect("valid chunk spec");

    let bin_path = std::env::temp_dir().join("tenrso_test_mttkrp_sources.bin");
    let arrow_path = std::env::temp_dir().join("tenrso_test_mttkrp_sources.arrow");
    let meta_path = std::env::temp_dir().join("tenrso_test_mttkrp_sources.arrow.chunks.json");
    write_tensor_binary(&bin_path, &tensor).expect("write tensor file");
    ArrowChunkStore::create(&arrow_path, &tensor, &spec).expect("write arrow chunk store");

    let dense = DenseChunkSource::new(tensor).expect("contiguous tensor");
    let mmap = MmapChunkSource::open(&bin_path).expect("mmap the tensor file");
    let arrow = ArrowChunkStore::open(&arrow_path).expect("open arrow chunk store");

    let config = MttkrpStreamConfig::new().max_memory_mb(8);
    let from_dense = StreamingMttkrp::new(config.clone())
        .mttkrp_all_modes(&dense, &spec, &fviews)
        .expect("dense source");
    let from_mmap = StreamingMttkrp::new(config.clone())
        .mttkrp_all_modes(&mmap, &spec, &fviews)
        .expect("mmap source");
    let from_arrow = StreamingMttkrp::new(config)
        .mttkrp_all_modes(&arrow, &spec, &fviews)
        .expect("arrow source");

    for mode in 0..shape.len() {
        assert_eq!(
            from_dense[mode], from_mmap[mode],
            "mode {} dense vs mmap",
            mode
        );
        assert_eq!(
            from_dense[mode], from_arrow[mode],
            "mode {} dense vs arrow",
            mode
        );
    }

    drop(mmap);
    drop(arrow);
    std::fs::remove_file(&bin_path).expect("clean up tensor file");
    std::fs::remove_file(&arrow_path).expect("clean up arrow file");
    std::fs::remove_file(&meta_path).expect("clean up sidecar");
}

/// A subset of modes is accumulated correctly and returned in ascending mode order.
#[test]
fn test_mode_subset() {
    let shape = vec![5, 6, 4, 3];
    let tensor = random_tensor(&shape, 700);
    let factors = random_factors(&shape, 2, 701);
    let fviews = views(&factors);
    let source = DenseChunkSource::new(tensor.clone()).expect("contiguous tensor");
    let spec = ChunkSpec::tile_size(&shape, &[2, 4, 3, 2]).expect("valid chunk spec");

    let requested = [3usize, 1]; // deliberately out of order
    let out = StreamingMttkrp::new(MttkrpStreamConfig::new())
        .mttkrp_modes(&source, &spec, &fviews, &requested)
        .expect("streaming MTTKRP over a mode subset");
    assert_eq!(out.len(), 2);

    let x = tensor.as_array().view();
    // Results come back ascending: [mode 1, mode 3].
    for (i, mode) in [1usize, 3].iter().enumerate() {
        let reference = tenrso_kernels::mttkrp(&x, &fviews, *mode).expect("in-core reference");
        assert!(rel_error(&out[i], &reference) < TOL, "mode {}", mode);
    }
}

/// Different chunk grids agree with each other (and with the reference) to tolerance — the
/// documented consequence of regrouping the sum.
#[test]
fn test_chunk_grid_independence_to_tolerance() {
    let shape = vec![12, 8, 6];
    let tensor = random_tensor(&shape, 800);
    let factors = random_factors(&shape, 4, 801);
    let fviews = views(&factors);
    let source = DenseChunkSource::new(tensor.clone()).expect("contiguous tensor");

    let x = tensor.as_array().view();
    let references: Vec<Array2<f64>> = (0..shape.len())
        .map(|m| tenrso_kernels::mttkrp(&x, &fviews, m).expect("in-core reference"))
        .collect();

    for chunk in [
        vec![12, 8, 6],
        vec![6, 4, 3],
        vec![5, 3, 4],
        vec![1, 8, 6],
        vec![12, 1, 1],
        vec![1, 1, 1],
    ] {
        let spec = ChunkSpec::tile_size(&shape, &chunk).expect("valid chunk spec");
        let out = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(16))
            .mttkrp_all_modes(&source, &spec, &fviews)
            .expect("streaming MTTKRP");
        for mode in 0..shape.len() {
            let err = rel_error(&out[mode], &references[mode]);
            assert!(
                err < TOL,
                "chunk {:?} mode {}: rel err {:e} exceeds tolerance",
                chunk,
                mode,
                err
            );
        }
    }
}
