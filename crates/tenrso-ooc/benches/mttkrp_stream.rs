//! Benchmarks for out-of-core (streaming) MTTKRP.
//!
//! What is being measured, and what is honestly *not*:
//!
//! * `in_core_*` are the `tenrso-kernels` baselines on a tensor already in RAM.
//! * `stream_dense_*` streams from a `DenseChunkSource` — the tensor is in RAM, so there is
//!   no disk in the loop. This isolates the **pure streaming overhead** (sub-box copies,
//!   per-chunk kernel invocations, accumulation) from I/O. Any speedup or slowdown here is
//!   compute.
//! * `stream_mmap_*` streams from a memory-mapped file. After the first pass the file is in
//!   the OS page cache, so this measures cached-file I/O plus streaming overhead, **not**
//!   cold-disk throughput. Cold-disk numbers are a property of the storage device, not of
//!   this code, and are deliberately not reported as if they were compute.
//! * `sweep_*` compares a full CP-ALS sweep done as `N` single-mode passes against the
//!   one-pass all-modes path, which is the reason the all-modes path exists.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use tenrso_core::DenseND;
use tenrso_ooc::chunk_source::{DenseChunkSource, MmapChunkSource};
use tenrso_ooc::mmap_io::write_tensor_binary;
use tenrso_ooc::{ChunkSpec, MttkrpStreamConfig, StreamingMttkrp};

const RANK: usize = 16;

fn tensor(shape: &[usize]) -> DenseND<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize % 1000) as f64) / 1000.0 - 0.5)
        .collect();
    DenseND::from_vec(data, shape).unwrap()
}

fn factors(shape: &[usize], rank: usize) -> Vec<Array2<f64>> {
    shape
        .iter()
        .enumerate()
        .map(|(k, &dim)| {
            Array2::from_shape_fn((dim, rank), |(i, r)| {
                (((i * 31 + r * 17 + k * 7) % 97) as f64) / 97.0 - 0.5
            })
        })
        .collect()
}

/// Streaming overhead vs the in-core kernel, tensor fully in RAM (no disk in the loop).
fn bench_streaming_overhead(c: &mut Criterion) {
    let shape = vec![80, 64, 48];
    let x = tensor(&shape);
    let fs = factors(&shape, RANK);
    let views: Vec<ArrayView2<f64>> = fs.iter().map(|f| f.view()).collect();
    let source = DenseChunkSource::new(x.clone()).unwrap();
    let elems: usize = shape.iter().product();

    let mut group = c.benchmark_group("mttkrp_mode0");
    group.throughput(Throughput::Elements(elems as u64));
    group.measurement_time(Duration::from_secs(6));

    group.bench_function("in_core", |b| {
        let view = x.as_array().view();
        b.iter(|| tenrso_kernels::mttkrp(&view, &views, 0).unwrap());
    });

    for chunk in [16usize, 32, 64] {
        let spec = ChunkSpec::tile_size(&shape, &[chunk, chunk, chunk]).unwrap();
        group.bench_with_input(
            BenchmarkId::new("stream_dense", format!("chunk{}", chunk)),
            &spec,
            |b, spec| {
                let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(64));
                b.iter(|| exec.mttkrp(&source, spec, &views, 0).unwrap());
            },
        );
    }

    group.finish();
}

/// One-pass all-modes (dimension tree) vs N separate single-mode passes: the CP-ALS sweep.
fn bench_sweep(c: &mut Criterion) {
    let shape = vec![80, 64, 48];
    let x = tensor(&shape);
    let fs = factors(&shape, RANK);
    let views: Vec<ArrayView2<f64>> = fs.iter().map(|f| f.view()).collect();
    let source = DenseChunkSource::new(x.clone()).unwrap();
    let spec = ChunkSpec::tile_size(&shape, &[32, 32, 32]).unwrap();
    let elems: usize = shape.iter().product();

    let mut group = c.benchmark_group("cp_als_sweep");
    group.throughput(Throughput::Elements(elems as u64));
    group.measurement_time(Duration::from_secs(6));

    group.bench_function("in_core_n_passes", |b| {
        let view = x.as_array().view();
        b.iter(|| {
            for mode in 0..shape.len() {
                let _ = tenrso_kernels::mttkrp(&view, &views, mode).unwrap();
            }
        });
    });

    group.bench_function("in_core_dimtree", |b| {
        let view = x.as_array().view();
        b.iter(|| tenrso_kernels::mttkrp_all_modes(&view, &views).unwrap());
    });

    group.bench_function("stream_n_passes", |b| {
        let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(64));
        b.iter(|| {
            for mode in 0..shape.len() {
                let _ = exec.mttkrp(&source, &spec, &views, mode).unwrap();
            }
        });
    });

    group.bench_function("stream_all_modes_one_pass", |b| {
        let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(64));
        b.iter(|| exec.mttkrp_all_modes(&source, &spec, &views).unwrap());
    });

    group.finish();
}

/// Streaming from a memory-mapped file (warm page cache) vs from RAM.
fn bench_mmap_source(c: &mut Criterion) {
    let shape = vec![80, 64, 48];
    let x = tensor(&shape);
    let fs = factors(&shape, RANK);
    let views: Vec<ArrayView2<f64>> = fs.iter().map(|f| f.view()).collect();
    let spec = ChunkSpec::tile_size(&shape, &[32, 32, 32]).unwrap();
    let elems: usize = shape.iter().product();

    let path = std::env::temp_dir().join("tenrso_bench_mttkrp_stream.bin");
    write_tensor_binary(&path, &x).unwrap();
    let mmap_source = MmapChunkSource::open(&path).unwrap();
    let dense_source = DenseChunkSource::new(x).unwrap();

    let mut group = c.benchmark_group("mttkrp_all_modes_source");
    group.throughput(Throughput::Elements(elems as u64));
    group.measurement_time(Duration::from_secs(6));

    group.bench_function("dense_memory", |b| {
        let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(64));
        b.iter(|| exec.mttkrp_all_modes(&dense_source, &spec, &views).unwrap());
    });

    group.bench_function("mmap_file_warm_cache", |b| {
        let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_mb(64));
        b.iter(|| exec.mttkrp_all_modes(&mmap_source, &spec, &views).unwrap());
    });

    // A budget far below the tensor size: the pass still completes, with a correspondingly
    // small window. The chunk must fit inside the budget, so use a finer grid here (a
    // 32^3 chunk is already 256 KiB and would not fit a 256 KiB budget).
    let fine_spec = ChunkSpec::tile_size(&shape, &[16, 16, 16]).unwrap();
    group.bench_function("mmap_file_tight_budget_256k", |b| {
        let mut exec = StreamingMttkrp::new(MttkrpStreamConfig::new().max_memory_bytes(256 * 1024));
        b.iter(|| {
            exec.mttkrp_all_modes(&mmap_source, &fine_spec, &views)
                .unwrap()
        });
    });

    group.finish();
    drop(mmap_source);
    let _ = std::fs::remove_file(&path);
}

criterion_group!(
    benches,
    bench_streaming_overhead,
    bench_sweep,
    bench_mmap_source
);
criterion_main!(benches);
