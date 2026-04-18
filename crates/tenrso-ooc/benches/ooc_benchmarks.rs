//! Criterion benchmarks for tenrso-ooc I/O and processing pipelines
//!
//! Covers: Arrow IPC, Parquet, mmap, batch I/O, compression, and streaming.
//!
//! All benchmarks use `std::env::temp_dir()` for I/O and clean up temp files
//! after each iteration. Data is seeded with `StdRng::seed_from_u64(42)` for
//! reproducibility.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use scirs2_core::RngExt;
use std::hint::black_box;
use tenrso_core::DenseND;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a DenseND<f64> tensor filled with seeded pseudo-random data.
fn make_random_tensor(shape: &[usize], seed: u64) -> DenseND<f64> {
    let len: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f64> = (0..len).map(|_| rng.random::<f64>()).collect();
    DenseND::from_vec(data, shape).expect("failed to create random tensor")
}

/// Return a unique temp-file path. The caller is responsible for cleanup.
fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("tenrso_ooc_bench_{}", name));
    p
}

// ---------------------------------------------------------------------------
// 1. Arrow IPC benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "arrow")]
fn bench_arrow_ipc(c: &mut Criterion) {
    use tenrso_ooc::arrow_io::{ArrowReader, ArrowWriter};

    let mut group = c.benchmark_group("arrow_ipc");

    // 256 x 256 x 16 = 1,048,576 f64 = 8 MiB
    let shape = [256, 256, 16];
    let total_elements = shape.iter().product::<usize>() as u64;
    let tensor = make_random_tensor(&shape, 42);

    group.throughput(Throughput::Elements(total_elements));

    // --- Write ---
    group.bench_function("write_256x256x16", |b| {
        b.iter(|| {
            let path = temp_path("arrow_write.arrow");
            let mut writer =
                ArrowWriter::new(&path).expect("arrow writer creation failed");
            writer.write(&tensor).expect("arrow write failed");
            writer.finish().expect("arrow finish failed");
            std::fs::remove_file(&path).ok();
        });
    });

    // --- Read (pre-write the file once, bench reads) ---
    let read_path = temp_path("arrow_read.arrow");
    {
        let mut w =
            ArrowWriter::new(&read_path).expect("arrow writer creation failed");
        w.write(&tensor).expect("arrow write failed");
        w.finish().expect("arrow finish failed");
    }

    group.bench_function("read_256x256x16", |b| {
        b.iter(|| {
            let mut reader =
                ArrowReader::open(&read_path).expect("arrow reader open failed");
            let loaded = reader.read().expect("arrow read failed");
            black_box(loaded);
        });
    });

    // --- Round-trip ---
    group.bench_function("roundtrip_256x256x16", |b| {
        b.iter(|| {
            let path = temp_path("arrow_rt.arrow");
            let mut writer =
                ArrowWriter::new(&path).expect("arrow writer creation failed");
            writer.write(&tensor).expect("arrow write failed");
            writer.finish().expect("arrow finish failed");

            let mut reader =
                ArrowReader::open(&path).expect("arrow reader open failed");
            let loaded = reader.read().expect("arrow read failed");
            std::fs::remove_file(&path).ok();
            black_box(loaded);
        });
    });

    std::fs::remove_file(&read_path).ok();
    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Parquet benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "parquet")]
fn bench_parquet(c: &mut Criterion) {
    use tenrso_ooc::parquet_io::{ParquetReader, ParquetWriter};

    let mut group = c.benchmark_group("parquet");

    // NOTE: ParquetReader reads only the first record batch. The default reader
    // batch size is 1024 rows, so the tensor element count must stay within that
    // limit. We use 32x32 = 1024 elements as the standard Parquet bench size.
    // For larger tensors, the ParquetReader would need a with_batch_size() option.
    let shape = [32, 32];
    let total_elements = shape.iter().product::<usize>() as u64;
    let tensor = make_random_tensor(&shape, 42);

    group.throughput(Throughput::Elements(total_elements));

    // --- Write ---
    group.bench_function("write_32x32", |b| {
        b.iter(|| {
            let path = temp_path("pq_write.parquet");
            let mut writer =
                ParquetWriter::new(&path).expect("parquet writer creation failed");
            writer.write(&tensor).expect("parquet write failed");
            // finish() consumes self
            writer.finish().expect("parquet finish failed");
            std::fs::remove_file(&path).ok();
        });
    });

    // --- Read (pre-write once) ---
    let read_path = temp_path("pq_read.parquet");
    {
        let mut w =
            ParquetWriter::new(&read_path).expect("parquet writer creation failed");
        w.write(&tensor).expect("parquet write failed");
        w.finish().expect("parquet finish failed");
    }

    group.bench_function("read_32x32", |b| {
        b.iter(|| {
            let reader =
                ParquetReader::open(&read_path).expect("parquet reader open failed");
            let loaded = reader.read().expect("parquet read failed");
            black_box(loaded);
        });
    });

    // --- Round-trip ---
    group.bench_function("roundtrip_32x32", |b| {
        b.iter(|| {
            let path = temp_path("pq_rt.parquet");
            let mut writer =
                ParquetWriter::new(&path).expect("parquet writer creation failed");
            writer.write(&tensor).expect("parquet write failed");
            writer.finish().expect("parquet finish failed");

            let reader =
                ParquetReader::open(&path).expect("parquet reader open failed");
            let loaded = reader.read().expect("parquet read failed");
            std::fs::remove_file(&path).ok();
            black_box(loaded);
        });
    });

    // --- Write-only scaling (not limited by reader batch size) ---
    for &side in &[32, 64, 128] {
        let s = [side, side];
        let elems = s.iter().product::<usize>() as u64;
        let t = make_random_tensor(&s, 42);

        group.throughput(Throughput::Elements(elems));
        group.bench_with_input(
            BenchmarkId::new("write_scale", format!("{side}x{side}")),
            &side,
            |b, _| {
                b.iter(|| {
                    let path = temp_path(&format!("pq_scale_{side}.parquet"));
                    let mut writer =
                        ParquetWriter::new(&path).expect("parquet writer creation failed");
                    writer.write(&t).expect("parquet write failed");
                    writer.finish().expect("parquet finish failed");
                    std::fs::remove_file(&path).ok();
                });
            },
        );
    }

    std::fs::remove_file(&read_path).ok();
    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Memory-mapped I/O benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "mmap")]
fn bench_mmap_io(c: &mut Criterion) {
    use tenrso_ooc::mmap_io::{write_tensor_binary, MmapTensor};

    let mut group = c.benchmark_group("mmap_io");

    let shape = [256, 256, 16];
    let total_elements = shape.iter().product::<usize>() as u64;
    let tensor = make_random_tensor(&shape, 42);

    group.throughput(Throughput::Elements(total_elements));

    // --- Binary write ---
    group.bench_function("write_256x256x16", |b| {
        b.iter(|| {
            let path = temp_path("mmap_write.bin");
            write_tensor_binary(&path, &tensor).expect("mmap write failed");
            std::fs::remove_file(&path).ok();
        });
    });

    // --- Binary read via read_tensor_binary ---
    let read_path = temp_path("mmap_read.bin");
    write_tensor_binary(&read_path, &tensor).expect("mmap write failed");

    group.bench_function("read_binary_256x256x16", |b| {
        b.iter(|| {
            let loaded = tenrso_ooc::mmap_io::read_tensor_binary(&read_path)
                .expect("binary read failed");
            black_box(loaded);
        });
    });

    // --- Mmap open + as_slice (zero-copy read) ---
    group.bench_function("mmap_open_slice_256x256x16", |b| {
        b.iter(|| {
            let mmap = MmapTensor::<f64>::open(&read_path).expect("mmap open failed");
            let slice = mmap.as_slice();
            black_box(slice.len());
            // intentionally keep mmap alive through black_box
            black_box(&mmap);
        });
    });

    // --- Mmap open + to_dense (copies data) ---
    group.bench_function("mmap_to_dense_256x256x16", |b| {
        b.iter(|| {
            let mmap = MmapTensor::<f64>::open(&read_path).expect("mmap open failed");
            let dense = mmap.to_dense().expect("to_dense failed");
            black_box(dense);
        });
    });

    // --- Scaling: different tensor sizes ---
    for &side in &[64, 128, 256] {
        let s = [side, side, 16];
        let elems = s.iter().product::<usize>() as u64;
        let t = make_random_tensor(&s, 42);
        let p = temp_path(&format!("mmap_scale_{side}.bin"));
        write_tensor_binary(&p, &t).expect("mmap write failed");

        group.throughput(Throughput::Elements(elems));
        group.bench_with_input(
            BenchmarkId::new("mmap_read_scale", format!("{side}x{side}x16")),
            &side,
            |b, _| {
                b.iter(|| {
                    let mmap =
                        MmapTensor::<f64>::open(&p).expect("mmap open failed");
                    black_box(mmap.as_slice().len());
                });
            },
        );
        std::fs::remove_file(&p).ok();
    }

    std::fs::remove_file(&read_path).ok();
    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Batch I/O benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "mmap")]
fn bench_batch_io(c: &mut Criterion) {
    use std::collections::HashMap;
    use tenrso_ooc::batch_io::{BatchConfig, BatchReader, BatchWriter};

    let mut group = c.benchmark_group("batch_io");

    let num_chunks = 8usize;
    let chunk_shape = [64, 64, 16];
    let elements_per_chunk = chunk_shape.iter().product::<usize>() as u64;
    let total_elements = elements_per_chunk * num_chunks as u64;

    group.throughput(Throughput::Elements(total_elements));

    // Prepare chunks
    let mut chunks = HashMap::new();
    let mut shapes = HashMap::new();
    for i in 0..num_chunks {
        let id = format!("{i}");
        let t = make_random_tensor(&chunk_shape, 42 + i as u64);
        shapes.insert(id.clone(), chunk_shape.to_vec());
        chunks.insert(id, t);
    }

    let config_seq = BatchConfig::default()
        .batch_size(num_chunks)
        .parallel(false);

    let config_par = BatchConfig::default()
        .batch_size(num_chunks)
        .parallel(true);

    // --- Sequential batch write ---
    group.bench_function("write_8chunks_seq", |b| {
        let base = temp_path("batch_write_seq");
        b.iter(|| {
            let writer = BatchWriter::new(config_seq.clone());
            let res = writer
                .write_batch(&chunks, &base)
                .expect("batch write failed");
            black_box(res.bytes_written);
        });
        std::fs::remove_dir_all(&base).ok();
    });

    // --- Parallel batch write ---
    group.bench_function("write_8chunks_par", |b| {
        let base = temp_path("batch_write_par");
        b.iter(|| {
            let writer = BatchWriter::new(config_par.clone());
            let res = writer
                .write_batch(&chunks, &base)
                .expect("batch write failed");
            black_box(res.bytes_written);
        });
        std::fs::remove_dir_all(&base).ok();
    });

    // --- Sequential batch read (pre-write once) ---
    let read_base = temp_path("batch_read_dir");
    {
        let writer = BatchWriter::new(config_seq.clone());
        writer
            .write_batch(&chunks, &read_base)
            .expect("batch write for read bench failed");
    }
    let chunk_ids: Vec<String> = (0..num_chunks).map(|i| format!("{i}")).collect();

    group.bench_function("read_8chunks_seq", |b| {
        b.iter(|| {
            let reader = BatchReader::new(config_seq.clone());
            let res = reader
                .read_batch(&chunk_ids, &shapes, &read_base)
                .expect("batch read failed");
            black_box(res.bytes_read);
        });
    });

    // --- Parallel batch read ---
    group.bench_function("read_8chunks_par", |b| {
        b.iter(|| {
            let reader = BatchReader::new(config_par.clone());
            let res = reader
                .read_batch(&chunk_ids, &shapes, &read_base)
                .expect("batch read failed");
            black_box(res.bytes_read);
        });
    });

    // --- Round-trip (write + read) ---
    group.bench_function("roundtrip_8chunks", |b| {
        let base = temp_path("batch_rt");
        b.iter(|| {
            let writer = BatchWriter::new(config_seq.clone());
            writer
                .write_batch(&chunks, &base)
                .expect("batch write failed");

            let reader = BatchReader::new(config_seq.clone());
            let res = reader
                .read_batch(&chunk_ids, &shapes, &base)
                .expect("batch read failed");
            black_box(res.bytes_read);
        });
        std::fs::remove_dir_all(&base).ok();
    });

    std::fs::remove_dir_all(&read_base).ok();
    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Compression benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "compression")]
fn bench_compression(c: &mut Criterion) {
    use tenrso_ooc::compression::{
        compress_bytes, compress_f64_slice, decompress_bytes, decompress_to_f64_vec,
        CompressionCodec,
    };

    let mut group = c.benchmark_group("compression");

    // 128 x 128 = 16384 f64 = 128 KiB -- realistic random data
    let shape = [128, 128];
    let total_elements = shape.iter().product::<usize>() as u64;
    let tensor = make_random_tensor(&shape, 42);
    let raw_bytes: Vec<u8> = tensor
        .as_slice()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    group.throughput(Throughput::Elements(total_elements));

    // --- LZ4 compress ---
    #[cfg(feature = "lz4-compression")]
    {
        group.bench_function("lz4_compress_128x128", |b| {
            b.iter(|| {
                let compressed =
                    compress_bytes(&raw_bytes, CompressionCodec::Lz4)
                        .expect("lz4 compress failed");
                black_box(compressed.len());
            });
        });

        // pre-compress for decompression bench
        let compressed_lz4 =
            compress_bytes(&raw_bytes, CompressionCodec::Lz4)
                .expect("lz4 compress setup failed");

        group.bench_function("lz4_decompress_128x128", |b| {
            b.iter(|| {
                let decompressed = decompress_bytes(&compressed_lz4)
                    .expect("lz4 decompress failed");
                black_box(decompressed.len());
            });
        });

        // f64-slice convenience path
        group.bench_function("lz4_f64_roundtrip_128x128", |b| {
            b.iter(|| {
                let compressed = compress_f64_slice(tensor.as_slice(), CompressionCodec::Lz4)
                    .expect("lz4 f64 compress failed");
                let decompressed =
                    decompress_to_f64_vec(&compressed).expect("lz4 f64 decompress failed");
                black_box(decompressed.len());
            });
        });
    }

    // --- Zstd compress ---
    #[cfg(feature = "zstd-compression")]
    {
        for &level in &[1, 3, 9] {
            let codec = CompressionCodec::Zstd { level };

            group.bench_function(
                format!("zstd_lvl{level}_compress_128x128"),
                |b| {
                    b.iter(|| {
                        let compressed =
                            compress_bytes(&raw_bytes, codec)
                                .expect("zstd compress failed");
                        black_box(compressed.len());
                    });
                },
            );
        }

        // Decompress at default level 3
        let compressed_zstd =
            compress_bytes(&raw_bytes, CompressionCodec::Zstd { level: 3 })
                .expect("zstd compress setup failed");

        group.bench_function("zstd_decompress_128x128", |b| {
            b.iter(|| {
                let decompressed = decompress_bytes(&compressed_zstd)
                    .expect("zstd decompress failed");
                black_box(decompressed.len());
            });
        });

        // f64-slice convenience path
        group.bench_function("zstd_f64_roundtrip_128x128", |b| {
            b.iter(|| {
                let compressed =
                    compress_f64_slice(tensor.as_slice(), CompressionCodec::Zstd { level: 3 })
                        .expect("zstd f64 compress failed");
                let decompressed =
                    decompress_to_f64_vec(&compressed).expect("zstd f64 decompress failed");
                black_box(decompressed.len());
            });
        });
    }

    // --- Scaling: compare codecs across sizes ---
    for &side in &[64, 128, 256] {
        let s = [side, side];
        let elems = s.iter().product::<usize>() as u64;
        let t = make_random_tensor(&s, 42);
        let bytes: Vec<u8> = t
            .as_slice()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        group.throughput(Throughput::Elements(elems));

        #[cfg(feature = "lz4-compression")]
        group.bench_with_input(
            BenchmarkId::new("lz4_compress_scale", format!("{side}x{side}")),
            &side,
            |b, _| {
                b.iter(|| {
                    let compressed =
                        compress_bytes(&bytes, CompressionCodec::Lz4)
                            .expect("lz4 scale compress failed");
                    black_box(compressed.len());
                });
            },
        );

        #[cfg(feature = "zstd-compression")]
        group.bench_with_input(
            BenchmarkId::new("zstd_compress_scale", format!("{side}x{side}")),
            &side,
            |b, _| {
                b.iter(|| {
                    let compressed =
                        compress_bytes(&bytes, CompressionCodec::Zstd { level: 3 })
                            .expect("zstd scale compress failed");
                    black_box(compressed.len());
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Streaming contraction benchmarks
// ---------------------------------------------------------------------------
//
// NOTE: Detailed streaming benchmarks already exist in `large_tensors.rs`
// (matmul_chunked, add_chunked, streaming_memory_limits, prefetch_strategies).
// Here we add a small complementary benchmark for the streaming add operation
// at the canonical I/O-bench tensor size so users can cross-compare formats.

fn bench_streaming_add(c: &mut Criterion) {
    use tenrso_ooc::{StreamConfig, StreamingExecutor};

    let mut group = c.benchmark_group("streaming_add");

    let shape = [256, 256, 16];
    let total_elements = shape.iter().product::<usize>() as u64;
    let a = make_random_tensor(&shape, 42);
    let b = make_random_tensor(&shape, 84);

    group.throughput(Throughput::Elements(total_elements));

    for &chunk in &[32, 64, 128] {
        let config = StreamConfig::new()
            .max_memory_mb(64)
            .chunk_size(vec![chunk]);

        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk),
            &chunk,
            |bencher, _| {
                let mut executor = StreamingExecutor::new(config.clone());
                bencher.iter(|| {
                    let result = executor
                        .add_chunked(&a, &b)
                        .expect("streaming add failed");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups and main
// ---------------------------------------------------------------------------

// Conditionally include feature-gated groups
#[cfg(feature = "arrow")]
criterion_group!(arrow_benches, bench_arrow_ipc);

#[cfg(feature = "parquet")]
criterion_group!(parquet_benches, bench_parquet);

#[cfg(feature = "mmap")]
criterion_group!(mmap_benches, bench_mmap_io);

#[cfg(feature = "mmap")]
criterion_group!(batch_benches, bench_batch_io);

#[cfg(feature = "compression")]
criterion_group!(compression_benches, bench_compression);

criterion_group!(streaming_benches, bench_streaming_add);

// Main entry point -- all default-enabled feature groups
criterion_main!(
    arrow_benches,
    parquet_benches,
    mmap_benches,
    batch_benches,
    compression_benches,
    streaming_benches
);
