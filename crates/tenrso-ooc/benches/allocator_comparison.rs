use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_ooc::allocators::AllocatorInfo;

fn bench_small_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocator_small");
    let info = AllocatorInfo::current();

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*size as u64 * 10000));
        group.bench_with_input(BenchmarkId::new(info.name, size), size, |b, &size| {
            b.iter(|| {
                let mut vec = Vec::with_capacity(10000);
                for _ in 0..10000 {
                    vec.push(vec![0u8; size]);
                }
                black_box(vec);
            });
        });
    }
    group.finish();
}

fn bench_large_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocator_large");
    let info = AllocatorInfo::current();

    for size in [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64 * 100));
        group.bench_with_input(BenchmarkId::new(info.name, size), size, |b, &size| {
            b.iter(|| {
                let mut vec = Vec::with_capacity(100);
                for _ in 0..100 {
                    vec.push(vec![0u8; size]);
                }
                black_box(vec);
            });
        });
    }
    group.finish();
}

fn bench_allocation_deallocation_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocator_pattern");
    let info = AllocatorInfo::current();

    group.bench_function(format!("{}_alloc_dealloc", info.name), |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let v = vec![0u8; 4096];
                black_box(v);
            }
        });
    });

    group.bench_function(format!("{}_reuse", info.name), |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(4096);
            for _ in 0..1000 {
                v.clear();
                v.resize(4096, 0);
                black_box(&v);
            }
        });
    });

    group.finish();
}

fn bench_tensor_chunk_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_chunks");
    let info = AllocatorInfo::current();

    // Simulate realistic tensor chunk allocations
    let chunk_size = 256 * 256 * 8; // 256x256 f64 tensor
    group.throughput(Throughput::Bytes(chunk_size as u64 * 100));

    group.bench_function(format!("{}_tensor_chunks", info.name), |b| {
        b.iter(|| {
            let mut chunks = Vec::with_capacity(100);
            for _ in 0..100 {
                chunks.push(vec![0f64; 256 * 256]);
            }
            black_box(chunks);
        });
    });

    group.finish();
}

fn bench_concurrent_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocator_concurrent");
    let info = AllocatorInfo::current();

    group.bench_function(format!("{}_single_thread", info.name), |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let v = vec![0u8; 4096];
                black_box(v);
            }
        });
    });

    // Note: Parallel benchmark disabled to avoid rayon dependency
    // Use sequential iteration instead
    #[cfg(feature = "parallel")]
    group.bench_function(format!("{}_threaded", info.name), |b| {
        b.iter(|| {
            (0..1000).for_each(|_| {
                let v = vec![0u8; 4096];
                black_box(v);
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_small_allocations,
    bench_large_allocations,
    bench_allocation_deallocation_pattern,
    bench_tensor_chunk_simulation,
    bench_concurrent_allocations,
);
criterion_main!(benches);
