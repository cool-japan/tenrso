//! Benchmarks for convolution operations with im2col + GEMM optimization
//!
//! These benchmarks demonstrate the performance improvements from the im2col
//! transformation combined with optimized matrix multiplication (GEMM).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tenrso_core::dense::ConvPadding;
use tenrso_core::DenseND;

/// Benchmark 2D convolution with various input sizes
fn bench_conv2d_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_by_size");

    // Test various realistic CNN layer sizes
    let sizes = vec![
        (32, 32, 3, 3),   // Small: 32x32 image, 3x3 kernel
        (64, 64, 3, 3),   // Medium: 64x64 image, 3x3 kernel
        (128, 128, 5, 5), // Large: 128x128 image, 5x5 kernel
        (224, 224, 7, 7), // ResNet input: 224x224 image, 7x7 kernel
    ];

    for (h, w, kh, kw) in sizes {
        let input = DenseND::<f64>::random_uniform(&[h, w], 0.0, 1.0);
        let kernel = DenseND::<f64>::random_uniform(&[kh, kw], -1.0, 1.0);

        group.bench_with_input(
            BenchmarkId::new("conv2d", format!("{}x{}_k{}x{}", h, w, kh, kw)),
            &(&input, &kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    let output = input.conv2d(kernel, 1, ConvPadding::Valid, 1).unwrap();
                    std::hint::black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batched 2D convolution (common in neural networks)
fn bench_conv2d_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_batched");

    // Typical CNN layer: batch of 32 images, 64x64 pixels
    let batch_sizes = vec![1, 8, 16, 32];

    for batch_size in batch_sizes {
        let input = DenseND::<f64>::random_uniform(&[batch_size, 64, 64], 0.0, 1.0);
        let kernel = DenseND::<f64>::random_uniform(&[3, 3], -1.0, 1.0);

        group.bench_with_input(
            BenchmarkId::new("batched", batch_size),
            &(&input, &kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    let output = input.conv2d(kernel, 1, ConvPadding::Valid, 1).unwrap();
                    std::hint::black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark convolution with different stride values
fn bench_conv2d_stride(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_stride");

    let input = DenseND::<f64>::random_uniform(&[128, 128], 0.0, 1.0);
    let kernel = DenseND::<f64>::random_uniform(&[3, 3], -1.0, 1.0);

    for stride in [1, 2, 4] {
        group.bench_with_input(BenchmarkId::new("stride", stride), &stride, |b, &stride| {
            b.iter(|| {
                let output = input
                    .conv2d(&kernel, stride, ConvPadding::Valid, 1)
                    .unwrap();
                std::hint::black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark multi-channel convolution (e.g., RGB images)
fn bench_conv2d_multichannel(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_multichannel");

    let channels = vec![1, 3, 16, 32]; // Grayscale, RGB, and feature maps

    for num_channels in channels {
        let input = DenseND::<f64>::random_uniform(&[8, num_channels, 64, 64], 0.0, 1.0);
        let kernel = DenseND::<f64>::random_uniform(&[3, 3], -1.0, 1.0);

        group.bench_with_input(
            BenchmarkId::new("channels", num_channels),
            &(&input, &kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    let output = input.conv2d(kernel, 1, ConvPadding::Valid, 1).unwrap();
                    std::hint::black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 1D convolution for sequence data
fn bench_conv1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv1d");

    // Common sequence lengths for audio/NLP
    let lengths = vec![256, 512, 1024, 2048];

    for length in lengths {
        let input = DenseND::<f64>::random_uniform(&[length], 0.0, 1.0);
        let kernel = DenseND::<f64>::random_uniform(&[7], -1.0, 1.0); // 7-tap filter

        group.bench_with_input(
            BenchmarkId::new("seq_length", length),
            &(&input, &kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    let output = input.conv1d(kernel, 1, ConvPadding::Valid, 1).unwrap();
                    std::hint::black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 3D convolution for volumetric data
fn bench_conv3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv3d");

    // Small volumetric sizes (3D convolution is expensive)
    let sizes = vec![
        (16, 16, 16, 3, 3, 3), // Small volume
        (32, 32, 32, 3, 3, 3), // Medium volume
    ];

    for (d, h, w, kd, kh, kw) in sizes {
        let input = DenseND::<f64>::random_uniform(&[d, h, w], 0.0, 1.0);
        let kernel = DenseND::<f64>::random_uniform(&[kd, kh, kw], -1.0, 1.0);

        group.bench_with_input(
            BenchmarkId::new("conv3d", format!("{}x{}x{}", d, h, w)),
            &(&input, &kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    let output = input.conv3d(kernel, 1, ConvPadding::Valid, 1).unwrap();
                    std::hint::black_box(output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_conv2d_sizes,
    bench_conv2d_batched,
    bench_conv2d_stride,
    bench_conv2d_multichannel,
    bench_conv1d,
    bench_conv3d,
);
criterion_main!(benches);
