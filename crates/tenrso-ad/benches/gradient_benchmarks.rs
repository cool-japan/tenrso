//! Performance benchmarks for gradient operations
//!
//! Benchmarks VJP, decomposition gradients, and gradient checking

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::{Array, Array1, Array2, IxDyn};
use std::hint::black_box;
use tenrso_ad::grad::{CpReconstructionGrad, TuckerReconstructionGrad};
use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig};
use tenrso_ad::vjp::{EinsumVjp, ElementwiseUnaryVjp, ReductionType, ReductionVjp, VjpOp};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction;
use tenrso_planner::EinsumSpec;

/// Benchmark einsum VJP for matrix multiplication
fn bench_einsum_vjp_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_vjp_matmul");

    for size in [16, 32, 64, 128, 256].iter() {
        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            let a_arr = Array2::from_shape_fn((size, size), |(i, j)| (i * size + j) as f64);
            let b_arr = Array2::from_shape_fn((size, size), |(i, j)| (i * size + j) as f64);
            let a = DenseND::from_array(a_arr.into_dyn());
            let b_mat = DenseND::from_array(b_arr.into_dyn());
            let cotangent = DenseND::ones(&[size, size]);
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

            bencher.iter(|| {
                let vjp_ctx =
                    EinsumVjp::new(spec.clone(), black_box(a.clone()), black_box(b_mat.clone()));
                let result = vjp_ctx.vjp(black_box(&cotangent)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark einsum VJP for tensor contraction
fn bench_einsum_vjp_tensor_contraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_vjp_tensor_contraction");

    // Simplified to use only implemented operations
    for size in [8, 16, 32, 64].iter() {
        let n_elements = size * size;
        group.throughput(Throughput::Elements(n_elements as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            // Use ij,jk->ik which is a simple matmul
            let a_arr = Array2::from_shape_fn((size, size), |(i, j)| (i * size + j) as f64);
            let b_arr = Array2::from_shape_fn((size, size), |(i, j)| (i * size + j) as f64);
            let a = DenseND::from_array(a_arr.into_dyn());
            let b_mat = DenseND::from_array(b_arr.into_dyn());
            let cotangent = DenseND::ones(&[size, size]);
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

            bencher.iter(|| {
                let vjp_ctx =
                    EinsumVjp::new(spec.clone(), black_box(a.clone()), black_box(b_mat.clone()));
                let result = vjp_ctx.vjp(black_box(&cotangent)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark element-wise VJP operations
fn bench_elementwise_vjp(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_vjp");

    for size in [1000, 10000, 100000, 1000000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            let x_arr = Array1::from_shape_fn(size, |i| (i as f64) * 0.001);
            let x = DenseND::from_array(x_arr.into_dyn());
            let cotangent = DenseND::ones(&[size]);

            bencher.iter(|| {
                let derivative = |x_elem: &f64| 2.0 * x_elem;
                let ctx = ElementwiseUnaryVjp::new(black_box(x.clone()), derivative);
                let grads = ctx.vjp(black_box(&cotangent)).unwrap();
                black_box(grads);
            });
        });
    }
    group.finish();
}

/// Benchmark reduction VJP operations
fn bench_reduction_vjp(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_vjp");

    for size in [100, 500, 1000, 2000].iter() {
        let n_elements = size * size;
        group.throughput(Throughput::Elements(n_elements as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            let input_shape = vec![size, size];
            let cotangent = DenseND::from_elem(&[], 1.0);

            bencher.iter(|| {
                let ctx = ReductionVjp::<f64>::new(
                    black_box(input_shape.clone()),
                    None,
                    ReductionType::Sum,
                );
                let grads = ctx.vjp(black_box(&cotangent)).unwrap();
                black_box(grads);
            });
        });
    }
    group.finish();
}

/// Benchmark CP decomposition gradients
fn bench_cp_reconstruction_grad(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_reconstruction_grad");

    for rank in [8, 16, 32, 64].iter() {
        group.throughput(Throughput::Elements((rank * 100 * 3) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(rank), rank, |bencher, &rank| {
            let shape = vec![50, 50, 40];
            let factors = vec![
                Array2::from_shape_fn((shape[0], rank), |(i, j)| (i * rank + j) as f64 * 0.01),
                Array2::from_shape_fn((shape[1], rank), |(i, j)| (i * rank + j) as f64 * 0.01),
                Array2::from_shape_fn((shape[2], rank), |(i, j)| (i * rank + j) as f64 * 0.01),
            ];
            let weights = Array1::ones(rank);
            let cotangent = DenseND::ones(&shape);
            let grad_ctx = CpReconstructionGrad::new(factors.clone(), Some(weights.clone()));

            bencher.iter(|| {
                let grads = grad_ctx
                    .compute_factor_gradients(black_box(&cotangent))
                    .unwrap();
                black_box(grads);
            });
        });
    }
    group.finish();
}

/// Benchmark Tucker decomposition gradients
fn bench_tucker_reconstruction_grad(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_reconstruction_grad");

    for core_size in [4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(core_size),
            core_size,
            |bencher, &core_size| {
                let shape = vec![30, 30, 30];
                let core_arr =
                    Array::from_shape_fn(IxDyn(&[core_size, core_size, core_size]), |idx| {
                        let i = idx[0] * core_size * core_size + idx[1] * core_size + idx[2];
                        i as f64 * 0.01
                    });
                let core = DenseND::from_array(core_arr);
                let factors = vec![
                    Array2::from_shape_fn((shape[0], core_size), |(i, j)| {
                        (i * core_size + j) as f64 * 0.01
                    }),
                    Array2::from_shape_fn((shape[1], core_size), |(i, j)| {
                        (i * core_size + j) as f64 * 0.01
                    }),
                    Array2::from_shape_fn((shape[2], core_size), |(i, j)| {
                        (i * core_size + j) as f64 * 0.01
                    }),
                ];
                let cotangent = DenseND::ones(&shape);
                let grad_ctx = TuckerReconstructionGrad::new(core.clone(), factors.clone());

                bencher.iter(|| {
                    let grads = grad_ctx
                        .compute_factor_gradients(black_box(&cotangent))
                        .unwrap();
                    black_box(grads);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark gradient checking
fn bench_gradcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradcheck");

    for size in [10, 20, 30].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            let x_arr = Array2::from_shape_fn((size, size), |(i, j)| (i * size + j) as f64 * 0.1);
            let y_arr = Array2::from_shape_fn((size, size), |(i, j)| (i * size + j) as f64 * 0.1);
            let x = DenseND::from_array(x_arr.into_dyn());
            let y = DenseND::from_array(y_arr.into_dyn());
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

            let config = GradCheckConfig {
                epsilon: 1e-5,
                rtol: 1e-3,
                atol: 1e-5,
                use_central_diff: true,
                verbose: false,
            };

            bencher.iter(|| {
                let result = check_gradient(
                    |a| execute_dense_contraction(&spec, a, &y),
                    |a, grad_y| {
                        let vjp = EinsumVjp::new(spec.clone(), a.clone(), y.clone());
                        let grads = vjp.vjp(grad_y)?;
                        Ok(grads[0].clone())
                    },
                    black_box(&x),
                    &DenseND::ones(&[size, size]),
                    &config,
                )
                .unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_einsum_vjp_matmul,
    bench_einsum_vjp_tensor_contraction,
    bench_elementwise_vjp,
    bench_reduction_vjp,
    bench_cp_reconstruction_grad,
    bench_tucker_reconstruction_grad,
    bench_gradcheck,
);
criterion_main!(benches);
