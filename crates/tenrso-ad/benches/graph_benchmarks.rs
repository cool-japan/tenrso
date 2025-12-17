//! Performance benchmarks for graph-based automatic differentiation
//!
//! Benchmarks the computation graph system including:
//! - Forward pass performance
//! - Backward pass performance
//! - Graph construction overhead
//! - Graph optimization impact

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::array;
use std::hint::black_box;
use tenrso_ad::graph::ComputationGraph;
use tenrso_ad::graph_optimizer::{GraphOptimizer, OptimizationPass};

/// Benchmark basic arithmetic operations in the graph
fn bench_graph_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_arithmetic");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            bencher.iter(|| {
                let graph = ComputationGraph::<f64>::new();
                let x_vec: Vec<f64> = (0..size).map(|i| i as f64).collect();
                let y_vec: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();

                let x = graph
                    .variable(
                        scirs2_core::ndarray_ext::Array::from_vec(x_vec).into_dyn(),
                        true,
                    )
                    .unwrap();
                let y = graph
                    .variable(
                        scirs2_core::ndarray_ext::Array::from_vec(y_vec).into_dyn(),
                        true,
                    )
                    .unwrap();

                // Build computation: (x + y) * x
                let sum = graph.add(&x, &y).unwrap();
                let result = graph.mul(&sum, &x).unwrap();

                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark forward + backward pass for a simple neural network layer
fn bench_graph_neural_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_neural_layer");

    for batch_size in [1, 16, 32, 64].iter() {
        let input_dim = 128;
        let output_dim = 64;

        group.throughput(Throughput::Elements(
            (batch_size * input_dim * output_dim) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |bencher, &batch_size| {
                bencher.iter(|| {
                    let graph = ComputationGraph::<f64>::new();

                    // Input batch
                    let x_arr =
                        scirs2_core::ndarray_ext::Array2::<f64>::zeros((batch_size, input_dim));
                    let x = graph.variable(x_arr.into_dyn(), true).unwrap();

                    // Weights and bias
                    let w_arr =
                        scirs2_core::ndarray_ext::Array2::<f64>::zeros((input_dim, output_dim));
                    let w = graph.variable(w_arr.into_dyn(), true).unwrap();

                    let b_arr = scirs2_core::ndarray_ext::Array1::<f64>::zeros(output_dim);
                    let b = graph.variable(b_arr.into_dyn(), true).unwrap();

                    // Forward: h = ReLU(x @ W + b)
                    let xw = graph.matmul(&x, &w).unwrap();
                    let xw_b = graph.add(&xw, &b).unwrap();
                    let h = graph.relu(&xw_b).unwrap();

                    // Loss (sum for simplicity)
                    let loss = graph.sum(&h).unwrap();

                    // Backward pass
                    graph.backward(&loss).unwrap();

                    black_box(loss);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark deep network forward + backward pass
fn bench_graph_deep_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_deep_network");

    for num_layers in [5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*num_layers as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_layers),
            num_layers,
            |bencher, &num_layers| {
                bencher.iter(|| {
                    let graph = ComputationGraph::<f64>::new();
                    let hidden_dim = 64;

                    // Input
                    let mut x = graph
                        .variable(
                            scirs2_core::ndarray_ext::Array2::<f64>::zeros((1, hidden_dim))
                                .into_dyn(),
                            true,
                        )
                        .unwrap();

                    // Build deep network
                    for _ in 0..num_layers {
                        let w = graph
                            .variable(
                                scirs2_core::ndarray_ext::Array2::<f64>::zeros((
                                    hidden_dim, hidden_dim,
                                ))
                                .into_dyn(),
                                false,
                            )
                            .unwrap();
                        let b = graph
                            .variable(
                                scirs2_core::ndarray_ext::Array1::<f64>::zeros(hidden_dim)
                                    .into_dyn(),
                                false,
                            )
                            .unwrap();

                        let wx = graph.matmul(&x, &w).unwrap();
                        let wxb = graph.add(&wx, &b).unwrap();
                        x = graph.relu(&wxb).unwrap();
                    }

                    let loss = graph.sum(&x).unwrap();
                    graph.backward(&loss).unwrap();

                    black_box(loss);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark graph construction overhead
fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");

    for num_ops in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*num_ops as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_ops),
            num_ops,
            |bencher, &num_ops| {
                bencher.iter(|| {
                    let graph = ComputationGraph::<f64>::new();
                    let mut x = graph
                        .variable(array![1.0, 2.0, 3.0].into_dyn(), true)
                        .unwrap();

                    // Chain operations
                    for _ in 0..num_ops {
                        x = graph.exp(&x).unwrap();
                    }

                    black_box(x);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark backward pass performance
fn bench_graph_backward_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_backward_only");

    for num_layers in [5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*num_layers as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_layers),
            num_layers,
            |bencher, &num_layers| {
                // Build graph once
                let graph = ComputationGraph::<f64>::new();
                let hidden_dim = 64;
                let mut x = graph
                    .variable(
                        scirs2_core::ndarray_ext::Array2::<f64>::zeros((1, hidden_dim)).into_dyn(),
                        true,
                    )
                    .unwrap();

                for _ in 0..num_layers {
                    let w = graph
                        .variable(
                            scirs2_core::ndarray_ext::Array2::<f64>::zeros((
                                hidden_dim, hidden_dim,
                            ))
                            .into_dyn(),
                            false,
                        )
                        .unwrap();
                    let wx = graph.matmul(&x, &w).unwrap();
                    x = graph.relu(&wx).unwrap();
                }
                let loss = graph.sum(&x).unwrap();

                // Benchmark only backward pass
                bencher.iter(|| {
                    graph.zero_grad();
                    graph.backward(black_box(&loss)).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark activation functions
fn bench_graph_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_activations");

    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            bencher.iter(|| {
                let graph = ComputationGraph::<f64>::new();
                let x_vec: Vec<f64> = (0..size).map(|i| i as f64 / size as f64).collect();
                let x = graph
                    .variable(
                        scirs2_core::ndarray_ext::Array::from_vec(x_vec).into_dyn(),
                        true,
                    )
                    .unwrap();

                // Test all activation functions
                let relu_out = graph.relu(&x).unwrap();
                let sigmoid_out = graph.sigmoid(&x).unwrap();
                let tanh_out = graph.tanh(&x).unwrap();

                // Sum to create loss
                let sum1 = graph.sum(&relu_out).unwrap();
                let sum2 = graph.sum(&sigmoid_out).unwrap();
                let sum3 = graph.sum(&tanh_out).unwrap();

                let combined = graph.add(&sum1, &sum2).unwrap();
                let loss = graph.add(&combined, &sum3).unwrap();

                graph.backward(&loss).unwrap();

                black_box(loss);
            });
        });
    }
    group.finish();
}

/// Benchmark graph optimization impact
fn bench_graph_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_optimization");

    // Benchmark operation fusion
    group.bench_function("fusion_matmul_bias_relu", |bencher| {
        bencher.iter(|| {
            let graph = ComputationGraph::<f64>::new();
            let x = graph
                .variable(
                    scirs2_core::ndarray_ext::Array2::<f64>::zeros((32, 64)).into_dyn(),
                    true,
                )
                .unwrap();
            let w = graph
                .variable(
                    scirs2_core::ndarray_ext::Array2::<f64>::zeros((64, 128)).into_dyn(),
                    false,
                )
                .unwrap();
            let b = graph
                .variable(
                    scirs2_core::ndarray_ext::Array1::<f64>::zeros(128).into_dyn(),
                    false,
                )
                .unwrap();

            let xw = graph.matmul(&x, &w).unwrap();
            let xwb = graph.add(&xw, &b).unwrap();
            let out = graph.relu(&xwb).unwrap();
            let loss = graph.sum(&out).unwrap();

            // Apply optimization
            let optimizer = GraphOptimizer::new().with_pass(OptimizationPass::OperationFusion);
            let _ = optimizer.optimize(&graph);

            black_box(loss);
        });
    });

    // Benchmark dead code elimination
    group.bench_function("dead_code_elimination", |bencher| {
        bencher.iter(|| {
            let graph = ComputationGraph::<f64>::new();
            let x = graph
                .variable(array![1.0, 2.0, 3.0].into_dyn(), true)
                .unwrap();

            // Create some dead code
            let _ = graph.exp(&x).unwrap();
            let _ = graph.log(&x).unwrap();

            // Only use this path
            let y = graph.pow(&x, 2.0).unwrap();
            let loss = graph.sum(&y).unwrap();

            // Apply optimization
            let optimizer = GraphOptimizer::new().with_pass(OptimizationPass::DeadCodeElimination);
            let _ = optimizer.optimize(&graph);

            black_box(loss);
        });
    });

    // Benchmark full optimization pipeline
    group.bench_function("full_optimization", |bencher| {
        bencher.iter(|| {
            let graph = ComputationGraph::<f64>::new();
            let x = graph
                .variable(
                    scirs2_core::ndarray_ext::Array2::<f64>::zeros((16, 32)).into_dyn(),
                    true,
                )
                .unwrap();
            let w = graph
                .variable(
                    scirs2_core::ndarray_ext::Array2::<f64>::zeros((32, 64)).into_dyn(),
                    false,
                )
                .unwrap();
            let b = graph
                .variable(
                    scirs2_core::ndarray_ext::Array1::<f64>::zeros(64).into_dyn(),
                    false,
                )
                .unwrap();

            // Create some dead code
            let _ = graph.sigmoid(&x).unwrap();

            // Main path with fusible operations
            let xw = graph.matmul(&x, &w).unwrap();
            let xwb = graph.add(&xw, &b).unwrap();
            let out = graph.relu(&xwb).unwrap();
            let loss = graph.sum(&out).unwrap();

            // Apply all optimizations
            let optimizer = GraphOptimizer::new().with_pass(OptimizationPass::All);
            let _ = optimizer.optimize(&graph);

            black_box(loss);
        });
    });

    group.finish();
}

/// Benchmark memory usage (indirectly via gradient storage)
fn bench_graph_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_memory");

    for num_vars in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*num_vars as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            num_vars,
            |bencher, &num_vars| {
                bencher.iter(|| {
                    let graph = ComputationGraph::<f64>::new();
                    let mut vars = Vec::new();

                    // Create many variables
                    for _ in 0..num_vars {
                        let v = graph
                            .variable(array![1.0, 2.0, 3.0].into_dyn(), true)
                            .unwrap();
                        vars.push(v);
                    }

                    // Sum them all
                    let mut total = vars[0];
                    for v in vars.iter().skip(1) {
                        total = graph.add(&total, v).unwrap();
                    }

                    let loss = graph.sum(&total).unwrap();
                    graph.backward(&loss).unwrap();

                    // Access all gradients
                    for v in &vars {
                        let _ = graph.gradient(v).unwrap();
                    }

                    black_box(loss);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_graph_arithmetic,
    bench_graph_neural_layer,
    bench_graph_deep_network,
    bench_graph_construction,
    bench_graph_backward_only,
    bench_graph_activations,
    bench_graph_optimization,
    bench_graph_memory,
);
criterion_main!(benches);
