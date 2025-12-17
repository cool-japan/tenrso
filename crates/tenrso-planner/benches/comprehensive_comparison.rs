//! Comprehensive comparison benchmark for all planning algorithms
//!
//! This benchmark suite compares all 6 planning algorithms across
//! different problem sizes, topologies, and characteristics.

use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use std::hint::black_box;
use tenrso_planner::{
    beam_search_planner, dp_planner, greedy_planner, simulated_annealing_planner, AdaptivePlanner,
    EinsumSpec, PlanHints, Planner,
};

/// Benchmark matrix chain multiplication with varying chain length
fn bench_matrix_chain_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_chain_by_size");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for chain_len in [3, 5, 7, 10].iter() {
        // Create chain: A1 × A2 × A3 × ... × An
        let mut indices = Vec::new();
        let mut shapes = Vec::new();

        for i in 0..*chain_len {
            indices.push(format!(
                "{}{}",
                (b'a' + i as u8) as char,
                (b'a' + i as u8 + 1) as char
            ));
            shapes.push(vec![10, 10]);
        }

        let output = format!("a{}", (b'a' + *chain_len as u8) as char);
        let spec_str = format!("{}->{}", indices.join(","), output);

        if let Ok(spec) = EinsumSpec::parse(&spec_str) {
            let hints = PlanHints::default();

            // Greedy
            group.bench_with_input(BenchmarkId::new("Greedy", chain_len), chain_len, |b, _| {
                b.iter(|| greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints)))
            });

            // DP (only for small sizes)
            if *chain_len <= 10 {
                group.bench_with_input(BenchmarkId::new("DP", chain_len), chain_len, |b, _| {
                    b.iter(|| dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints)))
                });
            }

            // Beam Search (k=5)
            group.bench_with_input(
                BenchmarkId::new("BeamSearch_k5", chain_len),
                chain_len,
                |b, _| {
                    b.iter(|| {
                        beam_search_planner(
                            black_box(&spec),
                            black_box(&shapes),
                            black_box(&hints),
                            5,
                        )
                    })
                },
            );

            // Adaptive
            group.bench_with_input(
                BenchmarkId::new("Adaptive", chain_len),
                chain_len,
                |b, _| {
                    let planner = AdaptivePlanner::new();
                    b.iter(|| {
                        planner.make_plan(
                            black_box(&spec_str),
                            black_box(&shapes),
                            black_box(&hints),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark star topology with varying number of branches
fn bench_star_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_topology");

    for num_branches in [3, 5, 8].iter() {
        // Create star: A1 contracts with A2, A3, A4, ...
        // All share index 'a', output removes 'a'
        let spec_str = if *num_branches == 3 {
            "ab,ac,ad->bcd"
        } else if *num_branches == 5 {
            "ab,ac,ad,ae,af->bcdef"
        } else {
            "ab,ac,ad,ae,af,ag,ah,ai->bcdefghi"
        };

        if let Ok(spec) = EinsumSpec::parse(spec_str) {
            let shapes = vec![vec![10, 10]; *num_branches];
            let hints = PlanHints::default();

            group.bench_with_input(
                BenchmarkId::new("Greedy", num_branches),
                num_branches,
                |b, _| {
                    b.iter(|| {
                        greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints))
                    })
                },
            );

            if *num_branches <= 8 {
                group.bench_with_input(
                    BenchmarkId::new("DP", num_branches),
                    num_branches,
                    |b, _| {
                        b.iter(|| {
                            dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints))
                        })
                    },
                );
            }

            group.bench_with_input(
                BenchmarkId::new("BeamSearch_k5", num_branches),
                num_branches,
                |b, _| {
                    b.iter(|| {
                        beam_search_planner(
                            black_box(&spec),
                            black_box(&shapes),
                            black_box(&hints),
                            5,
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark with varying dimension sizes
fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_scaling");

    let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
    let hints = PlanHints::default();

    for dim in [10, 50, 100, 200].iter() {
        let shapes = vec![vec![*dim, *dim], vec![*dim, *dim], vec![*dim, *dim]];

        group.bench_with_input(BenchmarkId::new("Greedy", dim), dim, |b, _| {
            b.iter(|| greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints)))
        });

        group.bench_with_input(BenchmarkId::new("DP", dim), dim, |b, _| {
            b.iter(|| dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints)))
        });

        group.bench_with_input(BenchmarkId::new("BeamSearch_k5", dim), dim, |b, _| {
            b.iter(|| {
                beam_search_planner(black_box(&spec), black_box(&shapes), black_box(&hints), 5)
            })
        });
    }

    group.finish();
}

/// Benchmark plan quality (FLOPs) vs planning time tradeoff
fn bench_quality_vs_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_vs_time");
    group.sample_size(20); // Reduce sample size for expensive operations

    let spec = EinsumSpec::parse("ij,jk,kl,lm,mn->in").unwrap();
    let shapes = vec![
        vec![20, 30],
        vec![30, 40],
        vec![40, 50],
        vec![50, 60],
        vec![60, 20],
    ];
    let hints = PlanHints::default();

    // Greedy (fastest)
    group.bench_function("Greedy", |b| {
        b.iter(|| greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints)))
    });

    // Beam Search (k=3, 5, 10)
    for k in [3, 5, 10].iter() {
        group.bench_function(format!("BeamSearch_k{}", k), |b| {
            b.iter(|| {
                beam_search_planner(black_box(&spec), black_box(&shapes), black_box(&hints), *k)
            })
        });
    }

    // DP (optimal but slower)
    group.bench_function("DP_optimal", |b| {
        b.iter(|| dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints)))
    });

    // Simulated Annealing (reduced iterations for benchmarking)
    group.bench_function("SA_300iter", |b| {
        b.iter(|| {
            simulated_annealing_planner(
                black_box(&spec),
                black_box(&shapes),
                black_box(&hints),
                1000.0,
                0.95,
                300,
            )
        })
    });

    group.finish();
}

/// Benchmark with cache-friendly vs cache-unfriendly problem sizes
fn bench_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effects");

    let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    let hints = PlanHints::default();

    // Small (cache-friendly)
    let small_shapes = vec![vec![8, 8], vec![8, 8]];
    group.bench_function("small_8x8", |b| {
        b.iter(|| {
            greedy_planner(
                black_box(&spec),
                black_box(&small_shapes),
                black_box(&hints),
            )
        })
    });

    // Medium
    let medium_shapes = vec![vec![64, 64], vec![64, 64]];
    group.bench_function("medium_64x64", |b| {
        b.iter(|| {
            greedy_planner(
                black_box(&spec),
                black_box(&medium_shapes),
                black_box(&hints),
            )
        })
    });

    // Large (cache-unfriendly)
    let large_shapes = vec![vec![512, 512], vec![512, 512]];
    group.bench_function("large_512x512", |b| {
        b.iter(|| {
            greedy_planner(
                black_box(&spec),
                black_box(&large_shapes),
                black_box(&hints),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_chain_by_size,
    bench_star_topology,
    bench_dimension_scaling,
    bench_quality_vs_time,
    bench_cache_effects,
);
criterion_main!(benches);
