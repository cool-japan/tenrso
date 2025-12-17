//! Benchmarks for sparse tensor operations
//!
//! Compares performance of sparse operations against dense baselines
//! and across different sparse formats.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::{Array1, Array2};
use std::hint::black_box;
use tenrso_sparse::{BcsrMatrix, CooTensor, CsrMatrix, DiaMatrix, EllMatrix};

/// Generate a random sparse matrix with specified density
fn random_sparse_matrix(nrows: usize, ncols: usize, density: f64) -> CsrMatrix<f64> {
    let nnz = ((nrows * ncols) as f64 * density).max(1.0) as usize;

    let mut indices = Vec::new();
    let mut values = Vec::new();

    // Simple pseudo-random generation for reproducibility
    let mut seed = 12345u64;
    for _ in 0..nnz {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let i = (seed % nrows as u64) as usize;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let j = (seed % ncols as u64) as usize;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let val = (seed % 10000) as f64 / 10000.0;

        indices.push(vec![i, j]);
        values.push(val);
    }

    let coo = CooTensor::new(indices, values, vec![nrows, ncols]).expect("Failed to create COO");
    CsrMatrix::from_coo(&coo).expect("Failed to convert to CSR")
}

/// Generate a random dense matrix
fn random_dense_matrix(nrows: usize, ncols: usize) -> Array2<f64> {
    let mut seed = 54321u64;
    Array2::from_shape_fn((nrows, ncols), |_| {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed % 10000) as f64 / 10000.0
    })
}

/// Generate a random dense vector
fn random_dense_vector(size: usize) -> Array1<f64> {
    let mut seed = 98765u64;
    Array1::from_shape_fn(size, |_| {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed % 10000) as f64 / 10000.0
    })
}

/// Benchmark SpMV (Sparse Matrix-Vector) operations
fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv");

    for size in [100, 500, 1000].iter() {
        for density in [0.01, 0.05, 0.1].iter() {
            let csr = random_sparse_matrix(*size, *size, *density);
            let x = random_dense_vector(*size);

            let nnz = csr.nnz();
            group.throughput(Throughput::Elements(nnz as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_d{}", size, size, density)),
                &(csr, x),
                |b, (csr, x)| {
                    b.iter(|| {
                        let result = csr.spmv(black_box(&x.view()));
                        let _ = black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SpMM (Sparse Matrix-Matrix) operations
fn bench_spmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmm");

    for size in [100, 500].iter() {
        for k in [10, 50].iter() {
            for density in [0.01, 0.05, 0.1].iter() {
                let csr = random_sparse_matrix(*size, *size, *density);
                let b = random_dense_matrix(*size, *k);

                let nnz = csr.nnz();
                group.throughput(Throughput::Elements((nnz * k) as u64));

                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("{}x{}_k{}_d{}", size, size, k, density)),
                    &(csr, b),
                    |bench, (csr, b)| {
                        bench.iter(|| {
                            let result = csr.spmm(black_box(&b.view())).expect("SpMM failed");
                            black_box(result);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark SpSpMM (Sparse-Sparse Matrix Multiply) operations
fn bench_spspmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("spspmm");

    for size in [100, 500].iter() {
        for density in [0.01, 0.05].iter() {
            let a = random_sparse_matrix(*size, *size, *density);
            let b = random_sparse_matrix(*size, *size, *density);

            let ops = (a.nnz() * b.nnz() / size) as u64;
            group.throughput(Throughput::Elements(ops));

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_d{}", size, size, density)),
                &(a, b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        let result = a.spspmm(black_box(b)).expect("SpSpMM failed");
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark format conversions
fn bench_format_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_conversion");

    for size in [100, 500, 1000].iter() {
        for density in [0.01, 0.05, 0.1].iter() {
            let csr = random_sparse_matrix(*size, *size, *density);
            let coo_data = csr.to_coo();

            let nnz = csr.nnz();
            group.throughput(Throughput::Elements(nnz as u64));

            // COO -> CSR
            group.bench_with_input(
                BenchmarkId::new("coo_to_csr", format!("{}x{}_d{}", size, size, density)),
                &coo_data,
                |b, coo| {
                    b.iter(|| {
                        let result = CsrMatrix::from_coo(black_box(coo));
                        let _ = black_box(result);
                    });
                },
            );

            // CSR -> COO
            group.bench_with_input(
                BenchmarkId::new("csr_to_coo", format!("{}x{}_d{}", size, size, density)),
                &csr,
                |b, csr| {
                    b.iter(|| {
                        let result = csr.to_coo();
                        black_box(result);
                    });
                },
            );

            // CSR -> CSC
            group.bench_with_input(
                BenchmarkId::new("csr_to_csc", format!("{}x{}_d{}", size, size, density)),
                &csr,
                |b, csr| {
                    b.iter(|| {
                        let result = csr.to_csc();
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark BCSR operations
fn bench_bcsr(c: &mut Criterion) {
    let mut group = c.benchmark_group("bcsr");

    for size in [100, 400].iter() {
        for block_size in [2, 4].iter() {
            for density in [0.05, 0.1].iter() {
                // Create dense matrix for BCSR
                let sparse_dense = random_dense_matrix(*size, *size);
                // Zero out elements to achieve desired density
                let mut bcsr_dense = sparse_dense.clone();
                let threshold = 1.0 - density;
                for elem in bcsr_dense.iter_mut() {
                    if *elem < threshold {
                        *elem = 0.0;
                    }
                }

                let bcsr = BcsrMatrix::from_dense(
                    &bcsr_dense.view(),
                    (*block_size, *block_size),
                    0.0, // threshold
                )
                .expect("Failed to create BCSR");
                let x = random_dense_vector(*size);

                let nnz = bcsr.nnz();
                if nnz == 0 {
                    continue; // Skip if no nonzeros
                }

                group.throughput(Throughput::Elements(nnz as u64));

                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "{}x{}_b{}_d{}",
                        size, size, block_size, density
                    )),
                    &(bcsr, x),
                    |b, (bcsr, x)| {
                        b.iter(|| {
                            let result = bcsr.spmv(black_box(&x.view())).expect("BCSR SpMV failed");
                            black_box(result);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark CSC operations
fn bench_csc(c: &mut Criterion) {
    let mut group = c.benchmark_group("csc");

    for size in [100, 500].iter() {
        for density in [0.01, 0.05, 0.1].iter() {
            let csr = random_sparse_matrix(*size, *size, *density);
            let csc = csr.to_csc();
            let x = random_dense_vector(*size);

            let nnz = csc.nnz();
            group.throughput(Throughput::Elements(nnz as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_d{}", size, size, density)),
                &(csc, x),
                |b, (csc, x)| {
                    b.iter(|| {
                        let result = csc.matvec(black_box(&x.view())).expect("CSC matvec failed");
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark format comparison for SpMV
fn bench_format_comparison_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_comparison_spmv");

    let size = 500;
    let density = 0.05;

    let csr = random_sparse_matrix(size, size, density);
    let csc = csr.to_csc();
    let x = random_dense_vector(size);

    let nnz = csr.nnz();
    group.throughput(Throughput::Elements(nnz as u64));

    // CSR SpMV
    group.bench_function("csr_spmv", |b| {
        b.iter(|| {
            let result = csr.spmv(black_box(&x.view()));
            let _ = black_box(result);
        });
    });

    // CSC matvec
    group.bench_function("csc_matvec", |b| {
        b.iter(|| {
            let result = csc.matvec(black_box(&x.view())).expect("CSC matvec failed");
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark sparse matrix addition
fn bench_sparse_add(c: &mut Criterion) {
    use tenrso_sparse::ops::sparse_add_csr;

    let mut group = c.benchmark_group("sparse_add");

    for size in [100, 500].iter() {
        for density in [0.01, 0.05, 0.1].iter() {
            let a = random_sparse_matrix(*size, *size, *density);
            let b = random_sparse_matrix(*size, *size, *density);

            let total_nnz = (a.nnz() + b.nnz()) as u64;
            group.throughput(Throughput::Elements(total_nnz));

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_d{}", size, size, density)),
                &(a, b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        let result = sparse_add_csr(black_box(a), black_box(b), 1.0, 1.0)
                            .expect("Sparse add failed");
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory footprint estimation
fn bench_memory_analysis(c: &mut Criterion) {
    use tenrso_sparse::utils::MemoryFootprint;

    let mut group = c.benchmark_group("memory_analysis");

    for size in [100, 500, 1000].iter() {
        for density in [0.01, 0.05, 0.1].iter() {
            let nnz = ((*size * *size) as f64 * density) as usize;

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_d{}", size, size, density)),
                &(*size, nnz),
                |b, &(size, nnz)| {
                    b.iter(|| {
                        let coo = MemoryFootprint::coo::<f64>(&[size, size], nnz);
                        let csr = MemoryFootprint::csr::<f64>(size, nnz);
                        let dense = MemoryFootprint::dense::<f64>(&[size, size]);
                        black_box((coo, csr, dense));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark format recommendation
fn bench_format_recommendation(c: &mut Criterion) {
    use tenrso_sparse::utils::{recommend_format, SparsityStats};

    let mut group = c.benchmark_group("format_recommendation");

    for ndim in [2, 3, 4].iter() {
        for density in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2].iter() {
            let shape = vec![100; *ndim];
            let total: usize = shape.iter().product();
            let nnz = (total as f64 * density) as usize;
            let stats = SparsityStats::from_shape_nnz(shape, nnz);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}D_d{}", ndim, density)),
                &stats,
                |b, stats| {
                    b.iter(|| {
                        let rec = recommend_format(black_box(stats));
                        black_box(rec);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark dense to sparse conversion with thresholding
fn bench_dense_to_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_to_sparse");

    for size in [100, 500].iter() {
        let dense = random_dense_matrix(*size, *size);

        group.throughput(Throughput::Elements((*size * *size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            &dense,
            |b, dense| {
                b.iter(|| {
                    // Threshold at 0.5 to create sparse matrix
                    let mut indices = Vec::new();
                    let mut values = Vec::new();

                    for (idx, &val) in dense.indexed_iter() {
                        if val > 0.5 {
                            indices.push(vec![idx.0, idx.1]);
                            values.push(val);
                        }
                    }

                    let coo =
                        CooTensor::new(black_box(indices), black_box(values), vec![*size, *size]);
                    let _ = black_box(coo);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "csf")]
/// Benchmark CSF operations
fn bench_csf(c: &mut Criterion) {
    use tenrso_sparse::CsfTensor;

    let mut group = c.benchmark_group("csf");

    for size in [20, 40].iter() {
        for density in [0.01, 0.05].iter() {
            // Create 3D sparse tensor
            let total = size * size * size;
            let nnz = ((total as f64) * density) as usize;

            let mut indices = Vec::new();
            let mut values = Vec::new();

            let mut seed = 12345u64;
            for _ in 0..nnz {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let i = (seed % *size as u64) as usize;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (seed % *size as u64) as usize;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let k = (seed % *size as u64) as usize;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (seed % 10000) as f64 / 10000.0;

                indices.push(vec![i, j, k]);
                values.push(val);
            }

            let coo =
                CooTensor::new(indices, values, vec![*size, *size, *size]).expect("Failed COO");

            group.throughput(Throughput::Elements(nnz as u64));

            // COO to CSF conversion
            group.bench_with_input(
                BenchmarkId::new("coo_to_csf", format!("{}^3_d{}", size, density)),
                &coo,
                |b, coo| {
                    b.iter(|| {
                        let csf = CsfTensor::from_coo(black_box(coo), &[0, 1, 2])
                            .expect("Failed CSF conversion");
                        black_box(csf);
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "csf"))]
fn bench_csf(_c: &mut Criterion) {}

/// Benchmark sparse tensor reductions
fn bench_reductions(c: &mut Criterion) {
    use tenrso_sparse::reductions::*;

    let mut group = c.benchmark_group("reductions");

    // Test different sizes and densities
    let configs = vec![
        (100, 100, 0.01),   // Small, very sparse
        (100, 100, 0.1),    // Small, sparse
        (1000, 1000, 0.01), // Large, very sparse
        (1000, 1000, 0.1),  // Large, sparse
    ];

    for (nrows, ncols, density) in configs {
        let size = nrows * ncols;
        let nnz = (size as f64 * density).max(1.0) as usize;

        // Create sparse matrix
        let sparse = random_sparse_matrix(nrows, ncols, density);

        // Convert to COO for reduction operations
        let coo = sparse.to_coo();

        group.throughput(Throughput::Elements(nnz as u64));

        // Benchmark global sum
        group.bench_with_input(
            BenchmarkId::new("sum", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = sum(black_box(coo));
                    black_box(result);
                });
            },
        );

        // Benchmark global max
        group.bench_with_input(
            BenchmarkId::new("max", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = max(black_box(coo)).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark global min
        group.bench_with_input(
            BenchmarkId::new("min", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = min(black_box(coo)).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark global mean
        group.bench_with_input(
            BenchmarkId::new("mean", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = mean(black_box(coo)).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark sum along axis 0 (column sums)
        group.bench_with_input(
            BenchmarkId::new("sum_axis0", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = sum_axis(black_box(coo), 0).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark sum along axis 1 (row sums)
        group.bench_with_input(
            BenchmarkId::new("sum_axis1", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = sum_axis(black_box(coo), 1).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark max along axis 0
        group.bench_with_input(
            BenchmarkId::new("max_axis0", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = max_axis(black_box(coo), 0).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark mean along axis 0
        group.bench_with_input(
            BenchmarkId::new("mean_axis0", format!("{}x{}_d{}", nrows, ncols, density)),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let result = mean_axis(black_box(coo), 0).unwrap();
                    black_box(result);
                });
            },
        );
    }

    // Benchmark 3D reductions
    let sizes_3d = vec![(10, 0.1), (20, 0.05), (50, 0.01)];

    for (size, density) in sizes_3d {
        let total = size * size * size;
        let nnz = (total as f64 * density).max(1.0) as usize;

        // Generate 3D sparse tensor
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut seed = 11111u64;

        for _ in 0..nnz {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let i = (seed % size as u64) as usize;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (seed % size as u64) as usize;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let k = (seed % size as u64) as usize;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let val = (seed % 10000) as f64 / 10000.0;

            indices.push(vec![i, j, k]);
            values.push(val);
        }

        let coo = CooTensor::new(indices, values, vec![size, size, size]).expect("Failed COO");

        group.throughput(Throughput::Elements(nnz as u64));

        // Benchmark 3D sum along different axes
        for axis in 0..3 {
            group.bench_with_input(
                BenchmarkId::new("sum_axis", format!("{}^3_d{}_axis{}", size, density, axis)),
                &(&coo, axis),
                |b, (coo, axis)| {
                    b.iter(|| {
                        let result = sum_axis(black_box(coo), *axis).unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark parallel SpMV operations
fn bench_par_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_spmv");

    for size in [100, 500, 1000, 2000] {
        for density in [0.01, 0.05, 0.1] {
            let csr = random_sparse_matrix(size, size, density);
            let x = random_dense_vector(size);

            let id = BenchmarkId::from_parameter(format!("{}x{}_d{}", size, size, density));
            group.throughput(Throughput::Elements((csr.nnz() * 2) as u64)); // 2 ops per nnz

            #[cfg(feature = "parallel")]
            group.bench_with_input(id, &(&csr, &x), |b, (csr, x)| {
                b.iter(|| black_box(tenrso_sparse::parallel::par_spmv(csr, &x.view()).unwrap()));
            });
        }
    }

    group.finish();
}

/// Benchmark parallel SpMM operations
fn bench_par_spmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_spmm");

    for size in [100, 500, 1000] {
        for density in [0.01, 0.05] {
            let csr = random_sparse_matrix(size, size, density);
            let b = random_dense_matrix(size, 10);

            let id = BenchmarkId::from_parameter(format!("{}x{}_d{}_k10", size, size, density));
            group.throughput(Throughput::Elements((csr.nnz() * 10 * 2) as u64));

            #[cfg(feature = "parallel")]
            group.bench_with_input(id, &(&csr, &b), |b_bench, (csr, b)| {
                b_bench
                    .iter(|| black_box(tenrso_sparse::parallel::par_spmm(csr, &b.view()).unwrap()));
            });
        }
    }

    group.finish();
}

/// Benchmark parallel format conversions
fn bench_par_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_conversions");

    for size in [100, 500, 1000] {
        for density in [0.01, 0.05, 0.1] {
            let indices: Vec<Vec<usize>> = (0..((size * size) as f64 * density) as usize)
                .map(|i| vec![i % size, (i * 7) % size])
                .collect();
            let values: Vec<f64> = (0..indices.len()).map(|i| i as f64).collect();
            let coo = CooTensor::new(indices, values, vec![size, size]).unwrap();

            let id =
                BenchmarkId::from_parameter(format!("COO->CSR_{}x{}_d{}", size, size, density));

            #[cfg(feature = "parallel")]
            group.bench_with_input(id, &coo, |b, coo| {
                b.iter(|| black_box(tenrso_sparse::parallel::par_coo_to_csr(coo).unwrap()));
            });
        }
    }

    group.finish();
}

/// Benchmark sequential vs parallel comparison
fn bench_seq_vs_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_vs_par_comparison");

    let sizes = [500, 1000, 2000];

    for size in sizes {
        let csr = random_sparse_matrix(size, size, 0.05);
        let x = random_dense_vector(size);

        // Sequential SpMV
        group.bench_with_input(
            BenchmarkId::new("spmv_seq", size),
            &(&csr, &x),
            |b, (csr, x)| {
                b.iter(|| black_box(csr.spmv(&x.view()).unwrap()));
            },
        );

        // Parallel SpMV
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("spmv_par", size),
            &(&csr, &x),
            |b, (csr, x)| {
                b.iter(|| black_box(tenrso_sparse::parallel::par_spmv(csr, &x.view()).unwrap()));
            },
        );

        let b = random_dense_matrix(size, 10);

        // Sequential SpMM
        group.bench_with_input(
            BenchmarkId::new("spmm_seq", size),
            &(&csr, &b),
            |b_bench, (csr, b)| {
                b_bench.iter(|| black_box(csr.spmm(&b.view()).unwrap()));
            },
        );

        // Parallel SpMM
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("spmm_par", size),
            &(&csr, &b),
            |b_bench, (csr, b)| {
                b_bench
                    .iter(|| black_box(tenrso_sparse::parallel::par_spmm(csr, &b.view()).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark ELL format SpMV operations
fn bench_ell_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("ell_spmv");

    for size in [100, 500, 1000] {
        for density in [0.01, 0.05, 0.1] {
            let csr = random_sparse_matrix(size, size, density);
            let ell = EllMatrix::from_csr(&csr);
            let x = random_dense_vector(size);

            let fill_eff = ell.fill_efficiency();
            let id = BenchmarkId::from_parameter(format!(
                "{}x{}_d{:.2}_eff{:.0}%",
                size,
                size,
                density,
                fill_eff * 100.0
            ));
            group.throughput(Throughput::Elements((csr.nnz() * 2) as u64));

            group.bench_with_input(id, &(&ell, &x), |b, (ell, x)| {
                b.iter(|| black_box(ell.spmv(x).unwrap()));
            });
        }
    }

    group.finish();
}

/// Generate a banded sparse matrix for DIA benchmarks
fn random_banded_matrix(size: usize, bandwidth: usize) -> CsrMatrix<f64> {
    let mut indices = Vec::new();
    let mut values = Vec::new();

    let mut seed = 11111u64;
    for i in 0..size {
        for offset in -(bandwidth as isize)..=(bandwidth as isize) {
            let j = i as isize + offset;
            if j >= 0 && j < size as isize {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (seed % 10000) as f64 / 10000.0;
                indices.push(vec![i, j as usize]);
                values.push(val);
            }
        }
    }

    let coo = CooTensor::new(indices, values, vec![size, size]).expect("Failed to create COO");
    CsrMatrix::from_coo(&coo).expect("Failed to convert to CSR")
}

/// Benchmark DIA format SpMV operations
fn bench_dia_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("dia_spmv");

    for size in [100, 500, 1000] {
        for bandwidth in [1, 2, 5] {
            let csr = random_banded_matrix(size, bandwidth);
            let dia = DiaMatrix::from_csr(&csr);
            let x = random_dense_vector(size);

            let (lower, upper) = dia.bandwidth();
            let id = BenchmarkId::from_parameter(format!(
                "{}x{}_bw{}({}+{})",
                size,
                size,
                lower + upper,
                lower,
                upper
            ));
            group.throughput(Throughput::Elements((csr.nnz() * 2) as u64));

            group.bench_with_input(id, &(&dia, &x), |b, (dia, x)| {
                b.iter(|| black_box(dia.spmv(x).unwrap()));
            });
        }
    }

    group.finish();
}

/// Benchmark ELL vs CSR vs DIA SpMV comparison
fn bench_format_spmv_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_format_comparison");

    let size = 500;

    // Test 1: Uniform sparsity (good for ELL)
    let csr_uniform = random_sparse_matrix(size, size, 0.05);
    let ell_uniform = EllMatrix::from_csr(&csr_uniform);
    let x = random_dense_vector(size);

    group.throughput(Throughput::Elements((csr_uniform.nnz() * 2) as u64));

    group.bench_function("CSR_uniform", |b| {
        b.iter(|| black_box(csr_uniform.spmv(&x.view()).unwrap()));
    });

    group.bench_function("ELL_uniform", |b| {
        b.iter(|| black_box(ell_uniform.spmv(&x).unwrap()));
    });

    // Test 2: Banded structure (good for DIA)
    let csr_banded = random_banded_matrix(size, 2);
    let dia_banded = DiaMatrix::from_csr(&csr_banded);

    group.throughput(Throughput::Elements((csr_banded.nnz() * 2) as u64));

    group.bench_function("CSR_banded", |b| {
        b.iter(|| black_box(csr_banded.spmv(&x.view()).unwrap()));
    });

    group.bench_function("DIA_banded", |b| {
        b.iter(|| black_box(dia_banded.spmv(&x).unwrap()));
    });

    group.finish();
}

/// Benchmark element-wise operations (new operations)
fn bench_element_wise_ops(c: &mut Criterion) {
    use tenrso_sparse::ops::*;

    let mut group = c.benchmark_group("element_wise_ops");

    for size in [500, 1000].iter() {
        for density in [0.01, 0.05].iter() {
            let a = random_sparse_matrix(*size, *size, *density);
            let b = random_sparse_matrix(*size, *size, *density);

            let nnz = a.nnz();
            group.throughput(Throughput::Elements(nnz as u64));

            // Benchmark divide
            group.bench_with_input(
                BenchmarkId::new("divide", format!("{}x{}_d{}", size, size, density)),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        let result = sparse_divide_csr(black_box(a), black_box(b));
                        let _ = black_box(result);
                    });
                },
            );

            // Benchmark clip
            group.bench_with_input(
                BenchmarkId::new("clip", format!("{}x{}_d{}", size, size, density)),
                &a,
                |bench, a| {
                    bench.iter(|| {
                        let result = sparse_clip_csr(black_box(a), 0.2, 0.8);
                        let _ = black_box(result);
                    });
                },
            );

            // Benchmark floor
            group.bench_with_input(
                BenchmarkId::new("floor", format!("{}x{}_d{}", size, size, density)),
                &a,
                |bench, a| {
                    bench.iter(|| {
                        let result = sparse_floor_csr(black_box(a));
                        let _ = black_box(result);
                    });
                },
            );

            // Benchmark ceil
            group.bench_with_input(
                BenchmarkId::new("ceil", format!("{}x{}_d{}", size, size, density)),
                &a,
                |bench, a| {
                    bench.iter(|| {
                        let result = sparse_ceil_csr(black_box(a));
                        let _ = black_box(result);
                    });
                },
            );

            // Benchmark round
            group.bench_with_input(
                BenchmarkId::new("round", format!("{}x{}_d{}", size, size, density)),
                &a,
                |bench, a| {
                    bench.iter(|| {
                        let result = sparse_round_csr(black_box(a));
                        let _ = black_box(result);
                    });
                },
            );

            // Benchmark atan2
            group.bench_with_input(
                BenchmarkId::new("atan2", format!("{}x{}_d{}", size, size, density)),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        let result = sparse_atan2_csr(black_box(a), black_box(b));
                        let _ = black_box(result);
                    });
                },
            );

            // Benchmark hypot
            group.bench_with_input(
                BenchmarkId::new("hypot", format!("{}x{}_d{}", size, size, density)),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        let result = sparse_hypot_csr(black_box(a), black_box(b));
                        let _ = black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_spmv,
    bench_spmm,
    bench_spspmm,
    bench_format_conversion,
    bench_bcsr,
    bench_csc,
    bench_format_comparison_spmv,
    bench_sparse_add,
    bench_memory_analysis,
    bench_format_recommendation,
    bench_dense_to_sparse,
    bench_csf,
    bench_reductions,
    bench_par_spmv,
    bench_par_spmm,
    bench_par_conversions,
    bench_seq_vs_par,
    bench_ell_spmv,
    bench_dia_spmv,
    bench_format_spmv_comparison,
    bench_element_wise_ops
);
criterion_main!(benches);
