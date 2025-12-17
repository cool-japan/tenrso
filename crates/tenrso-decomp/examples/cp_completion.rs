//! CP Tensor Completion Example
//!
//! Demonstrates how to use CP-WOPT (Weighted Optimization) for tensor completion,
//! i.e., predicting missing entries in tensors.
//!
//! Run with: cargo run --release --example cp_completion

use scirs2_core::ndarray_ext::Array;
use scirs2_core::random::{thread_rng, Rng};
use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, cp_completion, InitStrategy};

fn main() -> anyhow::Result<()> {
    println!("=== CP Tensor Completion Examples ===\n");

    example_1_basic_completion()?;
    example_2_recommender_system()?;
    example_3_low_rank_recovery()?;
    example_4_varying_missing_rates()?;
    example_5_comparison_with_full()?;

    Ok(())
}

/// Example 1: Basic tensor completion with 50% missing data
fn example_1_basic_completion() -> anyhow::Result<()> {
    println!("--- Example 1: Basic Tensor Completion ---");

    // Create a small tensor
    let data = Array::from_shape_fn((8, 8, 8), |(i, j, k)| {
        (i as f64 * 0.1 + j as f64 * 0.2 + k as f64 * 0.15).sin()
    });

    // Create mask: randomly observe 50% of entries
    let mut mask = Array::zeros(vec![8, 8, 8]);
    let mut rng = thread_rng();
    let mut n_observed = 0;

    for idx in mask.iter_mut() {
        if rng.random::<f64>() < 0.5 {
            *idx = 1.0;
            n_observed += 1;
        }
    }

    let tensor = DenseND::from_array(data.into_dyn());
    let mask_tensor = DenseND::from_array(mask.into_dyn());

    println!("Tensor shape: {:?}", tensor.shape());
    println!("Observed entries: {} / {}", n_observed, 8 * 8 * 8);
    println!(
        "Observation rate: {:.1}%",
        100.0 * n_observed as f64 / (8.0 * 8.0 * 8.0)
    );

    // Perform completion
    let cp = cp_completion(&tensor, &mask_tensor, 4, 100, 1e-4, InitStrategy::Svd)?;

    println!("Converged in {} iterations", cp.iters);
    println!("Fit on observed entries: {:.4}", cp.fit);

    // Reconstruct complete tensor
    let completed = cp.reconstruct(tensor.shape())?;
    println!("Completed tensor shape: {:?}\n", completed.shape());

    Ok(())
}

/// Example 2: Simulated recommender system (user-item-context)
fn example_2_recommender_system() -> anyhow::Result<()> {
    println!("--- Example 2: Recommender System Simulation ---");

    // Simulate a user-item-context tensor
    // Users x Items x Contexts (e.g., time of day, device type)
    let n_users = 20;
    let n_items = 30;
    let n_contexts = 5;

    // Create sparse observations (only 5% of ratings are observed)
    let mut data = Array::zeros(vec![n_users, n_items, n_contexts]);
    let mut mask = Array::zeros(vec![n_users, n_items, n_contexts]);
    let mut rng = thread_rng();
    let mut n_ratings = 0;

    for i in 0..n_users {
        for j in 0..n_items {
            for k in 0..n_contexts {
                if rng.random::<f64>() < 0.05 {
                    // 5% observation rate
                    // Simulate rating based on user/item/context factors
                    let rating = (i as f64 * 0.1 + j as f64 * 0.05 + k as f64 * 0.2) % 5.0 + 1.0;
                    data[[i, j, k]] = rating;
                    mask[[i, j, k]] = 1.0;
                    n_ratings += 1;
                }
            }
        }
    }

    let tensor = DenseND::from_array(data.into_dyn());
    let mask_tensor = DenseND::from_array(mask.into_dyn());

    println!(
        "Users: {}, Items: {}, Contexts: {}",
        n_users, n_items, n_contexts
    );
    println!(
        "Observed ratings: {} / {} ({:.2}%)",
        n_ratings,
        n_users * n_items * n_contexts,
        100.0 * n_ratings as f64 / (n_users * n_items * n_contexts) as f64
    );

    // Complete the tensor
    let cp = cp_completion(&tensor, &mask_tensor, 5, 200, 1e-5, InitStrategy::Random)?;

    println!("CP rank: 5");
    println!("Converged in {} iterations", cp.iters);
    println!("Fit on observed ratings: {:.4}", cp.fit);

    // Reconstruct to get predictions
    let _predictions = cp.reconstruct(&[n_users, n_items, n_contexts])?;

    println!("Predictions ready for all user-item-context combinations");
    println!("Can now recommend items for users in any context\n");

    Ok(())
}

/// Example 3: Low-rank tensor recovery
fn example_3_low_rank_recovery() -> anyhow::Result<()> {
    println!("--- Example 3: Low-Rank Tensor Recovery ---");

    // Create a perfect rank-3 tensor
    let factor1 = Array::from_shape_fn((10, 3), |(i, r)| (i + r) as f64 / 10.0);
    let factor2 = Array::from_shape_fn((10, 3), |(i, r)| (i + r + 1) as f64 / 10.0);
    let factor3 = Array::from_shape_fn((10, 3), |(i, r)| (i + r + 2) as f64 / 10.0);

    // Reconstruct the ground truth tensor
    let factor_views = vec![factor1.view(), factor2.view(), factor3.view()];
    let ground_truth_arr = tenrso_kernels::cp_reconstruct(&factor_views, None)?;
    let ground_truth = DenseND::from_array(ground_truth_arr);

    // Observe only 30% of entries
    let mut mask = Array::zeros(vec![10, 10, 10]);
    let mut rng = thread_rng();

    for idx in mask.iter_mut() {
        if rng.random::<f64>() < 0.3 {
            *idx = 1.0;
        }
    }

    let mask_tensor = DenseND::from_array(mask.into_dyn());

    println!("True rank: 3");
    println!("Observation rate: 30%");

    // Try to recover with correct rank
    let cp = cp_completion(&ground_truth, &mask_tensor, 3, 150, 1e-5, InitStrategy::Svd)?;

    println!("Completion with rank 3:");
    println!("  Iterations: {}", cp.iters);
    println!("  Fit: {:.4}", cp.fit);

    // Compute error on full tensor
    let recovered = cp.reconstruct(&[10, 10, 10])?;
    let diff = &ground_truth - &recovered;
    let error_norm = diff.frobenius_norm();
    let ground_truth_norm = ground_truth.frobenius_norm();
    let relative_error = error_norm / ground_truth_norm;

    println!("  Relative recovery error: {:.6}", relative_error);
    println!("  (includes both observed and missing entries)\n");

    Ok(())
}

/// Example 4: Effect of varying missing data rates
fn example_4_varying_missing_rates() -> anyhow::Result<()> {
    println!("--- Example 4: Varying Missing Data Rates ---");

    let tensor = DenseND::<f64>::random_uniform(&[12, 12, 12], 0.0, 1.0);

    for obs_rate in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        // Create mask with varying observation rates
        let mut mask = Array::zeros(vec![12, 12, 12]);
        let mut rng = thread_rng();

        for idx in mask.iter_mut() {
            if rng.random::<f64>() < *obs_rate {
                *idx = 1.0;
            }
        }

        let mask_tensor = DenseND::from_array(mask.into_dyn());

        // Complete
        let cp = cp_completion(&tensor, &mask_tensor, 6, 100, 1e-4, InitStrategy::Random)?;

        println!(
            "Observation rate {:.0}%: fit={:.4}, iters={}",
            obs_rate * 100.0,
            cp.fit,
            cp.iters
        );
    }

    println!();
    Ok(())
}

/// Example 5: Comparison with full tensor decomposition
fn example_5_comparison_with_full() -> anyhow::Result<()> {
    println!("--- Example 5: Completion vs. Full Decomposition ---");

    // Create a tensor
    let tensor = DenseND::<f64>::random_uniform(&[15, 15, 15], 0.0, 1.0);

    // Full CP decomposition (all entries observed)
    let cp_full = cp_als(&tensor, 5, 100, 1e-4, InitStrategy::Svd, None)?;

    println!("Full CP-ALS (100% observed):");
    println!("  Iterations: {}", cp_full.iters);
    println!("  Fit: {:.4}", cp_full.fit);

    // Completion with 70% observed
    let mut mask = Array::zeros(vec![15, 15, 15]);
    let mut rng = thread_rng();

    for idx in mask.iter_mut() {
        if rng.random::<f64>() < 0.7 {
            *idx = 1.0;
        }
    }

    let mask_tensor = DenseND::from_array(mask.into_dyn());
    let cp_completion = cp_completion(&tensor, &mask_tensor, 5, 100, 1e-4, InitStrategy::Svd)?;

    println!("\nCP Completion (70% observed):");
    println!("  Iterations: {}", cp_completion.iters);
    println!("  Fit on observed: {:.4}", cp_completion.fit);

    // Compare reconstructions
    let full_recon = cp_full.reconstruct(&[15, 15, 15])?;
    let completion_recon = cp_completion.reconstruct(&[15, 15, 15])?;

    let diff = &full_recon - &completion_recon;
    let relative_diff = diff.frobenius_norm() / full_recon.frobenius_norm();

    println!("\nReconstruction difference:");
    println!("  Relative norm: {:.4}", relative_diff);
    println!("  (difference between full vs. completion reconstructions)\n");

    Ok(())
}
