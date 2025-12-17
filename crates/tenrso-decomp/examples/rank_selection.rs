//! Comprehensive example demonstrating automated rank selection for tensor decompositions
//!
//! This example shows how to use the rank selection utilities to automatically
//! determine the optimal rank for CP, Tucker, and TT decompositions.
//!
//! Run with: cargo run --example rank_selection --release

use anyhow::Result;
use tenrso_core::DenseND;
use tenrso_decomp::{
    cp_als, cp_num_params, rank_selection::*, tucker_hosvd, tucker_hosvd_auto, tucker_num_params,
    InitStrategy, TuckerRankSelection,
};

fn main() -> Result<()> {
    println!("{}", "=".repeat(80));
    println!("Automated Rank Selection for Tensor Decompositions");
    println!("{}", "=".repeat(80));
    println!();

    example_1_information_criteria()?;
    example_2_scree_plot_analysis()?;
    example_3_cross_validation()?;
    example_4_combined_strategy()?;
    example_5_tucker_auto_rank()?;
    example_6_strategy_comparison()?;

    println!("\n{}", "=".repeat(80));
    println!("All rank selection examples completed successfully!");
    println!("{}", "=".repeat(80));

    Ok(())
}

/// Example 1: Using Information Criteria (BIC, AIC, MDL) for rank selection
fn example_1_information_criteria() -> Result<()> {
    println!("Example 1: Information Criteria for Rank Selection");
    println!("{}", "-".repeat(80));

    let tensor = DenseND::<f64>::random_uniform(&[30, 30, 30], 0.0, 1.0);
    let candidate_ranks = vec![3, 5, 7, 10, 12, 15];

    println!(
        "Evaluating CP decomposition for ranks: {:?}",
        candidate_ranks
    );
    println!();

    let mut errors = Vec::new();
    let mut params = Vec::new();

    // Evaluate all candidate ranks
    for &rank in &candidate_ranks {
        let cp = cp_als(&tensor, rank, 30, 1e-4, InitStrategy::Svd, None)?;
        let error = 1.0 - cp.fit;
        let num_params = cp_num_params(tensor.shape(), rank);

        errors.push(error);
        params.push(num_params);

        println!(
            "  Rank {:2}: fit={:.4}, error={:.4}, params={:5}",
            rank, cp.fit, error, num_params
        );
    }

    println!();

    // Use BIC for rank selection
    let num_obs: usize = tensor.shape().iter().product();
    let result_bic = select_rank_auto(
        &errors,
        &params,
        num_obs,
        RankSelectionStrategy::InformationCriterion(InformationCriterion::BIC),
    );

    println!("BIC Selection Results:");
    println!("  Selected rank: {}", result_bic.rank);
    println!("  Error: {:.6}", result_bic.error);
    println!("  BIC value: {:.2}", result_bic.criterion_value);
    println!(
        "  Improvement over rank-1: {:.1}%",
        result_bic.improvement_ratio() * 100.0
    );
    println!("  Has elbow: {}", result_bic.has_elbow());

    // Compare with AIC
    let result_aic = select_rank_auto(
        &errors,
        &params,
        num_obs,
        RankSelectionStrategy::InformationCriterion(InformationCriterion::AIC),
    );

    println!();
    println!("AIC vs BIC Comparison:");
    println!(
        "  BIC selected rank: {} (more conservative)",
        result_bic.rank
    );
    println!(
        "  AIC selected rank: {} (less conservative)",
        result_aic.rank
    );
    println!();

    Ok(())
}

/// Example 2: Scree Plot Analysis for variance-based rank selection
fn example_2_scree_plot_analysis() -> Result<()> {
    println!("Example 2: Scree Plot Analysis");
    println!("{}", "-".repeat(80));

    // Simulate singular values from a Tucker decomposition
    let singular_values = vec![
        50.0, 35.0, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1,
    ];

    println!("Analyzing singular values: {:?}", &singular_values[..6]);
    println!("  ... and {} more values", singular_values.len() - 6);
    println!();

    // Create scree plot data
    let scree = ScreePlotData::new(singular_values, 0.9, 0.95);

    println!("Variance Analysis:");
    println!("  Total components: {}", scree.singular_values.len());
    println!(
        "  Suggested rank (elbow): {:?}",
        scree.suggested_rank.unwrap_or(0)
    );
    println!(
        "  Rank for 90% variance: {:?}",
        scree.suggested_rank_90.unwrap_or(0)
    );
    println!(
        "  Rank for 95% variance: {:?}",
        scree.suggested_rank_95.unwrap_or(0)
    );
    println!();

    // Show variance explained by each component
    println!("Variance Explained by Component:");
    for (i, (&var, &cum_var)) in scree
        .variance_explained
        .iter()
        .zip(scree.cumulative_variance.iter())
        .enumerate()
        .take(8)
    {
        println!(
            "  Component {:2}: {:.1}% (cumulative: {:.1}%)",
            i + 1,
            var * 100.0,
            cum_var * 100.0
        );
    }

    // Test custom thresholds
    println!();
    println!("Custom Variance Thresholds:");
    for threshold in [0.70, 0.80, 0.85, 0.90, 0.95, 0.99] {
        if let Some(rank) = scree.rank_for_variance(threshold) {
            println!(
                "  {}% variance requires rank: {}",
                (threshold * 100.0) as usize,
                rank
            );
        }
    }
    println!();

    Ok(())
}

/// Example 3: Cross-Validation for rank selection
fn example_3_cross_validation() -> Result<()> {
    println!("Example 3: Cross-Validation Rank Selection");
    println!("{}", "-".repeat(80));

    let tensor = DenseND::<f64>::random_uniform(&[25, 25, 25], 0.0, 1.0);
    let candidate_ranks = vec![3, 5, 7, 9, 11];

    println!("Using 80% train / 20% validation split");
    println!("Candidate ranks: {:?}", candidate_ranks);
    println!();

    // Create train/validation split
    let (train_mask, val_mask) = create_cv_split(tensor.shape(), 0.8);

    let mut validation_errors = Vec::new();
    let mut params = Vec::new();

    println!("Evaluating ranks on validation set:");
    for &rank in &candidate_ranks {
        // Train on training set using completion
        use tenrso_decomp::cp_completion;
        let cp = cp_completion(&tensor, &train_mask, rank, 40, 1e-4, InitStrategy::Svd)?;

        // Evaluate on validation set
        let reconstruction = cp.reconstruct(tensor.shape())?;
        let val_error = masked_reconstruction_error(&tensor, &reconstruction, &val_mask);

        validation_errors.push(val_error);
        params.push(cp_num_params(tensor.shape(), rank));

        println!(
            "  Rank {:2}: train fit={:.4}, val error={:.6}",
            rank, cp.fit, val_error
        );
    }

    println!();

    // Select best rank based on validation error
    let num_obs: usize = tensor.shape().iter().product();
    let result = select_rank_auto(
        &validation_errors,
        &params,
        num_obs,
        RankSelectionStrategy::CrossValidation,
    );

    println!("Cross-Validation Results:");
    println!("  Selected rank: {}", result.rank);
    println!("  Validation error: {:.6}", result.error);
    println!();

    Ok(())
}

/// Example 4: Combined strategy (IC + elbow verification)
fn example_4_combined_strategy() -> Result<()> {
    println!("Example 4: Combined IC + Elbow Strategy");
    println!("{}", "-".repeat(80));

    let tensor = DenseND::<f64>::random_uniform(&[28, 28, 28], 0.0, 1.0);
    let candidate_ranks = vec![2, 4, 6, 8, 10, 12, 15, 18];

    println!("Evaluating ranks: {:?}", candidate_ranks);
    println!();

    let mut errors = Vec::new();
    let mut params = Vec::new();

    for &rank in &candidate_ranks {
        let cp = cp_als(&tensor, rank, 25, 1e-4, InitStrategy::Random, None)?;
        errors.push(1.0 - cp.fit);
        params.push(cp_num_params(tensor.shape(), rank));
    }

    let num_obs: usize = tensor.shape().iter().product();

    // Use combined strategy
    let result = select_rank_auto(
        &errors,
        &params,
        num_obs,
        RankSelectionStrategy::Combined(InformationCriterion::BIC),
    );

    println!("Combined Strategy Results:");
    println!("  Selected rank: {}", result.rank);
    println!("  Error: {:.6}", result.error);
    println!("  Elbow detected: {}", result.has_elbow());
    println!();

    // Show error curve
    println!("Error Curve:");
    for (i, &error) in errors.iter().enumerate() {
        let marker = if candidate_ranks[i] == result.rank {
            " ← SELECTED"
        } else if result.has_elbow() && candidate_ranks[i] == result.rank {
            " ← ELBOW"
        } else {
            ""
        };
        println!(
            "  Rank {:2}: error={:.6}{}",
            candidate_ranks[i], error, marker
        );
    }
    println!();

    Ok(())
}

/// Example 5: Tucker automatic rank selection
fn example_5_tucker_auto_rank() -> Result<()> {
    println!("Example 5: Tucker Automatic Rank Selection");
    println!("{}", "-".repeat(80));

    let tensor = DenseND::<f64>::random_uniform(&[40, 40, 40], 0.0, 1.0);

    println!("Original tensor shape: {:?}", tensor.shape());
    println!();

    // Energy-based selection (90% variance)
    println!("Strategy 1: Energy-based (90% variance)");
    let tucker_90 = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.9))?;
    let ranks_90: Vec<usize> = tucker_90.factors.iter().map(|f| f.ncols()).collect();
    println!("  Selected ranks: {:?}", ranks_90);
    println!("  Core shape: {:?}", tucker_90.core.shape());
    let params_90 = tucker_num_params(tensor.shape(), &ranks_90);
    println!("  Parameters: {}", params_90);
    let recon_90 = tucker_90.reconstruct()?;
    let error_90 = (&tensor - &recon_90).frobenius_norm() / tensor.frobenius_norm();
    println!("  Relative error: {:.6}", error_90);
    println!();

    // Energy-based selection (95% variance)
    println!("Strategy 2: Energy-based (95% variance)");
    let tucker_95 = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.95))?;
    let ranks_95: Vec<usize> = tucker_95.factors.iter().map(|f| f.ncols()).collect();
    println!("  Selected ranks: {:?}", ranks_95);
    let params_95 = tucker_num_params(tensor.shape(), &ranks_95);
    println!("  Parameters: {}", params_95);
    let recon_95 = tucker_95.reconstruct()?;
    let error_95 = (&tensor - &recon_95).frobenius_norm() / tensor.frobenius_norm();
    println!("  Relative error: {:.6}", error_95);
    println!();

    // Threshold-based selection
    println!("Strategy 3: Threshold-based (σ > 0.1 × σ_max)");
    let tucker_thresh = tucker_hosvd_auto(&tensor, TuckerRankSelection::Threshold(0.1))?;
    let ranks_thresh: Vec<usize> = tucker_thresh.factors.iter().map(|f| f.ncols()).collect();
    println!("  Selected ranks: {:?}", ranks_thresh);
    let recon_thresh = tucker_thresh.reconstruct()?;
    let error_thresh = (&tensor - &recon_thresh).frobenius_norm() / tensor.frobenius_norm();
    println!("  Relative error: {:.6}", error_thresh);
    println!();

    // Manual for comparison
    println!("Strategy 4: Manual (fixed ranks [20, 20, 20])");
    let tucker_manual = tucker_hosvd(&tensor, &[20, 20, 20])?;
    let recon_manual = tucker_manual.reconstruct()?;
    let error_manual = (&tensor - &recon_manual).frobenius_norm() / tensor.frobenius_norm();
    println!("  Relative error: {:.6}", error_manual);
    println!();

    Ok(())
}

/// Example 6: Comparing all rank selection strategies
fn example_6_strategy_comparison() -> Result<()> {
    println!("Example 6: Comprehensive Strategy Comparison");
    println!("{}", "-".repeat(80));

    let tensor = DenseND::<f64>::random_uniform(&[30, 30, 30], 0.0, 1.0);
    let candidate_ranks = vec![3, 5, 7, 9, 11, 13, 15];

    println!("Comparing all rank selection strategies");
    println!("Candidate ranks: {:?}", candidate_ranks);
    println!();

    // Evaluate all ranks
    let mut errors = Vec::new();
    let mut params = Vec::new();

    for &rank in &candidate_ranks {
        let cp = cp_als(&tensor, rank, 25, 1e-4, InitStrategy::Svd, None)?;
        errors.push(1.0 - cp.fit);
        params.push(cp_num_params(tensor.shape(), rank));
    }

    let num_obs: usize = tensor.shape().iter().product();

    // Define all strategies
    let strategies = vec![
        (
            "BIC",
            RankSelectionStrategy::InformationCriterion(InformationCriterion::BIC),
        ),
        (
            "AIC",
            RankSelectionStrategy::InformationCriterion(InformationCriterion::AIC),
        ),
        (
            "MDL",
            RankSelectionStrategy::InformationCriterion(InformationCriterion::MDL),
        ),
        ("Elbow", RankSelectionStrategy::ElbowDetection),
        (
            "Combined",
            RankSelectionStrategy::Combined(InformationCriterion::BIC),
        ),
    ];

    println!("Strategy Comparison Results:");
    println!("{:-<60}", "");
    println!(
        "{:<15} {:<10} {:<12} {:<12}",
        "Strategy", "Rank", "Error", "Has Elbow"
    );
    println!("{:-<60}", "");

    for (name, strategy) in strategies {
        let result = select_rank_auto(&errors, &params, num_obs, strategy);
        println!(
            "{:<15} {:<10} {:<12.6} {:<12}",
            name,
            result.rank,
            result.error,
            if result.has_elbow() { "Yes" } else { "No" }
        );
    }

    println!("{:-<60}", "");
    println!();

    println!("Recommendation:");
    println!("  • Use BIC for general-purpose rank selection (most conservative)");
    println!("  • Use AIC when you want slightly higher ranks");
    println!("  • Use Elbow when you want data-driven selection without IC");
    println!("  • Use Combined for robust selection with elbow verification");
    println!();

    Ok(())
}
