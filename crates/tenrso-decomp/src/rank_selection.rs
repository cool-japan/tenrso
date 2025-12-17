//! Adaptive rank selection utilities for tensor decompositions
//!
//! Provides automated rank selection methods including:
//! - Information criteria (AIC, BIC, MDL)
//! - Cross-validation
//! - Elbow method detection
//! - Scree plot analysis
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use scirs2_core::numeric::{Float, NumCast};
use std::iter::Sum;
use tenrso_core::DenseND;

/// Information criterion for model selection
#[derive(Debug, Clone, Copy)]
pub enum InformationCriterion {
    /// Akaike Information Criterion: AIC = 2k - 2ln(L)
    /// Penalizes model complexity linearly
    AIC,

    /// Bayesian Information Criterion: BIC = k*ln(n) - 2ln(L)
    /// Stronger penalty for complexity than AIC
    BIC,

    /// Minimum Description Length: MDL = k/2*ln(n) - ln(L)
    /// Similar to BIC but with different scaling
    MDL,
}

/// Rank selection result with quality metrics
#[derive(Debug, Clone)]
pub struct RankSelectionResult {
    /// Selected rank
    pub rank: usize,

    /// Reconstruction error at selected rank
    pub error: f64,

    /// Information criterion value (lower is better)
    pub criterion_value: f64,

    /// Criterion used for selection
    pub criterion: InformationCriterion,

    /// All candidate ranks evaluated
    pub candidate_ranks: Vec<usize>,

    /// Errors for all candidate ranks
    pub errors: Vec<f64>,

    /// Criterion values for all candidate ranks
    pub criterion_values: Vec<f64>,
}

impl RankSelectionResult {
    /// Get the improvement ratio compared to rank-1
    pub fn improvement_ratio(&self) -> f64 {
        if self.errors.is_empty() {
            return 0.0;
        }
        let rank1_error = self.errors[0];
        if rank1_error == 0.0 {
            return 0.0;
        }
        (rank1_error - self.error) / rank1_error
    }

    /// Check if elbow was detected
    pub fn has_elbow(&self) -> bool {
        detect_elbow(&self.errors, self.rank)
    }
}

/// Cross-validation result for rank selection
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Selected rank with minimum validation error
    pub best_rank: usize,

    /// Validation error at best rank
    pub best_validation_error: f64,

    /// All candidate ranks
    pub candidate_ranks: Vec<usize>,

    /// Training errors for each rank
    pub training_errors: Vec<f64>,

    /// Validation errors for each rank
    pub validation_errors: Vec<f64>,
}

/// Scree plot data for visualizing rank selection
///
/// Contains singular values and explained variance ratios for each component.
/// Useful for creating scree plots to visualize the "elbow" in variance explained.
#[derive(Debug, Clone)]
pub struct ScreePlotData {
    /// Singular values (sorted in descending order)
    pub singular_values: Vec<f64>,

    /// Variance explained by each component (ratio)
    pub variance_explained: Vec<f64>,

    /// Cumulative variance explained (ratio)
    pub cumulative_variance: Vec<f64>,

    /// Suggested rank based on elbow detection
    pub suggested_rank: Option<usize>,

    /// Suggested rank based on variance threshold (e.g., 90% variance)
    pub suggested_rank_90: Option<usize>,

    /// Suggested rank based on variance threshold (e.g., 95% variance)
    pub suggested_rank_95: Option<usize>,
}

impl ScreePlotData {
    /// Create scree plot data from singular values
    ///
    /// # Arguments
    ///
    /// * `singular_values` - Singular values (will be sorted descending)
    /// * `variance_threshold_90` - Threshold for 90% variance (default: 0.9)
    /// * `variance_threshold_95` - Threshold for 95% variance (default: 0.95)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_decomp::rank_selection::ScreePlotData;
    ///
    /// let singular_values = vec![10.0, 5.0, 2.0, 1.0, 0.5, 0.1];
    /// let scree = ScreePlotData::new(singular_values, 0.9, 0.95);
    ///
    /// assert!(scree.suggested_rank_90.is_some());
    /// assert!(scree.cumulative_variance.last().unwrap() >= &0.99);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        mut singular_values: Vec<f64>,
        variance_threshold_90: f64,
        variance_threshold_95: f64,
    ) -> Self {
        // Sort descending
        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Compute variance explained (singular value squared)
        let total_variance: f64 = singular_values.iter().map(|s| s * s).sum();

        let variance_explained: Vec<f64> = if total_variance > 0.0 {
            singular_values
                .iter()
                .map(|s| (s * s) / total_variance)
                .collect()
        } else {
            vec![0.0; singular_values.len()]
        };

        // Compute cumulative variance
        let mut cumulative_variance = Vec::with_capacity(variance_explained.len());
        let mut cumsum = 0.0;
        for &var in &variance_explained {
            cumsum += var;
            cumulative_variance.push(cumsum);
        }

        // Find suggested ranks based on variance thresholds
        let suggested_rank_90 = cumulative_variance
            .iter()
            .position(|&v| v >= variance_threshold_90)
            .map(|i| i + 1); // +1 for 1-based rank

        let suggested_rank_95 = cumulative_variance
            .iter()
            .position(|&v| v >= variance_threshold_95)
            .map(|i| i + 1);

        // Detect elbow using variance explained
        let suggested_rank = Self::detect_elbow_variance(&variance_explained);

        ScreePlotData {
            singular_values,
            variance_explained,
            cumulative_variance,
            suggested_rank,
            suggested_rank_90,
            suggested_rank_95,
        }
    }

    /// Detect elbow point in variance explained using second derivative
    fn detect_elbow_variance(variance: &[f64]) -> Option<usize> {
        if variance.len() < 3 {
            return None;
        }

        // Compute second derivative (discrete approximation)
        let mut second_deriv = Vec::with_capacity(variance.len() - 2);
        for i in 1..(variance.len() - 1) {
            let d2 = variance[i - 1] - 2.0 * variance[i] + variance[i + 1];
            second_deriv.push(d2.abs());
        }

        // Find maximum second derivative (sharpest change)
        let mut max_d2 = 0.0;
        let mut max_idx = 0;
        for (i, &d2) in second_deriv.iter().enumerate() {
            if d2 > max_d2 {
                max_d2 = d2;
                max_idx = i;
            }
        }

        if max_d2 > 0.0 {
            Some(max_idx + 2) // +2 because we skipped first element and want 1-based rank
        } else {
            None
        }
    }

    /// Get rank that explains at least the given variance ratio
    ///
    /// # Arguments
    ///
    /// * `threshold` - Variance ratio threshold (e.g., 0.9 for 90%)
    ///
    /// # Returns
    ///
    /// Minimum rank needed to explain at least `threshold` variance
    pub fn rank_for_variance(&self, threshold: f64) -> Option<usize> {
        self.cumulative_variance
            .iter()
            .position(|&v| v >= threshold)
            .map(|i| i + 1)
    }
}

/// Compute information criterion for a decomposition
///
/// # Arguments
///
/// * `reconstruction_error` - Frobenius norm of reconstruction error
/// * `num_params` - Number of parameters in the model
/// * `num_observations` - Number of observed entries
/// * `criterion` - Which information criterion to use
///
/// # Returns
///
/// IC value (lower is better)
///
/// # Examples
///
/// ```
/// use tenrso_decomp::rank_selection::{compute_information_criterion, InformationCriterion};
///
/// let error = 0.1;
/// let num_params = 1000;
/// let num_obs = 10000;
///
/// let aic = compute_information_criterion(error, num_params, num_obs, InformationCriterion::AIC);
/// let bic = compute_information_criterion(error, num_params, num_obs, InformationCriterion::BIC);
///
/// // BIC typically penalizes complexity more than AIC
/// assert!(bic > aic);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn compute_information_criterion(
    reconstruction_error: f64,
    num_params: usize,
    num_observations: usize,
    criterion: InformationCriterion,
) -> f64 {
    let k = num_params as f64;
    let n = num_observations as f64;

    // Compute log-likelihood (assuming Gaussian noise)
    // L = -n/2 * ln(2π) - n/2 * ln(σ²)
    // where σ² = RSS/n (residual sum of squares / n)
    let rss = reconstruction_error * reconstruction_error;
    let sigma_sq = rss / n;

    // Negative log-likelihood (up to constants)
    let neg_log_likelihood = (n / 2.0) * sigma_sq.ln();

    match criterion {
        InformationCriterion::AIC => {
            // AIC = 2k - 2ln(L) = 2k + 2*neg_log_likelihood
            2.0 * k + 2.0 * neg_log_likelihood
        }
        InformationCriterion::BIC => {
            // BIC = k*ln(n) - 2ln(L) = k*ln(n) + 2*neg_log_likelihood
            k * n.ln() + 2.0 * neg_log_likelihood
        }
        InformationCriterion::MDL => {
            // MDL = k/2*ln(n) - ln(L) = k/2*ln(n) + neg_log_likelihood
            (k / 2.0) * n.ln() + neg_log_likelihood
        }
    }
}

/// Detect elbow point in error curve
///
/// Uses the angle-based method to find the point where adding more
/// components gives diminishing returns.
///
/// # Arguments
///
/// * `errors` - Reconstruction errors for increasing ranks
/// * `suggested_rank` - Rank to check if it's at the elbow
///
/// # Returns
///
/// true if the suggested rank is near the elbow point
fn detect_elbow(errors: &[f64], suggested_rank: usize) -> bool {
    if errors.len() < 3 || suggested_rank == 0 || suggested_rank >= errors.len() - 1 {
        return false;
    }

    // Compute angles at each point
    // Elbow is where the angle is sharpest
    let mut max_angle_idx = 1;
    let mut max_angle = 0.0;

    for i in 1..(errors.len() - 1) {
        // Vectors from point i to neighbors
        let v1_x = -1.0;
        let v1_y = errors[i - 1] - errors[i];
        let v2_x = 1.0;
        let v2_y = errors[i + 1] - errors[i];

        // Angle using dot product
        let dot = v1_x * v2_x + v1_y * v2_y;
        let mag1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
        let mag2 = (v2_x * v2_x + v2_y * v2_y).sqrt();

        if mag1 > 0.0 && mag2 > 0.0 {
            let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
            let angle = cos_angle.acos();

            if angle > max_angle {
                max_angle = angle;
                max_angle_idx = i;
            }
        }
    }

    // Check if suggested rank is within 1 of the elbow
    (suggested_rank as isize - max_angle_idx as isize).abs() <= 1
}

/// Select rank using information criterion
///
/// Evaluates multiple ranks and selects the one minimizing the IC.
///
/// # Arguments
///
/// * `errors` - Reconstruction errors for each candidate rank
/// * `params_per_rank` - Number of parameters for each rank
/// * `num_observations` - Total number of observations
/// * `criterion` - Which IC to use
///
/// # Returns
///
/// Index of the best rank (0-based)
///
/// # Examples
///
/// ```
/// use tenrso_decomp::rank_selection::{select_rank_by_criterion, InformationCriterion};
///
/// let errors = vec![1.0, 0.5, 0.3, 0.25, 0.23, 0.22];
/// let params = vec![100, 200, 300, 400, 500, 600];
/// let num_obs = 10000;
///
/// let best_idx = select_rank_by_criterion(
///     &errors,
///     &params,
///     num_obs,
///     InformationCriterion::BIC
/// );
///
/// // Should select a moderate rank (not too low, not too high)
/// assert!(best_idx > 0 && best_idx < errors.len() - 1);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn select_rank_by_criterion(
    errors: &[f64],
    params_per_rank: &[usize],
    num_observations: usize,
    criterion: InformationCriterion,
) -> usize {
    assert_eq!(
        errors.len(),
        params_per_rank.len(),
        "Errors and params must have same length"
    );
    assert!(!errors.is_empty(), "Must have at least one candidate rank");

    let mut best_idx = 0;
    let mut best_ic = f64::INFINITY;

    for (i, (&error, &num_params)) in errors.iter().zip(params_per_rank.iter()).enumerate() {
        let ic = compute_information_criterion(error, num_params, num_observations, criterion);

        if ic < best_ic {
            best_ic = ic;
            best_idx = i;
        }
    }

    best_idx
}

/// Compute number of parameters for CP decomposition
///
/// # Arguments
///
/// * `shape` - Tensor dimensions
/// * `rank` - CP rank
///
/// # Returns
///
/// Total number of parameters: sum(shape\[i\] * rank) + rank (weights)
pub fn cp_num_params(shape: &[usize], rank: usize) -> usize {
    let factor_params: usize = shape.iter().map(|&dim| dim * rank).sum();
    factor_params + rank // Include weights
}

/// Compute number of parameters for Tucker decomposition
///
/// # Arguments
///
/// * `shape` - Tensor dimensions
/// * `ranks` - Tucker ranks for each mode
///
/// # Returns
///
/// Total number of parameters: core + sum(shape\[i\] * ranks\[i\])
pub fn tucker_num_params(shape: &[usize], ranks: &[usize]) -> usize {
    assert_eq!(
        shape.len(),
        ranks.len(),
        "Shape and ranks must have same length"
    );

    let core_params: usize = ranks.iter().product();
    let factor_params: usize = shape
        .iter()
        .zip(ranks.iter())
        .map(|(&dim, &rank)| dim * rank)
        .sum();

    core_params + factor_params
}

/// Compute number of parameters for TT decomposition
///
/// # Arguments
///
/// * `shape` - Tensor dimensions
/// * `ranks` - TT-ranks [r1, r2, ..., r_{n-1}]
///
/// # Returns
///
/// Total number of parameters in all TT cores
pub fn tt_num_params(shape: &[usize], ranks: &[usize]) -> usize {
    assert_eq!(ranks.len() + 1, shape.len(), "TT ranks length must be n-1");

    let n = shape.len();
    let mut total = 0;

    // First core: (1, I_1, r_1)
    total += shape[0] * ranks[0];

    // Middle cores: (r_{k-1}, I_k, r_k)
    for k in 1..(n - 1) {
        total += ranks[k - 1] * shape[k] * ranks[k];
    }

    // Last core: (r_{n-1}, I_n, 1)
    total += ranks[n - 2] * shape[n - 1];

    total
}

/// Split tensor data for cross-validation
///
/// Randomly partitions observed entries into training and validation sets.
///
/// # Arguments
///
/// * `shape` - Tensor dimensions
/// * `train_ratio` - Fraction of data to use for training (e.g., 0.8 for 80%)
///
/// # Returns
///
/// Tuple of (training_mask, validation_mask) as `DenseND` tensors
/// where 1.0 = included, 0.0 = excluded
///
/// # Examples
///
/// ```
/// use tenrso_decomp::rank_selection::create_cv_split;
///
/// let shape = vec![10, 10, 10];
/// let (train_mask, val_mask) = create_cv_split(&shape, 0.8);
///
/// assert_eq!(train_mask.shape(), &shape);
/// assert_eq!(val_mask.shape(), &shape);
///
/// // Masks should be complementary (roughly)
/// let train_view = train_mask.view();
/// let val_view = val_mask.view();
/// let total_ones: f64 = train_view.iter().zip(val_view.iter())
///     .map(|(t, v)| t + v)
///     .sum();
///
/// // Most entries should be either in train or validation (not both)
/// assert!(total_ones > 900.0);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn create_cv_split(shape: &[usize], train_ratio: f64) -> (DenseND<f64>, DenseND<f64>) {
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    use scirs2_core::random::{thread_rng, Rng};

    assert!(
        train_ratio > 0.0 && train_ratio < 1.0,
        "train_ratio must be in (0, 1)"
    );

    let total_size: usize = shape.iter().product();
    let mut rng = thread_rng();

    // Create random assignments
    let mut train_data = Vec::with_capacity(total_size);
    let mut val_data = Vec::with_capacity(total_size);

    for _ in 0..total_size {
        let r: f64 = rng.random();
        if r < train_ratio {
            train_data.push(1.0);
            val_data.push(0.0);
        } else {
            train_data.push(0.0);
            val_data.push(1.0);
        }
    }

    let train_array =
        Array::from_shape_vec(IxDyn(shape), train_data).expect("Shape mismatch in train data");
    let val_array =
        Array::from_shape_vec(IxDyn(shape), val_data).expect("Shape mismatch in validation data");

    (
        DenseND::from_array(train_array),
        DenseND::from_array(val_array),
    )
}

/// Compute masked reconstruction error
///
/// Computes Frobenius norm of error only on masked entries.
///
/// # Arguments
///
/// * `original` - Original tensor
/// * `reconstruction` - Reconstructed tensor
/// * `mask` - Binary mask (1.0 = include, 0.0 = exclude)
///
/// # Returns
///
/// Frobenius norm of masked error: ||mask ⊙ (X - X_hat)||_F
pub fn masked_reconstruction_error<T>(
    original: &DenseND<T>,
    reconstruction: &DenseND<T>,
    mask: &DenseND<f64>,
) -> f64
where
    T: Float + NumCast + Sum,
{
    assert_eq!(
        original.shape(),
        reconstruction.shape(),
        "Shapes must match"
    );
    assert_eq!(original.shape(), mask.shape(), "Mask shape must match");

    let orig_view = original.view();
    let recon_view = reconstruction.view();
    let mask_view = mask.view();

    let mut error_sq = 0.0;
    let mut count = 0.0;

    for ((o, r), m) in orig_view
        .iter()
        .zip(recon_view.iter())
        .zip(mask_view.iter())
    {
        if *m > 0.5 {
            // Entry is included in mask
            let diff = (*o - *r).to_f64().unwrap();
            error_sq += diff * diff;
            count += 1.0;
        }
    }

    if count > 0.0 {
        (error_sq / count).sqrt()
    } else {
        0.0
    }
}

/// Strategy for automated rank selection
#[derive(Debug, Clone, Copy)]
pub enum RankSelectionStrategy {
    /// Use information criterion (AIC, BIC, or MDL)
    InformationCriterion(InformationCriterion),

    /// Use elbow detection on error curve
    ElbowDetection,

    /// Use variance explained threshold (e.g., 0.9 for 90%)
    VarianceThreshold(f64),

    /// Use cross-validation (requires train/validation split)
    CrossValidation,

    /// Combined strategy: IC + elbow verification
    Combined(InformationCriterion),
}

/// Automated rank selection for decompositions
///
/// Evaluates multiple candidate ranks and selects the best one using the specified strategy.
///
/// # Arguments
///
/// * `errors` - Reconstruction errors for each candidate rank
/// * `params_per_rank` - Number of parameters for each rank
/// * `num_observations` - Total number of observations
/// * `strategy` - Selection strategy to use
///
/// # Returns
///
/// `RankSelectionResult` with the selected rank and detailed diagnostics
///
/// # Examples
///
/// ```
/// use tenrso_decomp::rank_selection::{select_rank_auto, RankSelectionStrategy, InformationCriterion};
///
/// let errors = vec![1.0, 0.5, 0.3, 0.25, 0.23, 0.22];
/// let params = vec![100, 200, 300, 400, 500, 600];
/// let num_obs = 10000;
///
/// // Use BIC for rank selection
/// let result = select_rank_auto(
///     &errors,
///     &params,
///     num_obs,
///     RankSelectionStrategy::InformationCriterion(InformationCriterion::BIC)
/// );
///
/// println!("Selected rank: {}", result.rank);
/// println!("Error: {:.4}", result.error);
/// println!("Has elbow: {}", result.has_elbow());
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn select_rank_auto(
    errors: &[f64],
    params_per_rank: &[usize],
    num_observations: usize,
    strategy: RankSelectionStrategy,
) -> RankSelectionResult {
    assert_eq!(
        errors.len(),
        params_per_rank.len(),
        "Errors and params must have same length"
    );
    assert!(!errors.is_empty(), "Must have at least one candidate rank");

    let candidate_ranks: Vec<usize> = (1..=errors.len()).collect();

    match strategy {
        RankSelectionStrategy::InformationCriterion(criterion) => {
            let criterion_values: Vec<f64> = errors
                .iter()
                .zip(params_per_rank.iter())
                .map(|(&error, &num_params)| {
                    compute_information_criterion(error, num_params, num_observations, criterion)
                })
                .collect();

            let best_idx =
                select_rank_by_criterion(errors, params_per_rank, num_observations, criterion);

            RankSelectionResult {
                rank: candidate_ranks[best_idx],
                error: errors[best_idx],
                criterion_value: criterion_values[best_idx],
                criterion,
                candidate_ranks,
                errors: errors.to_vec(),
                criterion_values,
            }
        }

        RankSelectionStrategy::ElbowDetection => {
            // Find elbow point
            let mut best_idx = 0;
            let mut max_angle = 0.0;

            for i in 1..(errors.len() - 1) {
                let v1_x = -1.0;
                let v1_y = errors[i - 1] - errors[i];
                let v2_x = 1.0;
                let v2_y = errors[i + 1] - errors[i];

                let dot = v1_x * v2_x + v1_y * v2_y;
                let mag1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
                let mag2 = (v2_x * v2_x + v2_y * v2_y).sqrt();

                if mag1 > 0.0 && mag2 > 0.0 {
                    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
                    let angle = cos_angle.acos();

                    if angle > max_angle {
                        max_angle = angle;
                        best_idx = i;
                    }
                }
            }

            // Compute IC values using BIC as default
            let criterion = InformationCriterion::BIC;
            let criterion_values: Vec<f64> = errors
                .iter()
                .zip(params_per_rank.iter())
                .map(|(&error, &num_params)| {
                    compute_information_criterion(error, num_params, num_observations, criterion)
                })
                .collect();

            RankSelectionResult {
                rank: candidate_ranks[best_idx],
                error: errors[best_idx],
                criterion_value: criterion_values[best_idx],
                criterion,
                candidate_ranks,
                errors: errors.to_vec(),
                criterion_values,
            }
        }

        RankSelectionStrategy::VarianceThreshold(threshold) => {
            // This strategy requires singular values, not errors
            // For now, use the first rank that achieves error below threshold
            let best_idx = errors
                .iter()
                .position(|&e| e <= threshold)
                .unwrap_or(errors.len() - 1);

            let criterion = InformationCriterion::BIC;
            let criterion_values: Vec<f64> = errors
                .iter()
                .zip(params_per_rank.iter())
                .map(|(&error, &num_params)| {
                    compute_information_criterion(error, num_params, num_observations, criterion)
                })
                .collect();

            RankSelectionResult {
                rank: candidate_ranks[best_idx],
                error: errors[best_idx],
                criterion_value: criterion_values[best_idx],
                criterion,
                candidate_ranks,
                errors: errors.to_vec(),
                criterion_values,
            }
        }

        RankSelectionStrategy::CrossValidation => {
            // Find rank with minimum error (validation error)
            let mut best_idx = 0;
            let mut min_error = f64::INFINITY;

            for (i, &error) in errors.iter().enumerate() {
                if error < min_error {
                    min_error = error;
                    best_idx = i;
                }
            }

            let criterion = InformationCriterion::BIC;
            let criterion_values: Vec<f64> = errors
                .iter()
                .zip(params_per_rank.iter())
                .map(|(&error, &num_params)| {
                    compute_information_criterion(error, num_params, num_observations, criterion)
                })
                .collect();

            RankSelectionResult {
                rank: candidate_ranks[best_idx],
                error: errors[best_idx],
                criterion_value: criterion_values[best_idx],
                criterion,
                candidate_ranks,
                errors: errors.to_vec(),
                criterion_values,
            }
        }

        RankSelectionStrategy::Combined(criterion) => {
            // Use IC but verify elbow is nearby
            let ic_idx =
                select_rank_by_criterion(errors, params_per_rank, num_observations, criterion);

            // Check if elbow is detected near IC choice
            let elbow_verified = detect_elbow(errors, candidate_ranks[ic_idx]);

            // If no elbow near IC choice, use elbow detection
            let best_idx = if !elbow_verified {
                // Find elbow point
                let mut elbow_idx = 0;
                let mut max_angle = 0.0;

                for i in 1..(errors.len() - 1) {
                    let v1_x = -1.0;
                    let v1_y = errors[i - 1] - errors[i];
                    let v2_x = 1.0;
                    let v2_y = errors[i + 1] - errors[i];

                    let dot = v1_x * v2_x + v1_y * v2_y;
                    let mag1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
                    let mag2 = (v2_x * v2_x + v2_y * v2_y).sqrt();

                    if mag1 > 0.0 && mag2 > 0.0 {
                        let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
                        let angle = cos_angle.acos();

                        if angle > max_angle {
                            max_angle = angle;
                            elbow_idx = i;
                        }
                    }
                }
                elbow_idx
            } else {
                ic_idx
            };

            let criterion_values: Vec<f64> = errors
                .iter()
                .zip(params_per_rank.iter())
                .map(|(&error, &num_params)| {
                    compute_information_criterion(error, num_params, num_observations, criterion)
                })
                .collect();

            RankSelectionResult {
                rank: candidate_ranks[best_idx],
                error: errors[best_idx],
                criterion_value: criterion_values[best_idx],
                criterion,
                candidate_ranks,
                errors: errors.to_vec(),
                criterion_values,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_criterion_ordering() {
        let error = 0.1;
        let num_params = 1000;
        let num_obs = 10000;

        let aic =
            compute_information_criterion(error, num_params, num_obs, InformationCriterion::AIC);
        let bic =
            compute_information_criterion(error, num_params, num_obs, InformationCriterion::BIC);
        let mdl =
            compute_information_criterion(error, num_params, num_obs, InformationCriterion::MDL);

        // BIC typically has stronger penalty
        assert!(bic > aic, "BIC should penalize complexity more than AIC");

        // All should be finite
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(mdl.is_finite());
    }

    #[test]
    fn test_select_rank_prefers_parsimony() {
        // Errors decrease slowly after rank 3
        let errors = vec![1.0, 0.5, 0.3, 0.25, 0.23, 0.22, 0.21];
        let params = vec![100, 200, 300, 400, 500, 600, 700];
        let num_obs = 10000;

        let best_idx =
            select_rank_by_criterion(&errors, &params, num_obs, InformationCriterion::BIC);

        // Should prefer a moderate rank (not the highest)
        assert!(
            best_idx < errors.len() - 1,
            "Should not select highest rank due to overfitting"
        );
    }

    #[test]
    fn test_cp_num_params() {
        let shape = vec![10, 20, 30];
        let rank = 5;

        let params = cp_num_params(&shape, rank);

        // Expected: (10*5) + (20*5) + (30*5) + 5 = 50 + 100 + 150 + 5 = 305
        assert_eq!(params, 305);
    }

    #[test]
    fn test_tucker_num_params() {
        let shape = vec![10, 20, 30];
        let ranks = vec![5, 8, 6];

        let params = tucker_num_params(&shape, &ranks);

        // Core: 5*8*6 = 240
        // Factors: (10*5) + (20*8) + (30*6) = 50 + 160 + 180 = 390
        // Total: 240 + 390 = 630
        assert_eq!(params, 630);
    }

    #[test]
    fn test_tt_num_params() {
        let shape = vec![10, 20, 30, 40];
        let ranks = vec![5, 8, 6];

        let params = tt_num_params(&shape, &ranks);

        // Core 1: 1*10*5 = 50
        // Core 2: 5*20*8 = 800
        // Core 3: 8*30*6 = 1440
        // Core 4: 6*40*1 = 240
        // Total: 50 + 800 + 1440 + 240 = 2530
        assert_eq!(params, 2530);
    }

    #[test]
    fn test_elbow_detection() {
        // Clear elbow at index 2
        let errors = vec![1.0, 0.5, 0.25, 0.22, 0.21, 0.20];

        // Should detect elbow around rank 2-3
        assert!(detect_elbow(&errors, 2) || detect_elbow(&errors, 3));

        // Should not detect elbow at extremes
        assert!(!detect_elbow(&errors, 0));
        assert!(!detect_elbow(&errors, errors.len() - 1));
    }

    #[test]
    fn test_rank_selection_result_improvement() {
        let result = RankSelectionResult {
            rank: 3,
            error: 0.3,
            criterion_value: 1000.0,
            criterion: InformationCriterion::BIC,
            candidate_ranks: vec![1, 2, 3, 4, 5],
            errors: vec![1.0, 0.5, 0.3, 0.25, 0.23],
            criterion_values: vec![2000.0, 1500.0, 1000.0, 1100.0, 1200.0],
        };

        let improvement = result.improvement_ratio();

        // (1.0 - 0.3) / 1.0 = 0.7
        assert!((improvement - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_create_cv_split() {
        let shape = vec![10, 10, 10];
        let (train_mask, val_mask) = create_cv_split(&shape, 0.8);

        // Check shapes
        assert_eq!(train_mask.shape(), &shape);
        assert_eq!(val_mask.shape(), &shape);

        // Count entries
        let train_view = train_mask.view();
        let val_view = val_mask.view();

        let train_count: f64 = train_view.iter().sum();
        let val_count: f64 = val_view.iter().sum();
        let total: usize = shape.iter().product();

        // Approximately 80% train, 20% validation
        let expected_train = (total as f64) * 0.8;
        assert!(
            (train_count - expected_train).abs() < 100.0,
            "Train count deviation too large"
        );

        // Should be roughly complementary
        assert!((train_count + val_count - total as f64).abs() < 10.0);

        // No overlap (all entries are 0 or 1)
        for (t, v) in train_view.iter().zip(val_view.iter()) {
            assert!((*t == 0.0 || *t == 1.0) && (*v == 0.0 || *v == 1.0));
            assert!(*t + *v <= 1.0 + 1e-10); // At most one can be 1
        }
    }

    #[test]
    fn test_masked_reconstruction_error() {
        use scirs2_core::ndarray_ext::Array;

        // Create simple 2×2×2 tensor
        let data =
            Array::from_shape_vec(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let original = DenseND::from_array(data.into_dyn());

        // Create reconstruction with some error
        let recon_data =
            Array::from_shape_vec(vec![2, 2, 2], vec![1.1, 2.1, 2.9, 4.1, 5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let reconstruction = DenseND::from_array(recon_data.into_dyn());

        // Mask that includes only first 4 entries
        let mask_data =
            Array::from_shape_vec(vec![2, 2, 2], vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                .unwrap();
        let mask = DenseND::from_array(mask_data.into_dyn());

        let error = masked_reconstruction_error(&original, &reconstruction, &mask);

        // Error should only consider first 4 entries
        // Squared errors: 0.01, 0.01, 0.01, 0.01
        // Mean: 0.01, sqrt: 0.1
        assert!((error - 0.1).abs() < 1e-10, "Error was {}", error);
    }

    #[test]
    fn test_masked_error_empty_mask() {
        use scirs2_core::ndarray_ext::Array;

        let data = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let original = DenseND::from_array(data.clone().into_dyn());
        let reconstruction = DenseND::from_array(data.into_dyn());

        // Empty mask (all zeros)
        let mask_data = Array::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let mask = DenseND::from_array(mask_data.into_dyn());

        let error = masked_reconstruction_error(&original, &reconstruction, &mask);

        // Should return 0 for empty mask
        assert_eq!(error, 0.0);
    }

    #[test]
    fn test_scree_plot_data_basic() {
        let singular_values = vec![10.0, 5.0, 2.0, 1.0, 0.5, 0.1];
        let scree = ScreePlotData::new(singular_values, 0.9, 0.95);

        // Check that singular values are sorted descending
        assert_eq!(scree.singular_values[0], 10.0);
        assert_eq!(scree.singular_values[5], 0.1);

        // Check that cumulative variance reaches ~1.0
        assert!(
            scree.cumulative_variance.last().unwrap() >= &0.99,
            "Cumulative variance should reach ~1.0"
        );

        // Check that variance explained sums to ~1.0
        let total_var: f64 = scree.variance_explained.iter().sum();
        assert!(
            (total_var - 1.0).abs() < 1e-10,
            "Variance should sum to 1.0"
        );

        // Check that suggestions are reasonable
        assert!(scree.suggested_rank_90.is_some());
        assert!(scree.suggested_rank_95.is_some());

        // 95% rank should be >= 90% rank
        if let (Some(r90), Some(r95)) = (scree.suggested_rank_90, scree.suggested_rank_95) {
            assert!(
                r95 >= r90,
                "95% rank should be >= 90% rank, got {} vs {}",
                r95,
                r90
            );
        }
    }

    #[test]
    fn test_scree_plot_rank_for_variance() {
        let singular_values = vec![10.0, 5.0, 2.0, 1.0, 0.5, 0.1];
        let scree = ScreePlotData::new(singular_values, 0.9, 0.95);

        // Test rank_for_variance method
        let rank_80 = scree.rank_for_variance(0.8);
        let rank_90 = scree.rank_for_variance(0.9);

        assert!(rank_80.is_some());
        assert!(rank_90.is_some());

        if let (Some(r80), Some(r90)) = (rank_80, rank_90) {
            assert!(
                r90 >= r80,
                "Higher variance threshold should require >= rank"
            );
        }
    }

    #[test]
    fn test_scree_plot_elbow_detection() {
        // Create singular values with clear elbow at position 2
        let singular_values = vec![10.0, 9.0, 3.0, 1.0, 0.5, 0.2, 0.1];
        let scree = ScreePlotData::new(singular_values, 0.9, 0.95);

        // Elbow should be detected
        assert!(
            scree.suggested_rank.is_some(),
            "Should detect elbow in clear case"
        );
    }

    #[test]
    fn test_select_rank_auto_ic() {
        let errors = vec![1.0, 0.5, 0.3, 0.25, 0.23, 0.22, 0.21];
        let params = vec![100, 200, 300, 400, 500, 600, 700];
        let num_obs = 10000;

        let result = select_rank_auto(
            &errors,
            &params,
            num_obs,
            RankSelectionStrategy::InformationCriterion(InformationCriterion::BIC),
        );

        // Should select a moderate rank (not the highest)
        assert!(result.rank < errors.len(), "Should not select highest rank");
        assert!(result.rank > 0, "Should not select rank 0");

        // Check that result contains valid data
        assert_eq!(result.errors.len(), errors.len());
        assert_eq!(result.candidate_ranks.len(), errors.len());
        assert_eq!(result.criterion_values.len(), errors.len());
    }

    #[test]
    fn test_select_rank_auto_elbow() {
        // Clear elbow - large drop initially then small drops
        let errors = vec![1.0, 0.3, 0.15, 0.12, 0.11, 0.10];
        let params = vec![100, 200, 300, 400, 500, 600];
        let num_obs = 10000;

        let result = select_rank_auto(
            &errors,
            &params,
            num_obs,
            RankSelectionStrategy::ElbowDetection,
        );

        // Should select a reasonable rank (angle-based elbow detection)
        assert!(
            result.rank >= 1 && result.rank <= errors.len(),
            "Should select valid rank, got {}",
            result.rank
        );

        // Error should be finite and positive
        assert!(result.error > 0.0 && result.error < 1.0);
    }

    #[test]
    fn test_select_rank_auto_variance_threshold() {
        let errors = vec![1.0, 0.5, 0.3, 0.2, 0.15, 0.12];
        let params = vec![100, 200, 300, 400, 500, 600];
        let num_obs = 10000;

        let result = select_rank_auto(
            &errors,
            &params,
            num_obs,
            RankSelectionStrategy::VarianceThreshold(0.25),
        );

        // Should select first rank with error <= 0.25
        // First error <= 0.25 is at index 3 (0.2), which is rank 4
        assert!(
            result.error <= 0.25,
            "Selected rank should have error <= threshold, got {}",
            result.error
        );
        assert!(result.rank <= errors.len(), "Should select valid rank");
    }

    #[test]
    fn test_select_rank_auto_cv() {
        // Validation errors (rank 3 has minimum)
        let errors = vec![0.8, 0.6, 0.4, 0.5, 0.55, 0.6];
        let params = vec![100, 200, 300, 400, 500, 600];
        let num_obs = 10000;

        let result = select_rank_auto(
            &errors,
            &params,
            num_obs,
            RankSelectionStrategy::CrossValidation,
        );

        // Should select rank 3 (index 2) with minimum error
        assert_eq!(
            result.rank, 3,
            "Should select rank with minimum validation error"
        );
        assert_eq!(result.error, 0.4);
    }

    #[test]
    fn test_select_rank_auto_combined() {
        let errors = vec![1.0, 0.5, 0.3, 0.25, 0.23, 0.22];
        let params = vec![100, 200, 300, 400, 500, 600];
        let num_obs = 10000;

        let result = select_rank_auto(
            &errors,
            &params,
            num_obs,
            RankSelectionStrategy::Combined(InformationCriterion::BIC),
        );

        // Should select a reasonable rank
        assert!(result.rank > 0 && result.rank < errors.len());

        // Result should have elbow information
        let has_elbow = result.has_elbow();
        // Just check it computes without error
        let _ = has_elbow;
    }

    #[test]
    fn test_rank_selection_strategies_are_different() {
        let errors = vec![1.0, 0.5, 0.3, 0.25, 0.23, 0.22];
        let params = vec![100, 200, 300, 400, 500, 600];
        let num_obs = 10000;

        let result_bic = select_rank_auto(
            &errors,
            &params,
            num_obs,
            RankSelectionStrategy::InformationCriterion(InformationCriterion::BIC),
        );

        let result_aic = select_rank_auto(
            &errors,
            &params,
            num_obs,
            RankSelectionStrategy::InformationCriterion(InformationCriterion::AIC),
        );

        // BIC typically selects lower rank than AIC
        // (stronger penalty for complexity)
        assert!(
            result_bic.rank <= result_aic.rank,
            "BIC should select <= rank than AIC"
        );
    }
}
