//! Mixed precision training support for automatic differentiation.
//!
//! This module provides utilities for training with reduced precision (fp16/bf16)
//! while maintaining numerical stability through loss scaling and gradient overflow detection.
//!
//! # Features
//!
//! - **Loss Scaling**: Automatic and dynamic loss scaling to prevent gradient underflow
//! - **Overflow Detection**: Detect and handle gradient overflow/underflow
//! - **Precision Conversion**: Safe conversion between fp32, fp16, and bf16
//! - **Mixed Precision State**: Track scaling factor and iteration counts
//!
//! # Examples
//!
//! ```rust,ignore
//! use tenrso_ad::mixed_precision::{MixedPrecisionConfig, GradScaler};
//! use scirs2_core::ndarray_ext::Array1;
//!
//! // Create a gradient scaler with automatic loss scaling
//! let config = MixedPrecisionConfig::default();
//! let mut scaler = GradScaler::new(config);
//!
//! // Scale loss before backward pass
//! let scaled_loss = scaler.scale(loss);
//!
//! // After computing gradients, check for overflow and update scale
//! let gradients = vec![grad1, grad2, grad3];
//! if scaler.check_overflow(&gradients) {
//!     // Skip optimizer step, scale will be reduced
//!     scaler.update(true);
//! } else {
//!     // Unscale gradients and perform optimizer step
//!     let unscaled = scaler.unscale(&gradients);
//!     // ... optimizer step ...
//!     scaler.update(false);
//! }
//! ```

use anyhow::Result;
use num_traits::{Float, NumCast};
use scirs2_core::ndarray_ext::{Array, IxDyn, ScalarOperand};
use std::marker::PhantomData;

/// Configuration for mixed precision training.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Initial loss scale factor (default: 2^16 = 65536.0)
    pub init_scale: f32,

    /// Growth factor for increasing scale (default: 2.0)
    pub growth_factor: f32,

    /// Backoff factor for decreasing scale (default: 0.5)
    pub backoff_factor: f32,

    /// Number of iterations without overflow before growing scale (default: 2000)
    pub growth_interval: usize,

    /// Enable dynamic loss scaling (default: true)
    pub dynamic: bool,

    /// Minimum scale factor (default: 1.0)
    pub min_scale: f32,

    /// Maximum scale factor (default: 2^24 = 16777216.0)
    pub max_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            dynamic: true,
            min_scale: 1.0,
            max_scale: 16777216.0, // 2^24
        }
    }
}

impl MixedPrecisionConfig {
    /// Create configuration for static (non-dynamic) loss scaling.
    pub fn static_scale(scale: f32) -> Self {
        Self {
            init_scale: scale,
            dynamic: false,
            ..Default::default()
        }
    }

    /// Create configuration optimized for fp16 precision.
    pub fn fp16_optimized() -> Self {
        Self {
            init_scale: 65536.0,
            growth_interval: 2000,
            ..Default::default()
        }
    }

    /// Create configuration optimized for bf16 precision.
    /// BF16 has larger dynamic range than fp16, so lower initial scale.
    pub fn bf16_optimized() -> Self {
        Self {
            init_scale: 256.0, // 2^8, bf16 has better dynamic range
            growth_interval: 1000,
            ..Default::default()
        }
    }
}

/// Gradient scaler for mixed precision training.
///
/// Manages loss scaling to prevent gradient underflow in reduced precision training.
/// Automatically adjusts scale factor based on overflow detection.
#[derive(Debug, Clone)]
pub struct GradScaler {
    config: MixedPrecisionConfig,
    scale: f32,
    growth_tracker: usize,
    _phantom: PhantomData<f32>,
}

impl GradScaler {
    /// Create a new gradient scaler with the given configuration.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let scale = config.init_scale;
        Self {
            config,
            scale,
            growth_tracker: 0,
            _phantom: PhantomData,
        }
    }

    /// Get the current scale factor.
    pub fn get_scale(&self) -> f32 {
        self.scale
    }

    /// Scale a loss value before backward pass.
    ///
    /// # Arguments
    /// * `loss` - The loss value to scale
    ///
    /// # Returns
    /// Scaled loss value
    pub fn scale<T: Float>(&self, loss: T) -> T {
        let scale_t = T::from(self.scale).unwrap_or_else(|| T::one());
        loss * scale_t
    }

    /// Unscale gradients after backward pass.
    ///
    /// # Arguments
    /// * `gradients` - Gradients to unscale (in-place)
    ///
    /// # Returns
    /// Unscaled gradients
    pub fn unscale<T>(&self, gradients: &[Array<T, IxDyn>]) -> Vec<Array<T, IxDyn>>
    where
        T: Float + ScalarOperand + 'static,
    {
        let inv_scale = T::from(1.0 / self.scale as f64).unwrap_or_else(|| T::one());
        gradients.iter().map(|grad| grad * inv_scale).collect()
    }

    /// Check if any gradients contain overflow (inf/nan).
    ///
    /// # Arguments
    /// * `gradients` - Gradients to check for overflow
    ///
    /// # Returns
    /// `true` if overflow detected, `false` otherwise
    pub fn check_overflow<T>(&self, gradients: &[Array<T, IxDyn>]) -> bool
    where
        T: Float,
    {
        gradients
            .iter()
            .any(|grad| grad.iter().any(|&v| !v.is_finite()))
    }

    /// Update the scale factor based on overflow status.
    ///
    /// # Arguments
    /// * `overflow` - Whether overflow was detected in current iteration
    pub fn update(&mut self, overflow: bool) {
        if !self.config.dynamic {
            return;
        }

        if overflow {
            // Reduce scale on overflow
            self.scale = (self.scale * self.config.backoff_factor).max(self.config.min_scale);
            self.growth_tracker = 0;
        } else {
            // Increment growth tracker
            self.growth_tracker += 1;

            // Increase scale after growth_interval successful iterations
            if self.growth_tracker >= self.config.growth_interval {
                self.scale = (self.scale * self.config.growth_factor).min(self.config.max_scale);
                self.growth_tracker = 0;
            }
        }
    }

    /// Reset the scaler to initial state.
    pub fn reset(&mut self) {
        self.scale = self.config.init_scale;
        self.growth_tracker = 0;
    }

    /// Get statistics about the current scaling state.
    pub fn stats(&self) -> ScalerStats {
        ScalerStats {
            current_scale: self.scale,
            growth_tracker: self.growth_tracker,
            growth_interval: self.config.growth_interval,
        }
    }
}

/// Statistics about gradient scaler state.
#[derive(Debug, Clone, Copy)]
pub struct ScalerStats {
    /// Current scale factor
    pub current_scale: f32,

    /// Iterations since last overflow
    pub growth_tracker: usize,

    /// Iterations needed before scale growth
    pub growth_interval: usize,
}

impl ScalerStats {
    /// Get percentage progress towards scale growth.
    pub fn growth_progress(&self) -> f32 {
        (self.growth_tracker as f32 / self.growth_interval as f32) * 100.0
    }
}

/// Precision type for mixed precision operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionType {
    /// Full precision (fp32)
    FP32,
    /// Half precision (fp16) - IEEE 754 half precision
    FP16,
    /// Brain floating point (bf16) - Google's bfloat16
    BF16,
}

/// Convert tensor to lower precision with optional rounding.
///
/// # Arguments
/// * `input` - Input tensor in fp32
/// * `precision` - Target precision type
///
/// # Returns
/// Tensor converted to specified precision (stored as fp32)
pub fn convert_to_lower_precision<T>(
    input: &Array<T, IxDyn>,
    precision: PrecisionType,
) -> Result<Array<T, IxDyn>>
where
    T: Float + NumCast + 'static,
{
    match precision {
        PrecisionType::FP32 => Ok(input.clone()),
        PrecisionType::FP16 => convert_to_fp16(input),
        PrecisionType::BF16 => convert_to_bf16(input),
    }
}

/// Convert tensor to fp16 precision (simulated in fp32).
///
/// FP16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
/// Range: ~6e-8 to ~65504
fn convert_to_fp16<T>(input: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>>
where
    T: Float + NumCast + 'static,
{
    const FP16_MAX: f32 = 65504.0;
    const FP16_MIN: f32 = 6.0e-8;

    let output = input.mapv(|v| {
        let v_f32 = v.to_f32().unwrap_or(0.0);

        // Clamp to fp16 range
        let clamped = if v_f32.abs() > FP16_MAX {
            v_f32.signum() * FP16_MAX
        } else if v_f32.abs() < FP16_MIN && v_f32 != 0.0 {
            0.0
        } else {
            v_f32
        };

        // Round to fp16 precision (10 mantissa bits)
        // This is a simplified simulation
        let rounded = (clamped * 1024.0).round() / 1024.0;

        T::from(rounded).unwrap_or_else(T::zero)
    });

    Ok(output)
}

/// Convert tensor to bf16 precision (simulated in fp32).
///
/// BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
/// Range: ~1e-38 to ~3.4e38 (same as fp32)
fn convert_to_bf16<T>(input: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>>
where
    T: Float + NumCast + 'static,
{
    let output = input.mapv(|v| {
        let v_f32 = v.to_f32().unwrap_or(0.0);

        // Round to bf16 precision (7 mantissa bits)
        // Truncate mantissa bits 8-23 (keep sign, exponent, and top 7 mantissa bits)
        let bits = v_f32.to_bits();
        let truncated_bits = bits & 0xFFFF_0000;
        let rounded = f32::from_bits(truncated_bits);

        T::from(rounded).unwrap_or_else(T::zero)
    });

    Ok(output)
}

/// Detect gradient underflow risk based on gradient magnitudes.
///
/// # Arguments
/// * `gradients` - Gradients to analyze
/// * `precision` - Target precision type
///
/// # Returns
/// `true` if gradients are at risk of underflow
pub fn detect_underflow_risk<T>(gradients: &[Array<T, IxDyn>], precision: PrecisionType) -> bool
where
    T: Float,
{
    let threshold = match precision {
        PrecisionType::FP32 => 1.0e-38,
        PrecisionType::FP16 => 6.0e-8,
        PrecisionType::BF16 => 1.0e-38,
    };

    // Check if significant portion of gradients are below threshold
    let total_elements: usize = gradients.iter().map(|g| g.len()).sum();
    let underflow_count: usize = gradients
        .iter()
        .map(|grad| {
            grad.iter()
                .filter(|&&v| v.abs() < T::from(threshold).unwrap_or_else(T::zero) && !v.is_zero())
                .count()
        })
        .sum();

    // If more than 10% of non-zero gradients are near underflow, flag it
    let underflow_ratio = underflow_count as f32 / total_elements as f32;
    underflow_ratio > 0.1
}

/// Analyze gradient distribution for mixed precision compatibility.
#[derive(Debug, Clone)]
pub struct GradientAnalysis {
    /// Percentage of gradients that would underflow
    pub underflow_percentage: f32,

    /// Percentage of gradients that would overflow
    pub overflow_percentage: f32,

    /// Recommended precision type
    pub recommended_precision: PrecisionType,

    /// Recommended loss scale factor
    pub recommended_scale: f32,
}

/// Analyze gradients for mixed precision training compatibility.
///
/// # Arguments
/// * `gradients` - Gradients to analyze
/// * `precision` - Target precision type
///
/// # Returns
/// Analysis results with recommendations
pub fn analyze_gradients<T>(
    gradients: &[Array<T, IxDyn>],
    precision: PrecisionType,
) -> Result<GradientAnalysis>
where
    T: Float,
{
    let (underflow_threshold, overflow_threshold) = match precision {
        PrecisionType::FP32 => (1.0e-38, 3.4e38),
        PrecisionType::FP16 => (6.0e-8, 65504.0),
        PrecisionType::BF16 => (1.0e-38, 3.4e38),
    };

    let total_elements: usize = gradients.iter().map(|g| g.len()).sum();

    let underflow_count: usize = gradients
        .iter()
        .map(|grad| {
            grad.iter()
                .filter(|&&v| {
                    let abs_v = v.abs();
                    !v.is_zero() && abs_v < T::from(underflow_threshold).unwrap_or_else(T::zero)
                })
                .count()
        })
        .sum();

    let overflow_count: usize = gradients
        .iter()
        .map(|grad| {
            grad.iter()
                .filter(|&&v| {
                    let abs_v = v.abs();
                    abs_v > T::from(overflow_threshold).unwrap_or_else(T::max_value)
                })
                .count()
        })
        .sum();

    let underflow_percentage = (underflow_count as f32 / total_elements as f32) * 100.0;
    let overflow_percentage = (overflow_count as f32 / total_elements as f32) * 100.0;

    // Determine recommended precision
    let recommended_precision = if overflow_percentage > 1.0 || underflow_percentage > 10.0 {
        PrecisionType::FP32 // Too risky for reduced precision
    } else if underflow_percentage > 5.0 {
        PrecisionType::BF16 // Better dynamic range than fp16
    } else {
        precision // Original target is fine
    };

    // Calculate recommended scale
    let max_abs: f32 = gradients
        .iter()
        .flat_map(|grad| grad.iter())
        .map(|&v| v.to_f32().unwrap_or(0.0).abs())
        .fold(0.0f32, f32::max);

    let recommended_scale = if max_abs > 0.0 {
        let scale = overflow_threshold as f32 / (max_abs * 10.0); // Safety factor of 10
        scale.clamp(1.0, 65536.0)
    } else {
        65536.0
    };

    Ok(GradientAnalysis {
        underflow_percentage,
        overflow_percentage,
        recommended_precision,
        recommended_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.init_scale, 65536.0);
        assert_eq!(config.growth_factor, 2.0);
        assert_eq!(config.backoff_factor, 0.5);
        assert_eq!(config.growth_interval, 2000);
        assert!(config.dynamic);
    }

    #[test]
    fn test_grad_scaler_creation() {
        let config = MixedPrecisionConfig::default();
        let scaler = GradScaler::new(config.clone());
        assert_eq!(scaler.get_scale(), config.init_scale);
    }

    #[test]
    fn test_grad_scaler_scale_loss() {
        let scaler = GradScaler::new(MixedPrecisionConfig::default());
        let loss = 0.5f32;
        let scaled = scaler.scale(loss);
        assert_eq!(scaled, loss * 65536.0);
    }

    #[test]
    fn test_grad_scaler_unscale() {
        let scaler = GradScaler::new(MixedPrecisionConfig::default());
        let grad = Array::from_vec(vec![65536.0f32, 131072.0]).into_dyn();
        let unscaled = scaler.unscale(&[grad]);

        assert_eq!(unscaled.len(), 1);
        assert!((unscaled[0][[0]] - 1.0).abs() < 1e-5);
        assert!((unscaled[0][[1]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_grad_scaler_overflow_detection() {
        let scaler = GradScaler::new(MixedPrecisionConfig::default());

        // No overflow
        let grad1 = Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        assert!(!scaler.check_overflow(&[grad1]));

        // With overflow (inf)
        let grad2 = Array::from_vec(vec![1.0f32, f32::INFINITY, 3.0]).into_dyn();
        assert!(scaler.check_overflow(&[grad2]));

        // With overflow (nan)
        let grad3 = Array::from_vec(vec![1.0f32, f32::NAN, 3.0]).into_dyn();
        assert!(scaler.check_overflow(&[grad3]));
    }

    #[test]
    fn test_grad_scaler_update_overflow() {
        let mut scaler = GradScaler::new(MixedPrecisionConfig::default());
        let initial_scale = scaler.get_scale();

        // Update with overflow should reduce scale
        scaler.update(true);
        assert!(scaler.get_scale() < initial_scale);
        assert_eq!(scaler.get_scale(), initial_scale * 0.5);
    }

    #[test]
    fn test_grad_scaler_update_no_overflow() {
        let mut scaler = GradScaler::new(MixedPrecisionConfig {
            growth_interval: 5,
            ..Default::default()
        });
        let initial_scale = scaler.get_scale();

        // Update without overflow should increment growth tracker
        for i in 0..4 {
            scaler.update(false);
            assert_eq!(scaler.get_scale(), initial_scale);
            assert_eq!(scaler.growth_tracker, i + 1);
        }

        // After growth_interval iterations, scale should increase
        scaler.update(false);
        assert!(scaler.get_scale() > initial_scale);
        assert_eq!(scaler.get_scale(), initial_scale * 2.0);
    }

    #[test]
    fn test_grad_scaler_static_mode() {
        let mut scaler = GradScaler::new(MixedPrecisionConfig::static_scale(1000.0));
        let initial_scale = scaler.get_scale();

        // Static mode should not change scale
        scaler.update(false);
        assert_eq!(scaler.get_scale(), initial_scale);

        scaler.update(true);
        assert_eq!(scaler.get_scale(), initial_scale);
    }

    #[test]
    fn test_convert_to_fp16() {
        let input = Array::from_vec(vec![
            1.0f32, 100.0, 10000.0, 70000.0, // Large values
            0.0001, 0.000001, 0.0, // Small values
        ])
        .into_dyn();

        let output = convert_to_fp16(&input).unwrap();

        // FP16 max is ~65504, so 70000 should be clamped
        assert!(output[[3]] <= 65504.0);

        // Very small values should become zero
        assert_eq!(output[[5]], 0.0);

        // Zero should remain zero
        assert_eq!(output[[6]], 0.0);
    }

    #[test]
    fn test_convert_to_bf16() {
        let input = Array::from_vec(vec![1.234_567_9_f32, 100.123_46, 0.000_123_456]).into_dyn();

        let output = convert_to_bf16(&input).unwrap();

        // BF16 has 7 mantissa bits (truncates lower 16 bits of fp32)
        // Values will be less precise than fp32
        // The bf16 format preserves the exponent range but reduces mantissa precision

        // Check that values are in reasonable range (not exact due to precision loss)
        assert!(
            output[[0]] > 0.0 && output[[0]] < 2.0,
            "output[0] = {}",
            output[[0]]
        );
        assert!(output[[1]] > 0.0, "output[1] = {}", output[[1]]);
        assert!(
            output[[2]] >= 0.0 && output[[2]] < 0.01,
            "output[2] = {}",
            output[[2]]
        );
    }

    #[test]
    fn test_detect_underflow_risk() {
        // FP16 with very small gradients
        let small_grads = vec![
            Array::from_vec(vec![1e-9f32, 2e-9, 3e-9]).into_dyn(),
            Array::from_vec(vec![1e-10f32, 2e-10]).into_dyn(),
        ];
        assert!(detect_underflow_risk(&small_grads, PrecisionType::FP16));

        // FP32 with reasonable gradients
        let normal_grads = vec![
            Array::from_vec(vec![1e-5f32, 2e-5, 3e-5]).into_dyn(),
            Array::from_vec(vec![1e-4f32, 2e-4]).into_dyn(),
        ];
        assert!(!detect_underflow_risk(&normal_grads, PrecisionType::FP32));
    }

    #[test]
    fn test_analyze_gradients() {
        let gradients = vec![
            Array::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn(),
            Array::from_vec(vec![0.1f32, 0.2]).into_dyn(),
        ];

        let analysis = analyze_gradients(&gradients, PrecisionType::FP16).unwrap();

        // Normal gradients should have low underflow/overflow
        assert!(analysis.underflow_percentage < 10.0);
        assert!(analysis.overflow_percentage < 1.0);
        assert_eq!(analysis.recommended_precision, PrecisionType::FP16);
    }

    #[test]
    fn test_scaler_stats() {
        let mut scaler = GradScaler::new(MixedPrecisionConfig {
            growth_interval: 100,
            ..Default::default()
        });

        // Update a few times
        for _ in 0..25 {
            scaler.update(false);
        }

        let stats = scaler.stats();
        assert_eq!(stats.current_scale, 65536.0);
        assert_eq!(stats.growth_tracker, 25);
        assert_eq!(stats.growth_interval, 100);
        assert_eq!(stats.growth_progress(), 25.0);
    }

    #[test]
    fn test_scaler_reset() {
        let mut scaler = GradScaler::new(MixedPrecisionConfig::default());

        // Make some changes
        scaler.update(true); // Reduce scale
        scaler.update(false);
        scaler.update(false);

        let scale_before = scaler.get_scale();
        let tracker_before = scaler.growth_tracker;

        // Reset should restore initial state
        scaler.reset();
        assert_eq!(scaler.get_scale(), 65536.0);
        assert_eq!(scaler.growth_tracker, 0);
        assert_ne!(scale_before, scaler.get_scale());
        assert_ne!(tracker_before, scaler.growth_tracker);
    }
}
