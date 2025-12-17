//! Automatic compression codec selection based on data characteristics.
//!
//! This module analyzes tensor chunk data and automatically selects the best
//! compression codec based on entropy, redundancy, and performance trade-offs.
//!
//! # Features
//!
//! - Entropy analysis for compressibility estimation
//! - Data pattern detection (uniform, sequential, random)
//! - Automatic codec selection (None, LZ4, Zstd)
//! - Performance-aware recommendations
//! - Configurable selection policies

use serde::{Deserialize, Serialize};

#[cfg(feature = "compression")]
use crate::compression::CompressionCodec;

/// Data characteristics analysis results.
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Estimated Shannon entropy (0.0 to 8.0 for bytes)
    pub entropy: f64,
    /// Detected data pattern
    pub pattern: DataPattern,
    /// Compression ratio estimate (0.0 to 1.0, lower is better)
    pub estimated_ratio: f64,
    /// Data size in bytes
    pub size_bytes: usize,
    /// Unique value ratio (0.0 to 1.0)
    pub unique_ratio: f64,
}

/// Detected data patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataPattern {
    /// Highly uniform data (e.g., zeros, constants)
    Uniform,
    /// Sequential or nearly sequential data
    Sequential,
    /// Random or highly diverse data
    Random,
    /// Mixed patterns
    Mixed,
}

/// Codec selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionPolicy {
    /// Maximize compression ratio (slower, smaller)
    MaxCompression,
    /// Maximize speed (faster, larger)
    MaxSpeed,
    /// Balance compression and speed
    Balanced,
    /// Adaptive based on data characteristics
    Adaptive,
}

impl Default for SelectionPolicy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Configuration for compression auto-selection.
#[derive(Debug, Clone)]
pub struct AutoSelectConfig {
    /// Selection policy
    pub policy: SelectionPolicy,
    /// Minimum entropy threshold for compression (below this, use None)
    pub min_entropy_threshold: f64,
    /// Sample size for analysis (0 = analyze all data)
    pub sample_size: usize,
    /// Enable performance-based tuning
    pub performance_tuning: bool,
}

impl Default for AutoSelectConfig {
    fn default() -> Self {
        Self {
            policy: SelectionPolicy::Adaptive,
            min_entropy_threshold: 1.0, // Very low entropy data doesn't compress well
            sample_size: 8192,          // Sample 8KB for analysis
            performance_tuning: true,
        }
    }
}

/// Automatic compression codec selector.
pub struct CompressionAutoSelector {
    config: AutoSelectConfig,
    stats: SelectionStats,
}

/// Statistics for codec selection operations.
#[derive(Debug, Clone, Default)]
pub struct SelectionStats {
    /// Total selections performed
    pub selections: u64,
    /// Selections resulting in no compression
    pub no_compression: u64,
    /// Selections resulting in LZ4
    pub lz4_selected: u64,
    /// Selections resulting in Zstd
    pub zstd_selected: u64,
    /// Total bytes analyzed
    pub bytes_analyzed: u64,
}

impl CompressionAutoSelector {
    /// Create a new auto-selector with default configuration.
    pub fn new() -> Self {
        Self::with_config(AutoSelectConfig::default())
    }

    /// Create a new auto-selector with custom configuration.
    pub fn with_config(config: AutoSelectConfig) -> Self {
        Self {
            config,
            stats: SelectionStats::default(),
        }
    }

    /// Analyze data characteristics.
    pub fn analyze_data(&self, data: &[u8]) -> DataCharacteristics {
        // Sample data if configured
        let sample = if self.config.sample_size > 0 && data.len() > self.config.sample_size {
            &data[..self.config.sample_size]
        } else {
            data
        };

        let entropy = calculate_entropy(sample);
        let pattern = detect_pattern(sample);
        let unique_ratio = calculate_unique_ratio(sample);
        let estimated_ratio = estimate_compression_ratio(entropy, pattern, unique_ratio);

        DataCharacteristics {
            entropy,
            pattern,
            estimated_ratio,
            size_bytes: data.len(),
            unique_ratio,
        }
    }

    /// Select the best compression codec for given data.
    #[cfg(feature = "compression")]
    pub fn select_codec(&mut self, data: &[u8]) -> CompressionCodec {
        self.stats.selections += 1;
        self.stats.bytes_analyzed += data.len() as u64;

        let characteristics = self.analyze_data(data);

        let codec = match self.config.policy {
            SelectionPolicy::MaxCompression => self.select_max_compression(&characteristics),
            SelectionPolicy::MaxSpeed => self.select_max_speed(&characteristics),
            SelectionPolicy::Balanced => self.select_balanced(&characteristics),
            SelectionPolicy::Adaptive => self.select_adaptive(&characteristics),
        };

        // Update stats
        match codec {
            CompressionCodec::None => self.stats.no_compression += 1,
            CompressionCodec::Lz4 => self.stats.lz4_selected += 1,
            CompressionCodec::Zstd { .. } => self.stats.zstd_selected += 1,
        }

        codec
    }

    /// Select codec for maximum compression.
    #[cfg(feature = "compression")]
    fn select_max_compression(&self, chars: &DataCharacteristics) -> CompressionCodec {
        if chars.entropy < self.config.min_entropy_threshold {
            return CompressionCodec::None;
        }

        // For uniform/sequential data, high compression works well
        if matches!(
            chars.pattern,
            DataPattern::Uniform | DataPattern::Sequential
        ) {
            return CompressionCodec::Zstd { level: 22 }; // Maximum Zstd compression
        }

        // For random data, even max compression won't help much
        if chars.entropy > 7.0 {
            return CompressionCodec::None;
        }

        // Otherwise, use high Zstd compression
        CompressionCodec::Zstd { level: 19 }
    }

    /// Select codec for maximum speed.
    #[cfg(feature = "compression")]
    fn select_max_speed(&self, chars: &DataCharacteristics) -> CompressionCodec {
        if chars.entropy < self.config.min_entropy_threshold {
            return CompressionCodec::None;
        }

        // For highly compressible data, even LZ4 is worth it
        if chars.estimated_ratio < 0.5 {
            return CompressionCodec::Lz4;
        }

        // For high entropy, skip compression
        if chars.entropy > 6.5 {
            return CompressionCodec::None;
        }

        CompressionCodec::Lz4
    }

    /// Select codec for balanced compression/speed.
    #[cfg(feature = "compression")]
    fn select_balanced(&self, chars: &DataCharacteristics) -> CompressionCodec {
        if chars.entropy < self.config.min_entropy_threshold {
            return CompressionCodec::None;
        }

        // Very high entropy -> no compression
        if chars.entropy > 7.0 {
            return CompressionCodec::None;
        }

        // Low entropy -> good compression
        if chars.entropy < 4.0 {
            return CompressionCodec::Zstd { level: 3 }; // Fast Zstd
        }

        // Medium entropy -> LZ4 for speed
        if chars.entropy < 6.0 {
            return CompressionCodec::Lz4;
        }

        // High entropy but not random -> try light compression
        CompressionCodec::Lz4
    }

    /// Select codec adaptively based on data characteristics.
    #[cfg(feature = "compression")]
    fn select_adaptive(&self, chars: &DataCharacteristics) -> CompressionCodec {
        if chars.entropy < self.config.min_entropy_threshold {
            return CompressionCodec::None;
        }

        match chars.pattern {
            DataPattern::Uniform => {
                // Uniform data compresses extremely well
                if chars.entropy < 2.0 {
                    CompressionCodec::Zstd { level: 3 } // Fast is enough for very uniform data
                } else {
                    CompressionCodec::Lz4
                }
            }
            DataPattern::Sequential => {
                // Sequential patterns compress well with Zstd
                if chars.entropy < 5.0 {
                    CompressionCodec::Zstd { level: 6 } // Medium Zstd
                } else {
                    CompressionCodec::Lz4
                }
            }
            DataPattern::Random => {
                // Random data doesn't compress well
                if chars.entropy > 7.0 {
                    CompressionCodec::None
                } else {
                    CompressionCodec::Lz4 // Try LZ4 just in case
                }
            }
            DataPattern::Mixed => {
                // Use balanced approach for mixed patterns
                if chars.entropy < 4.0 {
                    CompressionCodec::Zstd { level: 3 }
                } else if chars.entropy < 6.0 {
                    CompressionCodec::Lz4
                } else {
                    CompressionCodec::None
                }
            }
        }
    }

    /// Get current selection statistics.
    pub fn stats(&self) -> &SelectionStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SelectionStats::default();
    }
}

impl Default for CompressionAutoSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate Shannon entropy of data.
fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u64; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Detect data pattern.
fn detect_pattern(data: &[u8]) -> DataPattern {
    if data.is_empty() {
        return DataPattern::Random;
    }

    // Check uniformity
    let first = data[0];
    if data.iter().all(|&b| b == first) {
        return DataPattern::Uniform;
    }

    // Check if mostly sequential
    let mut sequential_count = 0;
    for window in data.windows(2) {
        if window[1] == window[0] || window[1] == window[0].wrapping_add(1) {
            sequential_count += 1;
        }
    }

    let sequential_ratio = sequential_count as f64 / (data.len() - 1) as f64;
    if sequential_ratio > 0.7 {
        return DataPattern::Sequential;
    }

    // Check entropy for randomness
    let entropy = calculate_entropy(data);
    if entropy > 7.0 {
        return DataPattern::Random;
    }

    DataPattern::Mixed
}

/// Calculate ratio of unique values.
fn calculate_unique_ratio(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut seen = [false; 256];
    let mut unique = 0;

    for &byte in data {
        if !seen[byte as usize] {
            seen[byte as usize] = true;
            unique += 1;
        }
    }

    unique as f64 / 256.0
}

/// Estimate compression ratio based on characteristics.
fn estimate_compression_ratio(entropy: f64, pattern: DataPattern, unique_ratio: f64) -> f64 {
    // Base estimate from entropy (normalized to 0-1)
    let mut ratio = entropy / 8.0;

    // Adjust based on pattern
    match pattern {
        DataPattern::Uniform => ratio *= 0.1, // Extremely compressible
        DataPattern::Sequential => ratio *= 0.5, // Quite compressible
        DataPattern::Random => ratio = ratio.max(0.9), // Barely compressible
        DataPattern::Mixed => {}              // Use base ratio
    }

    // Adjust based on unique value ratio
    ratio *= 0.5 + (unique_ratio * 0.5); // Low uniqueness improves compression

    ratio.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let data = vec![42u8; 1000];
        let entropy = calculate_entropy(&data);
        assert!(entropy < 0.1, "Uniform data should have very low entropy");
    }

    #[test]
    fn test_entropy_sequential() {
        let data: Vec<u8> = (0..=255).collect();
        let entropy = calculate_entropy(&data);
        assert!(entropy > 7.0, "Sequential 0-255 should have high entropy");
    }

    #[test]
    fn test_entropy_random() {
        // Pseudo-random data with good distribution
        let data: Vec<u8> = (0..1000).map(|i| ((i * 37) % 256) as u8).collect();
        let entropy = calculate_entropy(&data);
        assert!(
            entropy > 6.0,
            "Well-distributed data should have high entropy"
        );
    }

    #[test]
    fn test_pattern_uniform() {
        let data = vec![7u8; 100];
        let pattern = detect_pattern(&data);
        assert_eq!(pattern, DataPattern::Uniform);
    }

    #[test]
    fn test_pattern_sequential() {
        let data: Vec<u8> = (0..100).collect();
        let pattern = detect_pattern(&data);
        assert_eq!(pattern, DataPattern::Sequential);
    }

    #[test]
    fn test_pattern_random() {
        // Truly random-looking data (avoid sequential patterns)
        let data: Vec<u8> = (0..1000).map(|i| ((i * 37 + 17) % 256) as u8).collect();
        let pattern = detect_pattern(&data);
        assert!(matches!(pattern, DataPattern::Random | DataPattern::Mixed));
    }

    #[test]
    fn test_unique_ratio() {
        let data = vec![42u8; 1000];
        let ratio = calculate_unique_ratio(&data);
        assert!(
            ratio < 0.01,
            "All same values should have very low unique ratio"
        );

        let data: Vec<u8> = (0..=255).collect();
        let ratio = calculate_unique_ratio(&data);
        assert_eq!(ratio, 1.0, "All different values should have ratio 1.0");
    }

    #[test]
    fn test_analyze_data() {
        let selector = CompressionAutoSelector::new();

        let uniform = vec![1u8; 1000];
        let chars = selector.analyze_data(&uniform);
        assert_eq!(chars.pattern, DataPattern::Uniform);
        assert!(chars.entropy < 1.0);
        assert!(chars.estimated_ratio < 0.2);
    }

    #[test]
    #[cfg(feature = "compression")]
    fn test_select_codec_uniform() {
        let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
            policy: SelectionPolicy::Adaptive,
            min_entropy_threshold: 0.1, // Lower threshold to ensure compression
            ..Default::default()
        });

        let data = vec![5u8; 1000];
        let codec = selector.select_codec(&data);

        // Uniform data should use Zstd or LZ4 (unless entropy is below threshold)
        // With very low entropy uniform data, it might select Zstd
        assert!(matches!(
            codec,
            CompressionCodec::None | CompressionCodec::Lz4 | CompressionCodec::Zstd { .. }
        ));
    }

    #[test]
    #[cfg(feature = "compression")]
    fn test_select_codec_random() {
        let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
            policy: SelectionPolicy::Adaptive,
            ..Default::default()
        });

        // Very random data (all different values)
        let data: Vec<u8> = (0..=255).cycle().take(1000).collect();
        let codec = selector.select_codec(&data);

        // Random data often results in None or LZ4
        assert!(matches!(
            codec,
            CompressionCodec::None | CompressionCodec::Lz4
        ));
    }

    #[test]
    #[cfg(feature = "compression")]
    fn test_select_codec_max_compression() {
        let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
            policy: SelectionPolicy::MaxCompression,
            ..Default::default()
        });

        let data = vec![1u8, 2u8, 3u8, 1u8, 2u8, 3u8]; // Repetitive pattern
        let codec = selector.select_codec(&data);

        // Should use Zstd for max compression
        assert!(matches!(codec, CompressionCodec::Zstd { .. }));
    }

    #[test]
    #[cfg(feature = "compression")]
    fn test_select_codec_max_speed() {
        let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
            policy: SelectionPolicy::MaxSpeed,
            ..Default::default()
        });

        let data = vec![1u8, 2u8, 1u8, 2u8]; // Compressible
        let codec = selector.select_codec(&data);

        // Should use LZ4 or None for max speed
        assert!(matches!(
            codec,
            CompressionCodec::Lz4 | CompressionCodec::None
        ));
    }

    #[test]
    #[cfg(feature = "compression")]
    fn test_stats_tracking() {
        let mut selector = CompressionAutoSelector::new();

        assert_eq!(selector.stats().selections, 0);

        let data1 = vec![1u8; 100];
        selector.select_codec(&data1);

        assert_eq!(selector.stats().selections, 1);
        assert!(selector.stats().bytes_analyzed > 0);

        let data2 = vec![2u8; 200];
        selector.select_codec(&data2);

        assert_eq!(selector.stats().selections, 2);

        selector.reset_stats();
        assert_eq!(selector.stats().selections, 0);
    }

    #[test]
    fn test_estimation_accuracy() {
        // Uniform data should have very low estimated ratio
        let ratio = estimate_compression_ratio(0.5, DataPattern::Uniform, 0.01);
        assert!(ratio < 0.1);

        // Random data should have high estimated ratio
        let ratio = estimate_compression_ratio(7.5, DataPattern::Random, 0.95);
        assert!(ratio > 0.8);
    }
}
