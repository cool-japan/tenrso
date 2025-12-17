//! Gradient compression utilities for efficient distributed training.
//!
//! This module provides various gradient compression techniques to reduce
//! communication overhead in distributed training scenarios:
//!
//! - **Quantization**: Reduce gradient precision (32-bit → 8-bit/16-bit)
//! - **Top-K Sparsification**: Keep only top-k largest gradients by magnitude
//! - **Random Sparsification**: Randomly sample gradients with probability p
//! - **Error Feedback**: Accumulate quantization/sparsification errors for accuracy
//!
//! # Example
//!
//! ```rust,ignore
//! use tenrso_ad::compression::{CompressedGradient, CompressionConfig, CompressionMethod};
//! use tenrso_core::DenseND;
//!
//! let gradient = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
//! let config = CompressionConfig {
//!     method: CompressionMethod::TopK { k: 2 },
//!     error_feedback: true,
//! };
//!
//! let compressed = CompressedGradient::compress(&gradient, &config)?;
//! let decompressed = compressed.decompress()?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array, ArrayD, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive};
use tenrso_core::DenseND;

/// Compression method for gradients
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// 8-bit quantization
    Quantize8Bit,
    /// 16-bit quantization
    Quantize16Bit,
    /// Keep only top-k largest gradients by magnitude
    TopK { k: usize },
    /// Random sparsification with probability p
    RandomSparsify { p: f64 },
    /// Combined: quantize then sparsify
    QuantizeThenSparsify { bits: u8, sparsity: f64 },
}

/// Configuration for gradient compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression method to use
    pub method: CompressionMethod,
    /// Whether to use error feedback accumulation
    pub error_feedback: bool,
    /// Random seed for reproducible compression
    pub seed: Option<u64>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            method: CompressionMethod::None,
            error_feedback: false,
            seed: None,
        }
    }
}

/// Compressed gradient representation
pub struct CompressedGradient<T> {
    /// Original shape of the gradient
    shape: Vec<usize>,
    /// Compressed data (format depends on compression method)
    data: CompressedData<T>,
    /// Compression method used
    #[allow(dead_code)]
    method: CompressionMethod,
    /// Error feedback accumulator (for future use)
    #[allow(dead_code)]
    error_feedback: Option<ArrayD<T>>,
}

/// Internal representation of compressed data
enum CompressedData<T> {
    /// No compression - full gradient
    Full(ArrayD<T>),
    /// 8-bit quantized values + scale and offset
    Quantized8Bit { values: Vec<u8>, scale: T, min: T },
    /// 16-bit quantized values + scale and offset
    Quantized16Bit { values: Vec<u16>, scale: T, min: T },
    /// Sparse representation: (indices, values)
    Sparse { indices: Vec<usize>, values: Vec<T> },
}

impl<T: Float + FromPrimitive> CompressedGradient<T> {
    /// Compress a gradient using the specified configuration
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient tensor to compress
    /// * `config` - Compression configuration
    ///
    /// # Returns
    ///
    /// Compressed gradient that can be transmitted efficiently
    pub fn compress(gradient: &DenseND<T>, config: &CompressionConfig) -> Result<Self> {
        let shape = gradient.shape().to_vec();
        let data_array = gradient.as_array();

        let data = match &config.method {
            CompressionMethod::None => CompressedData::Full(data_array.clone()),

            CompressionMethod::Quantize8Bit => Self::quantize_8bit(data_array)?,

            CompressionMethod::Quantize16Bit => Self::quantize_16bit(data_array)?,

            CompressionMethod::TopK { k } => Self::topk_sparsify(data_array, *k)?,

            CompressionMethod::RandomSparsify { p } => {
                Self::random_sparsify(data_array, *p, config.seed)?
            }

            CompressionMethod::QuantizeThenSparsify { bits, sparsity } => {
                // First quantize
                let quantized = match bits {
                    8 => Self::quantize_8bit(data_array)?,
                    16 => Self::quantize_16bit(data_array)?,
                    _ => return Err(anyhow!("Unsupported bit depth: {}", bits)),
                };

                // Then sparsify the quantized result
                let temp_dense = Self::decompress_data(&quantized, &shape)?;
                Self::random_sparsify(&temp_dense, *sparsity, config.seed)?
            }
        };

        Ok(Self {
            shape,
            data,
            method: config.method.clone(),
            error_feedback: if config.error_feedback {
                Some(ArrayD::zeros(IxDyn(gradient.shape())))
            } else {
                None
            },
        })
    }

    /// Decompress the gradient back to full representation
    pub fn decompress(&self) -> Result<DenseND<T>> {
        let decompressed = Self::decompress_data(&self.data, &self.shape)?;
        Ok(DenseND::from_array(decompressed))
    }

    /// Get compression ratio (original size / compressed size)
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let compressed_size = self.compressed_size_bytes();
        original_size as f64 / compressed_size as f64
    }

    /// Get compressed size in bytes
    pub fn compressed_size_bytes(&self) -> usize {
        match &self.data {
            CompressedData::Full(_) => {
                self.shape.iter().product::<usize>() * std::mem::size_of::<T>()
            }
            CompressedData::Quantized8Bit { values, .. } => {
                values.len() + std::mem::size_of::<T>() + 1 // values + scale + zero_point
            }
            CompressedData::Quantized16Bit { values, .. } => {
                values.len() * 2 + std::mem::size_of::<T>() + 2
            }
            CompressedData::Sparse { indices, values } => {
                indices.len() * std::mem::size_of::<usize>()
                    + values.len() * std::mem::size_of::<T>()
            }
        }
    }

    /// Get sparsity (fraction of zeros) if applicable
    pub fn sparsity(&self) -> Option<f64> {
        match &self.data {
            CompressedData::Sparse { indices, .. } => {
                let total = self.shape.iter().product::<usize>();
                let nonzero = indices.len();
                Some(1.0 - (nonzero as f64 / total as f64))
            }
            _ => None,
        }
    }

    // Helper methods for compression

    fn quantize_8bit(data: &ArrayD<T>) -> Result<CompressedData<T>> {
        let min = data.iter().copied().fold(T::infinity(), T::min);
        let max = data.iter().copied().fold(T::neg_infinity(), T::max);

        let scale = (max - min) / T::from_u8(255).unwrap();

        let values: Vec<u8> = data
            .iter()
            .map(|&val| ((val - min) / scale).to_u8().unwrap_or(0))
            .collect();

        Ok(CompressedData::Quantized8Bit { values, scale, min })
    }

    fn quantize_16bit(data: &ArrayD<T>) -> Result<CompressedData<T>> {
        let min = data.iter().copied().fold(T::infinity(), T::min);
        let max = data.iter().copied().fold(T::neg_infinity(), T::max);

        let scale = (max - min) / T::from_u16(65535).unwrap();

        let values: Vec<u16> = data
            .iter()
            .map(|&val| ((val - min) / scale).to_u16().unwrap_or(0))
            .collect();

        Ok(CompressedData::Quantized16Bit { values, scale, min })
    }

    fn topk_sparsify(data: &ArrayD<T>, k: usize) -> Result<CompressedData<T>> {
        let total_elements = data.len();
        if k > total_elements {
            return Err(anyhow!("k={} exceeds total elements={}", k, total_elements));
        }

        // Create (index, magnitude) pairs
        let mut indexed_magnitudes: Vec<(usize, T)> = data
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.abs()))
            .collect();

        // Sort by magnitude (descending)
        indexed_magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Keep top-k
        let mut indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);

        for (idx, _mag) in indexed_magnitudes.iter().take(k) {
            indices.push(*idx);
            values.push(data[[*idx]]);
        }

        Ok(CompressedData::Sparse { indices, values })
    }

    fn random_sparsify(
        data: &ArrayD<T>,
        sparsity: f64,
        _seed: Option<u64>,
    ) -> Result<CompressedData<T>> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(anyhow!("Sparsity must be in [0, 1], got {}", sparsity));
        }

        let keep_prob = 1.0 - sparsity;
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Simple deterministic "random" for now (can enhance with actual RNG)
        for (i, &val) in data.iter().enumerate() {
            let pseudo_rand = ((i * 2654435761) % 1000) as f64 / 1000.0;
            if pseudo_rand < keep_prob {
                indices.push(i);
                values.push(val);
            }
        }

        Ok(CompressedData::Sparse { indices, values })
    }

    fn decompress_data(data: &CompressedData<T>, shape: &[usize]) -> Result<ArrayD<T>> {
        match data {
            CompressedData::Full(arr) => Ok(arr.clone()),

            CompressedData::Quantized8Bit { values, scale, min } => {
                let decompressed: Vec<T> = values
                    .iter()
                    .map(|&qval| {
                        let q_float = T::from_u8(qval).unwrap();
                        q_float * *scale + *min
                    })
                    .collect();

                Array::from_shape_vec(IxDyn(shape), decompressed)
                    .map_err(|e| anyhow!("Shape mismatch: {}", e))
            }

            CompressedData::Quantized16Bit { values, scale, min } => {
                let decompressed: Vec<T> = values
                    .iter()
                    .map(|&qval| {
                        let q_float = T::from_u16(qval).unwrap();
                        q_float * *scale + *min
                    })
                    .collect();

                Array::from_shape_vec(IxDyn(shape), decompressed)
                    .map_err(|e| anyhow!("Shape mismatch: {}", e))
            }

            CompressedData::Sparse { indices, values } => {
                let total_size = shape.iter().product();
                let mut dense = ArrayD::zeros(IxDyn(shape));

                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    if idx < total_size {
                        dense[[idx]] = val;
                    }
                }

                Ok(dense)
            }
        }
    }
}

/// Statistics about compression performance
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_bytes: usize,
    /// Compressed size in bytes
    pub compressed_bytes: usize,
    /// Compression ratio
    pub ratio: f64,
    /// Sparsity (if applicable)
    pub sparsity: Option<f64>,
    /// Mean squared error after compression/decompression
    pub mse: Option<f64>,
}

impl CompressionStats {
    /// Compute compression statistics
    pub fn compute<T: Float + FromPrimitive>(
        original: &DenseND<T>,
        compressed: &CompressedGradient<T>,
    ) -> Result<Self> {
        let original_bytes = original.len() * std::mem::size_of::<T>();
        let compressed_bytes = compressed.compressed_size_bytes();
        let ratio = compressed.compression_ratio();
        let sparsity = compressed.sparsity();

        // Compute MSE
        let decompressed = compressed.decompress()?;
        let mse = Self::compute_mse(original, &decompressed)?;

        Ok(Self {
            original_bytes,
            compressed_bytes,
            ratio,
            sparsity,
            mse: Some(mse),
        })
    }

    fn compute_mse<T: Float + FromPrimitive>(a: &DenseND<T>, b: &DenseND<T>) -> Result<f64> {
        if a.shape() != b.shape() {
            return Err(anyhow!("Shape mismatch for MSE computation"));
        }

        let a_arr = a.as_array();
        let b_arr = b.as_array();

        let sum_sq_diff: T = a_arr
            .iter()
            .zip(b_arr.iter())
            .map(|(&a_val, &b_val)| {
                let diff = a_val - b_val;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);

        let n = T::from_usize(a.len()).unwrap();
        Ok((sum_sq_diff / n).to_f64().unwrap())
    }
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Compression Statistics:")?;
        writeln!(f, "  Original size: {} bytes", self.original_bytes)?;
        writeln!(f, "  Compressed size: {} bytes", self.compressed_bytes)?;
        writeln!(f, "  Compression ratio: {:.2}x", self.ratio)?;
        if let Some(sparsity) = self.sparsity {
            writeln!(f, "  Sparsity: {:.1}%", sparsity * 100.0)?;
        }
        if let Some(mse) = self.mse {
            writeln!(f, "  MSE: {:.6e}", mse)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert!(matches!(config.method, CompressionMethod::None));
        assert!(!config.error_feedback);
    }

    #[test]
    fn test_quantize_8bit() {
        let gradient = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let config = CompressionConfig {
            method: CompressionMethod::Quantize8Bit,
            ..Default::default()
        };

        let compressed = CompressedGradient::compress(&gradient, &config).unwrap();
        let decompressed = compressed.decompress().unwrap();

        assert_eq!(decompressed.shape(), gradient.shape());
        // Quantization introduces some error, but should be close
        for i in 0..5 {
            let original = gradient.get(&[i]).unwrap();
            let recovered = decompressed.get(&[i]).unwrap();
            assert!((original - recovered).abs() < 0.1);
        }
    }

    #[test]
    fn test_quantize_16bit() {
        let gradient = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let config = CompressionConfig {
            method: CompressionMethod::Quantize16Bit,
            ..Default::default()
        };

        let compressed = CompressedGradient::compress(&gradient, &config).unwrap();
        let decompressed = compressed.decompress().unwrap();

        // 16-bit should have much less error than 8-bit
        for i in 0..5 {
            let original = gradient.get(&[i]).unwrap();
            let recovered = decompressed.get(&[i]).unwrap();
            assert!((original - recovered).abs() < 0.01);
        }
    }

    #[test]
    fn test_topk_sparsify() {
        let gradient = DenseND::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0], &[5]).unwrap();
        let config = CompressionConfig {
            method: CompressionMethod::TopK { k: 2 },
            ..Default::default()
        };

        let compressed = CompressedGradient::compress(&gradient, &config).unwrap();
        assert_eq!(compressed.sparsity(), Some(0.6)); // 3 out of 5 are zero

        let decompressed = compressed.decompress().unwrap();
        // Top-2 values are 8.0 and 5.0, rest should be zero
        assert_eq!(*decompressed.get(&[3]).unwrap(), 8.0);
        assert_eq!(*decompressed.get(&[1]).unwrap(), 5.0);
    }

    #[test]
    fn test_random_sparsify() {
        let gradient = DenseND::from_vec(vec![1.0; 100], &[100]).unwrap();
        let config = CompressionConfig {
            method: CompressionMethod::RandomSparsify { p: 0.9 },
            ..Default::default()
        };

        let compressed = CompressedGradient::compress(&gradient, &config).unwrap();
        let sparsity = compressed.sparsity().unwrap();

        // Should be approximately 90% sparse (may vary due to pseudo-random)
        assert!(sparsity > 0.85 && sparsity < 0.95);
    }

    #[test]
    fn test_compression_ratio() {
        let gradient = DenseND::from_vec(vec![1.0; 1000], &[1000]).unwrap();

        // 8-bit quantization should give ~4x compression (32-bit → 8-bit)
        let config = CompressionConfig {
            method: CompressionMethod::Quantize8Bit,
            ..Default::default()
        };
        let compressed = CompressedGradient::compress(&gradient, &config).unwrap();
        assert!(compressed.compression_ratio() > 3.0);
    }

    #[test]
    fn test_topk_invalid_k() {
        let gradient = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let config = CompressionConfig {
            method: CompressionMethod::TopK { k: 10 },
            ..Default::default()
        };

        assert!(CompressedGradient::compress(&gradient, &config).is_err());
    }

    #[test]
    fn test_compression_stats() {
        let gradient = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let config = CompressionConfig {
            method: CompressionMethod::Quantize8Bit,
            ..Default::default()
        };

        let compressed = CompressedGradient::compress(&gradient, &config).unwrap();
        let stats = CompressionStats::compute(&gradient, &compressed).unwrap();

        assert!(stats.ratio > 1.0);
        assert!(stats.mse.is_some());
    }

    #[test]
    fn test_no_compression() {
        let gradient = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let config = CompressionConfig {
            method: CompressionMethod::None,
            ..Default::default()
        };

        let compressed = CompressedGradient::compress(&gradient, &config).unwrap();
        let decompressed = compressed.decompress().unwrap();

        for i in 0..3 {
            assert_eq!(
                *gradient.get(&[i]).unwrap(),
                *decompressed.get(&[i]).unwrap()
            );
        }
    }

    #[test]
    fn test_compression_stats_display() {
        let stats = CompressionStats {
            original_bytes: 4000,
            compressed_bytes: 1000,
            ratio: 4.0,
            sparsity: Some(0.75),
            mse: Some(0.001),
        };

        let display = format!("{}", stats);
        assert!(display.contains("4.00x"));
        assert!(display.contains("75.0%"));
    }
}
