//! Compression support for spilled chunks
//!
//! This module provides transparent compression/decompression for chunks spilled to disk,
//! trading CPU time for reduced I/O bandwidth and storage requirements.
//!
//! # Supported Algorithms
//!
//! - **LZ4**: Fast compression with moderate ratios (~2-3x), minimal CPU overhead
//! - **Zstd**: High compression ratios (~3-5x), tunable compression levels
//! - **None**: No compression (fastest, largest disk usage)
//!
//! # Features
//!
//! - Adaptive compression based on data characteristics
//! - Transparent compression/decompression in spill operations
//! - Configurable compression levels
//! - Statistics tracking (compression ratio, time)
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::compression::{CompressionCodec, compress_bytes, decompress_bytes};
//!
//! let data = vec![1.0f64; 1000];
//! let bytes = bytemuck::cast_slice(&data);
//!
//! // Compress with LZ4
//! let compressed = compress_bytes(bytes, CompressionCodec::Lz4)?;
//! println!("Compression ratio: {:.2}x", bytes.len() as f64 / compressed.len() as f64);
//!
//! // Decompress
//! let decompressed = decompress_bytes(&compressed)?;
//! assert_eq!(bytes, decompressed.as_slice());
//! ```

use anyhow::{anyhow, Result};
use std::io::{Read, Write};

/// Compression codec for spilled chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    /// No compression (fastest, largest size)
    None,
    /// LZ4 compression (fast, moderate ratio ~2-3x)
    #[cfg(feature = "lz4-compression")]
    Lz4,
    /// Zstd compression (slower, high ratio ~3-5x)
    #[cfg(feature = "zstd-compression")]
    Zstd {
        /// Compression level (1-22, default 3)
        level: i32,
    },
}

impl Default for CompressionCodec {
    fn default() -> Self {
        // Choose default based on available features
        #[cfg(feature = "lz4-compression")]
        return CompressionCodec::Lz4;

        #[cfg(all(not(feature = "lz4-compression"), feature = "zstd-compression"))]
        return CompressionCodec::Zstd { level: 3 };

        #[cfg(not(any(feature = "lz4-compression", feature = "zstd-compression")))]
        return CompressionCodec::None;
    }
}

impl CompressionCodec {
    /// Create Zstd codec with custom compression level
    #[cfg(feature = "zstd-compression")]
    pub fn zstd(level: i32) -> Self {
        CompressionCodec::Zstd {
            level: level.clamp(1, 22),
        }
    }

    /// Check if this codec performs compression
    pub fn is_compressed(&self) -> bool {
        !matches!(self, CompressionCodec::None)
    }

    /// Get codec name for logging/debugging
    pub fn name(&self) -> &'static str {
        match self {
            CompressionCodec::None => "none",
            #[cfg(feature = "lz4-compression")]
            CompressionCodec::Lz4 => "lz4",
            #[cfg(feature = "zstd-compression")]
            CompressionCodec::Zstd { .. } => "zstd",
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression time in microseconds
    pub compression_time_us: u64,
    /// Decompression time in microseconds (if decompressed)
    pub decompression_time_us: Option<u64>,
}

impl CompressionStats {
    /// Calculate compression ratio (original / compressed)
    pub fn ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            return 0.0;
        }
        self.original_size as f64 / self.compressed_size as f64
    }

    /// Calculate space saved as percentage
    pub fn space_saved_pct(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        100.0 * (1.0 - self.compressed_size as f64 / self.original_size as f64)
    }

    /// Calculate compression throughput (MB/s)
    pub fn compression_throughput_mbps(&self) -> f64 {
        if self.compression_time_us == 0 {
            return 0.0;
        }
        let mb = self.original_size as f64 / (1024.0 * 1024.0);
        let seconds = self.compression_time_us as f64 / 1_000_000.0;
        mb / seconds
    }

    /// Calculate decompression throughput (MB/s)
    pub fn decompression_throughput_mbps(&self) -> Option<f64> {
        self.decompression_time_us.map(|us| {
            if us == 0 {
                return 0.0;
            }
            let mb = self.original_size as f64 / (1024.0 * 1024.0);
            let seconds = us as f64 / 1_000_000.0;
            mb / seconds
        })
    }
}

/// Compress raw bytes using the specified codec
///
/// # Arguments
///
/// * `data` - Raw bytes to compress
/// * `codec` - Compression codec to use
///
/// # Returns
///
/// Compressed bytes with codec metadata prepended
pub fn compress_bytes(data: &[u8], codec: CompressionCodec) -> Result<Vec<u8>> {
    let start = std::time::Instant::now();

    let result = match codec {
        CompressionCodec::None => {
            // No compression: just prepend magic byte
            let mut output = Vec::with_capacity(data.len() + 1);
            output.push(0u8); // Magic byte for "None"
            output.extend_from_slice(data);
            output
        }
        #[cfg(feature = "lz4-compression")]
        CompressionCodec::Lz4 => {
            let compressed =
                lz4::block::compress(data, Some(lz4::block::CompressionMode::FAST(1)), false)?;
            let mut output = Vec::with_capacity(compressed.len() + 9);
            output.push(1u8); // Magic byte for "Lz4"
            output.extend_from_slice(&(data.len() as u64).to_le_bytes());
            output.extend_from_slice(&compressed);
            output
        }
        #[cfg(feature = "zstd-compression")]
        CompressionCodec::Zstd { level } => {
            let mut encoder = zstd::Encoder::new(Vec::new(), level)?;
            encoder.write_all(data)?;
            let compressed = encoder.finish()?;
            let mut output = Vec::with_capacity(compressed.len() + 9);
            output.push(2u8); // Magic byte for "Zstd"
            output.extend_from_slice(&(data.len() as u64).to_le_bytes());
            output.extend_from_slice(&compressed);
            output
        }
    };

    let _elapsed = start.elapsed();
    // Compression statistics can be tracked via CompressionStats if needed

    Ok(result)
}

/// Decompress bytes that were compressed with `compress_bytes`
///
/// # Arguments
///
/// * `data` - Compressed bytes with codec metadata
///
/// # Returns
///
/// Original uncompressed bytes
pub fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Err(anyhow!("Empty compressed data"));
    }

    let start = std::time::Instant::now();
    let magic = data[0];

    let result = match magic {
        0 => {
            // No compression
            data[1..].to_vec()
        }
        1 => {
            // LZ4
            #[cfg(feature = "lz4-compression")]
            {
                if data.len() < 9 {
                    return Err(anyhow!("Invalid LZ4 compressed data: too short"));
                }
                let original_size = u64::from_le_bytes(data[1..9].try_into()?) as usize;
                let compressed = &data[9..];
                lz4::block::decompress(compressed, Some(original_size as i32))?
            }
            #[cfg(not(feature = "lz4-compression"))]
            {
                return Err(anyhow!("LZ4 compression not enabled"));
            }
        }
        2 => {
            // Zstd
            #[cfg(feature = "zstd-compression")]
            {
                if data.len() < 9 {
                    return Err(anyhow!("Invalid Zstd compressed data: too short"));
                }
                let _original_size = u64::from_le_bytes(data[1..9].try_into()?);
                let compressed = &data[9..];
                let mut decoder = zstd::Decoder::new(compressed)?;
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                decompressed
            }
            #[cfg(not(feature = "zstd-compression"))]
            {
                return Err(anyhow!("Zstd compression not enabled"));
            }
        }
        _ => {
            return Err(anyhow!("Unknown compression codec magic byte: {}", magic));
        }
    };

    let _elapsed = start.elapsed();
    // Decompression statistics can be tracked via CompressionStats if needed

    Ok(result)
}

/// Compress f64 tensor data
///
/// This is a convenience wrapper that handles the conversion from f64 slice to bytes.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
/// * `codec` - Compression codec to use
///
/// # Returns
///
/// Compressed bytes
pub fn compress_f64_slice(data: &[f64], codec: CompressionCodec) -> Result<Vec<u8>> {
    // Convert f64 slice to bytes using std::mem::size_of_val
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    compress_bytes(bytes, codec)
}

/// Decompress bytes to f64 tensor data
///
/// # Arguments
///
/// * `data` - Compressed bytes
///
/// # Returns
///
/// Vector of f64 values
pub fn decompress_to_f64_vec(data: &[u8]) -> Result<Vec<f64>> {
    let decompressed = decompress_bytes(data)?;
    if decompressed.len() % std::mem::size_of::<f64>() != 0 {
        return Err(anyhow!(
            "Decompressed data size {} is not a multiple of f64 size",
            decompressed.len()
        ));
    }

    let count = decompressed.len() / std::mem::size_of::<f64>();
    let mut result = Vec::with_capacity(count);
    unsafe {
        let ptr = decompressed.as_ptr() as *const f64;
        for i in 0..count {
            result.push(*ptr.add(i));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compression_roundtrip() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let compressed = compress_f64_slice(&data, CompressionCodec::None).unwrap();
        let decompressed = decompress_to_f64_vec(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[cfg(feature = "lz4-compression")]
    #[test]
    fn test_lz4_compression_roundtrip() {
        let data = vec![1.0f64; 1000]; // Highly compressible
        let compressed = compress_f64_slice(&data, CompressionCodec::Lz4).unwrap();
        let decompressed = decompress_to_f64_vec(&compressed).unwrap();
        assert_eq!(data, decompressed);

        // Check that compression actually reduced size
        let original_size = data.len() * std::mem::size_of::<f64>();
        assert!(compressed.len() < original_size);
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_zstd_compression_roundtrip() {
        let data = vec![1.0f64; 1000]; // Highly compressible
        let compressed = compress_f64_slice(&data, CompressionCodec::Zstd { level: 3 }).unwrap();
        let decompressed = decompress_to_f64_vec(&compressed).unwrap();
        assert_eq!(data, decompressed);

        // Check that compression actually reduced size
        let original_size = data.len() * std::mem::size_of::<f64>();
        assert!(compressed.len() < original_size);
    }

    #[cfg(all(feature = "lz4-compression", feature = "zstd-compression"))]
    #[test]
    fn test_compression_ratio_comparison() {
        // Test with highly compressible data
        let data = vec![1.0f64; 10000];

        let none = compress_f64_slice(&data, CompressionCodec::None).unwrap();
        let lz4 = compress_f64_slice(&data, CompressionCodec::Lz4).unwrap();
        let zstd = compress_f64_slice(&data, CompressionCodec::Zstd { level: 3 }).unwrap();

        println!("None: {} bytes", none.len());
        println!(
            "LZ4:  {} bytes ({:.2}x)",
            lz4.len(),
            none.len() as f64 / lz4.len() as f64
        );
        println!(
            "Zstd: {} bytes ({:.2}x)",
            zstd.len(),
            none.len() as f64 / zstd.len() as f64
        );

        // LZ4 and Zstd should compress significantly
        assert!(lz4.len() < none.len() / 10);
        assert!(zstd.len() < none.len() / 10);
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            original_size: 1000,
            compressed_size: 250,
            compression_time_us: 100,
            decompression_time_us: Some(50),
        };

        assert_eq!(stats.ratio(), 4.0);
        assert_eq!(stats.space_saved_pct(), 75.0);
        assert!(stats.compression_throughput_mbps() > 0.0);
        assert!(stats.decompression_throughput_mbps().unwrap() > 0.0);
    }

    #[test]
    fn test_codec_name() {
        assert_eq!(CompressionCodec::None.name(), "none");
        #[cfg(feature = "lz4-compression")]
        assert_eq!(CompressionCodec::Lz4.name(), "lz4");
        #[cfg(feature = "zstd-compression")]
        assert_eq!(CompressionCodec::Zstd { level: 3 }.name(), "zstd");
    }
}
