//! Data integrity validation for out-of-core tensor operations.
//!
//! This module provides checksumming and validation capabilities to ensure
//! data correctness when chunks are spilled to disk and loaded back.
//!
//! # Features
//!
//! - Multiple checksum algorithms (CRC32, XXHash, Blake3)
//! - Automatic validation on load
//! - Corruption detection and reporting
//! - Per-chunk metadata with checksums
//! - Configurable validation policies

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::hash::Hasher;

/// Checksum algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// No checksumming (fastest, no validation)
    None,
    /// CRC32 (fast, good for detecting errors)
    Crc32,
    /// XXHash64 (very fast, good distribution)
    XxHash64,
    /// Blake3 (cryptographically secure, slower)
    Blake3,
}

impl Default for ChecksumAlgorithm {
    fn default() -> Self {
        Self::XxHash64
    }
}

/// Chunk metadata with integrity information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkIntegrityMetadata {
    /// Chunk identifier
    pub chunk_id: String,
    /// Data size in bytes
    pub size_bytes: usize,
    /// Shape of the tensor chunk
    pub shape: Vec<usize>,
    /// Checksum algorithm used
    pub algorithm: ChecksumAlgorithm,
    /// Computed checksum value
    pub checksum: u64,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
}

impl ChunkIntegrityMetadata {
    /// Create new metadata for a chunk.
    pub fn new(
        chunk_id: String,
        size_bytes: usize,
        shape: Vec<usize>,
        algorithm: ChecksumAlgorithm,
        data: &[u8],
    ) -> Self {
        let checksum = compute_checksum(algorithm, data);
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            chunk_id,
            size_bytes,
            shape,
            algorithm,
            checksum,
            created_at,
        }
    }

    /// Validate data against this metadata.
    pub fn validate(&self, data: &[u8]) -> Result<()> {
        // Check size
        if data.len() != self.size_bytes {
            return Err(anyhow!(
                "Size mismatch for chunk {}: expected {} bytes, got {}",
                self.chunk_id,
                self.size_bytes,
                data.len()
            ));
        }

        // Check checksum
        if self.algorithm != ChecksumAlgorithm::None {
            let computed = compute_checksum(self.algorithm, data);
            if computed != self.checksum {
                return Err(anyhow!(
                    "Checksum mismatch for chunk {}: expected {:#x}, got {:#x}",
                    self.chunk_id,
                    self.checksum,
                    computed
                ));
            }
        }

        Ok(())
    }
}

/// Compute checksum for data using specified algorithm.
pub fn compute_checksum(algorithm: ChecksumAlgorithm, data: &[u8]) -> u64 {
    match algorithm {
        ChecksumAlgorithm::None => 0,
        ChecksumAlgorithm::Crc32 => {
            let crc = crc32fast::hash(data);
            u64::from(crc)
        }
        ChecksumAlgorithm::XxHash64 => {
            let mut hasher = xxhash_rust::xxh64::Xxh64::new(0);
            hasher.write(data);
            hasher.finish()
        }
        ChecksumAlgorithm::Blake3 => {
            let hash = blake3::hash(data);
            // Use first 8 bytes of hash as u64
            u64::from_le_bytes([
                hash.as_bytes()[0],
                hash.as_bytes()[1],
                hash.as_bytes()[2],
                hash.as_bytes()[3],
                hash.as_bytes()[4],
                hash.as_bytes()[5],
                hash.as_bytes()[6],
                hash.as_bytes()[7],
            ])
        }
    }
}

/// Validation policy for chunk loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Always validate checksums
    Strict,
    /// Validate only if metadata is present
    Opportunistic,
    /// Never validate (fastest, least safe)
    None,
}

impl Default for ValidationPolicy {
    fn default() -> Self {
        Self::Strict
    }
}

/// Data integrity checker for managing chunk validation.
pub struct IntegrityChecker {
    algorithm: ChecksumAlgorithm,
    policy: ValidationPolicy,
    stats: IntegrityStats,
}

/// Statistics for integrity checking operations.
#[derive(Debug, Clone, Default)]
pub struct IntegrityStats {
    /// Total validations performed
    pub validations: u64,
    /// Successful validations
    pub successes: u64,
    /// Failed validations (corrupted data)
    pub failures: u64,
    /// Total bytes validated
    pub bytes_validated: u64,
}

impl IntegrityChecker {
    /// Create a new integrity checker.
    pub fn new(algorithm: ChecksumAlgorithm, policy: ValidationPolicy) -> Self {
        Self {
            algorithm,
            policy,
            stats: IntegrityStats::default(),
        }
    }

    /// Create metadata for a chunk.
    pub fn create_metadata(
        &mut self,
        chunk_id: String,
        shape: Vec<usize>,
        data: &[u8],
    ) -> ChunkIntegrityMetadata {
        ChunkIntegrityMetadata::new(chunk_id, data.len(), shape, self.algorithm, data)
    }

    /// Validate chunk data against metadata.
    pub fn validate(&mut self, metadata: &ChunkIntegrityMetadata, data: &[u8]) -> Result<()> {
        self.stats.validations += 1;

        match self.policy {
            ValidationPolicy::None => {
                self.stats.successes += 1;
                Ok(())
            }
            ValidationPolicy::Opportunistic | ValidationPolicy::Strict => {
                match metadata.validate(data) {
                    Ok(()) => {
                        self.stats.successes += 1;
                        self.stats.bytes_validated += data.len() as u64;
                        Ok(())
                    }
                    Err(e) => {
                        self.stats.failures += 1;
                        Err(e)
                    }
                }
            }
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &IntegrityStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = IntegrityStats::default();
    }

    /// Get success rate (0.0 to 1.0).
    pub fn success_rate(&self) -> f64 {
        if self.stats.validations == 0 {
            1.0
        } else {
            self.stats.successes as f64 / self.stats.validations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checksum_algorithms() {
        let data = b"Hello, TenRSo!";

        // CRC32
        let crc32 = compute_checksum(ChecksumAlgorithm::Crc32, data);
        assert_ne!(crc32, 0);

        // XXHash64
        let xxhash = compute_checksum(ChecksumAlgorithm::XxHash64, data);
        assert_ne!(xxhash, 0);
        assert_ne!(xxhash, crc32);

        // Blake3
        let blake3 = compute_checksum(ChecksumAlgorithm::Blake3, data);
        assert_ne!(blake3, 0);
        assert_ne!(blake3, xxhash);

        // None
        let none = compute_checksum(ChecksumAlgorithm::None, data);
        assert_eq!(none, 0);
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = b"deterministic test data";

        let xxhash1 = compute_checksum(ChecksumAlgorithm::XxHash64, data);
        let xxhash2 = compute_checksum(ChecksumAlgorithm::XxHash64, data);
        assert_eq!(xxhash1, xxhash2);
    }

    #[test]
    fn test_checksum_sensitivity() {
        let data1 = b"Hello, TenRSo!";
        let data2 = b"Hello, TenRSo?"; // Single character different

        let hash1 = compute_checksum(ChecksumAlgorithm::XxHash64, data1);
        let hash2 = compute_checksum(ChecksumAlgorithm::XxHash64, data2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_metadata_creation() {
        let data = b"test chunk data";
        let metadata = ChunkIntegrityMetadata::new(
            "chunk_001".to_string(),
            data.len(),
            vec![5, 3],
            ChecksumAlgorithm::XxHash64,
            data,
        );

        assert_eq!(metadata.chunk_id, "chunk_001");
        assert_eq!(metadata.size_bytes, data.len());
        assert_eq!(metadata.shape, vec![5, 3]);
        assert_eq!(metadata.algorithm, ChecksumAlgorithm::XxHash64);
        assert_ne!(metadata.checksum, 0);
    }

    #[test]
    fn test_metadata_validation_success() {
        let data = b"test chunk data";
        let metadata = ChunkIntegrityMetadata::new(
            "chunk_001".to_string(),
            data.len(),
            vec![5, 3],
            ChecksumAlgorithm::XxHash64,
            data,
        );

        assert!(metadata.validate(data).is_ok());
    }

    #[test]
    fn test_metadata_validation_size_mismatch() {
        let data = b"test chunk data";
        let metadata = ChunkIntegrityMetadata::new(
            "chunk_001".to_string(),
            data.len(),
            vec![5, 3],
            ChecksumAlgorithm::XxHash64,
            data,
        );

        let wrong_size_data = b"different size";
        let result = metadata.validate(wrong_size_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Size mismatch"));
    }

    #[test]
    fn test_metadata_validation_checksum_mismatch() {
        let data = b"test chunk data";
        let metadata = ChunkIntegrityMetadata::new(
            "chunk_001".to_string(),
            data.len(),
            vec![5, 3],
            ChecksumAlgorithm::XxHash64,
            data,
        );

        // Same size but different content
        let corrupted_data = b"CORRUPTED  data";
        let result = metadata.validate(corrupted_data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Checksum mismatch"));
    }

    #[test]
    fn test_integrity_checker_creation() {
        let checker = IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);
        assert_eq!(checker.stats.validations, 0);
        assert_eq!(checker.stats.successes, 0);
        assert_eq!(checker.stats.failures, 0);
    }

    #[test]
    fn test_integrity_checker_validate_success() {
        let mut checker =
            IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);

        let data = b"test chunk data";
        let metadata = checker.create_metadata("chunk_001".to_string(), vec![5, 3], data);

        let result = checker.validate(&metadata, data);
        assert!(result.is_ok());
        assert_eq!(checker.stats.validations, 1);
        assert_eq!(checker.stats.successes, 1);
        assert_eq!(checker.stats.failures, 0);
        assert_eq!(checker.success_rate(), 1.0);
    }

    #[test]
    fn test_integrity_checker_validate_failure() {
        let mut checker =
            IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);

        let data = b"test chunk data";
        let metadata = checker.create_metadata("chunk_001".to_string(), vec![5, 3], data);

        let corrupted = b"CORRUPTED  data";
        let result = checker.validate(&metadata, corrupted);
        assert!(result.is_err());
        assert_eq!(checker.stats.validations, 1);
        assert_eq!(checker.stats.successes, 0);
        assert_eq!(checker.stats.failures, 1);
        assert_eq!(checker.success_rate(), 0.0);
    }

    #[test]
    fn test_integrity_checker_policy_none() {
        let mut checker =
            IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::None);

        let data = b"test chunk data";
        let metadata = checker.create_metadata("chunk_001".to_string(), vec![5, 3], data);

        // Even corrupted data passes with None policy
        let corrupted = b"CORRUPTED  data";
        let result = checker.validate(&metadata, corrupted);
        assert!(result.is_ok());
        assert_eq!(checker.stats.successes, 1);
        assert_eq!(checker.stats.failures, 0);
    }

    #[test]
    fn test_integrity_checker_stats_reset() {
        let mut checker =
            IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);

        let data = b"test chunk data";
        let metadata = checker.create_metadata("chunk_001".to_string(), vec![5, 3], data);

        checker.validate(&metadata, data).unwrap();
        assert_eq!(checker.stats.validations, 1);

        checker.reset_stats();
        assert_eq!(checker.stats.validations, 0);
        assert_eq!(checker.stats.successes, 0);
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut checker =
            IntegrityChecker::new(ChecksumAlgorithm::XxHash64, ValidationPolicy::Strict);

        // No validations yet
        assert_eq!(checker.success_rate(), 1.0);

        let data = b"test chunk data";
        let metadata = checker.create_metadata("chunk_001".to_string(), vec![5, 3], data);

        // 1 success
        checker.validate(&metadata, data).unwrap();
        assert_eq!(checker.success_rate(), 1.0);

        // 1 failure
        let corrupted = b"CORRUPTED  data";
        let _ = checker.validate(&metadata, corrupted);
        assert_eq!(checker.success_rate(), 0.5);
    }
}
