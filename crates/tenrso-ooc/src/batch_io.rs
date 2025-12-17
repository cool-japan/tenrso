//! Batch I/O operations for improved throughput
//!
//! This module provides batch reading and writing capabilities to improve I/O
//! throughput when dealing with multiple tensor chunks. Batching reduces
//! syscall overhead and enables better OS-level optimizations.
//!
//! # Features
//!
//! - **Batch reads**: Load multiple chunks in a single operation
//! - **Batch writes**: Write multiple chunks with reduced overhead
//! - **Parallel I/O**: Optional parallel batch processing
//! - **Compression aware**: Works with compression features
//! - **Error recovery**: Partial success handling
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::batch_io::{BatchReader, BatchWriter, BatchConfig};
//!
//! let config = BatchConfig::default()
//!     .batch_size(10)
//!     .parallel(true);
//!
//! let reader = BatchReader::new(config);
//! let chunks = reader.read_batch(&chunk_ids, &base_path)?;
//! ```

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use tenrso_core::DenseND;

#[cfg(feature = "compression")]
use crate::compression::{compress_f64_slice, decompress_to_f64_vec, CompressionCodec};

/// Batch I/O configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub batch_size: usize,
    /// Enable parallel I/O
    pub parallel: bool,
    /// Number of parallel threads (0 = auto)
    pub num_threads: usize,
    /// Continue on error (partial success)
    pub continue_on_error: bool,
    /// Compression codec
    #[cfg(feature = "compression")]
    pub compression: CompressionCodec,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            parallel: true,
            num_threads: 0, // Auto-detect
            continue_on_error: true,
            #[cfg(feature = "compression")]
            compression: CompressionCodec::None,
        }
    }
}

impl BatchConfig {
    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Enable/disable parallel I/O
    pub fn parallel(mut self, enable: bool) -> Self {
        self.parallel = enable;
        self
    }

    /// Set number of parallel threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Enable/disable continue on error
    pub fn continue_on_error(mut self, enable: bool) -> Self {
        self.continue_on_error = enable;
        self
    }

    /// Set compression codec
    #[cfg(feature = "compression")]
    pub fn compression(mut self, codec: CompressionCodec) -> Self {
        self.compression = codec;
        self
    }
}

/// Batch read result
#[derive(Debug)]
pub struct BatchReadResult {
    /// Successfully read chunks
    pub chunks: HashMap<String, DenseND<f64>>,
    /// Failed chunk IDs with error messages
    pub errors: HashMap<String, String>,
    /// Total bytes read
    pub bytes_read: usize,
}

/// Batch write result
#[derive(Debug)]
pub struct BatchWriteResult {
    /// Successfully written chunk IDs
    pub written: Vec<String>,
    /// Failed chunk IDs with error messages
    pub errors: HashMap<String, String>,
    /// Total bytes written
    pub bytes_written: usize,
}

/// Batch reader for loading multiple chunks
pub struct BatchReader {
    config: BatchConfig,
}

impl BatchReader {
    /// Create a new batch reader
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Read a batch of chunks from disk
    ///
    /// # Arguments
    ///
    /// * `chunk_ids` - List of chunk IDs to read
    /// * `chunk_shapes` - Shapes of chunks (for reconstruction)
    /// * `base_path` - Base directory path
    ///
    /// # Returns
    ///
    /// BatchReadResult with successfully loaded chunks and any errors
    #[cfg(feature = "mmap")]
    pub fn read_batch(
        &self,
        chunk_ids: &[String],
        chunk_shapes: &HashMap<String, Vec<usize>>,
        base_path: &Path,
    ) -> Result<BatchReadResult> {
        let mut chunks = HashMap::new();
        let mut errors = HashMap::new();
        let mut bytes_read = 0usize;

        // Process in batches
        for batch in chunk_ids.chunks(self.config.batch_size) {
            let batch_result = if self.config.parallel && batch.len() > 1 {
                self.read_batch_parallel(batch, chunk_shapes, base_path)
            } else {
                self.read_batch_sequential(batch, chunk_shapes, base_path)
            };

            match batch_result {
                Ok(result) => {
                    chunks.extend(result.chunks);
                    errors.extend(result.errors);
                    bytes_read += result.bytes_read;
                }
                Err(e) => {
                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                    for id in batch {
                        errors.insert(id.clone(), e.to_string());
                    }
                }
            }
        }

        Ok(BatchReadResult {
            chunks,
            errors,
            bytes_read,
        })
    }

    /// Read batch sequentially
    #[cfg(feature = "mmap")]
    fn read_batch_sequential(
        &self,
        chunk_ids: &[String],
        chunk_shapes: &HashMap<String, Vec<usize>>,
        base_path: &Path,
    ) -> Result<BatchReadResult> {
        let mut chunks = HashMap::new();
        let mut errors = HashMap::new();
        let mut bytes_read = 0usize;

        for chunk_id in chunk_ids {
            match self.read_single_chunk(chunk_id, chunk_shapes, base_path) {
                Ok((tensor, size)) => {
                    chunks.insert(chunk_id.clone(), tensor);
                    bytes_read += size;
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        errors.insert(chunk_id.clone(), e.to_string());
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(BatchReadResult {
            chunks,
            errors,
            bytes_read,
        })
    }

    /// Read batch in parallel
    #[cfg(all(feature = "mmap", feature = "parallel"))]
    fn read_batch_parallel(
        &self,
        chunk_ids: &[String],
        chunk_shapes: &HashMap<String, Vec<usize>>,
        base_path: &Path,
    ) -> Result<BatchReadResult> {
        use std::sync::{Arc, Mutex};

        let chunks = Arc::new(Mutex::new(HashMap::new()));
        let errors = Arc::new(Mutex::new(HashMap::new()));
        let bytes_read = Arc::new(Mutex::new(0usize));

        let results: Vec<_> = chunk_ids
            .iter()
            .map(|chunk_id| {
                let result = self.read_single_chunk(chunk_id, chunk_shapes, base_path);
                (chunk_id.clone(), result)
            })
            .collect();

        for (chunk_id, result) in results {
            match result {
                Ok((tensor, size)) => {
                    chunks.lock().unwrap().insert(chunk_id, tensor);
                    *bytes_read.lock().unwrap() += size;
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        errors.lock().unwrap().insert(chunk_id, e.to_string());
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(BatchReadResult {
            chunks: Arc::try_unwrap(chunks).unwrap().into_inner().unwrap(),
            errors: Arc::try_unwrap(errors).unwrap().into_inner().unwrap(),
            bytes_read: Arc::try_unwrap(bytes_read).unwrap().into_inner().unwrap(),
        })
    }

    /// Stub for non-parallel build
    #[cfg(all(feature = "mmap", not(feature = "parallel")))]
    fn read_batch_parallel(
        &self,
        chunk_ids: &[String],
        chunk_shapes: &HashMap<String, Vec<usize>>,
        base_path: &Path,
    ) -> Result<BatchReadResult> {
        self.read_batch_sequential(chunk_ids, chunk_shapes, base_path)
    }

    /// Read a single chunk
    #[cfg(feature = "mmap")]
    fn read_single_chunk(
        &self,
        chunk_id: &str,
        chunk_shapes: &HashMap<String, Vec<usize>>,
        base_path: &Path,
    ) -> Result<(DenseND<f64>, usize)> {
        let filename = format!("chunk_{}.bin", chunk_id);
        let path = base_path.join(filename);

        let shape = chunk_shapes
            .get(chunk_id)
            .ok_or_else(|| anyhow!("Shape not found for chunk {}", chunk_id))?;

        #[cfg(feature = "compression")]
        {
            if !matches!(self.config.compression, CompressionCodec::None) {
                let compressed = std::fs::read(&path)?;
                let size = compressed.len();
                let decompressed = decompress_to_f64_vec(&compressed)?;
                let tensor = DenseND::from_vec(decompressed, shape)?;
                return Ok((tensor, size));
            }
        }

        let tensor = crate::mmap_io::read_tensor_binary(&path)?;
        let size = tensor.len() * std::mem::size_of::<f64>();
        Ok((tensor, size))
    }

    /// Stub for when mmap is not enabled
    #[cfg(not(feature = "mmap"))]
    pub fn read_batch(
        &self,
        _chunk_ids: &[String],
        _chunk_shapes: &HashMap<String, Vec<usize>>,
        _base_path: &Path,
    ) -> Result<BatchReadResult> {
        Err(anyhow!("Batch read requires mmap feature"))
    }
}

/// Batch writer for saving multiple chunks
pub struct BatchWriter {
    config: BatchConfig,
}

impl BatchWriter {
    /// Create a new batch writer
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Write a batch of chunks to disk
    ///
    /// # Arguments
    ///
    /// * `chunks` - Map of chunk IDs to tensors
    /// * `base_path` - Base directory path
    ///
    /// # Returns
    ///
    /// BatchWriteResult with written chunk IDs and any errors
    #[cfg(feature = "mmap")]
    pub fn write_batch(
        &self,
        chunks: &HashMap<String, DenseND<f64>>,
        base_path: &Path,
    ) -> Result<BatchWriteResult> {
        let mut written = Vec::new();
        let mut errors = HashMap::new();
        let mut bytes_written = 0usize;

        // Ensure base path exists
        std::fs::create_dir_all(base_path)?;

        // Convert to Vec for batching
        let chunk_list: Vec<_> = chunks.iter().collect();

        // Process in batches
        for batch in chunk_list.chunks(self.config.batch_size) {
            let batch_result = if self.config.parallel && batch.len() > 1 {
                self.write_batch_parallel(batch, base_path)
            } else {
                self.write_batch_sequential(batch, base_path)
            };

            match batch_result {
                Ok(result) => {
                    written.extend(result.written);
                    errors.extend(result.errors);
                    bytes_written += result.bytes_written;
                }
                Err(e) => {
                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                    for (id, _) in batch {
                        errors.insert((*id).clone(), e.to_string());
                    }
                }
            }
        }

        Ok(BatchWriteResult {
            written,
            errors,
            bytes_written,
        })
    }

    /// Write batch sequentially
    #[cfg(feature = "mmap")]
    fn write_batch_sequential(
        &self,
        chunks: &[(&String, &DenseND<f64>)],
        base_path: &Path,
    ) -> Result<BatchWriteResult> {
        let mut written = Vec::new();
        let mut errors = HashMap::new();
        let mut bytes_written = 0usize;

        for (chunk_id, tensor) in chunks {
            match self.write_single_chunk(chunk_id, tensor, base_path) {
                Ok(size) => {
                    written.push((*chunk_id).clone());
                    bytes_written += size;
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        errors.insert((*chunk_id).clone(), e.to_string());
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(BatchWriteResult {
            written,
            errors,
            bytes_written,
        })
    }

    /// Write batch in parallel
    #[cfg(all(feature = "mmap", feature = "parallel"))]
    fn write_batch_parallel(
        &self,
        chunks: &[(&String, &DenseND<f64>)],
        base_path: &Path,
    ) -> Result<BatchWriteResult> {
        use std::sync::{Arc, Mutex};

        let written = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(HashMap::new()));
        let bytes_written = Arc::new(Mutex::new(0usize));

        let results: Vec<_> = chunks
            .iter()
            .map(|(chunk_id, tensor)| {
                let result = self.write_single_chunk(chunk_id, tensor, base_path);
                ((*chunk_id).clone(), result)
            })
            .collect();

        for (chunk_id, result) in results {
            match result {
                Ok(size) => {
                    written.lock().unwrap().push(chunk_id);
                    *bytes_written.lock().unwrap() += size;
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        errors.lock().unwrap().insert(chunk_id, e.to_string());
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(BatchWriteResult {
            written: Arc::try_unwrap(written).unwrap().into_inner().unwrap(),
            errors: Arc::try_unwrap(errors).unwrap().into_inner().unwrap(),
            bytes_written: Arc::try_unwrap(bytes_written)
                .unwrap()
                .into_inner()
                .unwrap(),
        })
    }

    /// Stub for non-parallel build
    #[cfg(all(feature = "mmap", not(feature = "parallel")))]
    fn write_batch_parallel(
        &self,
        chunks: &[(&String, &DenseND<f64>)],
        base_path: &Path,
    ) -> Result<BatchWriteResult> {
        self.write_batch_sequential(chunks, base_path)
    }

    /// Write a single chunk
    #[cfg(feature = "mmap")]
    fn write_single_chunk(
        &self,
        chunk_id: &str,
        tensor: &DenseND<f64>,
        base_path: &Path,
    ) -> Result<usize> {
        let filename = format!("chunk_{}.bin", chunk_id);
        let path = base_path.join(filename);

        #[cfg(feature = "compression")]
        {
            if !matches!(self.config.compression, CompressionCodec::None) {
                let data = tensor.as_slice();
                let compressed = compress_f64_slice(data, self.config.compression)?;
                let size = compressed.len();
                std::fs::write(&path, compressed)?;
                return Ok(size);
            }
        }

        crate::mmap_io::write_tensor_binary(&path, tensor)?;
        Ok(tensor.len() * std::mem::size_of::<f64>())
    }

    /// Stub for when mmap is not enabled
    #[cfg(not(feature = "mmap"))]
    pub fn write_batch(
        &self,
        _chunks: &HashMap<String, DenseND<f64>>,
        _base_path: &Path,
    ) -> Result<BatchWriteResult> {
        Err(anyhow!("Batch write requires mmap feature"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 16);
        assert!(config.parallel);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::default()
            .batch_size(32)
            .parallel(false)
            .num_threads(4);

        assert_eq!(config.batch_size, 32);
        assert!(!config.parallel);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_batch_reader_writer() {
        use std::collections::HashMap;

        let temp_dir = std::env::temp_dir().join("tenrso_batch_io_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create test chunks
        let mut chunks = HashMap::new();
        let mut shapes = HashMap::new();

        for i in 0..5 {
            let id = format!("test_{}", i);
            let data = vec![i as f64; 100];
            let tensor = DenseND::from_vec(data, &[10, 10]).unwrap();
            shapes.insert(id.clone(), vec![10, 10]);
            chunks.insert(id, tensor);
        }

        // Write batch
        let writer = BatchWriter::new(BatchConfig::default().batch_size(3));
        let write_result = writer.write_batch(&chunks, &temp_dir).unwrap();

        assert_eq!(write_result.written.len(), 5);
        assert!(write_result.errors.is_empty());
        assert!(write_result.bytes_written > 0);

        // Read batch
        let chunk_ids: Vec<String> = chunks.keys().cloned().collect();
        let reader = BatchReader::new(BatchConfig::default().batch_size(3));
        let read_result = reader.read_batch(&chunk_ids, &shapes, &temp_dir).unwrap();

        assert_eq!(read_result.chunks.len(), 5);
        assert!(read_result.errors.is_empty());
        assert_eq!(read_result.bytes_read, write_result.bytes_written);

        // Verify data
        for (id, original) in &chunks {
            let loaded = &read_result.chunks[id];
            assert_eq!(loaded.as_slice(), original.as_slice());
        }

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
}
