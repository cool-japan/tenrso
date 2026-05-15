//! Parquet I/O for tensors
//!
//! This module provides serialization and deserialization of tensors using Apache Parquet format.
//!
//! # Features
//!
//! - Write `DenseND<T>` tensors to Parquet files
//! - Read Parquet files back to `DenseND<T>` tensors
//! - Preserve tensor shape and metadata
//! - Support for columnar compression
//! - Efficient partial reads
//!
//! # Example
//!
//! ```ignore
//! use tenrso_core::DenseND;
//! use tenrso_ooc::parquet_io::{ParquetWriter, ParquetReader};
//!
//! // Write tensor to Parquet
//! let tensor = DenseND::<f64>::ones(&[10, 20, 30]);
//! let mut writer = ParquetWriter::new("tensor.parquet")?;
//! writer.write(&tensor)?;
//! writer.finish()?;
//!
//! // Read tensor from Parquet
//! let reader = ParquetReader::open("tensor.parquet")?;
//! let loaded: DenseND<f64> = reader.read()?;
//! ```

use anyhow::{anyhow, Result};
use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tenrso_core::DenseND;

/// Parquet writer for tensors
///
/// Writes `DenseND<T>` tensors to Parquet format files with columnar compression.
pub struct ParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
}

impl ParquetWriter {
    /// Create a new Parquet writer
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    ///
    /// # Returns
    ///
    /// A new `ParquetWriter` instance
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;

        // Define schema: data and shape_info columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Float64, false),
            Field::new("shape_info", DataType::Int64, false),
        ]));

        // Create writer with default compression
        let props = WriterProperties::builder().build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self { writer, schema })
    }

    /// Write a tensor to the Parquet file
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to write
    ///
    /// # Returns
    ///
    /// Ok(()) if successful
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails
    pub fn write(&mut self, tensor: &DenseND<f64>) -> Result<()> {
        // Flatten tensor data
        let data = tensor.as_slice();
        let shape = tensor.shape();
        let rank = shape.len();

        // Determine array length: max(data_len, rank + 1)
        let array_len = data.len().max(rank + 1);

        // Build data array, pad with zeros if needed
        let mut data_vec = data.to_vec();
        while data_vec.len() < array_len {
            data_vec.push(0.0);
        }
        let data_array: ArrayRef = Arc::new(Float64Array::from(data_vec));

        // Build shape_info array: [rank, dim0, dim1, ..., 0, 0, ...]
        let mut shape_info = Vec::with_capacity(array_len);
        shape_info.push(rank as i64);
        for &dim in shape {
            shape_info.push(dim as i64);
        }
        while shape_info.len() < array_len {
            shape_info.push(0);
        }

        let shape_array: ArrayRef = Arc::new(Int64Array::from(shape_info));

        // Create record batch
        let batch = RecordBatch::try_new(self.schema.clone(), vec![data_array, shape_array])?;

        // Write batch
        self.writer.write(&batch)?;

        Ok(())
    }

    /// Finish writing and close the file
    ///
    /// Must be called to ensure all data is written to disk.
    ///
    /// # Returns
    ///
    /// Ok(()) if successful
    ///
    /// # Errors
    ///
    /// Returns an error if closing fails
    pub fn finish(self) -> Result<()> {
        self.writer.close()?;
        Ok(())
    }
}

/// Parquet reader for tensors
///
/// Reads `DenseND<T>` tensors from Parquet format files.
pub struct ParquetReader {
    file_path: String,
}

impl ParquetReader {
    /// Open a Parquet file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the input file
    ///
    /// # Returns
    ///
    /// A new `ParquetReader` instance
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_string_lossy().to_string();
        Ok(Self { file_path })
    }

    /// Read a tensor from the Parquet file
    ///
    /// Reads the first record batch and reconstructs the tensor.
    ///
    /// # Returns
    ///
    /// The reconstructed `DenseND<f64>` tensor
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No data is found in the file
    /// - The data format is invalid
    /// - Shape reconstruction fails
    pub fn read(&self) -> Result<DenseND<f64>> {
        let file = File::open(&self.file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.build()?;

        // Read first batch
        let batch = reader
            .next()
            .ok_or_else(|| anyhow!("No data found in Parquet file"))??;

        // Extract data array
        let data_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("Invalid data array type"))?;

        // Extract shape_info array
        let shape_info_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("Invalid shape_info array type"))?;

        // First element is rank, next elements are shape dimensions
        let rank = shape_info_array.value(0) as usize;
        let shape: Vec<usize> = (1..=rank)
            .map(|i| shape_info_array.value(i) as usize)
            .collect();

        // Calculate expected data length from shape
        let expected_len: usize = shape.iter().product();

        // Convert data to vector, taking only expected_len elements
        let data: Vec<f64> = data_array.values()[..expected_len].to_vec();

        // Reconstruct tensor
        DenseND::from_vec(data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_parquet_write_read_roundtrip() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_tensor.parquet");

        // Create test tensor
        let original =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Write
        let mut writer = ParquetWriter::new(&path).unwrap();
        writer.write(&original).unwrap();
        writer.finish().unwrap();

        // Read
        let reader = ParquetReader::open(&path).unwrap();
        let loaded = reader.read().unwrap();

        // Verify
        assert_eq!(original.shape(), loaded.shape());

        let orig_view = original.view();
        let loaded_view = loaded.view();

        for i in 0..2 {
            for j in 0..3 {
                let diff: f64 = orig_view[[i, j]] - loaded_view[[i, j]];
                assert!(diff.abs() < 1e-10, "Mismatch at [{}, {}]", i, j);
            }
        }

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_parquet_3d_tensor() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_tensor_3d.parquet");

        // Create 3D test tensor
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let original = DenseND::<f64>::from_vec(data, &[2, 3, 4]).unwrap();

        // Write
        let mut writer = ParquetWriter::new(&path).unwrap();
        writer.write(&original).unwrap();
        writer.finish().unwrap();

        // Read
        let reader = ParquetReader::open(&path).unwrap();
        let loaded = reader.read().unwrap();

        // Verify
        assert_eq!(original.shape(), loaded.shape());
        assert_eq!(original.shape(), &[2, 3, 4]);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_parquet_single_element() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_tensor_single.parquet");

        // Create single element tensor
        let original = DenseND::<f64>::from_vec(vec![42.0], &[1]).unwrap();

        // Write
        let mut writer = ParquetWriter::new(&path).unwrap();
        writer.write(&original).unwrap();
        writer.finish().unwrap();

        // Read
        let reader = ParquetReader::open(&path).unwrap();
        let loaded = reader.read().unwrap();

        // Verify
        assert_eq!(original.shape(), loaded.shape());
        let diff: f64 = original.view()[[0]] - loaded.view()[[0]];
        assert!(diff.abs() < 1e-10);

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
