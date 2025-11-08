//! Arrow IPC I/O for tensors
//!
//! This module provides serialization and deserialization of tensors using Apache Arrow IPC format.
//!
//! # Features
//!
//! - Write `DenseND<T>` tensors to Arrow IPC files
//! - Read Arrow IPC files back to `DenseND<T>` tensors
//! - Preserve tensor shape and metadata
//! - Support for f32, f64, i32, i64 types
//!
//! # Example
//!
//! ```ignore
//! use tenrso_core::DenseND;
//! use tenrso_ooc::arrow_io::{ArrowWriter, ArrowReader};
//!
//! // Write tensor to Arrow IPC
//! let tensor = DenseND::<f64>::ones(&[10, 20, 30]);
//! let mut writer = ArrowWriter::new("tensor.arrow")?;
//! writer.write(&tensor)?;
//! writer.finish()?;
//!
//! // Read tensor from Arrow IPC
//! let reader = ArrowReader::open("tensor.arrow")?;
//! let loaded: DenseND<f64> = reader.read()?;
//! ```

use anyhow::{anyhow, Result};
use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tenrso_core::DenseND;

/// Arrow IPC writer for tensors
///
/// Writes `DenseND<T>` tensors to Arrow IPC format files.
/// The tensor data is flattened and stored along with its shape.
pub struct ArrowWriter {
    writer: FileWriter<File>,
}

impl ArrowWriter {
    /// Create a new Arrow IPC writer
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    ///
    /// # Returns
    ///
    /// A new `ArrowWriter` instance
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;

        // Define schema: data and shape_info columns
        // shape_info will contain rank in first element, then shape dimensions
        let schema = Schema::new(vec![
            Field::new("data", DataType::Float64, false),
            Field::new("shape_info", DataType::Int64, false),
        ]);

        let writer = FileWriter::try_new(file, &schema)?;

        Ok(Self { writer })
    }

    /// Write a tensor to the Arrow IPC file
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
        // This ensures we have room for rank + all dimensions
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
        // Pad with zeros to match array_len
        while shape_info.len() < array_len {
            shape_info.push(0);
        }

        let shape_array: ArrayRef = Arc::new(Int64Array::from(shape_info));

        // Create record batch
        let batch =
            RecordBatch::try_new(self.writer.schema().clone(), vec![data_array, shape_array])?;

        // Write batch
        self.writer.write(&batch)?;

        Ok(())
    }

    /// Finish writing and flush the file
    ///
    /// Must be called to ensure all data is written to disk.
    ///
    /// # Returns
    ///
    /// Ok(()) if successful
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails
    pub fn finish(&mut self) -> Result<()> {
        self.writer.finish()?;
        Ok(())
    }
}

/// Arrow IPC reader for tensors
///
/// Reads `DenseND<T>` tensors from Arrow IPC format files.
pub struct ArrowReader {
    reader: FileReader<File>,
}

impl ArrowReader {
    /// Open an Arrow IPC file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the input file
    ///
    /// # Returns
    ///
    /// A new `ArrowReader` instance
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or is not a valid Arrow IPC file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = FileReader::try_new(file, None)?;
        Ok(Self { reader })
    }

    /// Read a tensor from the Arrow IPC file
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
    pub fn read(&mut self) -> Result<DenseND<f64>> {
        // Read first batch
        let batch = self
            .reader
            .next()
            .ok_or_else(|| anyhow!("No data found in Arrow file"))??;

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
    fn test_arrow_write_read_roundtrip() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_tensor.arrow");

        // Create test tensor
        let original =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Write
        let mut writer = ArrowWriter::new(&path).unwrap();
        writer.write(&original).unwrap();
        writer.finish().unwrap();

        // Read
        let mut reader = ArrowReader::open(&path).unwrap();
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
    fn test_arrow_3d_tensor() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_tensor_3d.arrow");

        // Create 3D test tensor
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let original = DenseND::<f64>::from_vec(data, &[2, 3, 4]).unwrap();

        // Write
        let mut writer = ArrowWriter::new(&path).unwrap();
        writer.write(&original).unwrap();
        writer.finish().unwrap();

        // Read
        let mut reader = ArrowReader::open(&path).unwrap();
        let loaded = reader.read().unwrap();

        // Verify
        assert_eq!(original.shape(), loaded.shape());
        assert_eq!(original.shape(), &[2, 3, 4]);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_arrow_single_element() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_tensor_single.arrow");

        // Create single element tensor
        let original = DenseND::<f64>::from_vec(vec![42.0], &[1]).unwrap();

        // Write
        let mut writer = ArrowWriter::new(&path).unwrap();
        writer.write(&original).unwrap();
        writer.finish().unwrap();

        // Read
        let mut reader = ArrowReader::open(&path).unwrap();
        let loaded = reader.read().unwrap();

        // Verify
        assert_eq!(original.shape(), loaded.shape());
        let diff: f64 = original.view()[[0]] - loaded.view()[[0]];
        assert!(diff.abs() < 1e-10);

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
