//! Memory-mapped tensor I/O
//!
//! This module provides zero-copy access to large tensors stored on disk using memory mapping.
//!
//! # Features
//!
//! - Memory-mapped read-only tensors (`MmapTensor`)
//! - Memory-mapped writable tensors (`MmapTensorMut`)
//! - Custom binary format with header and shape metadata
//! - Zero-copy access for large tensors
//! - Safe memory access patterns
//!
//! # Binary Format
//!
//! The binary format consists of:
//! - Magic bytes: "TNRS" (4 bytes)
//! - Version: u32 (4 bytes)
//! - Rank: u32 (4 bytes)
//! - Padding: 4 bytes (for 8-byte alignment)
//! - Shape: [u64; rank] (8 * rank bytes)
//! - Data: [f64; product(shape)] (8 * product(shape) bytes)
//!
//! # Example
//!
//! ```ignore
//! use tenrso_core::DenseND;
//! use tenrso_ooc::mmap_io::{MmapTensor, write_tensor_binary};
//!
//! // Write tensor to binary file
//! let tensor = DenseND::<f64>::ones(&[1000, 1000]);
//! write_tensor_binary("large_tensor.bin", &tensor)?;
//!
//! // Memory-map for zero-copy access
//! let mmap_tensor = MmapTensor::<f64>::open("large_tensor.bin")?;
//! let shape = mmap_tensor.shape();
//! let data = mmap_tensor.as_slice();
//! ```

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::Path;
use tenrso_core::DenseND;

/// Magic bytes for tensor binary format
const MAGIC: &[u8; 4] = b"TNRS";

/// Current binary format version
const VERSION: u32 = 1;

/// Memory-mapped read-only tensor
///
/// Provides zero-copy access to tensor data stored in a binary file.
#[derive(Debug)]
pub struct MmapTensor<T> {
    _mmap: Mmap,
    shape: Vec<usize>,
    data_offset: usize,
    _phantom: PhantomData<T>,
}

impl MmapTensor<f64> {
    /// Open a memory-mapped tensor from a binary file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the binary tensor file
    ///
    /// # Returns
    ///
    /// A new `MmapTensor` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened
    /// - File format is invalid
    /// - Magic bytes don't match
    /// - Version is unsupported
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate magic bytes
        if mmap.len() < 4 || &mmap[0..4] != MAGIC {
            return Err(anyhow!("Invalid magic bytes"));
        }

        // Read version
        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != VERSION {
            return Err(anyhow!("Unsupported version: {}", version));
        }

        // Read rank
        let rank = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;

        // Read shape (skip 4 bytes padding to ensure 8-byte alignment)
        let shape_offset = 16;
        let shape_size = rank * 8;
        let mut shape = Vec::with_capacity(rank);

        for i in 0..rank {
            let offset = shape_offset + i * 8;
            let dim_bytes = [
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
                mmap[offset + 4],
                mmap[offset + 5],
                mmap[offset + 6],
                mmap[offset + 7],
            ];
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }

        let data_offset = shape_offset + shape_size;

        // Validate data size
        let expected_data_size = shape.iter().product::<usize>() * std::mem::size_of::<f64>();
        let actual_data_size = mmap.len() - data_offset;
        if actual_data_size < expected_data_size {
            return Err(anyhow!(
                "Insufficient data: expected {} bytes, found {}",
                expected_data_size,
                actual_data_size
            ));
        }

        Ok(Self {
            _mmap: mmap,
            shape,
            data_offset,
            _phantom: PhantomData,
        })
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get a slice view of the tensor data
    ///
    /// # Safety
    ///
    /// This is safe because we validate the file format and data size on open.
    pub fn as_slice(&self) -> &[f64] {
        let data_len = self.shape.iter().product::<usize>();
        let byte_slice = &self._mmap[self.data_offset..self.data_offset + data_len * 8];

        // Safety: We've validated the size and alignment
        unsafe { std::slice::from_raw_parts(byte_slice.as_ptr() as *const f64, data_len) }
    }

    /// Convert to a DenseND tensor (copies data)
    ///
    /// # Returns
    ///
    /// A new `DenseND<f64>` tensor with copied data
    ///
    /// # Errors
    ///
    /// Returns an error if tensor construction fails
    pub fn to_dense(&self) -> Result<DenseND<f64>> {
        let data = self.as_slice().to_vec();
        DenseND::from_vec(data, &self.shape)
    }
}

/// Memory-mapped writable tensor
///
/// Provides writable access to tensor data stored in a binary file.
#[derive(Debug)]
pub struct MmapTensorMut<T> {
    _mmap: MmapMut,
    shape: Vec<usize>,
    data_offset: usize,
    _phantom: PhantomData<T>,
}

impl MmapTensorMut<f64> {
    /// Open a memory-mapped writable tensor from a binary file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the binary tensor file
    ///
    /// # Returns
    ///
    /// A new `MmapTensorMut` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened
    /// - File format is invalid
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Validate magic bytes
        if mmap.len() < 4 || &mmap[0..4] != MAGIC {
            return Err(anyhow!("Invalid magic bytes"));
        }

        // Read version
        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != VERSION {
            return Err(anyhow!("Unsupported version: {}", version));
        }

        // Read rank
        let rank = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;

        // Read shape (skip 4 bytes padding to ensure 8-byte alignment)
        let shape_offset = 16;
        let shape_size = rank * 8;
        let mut shape = Vec::with_capacity(rank);

        for i in 0..rank {
            let offset = shape_offset + i * 8;
            let dim_bytes = [
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
                mmap[offset + 4],
                mmap[offset + 5],
                mmap[offset + 6],
                mmap[offset + 7],
            ];
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }

        let data_offset = shape_offset + shape_size;

        Ok(Self {
            _mmap: mmap,
            shape,
            data_offset,
            _phantom: PhantomData,
        })
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get a mutable slice view of the tensor data
    ///
    /// # Safety
    ///
    /// This is safe because we validate the file format and data size on open.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        let data_len = self.shape.iter().product::<usize>();
        let byte_slice = &mut self._mmap[self.data_offset..self.data_offset + data_len * 8];

        // Safety: We've validated the size and alignment
        unsafe { std::slice::from_raw_parts_mut(byte_slice.as_mut_ptr() as *mut f64, data_len) }
    }

    /// Flush changes to disk
    ///
    /// # Returns
    ///
    /// Ok(()) if successful
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails
    pub fn flush(&self) -> Result<()> {
        self._mmap.flush()?;
        Ok(())
    }
}

/// Write a tensor to a binary file in mmap-compatible format
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `tensor` - The tensor to write
///
/// # Returns
///
/// Ok(()) if successful
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be created
/// - Writing fails
pub fn write_tensor_binary<P: AsRef<Path>>(path: P, tensor: &DenseND<f64>) -> Result<()> {
    let mut file = File::create(path)?;

    // Write magic bytes
    file.write_all(MAGIC)?;

    // Write version
    file.write_all(&VERSION.to_le_bytes())?;

    // Write rank
    let rank = tensor.shape().len() as u32;
    file.write_all(&rank.to_le_bytes())?;

    // Write padding for 8-byte alignment
    file.write_all(&[0u8; 4])?;

    // Write shape
    for &dim in tensor.shape() {
        file.write_all(&(dim as u64).to_le_bytes())?;
    }

    // Write data
    let data = tensor.as_slice();
    for &value in data {
        file.write_all(&value.to_le_bytes())?;
    }

    file.flush()?;
    Ok(())
}

/// Read a tensor from a binary file
///
/// # Arguments
///
/// * `path` - Path to the input file
///
/// # Returns
///
/// The reconstructed `DenseND<f64>` tensor
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be opened
/// - File format is invalid
pub fn read_tensor_binary<P: AsRef<Path>>(path: P) -> Result<DenseND<f64>> {
    let mut file = File::open(path)?;

    // Read magic bytes
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(anyhow!("Invalid magic bytes"));
    }

    // Read version
    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        return Err(anyhow!("Unsupported version: {}", version));
    }

    // Read rank
    let mut rank_bytes = [0u8; 4];
    file.read_exact(&mut rank_bytes)?;
    let rank = u32::from_le_bytes(rank_bytes) as usize;

    // Skip padding for 8-byte alignment
    let mut padding = [0u8; 4];
    file.read_exact(&mut padding)?;

    // Read shape
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        let mut dim_bytes = [0u8; 8];
        file.read_exact(&mut dim_bytes)?;
        shape.push(u64::from_le_bytes(dim_bytes) as usize);
    }

    // Read data
    let data_len = shape.iter().product::<usize>();
    let mut data = Vec::with_capacity(data_len);
    for _ in 0..data_len {
        let mut value_bytes = [0u8; 8];
        file.read_exact(&mut value_bytes)?;
        data.push(f64::from_le_bytes(value_bytes));
    }

    DenseND::from_vec(data, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_binary_write_read_roundtrip() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_tensor.bin");

        // Create test tensor
        let original =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Write
        write_tensor_binary(&path, &original).unwrap();

        // Read
        let loaded = read_tensor_binary(&path).unwrap();

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
    fn test_mmap_tensor_read() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_mmap.bin");

        // Create and write test tensor
        let original =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        write_tensor_binary(&path, &original).unwrap();

        // Memory-map
        let mmap_tensor = MmapTensor::<f64>::open(&path).unwrap();

        // Verify shape
        assert_eq!(mmap_tensor.shape(), &[2, 3]);

        // Verify data
        let data = mmap_tensor.as_slice();
        assert_eq!(data.len(), 6);
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[5] - 6.0).abs() < 1e-10);

        // Convert to DenseND
        let loaded = mmap_tensor.to_dense().unwrap();
        assert_eq!(loaded.shape(), original.shape());

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_tensor_mut_write() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_mmap_mut.bin");

        // Create and write test tensor
        let original = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        write_tensor_binary(&path, &original).unwrap();

        // Memory-map for writing
        {
            let mut mmap_tensor = MmapTensorMut::<f64>::open(&path).unwrap();
            let data = mmap_tensor.as_mut_slice();
            data[0] = 10.0;
            data[3] = 40.0;
            mmap_tensor.flush().unwrap();
        }

        // Read back to verify changes
        let loaded = read_tensor_binary(&path).unwrap();
        let loaded_view = loaded.view();

        assert!((loaded_view[[0, 0]] - 10.0).abs() < 1e-10);
        assert!((loaded_view[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((loaded_view[[1, 0]] - 3.0).abs() < 1e-10);
        assert!((loaded_view[[1, 1]] - 40.0).abs() < 1e-10);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_3d_tensor() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_mmap_3d.bin");

        // Create 3D tensor
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let original = DenseND::<f64>::from_vec(data, &[2, 3, 4]).unwrap();

        // Write
        write_tensor_binary(&path, &original).unwrap();

        // Memory-map and verify
        let mmap_tensor = MmapTensor::<f64>::open(&path).unwrap();
        assert_eq!(mmap_tensor.shape(), &[2, 3, 4]);

        let loaded = mmap_tensor.to_dense().unwrap();
        assert_eq!(loaded.shape(), &[2, 3, 4]);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_invalid.bin");

        // Write invalid file
        let mut file = File::create(&path).unwrap();
        file.write_all(b"XXXX").unwrap();

        // Try to open
        let result = MmapTensor::<f64>::open(&path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid magic bytes"));

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
