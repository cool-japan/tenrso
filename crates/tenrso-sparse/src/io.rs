//! Matrix Market I/O format support
//!
//! This module provides reading and writing sparse matrices in the
//! [Matrix Market](https://math.nist.gov/MatrixMarket/) exchange format,
//! which is a standard ASCII format for representing sparse matrices.
//!
//! # Format
//!
//! The Matrix Market format consists of:
//! - A header line: `%%MatrixMarket matrix coordinate real general`
//! - Optional comment lines starting with `%`
//! - Size line: `nrows ncols nnz`
//! - Data lines: `row col value` (1-indexed)
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{CooTensor, io};
//! use std::io::Cursor;
//!
//! // Write a sparse matrix to Matrix Market format
//! let indices = vec![vec![0, 0], vec![1, 1], vec![2, 0]];
//! let values = vec![1.0, 2.0, 3.0];
//! let shape = vec![3, 3];
//! let coo = CooTensor::new(indices, values, shape).unwrap();
//!
//! let mut output = Vec::new();
//! io::write_matrix_market(&coo, &mut output).unwrap();
//!
//! // Read it back
//! let input = Cursor::new(output);
//! let coo_back = io::read_matrix_market::<f64>(input).unwrap();
//! assert_eq!(coo_back.nnz(), 3);
//! ```

use crate::{CooTensor, SparseError, SparseResult};
use scirs2_core::numeric::Float;
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;

/// Matrix Market format variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMarketFormat {
    /// Coordinate format (COO)
    Coordinate,
    /// Array format (dense)
    Array,
}

/// Matrix Market data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMarketDataType {
    /// Real numbers
    Real,
    /// Complex numbers
    Complex,
    /// Integer numbers
    Integer,
    /// Pattern (no values, only structure)
    Pattern,
}

/// Matrix Market symmetry type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMarketSymmetry {
    /// General (no symmetry)
    General,
    /// Symmetric
    Symmetric,
    /// SkewSymmetric
    SkewSymmetric,
    /// Hermitian
    Hermitian,
}

/// Matrix Market header information
#[derive(Debug, Clone)]
pub struct MatrixMarketHeader {
    /// Format (coordinate or array)
    pub format: MatrixMarketFormat,
    /// Data type
    pub data_type: MatrixMarketDataType,
    /// Symmetry type
    pub symmetry: MatrixMarketSymmetry,
}

impl MatrixMarketHeader {
    /// Parse header from first line
    pub fn parse(line: &str) -> SparseResult<Self> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 5 || parts[0] != "%%MatrixMarket" || parts[1] != "matrix" {
            return Err(SparseError::conversion(
                "Invalid Matrix Market header format".to_string(),
            ));
        }

        let format = match parts[2] {
            "coordinate" => MatrixMarketFormat::Coordinate,
            "array" => MatrixMarketFormat::Array,
            _ => {
                return Err(SparseError::conversion(format!(
                    "Unknown format: {}",
                    parts[2]
                )))
            }
        };

        let data_type = match parts[3] {
            "real" => MatrixMarketDataType::Real,
            "complex" => MatrixMarketDataType::Complex,
            "integer" => MatrixMarketDataType::Integer,
            "pattern" => MatrixMarketDataType::Pattern,
            _ => {
                return Err(SparseError::conversion(format!(
                    "Unknown data type: {}",
                    parts[3]
                )))
            }
        };

        let symmetry = match parts[4] {
            "general" => MatrixMarketSymmetry::General,
            "symmetric" => MatrixMarketSymmetry::Symmetric,
            "skew-symmetric" => MatrixMarketSymmetry::SkewSymmetric,
            "hermitian" => MatrixMarketSymmetry::Hermitian,
            _ => {
                return Err(SparseError::conversion(format!(
                    "Unknown symmetry: {}",
                    parts[4]
                )))
            }
        };

        Ok(Self {
            format,
            data_type,
            symmetry,
        })
    }

    /// Convert to header line string
    pub fn header_string(&self) -> String {
        let format_str = match self.format {
            MatrixMarketFormat::Coordinate => "coordinate",
            MatrixMarketFormat::Array => "array",
        };

        let data_type_str = match self.data_type {
            MatrixMarketDataType::Real => "real",
            MatrixMarketDataType::Complex => "complex",
            MatrixMarketDataType::Integer => "integer",
            MatrixMarketDataType::Pattern => "pattern",
        };

        let symmetry_str = match self.symmetry {
            MatrixMarketSymmetry::General => "general",
            MatrixMarketSymmetry::Symmetric => "symmetric",
            MatrixMarketSymmetry::SkewSymmetric => "skew-symmetric",
            MatrixMarketSymmetry::Hermitian => "hermitian",
        };

        format!(
            "%%MatrixMarket matrix {} {} {}",
            format_str, data_type_str, symmetry_str
        )
    }
}

/// Read a sparse matrix from Matrix Market format
///
/// # Complexity
///
/// O(nnz) for reading and parsing
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::io::read_matrix_market;
/// use std::io::Cursor;
///
/// let data = b"%%MatrixMarket matrix coordinate real general
/// 3 3 3
/// 1 1 1.0
/// 2 2 2.0
/// 3 1 3.0
/// ";
/// let reader = Cursor::new(data);
/// let coo = read_matrix_market::<f64>(reader).unwrap();
/// assert_eq!(coo.nnz(), 3);
/// assert_eq!(coo.shape(), &[3, 3]);
/// ```
pub fn read_matrix_market<T: Float + FromStr>(
    reader: impl std::io::Read,
) -> SparseResult<CooTensor<T>> {
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    // Read header
    reader
        .read_line(&mut line)
        .map_err(|e| SparseError::conversion(format!("Failed to read header: {}", e)))?;
    let header = MatrixMarketHeader::parse(line.trim())?;

    if header.format != MatrixMarketFormat::Coordinate {
        return Err(SparseError::conversion(
            "Only coordinate format is supported".to_string(),
        ));
    }

    // Skip comment lines
    loop {
        line.clear();
        reader
            .read_line(&mut line)
            .map_err(|e| SparseError::conversion(format!("Failed to read line: {}", e)))?;
        if !line.trim().starts_with('%') {
            break;
        }
    }

    // Parse size line
    let size_parts: Vec<&str> = line.split_whitespace().collect();
    if size_parts.len() != 3 {
        return Err(SparseError::conversion(
            "Invalid size line format".to_string(),
        ));
    }

    let nrows: usize = size_parts[0]
        .parse()
        .map_err(|_| SparseError::conversion("Invalid nrows".to_string()))?;
    let ncols: usize = size_parts[1]
        .parse()
        .map_err(|_| SparseError::conversion("Invalid ncols".to_string()))?;
    let nnz: usize = size_parts[2]
        .parse()
        .map_err(|_| SparseError::conversion("Invalid nnz".to_string()))?;

    let mut indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    // Read data lines
    for _ in 0..nnz {
        line.clear();
        reader
            .read_line(&mut line)
            .map_err(|e| SparseError::conversion(format!("Failed to read data line: {}", e)))?;

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(SparseError::conversion(
                "Invalid data line format".to_string(),
            ));
        }

        // Matrix Market uses 1-based indexing
        let row: usize = parts[0]
            .parse::<usize>()
            .map_err(|_| SparseError::conversion("Invalid row index".to_string()))?
            - 1;
        let col: usize = parts[1]
            .parse::<usize>()
            .map_err(|_| SparseError::conversion("Invalid column index".to_string()))?
            - 1;

        let value = if header.data_type == MatrixMarketDataType::Pattern {
            T::one()
        } else if parts.len() < 3 {
            return Err(SparseError::conversion(
                "Missing value in data line".to_string(),
            ));
        } else {
            parts[2]
                .parse::<T>()
                .map_err(|_| SparseError::conversion("Invalid value".to_string()))?
        };

        indices.push(vec![row, col]);
        values.push(value);

        // Handle symmetry
        if header.symmetry == MatrixMarketSymmetry::Symmetric && row != col {
            indices.push(vec![col, row]);
            values.push(value);
        } else if header.symmetry == MatrixMarketSymmetry::SkewSymmetric && row != col {
            indices.push(vec![col, row]);
            values.push(-value);
        }
    }

    CooTensor::new(indices, values, vec![nrows, ncols]).map_err(|e| e.into())
}

/// Write a sparse matrix to Matrix Market format
///
/// # Complexity
///
/// O(nnz) for writing
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CooTensor, io};
///
/// let indices = vec![vec![0, 0], vec![1, 1], vec![2, 0]];
/// let values = vec![1.0, 2.0, 3.0];
/// let shape = vec![3, 3];
/// let coo = CooTensor::new(indices, values, shape).unwrap();
///
/// let mut output = Vec::new();
/// io::write_matrix_market(&coo, &mut output).unwrap();
///
/// let output_str = String::from_utf8(output).unwrap();
/// assert!(output_str.contains("%%MatrixMarket"));
/// assert!(output_str.contains("3 3 3"));
/// ```
pub fn write_matrix_market<T: Float + std::fmt::Display>(
    coo: &CooTensor<T>,
    writer: &mut impl Write,
) -> SparseResult<()> {
    if coo.rank() != 2 {
        return Err(SparseError::validation(&format!(
            "Only 2D matrices supported, got {}D",
            coo.rank()
        )));
    }

    // Write header
    let header = MatrixMarketHeader {
        format: MatrixMarketFormat::Coordinate,
        data_type: MatrixMarketDataType::Real,
        symmetry: MatrixMarketSymmetry::General,
    };

    writeln!(writer, "{}", header.header_string())
        .map_err(|e| SparseError::conversion(format!("Failed to write header: {}", e)))?;

    // Write size line
    let nrows = coo.shape()[0];
    let ncols = coo.shape()[1];
    let nnz = coo.nnz();
    writeln!(writer, "{} {} {}", nrows, ncols, nnz)
        .map_err(|e| SparseError::conversion(format!("Failed to write size: {}", e)))?;

    // Write data lines (1-indexed)
    for (idx, val) in coo.indices().iter().zip(coo.values().iter()) {
        let row = idx[0] + 1; // Convert to 1-based
        let col = idx[1] + 1; // Convert to 1-based
        writeln!(writer, "{} {} {}", row, col, val)
            .map_err(|e| SparseError::conversion(format!("Failed to write data: {}", e)))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_header_parse() {
        let line = "%%MatrixMarket matrix coordinate real general";
        let header = MatrixMarketHeader::parse(line).unwrap();
        assert_eq!(header.format, MatrixMarketFormat::Coordinate);
        assert_eq!(header.data_type, MatrixMarketDataType::Real);
        assert_eq!(header.symmetry, MatrixMarketSymmetry::General);
    }

    #[test]
    fn test_header_to_string() {
        let header = MatrixMarketHeader {
            format: MatrixMarketFormat::Coordinate,
            data_type: MatrixMarketDataType::Real,
            symmetry: MatrixMarketSymmetry::Symmetric,
        };
        let s = header.header_string();
        assert_eq!(s, "%%MatrixMarket matrix coordinate real symmetric");
    }

    #[test]
    fn test_read_simple_matrix() {
        let data = b"%%MatrixMarket matrix coordinate real general
3 3 3
1 1 1.0
2 2 2.0
3 1 3.0
";
        let reader = Cursor::new(data);
        let coo = read_matrix_market::<f64>(reader).unwrap();
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.shape(), &[3, 3]);
    }

    #[test]
    fn test_read_with_comments() {
        let data = b"%%MatrixMarket matrix coordinate real general
% This is a comment
% Another comment
3 3 2
1 1 1.5
2 2 2.5
";
        let reader = Cursor::new(data);
        let coo = read_matrix_market::<f64>(reader).unwrap();
        assert_eq!(coo.nnz(), 2);
    }

    #[test]
    fn test_read_symmetric_matrix() {
        let data = b"%%MatrixMarket matrix coordinate real symmetric
3 3 2
1 1 1.0
2 1 2.0
";
        let reader = Cursor::new(data);
        let coo = read_matrix_market::<f64>(reader).unwrap();
        // Should expand symmetric entry
        assert_eq!(coo.nnz(), 3); // 1 diagonal + 2 off-diagonal (2,1) and (1,2)
    }

    #[test]
    fn test_read_pattern_matrix() {
        let data = b"%%MatrixMarket matrix coordinate pattern general
3 3 2
1 1
2 2
";
        let reader = Cursor::new(data);
        let coo = read_matrix_market::<f64>(reader).unwrap();
        assert_eq!(coo.nnz(), 2);
        // Pattern matrices have value 1.0
        assert_eq!(coo.values()[0], 1.0);
    }

    #[test]
    fn test_write_matrix() {
        let indices = vec![vec![0, 0], vec![1, 1], vec![2, 0]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 3];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let mut output = Vec::new();
        write_matrix_market(&coo, &mut output).unwrap();

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("%%MatrixMarket matrix coordinate real general"));
        assert!(output_str.contains("3 3 3"));
        assert!(output_str.contains("1 1 1"));
        assert!(output_str.contains("2 2 2"));
        assert!(output_str.contains("3 1 3"));
    }

    #[test]
    fn test_roundtrip() {
        let indices = vec![vec![0, 0], vec![1, 1], vec![0, 2]];
        let values = vec![1.5, 2.5, 3.5];
        let shape = vec![2, 3];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let mut buffer = Vec::new();
        write_matrix_market(&coo, &mut buffer).unwrap();

        let reader = Cursor::new(buffer);
        let coo_back = read_matrix_market::<f64>(reader).unwrap();

        assert_eq!(coo.nnz(), coo_back.nnz());
        assert_eq!(coo.shape(), coo_back.shape());
    }

    #[test]
    fn test_invalid_header() {
        let data = b"Invalid header
3 3 2
1 1 1.0
";
        let reader = Cursor::new(data);
        assert!(read_matrix_market::<f64>(reader).is_err());
    }

    #[test]
    fn test_3d_matrix_write_error() {
        let indices = vec![vec![0, 0, 0], vec![1, 1, 1]];
        let values = vec![1.0, 2.0];
        let shape = vec![2, 2, 2];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let mut output = Vec::new();
        assert!(write_matrix_market(&coo, &mut output).is_err());
    }
}
