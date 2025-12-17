//! Unified error types for sparse tensor operations
//!
//! This module provides a centralized error handling system for all sparse tensor
//! formats and operations in tenrso-sparse.
//!
//! # Design
//!
//! - **`SparseError`**: Top-level enum covering all error cases
//! - **Format-specific errors**: COO, CSR, CSC, BCSR, CSF, HiCOO
//! - **Operation errors**: SpMV, SpMM, SpSpMM, masked operations
//! - **Conversion errors**: Format conversions, dense/sparse
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::error::{SparseError, ValidationError};
//!
//! fn validate_shape(shape: &[usize]) -> Result<(), SparseError> {
//!     if shape.is_empty() {
//!         return Err(SparseError::Validation(ValidationError::EmptyShape));
//!     }
//!     if shape.contains(&0) {
//!         return Err(SparseError::Validation(ValidationError::ZeroInShape));
//!     }
//!     Ok(())
//! }
//! ```

use thiserror::Error;

/// Top-level error type for all sparse tensor operations
#[derive(Error, Debug)]
pub enum SparseError {
    /// Validation errors (shape, indices, values)
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    /// Shape mismatch errors
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(#[from] ShapeMismatchError),

    /// Format conversion errors
    #[error("Conversion error: {0}")]
    Conversion(#[from] ConversionError),

    /// Sparse operation errors
    #[error("Operation error: {0}")]
    Operation(#[from] OperationError),

    /// Index errors
    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    /// CSR format errors
    #[error("CSR error: {0}")]
    Csr(#[from] crate::csr::CsrError),

    /// CSC format errors
    #[error("CSC error: {0}")]
    Csc(#[from] crate::csc::CscError),

    /// COO format errors
    #[error("COO error: {0}")]
    Coo(#[from] crate::coo::CooError),

    /// Generic error with message
    #[error("{0}")]
    Other(String),
}

/// Validation errors for tensor properties
#[derive(Error, Debug, Clone)]
pub enum ValidationError {
    #[error("Shape cannot be empty")]
    EmptyShape,

    #[error("Shape cannot contain zeros")]
    ZeroInShape,

    #[error("Shape dimensions must match: expected {expected}, got {got}")]
    ShapeDimsMismatch { expected: usize, got: usize },

    #[error("Indices and values must have same length: {indices} indices vs {values} values")]
    LengthMismatch { indices: usize, values: usize },

    #[error("Invalid block shape: {reason}")]
    InvalidBlockShape { reason: String },

    #[error("Invalid mode order: {reason}")]
    InvalidModeOrder { reason: String },

    #[error("Empty tensor: {reason}")]
    EmptyTensor { reason: String },

    #[error("Invalid pointer array: {reason}")]
    InvalidPointers { reason: String },

    #[error("Unsorted indices at position {position}")]
    UnsortedIndices { position: usize },

    #[error("Duplicate indices at position {position}")]
    DuplicateIndices { position: usize },
}

/// Shape mismatch errors for tensor operations
#[derive(Error, Debug, Clone)]
pub enum ShapeMismatchError {
    #[error("Matrix multiplication dimension mismatch: ({m1}×{n1}) × ({m2}×{n2})")]
    MatMul {
        m1: usize,
        n1: usize,
        m2: usize,
        n2: usize,
    },

    #[error("Matrix-vector multiplication dimension mismatch: ({m}×{n}) × ({v},)")]
    MatVec { m: usize, n: usize, v: usize },

    #[error("Tensor shape mismatch: expected {expected:?}, got {got:?}")]
    Tensor {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Block shape mismatch: tensor shape {tensor_shape:?} incompatible with block shape {block_shape:?}")]
    Block {
        tensor_shape: Vec<usize>,
        block_shape: Vec<usize>,
    },

    #[error("Mode count mismatch: expected {expected}, got {got}")]
    ModeCount { expected: usize, got: usize },
}

/// Format conversion errors
#[derive(Error, Debug, Clone)]
pub enum ConversionError {
    #[error("Cannot convert from {from} to {to}: {reason}")]
    Incompatible {
        from: String,
        to: String,
        reason: String,
    },

    #[error("Conversion failed: {reason}")]
    Failed { reason: String },

    #[error("Unsupported conversion from {from} to {to}")]
    Unsupported { from: String, to: String },

    #[error("Dimension mismatch during conversion: expected {expected}D, got {got}D")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Sparse operation errors
#[derive(Error, Debug, Clone)]
pub enum OperationError {
    #[error("Operation not supported for format {format}: {operation}")]
    Unsupported { format: String, operation: String },

    #[error("Operation failed: {reason}")]
    Failed { reason: String },

    #[error("Numerical error in operation {operation}: {reason}")]
    Numerical { operation: String, reason: String },

    #[error("Buffer overflow in operation {operation}")]
    BufferOverflow { operation: String },

    #[error("Invalid mask: {reason}")]
    InvalidMask { reason: String },
}

/// Index errors
#[derive(Error, Debug, Clone)]
pub enum IndexError {
    #[error("Index out of bounds: index {index:?} exceeds shape {shape:?}")]
    OutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    #[error("Invalid index: {reason}")]
    Invalid { reason: String },

    #[error("Index dimension mismatch: expected {expected}D, got {got}D")]
    DimensionMismatch { expected: usize, got: usize },
}

// Note: Conversion implementations from format-specific errors can be added later
// For now, we use the new unified error types for new code

/// Result type alias for sparse tensor operations
pub type SparseResult<T> = Result<T, SparseError>;

// Convenience constructors for common error patterns
impl SparseError {
    /// Create an index out of bounds error
    pub fn index_out_of_bounds(index: Vec<usize>, shape: Vec<usize>) -> Self {
        SparseError::Index(IndexError::OutOfBounds { index, shape })
    }

    /// Create a validation error with a message
    pub fn validation(msg: &str) -> Self {
        SparseError::Other(msg.to_string())
    }

    /// Create a conversion error with a message
    pub fn conversion(msg: String) -> Self {
        SparseError::Conversion(ConversionError::Failed { reason: msg })
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: Vec<usize>, got: Vec<usize>) -> Self {
        SparseError::ShapeMismatch(ShapeMismatchError::Tensor { expected, got })
    }

    /// Create an operation error with a message
    pub fn operation(msg: &str) -> Self {
        SparseError::Operation(OperationError::Failed {
            reason: msg.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error() {
        let err = ValidationError::EmptyShape;
        assert_eq!(err.to_string(), "Shape cannot be empty");
    }

    #[test]
    fn test_shape_mismatch_error() {
        let err = ShapeMismatchError::MatMul {
            m1: 3,
            n1: 4,
            m2: 5,
            n2: 6,
        };
        assert_eq!(
            err.to_string(),
            "Matrix multiplication dimension mismatch: (3×4) × (5×6)"
        );
    }

    #[test]
    fn test_sparse_error_from_validation() {
        let err: SparseError = ValidationError::ZeroInShape.into();
        assert!(matches!(err, SparseError::Validation(_)));
    }

    #[test]
    fn test_index_error() {
        let err = IndexError::OutOfBounds {
            index: vec![1, 2, 3],
            shape: vec![1, 2, 2],
        };
        assert_eq!(
            err.to_string(),
            "Index out of bounds: index [1, 2, 3] exceeds shape [1, 2, 2]"
        );
    }
}
