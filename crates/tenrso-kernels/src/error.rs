//! Error types for tensor kernel operations
//!
//! This module provides structured error types for kernel operations,
//! making error handling more robust and informative.

use std::fmt;

/// Error type for tensor kernel operations
#[derive(Debug, Clone, PartialEq)]
pub enum KernelError {
    /// Dimension mismatch between operands
    DimensionMismatch {
        operation: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: String,
    },

    /// Invalid mode/axis specification
    InvalidMode {
        mode: usize,
        max_mode: usize,
        context: String,
    },

    /// Rank mismatch (e.g., different CP ranks in factor matrices)
    RankMismatch {
        operation: String,
        expected_rank: usize,
        actual_rank: usize,
        factor_index: usize,
    },

    /// Empty input not allowed
    EmptyInput {
        operation: String,
        parameter: String,
    },

    /// Invalid tile/block size
    InvalidTileSize {
        operation: String,
        tile_size: usize,
        reason: String,
    },

    /// Shape incompatibility
    IncompatibleShapes {
        operation: String,
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
        reason: String,
    },

    /// Generic operation error with context
    OperationError { operation: String, message: String },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::DimensionMismatch {
                operation,
                expected,
                actual,
                context,
            } => write!(
                f,
                "{}: dimension mismatch - expected {:?}, got {:?}. {}",
                operation, expected, actual, context
            ),

            KernelError::InvalidMode {
                mode,
                max_mode,
                context,
            } => write!(
                f,
                "Invalid mode {}: must be < {}. {}",
                mode, max_mode, context
            ),

            KernelError::RankMismatch {
                operation,
                expected_rank,
                actual_rank,
                factor_index,
            } => write!(
                f,
                "{}: rank mismatch at factor {}: expected rank {}, got {}",
                operation, factor_index, expected_rank, actual_rank
            ),

            KernelError::EmptyInput {
                operation,
                parameter,
            } => write!(
                f,
                "{}: empty input not allowed for parameter '{}'",
                operation, parameter
            ),

            KernelError::InvalidTileSize {
                operation,
                tile_size,
                reason,
            } => write!(
                f,
                "{}: invalid tile size {}: {}",
                operation, tile_size, reason
            ),

            KernelError::IncompatibleShapes {
                operation,
                shape_a,
                shape_b,
                reason,
            } => write!(
                f,
                "{}: incompatible shapes {:?} and {:?}: {}",
                operation, shape_a, shape_b, reason
            ),

            KernelError::OperationError { operation, message } => {
                write!(f, "{}: {}", operation, message)
            }
        }
    }
}

impl std::error::Error for KernelError {}

/// Result type for kernel operations
pub type KernelResult<T> = Result<T, KernelError>;

impl KernelError {
    /// Create a dimension mismatch error
    pub fn dimension_mismatch(
        operation: impl Into<String>,
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: impl Into<String>,
    ) -> Self {
        KernelError::DimensionMismatch {
            operation: operation.into(),
            expected,
            actual,
            context: context.into(),
        }
    }

    /// Create an invalid mode error
    pub fn invalid_mode(mode: usize, max_mode: usize, context: impl Into<String>) -> Self {
        KernelError::InvalidMode {
            mode,
            max_mode,
            context: context.into(),
        }
    }

    /// Create a rank mismatch error
    pub fn rank_mismatch(
        operation: impl Into<String>,
        expected_rank: usize,
        actual_rank: usize,
        factor_index: usize,
    ) -> Self {
        KernelError::RankMismatch {
            operation: operation.into(),
            expected_rank,
            actual_rank,
            factor_index,
        }
    }

    /// Create an empty input error
    pub fn empty_input(operation: impl Into<String>, parameter: impl Into<String>) -> Self {
        KernelError::EmptyInput {
            operation: operation.into(),
            parameter: parameter.into(),
        }
    }

    /// Create an invalid tile size error
    pub fn invalid_tile_size(
        operation: impl Into<String>,
        tile_size: usize,
        reason: impl Into<String>,
    ) -> Self {
        KernelError::InvalidTileSize {
            operation: operation.into(),
            tile_size,
            reason: reason.into(),
        }
    }

    /// Create an incompatible shapes error
    pub fn incompatible_shapes(
        operation: impl Into<String>,
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
        reason: impl Into<String>,
    ) -> Self {
        KernelError::IncompatibleShapes {
            operation: operation.into(),
            shape_a,
            shape_b,
            reason: reason.into(),
        }
    }

    /// Create a generic operation error
    pub fn operation_error(operation: impl Into<String>, message: impl Into<String>) -> Self {
        KernelError::OperationError {
            operation: operation.into(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_display() {
        let err = KernelError::dimension_mismatch(
            "khatri_rao",
            vec![10, 5],
            vec![10, 3],
            "Number of columns must match",
        );

        let msg = format!("{}", err);
        assert!(msg.contains("khatri_rao"));
        assert!(msg.contains("dimension mismatch"));
        assert!(msg.contains("[10, 5]"));
        assert!(msg.contains("[10, 3]"));
    }

    #[test]
    fn test_invalid_mode_display() {
        let err = KernelError::invalid_mode(3, 3, "Tensor has only 3 modes");

        let msg = format!("{}", err);
        assert!(msg.contains("Invalid mode 3"));
        assert!(msg.contains("must be < 3"));
    }

    #[test]
    fn test_rank_mismatch_display() {
        let err = KernelError::rank_mismatch("mttkrp", 5, 3, 2);

        let msg = format!("{}", err);
        assert!(msg.contains("mttkrp"));
        assert!(msg.contains("factor 2"));
        assert!(msg.contains("expected rank 5"));
        assert!(msg.contains("got 3"));
    }

    #[test]
    fn test_empty_input_display() {
        let err = KernelError::empty_input("outer_product", "vectors");

        let msg = format!("{}", err);
        assert!(msg.contains("outer_product"));
        assert!(msg.contains("empty input"));
        assert!(msg.contains("vectors"));
    }

    #[test]
    fn test_invalid_tile_size_display() {
        let err = KernelError::invalid_tile_size("mttkrp_blocked", 0, "must be positive");

        let msg = format!("{}", err);
        assert!(msg.contains("mttkrp_blocked"));
        assert!(msg.contains("invalid tile size 0"));
        assert!(msg.contains("must be positive"));
    }

    #[test]
    fn test_incompatible_shapes_display() {
        let err = KernelError::incompatible_shapes(
            "hadamard",
            vec![2, 3],
            vec![2, 4],
            "Element-wise multiplication requires same shape",
        );

        let msg = format!("{}", err);
        assert!(msg.contains("hadamard"));
        assert!(msg.contains("[2, 3]"));
        assert!(msg.contains("[2, 4]"));
        assert!(msg.contains("Element-wise multiplication"));
    }
}
