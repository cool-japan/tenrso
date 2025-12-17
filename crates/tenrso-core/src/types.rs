//! Core type definitions for TenRSo tensors.
//!
//! This module defines the fundamental types used throughout the TenRSo stack:
//!
//! - Type aliases for tensor dimensions ([`Axis`], [`Rank`], [`Shape`])
//! - Axis metadata ([`AxisMeta`]) for symbolic axis naming
//! - Tensor representations ([`TensorRepr`]) supporting Dense/Sparse/LowRank
//! - Unified tensor handle ([`TensorHandle`]) for polymorphic tensor operations
//!
//! # Examples
//!
//! Creating and manipulating tensor metadata:
//!
//! ```
//! use tenrso_core::{AxisMeta, TensorHandle, DenseND};
//!
//! // Create axis metadata
//! let axes = vec![
//!     AxisMeta::new("batch", 32),
//!     AxisMeta::new("features", 128),
//! ];
//!
//! // Create a tensor with metadata
//! let tensor = DenseND::<f64>::zeros(&[32, 128]);
//! let handle = TensorHandle::from_dense(tensor, axes);
//!
//! assert_eq!(handle.rank(), 2);
//! assert_eq!(handle.axes[0].name, "batch");
//! ```

use scirs2_core::numeric::Num;
use smallvec::SmallVec;

// Re-export the actual DenseND implementation
pub use crate::dense::DenseND;

/// Type alias for tensor axis index.
///
/// Used to identify specific dimensions in multi-dimensional tensors.
/// Zero-indexed (0 is the first axis).
///
/// # Examples
///
/// ```
/// use tenrso_core::{Axis, DenseND};
///
/// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
/// let mode: Axis = 1; // Second axis (size 3)
/// let unfolded = tensor.unfold(mode).unwrap();
/// assert_eq!(unfolded.shape(), &[3, 8]);
/// ```
pub type Axis = usize;

/// Type alias for tensor rank (number of dimensions).
///
/// Represents the dimensionality of a tensor (e.g., 2 for matrices, 3 for 3D tensors).
///
/// # Examples
///
/// ```
/// use tenrso_core::{Rank, DenseND};
///
/// let matrix = DenseND::<f64>::zeros(&[2, 3]);
/// let rank: Rank = matrix.rank();
/// assert_eq!(rank, 2);
///
/// let tensor_3d = DenseND::<f64>::zeros(&[2, 3, 4]);
/// assert_eq!(tensor_3d.rank(), 3);
/// ```
pub type Rank = usize;

/// Shape type using SmallVec to avoid heap allocation for common cases.
///
/// Optimized for tensors with up to 6 dimensions (covers most use cases).
/// Automatically falls back to heap allocation for higher-rank tensors.
///
/// # Examples
///
/// ```
/// use tenrso_core::{Shape, DenseND, TensorHandle};
///
/// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
/// let handle = TensorHandle::from_dense_auto(tensor);
///
/// // Get shape as SmallVec
/// let shape: Shape = handle.shape();
/// assert_eq!(shape.len(), 3);
/// assert_eq!(&shape[..], &[2, 3, 4]);
/// ```
pub type Shape = SmallVec<[usize; 6]>;

/// Padding mode for array padding operations.
///
/// Defines how values are padded at the edges of arrays.
///
/// # Modes
///
/// - `Constant`: Pads with a constant value
/// - `Edge`: Pads with the edge values of the array
/// - `Reflect`: Pads with the reflection of values at the edge (mirrored)
/// - `Wrap`: Pads with wrap-around (periodic boundary conditions)
///
/// # Examples
///
/// ```
/// use tenrso_core::{DenseND, PadMode};
///
/// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
///
/// // Different padding modes
/// let const_pad = tensor.pad(&[(1, 1)], PadMode::Constant, 0.0).unwrap();
/// let edge_pad = tensor.pad(&[(1, 1)], PadMode::Edge, 0.0).unwrap();
/// let reflect_pad = tensor.pad(&[(1, 1)], PadMode::Reflect, 0.0).unwrap();
/// let wrap_pad = tensor.pad(&[(1, 1)], PadMode::Wrap, 0.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Pad with a constant value
    Constant,
    /// Pad with edge values
    Edge,
    /// Pad with reflection of the array
    Reflect,
    /// Pad with wrap-around (periodic)
    Wrap,
}

/// Metadata for a single tensor axis.
///
/// Provides symbolic naming for tensor dimensions, making code more readable
/// and enabling better debugging output.
///
/// # Examples
///
/// ```
/// use tenrso_core::AxisMeta;
///
/// // Create axis metadata for a batch dimension
/// let batch_axis = AxisMeta::new("batch", 32);
/// assert_eq!(batch_axis.name, "batch");
/// assert_eq!(batch_axis.size, 32);
///
/// // Create metadata for multiple axes
/// let axes = vec![
///     AxisMeta::new("height", 224),
///     AxisMeta::new("width", 224),
///     AxisMeta::new("channels", 3),
/// ];
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AxisMeta {
    /// Symbolic name for this axis (e.g., "batch", "time", "features")
    pub name: String,
    /// Size of this dimension (number of elements along this axis)
    pub size: usize,
}

impl AxisMeta {
    /// Create new axis metadata.
    ///
    /// # Arguments
    ///
    /// * `name` - Symbolic name for the axis (e.g., "batch", "height")
    /// * `size` - Number of elements along this axis
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::AxisMeta;
    ///
    /// let axis = AxisMeta::new("time_steps", 100);
    /// assert_eq!(axis.name, "time_steps");
    /// assert_eq!(axis.size, 100);
    /// ```
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }
}

/// Sparse N-dimensional tensor (placeholder).
///
/// This type will be fully implemented in the `tenrso-sparse` crate.
/// It will support various sparse formats: COO, CSR, BCSR, CSF, and HiCOO.
///
/// # Future Usage
///
/// ```ignore
/// // Will be available in future releases
/// use tenrso_core::SparseND;
///
/// let sparse = SparseND::from_coo(indices, values, shape);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseND<T> {
    _phantom: std::marker::PhantomData<T>,
}

/// Low-rank tensor representation (CP/Tucker/TT).
///
/// This type will be fully implemented in the `tenrso-decomp` crate.
/// It will support various decomposition formats: CP (CANDECOMP/PARAFAC),
/// Tucker, and Tensor-Train (TT).
///
/// # Future Usage
///
/// ```ignore
/// // Will be available in future releases
/// use tenrso_core::LowRank;
///
/// let cp = LowRank::cp(factors, rank);
/// let tucker = LowRank::tucker(core, factors);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LowRank<T> {
    _phantom: std::marker::PhantomData<T>,
}

/// Unified tensor representation supporting multiple storage formats.
///
/// This enum allows tensors to be stored in different representations
/// (dense, sparse, low-rank) while providing a uniform API through [`TensorHandle`].
///
/// # Variants
///
/// - **Dense:** Standard dense tensor storage (all elements in memory)
/// - **Sparse:** Sparse tensor storage (only non-zero elements stored)
/// - **LowRank:** Low-rank decomposition (factors stored separately)
///
/// # Examples
///
/// ```
/// use tenrso_core::{TensorRepr, DenseND};
///
/// // Create a dense representation
/// let dense = DenseND::<f64>::zeros(&[10, 20]);
/// let repr = TensorRepr::Dense(dense);
///
/// match repr {
///     TensorRepr::Dense(d) => println!("Dense tensor with shape {:?}", d.shape()),
///     TensorRepr::Sparse(_) => println!("Sparse tensor"),
///     TensorRepr::LowRank(_) => println!("Low-rank tensor"),
/// }
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "T: serde::Serialize")))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "T: serde::Deserialize<'de>"))
)]
pub enum TensorRepr<T>
where
    T: Clone + Num,
{
    /// Dense tensor storage - all elements stored in contiguous memory
    Dense(DenseND<T>),
    /// Sparse tensor storage - only non-zero elements stored (COO/CSR/BCSR/CSF/HiCOO)
    Sparse(SparseND<T>),
    /// Low-rank decomposition - tensor represented as factorized form (CP/Tucker/TT)
    LowRank(LowRank<T>),
}

impl<T: std::fmt::Debug + Clone + Num> std::fmt::Debug for TensorRepr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(d) => f.debug_tuple("Dense").field(d).finish(),
            Self::Sparse(_) => f.debug_tuple("Sparse").field(&"<sparse>").finish(),
            Self::LowRank(_) => f.debug_tuple("LowRank").field(&"<low-rank>").finish(),
        }
    }
}

/// Main tensor handle unifying all representations.
///
/// `TensorHandle` provides a unified interface for working with tensors regardless
/// of their internal representation (dense, sparse, or low-rank). This enables
/// polymorphic tensor operations and automatic representation selection.
///
/// # Design Goals
///
/// - **Unified API:** Same interface for dense, sparse, and low-rank tensors
/// - **Metadata Tracking:** Axis names and sizes for better debugging
/// - **Type Safety:** Compile-time guarantees about tensor properties
/// - **Zero-Cost Abstraction:** No runtime overhead when representation is known
///
/// # Examples
///
/// ## Creating from Dense Tensors
///
/// ```
/// use tenrso_core::{DenseND, TensorHandle, AxisMeta};
///
/// // With explicit axis metadata
/// let tensor = DenseND::<f64>::zeros(&[32, 128]);
/// let axes = vec![
///     AxisMeta::new("batch", 32),
///     AxisMeta::new("features", 128),
/// ];
/// let handle = TensorHandle::from_dense(tensor, axes);
///
/// assert_eq!(handle.rank(), 2);
/// assert_eq!(handle.axes[0].name, "batch");
/// ```
///
/// ## Automatic Axis Naming
///
/// ```
/// use tenrso_core::{DenseND, TensorHandle};
///
/// let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
/// let handle = TensorHandle::from_dense_auto(tensor);
///
/// // Axes are named "axis_0", "axis_1", "axis_2"
/// assert_eq!(handle.axes[0].name, "axis_0");
/// assert_eq!(handle.axes[1].name, "axis_1");
/// assert_eq!(handle.axes[2].name, "axis_2");
/// ```
///
/// ## Querying Tensor Properties
///
/// ```
/// use tenrso_core::{DenseND, TensorHandle};
///
/// let tensor = DenseND::<f64>::zeros(&[10, 20, 30]);
/// let handle = TensorHandle::from_dense_auto(tensor);
///
/// // Get rank and shape
/// assert_eq!(handle.rank(), 3);
/// assert_eq!(handle.shape().as_slice(), &[10, 20, 30]);
///
/// // Access the underlying dense tensor
/// if let Some(dense) = handle.as_dense() {
///     assert_eq!(dense.len(), 6000);
/// }
/// ```
///
/// ## Converting Between Representations
///
/// ```
/// use tenrso_core::{DenseND, TensorHandle};
///
/// let tensor = DenseND::<f64>::random_uniform(&[5, 5], 0.0, 1.0);
/// let handle = TensorHandle::from_dense_auto(tensor);
///
/// // Convert to dense (no-op if already dense)
/// let dense = handle.to_dense().unwrap();
/// assert_eq!(dense.shape(), &[5, 5]);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "T: serde::Serialize")))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "T: serde::Deserialize<'de>"))
)]
pub struct TensorHandle<T>
where
    T: Clone + Num,
{
    /// Internal representation (Dense/Sparse/LowRank)
    pub repr: TensorRepr<T>,
    /// Axis metadata with symbolic names and sizes
    pub axes: Vec<AxisMeta>,
}

impl<T: std::fmt::Debug + Clone + Num> std::fmt::Debug for TensorHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorHandle")
            .field("repr", &self.repr)
            .field("axes", &self.axes)
            .finish()
    }
}

impl<T> TensorHandle<T>
where
    T: Clone + Num,
{
    /// Create a tensor handle from a dense tensor
    ///
    /// # Arguments
    ///
    /// * `dense` - The dense tensor
    /// * `axes` - Axis metadata (must match tensor shape)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, TensorHandle, AxisMeta};
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3]);
    /// let axes = vec![
    ///     AxisMeta::new("rows", 2),
    ///     AxisMeta::new("cols", 3),
    /// ];
    /// let handle = TensorHandle::from_dense(tensor, axes);
    /// assert_eq!(handle.rank(), 2);
    /// ```
    pub fn from_dense(dense: DenseND<T>, axes: Vec<AxisMeta>) -> Self {
        assert_eq!(
            dense.shape().len(),
            axes.len(),
            "Axis metadata length must match tensor rank"
        );
        for (i, axis) in axes.iter().enumerate() {
            assert_eq!(
                axis.size,
                dense.shape()[i],
                "Axis {} size mismatch: expected {}, got {}",
                i,
                dense.shape()[i],
                axis.size
            );
        }
        Self {
            repr: TensorRepr::Dense(dense),
            axes,
        }
    }

    /// Create a tensor handle from a dense tensor with automatic axis naming
    ///
    /// Axes are named "axis_0", "axis_1", etc.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, TensorHandle};
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let handle = TensorHandle::from_dense_auto(tensor);
    /// assert_eq!(handle.rank(), 3);
    /// assert_eq!(handle.axes[0].name, "axis_0");
    /// ```
    pub fn from_dense_auto(dense: DenseND<T>) -> Self {
        let axes = dense
            .shape()
            .iter()
            .enumerate()
            .map(|(i, &size)| AxisMeta::new(format!("axis_{}", i), size))
            .collect();
        Self {
            repr: TensorRepr::Dense(dense),
            axes,
        }
    }

    /// Get the rank (number of dimensions) of this tensor.
    ///
    /// # Complexity
    ///
    /// O(1) - reads cached value from axis metadata
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, TensorHandle};
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let handle = TensorHandle::from_dense_auto(tensor);
    ///
    /// assert_eq!(handle.rank(), 3);
    /// ```
    pub fn rank(&self) -> Rank {
        self.axes.len()
    }

    /// Get the shape of this tensor.
    ///
    /// Returns a `SmallVec` containing the size of each dimension.
    ///
    /// # Complexity
    ///
    /// O(rank) - constructs SmallVec from axis metadata
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, TensorHandle};
    ///
    /// let tensor = DenseND::<f64>::zeros(&[10, 20, 30]);
    /// let handle = TensorHandle::from_dense_auto(tensor);
    ///
    /// let shape = handle.shape();
    /// assert_eq!(shape.as_slice(), &[10, 20, 30]);
    /// ```
    pub fn shape(&self) -> Shape {
        self.axes.iter().map(|a| a.size).collect()
    }

    /// Get a reference to the dense representation if this is a dense tensor.
    ///
    /// Returns `None` if the tensor is stored in sparse or low-rank format.
    ///
    /// # Complexity
    ///
    /// O(1) - pattern matching on enum variant
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, TensorHandle};
    ///
    /// let tensor = DenseND::<f64>::ones(&[5, 5]);
    /// let handle = TensorHandle::from_dense_auto(tensor);
    ///
    /// // Access the underlying dense tensor
    /// if let Some(dense) = handle.as_dense() {
    ///     assert_eq!(dense.shape(), &[5, 5]);
    ///     assert_eq!(dense.len(), 25);
    /// }
    /// ```
    pub fn as_dense(&self) -> Option<&DenseND<T>> {
        match &self.repr {
            TensorRepr::Dense(d) => Some(d),
            _ => None,
        }
    }

    /// Get a mutable reference to the dense representation if this is a dense tensor.
    ///
    /// Returns `None` if the tensor is stored in sparse or low-rank format.
    ///
    /// # Complexity
    ///
    /// O(1) - pattern matching on enum variant
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, TensorHandle};
    ///
    /// let tensor = DenseND::<f64>::zeros(&[3, 3]);
    /// let mut handle = TensorHandle::from_dense_auto(tensor);
    ///
    /// // Modify the underlying dense tensor
    /// if let Some(dense) = handle.as_dense_mut() {
    ///     dense.as_array_mut()[[0, 0]] = 1.0;
    /// }
    ///
    /// assert_eq!(handle.as_dense().unwrap()[&[0, 0]], 1.0);
    /// ```
    pub fn as_dense_mut(&mut self) -> Option<&mut DenseND<T>> {
        match &mut self.repr {
            TensorRepr::Dense(d) => Some(d),
            _ => None,
        }
    }

    /// Convert this tensor to dense representation if not already dense.
    ///
    /// For dense tensors, this clones the underlying data.
    /// For sparse and low-rank tensors, this will be implemented in future releases.
    ///
    /// # Complexity
    ///
    /// - **Dense:** O(n) where n is the number of elements (clone operation)
    /// - **Sparse:** Not yet implemented
    /// - **LowRank:** Not yet implemented
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is sparse or low-rank (not yet implemented).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, TensorHandle};
    ///
    /// let tensor = DenseND::<f64>::ones(&[4, 4]);
    /// let handle = TensorHandle::from_dense_auto(tensor);
    ///
    /// // Convert to dense (clones the data)
    /// let dense = handle.to_dense().unwrap();
    /// assert_eq!(dense.shape(), &[4, 4]);
    /// assert_eq!(dense.len(), 16);
    /// ```
    pub fn to_dense(&self) -> anyhow::Result<DenseND<T>> {
        match &self.repr {
            TensorRepr::Dense(d) => Ok(d.clone()),
            TensorRepr::Sparse(_) => {
                anyhow::bail!("Sparse to dense conversion not yet implemented")
            }
            TensorRepr::LowRank(_) => {
                anyhow::bail!("Low-rank to dense conversion not yet implemented")
            }
        }
    }
}
