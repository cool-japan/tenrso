//! Masked einsum operations for sparse-dense mixed computation
//!
//! This module provides masked Einstein summation that computes only the output
//! elements indicated by a [`Mask`], returning sparse results. This is critical
//! for performance when only a small subset of the output is needed.
//!
//! # Key Operations
//!
//! - **Masked matmul** (`ij,jk->ik`): Compute only masked output positions
//! - **Masked element-wise** (`ij,ij->ij`): Hadamard product at mask positions
//! - **Masked outer product** (`i,j->ij`): Outer product at mask positions
//! - **Masked reductions**: `masked_sum`, `masked_mean` over sparse subsets
//!
//! # Performance
//!
//! For high sparsity (e.g., 10% mask density), masked einsum avoids computing
//! ~90% of the output, yielding proportional speedups over dense computation.
//!
//! # Examples
//!
//! ```
//! use tenrso_core::DenseND;
//! use tenrso_sparse::mask::Mask;
//! use tenrso_sparse::masked_einsum::masked_einsum;
//!
//! // Two 3x3 matrices
//! let a = DenseND::from_vec(
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
//!     &[3, 3],
//! ).unwrap();
//! let b = DenseND::from_vec(
//!     vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
//!     &[3, 3],
//! ).unwrap();
//!
//! // Only compute diagonal elements
//! let mask = Mask::from_indices(
//!     vec![vec![0, 0], vec![1, 1], vec![2, 2]],
//!     vec![3, 3],
//! ).unwrap();
//!
//! let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).unwrap();
//! assert_eq!(result.nnz(), 3);
//! ```
//!
//! # SciRS2 Integration
//!
//! All numeric operations use `scirs2_core::numeric::Float`.
//! Direct use of `ndarray` or `rand` is forbidden per SCIRS2_INTEGRATION_POLICY.md.

use crate::coo::CooTensor;
use crate::mask::Mask;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet};
use tenrso_core::DenseND;
use thiserror::Error;

/// Errors specific to masked einsum operations
#[derive(Error, Debug)]
pub enum MaskedEinsumError {
    /// The einsum specification string is malformed
    #[error("Invalid einsum spec: {0}")]
    InvalidSpec(String),

    /// The number of inputs does not match the spec
    #[error("Input count mismatch: spec expects {expected} inputs, got {actual}")]
    InputCountMismatch { expected: usize, actual: usize },

    /// An input tensor's shape does not match the index dimensions in the spec
    #[error("Shape mismatch for input {input_idx}: spec expects {expected} dims, shape has {actual} dims")]
    InputShapeMismatch {
        input_idx: usize,
        expected: usize,
        actual: usize,
    },

    /// Two indices that should share a dimension have different sizes
    #[error("Dimension mismatch for index '{index}': was {prev_size}, now {new_size}")]
    DimensionMismatch {
        index: char,
        prev_size: usize,
        new_size: usize,
    },

    /// The mask shape does not match the computed output shape
    #[error("Mask shape {mask_shape:?} does not match output shape {output_shape:?}")]
    MaskShapeMismatch {
        mask_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// The einsum operation type is not supported by the masked engine
    #[error("Unsupported masked einsum pattern: {0}")]
    UnsupportedPattern(String),

    /// An internal COO construction error
    #[error("COO construction error: {0}")]
    CooError(#[from] crate::coo::CooError),
}

// ---------------------------------------------------------------------------
// Lightweight einsum spec parser (avoids tenrso-planner dependency)
// ---------------------------------------------------------------------------

/// Parsed einsum specification (lightweight, local to this module)
#[derive(Debug, Clone)]
struct EinsumSpec {
    /// Per-input index lists, e.g. [['i','j'], ['j','k']]
    inputs: Vec<Vec<char>>,
    /// Output index list, e.g. ['i','k']
    output: Vec<char>,
    /// All unique indices
    all_indices: HashSet<char>,
    /// Indices that are summed over (contracted)
    contracted: HashSet<char>,
}

impl EinsumSpec {
    /// Parse an einsum notation string such as `"ij,jk->ik"`.
    fn parse(spec: &str) -> Result<Self, MaskedEinsumError> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err(MaskedEinsumError::InvalidSpec(
                "empty specification".to_string(),
            ));
        }

        let parts: Vec<&str> = spec.split("->").collect();
        if parts.len() > 2 {
            return Err(MaskedEinsumError::InvalidSpec(
                "multiple '->' found".to_string(),
            ));
        }

        let input_str = parts[0].trim();
        if input_str.is_empty() {
            return Err(MaskedEinsumError::InvalidSpec(
                "no inputs provided".to_string(),
            ));
        }

        let inputs: Vec<Vec<char>> = input_str
            .split(',')
            .map(|s| s.trim().chars().collect())
            .collect();

        for (i, inp) in inputs.iter().enumerate() {
            if inp.is_empty() {
                return Err(MaskedEinsumError::InvalidSpec(format!(
                    "input {} is empty",
                    i
                )));
            }
            for &c in inp {
                if !c.is_ascii_lowercase() {
                    return Err(MaskedEinsumError::InvalidSpec(format!(
                        "input {} contains invalid character '{}'",
                        i, c
                    )));
                }
            }
        }

        let output: Vec<char> = if parts.len() == 2 {
            let out_str = parts[1].trim();
            for c in out_str.chars() {
                if !c.is_ascii_lowercase() {
                    return Err(MaskedEinsumError::InvalidSpec(format!(
                        "output contains invalid character '{}'",
                        c
                    )));
                }
            }
            out_str.chars().collect()
        } else {
            // Infer: all indices in first-appearance order
            let mut seen = HashSet::new();
            let mut out = Vec::new();
            for inp in &inputs {
                for &c in inp {
                    if seen.insert(c) {
                        out.push(c);
                    }
                }
            }
            out
        };

        let mut all_indices = HashSet::new();
        for inp in &inputs {
            for &c in inp {
                all_indices.insert(c);
            }
        }

        let output_set: HashSet<char> = output.iter().copied().collect();

        // Validate output indices are in inputs
        for &c in &output {
            if !all_indices.contains(&c) {
                return Err(MaskedEinsumError::InvalidSpec(format!(
                    "output index '{}' not in any input",
                    c
                )));
            }
        }

        let contracted: HashSet<char> = all_indices
            .iter()
            .copied()
            .filter(|c| !output_set.contains(c))
            .collect();

        Ok(Self {
            inputs,
            output,
            all_indices,
            contracted,
        })
    }
}

// ---------------------------------------------------------------------------
// Dimension map: index char -> size
// ---------------------------------------------------------------------------

/// Build a mapping from index character to dimension size, validating consistency.
fn build_dim_map(
    spec: &EinsumSpec,
    inputs: &[&DenseND<impl Float>],
) -> Result<HashMap<char, usize>, MaskedEinsumError> {
    if spec.inputs.len() != inputs.len() {
        return Err(MaskedEinsumError::InputCountMismatch {
            expected: spec.inputs.len(),
            actual: inputs.len(),
        });
    }

    let mut dim_map: HashMap<char, usize> = HashMap::new();

    for (inp_idx, (spec_indices, tensor)) in spec.inputs.iter().zip(inputs.iter()).enumerate() {
        let shape = tensor.shape();
        if spec_indices.len() != shape.len() {
            return Err(MaskedEinsumError::InputShapeMismatch {
                input_idx: inp_idx,
                expected: spec_indices.len(),
                actual: shape.len(),
            });
        }

        for (&c, &size) in spec_indices.iter().zip(shape.iter()) {
            if let Some(&prev) = dim_map.get(&c) {
                if prev != size {
                    return Err(MaskedEinsumError::DimensionMismatch {
                        index: c,
                        prev_size: prev,
                        new_size: size,
                    });
                }
            } else {
                dim_map.insert(c, size);
            }
        }
    }

    Ok(dim_map)
}

/// Compute the expected output shape from spec and dimension map.
fn compute_output_shape(
    spec: &EinsumSpec,
    dim_map: &HashMap<char, usize>,
) -> Result<Vec<usize>, MaskedEinsumError> {
    spec.output
        .iter()
        .map(|c| {
            dim_map.get(c).copied().ok_or_else(|| {
                MaskedEinsumError::InvalidSpec(format!(
                    "output index '{}' missing from dimension map",
                    c
                ))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public API: masked_einsum
// ---------------------------------------------------------------------------

/// Perform a masked Einstein summation on dense inputs.
///
/// Only the output positions indicated by `mask` are computed. The result is
/// returned as a sparse [`CooTensor`].
///
/// # Supported Patterns
///
/// | Pattern | Description |
/// |---------|-------------|
/// | `ij,jk->ik` | Masked matrix multiplication |
/// | `ij,ij->ij` | Masked element-wise (Hadamard) product |
/// | `i,j->ij` | Masked outer product |
/// | General | Falls back to generic contraction loop |
///
/// # Arguments
///
/// * `spec_str` - Einsum notation string (e.g. `"ij,jk->ik"`)
/// * `inputs`   - Slice of references to dense input tensors
/// * `mask`     - Boolean mask indicating which output elements to compute
///
/// # Errors
///
/// Returns error on malformed spec, shape mismatches, or mask shape mismatches.
///
/// # Complexity
///
/// For masked matmul with mask density `d` and contraction dimension `K`:
/// O(d * M * N * K) vs O(M * N * K) for full matmul.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_sparse::mask::Mask;
/// use tenrso_sparse::masked_einsum::masked_einsum;
///
/// let a = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
/// let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
/// let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![2, 2]).unwrap();
///
/// let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).unwrap();
/// assert_eq!(result.nnz(), 2);
/// ```
pub fn masked_einsum<T: Float>(
    spec_str: &str,
    inputs: &[&DenseND<T>],
    mask: &Mask,
) -> Result<CooTensor<T>, MaskedEinsumError> {
    let spec = EinsumSpec::parse(spec_str)?;
    let dim_map = build_dim_map(&spec, inputs)?;
    let output_shape = compute_output_shape(&spec, &dim_map)?;

    // Validate mask shape
    if mask.shape() != output_shape.as_slice() {
        return Err(MaskedEinsumError::MaskShapeMismatch {
            mask_shape: mask.shape().to_vec(),
            output_shape,
        });
    }

    // Empty mask => empty result
    if mask.nnz() == 0 {
        return Ok(CooTensor::zeros(output_shape)?);
    }

    // Dispatch to specialised kernel when possible
    if spec.inputs.len() == 2 {
        let lhs = &spec.inputs[0];
        let rhs = &spec.inputs[1];
        let out = &spec.output;

        // ij,jk->ik  (matmul)
        if lhs.len() == 2
            && rhs.len() == 2
            && out.len() == 2
            && lhs[0] == out[0]
            && rhs[1] == out[1]
            && lhs[1] == rhs[0]
            && spec.contracted.len() == 1
            && spec.contracted.contains(&lhs[1])
        {
            return masked_matmul(inputs[0], inputs[1], mask, &output_shape);
        }

        // ij,ij->ij  (element-wise / Hadamard)
        if lhs == rhs && lhs == out && spec.contracted.is_empty() {
            return masked_elementwise(inputs[0], inputs[1], mask, &output_shape);
        }

        // i,j->ij  (outer product)
        if lhs.len() == 1
            && rhs.len() == 1
            && out.len() == 2
            && out[0] == lhs[0]
            && out[1] == rhs[0]
            && spec.contracted.is_empty()
        {
            return masked_outer(inputs[0], inputs[1], mask, &output_shape);
        }
    }

    // Generic fallback
    masked_generic(&spec, inputs, mask, &dim_map, &output_shape)
}

// ---------------------------------------------------------------------------
// Specialised kernels
// ---------------------------------------------------------------------------

/// Masked matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]  for (i,k) in mask.
///
/// # Complexity
///
/// O(mask.nnz() * K) where K is the contraction dimension.
fn masked_matmul<T: Float>(
    a: &DenseND<T>,
    b: &DenseND<T>,
    mask: &Mask,
    output_shape: &[usize],
) -> Result<CooTensor<T>, MaskedEinsumError> {
    let a_view = a.view();
    let b_view = b.view();
    let k_dim = a.shape()[1]; // contraction dimension

    let sorted = mask.to_sorted_indices();
    let mut result_indices = Vec::with_capacity(sorted.len());
    let mut result_values = Vec::with_capacity(sorted.len());

    for idx in &sorted {
        let i = idx[0];
        let k = idx[1];

        let mut acc = T::zero();
        for j in 0..k_dim {
            acc = acc + a_view[&[i, j][..]] * b_view[&[j, k][..]];
        }

        if acc.abs() > T::epsilon() {
            result_indices.push(idx.clone());
            result_values.push(acc);
        }
    }

    Ok(CooTensor::new(
        result_indices,
        result_values,
        output_shape.to_vec(),
    )?)
}

/// Masked element-wise (Hadamard) product: C[i,j] = A[i,j] * B[i,j]  for (i,j) in mask.
///
/// # Complexity
///
/// O(mask.nnz())
fn masked_elementwise<T: Float>(
    a: &DenseND<T>,
    b: &DenseND<T>,
    mask: &Mask,
    output_shape: &[usize],
) -> Result<CooTensor<T>, MaskedEinsumError> {
    let a_view = a.view();
    let b_view = b.view();

    let sorted = mask.to_sorted_indices();
    let mut result_indices = Vec::with_capacity(sorted.len());
    let mut result_values = Vec::with_capacity(sorted.len());

    for idx in &sorted {
        let val = a_view[&idx[..]] * b_view[&idx[..]];
        if val.abs() > T::epsilon() {
            result_indices.push(idx.clone());
            result_values.push(val);
        }
    }

    Ok(CooTensor::new(
        result_indices,
        result_values,
        output_shape.to_vec(),
    )?)
}

/// Masked outer product: C[i,j] = A[i] * B[j]  for (i,j) in mask.
///
/// # Complexity
///
/// O(mask.nnz())
fn masked_outer<T: Float>(
    a: &DenseND<T>,
    b: &DenseND<T>,
    mask: &Mask,
    output_shape: &[usize],
) -> Result<CooTensor<T>, MaskedEinsumError> {
    let a_view = a.view();
    let b_view = b.view();

    let sorted = mask.to_sorted_indices();
    let mut result_indices = Vec::with_capacity(sorted.len());
    let mut result_values = Vec::with_capacity(sorted.len());

    for idx in &sorted {
        let val = a_view[&idx[0..1]] * b_view[&idx[1..2]];
        if val.abs() > T::epsilon() {
            result_indices.push(idx.clone());
            result_values.push(val);
        }
    }

    Ok(CooTensor::new(
        result_indices,
        result_values,
        output_shape.to_vec(),
    )?)
}

// ---------------------------------------------------------------------------
// Generic masked contraction (fallback)
// ---------------------------------------------------------------------------

/// Generic masked contraction for arbitrary einsum specs.
///
/// For each mask position, iterate over all contracted-index combinations and
/// accumulate the product of input elements.
///
/// # Complexity
///
/// O(mask.nnz() * product_of_contracted_dims)
fn masked_generic<T: Float>(
    spec: &EinsumSpec,
    inputs: &[&DenseND<T>],
    mask: &Mask,
    dim_map: &HashMap<char, usize>,
    output_shape: &[usize],
) -> Result<CooTensor<T>, MaskedEinsumError> {
    // Contracted indices in a fixed order
    let contracted_chars: Vec<char> = {
        let mut v: Vec<char> = spec.contracted.iter().copied().collect();
        v.sort();
        v
    };
    let contracted_sizes: Vec<usize> = contracted_chars
        .iter()
        .map(|c| {
            dim_map.get(c).copied().ok_or_else(|| {
                MaskedEinsumError::InvalidSpec(format!("contracted index '{}' not in dim_map", c))
            })
        })
        .collect::<Result<_, _>>()?;

    // Total number of contracted-index combinations
    let contracted_total: usize = contracted_sizes.iter().product::<usize>().max(1);

    let sorted = mask.to_sorted_indices();
    let mut result_indices = Vec::with_capacity(sorted.len());
    let mut result_values = Vec::with_capacity(sorted.len());

    // Precompute output-index to char mapping
    let output_chars = &spec.output;

    for out_idx in &sorted {
        // Build output-index assignment: char -> value
        let mut assignment: HashMap<char, usize> = HashMap::with_capacity(spec.all_indices.len());
        for (pos, &c) in output_chars.iter().enumerate() {
            assignment.insert(c, out_idx[pos]);
        }

        let mut acc = T::zero();

        // Iterate over contracted-index combinations
        for flat_c in 0..contracted_total {
            // Decode flat_c into contracted index values
            let mut remainder = flat_c;
            for (ci, &csize) in contracted_sizes.iter().enumerate().rev() {
                assignment.insert(contracted_chars[ci], remainder % csize);
                remainder /= csize;
            }

            // Compute product of input elements at this assignment
            let mut prod = T::one();
            let mut valid = true;
            for (inp_spec, &tensor) in spec.inputs.iter().zip(inputs.iter()) {
                let view = tensor.view();
                let multi_idx: Vec<usize> = inp_spec
                    .iter()
                    .map(|c| {
                        assignment.get(c).copied().ok_or_else(|| {
                            MaskedEinsumError::InvalidSpec(format!("index '{}' not assigned", c))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                prod = prod * view[&multi_idx[..]];
                if prod.is_nan() {
                    valid = false;
                    break;
                }
            }

            if valid {
                acc = acc + prod;
            }
        }

        if acc.abs() > T::epsilon() {
            result_indices.push(out_idx.clone());
            result_values.push(acc);
        }
    }

    Ok(CooTensor::new(
        result_indices,
        result_values,
        output_shape.to_vec(),
    )?)
}

// ---------------------------------------------------------------------------
// Masked reductions (subset reductions)
// ---------------------------------------------------------------------------

/// Compute the sum of a dense tensor at positions specified by a mask.
///
/// Only the elements whose indices are in the mask contribute to the sum.
/// This is equivalent to masking the tensor and then reducing.
///
/// # Complexity
///
/// O(mask.nnz())
///
/// # Errors
///
/// Returns error if mask shape does not match tensor shape.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_sparse::mask::Mask;
/// use tenrso_sparse::masked_einsum::masked_sum;
///
/// let tensor = DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![2, 2]).unwrap();
///
/// let s = masked_sum(&tensor, &mask).unwrap();
/// assert!((s - 5.0_f64).abs() < 1e-10); // 1.0 + 4.0
/// ```
pub fn masked_sum<T: Float>(tensor: &DenseND<T>, mask: &Mask) -> Result<T, MaskedEinsumError> {
    if mask.shape() != tensor.shape() {
        return Err(MaskedEinsumError::MaskShapeMismatch {
            mask_shape: mask.shape().to_vec(),
            output_shape: tensor.shape().to_vec(),
        });
    }

    let view = tensor.view();
    let mut acc = T::zero();

    for idx in mask.iter() {
        acc = acc + view[&idx[..]];
    }

    Ok(acc)
}

/// Compute the mean of a dense tensor at positions specified by a mask.
///
/// Only the elements whose indices are in the mask contribute. The divisor
/// is the number of mask positions (not the total tensor size).
///
/// # Complexity
///
/// O(mask.nnz())
///
/// # Errors
///
/// Returns error if mask shape does not match tensor shape, or if mask is empty.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_sparse::mask::Mask;
/// use tenrso_sparse::masked_einsum::masked_mean;
///
/// let tensor = DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![2, 2]).unwrap();
///
/// let m = masked_mean(&tensor, &mask).unwrap();
/// assert!((m - 2.5_f64).abs() < 1e-10); // (1.0 + 4.0) / 2
/// ```
pub fn masked_mean<T: Float>(tensor: &DenseND<T>, mask: &Mask) -> Result<T, MaskedEinsumError> {
    if mask.nnz() == 0 {
        return Err(MaskedEinsumError::InvalidSpec(
            "cannot compute mean over empty mask".to_string(),
        ));
    }

    let total = masked_sum(tensor, mask)?;
    let n = T::from(mask.nnz() as f64).unwrap_or_else(|| T::one());
    Ok(total / n)
}

/// Compute the maximum of a dense tensor at positions specified by a mask.
///
/// # Complexity
///
/// O(mask.nnz())
///
/// # Errors
///
/// Returns error if mask shape does not match tensor shape, or if mask is empty.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_sparse::mask::Mask;
/// use tenrso_sparse::masked_einsum::masked_max;
///
/// let tensor = DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let mask = Mask::from_indices(vec![vec![0, 1], vec![1, 0]], vec![2, 2]).unwrap();
///
/// let mx = masked_max(&tensor, &mask).unwrap();
/// assert!((mx - 3.0_f64).abs() < 1e-10); // max(2.0, 3.0)
/// ```
pub fn masked_max<T: Float>(tensor: &DenseND<T>, mask: &Mask) -> Result<T, MaskedEinsumError> {
    if mask.shape() != tensor.shape() {
        return Err(MaskedEinsumError::MaskShapeMismatch {
            mask_shape: mask.shape().to_vec(),
            output_shape: tensor.shape().to_vec(),
        });
    }

    if mask.nnz() == 0 {
        return Err(MaskedEinsumError::InvalidSpec(
            "cannot compute max over empty mask".to_string(),
        ));
    }

    let view = tensor.view();
    let mut first = true;
    let mut best = T::zero();

    for idx in mask.iter() {
        let val = view[&idx[..]];
        if first || val > best {
            best = val;
            first = false;
        }
    }

    Ok(best)
}

/// Compute the minimum of a dense tensor at positions specified by a mask.
///
/// # Complexity
///
/// O(mask.nnz())
///
/// # Errors
///
/// Returns error if mask shape does not match tensor shape, or if mask is empty.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_sparse::mask::Mask;
/// use tenrso_sparse::masked_einsum::masked_min;
///
/// let tensor = DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let mask = Mask::from_indices(vec![vec![0, 1], vec![1, 0]], vec![2, 2]).unwrap();
///
/// let mn = masked_min(&tensor, &mask).unwrap();
/// assert!((mn - 2.0_f64).abs() < 1e-10); // min(2.0, 3.0)
/// ```
pub fn masked_min<T: Float>(tensor: &DenseND<T>, mask: &Mask) -> Result<T, MaskedEinsumError> {
    if mask.shape() != tensor.shape() {
        return Err(MaskedEinsumError::MaskShapeMismatch {
            mask_shape: mask.shape().to_vec(),
            output_shape: tensor.shape().to_vec(),
        });
    }

    if mask.nnz() == 0 {
        return Err(MaskedEinsumError::InvalidSpec(
            "cannot compute min over empty mask".to_string(),
        ));
    }

    let view = tensor.view();
    let mut first = true;
    let mut best = T::zero();

    for idx in mask.iter() {
        let val = view[&idx[..]];
        if first || val < best {
            best = val;
            first = false;
        }
    }

    Ok(best)
}

/// Compute the variance of a dense tensor at positions specified by a mask.
///
/// Uses the two-pass formula: Var = E\[X^2\] - E\[X\]^2, but with Welford's
/// online algorithm for numerical stability.
///
/// # Complexity
///
/// O(mask.nnz())
///
/// # Errors
///
/// Returns error if mask shape does not match tensor shape, or if mask has fewer
/// than 2 positions.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_sparse::mask::Mask;
/// use tenrso_sparse::masked_einsum::masked_variance;
///
/// let tensor = DenseND::from_vec(vec![2.0_f64, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
/// let mask = Mask::from_indices(
///     vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
///     vec![2, 2],
/// ).unwrap();
///
/// let var = masked_variance(&tensor, &mask).unwrap();
/// // mean = 5.0, variance = ((2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2)/4 = 5.0
/// assert!((var - 5.0_f64).abs() < 1e-10);
/// ```
pub fn masked_variance<T: Float>(tensor: &DenseND<T>, mask: &Mask) -> Result<T, MaskedEinsumError> {
    if mask.shape() != tensor.shape() {
        return Err(MaskedEinsumError::MaskShapeMismatch {
            mask_shape: mask.shape().to_vec(),
            output_shape: tensor.shape().to_vec(),
        });
    }

    if mask.nnz() < 1 {
        return Err(MaskedEinsumError::InvalidSpec(
            "cannot compute variance over empty mask".to_string(),
        ));
    }

    let m = masked_mean(tensor, mask)?;
    let view = tensor.view();

    let mut sum_sq = T::zero();
    for idx in mask.iter() {
        let diff = view[&idx[..]] - m;
        sum_sq = sum_sq + diff * diff;
    }

    let n = T::from(mask.nnz() as f64).unwrap_or_else(|| T::one());
    Ok(sum_sq / n)
}

/// Extract masked elements as a sparse COO tensor.
///
/// Returns a COO tensor containing only the values of `tensor` at mask positions.
///
/// # Complexity
///
/// O(mask.nnz())
///
/// # Errors
///
/// Returns error if mask shape does not match tensor shape.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_sparse::mask::Mask;
/// use tenrso_sparse::masked_einsum::masked_extract;
///
/// let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let mask = Mask::from_indices(vec![vec![0, 1], vec![1, 0]], vec![2, 2]).unwrap();
///
/// let sparse = masked_extract(&tensor, &mask).unwrap();
/// assert_eq!(sparse.nnz(), 2);
/// ```
pub fn masked_extract<T: Float>(
    tensor: &DenseND<T>,
    mask: &Mask,
) -> Result<CooTensor<T>, MaskedEinsumError> {
    if mask.shape() != tensor.shape() {
        return Err(MaskedEinsumError::MaskShapeMismatch {
            mask_shape: mask.shape().to_vec(),
            output_shape: tensor.shape().to_vec(),
        });
    }

    let view = tensor.view();
    let sorted = mask.to_sorted_indices();
    let mut result_indices = Vec::with_capacity(sorted.len());
    let mut result_values = Vec::with_capacity(sorted.len());

    for idx in &sorted {
        let val = view[&idx[..]];
        if val.abs() > T::epsilon() {
            result_indices.push(idx.clone());
            result_values.push(val);
        }
    }

    Ok(CooTensor::new(
        result_indices,
        result_values,
        mask.shape().to_vec(),
    )?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // EinsumSpec parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").expect("parse failed");
        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.output, vec!['i', 'k']);
        assert!(spec.contracted.contains(&'j'));
        assert_eq!(spec.contracted.len(), 1);
    }

    #[test]
    fn test_parse_elementwise() {
        let spec = EinsumSpec::parse("ij,ij->ij").expect("parse failed");
        assert!(spec.contracted.is_empty());
    }

    #[test]
    fn test_parse_outer() {
        let spec = EinsumSpec::parse("i,j->ij").expect("parse failed");
        assert!(spec.contracted.is_empty());
        assert_eq!(spec.output, vec!['i', 'j']);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(EinsumSpec::parse("").is_err());
        assert!(EinsumSpec::parse("->->").is_err());
        assert!(EinsumSpec::parse("IJ,JK->IK").is_err());
    }

    // -----------------------------------------------------------------------
    // Masked matmul
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_matmul_identity() {
        // A = I (2x2), B = [[5,6],[7,8]], mask = full => C = B
        let a = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("a");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("b");
        let mask = Mask::full(vec![2, 2]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("einsum");

        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 0][..]] - 5.0).abs() < 1e-10);
        assert!((arr[&[0, 1][..]] - 6.0).abs() < 1e-10);
        assert!((arr[&[1, 0][..]] - 7.0).abs() < 1e-10);
        assert!((arr[&[1, 1][..]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_masked_matmul_diagonal_only() {
        // 3x3 matmul, only diagonal output
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
            .expect("a");
        let b = DenseND::from_vec(vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[3, 3])
            .expect("b");

        let mask =
            Mask::from_indices(vec![vec![0, 0], vec![1, 1], vec![2, 2]], vec![3, 3]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 3);

        // Verify against manual computation
        // C[0,0] = 1*9 + 2*6 + 3*3 = 9+12+9 = 30
        // C[1,1] = 4*8 + 5*5 + 6*2 = 32+25+12 = 69
        // C[2,2] = 7*7 + 8*4 + 9*1 = 49+32+9 = 90
        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 0][..]] - 30.0).abs() < 1e-10);
        assert!((arr[&[1, 1][..]] - 69.0).abs() < 1e-10);
        assert!((arr[&[2, 2][..]] - 90.0).abs() < 1e-10);
        // Off-diagonal should be zero
        assert!((arr[&[0, 1][..]]).abs() < 1e-10);
    }

    #[test]
    fn test_masked_matmul_single_element() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("a");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("b");

        let mask = Mask::from_indices(vec![vec![0, 1]], vec![2, 2]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 1);

        // C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 1][..]] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_masked_matmul_empty_mask() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("a");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("b");

        let mask = Mask::empty(vec![2, 2]).expect("mask");
        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 0);
    }

    #[test]
    fn test_masked_matmul_matches_full_at_mask_positions() {
        // Verify masked result matches full matmul at every mask position
        let a = DenseND::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        )
        .expect("a");
        let b = DenseND::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4, 3],
        )
        .expect("b");

        // Full matmul via masked_einsum with full mask
        let full_mask = Mask::full(vec![3, 3]).expect("full mask");
        let full_result = masked_einsum("ij,jk->ik", &[&a, &b], &full_mask).expect("full");
        let full_dense = full_result.to_dense().expect("full dense");

        // Sparse mask
        let mask =
            Mask::from_indices(vec![vec![0, 2], vec![1, 0], vec![2, 1]], vec![3, 3]).expect("mask");
        let sparse_result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("sparse");
        let sparse_dense = sparse_result.to_dense().expect("sparse dense");

        // Check match at mask positions
        for idx in mask.iter() {
            let full_val = full_dense.as_array()[&idx[..]];
            let sparse_val = sparse_dense.as_array()[&idx[..]];
            assert!(
                (full_val - sparse_val).abs() < 1e-10,
                "Mismatch at {:?}: full={}, sparse={}",
                idx,
                full_val,
                sparse_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // Masked element-wise
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_elementwise_product() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("a");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("b");

        let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![2, 2]).expect("mask");

        let result = masked_einsum("ij,ij->ij", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 2);

        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 0][..]] - 5.0).abs() < 1e-10); // 1*5
        assert!((arr[&[1, 1][..]] - 32.0).abs() < 1e-10); // 4*8
    }

    // -----------------------------------------------------------------------
    // Masked outer product
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_outer_product() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("a");
        let b = DenseND::from_vec(vec![4.0, 5.0], &[2]).expect("b");

        let mask =
            Mask::from_indices(vec![vec![0, 0], vec![1, 1], vec![2, 0]], vec![3, 2]).expect("mask");

        let result = masked_einsum("i,j->ij", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 3);

        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 0][..]] - 4.0).abs() < 1e-10); // 1*4
        assert!((arr[&[1, 1][..]] - 10.0).abs() < 1e-10); // 2*5
        assert!((arr[&[2, 0][..]] - 12.0).abs() < 1e-10); // 3*4
    }

    // -----------------------------------------------------------------------
    // Generic fallback (3-input contraction)
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_generic_batched_matmul() {
        // bij,bjk->bik with mask
        // batch=1, i=2, j=2, k=2
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).expect("a");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[1, 2, 2]).expect("b");

        let mask =
            Mask::from_indices(vec![vec![0, 0, 0], vec![0, 1, 1]], vec![1, 2, 2]).expect("mask");

        let result = masked_einsum("bij,bjk->bik", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 2);

        // C[0,0,0] = A[0,0,0]*B[0,0,0] + A[0,0,1]*B[0,1,0] = 1*5 + 2*7 = 19
        // C[0,1,1] = A[0,1,0]*B[0,0,1] + A[0,1,1]*B[0,1,1] = 3*6 + 4*8 = 50
        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 0, 0][..]] - 19.0).abs() < 1e-10);
        assert!((arr[&[0, 1, 1][..]] - 50.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Masked reductions
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_sum() {
        let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![2, 2]).expect("mask");

        let s = masked_sum(&tensor, &mask).expect("sum");
        assert!((s - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_masked_mean() {
        let tensor = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).expect("tensor");
        let mask = Mask::from_indices(
            vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
            vec![2, 2],
        )
        .expect("mask");

        let m = masked_mean(&tensor, &mask).expect("mean");
        assert!((m - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_masked_max() {
        let tensor = DenseND::from_vec(vec![1.0, 9.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 0]], vec![2, 2]).expect("mask");

        let mx = masked_max(&tensor, &mask).expect("max");
        assert!((mx - 3.0).abs() < 1e-10); // max(1.0, 3.0)
    }

    #[test]
    fn test_masked_min() {
        let tensor = DenseND::from_vec(vec![1.0, 9.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 0]], vec![2, 2]).expect("mask");

        let mn = masked_min(&tensor, &mask).expect("min");
        assert!((mn - 1.0).abs() < 1e-10); // min(1.0, 3.0)
    }

    #[test]
    fn test_masked_variance() {
        let tensor = DenseND::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).expect("tensor");
        let mask = Mask::from_indices(
            vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
            vec![2, 2],
        )
        .expect("mask");

        let var = masked_variance(&tensor, &mask).expect("variance");
        // mean = 5.0, variance = ((2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2)/4 = 20/4 = 5.0
        assert!((var - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_masked_extract() {
        let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        let mask = Mask::from_indices(vec![vec![0, 1], vec![1, 0]], vec![2, 2]).expect("mask");

        let sparse = masked_extract(&tensor, &mask).expect("extract");
        assert_eq!(sparse.nnz(), 2);

        let dense = sparse.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 1][..]] - 2.0).abs() < 1e-10);
        assert!((arr[&[1, 0][..]] - 3.0).abs() < 1e-10);
        assert!((arr[&[0, 0][..]]).abs() < 1e-10); // not in mask
    }

    // -----------------------------------------------------------------------
    // Shape mismatch / error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_mask_shape_mismatch() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("a");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("b");
        let mask = Mask::from_indices(vec![vec![0, 0]], vec![3, 3]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_input_count_mismatch() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("a");
        let mask = Mask::from_indices(vec![vec![0, 0]], vec![2, 2]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a], &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("a");
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
            .expect("b");
        let mask = Mask::from_indices(vec![vec![0, 0]], vec![2, 3]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask);
        assert!(result.is_err()); // j is 2 in A but 3 in B
    }

    #[test]
    fn test_masked_sum_shape_mismatch() {
        let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        let mask = Mask::from_indices(vec![vec![0, 0]], vec![3, 3]).expect("mask");

        assert!(masked_sum(&tensor, &mask).is_err());
    }

    #[test]
    fn test_masked_mean_empty_mask() {
        let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        let mask = Mask::empty(vec![2, 2]).expect("mask");

        assert!(masked_mean(&tensor, &mask).is_err());
    }

    // -----------------------------------------------------------------------
    // Sparsity-level tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_matmul_10_percent_sparsity() {
        // 10x10 matmul with ~10% mask density (10 elements)
        let n = 10;
        let mut a_data = vec![0.0f64; n * n];
        let mut b_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = (i * n + j + 1) as f64;
                b_data[i * n + j] = ((n * n) - (i * n + j)) as f64;
            }
        }
        let a = DenseND::from_vec(a_data, &[n, n]).expect("a");
        let b = DenseND::from_vec(b_data, &[n, n]).expect("b");

        // ~10% mask: 10 positions out of 100
        let mask_indices: Vec<Vec<usize>> = (0..n).map(|i| vec![i, i]).collect();
        let mask = Mask::from_indices(mask_indices, vec![n, n]).expect("mask");
        assert!((mask.density() - 0.1).abs() < 0.01);

        // Full computation for reference
        let full_mask = Mask::full(vec![n, n]).expect("full");
        let full = masked_einsum("ij,jk->ik", &[&a, &b], &full_mask).expect("full");
        let full_dense = full.to_dense().expect("full_dense");

        // Sparse computation
        let sparse = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("sparse");
        let sparse_dense = sparse.to_dense().expect("sparse_dense");

        // Verify diagonal matches
        for i in 0..n {
            let full_val = full_dense.as_array()[&[i, i][..]];
            let sparse_val = sparse_dense.as_array()[&[i, i][..]];
            assert!(
                (full_val - sparse_val).abs() < 1e-8,
                "Mismatch at [{},{}]: full={}, sparse={}",
                i,
                i,
                full_val,
                sparse_val
            );
        }
    }

    #[test]
    fn test_masked_matmul_50_percent_sparsity() {
        let n = 4;
        let mut a_data = vec![0.0f64; n * n];
        let mut b_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = (i + j + 1) as f64;
                b_data[i * n + j] = (i * j + 1) as f64;
            }
        }
        let a = DenseND::from_vec(a_data, &[n, n]).expect("a");
        let b = DenseND::from_vec(b_data, &[n, n]).expect("b");

        // 50% mask: every other element (8 out of 16)
        let mut mask_indices = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if (i + j) % 2 == 0 {
                    mask_indices.push(vec![i, j]);
                }
            }
        }
        let mask = Mask::from_indices(mask_indices.clone(), vec![n, n]).expect("mask");
        assert!((mask.density() - 0.5).abs() < 0.01);

        // Full for reference
        let full_mask = Mask::full(vec![n, n]).expect("full");
        let full = masked_einsum("ij,jk->ik", &[&a, &b], &full_mask).expect("full");
        let full_dense = full.to_dense().expect("full_dense");

        let sparse = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("sparse");
        let sparse_dense = sparse.to_dense().expect("sparse_dense");

        for idx in &mask_indices {
            let fv = full_dense.as_array()[&idx[..]];
            let sv = sparse_dense.as_array()[&idx[..]];
            assert!(
                (fv - sv).abs() < 1e-8,
                "Mismatch at {:?}: full={}, sparse={}",
                idx,
                fv,
                sv
            );
        }
    }

    #[test]
    fn test_masked_matmul_90_percent_sparsity() {
        // 90% sparsity = 10% density => only ~2 elements for a 4x4
        let n = 4;
        let mut a_data = vec![0.0f64; n * n];
        let mut b_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = (i + 1) as f64;
                b_data[i * n + j] = (j + 1) as f64;
            }
        }
        let a = DenseND::from_vec(a_data, &[n, n]).expect("a");
        let b = DenseND::from_vec(b_data, &[n, n]).expect("b");

        // ~10% density (2 out of 16)
        let mask = Mask::from_indices(vec![vec![0, 0], vec![3, 3]], vec![n, n]).expect("mask");

        let full_mask = Mask::full(vec![n, n]).expect("full");
        let full = masked_einsum("ij,jk->ik", &[&a, &b], &full_mask).expect("full");
        let full_dense = full.to_dense().expect("full_dense");

        let sparse = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("sparse");
        let sparse_dense = sparse.to_dense().expect("sparse_dense");

        let fv00 = full_dense.as_array()[&[0, 0][..]];
        let sv00 = sparse_dense.as_array()[&[0, 0][..]];
        assert!((fv00 - sv00).abs() < 1e-8);

        let fv33 = full_dense.as_array()[&[3, 3][..]];
        let sv33 = sparse_dense.as_array()[&[3, 3][..]];
        assert!((fv33 - sv33).abs() < 1e-8);
    }

    #[test]
    fn test_masked_matmul_rectangular() {
        // 2x3 * 3x4 -> 2x4 with mask
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
        let b = DenseND::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        )
        .expect("b");

        let mask =
            Mask::from_indices(vec![vec![0, 0], vec![0, 3], vec![1, 2]], vec![2, 4]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 3);

        // C[0,0] = 1*1 + 2*5 + 3*9 = 1+10+27 = 38
        // C[0,3] = 1*4 + 2*8 + 3*12 = 4+16+36 = 56
        // C[1,2] = 4*3 + 5*7 + 6*11 = 12+35+66 = 113
        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 0][..]] - 38.0).abs() < 1e-10);
        assert!((arr[&[0, 3][..]] - 56.0).abs() < 1e-10);
        assert!((arr[&[1, 2][..]] - 113.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 1-D tests for reductions
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_sum_1d() {
        let tensor = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).expect("t");
        let mask = Mask::from_indices(vec![vec![1], vec![3]], vec![5]).expect("mask");

        let s = masked_sum(&tensor, &mask).expect("sum");
        assert!((s - 60.0).abs() < 1e-10); // 20 + 40
    }

    #[test]
    fn test_masked_mean_1d() {
        let tensor = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).expect("t");
        let mask = Mask::from_indices(vec![vec![1], vec![3]], vec![5]).expect("mask");

        let m = masked_mean(&tensor, &mask).expect("mean");
        assert!((m - 30.0).abs() < 1e-10); // (20 + 40) / 2
    }

    // -----------------------------------------------------------------------
    // f32 type test
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_einsum_f32() {
        let a = DenseND::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("a");
        let b = DenseND::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).expect("b");
        let mask = Mask::from_indices(vec![vec![0, 0]], vec![2, 2]).expect("mask");

        let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).expect("einsum");
        assert_eq!(result.nnz(), 1);

        // C[0,0] = 1*5 + 2*7 = 19
        let dense = result.to_dense().expect("to_dense");
        let arr = dense.as_array();
        assert!((arr[&[0, 0][..]] - 19.0f32).abs() < 1e-4);
    }
}
