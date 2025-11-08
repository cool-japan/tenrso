//! Outer product operations for tensor construction
//!
//! The outer product is fundamental for CP decomposition reconstruction.
//! For vectors v₁, v₂, ..., vₙ, the outer product creates a tensor where
//! `T[i₁, i₂, ..., iₙ] = v₁[i₁] × v₂[i₂] × ... × vₙ[iₙ]`
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array, Array1, Array2, ArrayView1, ArrayView2, IxDyn};
use scirs2_core::numeric::Num;

/// Compute the outer product of two vectors to form a matrix
///
/// For vectors u (length I) and v (length J), computes matrix M where
/// `M[i,j] = u[i] × v[j]`
///
/// # Arguments
///
/// * `u` - First vector with length I
/// * `v` - Second vector with length J
///
/// # Returns
///
/// A matrix with shape (I, J)
///
/// # Complexity
///
/// Time: O(I × J)
/// Space: O(I × J)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::outer_product_2;
///
/// let u = array![1.0, 2.0, 3.0];
/// let v = array![4.0, 5.0];
/// let m = outer_product_2(&u.view(), &v.view());
///
/// assert_eq!(m.shape(), &[3, 2]);
/// assert_eq!(m[[0, 0]], 4.0);   // 1*4
/// assert_eq!(m[[0, 1]], 5.0);   // 1*5
/// assert_eq!(m[[1, 0]], 8.0);   // 2*4
/// assert_eq!(m[[2, 1]], 15.0);  // 3*5
/// ```
pub fn outer_product_2<T>(u: &ArrayView1<T>, v: &ArrayView1<T>) -> Array2<T>
where
    T: Clone + Num,
{
    let i = u.len();
    let j = v.len();

    let mut result = Array2::<T>::zeros((i, j));

    for (row, u_val) in u.iter().enumerate() {
        for (col, v_val) in v.iter().enumerate() {
            result[[row, col]] = u_val.clone() * v_val.clone();
        }
    }

    result
}

/// Compute the outer product of multiple vectors to form an N-D tensor
///
/// For vectors v₁, v₂, ..., vₙ with lengths I₁, I₂, ..., Iₙ, computes tensor T where
/// `T[i₁, i₂, ..., iₙ] = v₁[i₁] × v₂[i₂] × ... × vₙ[iₙ]`
///
/// This is the core operation for reconstructing CP decompositions from factor vectors.
///
/// # Arguments
///
/// * `vectors` - Slice of vectors to compute outer product over
///
/// # Returns
///
/// An N-dimensional tensor where N is the number of input vectors
///
/// # Errors
///
/// Returns error if no vectors are provided
///
/// # Complexity
///
/// Time: O(∏ᵢ Iᵢ) where Iᵢ is the length of vector i
/// Space: O(∏ᵢ Iᵢ)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::outer_product;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![3.0, 4.0, 5.0];
/// let v3 = array![6.0, 7.0];
///
/// let tensor = outer_product(&[v1.view(), v2.view(), v3.view()]).unwrap();
/// assert_eq!(tensor.shape(), &[2, 3, 2]);
///
/// // T[0,0,0] = 1.0 * 3.0 * 6.0 = 18.0
/// assert_eq!(tensor[[0, 0, 0]], 18.0);
/// ```
pub fn outer_product<T>(vectors: &[ArrayView1<T>]) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num,
{
    if vectors.is_empty() {
        anyhow::bail!("Need at least one vector for outer product");
    }

    // Compute output shape
    let shape: Vec<usize> = vectors.iter().map(|v| v.len()).collect();
    let total_size: usize = shape.iter().product();

    // Allocate output tensor
    let mut result = Array::<T, IxDyn>::zeros(IxDyn(&shape));

    // Compute outer product by iterating through all multi-indices
    for flat_idx in 0..total_size {
        let multi_idx = flat_to_multi_index(flat_idx, &shape);

        // Compute product of all vector elements at this index
        let mut prod = T::one();
        for (dim, &idx) in multi_idx.iter().enumerate() {
            prod = prod * vectors[dim][idx].clone();
        }

        result[&multi_idx[..]] = prod;
    }

    Ok(result)
}

/// Compute weighted outer product of multiple vectors (for CP reconstruction)
///
/// For vectors v₁, v₂, ..., vₙ and weight λ, computes tensor T where
/// `T[i₁, i₂, ..., iₙ] = λ × v₁[i₁] × v₂[i₂] × ... × vₙ[iₙ]`
///
/// This is used in CP decomposition reconstruction where each rank-1 component
/// has an associated weight.
///
/// # Arguments
///
/// * `vectors` - Slice of vectors (one per mode)
/// * `weight` - Scalar weight for this component
///
/// # Returns
///
/// An N-dimensional tensor
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::outer_product_weighted;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![3.0, 4.0];
/// let weight = 2.0;
///
/// let tensor = outer_product_weighted(&[v1.view(), v2.view()], weight).unwrap();
/// assert_eq!(tensor.shape(), &[2, 2]);
/// assert_eq!(tensor[[0, 0]], 6.0);  // 2.0 * 1.0 * 3.0
/// ```
pub fn outer_product_weighted<T>(vectors: &[ArrayView1<T>], weight: T) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num,
{
    let mut result = outer_product(vectors)?;
    // Multiply each element by weight
    result.mapv_inplace(|x| x * weight.clone());
    Ok(result)
}

/// Compute the sum of outer products for CP reconstruction
///
/// For factor matrices A₁, A₂, ..., Aₙ (each Iₖ × R) and optional weights λ,
/// reconstructs the tensor as:
/// `T = ∑ᵣ λᵣ × (A₁[:,r] ⊗ A₂[:,r] ⊗ ... ⊗ Aₙ[:,r])`
///
/// This is the standard CP decomposition reconstruction.
///
/// # Arguments
///
/// * `factors` - Factor matrices (one per mode)
/// * `weights` - Optional weights for each rank-1 component (defaults to 1.0)
///
/// # Returns
///
/// Reconstructed N-dimensional tensor
///
/// # Errors
///
/// Returns error if:
/// - No factors provided
/// - Factors have different numbers of columns (ranks)
/// - Number of weights doesn't match rank
///
/// # Complexity
///
/// Time: O(R × ∏ᵢ Iᵢ) where R is rank
/// Space: O(∏ᵢ Iᵢ)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::cp_reconstruct;
///
/// // Rank-2 CP decomposition of 2×3 matrix
/// let a1 = array![[1.0, 0.0], [0.0, 1.0]];  // 2×2
/// let a2 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];  // 3×2
///
/// let tensor = cp_reconstruct(&[a1.view(), a2.view()], None).unwrap();
/// assert_eq!(tensor.shape(), &[2, 3]);
/// ```
pub fn cp_reconstruct<T>(
    factors: &[ArrayView2<T>],
    weights: Option<&ArrayView1<T>>,
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num,
{
    if factors.is_empty() {
        anyhow::bail!("Need at least one factor matrix for CP reconstruction");
    }

    let rank = factors[0].shape()[1];

    // Validate all factors have same rank
    for (i, factor) in factors.iter().enumerate() {
        if factor.shape()[1] != rank {
            anyhow::bail!(
                "Factor {} has {} columns, expected {}",
                i,
                factor.shape()[1],
                rank
            );
        }
    }

    // Validate weights if provided
    if let Some(w) = weights {
        if w.len() != rank {
            anyhow::bail!("Weights length {} must match rank {}", w.len(), rank);
        }
    }

    // Compute tensor shape
    let shape: Vec<usize> = factors.iter().map(|f| f.shape()[0]).collect();
    let mut result = Array::<T, IxDyn>::zeros(IxDyn(&shape));

    // Sum over all rank-1 components
    for r in 0..rank {
        // Extract r-th column from each factor matrix
        let vectors: Vec<Array1<T>> = factors.iter().map(|f| f.column(r).to_owned()).collect();
        let vector_views: Vec<ArrayView1<T>> = vectors.iter().map(|v| v.view()).collect();

        // Compute weighted outer product
        let weight = weights.map(|w| w[r].clone()).unwrap_or_else(T::one);
        let component = outer_product_weighted(&vector_views, weight)?;

        // Add to result
        result = result + component;
    }

    Ok(result)
}

/// Convert flat index to multi-dimensional index
fn flat_to_multi_index(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut multi_idx = vec![0; shape.len()];
    for (dim, &size) in shape.iter().enumerate().rev() {
        multi_idx[dim] = flat_idx % size;
        flat_idx /= size;
    }
    multi_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_outer_product_2() {
        let u = array![1.0, 2.0, 3.0];
        let v = array![4.0, 5.0];
        let m = outer_product_2(&u.view(), &v.view());

        assert_eq!(m.shape(), &[3, 2]);
        assert_eq!(m[[0, 0]], 4.0);
        assert_eq!(m[[0, 1]], 5.0);
        assert_eq!(m[[1, 0]], 8.0);
        assert_eq!(m[[1, 1]], 10.0);
        assert_eq!(m[[2, 0]], 12.0);
        assert_eq!(m[[2, 1]], 15.0);
    }

    #[test]
    fn test_outer_product_3d() {
        let v1 = array![1.0, 2.0];
        let v2 = array![3.0, 4.0];
        let v3 = array![5.0, 6.0];

        let tensor = outer_product(&[v1.view(), v2.view(), v3.view()]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2, 2]);

        // Check a few values
        assert_eq!(tensor[[0, 0, 0]], 15.0); // 1*3*5
        assert_eq!(tensor[[0, 0, 1]], 18.0); // 1*3*6
        assert_eq!(tensor[[1, 1, 1]], 48.0); // 2*4*6
    }

    #[test]
    fn test_outer_product_weighted() {
        let v1 = array![1.0, 2.0];
        let v2 = array![3.0, 4.0];
        let weight = 2.0;

        let tensor = outer_product_weighted(&[v1.view(), v2.view()], weight).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor[[0, 0]], 6.0); // 2*1*3
        assert_eq!(tensor[[0, 1]], 8.0); // 2*1*4
        assert_eq!(tensor[[1, 0]], 12.0); // 2*2*3
        assert_eq!(tensor[[1, 1]], 16.0); // 2*2*4
    }

    #[test]
    fn test_cp_reconstruct_rank1() {
        // Rank-1 reconstruction: single outer product
        let a1 = array![[2.0], [3.0]];
        let a2 = array![[4.0], [5.0], [6.0]];

        let tensor = cp_reconstruct(&[a1.view(), a2.view()], None).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);

        assert_eq!(tensor[[0, 0]], 8.0); // 2*4
        assert_eq!(tensor[[0, 1]], 10.0); // 2*5
        assert_eq!(tensor[[0, 2]], 12.0); // 2*6
        assert_eq!(tensor[[1, 0]], 12.0); // 3*4
        assert_eq!(tensor[[1, 1]], 15.0); // 3*5
        assert_eq!(tensor[[1, 2]], 18.0); // 3*6
    }

    #[test]
    fn test_cp_reconstruct_rank2() {
        // Rank-2 reconstruction
        let a1 = array![[1.0, 0.0], [0.0, 1.0]];
        let a2 = array![[1.0, 0.0], [0.0, 1.0]];

        let tensor = cp_reconstruct(&[a1.view(), a2.view()], None).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);

        // Should reconstruct identity matrix
        assert_eq!(tensor[[0, 0]], 1.0);
        assert_eq!(tensor[[0, 1]], 0.0);
        assert_eq!(tensor[[1, 0]], 0.0);
        assert_eq!(tensor[[1, 1]], 1.0);
    }

    #[test]
    fn test_cp_reconstruct_with_weights() {
        let a1 = array![[1.0, 2.0], [3.0, 4.0]];
        let a2 = array![[5.0, 6.0], [7.0, 8.0]];
        let weights = array![2.0, 3.0];

        let tensor = cp_reconstruct(&[a1.view(), a2.view()], Some(&weights.view())).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);

        // T[0,0] = 2.0*(1*5) + 3.0*(2*6) = 10 + 36 = 46
        assert_eq!(tensor[[0, 0]], 46.0);
    }

    #[test]
    fn test_flat_to_multi_index() {
        let shape = vec![2, 3, 4];

        assert_eq!(flat_to_multi_index(0, &shape), vec![0, 0, 0]);
        assert_eq!(flat_to_multi_index(1, &shape), vec![0, 0, 1]);
        assert_eq!(flat_to_multi_index(4, &shape), vec![0, 1, 0]);
        assert_eq!(flat_to_multi_index(23, &shape), vec![1, 2, 3]);
    }

    #[test]
    fn test_outer_product_single_vector() {
        let v = array![1.0, 2.0, 3.0];
        let tensor = outer_product(&[v.view()]).unwrap();

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor[[0]], 1.0);
        assert_eq!(tensor[[1]], 2.0);
        assert_eq!(tensor[[2]], 3.0);
    }

    #[test]
    #[should_panic(expected = "at least one vector")]
    fn test_outer_product_empty() {
        let _ = outer_product::<f64>(&[]).unwrap();
    }

    #[test]
    #[should_panic(expected = "columns")]
    fn test_cp_reconstruct_mismatched_ranks() {
        let a1 = array![[1.0, 2.0], [3.0, 4.0]]; // Rank 2
        let a2 = array![[5.0], [6.0]]; // Rank 1

        cp_reconstruct(&[a1.view(), a2.view()], None).unwrap();
    }
}
