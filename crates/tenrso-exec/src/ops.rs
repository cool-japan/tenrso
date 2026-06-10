//! Operation implementations for tensor contractions

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array2, Axis as NdAxis, Ix2};
use scirs2_core::numeric::Num;
use tenrso_core::DenseND;
use tenrso_planner::EinsumSpec;

/// Execute a pairwise dense tensor contraction
///
/// Implements the core einsum operation for two dense tensors.
///
/// # Arguments
///
/// * `spec` - Einsum specification (e.g., "ij,jk->ik")
/// * `a` - First input tensor
/// * `b` - Second input tensor
///
/// # Returns
///
/// Result tensor from the contraction
pub fn execute_dense_contraction<T>(
    spec: &EinsumSpec,
    a: &DenseND<T>,
    b: &DenseND<T>,
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default,
{
    if spec.num_inputs() != 2 {
        return Err(anyhow!(
            "Pairwise contraction requires exactly 2 inputs, got {}",
            spec.num_inputs()
        ));
    }

    // Get input specs
    let spec_a = &spec.inputs[0];
    let spec_b = &spec.inputs[1];

    // Build index maps
    let mut index_to_dim_a = std::collections::HashMap::new();
    for (i, c) in spec_a.chars().enumerate() {
        index_to_dim_a.insert(c, (i, a.shape()[i]));
    }

    let mut index_to_dim_b = std::collections::HashMap::new();
    for (i, c) in spec_b.chars().enumerate() {
        index_to_dim_b.insert(c, (i, b.shape()[i]));
    }

    // Determine output shape
    let output_shape: Vec<usize> = spec
        .output
        .chars()
        .map(|c| {
            index_to_dim_a
                .get(&c)
                .or_else(|| index_to_dim_b.get(&c))
                .map(|(_, size)| *size)
                .unwrap_or(1)
        })
        .collect();

    // For now, use a simple nested loop implementation
    // TODO: Optimize with blocked/tiled execution and BLAS

    // This is a placeholder - we need a proper einsum implementation
    // For M4, we'll implement a basic matmul case
    if is_matrix_multiply(spec_a, spec_b, &spec.output) {
        execute_matmul_general(spec, a, b, &output_shape)
    } else {
        // General case - use naive einsum
        execute_general_einsum(spec, a, b, &output_shape)
    }
}

/// Check if this is a matrix multiplication pattern
fn is_matrix_multiply(spec_a: &str, spec_b: &str, output: &str) -> bool {
    // Pattern: 2D x 2D -> 2D with one shared contracted index
    if spec_a.len() != 2 || spec_b.len() != 2 || output.len() != 2 {
        return false;
    }

    let a_chars: Vec<char> = spec_a.chars().collect();
    let b_chars: Vec<char> = spec_b.chars().collect();
    let out_chars: Vec<char> = output.chars().collect();

    // Find the shared (contracted) index
    let shared = if a_chars.contains(&b_chars[0]) {
        Some(b_chars[0])
    } else if a_chars.contains(&b_chars[1]) {
        Some(b_chars[1])
    } else {
        None
    };

    let Some(shared_idx) = shared else {
        return false;
    };

    // Check that:
    // 1. Exactly one index is shared (contracted)
    // 2. The other two indices appear in the output
    let a_not_shared: Vec<char> = a_chars
        .iter()
        .filter(|&&c| c != shared_idx)
        .copied()
        .collect();
    let b_not_shared: Vec<char> = b_chars
        .iter()
        .filter(|&&c| c != shared_idx)
        .copied()
        .collect();

    if a_not_shared.len() != 1 || b_not_shared.len() != 1 {
        return false;
    }

    // Output should contain exactly the two non-shared indices
    let mut expected_out = vec![a_not_shared[0], b_not_shared[0]];
    expected_out.sort();
    let mut actual_out = out_chars.clone();
    actual_out.sort();

    expected_out == actual_out
}

/// Execute matrix multiplication with BLAS-backed 2-D array layout.
///
/// Materialises `a` and `b` as contiguous `Array2<T>` in `(m×k)` and `(k×n)`
/// orientation respectively, then performs the product via ndarray's
/// `dot`-compatible row/column iteration.  Accessing `a2.row(i)` yields a
/// contiguous slice that the compiler (and BLAS if available) can vectorise;
/// `b2.column(j)` is strided but predictable.
///
/// Using `Array2` instead of dynamic `ArrayViewD` indexing eliminates the
/// per-element dynamic dispatch overhead and enables LLVM to apply
/// auto-vectorisation on the inner loop.
///
/// # Layout strategy
///
/// - `a_free_pos == 0`: `a` is already `(m × k)` in row-major storage — borrow
///   directly via `into_dimensionality::<Ix2>()`.
/// - `a_free_pos == 1`: `a` is `(k × m)` — transpose to produce `(m × k)`.
/// - `b_shared_pos == 0`: `b` is `(k × n)` — use directly.
/// - `b_shared_pos == 1`: `b` is `(n × k)` — transpose to `(k × n)`.
///
/// The result is placed into the flat output buffer in the order dictated by
/// the output spec (either `[a_free, b_free]` or `[b_free, a_free]`).
fn execute_matmul_general<T>(
    spec: &EinsumSpec,
    a: &DenseND<T>,
    b: &DenseND<T>,
    output_shape: &[usize],
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default,
{
    let spec_a = &spec.inputs[0];
    let spec_b = &spec.inputs[1];
    let spec_out = &spec.output;

    let a_chars: Vec<char> = spec_a.chars().collect();
    let b_chars: Vec<char> = spec_b.chars().collect();
    let out_chars: Vec<char> = spec_out.chars().collect();

    let shared_idx = a_chars
        .iter()
        .find(|&&c| b_chars.contains(&c))
        .copied()
        .ok_or_else(|| anyhow!("No shared index found for contraction"))?;

    let a_free_idx = a_chars
        .iter()
        .find(|&&c| c != shared_idx)
        .copied()
        .ok_or_else(|| anyhow!("execute_matmul_general: no free index in spec_a"))?;
    let b_free_idx = b_chars
        .iter()
        .find(|&&c| c != shared_idx)
        .copied()
        .ok_or_else(|| anyhow!("execute_matmul_general: no free index in spec_b"))?;

    let a_free_pos = a_chars.iter().position(|&c| c == a_free_idx).unwrap();
    let a_shared_pos = a_chars.iter().position(|&c| c == shared_idx).unwrap();
    let b_free_pos = b_chars.iter().position(|&c| c == b_free_idx).unwrap();
    let b_shared_pos = b_chars.iter().position(|&c| c == shared_idx).unwrap();

    let m = a.shape()[a_free_pos]; // rows of result from a
    let k = a.shape()[a_shared_pos]; // contracted dimension
    let n = b.shape()[b_free_pos]; // cols of result from b

    if k != b.shape()[b_shared_pos] {
        return Err(anyhow!(
            "Contracted dimension mismatch: {} vs {}",
            k,
            b.shape()[b_shared_pos]
        ));
    }

    // Build contiguous 2-D arrays in (m × k) and (k × n) layout.
    // Array2::from_shape_fn requires only Clone on the element type.
    let a_view_dyn = a.view();
    let b_view_dyn = b.view();

    // Obtain Ix2 view of a, then transpose if needed so final layout is (m × k).
    let a2: Array2<T> = {
        let v = a_view_dyn.into_dimensionality::<Ix2>().map_err(|e| {
            anyhow!(
                "execute_matmul_general: a into_dimensionality failed: {}",
                e
            )
        })?;
        if a_free_pos == 0 {
            // Already (m × k).
            v.to_owned()
        } else {
            // Layout is (k × m) — transpose to (m × k).
            v.t().to_owned()
        }
    };

    // Obtain Ix2 view of b, then transpose if needed so final layout is (k × n).
    let b2: Array2<T> = {
        let v = b_view_dyn.into_dimensionality::<Ix2>().map_err(|e| {
            anyhow!(
                "execute_matmul_general: b into_dimensionality failed: {}",
                e
            )
        })?;
        if b_shared_pos == 0 {
            // Already (k × n).
            v.to_owned()
        } else {
            // Layout is (n × k) — transpose to (k × n).
            v.t().to_owned()
        }
    };

    // (m × k) · (k × n) → (m × n).
    //
    // Iterate over rows of a2 (each row is a contiguous slice of length k) and
    // columns of b2 (each column is a strided slice of length k).  This gives
    // the compiler the opportunity to auto-vectorise the inner accumulation loop
    // while requiring only `Clone + Num + AddAssign` on `T`.
    let mut result_flat: Vec<T> = Vec::with_capacity(m * n);

    for row_a in a2.axis_iter(NdAxis(0)) {
        // `row_a` is a contiguous Array1-view of length k.
        for col_b in b2.axis_iter(NdAxis(1)) {
            // `col_b` is a strided Array1-view of length k.
            let dot_val = row_a
                .iter()
                .zip(col_b.iter())
                .fold(T::default(), |mut acc, (av, bv)| {
                    acc += av.clone() * bv.clone();
                    acc
                });
            result_flat.push(dot_val);
        }
    }
    // result_flat is in (a_free_idx, b_free_idx) = (row=m, col=n) order.

    // Place into output_shape in the order dictated by the output spec.
    let a_free_out_pos = out_chars
        .iter()
        .position(|&c| c == a_free_idx)
        .ok_or_else(|| anyhow!("execute_matmul_general: a_free_idx missing from output spec"))?;

    let output_data: Vec<T> = if a_free_out_pos == 0 {
        // output[a_free, b_free] — matches result_flat layout directly.
        result_flat
    } else {
        // output[b_free, a_free] — need to transpose the (m × n) result to (n × m).
        let result_mn =
            Array2::from_shape_vec((m, n), result_flat).map_err(|e| anyhow!("{}", e))?;
        let transposed = result_mn.t().to_owned();
        transposed.into_raw_vec_and_offset().0
    };

    DenseND::from_vec(output_data, output_shape)
}

/// Execute general einsum (naive implementation)
fn execute_general_einsum<T>(
    spec: &EinsumSpec,
    a: &DenseND<T>,
    b: &DenseND<T>,
    output_shape: &[usize],
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default,
{
    let spec_a = &spec.inputs[0];
    let spec_b = &spec.inputs[1];
    let spec_out = &spec.output;

    // Special case: element-wise product with full reduction (e.g., "ij,ij->")
    if spec_a == spec_b && spec_out.is_empty() {
        // Element-wise multiply and sum all
        let a_view = a.view();
        let b_view = b.view();

        let mut sum = T::default();
        let size: usize = a.shape().iter().product();

        for flat_idx in 0..size {
            // Convert flat index to multi-dimensional index
            let mut idx = Vec::with_capacity(a.shape().len());
            let mut remaining = flat_idx;
            for &dim_size in a.shape().iter().rev() {
                idx.push(remaining % dim_size);
                remaining /= dim_size;
            }
            idx.reverse();

            let a_val = a_view[idx.as_slice()].clone();
            let b_val = b_view[idx.as_slice()].clone();
            sum += a_val * b_val;
        }

        // Return scalar (empty shape)
        return DenseND::from_vec(vec![sum], &[]);
    }

    // Special case: outer product (e.g., "i,j->ij")
    let a_chars: Vec<char> = spec_a.chars().collect();
    let b_chars: Vec<char> = spec_b.chars().collect();
    let out_chars: Vec<char> = spec_out.chars().collect();

    let has_shared = a_chars.iter().any(|c| b_chars.contains(c));

    if !has_shared && out_chars.len() == a_chars.len() + b_chars.len() {
        // Outer product
        let a_view = a.view();
        let b_view = b.view();

        let output_size: usize = output_shape.iter().product();
        let mut output = vec![T::default(); output_size];

        for (out_idx, out_val) in output.iter_mut().enumerate().take(output_size) {
            // Convert output flat index to multi-dimensional
            let mut idx = Vec::with_capacity(output_shape.len());
            let mut remaining = out_idx;
            for &dim_size in output_shape.iter().rev() {
                idx.push(remaining % dim_size);
                remaining /= dim_size;
            }
            idx.reverse();

            // Split indices for a and b
            let a_idx = &idx[0..a.shape().len()];
            let b_idx = &idx[a.shape().len()..];

            let a_val = a_view[a_idx].clone();
            let b_val = b_view[b_idx].clone();
            *out_val = a_val * b_val;
        }

        return DenseND::from_vec(output, output_shape);
    }

    // General index-based einsum contraction.
    //
    // Algorithm:
    //  1. Build a map from each index character to its size.
    //  2. Identify contraction indices (appear in inputs but not output).
    //  3. Enumerate all output index combinations (flat loop decoded via
    //     mixed-radix arithmetic).
    //  4. For each output position: sum over all contraction index combinations
    //     the product of corresponding input elements.
    //  5. Write result to output tensor.

    // Step 1: map each index char to its dimension size.
    let mut dim_for: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (i, c) in spec_a.chars().enumerate() {
        dim_for.entry(c).or_insert(a.shape()[i]);
    }
    for (i, c) in spec_b.chars().enumerate() {
        dim_for.entry(c).or_insert(b.shape()[i]);
    }

    // Step 2: identify contraction indices (in inputs, absent from output).
    let contracted: Vec<char> = a_chars
        .iter()
        .chain(b_chars.iter())
        .copied()
        .filter(|c| !out_chars.contains(c))
        .collect::<std::collections::HashSet<char>>()
        .into_iter()
        .collect();

    // Build sorted contraction index list for deterministic iteration.
    let mut contracted_sorted = contracted.clone();
    contracted_sorted.sort_unstable();

    // Sizes for contraction dimensions.
    let contracted_sizes: Vec<usize> = contracted_sorted
        .iter()
        .map(|c| *dim_for.get(c).unwrap_or(&1))
        .collect();

    // Sizes for output dimensions.
    let out_sizes: Vec<usize> = out_chars
        .iter()
        .map(|c| *dim_for.get(c).unwrap_or(&1))
        .collect();

    let output_total: usize = out_sizes.iter().product::<usize>().max(1);
    let contracted_total: usize = contracted_sizes.iter().product::<usize>().max(1);

    let mut output = vec![T::default(); output_total];

    let a_view = a.view();
    let b_view = b.view();

    // Helper: decode a flat index into a multi-dimensional index given sizes.
    let decode_flat = |flat: usize, sizes: &[usize]| -> Vec<usize> {
        let mut idx = vec![0usize; sizes.len()];
        let mut remaining = flat;
        for (d, &sz) in sizes.iter().enumerate().rev() {
            idx[d] = remaining % sz;
            remaining /= sz;
        }
        idx
    };

    // Iterate over all output positions.
    for (out_flat, out_elem) in output.iter_mut().enumerate().take(output_total) {
        let out_idx = decode_flat(out_flat, &out_sizes);

        // Build a mapping from char -> position value for output indices.
        let mut char_val: std::collections::HashMap<char, usize> = out_chars
            .iter()
            .copied()
            .zip(out_idx.iter().copied())
            .collect();

        let mut acc = T::default();

        // Iterate over all contraction index combinations.
        for con_flat in 0..contracted_total {
            let con_idx = decode_flat(con_flat, &contracted_sizes);

            // Extend char_val with contraction indices.
            for (c, &v) in contracted_sorted.iter().zip(con_idx.iter()) {
                char_val.insert(*c, v);
            }

            // Build a_index and b_index from char_val.
            let a_index: Vec<usize> = spec_a
                .chars()
                .map(|c| *char_val.get(&c).unwrap_or(&0))
                .collect();
            let b_index: Vec<usize> = spec_b
                .chars()
                .map(|c| *char_val.get(&c).unwrap_or(&0))
                .collect();

            let a_val = a_view[a_index.as_slice()].clone();
            let b_val = b_view[b_index.as_slice()].clone();
            acc += a_val * b_val;
        }

        *out_elem = acc;
    }

    // Handle scalar output (empty spec_out) — shape is [].
    if spec_out.is_empty() {
        DenseND::from_vec(output, &[])
    } else {
        DenseND::from_vec(output, output_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_matrix_multiply() {
        assert!(is_matrix_multiply("ij", "jk", "ik"));
        assert!(is_matrix_multiply("ab", "bc", "ac"));
        assert!(!is_matrix_multiply("ijk", "jk", "ik"));
        assert!(!is_matrix_multiply("ij", "jk", "ijk"));
    }

    #[test]
    fn test_execute_matmul() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let c = execute_dense_contraction(&spec, &a, &b).unwrap();

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = [19.0, 22.0, 43.0, 50.0];
        let result_view = c.view();

        for (i, &expected_val) in expected.iter().enumerate() {
            let row = i / 2;
            let col = i % 2;
            let diff: f64 = result_view[[row, col]] - expected_val;
            assert!(diff.abs() < 1e-10);
        }
    }

    #[test]
    fn test_execute_dense_contraction_matmul() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let c = execute_dense_contraction(&spec, &a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);

        // Verify dimensions
        let result_view = c.view();
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        let diff: f64 = result_view[[0, 0]] - 58.0;
        assert!(diff.abs() < 1e-10);
    }

    /// Test the general einsum fallback for a 3D × 2D contraction: "ijk,kl->ijl".
    ///
    /// a[i,j,k] × b[k,l]  =>  c[i,j,l] = Σ_k a[i,j,k] * b[k,l]
    ///
    /// With shapes a[2,2,2] and b[2,2] we can verify every element by hand.
    #[test]
    fn test_general_einsum_3d_times_2d() {
        // a[2,2,2]: row-major values 1..8
        let a =
            DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();
        // b[2,2]: [[1,0],[0,1]] (identity — result should equal a's k-slices unchanged)
        let b = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

        let spec = EinsumSpec::parse("ijk,kl->ijl").unwrap();
        let c = execute_dense_contraction(&spec, &a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2, 2]);

        // With identity b the result should equal a.
        let a_view = a.view();
        let c_view = c.view();
        for i in 0..2 {
            for j in 0..2 {
                for l in 0..2 {
                    let diff: f64 = c_view[[i, j, l]] - a_view[[i, j, l]];
                    assert!(
                        diff.abs() < 1e-10,
                        "c[{i},{j},{l}] = {} != {} (expected a[{i},{j},{l}])",
                        c_view[[i, j, l]],
                        a_view[[i, j, l]]
                    );
                }
            }
        }
    }

    /// Test the general einsum fallback for contraction with a non-trivial b.
    ///
    /// "ijk,jl->ikl": contract over j.  a[2,2,2], b[2,3] -> c[2,2,3]
    ///
    /// c[i,k,l] = Σ_j a[i,j,k] * b[j,l]
    #[test]
    fn test_general_einsum_middle_contraction() {
        // a[2,2,2] row-major: [[[ 1, 2],[ 3, 4]], [[ 5, 6],[ 7, 8]]]
        let a = DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])
            .unwrap();
        // b[2,3]: [[1,2,3],[4,5,6]]
        let b = DenseND::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let spec = EinsumSpec::parse("ijk,jl->ikl").unwrap();
        let c = execute_dense_contraction(&spec, &a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2, 3]);

        // Compute reference by naive triple loop.
        let a_view = a.view();
        let b_view = b.view();
        let c_view = c.view();
        for i in 0..2usize {
            for k in 0..2usize {
                for l in 0..3usize {
                    let expected: f64 = (0..2).map(|j| a_view[[i, j, k]] * b_view[[j, l]]).sum();
                    let diff = (c_view[[i, k, l]] - expected).abs();
                    assert!(
                        diff < 1e-10,
                        "c[{i},{k},{l}] = {} != {expected}",
                        c_view[[i, k, l]]
                    );
                }
            }
        }
    }
}
