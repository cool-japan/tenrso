//! Operation implementations for tensor contractions

use anyhow::{anyhow, Result};
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

    if shared.is_none() {
        return false;
    }

    let shared_idx = shared.unwrap();

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

/// Execute matrix multiplication with general index ordering
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

    // Find the contracted (shared) index
    let a_chars: Vec<char> = spec_a.chars().collect();
    let b_chars: Vec<char> = spec_b.chars().collect();

    let shared_idx = a_chars
        .iter()
        .find(|&&c| b_chars.contains(&c))
        .copied()
        .ok_or_else(|| anyhow!("No shared index found for contraction"))?;

    // Find non-shared indices and their positions
    let a_free_idx = a_chars.iter().find(|&&c| c != shared_idx).copied().unwrap();
    let b_free_idx = b_chars.iter().find(|&&c| c != shared_idx).copied().unwrap();

    // Find positions in tensors
    let a_free_pos = a_chars.iter().position(|&c| c == a_free_idx).unwrap();
    let a_shared_pos = a_chars.iter().position(|&c| c == shared_idx).unwrap();
    let b_free_pos = b_chars.iter().position(|&c| c == b_free_idx).unwrap();
    let b_shared_pos = b_chars.iter().position(|&c| c == shared_idx).unwrap();

    // Get dimensions
    let a_free_dim = a.shape()[a_free_pos];
    let b_free_dim = b.shape()[b_free_pos];
    let contracted_dim = a.shape()[a_shared_pos];

    // Verify contracted dimension matches
    if contracted_dim != b.shape()[b_shared_pos] {
        return Err(anyhow!(
            "Contracted dimension mismatch: {} vs {}",
            contracted_dim,
            b.shape()[b_shared_pos]
        ));
    }

    // Find output order
    let out_chars: Vec<char> = spec_out.chars().collect();
    let a_free_out_pos = out_chars.iter().position(|&c| c == a_free_idx).unwrap();
    let _b_free_out_pos = out_chars.iter().position(|&c| c == b_free_idx).unwrap();

    let mut output = vec![T::default(); output_shape.iter().product()];
    let a_view = a.view();
    let b_view = b.view();

    // Perform contraction
    for a_i in 0..a_free_dim {
        for b_i in 0..b_free_dim {
            let mut sum = T::default();
            for k in 0..contracted_dim {
                // Build indices for a and b based on their actual layout
                let a_idx = if a_free_pos == 0 { [a_i, k] } else { [k, a_i] };
                let b_idx = if b_free_pos == 0 { [b_i, k] } else { [k, b_i] };

                let a_val = a_view[a_idx].clone();
                let b_val = b_view[b_idx].clone();
                sum += a_val * b_val;
            }

            // Write to output in correct order
            let out_idx = if a_free_out_pos == 0 {
                a_i * b_free_dim + b_i
            } else {
                b_i * a_free_dim + a_i
            };
            output[out_idx] = sum;
        }
    }

    DenseND::from_vec(output, output_shape)
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

    // TODO: Implement full general einsum
    // For now, return error for unsupported cases
    Err(anyhow!(
        "General einsum not yet fully implemented for spec: {},{},->{}",
        spec_a,
        spec_b,
        spec_out
    ))
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
}
