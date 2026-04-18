//! TT arithmetic operations: addition, inner product, Hadamard product.
//!
//! Each operation acts directly on TT cores without materialising the full
//! tensor, preserving the O(sum of core sizes) storage advantage of the
//! TT format.
//!
//! All array operations use `scirs2_core::ndarray_ext`.

use scirs2_core::ndarray_ext::{Array2, Array3};
use scirs2_core::numeric::{Float, NumAssign, NumCast};

use super::types::{TTDecomp, TTError};

/// Addition of two TT decompositions
///
/// Computes the TT decomposition of X + Y where X and Y are in TT format.
///
/// # Algorithm
///
/// For TT decompositions with cores Gₖ and Hₖ, the sum has cores:
/// - First core: [G₁ H₁]
/// - Middle cores: [[Gₖ 0], [0 Hₖ]]
/// - Last core: [Gₙ; Hₙ] (vertical concatenation)
///
/// The resulting TT-ranks are r₁ + s₁, r₂ + s₂, ..., rₙ₋₁ + sₙ₋₁
/// where rᵢ and sᵢ are the ranks of X and Y respectively.
///
/// # Arguments
///
/// * `tt1` - First TT decomposition
/// * `tt2` - Second TT decomposition
///
/// # Returns
///
/// New TTDecomp representing the sum
///
/// # Errors
///
/// Returns error if tensors have different shapes
///
/// # Complexity
///
/// Time: O(N) where N is the number of cores
/// Space: O(R₁ × R₂) where R₁, R₂ are max TT-ranks
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::{tt_svd, tt_add};
///
/// let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
/// let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
///
/// let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
/// let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();
///
/// let tt_sum = tt_add(&tt1, &tt2).unwrap();
/// println!("Sum TT-ranks: {:?}", tt_sum.ranks);
/// ```
pub fn tt_add<T>(tt1: &TTDecomp<T>, tt2: &TTDecomp<T>) -> Result<TTDecomp<T>, TTError>
where
    T: Float + NumCast + 'static,
{
    // Validate shapes match
    if tt1.shape != tt2.shape {
        return Err(TTError::ShapeMismatch(format!(
            "Shape mismatch: {:?} vs {:?}",
            tt1.shape, tt2.shape
        )));
    }

    let n_modes = tt1.cores.len();
    if n_modes != tt2.cores.len() {
        return Err(TTError::ShapeMismatch(format!(
            "Number of modes mismatch: {} vs {}",
            n_modes,
            tt2.cores.len()
        )));
    }

    let mut new_cores = Vec::with_capacity(n_modes);
    let mut new_ranks = Vec::with_capacity(n_modes - 1);

    for k in 0..n_modes {
        let core1 = &tt1.cores[k];
        let core2 = &tt2.cores[k];

        let (r1_left, n1, r1_right) = core1.dim();
        let (r2_left, n2, r2_right) = core2.dim();

        if n1 != n2 {
            return Err(TTError::ShapeMismatch(format!(
                "Mode {} size mismatch: {} vs {}",
                k, n1, n2
            )));
        }

        let n_k = n1;

        // Handle boundary conditions
        if k == 0 {
            // First core: both left ranks should be 1
            if r1_left != 1 || r2_left != 1 {
                return Err(TTError::ShapeMismatch(format!(
                    "First cores must have left rank 1, got {} and {}",
                    r1_left, r2_left
                )));
            }

            let r_right = r1_right + r2_right;
            // Concatenate horizontally: [G₁ H₁] with shape (1, n, r1_right + r2_right)
            let mut new_core = Array3::<T>::zeros((1, n_k, r_right));

            for j in 0..n_k {
                for l in 0..r1_right {
                    new_core[[0, j, l]] = core1[[0, j, l]];
                }
                for l in 0..r2_right {
                    new_core[[0, j, r1_right + l]] = core2[[0, j, l]];
                }
            }

            new_cores.push(new_core);
            if k < n_modes - 1 {
                new_ranks.push(r_right);
            }
        } else if k == n_modes - 1 {
            // Last core: both right ranks should be 1
            if r1_right != 1 || r2_right != 1 {
                return Err(TTError::ShapeMismatch(format!(
                    "Last cores must have right rank 1, got {} and {}",
                    r1_right, r2_right
                )));
            }

            let r_left = r1_left + r2_left;
            // Concatenate vertically: [Gₙ; Hₙ] with shape (r1_left + r2_left, n, 1)
            let mut new_core = Array3::<T>::zeros((r_left, n_k, 1));

            for i in 0..r1_left {
                for j in 0..n_k {
                    new_core[[i, j, 0]] = core1[[i, j, 0]];
                }
            }
            for i in 0..r2_left {
                for j in 0..n_k {
                    new_core[[r1_left + i, j, 0]] = core2[[i, j, 0]];
                }
            }

            new_cores.push(new_core);
        } else {
            // Middle cores: block diagonal structure
            let r_left = r1_left + r2_left;
            let r_right = r1_right + r2_right;
            let mut new_core = Array3::<T>::zeros((r_left, n_k, r_right));

            // Copy core1 to top-left block
            for i in 0..r1_left {
                for j in 0..n_k {
                    for l in 0..r1_right {
                        new_core[[i, j, l]] = core1[[i, j, l]];
                    }
                }
            }

            // Copy core2 to bottom-right block
            for i in 0..r2_left {
                for j in 0..n_k {
                    for l in 0..r2_right {
                        new_core[[r1_left + i, j, r1_right + l]] = core2[[i, j, l]];
                    }
                }
            }

            new_cores.push(new_core);
            if k < n_modes - 1 {
                new_ranks.push(r_right);
            }
        }
    }

    Ok(TTDecomp {
        cores: new_cores,
        ranks: new_ranks,
        shape: tt1.shape.clone(),
        error: None,
    })
}

/// Inner product (dot product) of two TT decompositions
///
/// Computes ⟨X, Y⟩ where X and Y are in TT format, without explicit reconstruction.
///
/// # Algorithm
///
/// The inner product can be computed efficiently by contracting TT cores:
/// ⟨X, Y⟩ = trace(M₁ × M₂ × ... × Mₙ)
/// where Mₖ\[r,s\] = Σᵢ Gₖ\[r,i,:\] · Hₖ\[s,i,:\]
///
/// # Arguments
///
/// * `tt1` - First TT decomposition
/// * `tt2` - Second TT decomposition
///
/// # Returns
///
/// Scalar inner product value
///
/// # Errors
///
/// Returns error if tensors have different shapes
///
/// # Complexity
///
/// Time: O(N × R₁² × R₂² × I) where R₁, R₂ are max TT-ranks, I is max mode size
/// Space: O(R₁ × R₂)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::{tt_svd, tt_dot};
///
/// let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
/// let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
///
/// let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
/// let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();
///
/// let inner_prod = tt_dot(&tt1, &tt2).unwrap();
/// println!("Inner product: {}", inner_prod);
/// ```
pub fn tt_dot<T>(tt1: &TTDecomp<T>, tt2: &TTDecomp<T>) -> Result<T, TTError>
where
    T: Float + NumCast + NumAssign + 'static,
{
    // Validate shapes match
    if tt1.shape != tt2.shape {
        return Err(TTError::ShapeMismatch(format!(
            "Shape mismatch: {:?} vs {:?}",
            tt1.shape, tt2.shape
        )));
    }

    let n_modes = tt1.cores.len();
    if n_modes != tt2.cores.len() {
        return Err(TTError::ShapeMismatch(format!(
            "Number of modes mismatch: {} vs {}",
            n_modes,
            tt2.cores.len()
        )));
    }

    // Start with first core contraction: M₁[r,s] = Σᵢ G₁[1,i,r] · H₁[1,i,s]
    let core1_1 = &tt1.cores[0];
    let core1_2 = &tt2.cores[0];
    let (_, n_0, r1_0) = core1_1.dim();
    let (_, _, r2_0) = core1_2.dim();

    let mut m = Array2::<T>::zeros((r1_0, r2_0));
    for i in 0..n_0 {
        for r in 0..r1_0 {
            for s in 0..r2_0 {
                m[[r, s]] += core1_1[[0, i, r]] * core1_2[[0, i, s]];
            }
        }
    }

    // Contract with subsequent cores
    for k in 1..n_modes {
        let core1_k = &tt1.cores[k];
        let core2_k = &tt2.cores[k];
        let (r1_left, n_k, r1_right) = core1_k.dim();
        let (r2_left, _, r2_right) = core2_k.dim();

        let mut m_next = Array2::<T>::zeros((r1_right, r2_right));

        // M_{k+1}[r',s'] = Σᵣ Σₛ Σᵢ M_k[r,s] · G_k[r,i,r'] · H_k[s,i,s']
        for r in 0..r1_left {
            for s in 0..r2_left {
                let m_rs = m[[r, s]];
                if m_rs.abs() < T::epsilon() {
                    continue; // Skip if negligible
                }

                for i in 0..n_k {
                    for r_prime in 0..r1_right {
                        for s_prime in 0..r2_right {
                            m_next[[r_prime, s_prime]] +=
                                m_rs * core1_k[[r, i, r_prime]] * core2_k[[s, i, s_prime]];
                        }
                    }
                }
            }
        }

        m = m_next;
    }

    // For the last core, both right ranks should be 1
    if m.shape() != [1, 1] {
        return Err(TTError::ShapeMismatch(format!(
            "Final contraction should be 1×1, got {:?}",
            m.shape()
        )));
    }

    Ok(m[[0, 0]])
}

/// Element-wise (Hadamard) product of two TT decompositions
///
/// Computes the TT decomposition of X ⊙ Y (element-wise product)
/// where X and Y are in TT format.
///
/// # Algorithm
///
/// For TT decompositions with cores Gₖ and Hₖ, the Hadamard product has cores:
/// Cₖ[r₁r₂, i, r₁'r₂'] = Gₖ[r₁, i, r₁'] · Hₖ[r₂, i, r₂']
///
/// The resulting TT-ranks are r₁×s₁, r₂×s₂, ..., rₙ₋₁×sₙ₋₁
///
/// # Arguments
///
/// * `tt1` - First TT decomposition
/// * `tt2` - Second TT decomposition
///
/// # Returns
///
/// New TTDecomp representing the Hadamard product
///
/// # Errors
///
/// Returns error if tensors have different shapes
///
/// # Complexity
///
/// Time: O(N × R₁² × R₂² × I) where R₁, R₂ are max TT-ranks
/// Space: O(R₁² × R₂²) per core
///
/// # Note
///
/// The resulting TT-ranks grow as the product of input ranks.
/// Consider applying `tt_round` after this operation to reduce ranks.
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tt::{tt_svd, tt_hadamard};
///
/// let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
/// let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
///
/// let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
/// let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();
///
/// let tt_prod = tt_hadamard(&tt1, &tt2).unwrap();
/// println!("Product TT-ranks: {:?}", tt_prod.ranks);
///
/// // Note: Ranks grow multiplicatively (4×4=16 for each bond)
/// // Consider using tt_round with larger tensor sizes to reduce storage
/// ```
pub fn tt_hadamard<T>(tt1: &TTDecomp<T>, tt2: &TTDecomp<T>) -> Result<TTDecomp<T>, TTError>
where
    T: Float + NumCast + 'static,
{
    // Validate shapes match
    if tt1.shape != tt2.shape {
        return Err(TTError::ShapeMismatch(format!(
            "Shape mismatch: {:?} vs {:?}",
            tt1.shape, tt2.shape
        )));
    }

    let n_modes = tt1.cores.len();
    if n_modes != tt2.cores.len() {
        return Err(TTError::ShapeMismatch(format!(
            "Number of modes mismatch: {} vs {}",
            n_modes,
            tt2.cores.len()
        )));
    }

    let mut new_cores = Vec::with_capacity(n_modes);
    let mut new_ranks = Vec::with_capacity(n_modes - 1);

    for k in 0..n_modes {
        let core1 = &tt1.cores[k];
        let core2 = &tt2.cores[k];

        let (r1_left, n1, r1_right) = core1.dim();
        let (r2_left, n2, r2_right) = core2.dim();

        if n1 != n2 {
            return Err(TTError::ShapeMismatch(format!(
                "Mode {} size mismatch: {} vs {}",
                k, n1, n2
            )));
        }

        let n_k = n1;
        let r_left = r1_left * r2_left;
        let r_right = r1_right * r2_right;

        // Create new core: Cₖ[r₁r₂, i, r₁'r₂'] = Gₖ[r₁, i, r₁'] · Hₖ[r₂, i, r₂']
        let mut new_core = Array3::<T>::zeros((r_left, n_k, r_right));

        for r1 in 0..r1_left {
            for r2 in 0..r2_left {
                let r_idx = r1 * r2_left + r2;

                for i in 0..n_k {
                    for r1_prime in 0..r1_right {
                        for r2_prime in 0..r2_right {
                            let r_prime_idx = r1_prime * r2_right + r2_prime;
                            new_core[[r_idx, i, r_prime_idx]] =
                                core1[[r1, i, r1_prime]] * core2[[r2, i, r2_prime]];
                        }
                    }
                }
            }
        }

        new_cores.push(new_core);

        if k < n_modes - 1 {
            new_ranks.push(r_right);
        }
    }

    Ok(TTDecomp {
        cores: new_cores,
        ranks: new_ranks,
        shape: tt1.shape.clone(),
        error: None,
    })
}
