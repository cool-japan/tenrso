//! Complete Tensor Train (TT) workflow example using tenrso-kernels TT operations
//!
//! This example demonstrates a full TT decomposition workflow including:
//! - Creating TT cores
//! - Computing TT norm
//! - Computing TT dot products
//! - Orthogonalizing TT cores (left and right)
//! - Rounding TT representation
//!
//! Run with: cargo run --example tt_workflow

use scirs2_core::ndarray_ext::Array3;
use tenrso_kernels::{tt_dot, tt_left_orthogonalize, tt_norm, tt_right_orthogonalize, tt_round};

fn main() {
    println!("=== Tensor Train (TT) Workflow Example ===\n");

    // Problem setup
    let d: usize = 5; // Number of dimensions
    let n: usize = 10; // Mode size
    let r: usize = 4; // TT rank

    println!("Problem:");
    println!("  Tensor order: d = {}", d);
    println!("  Mode size: n = {}", n);
    println!("  TT rank: r = {}", r);
    println!(
        "  Full tensor size: {}^{} = {} elements",
        n,
        d,
        n.pow(d as u32)
    );
    println!(
        "  TT representation: {} parameters (compression ratio: {:.2}x)\n",
        compute_tt_parameters(d, n, r),
        n.pow(d as u32) as f64 / compute_tt_parameters(d, n, r) as f64
    );

    // Step 1: Create initial TT cores (random initialization)
    println!("Step 1: Creating random TT cores...");
    let cores = create_random_tt_cores(d, n, r, 42);
    println!("  Created {} cores with shapes:", cores.len());
    for (i, core) in cores.iter().enumerate() {
        let shape = core.shape();
        println!("    Core {}: [{}, {}, {}]", i, shape[0], shape[1], shape[2]);
    }

    // Step 2: Compute TT norm
    println!("\nStep 2: Computing TT norm...");
    let core_views: Vec<_> = cores.iter().map(|c| c.view()).collect();
    let norm = tt_norm(&core_views).expect("Failed to compute TT norm");
    println!("  TT norm: {:.6}", norm);
    println!("  (Computed without materializing the full tensor)");

    // Step 3: Compute TT dot product (with itself)
    println!("\nStep 3: Computing TT self dot product...");
    let dot_self = tt_dot(&core_views, &core_views).expect("Failed to compute TT dot product");
    println!("  ⟨X, X⟩ = {:.6}", dot_self);
    println!("  ||X||² = {:.6}", norm * norm);
    println!(
        "  Verification: |⟨X,X⟩ - ||X||²| = {:.2e}",
        (dot_self - norm * norm).abs()
    );

    // Step 4: Create another TT tensor for dot product
    println!("\nStep 4: Creating second TT tensor...");
    let cores_y = create_random_tt_cores(d, n, r, 123);
    let core_views_y: Vec<_> = cores_y.iter().map(|c| c.view()).collect();
    let norm_y = tt_norm(&core_views_y).expect("Failed to compute norm of Y");
    println!("  ||Y|| = {:.6}", norm_y);

    // Compute dot product between X and Y
    let dot_xy = tt_dot(&core_views, &core_views_y).expect("Failed to compute dot product");
    println!("  ⟨X, Y⟩ = {:.6}", dot_xy);

    // Compute cosine similarity
    let cosine_sim = dot_xy / (norm * norm_y);
    println!("  Cosine similarity: {:.6}", cosine_sim);

    // Step 5: Left-orthogonalize TT cores (using simpler smaller tensor)
    println!("\nStep 5: Left-orthogonalizing TT cores (smaller example)...");
    let d_small = 3;
    let n_small = 5;
    let r_small = 2;
    let mut cores_small = create_random_tt_cores(d_small, n_small, r_small, 777);
    let core_views_small: Vec<_> = cores_small.iter().map(|c| c.view()).collect();
    let norm_before = tt_norm(&core_views_small).expect("Failed to compute norm");

    tt_left_orthogonalize(&mut cores_small).expect("Failed to left-orthogonalize");
    let core_views_ortho: Vec<_> = cores_small.iter().map(|c| c.view()).collect();
    let norm_after =
        tt_norm(&core_views_ortho).expect("Failed to compute norm after orthogonalization");
    println!("  Norm before orthogonalization: {:.6}", norm_before);
    println!("  Norm after orthogonalization: {:.6}", norm_after);
    println!(
        "  Norm preservation error: {:.2e}",
        (norm_before - norm_after).abs()
    );

    // Step 6: Right-orthogonalize TT cores
    println!("\nStep 6: Right-orthogonalizing TT cores (smaller example)...");
    let mut cores_small2 = create_random_tt_cores(d_small, n_small, r_small, 888);
    tt_right_orthogonalize(&mut cores_small2).expect("Failed to right-orthogonalize");
    let core_views_right: Vec<_> = cores_small2.iter().map(|c| c.view()).collect();
    let norm_right =
        tt_norm(&core_views_right).expect("Failed to compute norm after right orthogonalization");
    println!("  Norm after right-orthogonalization: {:.6}", norm_right);

    // Step 7: TT rounding (with very small epsilon for demonstration)
    println!("\nStep 7: Applying TT rounding (smaller example)...");
    let epsilon = 1e-8;
    let mut cores_rounded = cores_small.clone();
    tt_round(&mut cores_rounded, None, epsilon).expect("Failed to round TT");
    let core_views_rounded: Vec<_> = cores_rounded.iter().map(|c| c.view()).collect();
    let norm_rounded = tt_norm(&core_views_rounded).expect("Failed to compute norm after rounding");
    println!("  Epsilon: {:.2e}", epsilon);
    println!("  Norm after rounding: {:.6}", norm_rounded);
    println!(
        "  Approximation error: {:.2e}",
        (norm_before - norm_rounded).abs()
    );
    println!("  (Note: Current rounding uses orthogonalization only)");

    // Final summary
    println!("\n=== TT Workflow Complete ===\n");

    println!("Summary of TT operations:");
    println!("  ✓ TT core creation and validation");
    println!("  ✓ TT norm computation (O(d×r³) complexity)");
    println!("  ✓ TT dot product computation");
    println!("  ✓ TT cosine similarity");
    println!("  ✓ Left-orthogonalization (norm-preserving)");
    println!("  ✓ Right-orthogonalization (norm-preserving)");
    println!("  ✓ TT rounding with epsilon control");

    println!("\nTT advantages demonstrated:");
    println!(
        "  • Storage: {} params vs {} full tensor elements",
        compute_tt_parameters(d, n, r),
        n.pow(d as u32)
    );
    println!(
        "  • Compression: {:.2}x reduction",
        n.pow(d as u32) as f64 / compute_tt_parameters(d, n, r) as f64
    );
    println!("  • Efficient operations: No full tensor materialization needed");
    println!("  • Numerical stability: Orthogonalization support");
}

/// Create random TT cores with proper boundary ranks (r₀ = rₐ = 1)
fn create_random_tt_cores(d: usize, n: usize, r: usize, seed: u64) -> Vec<Array3<f64>> {
    let mut cores = Vec::with_capacity(d);

    for i in 0..d {
        let (r_left, r_right) = if i == 0 {
            (1, r) // First core: [1, n, r]
        } else if i == d - 1 {
            (r, 1) // Last core: [r, n, 1]
        } else {
            (r, r) // Middle cores: [r, n, r]
        };

        // Simple deterministic pseudo-random generation
        let core = Array3::from_shape_fn((r_left, n, r_right), |(i0, i1, i2)| {
            let idx = i0 * n * r_right + i1 * r_right + i2;
            let s = (seed as f64 + i as f64 + idx as f64) * 0.1;
            s.sin() * 0.5 + 0.5 // Values in [0, 1]
        });

        cores.push(core);
    }

    cores
}

/// Compute the number of parameters in a TT representation
fn compute_tt_parameters(d: usize, n: usize, r: usize) -> usize {
    if d == 0 {
        return 0;
    }
    if d == 1 {
        return n;
    }

    // First core: 1 × n × r = n × r
    let first = n * r;

    // Middle cores: (d-2) × r × n × r
    let middle = if d > 2 { (d - 2) * r * n * r } else { 0 };

    // Last core: r × n × 1 = r × n
    let last = r * n;

    first + middle + last
}
