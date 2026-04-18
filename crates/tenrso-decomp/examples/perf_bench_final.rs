use tenrso_core::DenseND;
use tenrso_decomp::cp::{cp_als, InitStrategy};
use tenrso_decomp::tucker::{tucker_hooi, tucker_hosvd};
use tenrso_decomp::tt::tt_svd;
use std::time::Instant;

fn main() {
    // CP-ALS benchmark: 256^3, rank 64, 10 iters
    println!("=== CP-ALS: 256^3, r64, 10 iters ===");
    let tensor_cp = DenseND::<f64>::random_uniform(&[256, 256, 256], 0.0, 1.0);
    let t0 = Instant::now();
    let cp = cp_als(&tensor_cp, 64, 10, 1e-8, InitStrategy::Random, None);
    let cp_time = t0.elapsed();
    match cp {
        Ok(r) => println!("  Time: {:.2}s  fit={:.4}  iters={}", cp_time.as_secs_f64(), r.fit, r.iters),
        Err(e) => println!("  FAILED: {}", e),
    }

    // Tucker-HOOI: 256x256x64, ranks [32,32,16], 10 iters
    println!("\n=== Tucker-HOOI: 256x256x64, r[32,32,16], 10 iters ===");
    let tensor_tucker = DenseND::<f64>::random_uniform(&[256, 256, 64], 0.0, 1.0);
    let t1 = Instant::now();
    let tucker = tucker_hooi(&tensor_tucker, &[32, 32, 16], 10, 1e-4);
    let tucker_time = t1.elapsed();
    match tucker {
        Ok(r) => println!("  Time: {:.2}s  iters={}", tucker_time.as_secs_f64(), r.iters),
        Err(e) => println!("  FAILED: {}", e),
    }

    // Tucker-HOSVD: 512x512x128, r[64,64,32]
    println!("\n=== Tucker-HOSVD: 512x512x128, r[64,64,32] ===");
    let tensor_hosvd = DenseND::<f64>::random_uniform(&[512, 512, 128], 0.0, 1.0);
    let t2 = Instant::now();
    let hosvd = tucker_hosvd(&tensor_hosvd, &[64, 64, 32]);
    let hosvd_time = t2.elapsed();
    match hosvd {
        Ok(r) => println!("  Time: {:.2}s  iters={}", hosvd_time.as_secs_f64(), r.iters),
        Err(e) => println!("  FAILED: {}", e),
    }

    // TT-SVD: 32^4, ranks [16,16,16]
    println!("\n=== TT-SVD: 32^4, r[16,16,16] ===");
    let tensor_tt = DenseND::<f64>::random_uniform(&[32, 32, 32, 32], 0.0, 1.0);
    let t3 = Instant::now();
    let tt = tt_svd(&tensor_tt, &[16, 16, 16], 1e-6);
    let tt_time = t3.elapsed();
    match tt {
        Ok(_) => println!("  Time: {:.2}s", tt_time.as_secs_f64()),
        Err(e) => println!("  FAILED: {}", e),
    }
}
