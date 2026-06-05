//! Binary I/O round-trip example for `DenseND<T>` and `TensorHandle<T>`.
//!
//! Demonstrates `save_binary` / `load_binary` using oxicode encoding.
//! Requires the `binary` feature:
//!
//! ```bash
//! cargo run -p tenrso-core --example binary_io --features binary
//! ```

use tenrso_core::{AxisMeta, DenseND, TensorHandle};

fn main() -> anyhow::Result<()> {
    println!("=== TenRSo Binary I/O Example ===\n");

    example_dense_nd_f64()?;
    example_dense_nd_f32()?;
    example_tensor_handle()?;
    example_large_tensor()?;

    println!("\n=== All binary I/O examples completed successfully! ===");
    Ok(())
}

// ------------------------------------------------------------------ helpers --

fn assert_tensors_equal(a: &DenseND<f64>, b: &DenseND<f64>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{}: shape mismatch", label);
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-15,
        "{}: element mismatch (max diff {})",
        label,
        max_diff
    );
    println!("  [OK] {} — max |diff| = {:.2e}", label, max_diff);
}

fn assert_tensors_equal_f32(a: &DenseND<f32>, b: &DenseND<f32>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{}: shape mismatch", label);
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-7,
        "{}: element mismatch (max diff {})",
        label,
        max_diff
    );
    println!("  [OK] {} — max |diff| = {:.2e}", label, max_diff);
}

// ------------------------------------------------------- example functions --

fn example_dense_nd_f64() -> anyhow::Result<()> {
    println!("--- DenseND<f64> round-trip ---");

    let tmp = std::env::temp_dir().join("tenrso_binary_io_f64.bin");

    // 3-D tensor with known values
    let data: Vec<f64> = (0..60).map(|i| i as f64 * 0.5 + 0.1).collect();
    let tensor = DenseND::from_vec(data, &[3, 4, 5])?;

    println!("  original shape  : {:?}", tensor.shape());
    println!("  original [0,0,0]: {}", tensor[&[0, 0, 0]]);

    tensor.save_binary(&tmp)?;
    println!("  saved to         : {:?}", tmp);

    let loaded = DenseND::<f64>::load_binary(&tmp)?;
    println!("  loaded  shape   : {:?}", loaded.shape());

    assert_tensors_equal(&tensor, &loaded, "DenseND<f64> 3D round-trip");

    // Clean up — ignore errors (e.g. permission denied in CI)
    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

fn example_dense_nd_f32() -> anyhow::Result<()> {
    println!("\n--- DenseND<f32> round-trip ---");

    let tmp = std::env::temp_dir().join("tenrso_binary_io_f32.bin");

    // 1-D tensor
    let data: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
    let tensor = DenseND::from_vec(data, &[100])?;

    println!("  original shape  : {:?}", tensor.shape());

    tensor.save_binary(&tmp)?;

    let loaded = DenseND::<f32>::load_binary(&tmp)?;
    assert_tensors_equal_f32(&tensor, &loaded, "DenseND<f32> 1D round-trip");

    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

fn example_tensor_handle() -> anyhow::Result<()> {
    println!("\n--- TensorHandle<f64> round-trip ---");

    let tmp = std::env::temp_dir().join("tenrso_binary_io_handle.bin");

    let dense = DenseND::<f64>::random_uniform(&[6, 7], 0.0, 1.0);
    let axes = vec![AxisMeta::new("rows", 6), AxisMeta::new("cols", 7)];
    let handle = TensorHandle::from_dense(dense, axes);

    println!(
        "  original axes   : {:?}",
        handle.axes.iter().map(|a| &a.name).collect::<Vec<_>>()
    );
    println!("  original shape  : {:?}", handle.shape());

    handle.save_binary(&tmp)?;

    // Load back as a TensorHandle — axes are auto-named axis_0, axis_1, …
    let loaded_handle = TensorHandle::<f64>::load_binary(&tmp)?;
    println!("  loaded shape    : {:?}", loaded_handle.shape());

    let orig_dense = handle.as_dense().unwrap();
    let load_dense = loaded_handle.as_dense().unwrap();
    assert_tensors_equal(orig_dense, load_dense, "TensorHandle<f64> round-trip");

    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

fn example_large_tensor() -> anyhow::Result<()> {
    println!("\n--- Large DenseND<f64> (256×256) round-trip ---");

    let tmp = std::env::temp_dir().join("tenrso_binary_io_large.bin");

    let tensor = DenseND::<f64>::random_uniform(&[256, 256], -1.0, 1.0);
    let byte_size = tensor.size_bytes();
    println!(
        "  tensor size     : {} bytes ({} KiB)",
        byte_size,
        byte_size / 1024
    );

    let t0 = std::time::Instant::now();
    tensor.save_binary(&tmp)?;
    let save_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let loaded = DenseND::<f64>::load_binary(&tmp)?;
    let load_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("  save time       : {:.2} ms", save_ms);
    println!("  load time       : {:.2} ms", load_ms);

    assert_tensors_equal(&tensor, &loaded, "Large tensor round-trip");

    let _ = std::fs::remove_file(&tmp);
    Ok(())
}
