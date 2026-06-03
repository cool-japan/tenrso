//! Integration tests for binary serialization of `DenseND<T>` and `TensorHandle<T>`.
//!
//! Compile and run with:
//! ```bash
//! cargo nextest run -p tenrso-core --features binary
//! ```

#[cfg(feature = "binary")]
mod binary_io_tests {
    use tenrso_core::{AxisMeta, DenseND, TensorHandle};

    // ------------------------------------------------------------------ helpers --

    fn max_diff_f64(a: &DenseND<f64>, b: &DenseND<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn max_diff_f32(a: &DenseND<f32>, b: &DenseND<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    // -------------------------------------------------------- DenseND<f64> tests --

    #[test]
    fn test_round_trip_f64_3d() {
        let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let path = std::env::temp_dir().join("tenrso_test_binary_f64_3d.bin");

        tensor.save_binary(&path).unwrap();
        let loaded = DenseND::<f64>::load_binary(&path).unwrap();

        assert_eq!(tensor.shape(), loaded.shape());
        assert!(
            max_diff_f64(&tensor, &loaded) < 1e-15,
            "f64 3D round-trip: max diff exceeded tolerance"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_round_trip_f64_1d() {
        let data: Vec<f64> = (0..128).map(|i| i as f64 * std::f64::consts::PI).collect();
        let tensor = DenseND::from_vec(data, &[128]).unwrap();
        let path = std::env::temp_dir().join("tenrso_test_binary_f64_1d.bin");

        tensor.save_binary(&path).unwrap();
        let loaded = DenseND::<f64>::load_binary(&path).unwrap();

        assert_eq!(tensor.shape(), loaded.shape());
        assert!(
            max_diff_f64(&tensor, &loaded) < 1e-15,
            "f64 1D round-trip: data mismatch"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_round_trip_f64_scalar() {
        // 0-dimensional / single-element tensor
        let tensor = DenseND::<f64>::from_elem(&[1], 42.0);
        let path = std::env::temp_dir().join("tenrso_test_binary_f64_scalar.bin");

        tensor.save_binary(&path).unwrap();
        let loaded = DenseND::<f64>::load_binary(&path).unwrap();

        assert_eq!(tensor.shape(), loaded.shape());
        assert!(
            max_diff_f64(&tensor, &loaded) < 1e-15,
            "f64 scalar round-trip: data mismatch"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_round_trip_f64_high_rank() {
        // 5-dimensional tensor
        let tensor = DenseND::<f64>::random_uniform(&[2, 3, 4, 5, 6], -1.0, 1.0);
        let path = std::env::temp_dir().join("tenrso_test_binary_f64_5d.bin");

        tensor.save_binary(&path).unwrap();
        let loaded = DenseND::<f64>::load_binary(&path).unwrap();

        assert_eq!(tensor.shape(), loaded.shape());
        assert_eq!(tensor.rank(), 5);
        assert!(
            max_diff_f64(&tensor, &loaded) < 1e-15,
            "f64 5D round-trip: data mismatch"
        );

        let _ = std::fs::remove_file(&path);
    }

    // -------------------------------------------------------- DenseND<f32> tests --

    #[test]
    fn test_round_trip_f32_2d() {
        // random_uniform requires T: From<f64> which f32 does not implement,
        // so we construct an f32 tensor from explicit values instead.
        let data: Vec<f32> = (0..256)
            .map(|i: u32| (i as f32) * 0.390_625_f32) // 100.0 / 256.0
            .collect();
        let tensor = DenseND::<f32>::from_vec(data, &[16, 16]).unwrap();
        let path = std::env::temp_dir().join("tenrso_test_binary_f32_2d.bin");

        tensor.save_binary(&path).unwrap();
        let loaded = DenseND::<f32>::load_binary(&path).unwrap();

        assert_eq!(tensor.shape(), loaded.shape());
        assert!(
            max_diff_f32(&tensor, &loaded) < 1e-7,
            "f32 2D round-trip: data mismatch"
        );

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------- TensorHandle tests --

    #[test]
    fn test_tensor_handle_round_trip() {
        let dense = DenseND::<f64>::random_uniform(&[8, 12], -5.0, 5.0);
        let axes = vec![AxisMeta::new("rows", 8), AxisMeta::new("cols", 12)];
        let handle = TensorHandle::from_dense(dense, axes);

        let path = std::env::temp_dir().join("tenrso_test_binary_handle.bin");

        handle.save_binary(&path).unwrap();
        let loaded = TensorHandle::<f64>::load_binary(&path).unwrap();

        assert_eq!(handle.shape(), loaded.shape());

        let orig = handle.as_dense().unwrap();
        let load = loaded.as_dense().unwrap();
        assert!(
            max_diff_f64(orig, load) < 1e-15,
            "TensorHandle round-trip: element mismatch"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tensor_handle_axis_reconstruction() {
        // After load_binary, axes are auto-named axis_0, axis_1, …
        let dense = DenseND::<f64>::zeros(&[3, 4, 5]);
        let handle = TensorHandle::from_dense_auto(dense);

        let path = std::env::temp_dir().join("tenrso_test_binary_handle_axes.bin");

        handle.save_binary(&path).unwrap();
        let loaded = TensorHandle::<f64>::load_binary(&path).unwrap();

        assert_eq!(loaded.rank(), 3);
        assert_eq!(loaded.axes[0].name, "axis_0");
        assert_eq!(loaded.axes[1].name, "axis_1");
        assert_eq!(loaded.axes[2].name, "axis_2");
        assert_eq!(loaded.axes[0].size, 3);
        assert_eq!(loaded.axes[1].size, 4);
        assert_eq!(loaded.axes[2].size, 5);

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------- file-level checks --

    #[test]
    fn test_file_is_non_empty_after_save() {
        let tensor = DenseND::<f64>::ones(&[10, 10]);
        let path = std::env::temp_dir().join("tenrso_test_binary_nonempty.bin");

        tensor.save_binary(&path).unwrap();

        let metadata = std::fs::metadata(&path).unwrap();
        assert!(
            metadata.len() > 0,
            "binary file should have non-zero size after save"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_missing_file_returns_error() {
        let path = std::env::temp_dir().join("tenrso_test_nonexistent_file_xyz_123.bin");
        // Make sure the path does not exist
        let _ = std::fs::remove_file(&path);

        let result = DenseND::<f64>::load_binary(&path);
        assert!(
            result.is_err(),
            "load_binary should return Err for a missing file"
        );
    }

    #[test]
    fn test_corrupted_bytes_return_error() {
        let path = std::env::temp_dir().join("tenrso_test_binary_corrupt.bin");
        // Write garbage bytes
        std::fs::write(&path, b"this is not a valid bincode tensor payload").unwrap();

        let result = DenseND::<f64>::load_binary(&path);
        assert!(
            result.is_err(),
            "load_binary should return Err for corrupted bytes"
        );

        let _ = std::fs::remove_file(&path);
    }

    // ---------------------------------------------------------- shape fidelity --

    #[test]
    fn test_shape_preserved_exactly() {
        let shapes: &[&[usize]] = &[&[1], &[1, 1], &[7], &[3, 3], &[2, 4, 8], &[5, 1, 3, 2]];

        for (idx, shape) in shapes.iter().enumerate() {
            let tensor = DenseND::<f64>::zeros(shape);
            let path = std::env::temp_dir().join(format!("tenrso_test_binary_shape_{}.bin", idx));

            tensor.save_binary(&path).unwrap();
            let loaded = DenseND::<f64>::load_binary(&path).unwrap();

            assert_eq!(
                tensor.shape(),
                loaded.shape(),
                "shape {:?} not preserved",
                shape
            );

            let _ = std::fs::remove_file(&path);
        }
    }
}
