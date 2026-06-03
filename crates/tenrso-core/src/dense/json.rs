//! JSON serialization/deserialization for `DenseND<T>` using serde_json.
//!
//! # Format
//!
//! Files are UTF-8 JSON. The layout is produced by `serde_json::to_writer_pretty`,
//! which is self-describing (field names are present) but verbose compared to
//! binary. The format is stable across compatible versions of `serde` and `DenseND`.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "json")]
//! # {
//! use tenrso_core::DenseND;
//!
//! let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
//! let path = std::env::temp_dir().join("my_tensor.json");
//! tensor.save_json(&path).unwrap();
//! let loaded = DenseND::<f64>::load_json(&path).unwrap();
//! assert_eq!(tensor.shape(), loaded.shape());
//! # }
//! ```

use std::io::{BufReader, BufWriter};
use std::path::Path;

use scirs2_core::numeric::Num;

use super::DenseND;

impl<T> DenseND<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone + Num + 'static,
{
    /// Save this tensor to a JSON file using pretty-printed serde_json encoding.
    ///
    /// The resulting file is valid UTF-8 JSON produced by
    /// `serde_json::to_writer_pretty`. It captures the full tensor state
    /// including shape and all element values and is human-readable.
    ///
    /// # Arguments
    ///
    /// * `path` - Destination path for the JSON file. Created or truncated.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created or written to
    /// - serde_json serialization fails (e.g. a non-finite float that is not
    ///   representable in JSON)
    ///
    /// # Complexity
    ///
    /// O(n) in the number of elements — each element is formatted and streamed
    /// through a `BufWriter` to the file.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "json")]
    /// # {
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f32>::ones(&[8, 8]);
    /// tensor.save_json(std::path::Path::new("/tmp/tensor.json")).unwrap();
    /// # }
    /// ```
    pub fn save_json(&self, path: &Path) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| anyhow::anyhow!("cannot create JSON file {:?}: {}", path, e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| anyhow::anyhow!("serde_json encode failed for path {:?}: {}", path, e))?;
        Ok(())
    }

    /// Load a tensor from a JSON file written by [`DenseND::save_json`].
    ///
    /// The element type `T` must be compatible with the JSON representation
    /// stored in the file. Numeric types are coerced by serde_json's default
    /// visitor rules (e.g. `f32` can load values originally written as `f64`
    /// provided they are in range).
    ///
    /// # Arguments
    ///
    /// * `path` - Source path for the JSON file to read.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or read
    /// - The bytes do not represent valid JSON for `DenseND<T>`
    ///
    /// # Complexity
    ///
    /// O(n) in the number of elements — reads and parses the file via a
    /// streaming `BufReader`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "json")]
    /// # {
    /// use tenrso_core::DenseND;
    ///
    /// let loaded = DenseND::<f32>::load_json(std::path::Path::new("/tmp/tensor.json")).unwrap();
    /// # }
    /// ```
    pub fn load_json(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("cannot open JSON file {:?}: {}", path, e))?;
        let reader = BufReader::new(file);
        let tensor = serde_json::from_reader(reader)
            .map_err(|e| anyhow::anyhow!("serde_json decode failed for path {:?}: {}", path, e))?;
        Ok(tensor)
    }

    /// Serialize this tensor to a pretty-printed JSON `String`.
    ///
    /// Produces the same JSON representation as [`DenseND::save_json`] but
    /// returns it in memory rather than writing to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - serde_json serialization fails (e.g. a non-finite float value)
    ///
    /// # Complexity
    ///
    /// O(n) in the number of elements — allocates a `String` proportional to
    /// the number of formatted values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "json")]
    /// # {
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[2, 3]);
    /// let json_str = tensor.to_json_string().unwrap();
    /// assert!(json_str.contains("data"));
    /// # }
    /// ```
    pub fn to_json_string(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("serde_json stringify failed: {}", e))
    }

    /// Deserialize a tensor from a JSON string slice.
    ///
    /// Accepts any valid JSON representation of `DenseND<T>`, such as one
    /// produced by [`DenseND::to_json_string`].
    ///
    /// # Arguments
    ///
    /// * `s` - A string slice containing the JSON to deserialize.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `s` is not valid JSON
    /// - The JSON structure does not match `DenseND<T>`
    ///
    /// # Complexity
    ///
    /// O(n) in the number of elements — parses the string in a single pass.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "json")]
    /// # {
    /// use tenrso_core::DenseND;
    ///
    /// let original = DenseND::<f64>::ones(&[2, 3]);
    /// let json_str = original.to_json_string().unwrap();
    /// let loaded = DenseND::<f64>::from_json_str(&json_str).unwrap();
    /// assert_eq!(original.shape(), loaded.shape());
    /// # }
    /// ```
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        serde_json::from_str(s).map_err(|e| anyhow::anyhow!("serde_json parse failed: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a sequential f64 tensor with the given shape.
    fn seq_f64(shape: &[usize]) -> DenseND<f64> {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        DenseND::from_vec(data, shape).unwrap()
    }

    /// Helper: create a sequential f32 tensor with the given shape.
    fn seq_f32(shape: &[usize]) -> DenseND<f32> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        DenseND::from_vec(data, shape).unwrap()
    }

    #[test]
    fn round_trip_f64() {
        let tensor = seq_f64(&[2, 3, 4]);
        let path = std::env::temp_dir().join("tenrso_json_rt_f64.json");
        tensor.save_json(&path).unwrap();
        let loaded = DenseND::<f64>::load_json(&path).unwrap();
        assert_eq!(tensor.shape(), loaded.shape());
        for (a, b) in tensor.iter().zip(loaded.iter()) {
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn round_trip_f32() {
        let tensor = seq_f32(&[3, 5]);
        let path = std::env::temp_dir().join("tenrso_json_rt_f32.json");
        tensor.save_json(&path).unwrap();
        let loaded = DenseND::<f32>::load_json(&path).unwrap();
        assert_eq!(tensor.shape(), loaded.shape());
        for (a, b) in tensor.iter().zip(loaded.iter()) {
            assert_eq!(a, b);
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn string_round_trip() {
        let original = seq_f64(&[4, 4]);
        let json_str = original.to_json_string().unwrap();
        let loaded = DenseND::<f64>::from_json_str(&json_str).unwrap();
        assert_eq!(original.shape(), loaded.shape());
        for (a, b) in original.iter().zip(loaded.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn multi_shape() {
        let shapes: &[&[usize]] = &[&[10], &[3, 4], &[2, 3, 5], &[2, 2, 2, 2]];
        for shape in shapes {
            let tensor = seq_f64(shape);
            let json_str = tensor.to_json_string().unwrap();
            let loaded = DenseND::<f64>::from_json_str(&json_str).unwrap();
            assert_eq!(
                tensor.shape(),
                loaded.shape(),
                "shape mismatch for {:?}",
                shape
            );
            for (a, b) in tensor.iter().zip(loaded.iter()) {
                assert_eq!(a, b, "value mismatch for shape {:?}", shape);
            }
        }
    }

    #[test]
    fn empty_values() {
        // Zeros tensor round-trip
        let tensor = DenseND::<f64>::zeros(&[4, 6]);
        let json_str = tensor.to_json_string().unwrap();
        let loaded = DenseND::<f64>::from_json_str(&json_str).unwrap();
        assert_eq!(tensor.shape(), loaded.shape());
        for (a, b) in tensor.iter().zip(loaded.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn scalar_single_element() {
        // 1D tensor of length 1 acts as the closest supported "scalar"
        let tensor = DenseND::<f64>::from_vec(vec![42.0], &[1]).unwrap();
        let json_str = tensor.to_json_string().unwrap();
        let loaded = DenseND::<f64>::from_json_str(&json_str).unwrap();
        assert_eq!(loaded.shape(), &[1]);
        assert_eq!(loaded.to_vec()[0], 42.0);
    }

    #[test]
    fn bad_file_path() {
        let path = std::path::Path::new("/nonexistent_dir/no_such_file.json");
        let result = DenseND::<f64>::load_json(path);
        assert!(
            result.is_err(),
            "expected error loading non-existent file, got Ok"
        );
    }

    #[test]
    fn invalid_json_str() {
        let result = DenseND::<f64>::from_json_str("not valid json {{{");
        assert!(result.is_err(), "expected error from invalid JSON, got Ok");
    }

    #[test]
    fn large_tensor() {
        let tensor = DenseND::<f64>::zeros(&[100, 100]);
        let path = std::env::temp_dir().join("tenrso_json_large.json");
        tensor.save_json(&path).unwrap();
        let loaded = DenseND::<f64>::load_json(&path).unwrap();
        assert_eq!(tensor.shape(), loaded.shape());
        assert_eq!(loaded.len(), 10_000);
        for v in loaded.iter() {
            assert_eq!(*v, 0.0);
        }
        std::fs::remove_file(&path).unwrap();
    }
}
