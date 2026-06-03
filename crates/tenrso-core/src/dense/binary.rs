//! Binary serialization/deserialization for `DenseND<T>` using bincode v2.
//!
//! This module provides feature-gated (`binary`) save/load operations that
//! write and read the full tensor (data + shape) as a compact binary blob.
//! The format relies on serde-compatible encoding via `bincode::serde`.
//!
//! # Format
//!
//! The on-disk layout is produced by `bincode::serde::encode_to_vec` with the
//! standard configuration.  The file is **not** self-describing; the element
//! type `T` must match between the writing and reading call sites.  A future
//! version may add a small header for type/version metadata.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "binary")]
//! # {
//! use tenrso_core::DenseND;
//!
//! let tensor = DenseND::<f64>::from_vec(
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//!     &[2, 3],
//! ).unwrap();
//!
//! let path = std::env::temp_dir().join("my_tensor.bin");
//! tensor.save_binary(&path).unwrap();
//!
//! let loaded = DenseND::<f64>::load_binary(&path).unwrap();
//! assert_eq!(tensor.shape(), loaded.shape());
//! # }
//! ```

use std::io::Write;
use std::path::Path;

use scirs2_core::numeric::Num;

use super::DenseND;

impl<T> DenseND<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone + Num + 'static,
{
    /// Save this tensor to a binary file using bincode v2 encoding.
    ///
    /// The resulting file is a flat byte-stream produced by
    /// `bincode::serde::encode_to_vec` with the standard configuration.
    /// It captures the full tensor state including shape and all element
    /// values.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created or written to
    /// - bincode serialization fails (e.g. an unsupported element type)
    ///
    /// # Complexity
    ///
    /// O(n) in the number of elements — a single allocation for the encoded
    /// buffer followed by a single `write_all` call.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "binary")]
    /// # {
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f32>::ones(&[8, 8]);
    /// tensor.save_binary(std::path::Path::new("/tmp/tensor.bin")).unwrap();
    /// # }
    /// ```
    pub fn save_binary(&self, path: &Path) -> anyhow::Result<()> {
        let encoded = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("bincode encode failed for path {:?}: {}", path, e))?;
        let mut f = std::fs::File::create(path)
            .map_err(|e| anyhow::anyhow!("cannot create binary file {:?}: {}", path, e))?;
        f.write_all(&encoded)
            .map_err(|e| anyhow::anyhow!("write failed to {:?}: {}", path, e))?;
        Ok(())
    }

    /// Load a tensor from a binary file written by [`DenseND::save_binary`].
    ///
    /// The element type `T` must be the **same** type used when saving.  No
    /// runtime type-checking is performed; mismatches will result in a
    /// deserialization error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The bytes do not represent a valid bincode-encoded `DenseND<T>`
    ///
    /// # Complexity
    ///
    /// O(n) in the number of elements — reads the full file into memory then
    /// decodes in-place.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "binary")]
    /// # {
    /// use tenrso_core::DenseND;
    ///
    /// let loaded = DenseND::<f32>::load_binary(std::path::Path::new("/tmp/tensor.bin")).unwrap();
    /// # }
    /// ```
    pub fn load_binary(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("cannot read binary file {:?}: {}", path, e))?;
        let (tensor, _consumed) =
            bincode::serde::decode_from_slice::<Self, _>(&bytes, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("bincode decode failed for path {:?}: {}", path, e))?;
        Ok(tensor)
    }
}
