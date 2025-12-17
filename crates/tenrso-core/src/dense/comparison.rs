//! Comparison and logical operations on tensors
//!
//! This module provides element-wise comparison operations, masking,
//! and logical reductions.

use super::types::DenseND;
use scirs2_core::numeric::Num;

impl<T> DenseND<T>
where
    T: Clone + Num + PartialOrd,
{
    /// Element-wise greater than comparison
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let b = DenseND::<f64>::from_vec(vec![2.0, 2.0, 2.0, 5.0], &[2, 2]).unwrap();
    ///
    /// let result = a.gt(&b).unwrap();
    /// assert_eq!(result, vec![false, false, true, false]);
    /// ```
    pub fn gt(&self, other: &Self) -> anyhow::Result<Vec<bool>> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape());
        }
        let self_flat = self
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        let other_flat = other
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        Ok(self_flat
            .iter()
            .zip(other_flat.iter())
            .map(|(a, b)| a > b)
            .collect())
    }

    /// Element-wise less than comparison
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let b = DenseND::<f64>::from_vec(vec![2.0, 2.0, 2.0, 5.0], &[2, 2]).unwrap();
    ///
    /// let result = a.lt(&b).unwrap();
    /// assert_eq!(result, vec![true, false, false, true]);
    /// ```
    pub fn lt(&self, other: &Self) -> anyhow::Result<Vec<bool>> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape());
        }
        let self_flat = self
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        let other_flat = other
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        Ok(self_flat
            .iter()
            .zip(other_flat.iter())
            .map(|(a, b)| a < b)
            .collect())
    }

    /// Element-wise greater than or equal comparison
    pub fn gte(&self, other: &Self) -> anyhow::Result<Vec<bool>> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape());
        }
        let self_flat = self
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        let other_flat = other
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        Ok(self_flat
            .iter()
            .zip(other_flat.iter())
            .map(|(a, b)| a >= b)
            .collect())
    }

    /// Element-wise less than or equal comparison
    pub fn lte(&self, other: &Self) -> anyhow::Result<Vec<bool>> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape());
        }
        let self_flat = self
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        let other_flat = other
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        Ok(self_flat
            .iter()
            .zip(other_flat.iter())
            .map(|(a, b)| a <= b)
            .collect())
    }

    /// Element-wise equality comparison
    pub fn eq_elementwise(&self, other: &Self) -> anyhow::Result<Vec<bool>> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape());
        }
        let self_flat = self
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        let other_flat = other
            .data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Cannot compare non-contiguous tensor"))?;
        Ok(self_flat
            .iter()
            .zip(other_flat.iter())
            .map(|(a, b)| a == b)
            .collect())
    }

    /// Create a mask for values greater than a threshold
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let mask = tensor.mask_gt(2.5);
    ///
    /// assert_eq!(mask, vec![false, false, true, true]);
    /// ```
    pub fn mask_gt(&self, threshold: T) -> Vec<bool> {
        self.data.iter().map(|x| x > &threshold).collect()
    }

    /// Create a mask for values less than a threshold
    pub fn mask_lt(&self, threshold: T) -> Vec<bool> {
        self.data.iter().map(|x| x < &threshold).collect()
    }

    /// Get the count of elements satisfying a condition
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let count = tensor.count_if(|&x| x > 3.0);
    ///
    /// assert_eq!(count, 2); // 4.0 and 5.0
    /// ```
    pub fn count_if<F>(&self, predicate: F) -> usize
    where
        F: Fn(&T) -> bool,
    {
        self.data.iter().filter(|x| predicate(x)).count()
    }

    /// Test if all elements satisfy a predicate
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4]).unwrap();
    /// assert!(tensor.all(|&x| x > 0.0));
    /// assert!(!tensor.all(|&x| x > 5.0));
    /// ```
    pub fn all<F>(&self, predicate: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        self.data.iter().all(predicate)
    }

    /// Test if any element satisfies a predicate
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// assert!(tensor.any(|&x| x > 3.0));
    /// assert!(!tensor.any(|&x| x > 10.0));
    /// ```
    pub fn any<F>(&self, predicate: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        self.data.iter().any(predicate)
    }

    /// Select elements from two tensors based on a condition.
    ///
    /// Returns a tensor where each element is selected from `true_values` if the
    /// corresponding condition is true, otherwise from `false_values`.
    ///
    /// This is the tensor equivalent of `condition ? true_values : false_values`.
    ///
    /// # Arguments
    ///
    /// * `condition` - Boolean array (same shape as self)
    /// * `true_values` - Values to use where condition is true
    /// * `false_values` - Values to use where condition is false
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let x = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// let condition = vec![true, false, true, false];
    /// let a = DenseND::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
    /// let b = DenseND::<f64>::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[4]).unwrap();
    ///
    /// let result = DenseND::where_select(&condition, &a, &b).unwrap();
    /// assert_eq!(result[&[0]], 10.0);   // condition[0] = true, select from a
    /// assert_eq!(result[&[1]], 200.0);  // condition[1] = false, select from b
    /// assert_eq!(result[&[2]], 30.0);   // condition[2] = true, select from a
    /// assert_eq!(result[&[3]], 400.0);  // condition[3] = false, select from b
    /// ```
    pub fn where_select(
        condition: &[bool],
        true_values: &Self,
        false_values: &Self,
    ) -> anyhow::Result<Self> {
        if true_values.shape() != false_values.shape() {
            anyhow::bail!(
                "Shape mismatch: true_values {:?} vs false_values {:?}",
                true_values.shape(),
                false_values.shape()
            );
        }

        if condition.len() != true_values.len() {
            anyhow::bail!(
                "Condition length {} does not match tensor size {}",
                condition.len(),
                true_values.len()
            );
        }

        let result_data: Vec<T> = condition
            .iter()
            .zip(true_values.as_slice().iter())
            .zip(false_values.as_slice().iter())
            .map(|((&cond, t), f)| if cond { t.clone() } else { f.clone() })
            .collect();

        Self::from_vec(result_data, true_values.shape())
    }

    /// Replace elements based on a condition (in-place variant).
    ///
    /// Replaces elements with `value` where the condition is true.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// let condition = vec![false, true, false, true];
    /// tensor.where_replace(&condition, -1.0).unwrap();
    ///
    /// assert_eq!(tensor[&[0]], 1.0);
    /// assert_eq!(tensor[&[1]], -1.0);
    /// assert_eq!(tensor[&[2]], 3.0);
    /// assert_eq!(tensor[&[3]], -1.0);
    /// ```
    pub fn where_replace(&mut self, condition: &[bool], value: T) -> anyhow::Result<()> {
        if condition.len() != self.len() {
            anyhow::bail!(
                "Condition length {} does not match tensor size {}",
                condition.len(),
                self.len()
            );
        }

        for (i, &cond) in condition.iter().enumerate() {
            if cond {
                self.data.as_slice_mut().unwrap()[i] = value.clone();
            }
        }

        Ok(())
    }

    /// Select values based on condition, replacing others with a default.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// let condition = vec![true, false, true, false];
    /// let result = tensor.select_or_default(&condition, 0.0).unwrap();
    ///
    /// assert_eq!(result[&[0]], 1.0);
    /// assert_eq!(result[&[1]], 0.0);
    /// assert_eq!(result[&[2]], 3.0);
    /// assert_eq!(result[&[3]], 0.0);
    /// ```
    pub fn select_or_default(&self, condition: &[bool], default: T) -> anyhow::Result<Self> {
        if condition.len() != self.len() {
            anyhow::bail!(
                "Condition length {} does not match tensor size {}",
                condition.len(),
                self.len()
            );
        }

        let result_data: Vec<T> = condition
            .iter()
            .zip(self.as_slice().iter())
            .map(|(&cond, val)| if cond { val.clone() } else { default.clone() })
            .collect();

        Self::from_vec(result_data, self.shape())
    }
}
