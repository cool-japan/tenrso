//! Element-wise operations on tensors
//!
//! This module provides element-wise mathematical operations and transformations
//! including mathematical functions (sqrt, exp, ln, pow), clipping, and mapping.

use super::types::DenseND;
use scirs2_core::numeric::Num;

impl<T> DenseND<T>
where
    T: Clone + Num + PartialOrd,
{
    /// Apply element-wise absolute value.
    ///
    /// Returns a new tensor where each element is the absolute value of the input.
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
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![-1.0, 2.0, -3.0, 4.0],
    ///     &[2, 2]
    /// ).unwrap();
    ///
    /// let abs_tensor = tensor.abs();
    /// assert_eq!(abs_tensor[&[0, 0]], 1.0);
    /// assert_eq!(abs_tensor[&[0, 1]], 2.0);
    /// assert_eq!(abs_tensor[&[1, 0]], 3.0);
    /// assert_eq!(abs_tensor[&[1, 1]], 4.0);
    /// ```
    pub fn abs(&self) -> Self
    where
        T: scirs2_core::numeric::Signed,
    {
        let abs_data: Vec<T> = self.data.iter().map(|x| x.abs()).collect();
        Self::from_vec(abs_data, self.shape()).unwrap()
    }

    /// Clip values to be within [min_val, max_val]
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let clipped = tensor.clip(2.0, 5.0);
    ///
    /// assert_eq!(clipped[&[0, 0]], 2.0); // 1.0 clipped to 2.0
    /// assert_eq!(clipped[&[0, 1]], 2.0);
    /// assert_eq!(clipped[&[0, 2]], 3.0);
    /// assert_eq!(clipped[&[1, 2]], 5.0); // 6.0 clipped to 5.0
    /// ```
    pub fn clip(&self, min_val: T, max_val: T) -> Self {
        let clipped = self.data.mapv(|x| {
            if x < min_val {
                min_val.clone()
            } else if x > max_val {
                max_val.clone()
            } else {
                x
            }
        });
        Self { data: clipped }
    }
}

impl<T> DenseND<T>
where
    T: Clone + scirs2_core::numeric::Float,
{
    /// Apply element-wise square root.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[2, 2]).unwrap();
    /// let sqrt_tensor = tensor.sqrt();
    /// assert_eq!(sqrt_tensor[&[0, 0]], 1.0);
    /// assert_eq!(sqrt_tensor[&[0, 1]], 2.0);
    /// assert_eq!(sqrt_tensor[&[1, 0]], 3.0);
    /// assert_eq!(sqrt_tensor[&[1, 1]], 4.0);
    /// ```
    pub fn sqrt(&self) -> Self {
        let sqrt_data: Vec<T> = self.data.iter().map(|x| x.sqrt()).collect();
        Self::from_vec(sqrt_data, self.shape()).unwrap()
    }

    /// Apply element-wise exponential (e^x).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
    /// let exp_tensor = tensor.exp();
    /// assert!((exp_tensor[&[0]] - 1.0).abs() < 1e-10);
    /// assert!((exp_tensor[&[1]] - 2.718281828).abs() < 1e-6);
    /// ```
    pub fn exp(&self) -> Self {
        let exp_data: Vec<T> = self.data.iter().map(|x| x.exp()).collect();
        Self::from_vec(exp_data, self.shape()).unwrap()
    }

    /// Apply element-wise natural logarithm (ln(x)).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.718281828, 7.389], &[3]).unwrap();
    /// let log_tensor = tensor.ln();
    /// assert!((log_tensor[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((log_tensor[&[1]] - 1.0).abs() < 1e-6);
    /// ```
    pub fn ln(&self) -> Self {
        let log_data: Vec<T> = self.data.iter().map(|x| x.ln()).collect();
        Self::from_vec(log_data, self.shape()).unwrap()
    }

    /// Apply element-wise power (x^n).
    ///
    /// # Arguments
    ///
    /// * `n` - The exponent
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let squared = tensor.powf(2.0);
    /// assert_eq!(squared[&[0, 0]], 1.0);
    /// assert_eq!(squared[&[0, 1]], 4.0);
    /// assert_eq!(squared[&[1, 0]], 9.0);
    /// assert_eq!(squared[&[1, 1]], 16.0);
    /// ```
    pub fn powf(&self, n: T) -> Self {
        let pow_data: Vec<T> = self.data.iter().map(|x| x.powf(n)).collect();
        Self::from_vec(pow_data, self.shape()).unwrap()
    }

    // ========== Trigonometric Functions ==========

    /// Apply element-wise sine function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, std::f64::consts::PI / 2.0], &[2]).unwrap();
    /// let result = tensor.sin();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 1.0).abs() < 1e-10);
    /// ```
    pub fn sin(&self) -> Self {
        let sin_data: Vec<T> = self.data.iter().map(|x| x.sin()).collect();
        Self::from_vec(sin_data, self.shape()).unwrap()
    }

    /// Apply element-wise cosine function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, std::f64::consts::PI], &[2]).unwrap();
    /// let result = tensor.cos();
    /// assert!((result[&[0]] - 1.0).abs() < 1e-10);
    /// assert!((result[&[1]] + 1.0).abs() < 1e-10);
    /// ```
    pub fn cos(&self) -> Self {
        let cos_data: Vec<T> = self.data.iter().map(|x| x.cos()).collect();
        Self::from_vec(cos_data, self.shape()).unwrap()
    }

    /// Apply element-wise tangent function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, std::f64::consts::PI / 4.0], &[2]).unwrap();
    /// let result = tensor.tan();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 1.0).abs() < 1e-10);
    /// ```
    pub fn tan(&self) -> Self {
        let tan_data: Vec<T> = self.data.iter().map(|x| x.tan()).collect();
        Self::from_vec(tan_data, self.shape()).unwrap()
    }

    /// Apply element-wise arcsine (inverse sine) function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 0.5, 1.0], &[3]).unwrap();
    /// let result = tensor.asin();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[2]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
    /// ```
    pub fn asin(&self) -> Self {
        let asin_data: Vec<T> = self.data.iter().map(|x| x.asin()).collect();
        Self::from_vec(asin_data, self.shape()).unwrap()
    }

    /// Apply element-wise arccosine (inverse cosine) function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 0.0, -1.0], &[3]).unwrap();
    /// let result = tensor.acos();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
    /// ```
    pub fn acos(&self) -> Self {
        let acos_data: Vec<T> = self.data.iter().map(|x| x.acos()).collect();
        Self::from_vec(acos_data, self.shape()).unwrap()
    }

    /// Apply element-wise arctangent (inverse tangent) function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    /// let result = tensor.atan();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - std::f64::consts::PI / 4.0).abs() < 1e-10);
    /// ```
    pub fn atan(&self) -> Self {
        let atan_data: Vec<T> = self.data.iter().map(|x| x.atan()).collect();
        Self::from_vec(atan_data, self.shape()).unwrap()
    }

    /// Apply element-wise two-argument arctangent function (atan2).
    ///
    /// Computes atan2(self, other) for each element pair.
    ///
    /// # Arguments
    ///
    /// * `other` - The x-coordinates (denominator)
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
    /// let y = DenseND::<f64>::from_vec(vec![1.0, 1.0], &[2]).unwrap();
    /// let x = DenseND::<f64>::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    /// let result = y.atan2(&x).unwrap();
    /// assert!((result[&[0]] - std::f64::consts::PI / 4.0).abs() < 1e-10);
    /// assert!((result[&[1]] - std::f64::consts::PI / 2.0).abs() < 1e-10);
    /// ```
    pub fn atan2(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.shape() == other.shape(),
            "Shapes must match for atan2: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );
        let atan2_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(y, x)| y.atan2(*x))
            .collect();
        Ok(Self::from_vec(atan2_data, self.shape()).unwrap())
    }

    // ========== Hyperbolic Functions ==========

    /// Apply element-wise hyperbolic sine function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    /// let result = tensor.sinh();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 1.1752011936438014).abs() < 1e-10);
    /// ```
    pub fn sinh(&self) -> Self {
        let sinh_data: Vec<T> = self.data.iter().map(|x| x.sinh()).collect();
        Self::from_vec(sinh_data, self.shape()).unwrap()
    }

    /// Apply element-wise hyperbolic cosine function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    /// let result = tensor.cosh();
    /// assert!((result[&[0]] - 1.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 1.5430806348152437).abs() < 1e-10);
    /// ```
    pub fn cosh(&self) -> Self {
        let cosh_data: Vec<T> = self.data.iter().map(|x| x.cosh()).collect();
        Self::from_vec(cosh_data, self.shape()).unwrap()
    }

    /// Apply element-wise hyperbolic tangent function.
    ///
    /// Note: This is the mathematical tanh function. For the activation function,
    /// use `tanh_activation()` which has the same implementation but clearer naming.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    /// let result = tensor.tanh();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 0.7615941559557649).abs() < 1e-10);
    /// ```
    pub fn tanh(&self) -> Self {
        let tanh_data: Vec<T> = self.data.iter().map(|x| x.tanh()).collect();
        Self::from_vec(tanh_data, self.shape()).unwrap()
    }

    /// Apply element-wise inverse hyperbolic sine function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    /// let result = tensor.asinh();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 0.881373587019543).abs() < 1e-10);
    /// ```
    pub fn asinh(&self) -> Self {
        let asinh_data: Vec<T> = self.data.iter().map(|x| x.asinh()).collect();
        Self::from_vec(asinh_data, self.shape()).unwrap()
    }

    /// Apply element-wise inverse hyperbolic cosine function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    /// let result = tensor.acosh();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 1.3169578969248166).abs() < 1e-10);
    /// ```
    pub fn acosh(&self) -> Self {
        let acosh_data: Vec<T> = self.data.iter().map(|x| x.acosh()).collect();
        Self::from_vec(acosh_data, self.shape()).unwrap()
    }

    /// Apply element-wise inverse hyperbolic tangent function.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 0.5], &[2]).unwrap();
    /// let result = tensor.atanh();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 0.5493061443340548).abs() < 1e-10);
    /// ```
    pub fn atanh(&self) -> Self {
        let atanh_data: Vec<T> = self.data.iter().map(|x| x.atanh()).collect();
        Self::from_vec(atanh_data, self.shape()).unwrap()
    }

    // ========== Additional Logarithmic & Exponential Functions ==========

    /// Apply element-wise base-2 logarithm.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 4.0, 8.0], &[4]).unwrap();
    /// let result = tensor.log2();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 1.0).abs() < 1e-10);
    /// assert!((result[&[2]] - 2.0).abs() < 1e-10);
    /// assert!((result[&[3]] - 3.0).abs() < 1e-10);
    /// ```
    pub fn log2(&self) -> Self {
        let log2_data: Vec<T> = self.data.iter().map(|x| x.log2()).collect();
        Self::from_vec(log2_data, self.shape()).unwrap()
    }

    /// Apply element-wise base-10 logarithm.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 10.0, 100.0], &[3]).unwrap();
    /// let result = tensor.log10();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 1.0).abs() < 1e-10);
    /// assert!((result[&[2]] - 2.0).abs() < 1e-10);
    /// ```
    pub fn log10(&self) -> Self {
        let log10_data: Vec<T> = self.data.iter().map(|x| x.log10()).collect();
        Self::from_vec(log10_data, self.shape()).unwrap()
    }

    /// Apply element-wise ln(1 + x).
    ///
    /// More accurate than ln for values close to zero.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    /// let result = tensor.log1p();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 2.0_f64.ln()).abs() < 1e-10);
    /// ```
    pub fn log1p(&self) -> Self {
        let log1p_data: Vec<T> = self.data.iter().map(|x| x.ln_1p()).collect();
        Self::from_vec(log1p_data, self.shape()).unwrap()
    }

    /// Apply element-wise exp(x) - 1.
    ///
    /// More accurate than exp for values close to zero.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    /// let result = tensor.expm1();
    /// assert!((result[&[0]] - 0.0).abs() < 1e-10);
    /// assert!((result[&[1]] - (std::f64::consts::E - 1.0)).abs() < 1e-10);
    /// ```
    pub fn expm1(&self) -> Self {
        let expm1_data: Vec<T> = self.data.iter().map(|x| x.exp_m1()).collect();
        Self::from_vec(expm1_data, self.shape()).unwrap()
    }

    /// Apply element-wise 2^x.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4]).unwrap();
    /// let result = tensor.exp2();
    /// assert!((result[&[0]] - 1.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 2.0).abs() < 1e-10);
    /// assert!((result[&[2]] - 4.0).abs() < 1e-10);
    /// assert!((result[&[3]] - 8.0).abs() < 1e-10);
    /// ```
    pub fn exp2(&self) -> Self {
        let exp2_data: Vec<T> = self.data.iter().map(|x| x.exp2()).collect();
        Self::from_vec(exp2_data, self.shape()).unwrap()
    }

    // ========== Power & Root Functions ==========

    /// Apply element-wise square (x^2).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// let result = tensor.square();
    /// assert_eq!(result[&[0]], 1.0);
    /// assert_eq!(result[&[1]], 4.0);
    /// assert_eq!(result[&[2]], 9.0);
    /// assert_eq!(result[&[3]], 16.0);
    /// ```
    pub fn square(&self) -> Self {
        let square_data: Vec<T> = self.data.iter().map(|x| *x * *x).collect();
        Self::from_vec(square_data, self.shape()).unwrap()
    }

    /// Apply element-wise cube (x^3).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let result = tensor.cube();
    /// assert_eq!(result[&[0]], 1.0);
    /// assert_eq!(result[&[1]], 8.0);
    /// assert_eq!(result[&[2]], 27.0);
    /// ```
    pub fn cube(&self) -> Self {
        let cube_data: Vec<T> = self.data.iter().map(|x| *x * *x * *x).collect();
        Self::from_vec(cube_data, self.shape()).unwrap()
    }

    /// Apply element-wise cube root.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 8.0, 27.0], &[3]).unwrap();
    /// let result = tensor.cbrt();
    /// assert!((result[&[0]] - 1.0).abs() < 1e-10);
    /// assert!((result[&[1]] - 2.0).abs() < 1e-10);
    /// assert!((result[&[2]] - 3.0).abs() < 1e-10);
    /// ```
    pub fn cbrt(&self) -> Self {
        let cbrt_data: Vec<T> = self.data.iter().map(|x| x.cbrt()).collect();
        Self::from_vec(cbrt_data, self.shape()).unwrap()
    }

    /// Apply element-wise reciprocal (1/x).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 4.0], &[3]).unwrap();
    /// let result = tensor.recip();
    /// assert_eq!(result[&[0]], 1.0);
    /// assert_eq!(result[&[1]], 0.5);
    /// assert_eq!(result[&[2]], 0.25);
    /// ```
    pub fn recip(&self) -> Self {
        let recip_data: Vec<T> = self.data.iter().map(|x| x.recip()).collect();
        Self::from_vec(recip_data, self.shape()).unwrap()
    }

    // ========== Rounding Functions ==========

    /// Apply element-wise rounding to nearest integer.
    ///
    /// Uses banker's rounding (round half to even) which is the default Rust behavior.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.2, 1.6, 1.8, 2.3], &[4]).unwrap();
    /// let result = tensor.round();
    /// assert_eq!(result[&[0]], 1.0);
    /// assert_eq!(result[&[1]], 2.0);
    /// assert_eq!(result[&[2]], 2.0);
    /// assert_eq!(result[&[3]], 2.0);
    /// ```
    pub fn round(&self) -> Self {
        let round_data: Vec<T> = self.data.iter().map(|x| x.round()).collect();
        Self::from_vec(round_data, self.shape()).unwrap()
    }

    /// Apply element-wise floor (round down to nearest integer).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.2, 1.8, -1.2, -1.8], &[4]).unwrap();
    /// let result = tensor.floor();
    /// assert_eq!(result[&[0]], 1.0);
    /// assert_eq!(result[&[1]], 1.0);
    /// assert_eq!(result[&[2]], -2.0);
    /// assert_eq!(result[&[3]], -2.0);
    /// ```
    pub fn floor(&self) -> Self {
        let floor_data: Vec<T> = self.data.iter().map(|x| x.floor()).collect();
        Self::from_vec(floor_data, self.shape()).unwrap()
    }

    /// Apply element-wise ceiling (round up to nearest integer).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.2, 1.8, -1.2, -1.8], &[4]).unwrap();
    /// let result = tensor.ceil();
    /// assert_eq!(result[&[0]], 2.0);
    /// assert_eq!(result[&[1]], 2.0);
    /// assert_eq!(result[&[2]], -1.0);
    /// assert_eq!(result[&[3]], -1.0);
    /// ```
    pub fn ceil(&self) -> Self {
        let ceil_data: Vec<T> = self.data.iter().map(|x| x.ceil()).collect();
        Self::from_vec(ceil_data, self.shape()).unwrap()
    }

    /// Apply element-wise truncation (round towards zero).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.8, -1.8, 2.3, -2.3], &[4]).unwrap();
    /// let result = tensor.trunc();
    /// assert_eq!(result[&[0]], 1.0);
    /// assert_eq!(result[&[1]], -1.0);
    /// assert_eq!(result[&[2]], 2.0);
    /// assert_eq!(result[&[3]], -2.0);
    /// ```
    pub fn trunc(&self) -> Self {
        let trunc_data: Vec<T> = self.data.iter().map(|x| x.trunc()).collect();
        Self::from_vec(trunc_data, self.shape()).unwrap()
    }

    /// Apply element-wise fractional part extraction.
    ///
    /// Returns the fractional part of each element (x - trunc(x)).
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
    /// let tensor = DenseND::<f64>::from_vec(vec![1.25, -1.75, 2.5], &[3]).unwrap();
    /// let result = tensor.fract();
    /// assert!((result[&[0]] - 0.25).abs() < 1e-10);
    /// assert!((result[&[1]] + 0.75).abs() < 1e-10);
    /// assert!((result[&[2]] - 0.5).abs() < 1e-10);
    /// ```
    pub fn fract(&self) -> Self {
        let fract_data: Vec<T> = self.data.iter().map(|x| x.fract()).collect();
        Self::from_vec(fract_data, self.shape()).unwrap()
    }

    // ========== Sign Functions ==========

    /// Apply element-wise sign function.
    ///
    /// Returns -1.0 for negative, 1.0 for positive, and a value determined by the
    /// platform for zero values.
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
    /// let tensor = DenseND::<f64>::from_vec(vec![-2.5, 3.7], &[2]).unwrap();
    /// let result = tensor.signum();
    /// assert_eq!(result[&[0]], -1.0);
    /// assert_eq!(result[&[1]], 1.0);
    /// ```
    pub fn signum(&self) -> Self {
        let signum_data: Vec<T> = self.data.iter().map(|x| x.signum()).collect();
        Self::from_vec(signum_data, self.shape()).unwrap()
    }
}

impl<T> DenseND<T>
where
    T: Clone + Num,
{
    /// Fill all elements with a value
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::zeros(&[2, 3]);
    /// tensor.fill(5.0);
    ///
    /// assert_eq!(tensor[&[0, 0]], 5.0);
    /// assert_eq!(tensor[&[1, 2]], 5.0);
    /// ```
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    /// Apply a function element-wise
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let doubled = tensor.map(|x| x * 2.0);
    ///
    /// assert_eq!(doubled[&[0, 0]], 2.0);
    /// assert_eq!(doubled[&[1, 1]], 8.0);
    /// ```
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let mapped = self.data.mapv(f);
        Self { data: mapped }
    }

    /// Apply a function element-wise in-place
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// tensor.map_inplace(|x| x * 2.0);
    ///
    /// assert_eq!(tensor[&[0, 0]], 2.0);
    /// assert_eq!(tensor[&[1, 1]], 8.0);
    /// ```
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.data.mapv_inplace(f);
    }
}

// Activation functions for neural networks
impl<T> DenseND<T>
where
    T: Clone
        + scirs2_core::numeric::Float
        + scirs2_core::numeric::FromPrimitive
        + scirs2_core::numeric::NumCast
        + std::iter::Sum
        + std::ops::Mul<Output = T>,
{
    /// Apply ReLU (Rectified Linear Unit) activation function.
    ///
    /// ReLU(x) = max(0, x)
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
    /// let tensor = DenseND::<f64>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
    /// let activated = tensor.relu();
    ///
    /// assert_eq!(activated[&[0]], 0.0);
    /// assert_eq!(activated[&[1]], 0.0);
    /// assert_eq!(activated[&[2]], 0.0);
    /// assert_eq!(activated[&[3]], 1.0);
    /// assert_eq!(activated[&[4]], 2.0);
    /// ```
    pub fn relu(&self) -> Self {
        let zero = T::zero();
        let relu_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| if x > zero { x } else { zero })
            .collect();
        Self::from_vec(relu_data, self.shape()).unwrap()
    }

    /// Apply Leaky ReLU activation function.
    ///
    /// LeakyReLU(x) = max(alpha * x, x)
    ///
    /// # Arguments
    ///
    /// * `alpha` - Slope for negative values (typically 0.01)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-2.0, -1.0, 0.0, 1.0], &[4]).unwrap();
    /// let activated = tensor.leaky_relu(0.01);
    ///
    /// assert_eq!(activated[&[0]], -0.02);
    /// assert_eq!(activated[&[1]], -0.01);
    /// assert_eq!(activated[&[3]], 1.0);
    /// ```
    pub fn leaky_relu(&self, alpha: T) -> Self {
        let zero = T::zero();
        let leaky_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| if x > zero { x } else { alpha * x })
            .collect();
        Self::from_vec(leaky_data, self.shape()).unwrap()
    }

    /// Apply ELU (Exponential Linear Unit) activation function.
    ///
    /// ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
    ///
    /// # Arguments
    ///
    /// * `alpha` - Scale for negative values (typically 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    /// let activated = tensor.elu(1.0);
    ///
    /// assert_eq!(activated[&[1]], 0.0);
    /// assert_eq!(activated[&[2]], 1.0);
    /// ```
    pub fn elu(&self, alpha: T) -> Self {
        let zero = T::zero();
        let one = T::one();
        let elu_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| if x > zero { x } else { alpha * (x.exp() - one) })
            .collect();
        Self::from_vec(elu_data, self.shape()).unwrap()
    }

    /// Apply sigmoid activation function.
    ///
    /// sigmoid(x) = 1 / (1 + exp(-x))
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    /// let activated = tensor.sigmoid();
    ///
    /// assert!((activated[&[1]] - 0.5).abs() < 1e-10);
    /// ```
    pub fn sigmoid(&self) -> Self {
        let one = T::one();
        let sigmoid_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| one / (one + (-x).exp()))
            .collect();
        Self::from_vec(sigmoid_data, self.shape()).unwrap()
    }

    /// Apply tanh (hyperbolic tangent) activation function.
    ///
    /// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    /// let activated = tensor.tanh_activation();
    ///
    /// assert!((activated[&[1]] - 0.0).abs() < 1e-10);
    /// ```
    pub fn tanh_activation(&self) -> Self {
        let tanh_data: Vec<T> = self.data.iter().map(|&x| x.tanh()).collect();
        Self::from_vec(tanh_data, self.shape()).unwrap()
    }

    /// Apply Swish/SiLU activation function.
    ///
    /// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    /// let activated = tensor.swish();
    ///
    /// assert_eq!(activated[&[1]], 0.0);
    /// ```
    pub fn swish(&self) -> Self {
        let one = T::one();
        let swish_data: Vec<T> = self.data.iter().map(|&x| x / (one + (-x).exp())).collect();
        Self::from_vec(swish_data, self.shape()).unwrap()
    }

    /// Apply GELU (Gaussian Error Linear Unit) activation function.
    ///
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
    /// let activated = tensor.gelu();
    ///
    /// assert!((activated[&[1]] - 0.0).abs() < 1e-10);
    /// ```
    pub fn gelu(&self) -> Self {
        let half = T::from_f64(0.5).unwrap();
        let one = T::one();
        let coeff = T::from_f64(0.7978845608028654).unwrap(); // sqrt(2/pi)
        let cubic_coeff = T::from_f64(0.044715).unwrap();

        let gelu_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| {
                let x3 = x * x * x;
                let inner = coeff * (x + cubic_coeff * x3);
                half * x * (one + inner.tanh())
            })
            .collect();
        Self::from_vec(gelu_data, self.shape()).unwrap()
    }

    /// Clip gradient values to be within [-clip_value, clip_value].
    ///
    /// This is commonly used to prevent exploding gradients in neural network training.
    ///
    /// # Arguments
    ///
    /// * `clip_value` - Maximum absolute value
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let gradients = DenseND::<f64>::from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5]).unwrap();
    /// let clipped = gradients.clip_by_value(5.0);
    ///
    /// assert_eq!(clipped[&[0]], -5.0);
    /// assert_eq!(clipped[&[1]], -1.0);
    /// assert_eq!(clipped[&[2]], 0.0);
    /// assert_eq!(clipped[&[3]], 1.0);
    /// assert_eq!(clipped[&[4]], 5.0);
    /// ```
    pub fn clip_by_value(&self, clip_value: T) -> Self {
        let neg_clip = -clip_value;
        let clipped_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| {
                if x > clip_value {
                    clip_value
                } else if x < neg_clip {
                    neg_clip
                } else {
                    x
                }
            })
            .collect();
        Self::from_vec(clipped_data, self.shape()).unwrap()
    }

    /// Clip gradient tensor by L2 norm.
    ///
    /// If the L2 norm of the gradient exceeds max_norm, scale the entire tensor
    /// so that its L2 norm equals max_norm.
    ///
    /// # Arguments
    ///
    /// * `max_norm` - Maximum L2 norm
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let gradients = DenseND::<f64>::from_vec(vec![3.0, 4.0], &[2]).unwrap();
    /// let clipped = gradients.clip_by_norm(2.5);
    ///
    /// // Original norm is 5.0, should be scaled to 2.5
    /// let norm = clipped.norm_l2();
    /// assert!((norm - 2.5).abs() < 1e-10);
    /// ```
    pub fn clip_by_norm(&self, max_norm: T) -> Self {
        let norm = self.norm_l2();
        if norm <= max_norm {
            self.clone()
        } else {
            let scale = max_norm / norm;
            let clipped_data: Vec<T> = self.data.iter().map(|&x| x * scale).collect();
            Self::from_vec(clipped_data, self.shape()).unwrap()
        }
    }
}
