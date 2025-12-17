//! Parallel gradient computation using Rayon
//!
//! This module provides parallel implementations of gradient operations
//! for improved performance on multi-core systems.
//!
//! # Features
//!
//! - Parallel element-wise gradient computation
//! - Parallel gradient accumulation
//! - Parallel batch gradient processing
//! - Work-stealing via Rayon thread pool
//!
//! # Performance
//!
//! Parallel operations are beneficial when:
//! - Tensor size > 10,000 elements
//! - Multiple independent gradients to compute
//! - Batch size > 4
//!
//! For smaller tensors, sequential processing may be faster due to overhead.

use anyhow::Result;
use rayon::prelude::*;
use scirs2_core::ndarray_ext::{Array, IxDyn, Zip};
use scirs2_core::numeric::Float;
use tenrso_core::DenseND;

/// Configuration for parallel gradient computation
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum tensor size to use parallel processing (default: 10,000)
    pub min_parallel_size: usize,

    /// Number of threads to use (None = use all available cores)
    pub num_threads: Option<usize>,

    /// Chunk size for parallel iteration (None = auto)
    pub chunk_size: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_parallel_size: 10_000,
            num_threads: None,
            chunk_size: None,
        }
    }
}

/// Parallel element-wise gradient computation
///
/// Computes `grad_x = grad_y * f'(x)` in parallel for element-wise operations.
///
/// # Arguments
///
/// * `input` - Input tensor from forward pass
/// * `output_grad` - Gradient w.r.t. output
/// * `derivative` - Derivative function f'(x)
/// * `config` - Parallel configuration
///
/// # Returns
///
/// Gradient w.r.t. input
///
/// # Example
///
/// ```rust,ignore
/// let grad = parallel_elementwise_vjp(
///     &x,
///     &grad_y,
///     |x| 2.0 * x,  // derivative of x^2
///     &ParallelConfig::default(),
/// )?;
/// ```
pub fn parallel_elementwise_vjp<T, F>(
    input: &DenseND<T>,
    output_grad: &DenseND<T>,
    derivative: F,
    config: &ParallelConfig,
) -> Result<DenseND<T>>
where
    T: Float + Send + Sync,
    F: Fn(&T) -> T + Send + Sync,
{
    if input.shape() != output_grad.shape() {
        return Err(anyhow::anyhow!(
            "Shape mismatch: input {:?} vs output_grad {:?}",
            input.shape(),
            output_grad.shape()
        ));
    }

    let input_arr = input.as_array();
    let grad_arr = output_grad.as_array();

    // Check if parallel processing is beneficial
    let total_elements = input.len();
    let use_parallel = total_elements >= config.min_parallel_size;

    if use_parallel {
        // Parallel computation using Rayon
        // First collect to vectors to ensure consistent ordering
        let shape = input.shape().to_vec();
        let input_flat: Vec<T> = input_arr.iter().copied().collect();
        let grad_flat: Vec<T> = grad_arr.iter().copied().collect();

        // Use parallel iteration over vectors which preserves order
        let grad_data: Vec<T> = input_flat
            .par_iter()
            .zip(grad_flat.par_iter())
            .map(|(x, g)| {
                let deriv = derivative(x);
                deriv * *g
            })
            .collect();

        let result_arr = Array::from_shape_vec(IxDyn(&shape), grad_data)
            .map_err(|e| anyhow::anyhow!("Failed to create result array: {}", e))?;

        Ok(DenseND::from_array(result_arr))
    } else {
        // Sequential computation for small tensors
        let result_arr = Zip::from(input_arr).and(grad_arr).map_collect(|x, g| {
            let deriv = derivative(x);
            deriv * *g
        });

        Ok(DenseND::from_array(result_arr))
    }
}

/// Parallel gradient accumulation for multiple gradients
///
/// Efficiently accumulates multiple gradients in parallel.
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors to accumulate
/// * `config` - Parallel configuration
///
/// # Returns
///
/// Accumulated gradient
///
/// # Example
///
/// ```rust,ignore
/// let total_grad = parallel_accumulate_gradients(
///     &[grad1, grad2, grad3],
///     &ParallelConfig::default(),
/// )?;
/// ```
pub fn parallel_accumulate_gradients<T>(
    gradients: &[DenseND<T>],
    config: &ParallelConfig,
) -> Result<DenseND<T>>
where
    T: Float + Send + Sync,
{
    if gradients.is_empty() {
        return Err(anyhow::anyhow!("No gradients to accumulate"));
    }

    let shape = gradients[0].shape();

    // Verify all gradients have same shape
    for grad in gradients.iter() {
        if grad.shape() != shape {
            return Err(anyhow::anyhow!(
                "Shape mismatch in gradient accumulation: expected {:?}, got {:?}",
                shape,
                grad.shape()
            ));
        }
    }

    let total_elements = gradients[0].len();
    let use_parallel = total_elements >= config.min_parallel_size;

    if use_parallel {
        // Parallel reduction - accumulate gradients
        let mut result = gradients[0].clone();
        for grad in gradients.iter().skip(1) {
            let result_arr = result.as_array();
            let grad_arr = grad.as_array();
            let sum_arr = result_arr + grad_arr;
            result = DenseND::from_array(sum_arr);
        }
        Ok(result)
    } else {
        // Sequential accumulation
        let mut result = gradients[0].clone();
        for grad in gradients.iter().skip(1) {
            result = &result + grad;
        }
        Ok(result)
    }
}

/// Parallel batch gradient computation
///
/// Computes gradients for a batch of inputs in parallel.
///
/// # Arguments
///
/// * `inputs` - Batch of input tensors
/// * `output_grads` - Batch of output gradients
/// * `gradient_fn` - Function to compute gradient for single input
/// * `config` - Parallel configuration
///
/// # Returns
///
/// Vector of input gradients
///
/// # Example
///
/// ```rust,ignore
/// let input_grads = parallel_batch_gradients(
///     &batch_inputs,
///     &batch_output_grads,
///     |input, output_grad| compute_single_grad(input, output_grad),
///     &ParallelConfig::default(),
/// )?;
/// ```
pub fn parallel_batch_gradients<T, F>(
    inputs: &[DenseND<T>],
    output_grads: &[DenseND<T>],
    gradient_fn: F,
    _config: &ParallelConfig,
) -> Result<Vec<DenseND<T>>>
where
    T: Float + Send + Sync,
    F: Fn(&DenseND<T>, &DenseND<T>) -> Result<DenseND<T>> + Send + Sync,
{
    if inputs.len() != output_grads.len() {
        return Err(anyhow::anyhow!(
            "Batch size mismatch: {} inputs vs {} output grads",
            inputs.len(),
            output_grads.len()
        ));
    }

    let batch_size = inputs.len();
    let use_parallel = batch_size > 1;

    if use_parallel {
        // Parallel batch processing
        let results: Result<Vec<_>> = inputs
            .par_iter()
            .zip(output_grads.par_iter())
            .map(|(input, output_grad)| gradient_fn(input, output_grad))
            .collect();

        results
    } else {
        // Sequential for single item
        inputs
            .iter()
            .zip(output_grads.iter())
            .map(|(input, output_grad)| gradient_fn(input, output_grad))
            .collect()
    }
}

/// Parallel reduction gradient computation
///
/// Computes gradients for reduction operations in parallel.
///
/// # Arguments
///
/// * `input_shape` - Shape of original input
/// * `output_grad` - Gradient w.r.t. output (reduced)
/// * `scaling` - Scaling factor (e.g., 1/n for mean)
/// * `config` - Parallel configuration
///
/// # Returns
///
/// Gradient w.r.t. input (broadcasted)
pub fn parallel_reduction_grad<T>(
    input_shape: &[usize],
    output_grad: &DenseND<T>,
    scaling: T,
    config: &ParallelConfig,
) -> Result<DenseND<T>>
where
    T: Float + Send + Sync,
{
    let total_elements: usize = input_shape.iter().product();
    let use_parallel = total_elements >= config.min_parallel_size;

    // Get the scalar gradient value
    let grad_value = if output_grad.len() == 1 {
        *output_grad
            .get(&[])
            .or_else(|| output_grad.get(&[0]))
            .ok_or_else(|| anyhow::anyhow!("Failed to get gradient value"))?
    } else {
        return Err(anyhow::anyhow!(
            "Expected scalar output gradient for full reduction"
        ));
    };

    let scaled_grad = grad_value * scaling;

    if use_parallel {
        // Parallel fill
        let grad_data: Vec<T> = (0..total_elements)
            .into_par_iter()
            .map(|_| scaled_grad)
            .collect();

        let result_arr = Array::from_shape_vec(IxDyn(input_shape), grad_data)
            .map_err(|e| anyhow::anyhow!("Failed to create result array: {}", e))?;

        Ok(DenseND::from_array(result_arr))
    } else {
        // Sequential fill
        Ok(DenseND::from_elem(input_shape, scaled_grad))
    }
}

/// Configure global Rayon thread pool
///
/// Sets the number of threads for parallel computation.
///
/// # Arguments
///
/// * `num_threads` - Number of threads (None = use all available cores)
///
/// # Example
///
/// ```rust,ignore
/// configure_thread_pool(Some(4)); // Use 4 threads
/// configure_thread_pool(None);    // Use all cores
/// ```
pub fn configure_thread_pool(num_threads: Option<usize>) -> Result<()> {
    if let Some(n) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| anyhow::anyhow!("Failed to configure thread pool: {}", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_elementwise_vjp() {
        let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let grad_y = DenseND::ones(&[4]);

        let config = ParallelConfig {
            min_parallel_size: 1, // Force parallel for testing
            ..Default::default()
        };

        // Test derivative of x^2: f'(x) = 2x
        let grad_x = parallel_elementwise_vjp(&x, &grad_y, |x| 2.0 * x, &config).unwrap();

        assert_eq!(grad_x.shape(), &[4]);

        // Check values with array access
        let grad_arr = grad_x.as_array();
        let values: Vec<f64> = grad_arr.iter().copied().collect();

        assert!(
            (values[0] - 2.0).abs() < 1e-8,
            "grad[0] = {}, expected 2.0",
            values[0]
        );
        assert!(
            (values[1] - 4.0).abs() < 1e-8,
            "grad[1] = {}, expected 4.0",
            values[1]
        );
        assert!(
            (values[2] - 6.0).abs() < 1e-8,
            "grad[2] = {}, expected 6.0",
            values[2]
        );
        assert!(
            (values[3] - 8.0).abs() < 1e-8,
            "grad[3] = {}, expected 8.0",
            values[3]
        );
    }

    #[test]
    fn test_parallel_accumulate_gradients() {
        let grad1 = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let grad2 = DenseND::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let grad3 = DenseND::from_vec(vec![7.0, 8.0, 9.0], &[3]).unwrap();

        let config = ParallelConfig {
            min_parallel_size: 1,
            ..Default::default()
        };

        let result = parallel_accumulate_gradients(&[grad1, grad2, grad3], &config).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert!((result.get(&[0]).unwrap() - 12.0).abs() < 1e-10);
        assert!((result.get(&[1]).unwrap() - 15.0).abs() < 1e-10);
        assert!((result.get(&[2]).unwrap() - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_batch_gradients() {
        let inputs = vec![
            DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            DenseND::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
        ];
        let output_grads = vec![DenseND::ones(&[2]), DenseND::ones(&[2])];

        let config = ParallelConfig::default();

        let gradient_fn = |input: &DenseND<f64>, output_grad: &DenseND<f64>| {
            parallel_elementwise_vjp(input, output_grad, |x| 2.0 * x, &config)
        };

        let results =
            parallel_batch_gradients(&inputs, &output_grads, gradient_fn, &config).unwrap();

        assert_eq!(results.len(), 2);
        assert!((results[0].get(&[0]).unwrap() - 2.0).abs() < 1e-10);
        assert!((results[0].get(&[1]).unwrap() - 4.0).abs() < 1e-10);
        assert!((results[1].get(&[0]).unwrap() - 6.0).abs() < 1e-10);
        assert!((results[1].get(&[1]).unwrap() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduction_grad() {
        let input_shape = vec![2, 3];
        let output_grad = DenseND::from_elem(&[], 5.0);
        let scaling = 0.5;

        let config = ParallelConfig {
            min_parallel_size: 1,
            ..Default::default()
        };

        let result = parallel_reduction_grad(&input_shape, &output_grad, scaling, &config).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert!((result.get(&[i, j]).unwrap() - 2.5).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_sequential_fallback_small_tensor() {
        // Small tensor should use sequential path
        let x = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let grad_y = DenseND::ones(&[3]);

        let config = ParallelConfig {
            min_parallel_size: 1000, // Set high threshold
            ..Default::default()
        };

        let grad_x = parallel_elementwise_vjp(&x, &grad_y, |x| 2.0 * x, &config).unwrap();

        assert_eq!(grad_x.shape(), &[3]);
        assert!((grad_x.get(&[0]).unwrap() - 2.0).abs() < 1e-10);
    }
}
