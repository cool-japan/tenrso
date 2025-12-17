//! Advanced optimization algorithms for gradient-based training.
//!
//! This module provides production-ready optimizers that work seamlessly with
//! the tenrso-ad gradient computation system.
//!
//! # Optimizers
//!
//! - **SGD**: Stochastic Gradient Descent with optional momentum and weight decay
//! - **Adam**: Adaptive Moment Estimation
//! - **AdamW**: Adam with decoupled weight decay
//! - **RAdam**: Rectified Adam with variance rectification
//! - **RMSprop**: Root Mean Square Propagation
//! - **AdaGrad**: Adaptive Gradient Algorithm
//!
//! # Learning Rate Schedulers
//!
//! - **StepLR**: Step-wise learning rate decay
//! - **ExponentialLR**: Exponential learning rate decay
//! - **CosineAnnealingLR**: Cosine annealing schedule
//! - **ReduceLROnPlateau**: Reduce LR when metric plateaus
//! - **WarmupLRScheduler**: Linear warmup wrapper for any scheduler
//!
//! # Training Utilities
//!
//! - **GradientAccumulator**: Accumulate gradients over multiple micro-batches
//!
//! # Example
//!
//! ```rust,ignore
//! use tenrso_ad::optimizers::{Adam, OptimizerConfig};
//! use scirs2_core::ndarray_ext::Array1;
//!
//! // Create optimizer
//! let config = OptimizerConfig::adam()
//!     .learning_rate(0.001)
//!     .beta1(0.9)
//!     .beta2(0.999)
//!     .epsilon(1e-8);
//!
//! let mut optimizer = Adam::new(config);
//!
//! // Training loop
//! for epoch in 0..100 {
//!     let gradients = compute_gradients(&params);
//!     optimizer.step(&params, &gradients)?;
//! }
//! ```

use anyhow::{Context, Result};
use scirs2_core::ndarray_ext::ArrayD;
use scirs2_core::numeric::Float;

/// Configuration for optimizers
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum coefficient (for SGD)
    pub momentum: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Beta1 for Adam-family optimizers
    pub beta1: f64,
    /// Beta2 for Adam-family optimizers
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Nesterov momentum (for SGD)
    pub nesterov: bool,
    /// Dampening for momentum (for SGD)
    pub dampening: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::sgd()
    }
}

impl OptimizerConfig {
    /// Create SGD configuration
    pub fn sgd() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            nesterov: false,
            dampening: 0.0,
        }
    }

    /// Create Adam configuration
    pub fn adam() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            nesterov: false,
            dampening: 0.0,
        }
    }

    /// Create AdamW configuration
    pub fn adamw() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.0,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            nesterov: false,
            dampening: 0.0,
        }
    }

    /// Create RMSprop configuration
    pub fn rmsprop() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-8,
            nesterov: false,
            dampening: 0.0,
        }
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set beta1
    pub fn beta1(mut self, b1: f64) -> Self {
        self.beta1 = b1;
        self
    }

    /// Set beta2
    pub fn beta2(mut self, b2: f64) -> Self {
        self.beta2 = b2;
        self
    }

    /// Set epsilon
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Enable Nesterov momentum
    pub fn nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }

    /// Set dampening
    pub fn dampening(mut self, d: f64) -> Self {
        self.dampening = d;
        self
    }
}

/// Optimizer trait for parameter updates
pub trait Optimizer<T: Float> {
    /// Perform single optimization step
    fn step(&mut self, params: &mut ArrayD<T>, gradients: &ArrayD<T>) -> Result<()>;

    /// Get current learning rate
    fn get_lr(&self) -> f64;

    /// Set learning rate
    fn set_lr(&mut self, lr: f64);

    /// Reset optimizer state
    fn reset(&mut self);

    /// Get number of steps performed
    fn num_steps(&self) -> usize;
}

/// Stochastic Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct SGD<T: Float> {
    config: OptimizerConfig,
    velocity: Option<ArrayD<T>>,
    step_count: usize,
}

impl<T: Float + 'static> SGD<T> {
    /// Create new SGD optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            velocity: None,
            step_count: 0,
        }
    }
}

impl<T: Float + 'static> Optimizer<T> for SGD<T> {
    fn step(&mut self, params: &mut ArrayD<T>, gradients: &ArrayD<T>) -> Result<()> {
        let lr = T::from(self.config.learning_rate).context("Failed to convert learning rate")?;
        let momentum = T::from(self.config.momentum).context("Failed to convert momentum")?;
        let weight_decay =
            T::from(self.config.weight_decay).context("Failed to convert weight decay")?;
        let dampening = T::from(self.config.dampening).context("Failed to convert dampening")?;

        let mut grad = gradients.clone();

        // Add weight decay
        if self.config.weight_decay > 0.0 {
            grad = &grad + &(params.mapv(|x| x * weight_decay));
        }

        // Apply momentum
        if self.config.momentum > 0.0 {
            if let Some(ref mut v) = self.velocity {
                // v = momentum * v + (1 - dampening) * grad
                *v = v.mapv(|x| x * momentum) + grad.mapv(|x| x * (T::one() - dampening));

                if self.config.nesterov {
                    // params = params - lr * (grad + momentum * v)
                    grad = &grad + &v.mapv(|x| x * momentum);
                } else {
                    // params = params - lr * v
                    grad = v.clone();
                }
            } else {
                self.velocity = Some(grad.clone());
                grad = self.velocity.as_ref().unwrap().clone();
            }
        }

        // Update parameters: params = params - lr * grad
        *params = &*params - &grad.mapv(|x| x * lr);

        self.step_count += 1;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.velocity = None;
        self.step_count = 0;
    }

    fn num_steps(&self) -> usize {
        self.step_count
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
#[derive(Debug, Clone)]
pub struct Adam<T: Float> {
    config: OptimizerConfig,
    m: Option<ArrayD<T>>, // First moment estimate
    v: Option<ArrayD<T>>, // Second moment estimate
    step_count: usize,
}

impl<T: Float + 'static> Adam<T> {
    /// Create new Adam optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: None,
            v: None,
            step_count: 0,
        }
    }
}

impl<T: Float + 'static> Optimizer<T> for Adam<T> {
    fn step(&mut self, params: &mut ArrayD<T>, gradients: &ArrayD<T>) -> Result<()> {
        let lr = T::from(self.config.learning_rate).context("Failed to convert learning rate")?;
        let beta1 = T::from(self.config.beta1).context("Failed to convert beta1")?;
        let beta2 = T::from(self.config.beta2).context("Failed to convert beta2")?;
        let epsilon = T::from(self.config.epsilon).context("Failed to convert epsilon")?;
        let weight_decay =
            T::from(self.config.weight_decay).context("Failed to convert weight decay")?;

        let mut grad = gradients.clone();

        // Add weight decay (L2 regularization, not decoupled)
        if self.config.weight_decay > 0.0 {
            grad = &grad + &(params.mapv(|x| x * weight_decay));
        }

        // Initialize moments if needed
        if self.m.is_none() {
            self.m = Some(ArrayD::zeros(grad.raw_dim()));
            self.v = Some(ArrayD::zeros(grad.raw_dim()));
        }

        self.step_count += 1;

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
        *m = m.mapv(|x| x * beta1) + grad.mapv(|x| x * (T::one() - beta1));

        // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
        *v = v.mapv(|x| x * beta2) + grad.mapv(|x| x * x * (T::one() - beta2));

        // Bias correction
        let beta1_t = T::from(self.config.beta1.powi(self.step_count as i32))
            .context("Failed to compute beta1^t")?;
        let beta2_t = T::from(self.config.beta2.powi(self.step_count as i32))
            .context("Failed to compute beta2^t")?;

        let m_hat = m.mapv(|x| x / (T::one() - beta1_t));
        let v_hat = v.mapv(|x| x / (T::one() - beta2_t));

        // Update parameters: params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
        let update = m_hat
            .iter()
            .zip(v_hat.iter())
            .map(|(m, v)| *m / (v.sqrt() + epsilon) * lr)
            .collect::<Vec<_>>();

        let update_array = ArrayD::from_shape_vec(params.raw_dim(), update)
            .context("Failed to create update array")?;

        *params = &*params - &update_array;

        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.step_count = 0;
    }

    fn num_steps(&self) -> usize {
        self.step_count
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
#[derive(Debug, Clone)]
pub struct AdamW<T: Float> {
    config: OptimizerConfig,
    m: Option<ArrayD<T>>,
    v: Option<ArrayD<T>>,
    step_count: usize,
}

impl<T: Float + 'static> AdamW<T> {
    /// Create new AdamW optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: None,
            v: None,
            step_count: 0,
        }
    }
}

impl<T: Float + 'static> Optimizer<T> for AdamW<T> {
    fn step(&mut self, params: &mut ArrayD<T>, gradients: &ArrayD<T>) -> Result<()> {
        let lr = T::from(self.config.learning_rate).context("Failed to convert learning rate")?;
        let beta1 = T::from(self.config.beta1).context("Failed to convert beta1")?;
        let beta2 = T::from(self.config.beta2).context("Failed to convert beta2")?;
        let epsilon = T::from(self.config.epsilon).context("Failed to convert epsilon")?;
        let weight_decay =
            T::from(self.config.weight_decay).context("Failed to convert weight decay")?;

        let grad = gradients.clone();

        // Initialize moments if needed
        if self.m.is_none() {
            self.m = Some(ArrayD::zeros(grad.raw_dim()));
            self.v = Some(ArrayD::zeros(grad.raw_dim()));
        }

        self.step_count += 1;

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update first moment
        *m = m.mapv(|x| x * beta1) + grad.mapv(|x| x * (T::one() - beta1));

        // Update second moment
        *v = v.mapv(|x| x * beta2) + grad.mapv(|x| x * x * (T::one() - beta2));

        // Bias correction
        let beta1_t = T::from(self.config.beta1.powi(self.step_count as i32))
            .context("Failed to compute beta1^t")?;
        let beta2_t = T::from(self.config.beta2.powi(self.step_count as i32))
            .context("Failed to compute beta2^t")?;

        let m_hat = m.mapv(|x| x / (T::one() - beta1_t));
        let v_hat = v.mapv(|x| x / (T::one() - beta2_t));

        // AdamW: Decoupled weight decay
        // params = params - lr * weight_decay * params - lr * m_hat / (sqrt(v_hat) + epsilon)
        let adam_update = m_hat
            .iter()
            .zip(v_hat.iter())
            .map(|(m, v)| *m / (v.sqrt() + epsilon) * lr)
            .collect::<Vec<_>>();

        let adam_array = ArrayD::from_shape_vec(params.raw_dim(), adam_update)
            .context("Failed to create Adam update array")?;

        // Apply decoupled weight decay and Adam update
        if self.config.weight_decay > 0.0 {
            *params = params.mapv(|x| x * (T::one() - lr * weight_decay)) - &adam_array;
        } else {
            *params = &*params - &adam_array;
        }

        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.step_count = 0;
    }

    fn num_steps(&self) -> usize {
        self.step_count
    }
}

/// RMSprop optimizer
#[derive(Debug, Clone)]
pub struct RMSprop<T: Float> {
    config: OptimizerConfig,
    square_avg: Option<ArrayD<T>>,
    step_count: usize,
}

impl<T: Float + 'static> RMSprop<T> {
    /// Create new RMSprop optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            square_avg: None,
            step_count: 0,
        }
    }
}

impl<T: Float + 'static> Optimizer<T> for RMSprop<T> {
    fn step(&mut self, params: &mut ArrayD<T>, gradients: &ArrayD<T>) -> Result<()> {
        let lr = T::from(self.config.learning_rate).context("Failed to convert learning rate")?;
        let alpha = T::from(self.config.beta2).context("Failed to convert alpha")?;
        let epsilon = T::from(self.config.epsilon).context("Failed to convert epsilon")?;

        let grad = gradients.clone();

        // Initialize square average if needed
        if self.square_avg.is_none() {
            self.square_avg = Some(ArrayD::zeros(grad.raw_dim()));
        }

        let sq_avg = self.square_avg.as_mut().unwrap();

        // Update square average: sq_avg = alpha * sq_avg + (1 - alpha) * grad^2
        *sq_avg = sq_avg.mapv(|x| x * alpha) + grad.mapv(|x| x * x * (T::one() - alpha));

        // Update parameters: params = params - lr * grad / (sqrt(sq_avg) + epsilon)
        let update = grad
            .iter()
            .zip(sq_avg.iter())
            .map(|(g, sq)| *g / (sq.sqrt() + epsilon) * lr)
            .collect::<Vec<_>>();

        let update_array = ArrayD::from_shape_vec(params.raw_dim(), update)
            .context("Failed to create update array")?;

        *params = &*params - &update_array;

        self.step_count += 1;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.square_avg = None;
        self.step_count = 0;
    }

    fn num_steps(&self) -> usize {
        self.step_count
    }
}

/// AdaGrad optimizer
#[derive(Debug, Clone)]
pub struct AdaGrad<T: Float> {
    config: OptimizerConfig,
    sum_squares: Option<ArrayD<T>>,
    step_count: usize,
}

impl<T: Float + 'static> AdaGrad<T> {
    /// Create new AdaGrad optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            sum_squares: None,
            step_count: 0,
        }
    }
}

impl<T: Float + 'static> Optimizer<T> for AdaGrad<T> {
    fn step(&mut self, params: &mut ArrayD<T>, gradients: &ArrayD<T>) -> Result<()> {
        let lr = T::from(self.config.learning_rate).context("Failed to convert learning rate")?;
        let epsilon = T::from(self.config.epsilon).context("Failed to convert epsilon")?;

        let grad = gradients.clone();

        // Initialize sum of squares if needed
        if self.sum_squares.is_none() {
            self.sum_squares = Some(ArrayD::zeros(grad.raw_dim()));
        }

        let sum_sq = self.sum_squares.as_mut().unwrap();

        // Accumulate squared gradients: sum_sq = sum_sq + grad^2
        *sum_sq = &*sum_sq + &grad.mapv(|x| x * x);

        // Update parameters: params = params - lr * grad / (sqrt(sum_sq) + epsilon)
        let update = grad
            .iter()
            .zip(sum_sq.iter())
            .map(|(g, sq)| *g / (sq.sqrt() + epsilon) * lr)
            .collect::<Vec<_>>();

        let update_array = ArrayD::from_shape_vec(params.raw_dim(), update)
            .context("Failed to create update array")?;

        *params = &*params - &update_array;

        self.step_count += 1;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.sum_squares = None;
        self.step_count = 0;
    }

    fn num_steps(&self) -> usize {
        self.step_count
    }
}

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get current learning rate
    fn get_lr(&self) -> f64;

    /// Step the scheduler (call after each epoch or batch)
    fn step(&mut self);

    /// Step with metric (for ReduceLROnPlateau)
    fn step_with_metric(&mut self, _metric: f64) {
        self.step();
    }

    /// Reset scheduler
    fn reset(&mut self);
}

/// Step-wise learning rate scheduler
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f64,
    current_lr: f64,
    step_size: usize,
    gamma: f64,
    current_step: usize,
}

impl StepLR {
    /// Create new StepLR scheduler
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step.is_multiple_of(self.step_size) {
            self.current_lr *= self.gamma;
        }
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_step = 0;
    }
}

/// Exponential learning rate scheduler
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    initial_lr: f64,
    current_lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    /// Create new ExponentialLR scheduler
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            gamma,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_lr *= self.gamma;
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
    }
}

/// Cosine annealing learning rate scheduler
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    current_lr: f64,
    t_max: usize,
    current_step: usize,
}

impl CosineAnnealingLR {
    /// Create new CosineAnnealingLR scheduler
    pub fn new(initial_lr: f64, t_max: usize, min_lr: f64) -> Self {
        Self {
            initial_lr,
            min_lr,
            current_lr: initial_lr,
            t_max,
            current_step: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_step += 1;
        let progress = (self.current_step % self.t_max) as f64 / self.t_max as f64;
        let cos_val = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_val;
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_step = 0;
    }
}

/// Reduce LR on plateau scheduler
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    initial_lr: f64,
    current_lr: f64,
    factor: f64,
    patience: usize,
    min_lr: f64,
    best_metric: Option<f64>,
    wait_count: usize,
    mode: PlateauMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    Min,
    Max,
}

impl ReduceLROnPlateau {
    /// Create new ReduceLROnPlateau scheduler
    pub fn new(
        initial_lr: f64,
        factor: f64,
        patience: usize,
        min_lr: f64,
        mode: PlateauMode,
    ) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            factor,
            patience,
            min_lr,
            best_metric: None,
            wait_count: 0,
            mode,
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn step(&mut self) {
        // No-op, use step_with_metric instead
    }

    fn step_with_metric(&mut self, metric: f64) {
        let is_better = match (&self.best_metric, self.mode) {
            (None, _) => true,
            (Some(best), PlateauMode::Min) => metric < *best,
            (Some(best), PlateauMode::Max) => metric > *best,
        };

        if is_better {
            self.best_metric = Some(metric);
            self.wait_count = 0;
        } else {
            self.wait_count += 1;
            if self.wait_count >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.wait_count = 0;
            }
        }
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.best_metric = None;
        self.wait_count = 0;
    }
}

/// Warmup learning rate scheduler that wraps any base scheduler
///
/// Linearly increases the learning rate from 0 to the initial LR over `warmup_steps`,
/// then delegates to the base scheduler.
#[derive(Debug, Clone)]
pub struct WarmupLRScheduler<S: LRScheduler> {
    base_scheduler: S,
    warmup_steps: usize,
    initial_lr: f64,
    current_step: usize,
}

impl<S: LRScheduler> WarmupLRScheduler<S> {
    /// Create new warmup scheduler
    ///
    /// # Arguments
    ///
    /// * `base_scheduler` - The scheduler to wrap
    /// * `warmup_steps` - Number of steps to warm up over
    pub fn new(base_scheduler: S, warmup_steps: usize) -> Self {
        let initial_lr = base_scheduler.get_lr();
        Self {
            base_scheduler,
            warmup_steps,
            initial_lr,
            current_step: 0,
        }
    }
}

impl<S: LRScheduler> LRScheduler for WarmupLRScheduler<S> {
    fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup from 0 to initial_lr
            self.initial_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            self.base_scheduler.get_lr()
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step >= self.warmup_steps {
            self.base_scheduler.step();
        }
    }

    fn step_with_metric(&mut self, metric: f64) {
        self.current_step += 1;
        if self.current_step >= self.warmup_steps {
            self.base_scheduler.step_with_metric(metric);
        }
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.base_scheduler.reset();
    }
}

/// Rectified Adam (RAdam) optimizer
///
/// RAdam automatically rectifies the variance of adaptive learning rate in the early stage
/// of training, achieving more stable and effective training than vanilla Adam.
///
/// Reference: "On the Variance of the Adaptive Learning Rate and Beyond" (Liu et al., 2019)
#[derive(Debug, Clone)]
pub struct RAdam<T: Float> {
    config: OptimizerConfig,
    m: Option<ArrayD<T>>, // First moment
    v: Option<ArrayD<T>>, // Second moment
    steps: usize,
}

impl<T: Float> RAdam<T> {
    /// Create new RAdam optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: None,
            v: None,
            steps: 0,
        }
    }

    /// Compute the rectification term for adaptive learning rate
    fn get_rectification_term(&self) -> Option<f64> {
        let rho_inf = 2.0 / (1.0 - self.config.beta2) - 1.0;
        let rho_t = rho_inf
            - 2.0 * (self.steps as f64) * self.config.beta2.powi(self.steps as i32)
                / (1.0 - self.config.beta2.powi(self.steps as i32));

        if rho_t > 5.0 {
            let r_t = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                .sqrt();
            Some(r_t)
        } else {
            None // Use SGD-like update (no adaptive learning rate correction)
        }
    }
}

impl<T: Float> Optimizer<T> for RAdam<T> {
    fn step(&mut self, params: &mut ArrayD<T>, gradients: &ArrayD<T>) -> Result<()> {
        if params.shape() != gradients.shape() {
            return Err(anyhow::anyhow!(
                "Parameter and gradient shapes must match: {:?} vs {:?}",
                params.shape(),
                gradients.shape()
            ));
        }

        self.steps += 1;

        // Initialize moments if needed
        if self.m.is_none() {
            self.m = Some(ArrayD::zeros(params.raw_dim()));
            self.v = Some(ArrayD::zeros(params.raw_dim()));
        }

        // Compute rectification term before borrowing
        let rect_term = self.get_rectification_term();

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first and second moments
        m.zip_mut_with(gradients, |m_val, &g_val| {
            let beta1 = T::from(self.config.beta1).unwrap();
            *m_val = *m_val * beta1 + g_val * (T::one() - beta1);
        });

        v.zip_mut_with(gradients, |v_val, &g_val| {
            let beta2 = T::from(self.config.beta2).unwrap();
            *v_val = *v_val * beta2 + g_val * g_val * (T::one() - beta2);
        });

        // Bias correction
        let bias_correction1 = 1.0 - self.config.beta1.powi(self.steps as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.steps as i32);
        let m_hat_scale = T::from(1.0 / bias_correction1).unwrap();

        // Check if we should use rectified adaptive learning rate
        if let Some(rect_term) = rect_term {
            // Rectified adaptive learning rate
            let lr = T::from(self.config.learning_rate * rect_term).unwrap();
            let v_hat_scale = T::from(1.0 / bias_correction2.sqrt()).unwrap();
            let eps = T::from(self.config.epsilon).unwrap();

            // Proper element-wise update: params = params - lr * m_hat / (sqrt(v_hat) + eps)
            for i in 0..params.len() {
                let m_val = m.as_slice().unwrap()[i];
                let v_val = v.as_slice().unwrap()[i];
                let p_val = &mut params.as_slice_mut().unwrap()[i];

                let m_hat = m_val * m_hat_scale;
                let v_hat = v_val * v_hat_scale;
                let update = m_hat / (v_hat.sqrt() + eps);
                *p_val = *p_val - lr * update;
            }
        } else {
            // Use SGD-like update (momentum only, no adaptive LR)
            let lr = T::from(self.config.learning_rate).unwrap();
            params.zip_mut_with(&*m, |p_val, &m_val| {
                *p_val = *p_val - lr * (m_val * m_hat_scale);
            });
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.steps = 0;
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn num_steps(&self) -> usize {
        self.steps
    }
}

/// Gradient accumulator for micro-batching
///
/// Accumulates gradients over multiple micro-batches before applying an optimizer step.
/// This is useful for:
/// - Training with large effective batch sizes on limited memory
/// - Simulating larger batches for better gradient estimates
/// - Reducing optimizer step frequency for efficiency
pub struct GradientAccumulator<T: Float> {
    accumulated_grads: Option<ArrayD<T>>,
    accumulation_steps: usize,
    current_step: usize,
}

impl<T: Float> GradientAccumulator<T> {
    /// Create new gradient accumulator
    ///
    /// # Arguments
    ///
    /// * `accumulation_steps` - Number of micro-batches to accumulate before optimizer step
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulated_grads: None,
            accumulation_steps,
            current_step: 0,
        }
    }

    /// Accumulate gradients from a micro-batch
    ///
    /// Returns `Some(averaged_grads)` when ready to perform optimizer step, `None` otherwise
    pub fn accumulate(&mut self, gradients: &ArrayD<T>) -> Option<ArrayD<T>> {
        if self.accumulated_grads.is_none() {
            self.accumulated_grads = Some(ArrayD::zeros(gradients.raw_dim()));
        }

        let acc = self.accumulated_grads.as_mut().unwrap();
        acc.zip_mut_with(gradients, |a, &g| *a = *a + g);

        self.current_step += 1;

        if self.current_step >= self.accumulation_steps {
            // Average the accumulated gradients
            let scale = T::from(1.0 / self.accumulation_steps as f64).unwrap();
            let averaged = acc.mapv(|v| v * scale);

            // Reset accumulator
            self.reset();

            Some(averaged)
        } else {
            None
        }
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.accumulated_grads = None;
        self.current_step = 0;
    }

    /// Get current accumulation step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Check if ready for optimizer step
    pub fn is_ready(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array1;

    #[test]
    fn test_sgd_basic() {
        let config = OptimizerConfig::sgd().learning_rate(0.1);
        let mut optimizer = SGD::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();

        // params = params - lr * grads
        // [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
        assert!((params[[0]] - 0.99).abs() < 1e-10);
        assert!((params[[1]] - 1.98).abs() < 1e-10);
        assert!((params[[2]] - 2.97).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_momentum() {
        let config = OptimizerConfig::sgd().learning_rate(0.1).momentum(0.9);
        let mut optimizer = SGD::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.num_steps(), 1);
    }

    #[test]
    fn test_adam_initialization() {
        let config = OptimizerConfig::adam();
        let optimizer = Adam::<f64>::new(config);
        assert_eq!(optimizer.num_steps(), 0);
        assert_eq!(optimizer.get_lr(), 0.001);
    }

    #[test]
    fn test_adam_step() {
        let config = OptimizerConfig::adam().learning_rate(0.01);
        let mut optimizer = Adam::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.num_steps(), 1);

        // Verify parameters changed
        assert!((params[[0]] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_adamw_decoupled_weight_decay() {
        let config = OptimizerConfig::adamw()
            .learning_rate(0.01)
            .weight_decay(0.01);
        let mut optimizer = AdamW::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.num_steps(), 1);
    }

    #[test]
    fn test_rmsprop_basic() {
        let config = OptimizerConfig::rmsprop().learning_rate(0.01);
        let mut optimizer = RMSprop::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.num_steps(), 1);
    }

    #[test]
    fn test_adagrad_accumulation() {
        let config = OptimizerConfig::sgd().learning_rate(0.1);
        let mut optimizer = AdaGrad::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.num_steps(), 2);
    }

    #[test]
    fn test_step_lr_scheduler() {
        let mut scheduler = StepLR::new(0.1, 10, 0.5);
        assert_eq!(scheduler.get_lr(), 0.1);

        for _ in 0..9 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step(); // Step 10
        assert_eq!(scheduler.get_lr(), 0.05);
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        let mut scheduler = ExponentialLR::new(0.1, 0.9);
        assert_eq!(scheduler.get_lr(), 0.1);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.09).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
        assert_eq!(scheduler.get_lr(), 0.1);

        for _ in 0..50 {
            scheduler.step();
        }
        // At halfway point (50 steps out of 100), LR should be near minimum
        // With cosine schedule: lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * t/T)) / 2
        // At t=50, T=100: (1 + cos(pi * 0.5)) / 2 = (1 + 0) / 2 = 0.5
        // So LR should be around 0.05
        assert!(scheduler.get_lr() < 0.06);
        assert!(scheduler.get_lr() > 0.04);
    }

    #[test]
    fn test_reduce_lr_on_plateau() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, 0.5, 3, 0.001, PlateauMode::Min);
        assert_eq!(scheduler.get_lr(), 0.1);

        // Metric improving
        scheduler.step_with_metric(1.0);
        scheduler.step_with_metric(0.9);
        assert_eq!(scheduler.get_lr(), 0.1);

        // Metric plateauing
        scheduler.step_with_metric(0.9);
        scheduler.step_with_metric(0.9);
        scheduler.step_with_metric(0.9);
        // After patience steps, LR should reduce
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_optimizer_reset() {
        let config = OptimizerConfig::adam();
        let mut optimizer = Adam::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.num_steps(), 1);

        optimizer.reset();
        assert_eq!(optimizer.num_steps(), 0);
    }

    #[test]
    fn test_lr_update() {
        let config = OptimizerConfig::sgd().learning_rate(0.1);
        let mut optimizer = SGD::<f64>::new(config);

        assert_eq!(optimizer.get_lr(), 0.1);
        optimizer.set_lr(0.01);
        assert_eq!(optimizer.get_lr(), 0.01);
    }

    #[test]
    fn test_warmup_scheduler() {
        let base = StepLR::new(0.1, 100, 0.5);
        let mut scheduler = WarmupLRScheduler::new(base, 10);

        // At step 0, LR should be 0
        assert_eq!(scheduler.get_lr(), 0.0);

        // At step 5 (half warmup), LR should be 0.05
        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-10);

        // After warmup (step 10+), should use base scheduler
        for _ in 0..5 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_radam_initialization() {
        let config = OptimizerConfig::adam();
        let optimizer = RAdam::<f64>::new(config);
        assert_eq!(optimizer.num_steps(), 0);
        assert_eq!(optimizer.get_lr(), 0.001);
    }

    #[test]
    fn test_radam_step() {
        let config = OptimizerConfig::adam().learning_rate(0.01);
        let mut optimizer = RAdam::<f64>::new(config);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]).into_dyn();

        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.num_steps(), 1);

        // Verify parameters changed
        assert!((params[[0]] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_radam_rectification() {
        let config = OptimizerConfig::adam();
        let optimizer = RAdam::<f64>::new(config);

        // Early steps should not have rectification
        assert!(optimizer.get_rectification_term().is_none());
    }

    #[test]
    fn test_gradient_accumulator_basic() {
        let mut accumulator = GradientAccumulator::<f64>::new(4);

        let grad1 = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grad2 = Array1::from_vec(vec![2.0, 4.0, 6.0]).into_dyn();
        let grad3 = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let grad4 = Array1::from_vec(vec![0.0, 0.0, 0.0]).into_dyn();

        assert!(accumulator.accumulate(&grad1).is_none());
        assert_eq!(accumulator.current_step(), 1);
        assert!(!accumulator.is_ready());

        assert!(accumulator.accumulate(&grad2).is_none());
        assert_eq!(accumulator.current_step(), 2);

        assert!(accumulator.accumulate(&grad3).is_none());
        assert_eq!(accumulator.current_step(), 3);

        // Fourth accumulation should return averaged gradients
        let result = accumulator.accumulate(&grad4);
        assert!(result.is_some());
        let averaged = result.unwrap();

        // Average should be (1+2+1+0)/4 = 1.0, (2+4+2+0)/4 = 2.0, (3+6+3+0)/4 = 3.0
        assert!((averaged[[0]] - 1.0).abs() < 1e-10);
        assert!((averaged[[1]] - 2.0).abs() < 1e-10);
        assert!((averaged[[2]] - 3.0).abs() < 1e-10);

        // Accumulator should be reset
        assert_eq!(accumulator.current_step(), 0);
    }

    #[test]
    fn test_gradient_accumulator_reset() {
        let mut accumulator = GradientAccumulator::<f64>::new(2);

        let grad = Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        accumulator.accumulate(&grad);
        assert_eq!(accumulator.current_step(), 1);

        accumulator.reset();
        assert_eq!(accumulator.current_step(), 0);
        assert!(!accumulator.is_ready());
    }
}
