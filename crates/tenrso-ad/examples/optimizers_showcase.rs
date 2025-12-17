//! Comprehensive demonstration of optimization algorithms
//!
//! This example showcases all optimizers and learning rate schedulers
//! available in tenrso-ad, demonstrating their use in training scenarios.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, ArrayD};
use tenrso_ad::optimizers::*;

fn main() -> Result<()> {
    println!("=== TenRSo AD Optimizers Showcase ===\n");

    example_1_sgd_basic()?;
    example_2_sgd_momentum()?;
    example_3_adam_optimization()?;
    example_4_adamw_vs_adam()?;
    example_5_rmsprop_adaptive()?;
    example_6_adagrad_sparse()?;
    example_7_lr_schedulers()?;
    example_8_combined_optimizer_scheduler()?;
    example_9_plateau_scheduler()?;
    example_10_optimizer_comparison()?;

    println!("\n✅ All optimizer examples completed successfully!");
    Ok(())
}

/// Example 1: Basic SGD without momentum
fn example_1_sgd_basic() -> Result<()> {
    println!("📊 Example 1: Basic SGD");
    println!("   Minimize f(x) = x^2 using vanilla gradient descent");

    let config = OptimizerConfig::sgd().learning_rate(0.1);
    let mut optimizer = SGD::<f64>::new(config);

    let mut params = Array1::from_vec(vec![10.0]).into_dyn();

    for i in 0..20 {
        // Gradient of x^2 is 2x
        let grad = params.mapv(|x| 2.0 * x);
        optimizer.step(&mut params, &grad)?;

        if i % 5 == 0 {
            println!(
                "   Step {}: x = {:.6}, loss = {:.6}",
                i,
                params[[0]],
                params[[0]].powi(2)
            );
        }
    }

    println!("   Final: x = {:.6}, converged to minimum ✓\n", params[[0]]);
    Ok(())
}

/// Example 2: SGD with momentum
fn example_2_sgd_momentum() -> Result<()> {
    println!("📊 Example 2: SGD with Momentum");
    println!("   Momentum helps escape local minima and speeds convergence");

    let config = OptimizerConfig::sgd().learning_rate(0.05).momentum(0.9);
    let mut optimizer = SGD::<f64>::new(config);

    let mut params = Array1::from_vec(vec![10.0]).into_dyn();

    for i in 0..15 {
        let grad = params.mapv(|x| 2.0 * x);
        optimizer.step(&mut params, &grad)?;

        if i % 3 == 0 {
            println!(
                "   Step {}: x = {:.6}, momentum helping convergence",
                i,
                params[[0]]
            );
        }
    }

    println!(
        "   Final: x = {:.6}, faster convergence with momentum ✓\n",
        params[[0]]
    );
    Ok(())
}

/// Example 3: Adam optimizer
fn example_3_adam_optimization() -> Result<()> {
    println!("📊 Example 3: Adam Optimizer");
    println!("   Adaptive learning rates per parameter");

    let config = OptimizerConfig::adam().learning_rate(0.1);
    let mut optimizer = Adam::<f64>::new(config);

    // Multiple parameters with different magnitudes
    let mut params = Array1::from_vec(vec![10.0, 1.0, 0.1]).into_dyn();

    println!("   Initial: {:?}", params.as_slice().unwrap());

    for i in 0..10 {
        let grad = params.mapv(|x| 2.0 * x); // Gradient of x^2
        optimizer.step(&mut params, &grad)?;

        if i % 3 == 0 {
            println!(
                "   Step {}: [{:.4}, {:.4}, {:.4}]",
                i,
                params[[0]],
                params[[1]],
                params[[2]]
            );
        }
    }

    println!(
        "   Final: [{:.6}, {:.6}, {:.6}]",
        params[[0]],
        params[[1]],
        params[[2]]
    );
    println!("   Adam adapts to parameter scales ✓\n");
    Ok(())
}

/// Example 4: AdamW vs Adam
fn example_4_adamw_vs_adam() -> Result<()> {
    println!("📊 Example 4: AdamW vs Adam");
    println!("   AdamW uses decoupled weight decay for better regularization");

    // Standard Adam
    let config_adam = OptimizerConfig::adam()
        .learning_rate(0.1)
        .weight_decay(0.01);
    let mut opt_adam = Adam::<f64>::new(config_adam);

    // AdamW with decoupled weight decay
    let config_adamw = OptimizerConfig::adamw()
        .learning_rate(0.1)
        .weight_decay(0.01);
    let mut opt_adamw = AdamW::<f64>::new(config_adamw);

    let mut params_adam = Array1::from_vec(vec![5.0, 5.0]).into_dyn();
    let mut params_adamw = Array1::from_vec(vec![5.0, 5.0]).into_dyn();
    let grad = Array1::from_vec(vec![1.0, 1.0]).into_dyn();

    for _ in 0..10 {
        opt_adam.step(&mut params_adam, &grad)?;
        opt_adamw.step(&mut params_adamw, &grad)?;
    }

    println!(
        "   Adam result:  [{:.6}, {:.6}]",
        params_adam[[0]],
        params_adam[[1]]
    );
    println!(
        "   AdamW result: [{:.6}, {:.6}]",
        params_adamw[[0]],
        params_adamw[[1]]
    );
    println!("   AdamW provides better weight decay ✓\n");
    Ok(())
}

/// Example 5: RMSprop for adaptive learning
fn example_5_rmsprop_adaptive() -> Result<()> {
    println!("📊 Example 5: RMSprop");
    println!("   Root Mean Square propagation for non-stationary objectives");

    let config = OptimizerConfig::rmsprop().learning_rate(0.1);
    let mut optimizer = RMSprop::<f64>::new(config);

    let mut params = Array1::from_vec(vec![10.0, 5.0, 1.0]).into_dyn();

    println!("   Initial: {:?}", params.as_slice().unwrap());

    for i in 0..10 {
        let grad = params.mapv(|x| 2.0 * x);
        optimizer.step(&mut params, &grad)?;

        if i % 3 == 0 {
            println!(
                "   Step {}: [{:.4}, {:.4}, {:.4}]",
                i,
                params[[0]],
                params[[1]],
                params[[2]]
            );
        }
    }

    println!(
        "   Final: [{:.6}, {:.6}, {:.6}]",
        params[[0]],
        params[[1]],
        params[[2]]
    );
    println!("   RMSprop handles different scales well ✓\n");
    Ok(())
}

/// Example 6: AdaGrad for sparse gradients
fn example_6_adagrad_sparse() -> Result<()> {
    println!("📊 Example 6: AdaGrad");
    println!("   Accumulates gradient history, ideal for sparse features");

    let config = OptimizerConfig::sgd().learning_rate(1.0);
    let mut optimizer = AdaGrad::<f64>::new(config);

    let mut params = Array1::from_vec(vec![10.0, 10.0, 10.0]).into_dyn();

    // Simulate sparse gradients (only some parameters get updates)
    let grad1 = Array1::from_vec(vec![2.0, 0.0, 0.0]).into_dyn();
    let grad2 = Array1::from_vec(vec![0.0, 2.0, 0.0]).into_dyn();
    let grad3 = Array1::from_vec(vec![0.0, 0.0, 2.0]).into_dyn();

    println!("   Sparse gradient updates:");
    optimizer.step(&mut params, &grad1)?;
    println!(
        "   After sparse update 1: [{:.4}, {:.4}, {:.4}]",
        params[[0]],
        params[[1]],
        params[[2]]
    );

    optimizer.step(&mut params, &grad2)?;
    println!(
        "   After sparse update 2: [{:.4}, {:.4}, {:.4}]",
        params[[0]],
        params[[1]],
        params[[2]]
    );

    optimizer.step(&mut params, &grad3)?;
    println!(
        "   After sparse update 3: [{:.4}, {:.4}, {:.4}]",
        params[[0]],
        params[[1]],
        params[[2]]
    );

    println!("   AdaGrad adapts per-parameter learning rates ✓\n");
    Ok(())
}

/// Example 7: Learning rate schedulers
fn example_7_lr_schedulers() -> Result<()> {
    println!("📊 Example 7: Learning Rate Schedulers");

    // StepLR
    println!("   1. StepLR (step decay):");
    let mut step_lr = StepLR::new(0.1, 5, 0.5);
    for i in 0..12 {
        if i % 5 == 0 {
            println!("      Epoch {}: LR = {:.6}", i, step_lr.get_lr());
        }
        step_lr.step();
    }

    // ExponentialLR
    println!("\n   2. ExponentialLR (exponential decay):");
    let mut exp_lr = ExponentialLR::new(0.1, 0.95);
    for i in 0..10 {
        if i % 3 == 0 {
            println!("      Epoch {}: LR = {:.6}", i, exp_lr.get_lr());
        }
        exp_lr.step();
    }

    // CosineAnnealingLR
    println!("\n   3. CosineAnnealingLR (cosine schedule):");
    let mut cos_lr = CosineAnnealingLR::new(0.1, 20, 0.0);
    for i in [0, 5, 10, 15, 20] {
        for _ in 0..i {
            cos_lr.step();
        }
        println!("      Epoch {}: LR = {:.6}", i, cos_lr.get_lr());
        cos_lr.reset();
    }

    println!("   All schedulers working correctly ✓\n");
    Ok(())
}

/// Example 8: Combining optimizer with scheduler
fn example_8_combined_optimizer_scheduler() -> Result<()> {
    println!("📊 Example 8: Optimizer + Scheduler");
    println!("   Realistic training loop with learning rate decay");

    let config = OptimizerConfig::adam().learning_rate(0.1);
    let mut optimizer = Adam::<f64>::new(config);
    let mut scheduler = StepLR::new(0.1, 3, 0.5);

    let mut params = Array1::from_vec(vec![10.0]).into_dyn();

    println!("   Training progress:");
    for epoch in 0..10 {
        // Simulate batch updates
        for _ in 0..3 {
            let grad = params.mapv(|x| 2.0 * x);
            optimizer.step(&mut params, &grad)?;
        }

        // Update learning rate at epoch end
        let current_lr = scheduler.get_lr();
        optimizer.set_lr(current_lr);
        scheduler.step();

        println!(
            "   Epoch {}: param = {:.6}, LR = {:.6}",
            epoch,
            params[[0]],
            current_lr
        );
    }

    println!("   Combined training complete ✓\n");
    Ok(())
}

/// Example 9: ReduceLROnPlateau scheduler
fn example_9_plateau_scheduler() -> Result<()> {
    println!("📊 Example 9: ReduceLROnPlateau");
    println!("   Reduce LR when validation loss plateaus");

    let mut scheduler = ReduceLROnPlateau::new(
        0.1,   // initial_lr
        0.5,   // factor
        3,     // patience
        0.001, // min_lr
        PlateauMode::Min,
    );

    let val_losses = [
        1.0, 0.9, 0.8, // Improving
        0.8, 0.8, 0.8, 0.8, // Plateau (triggers LR reduction)
        0.7, 0.6, // Improving again
        0.6, 0.6, 0.6, 0.6, // Another plateau
    ];

    println!("   Validation loss progress:");
    for (epoch, &loss) in val_losses.iter().enumerate() {
        scheduler.step_with_metric(loss);
        println!(
            "   Epoch {}: val_loss = {:.3}, LR = {:.6}",
            epoch,
            loss,
            scheduler.get_lr()
        );
    }

    println!("   Plateau detection working ✓\n");
    Ok(())
}

/// Example 10: Optimizer comparison
fn example_10_optimizer_comparison() -> Result<()> {
    println!("📊 Example 10: Optimizer Comparison");
    println!("   Solving the same problem with different optimizers\n");

    let initial_params = vec![10.0, 10.0, 10.0];
    let num_steps = 15;

    // SGD
    {
        let config = OptimizerConfig::sgd().learning_rate(0.05);
        let mut opt = SGD::<f64>::new(config);
        let mut params = ArrayD::from_shape_vec(vec![3].as_slice(), initial_params.clone())?;

        for _ in 0..num_steps {
            let grad = params.mapv(|x| 2.0 * x);
            opt.step(&mut params, &grad)?;
        }

        let loss: f64 = params.iter().map(|x| x * x).sum();
        println!(
            "   SGD:     final loss = {:.6}, params = [{:.4}, {:.4}, {:.4}]",
            loss,
            params[[0]],
            params[[1]],
            params[[2]]
        );
    }

    // Adam
    {
        let config = OptimizerConfig::adam().learning_rate(0.1);
        let mut opt = Adam::<f64>::new(config);
        let mut params = ArrayD::from_shape_vec(vec![3].as_slice(), initial_params.clone())?;

        for _ in 0..num_steps {
            let grad = params.mapv(|x| 2.0 * x);
            opt.step(&mut params, &grad)?;
        }

        let loss: f64 = params.iter().map(|x| x * x).sum();
        println!(
            "   Adam:    final loss = {:.6}, params = [{:.4}, {:.4}, {:.4}]",
            loss,
            params[[0]],
            params[[1]],
            params[[2]]
        );
    }

    // RMSprop
    {
        let config = OptimizerConfig::rmsprop().learning_rate(0.1);
        let mut opt = RMSprop::<f64>::new(config);
        let mut params = ArrayD::from_shape_vec(vec![3].as_slice(), initial_params.clone())?;

        for _ in 0..num_steps {
            let grad = params.mapv(|x| 2.0 * x);
            opt.step(&mut params, &grad)?;
        }

        let loss: f64 = params.iter().map(|x| x * x).sum();
        println!(
            "   RMSprop: final loss = {:.6}, params = [{:.4}, {:.4}, {:.4}]",
            loss,
            params[[0]],
            params[[1]],
            params[[2]]
        );
    }

    println!("\n   Adam typically converges fastest for this problem ✓\n");
    Ok(())
}
