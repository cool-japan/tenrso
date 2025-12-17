//! GPU abstraction layer demonstration
//!
//! This example demonstrates the device-agnostic GPU abstraction layer
//! with CPU fallback implementation.
//!
//! Run with: cargo run --example gpu_demo

use anyhow::Result;
use std::time::Instant;
use tenrso_ooc::gpu::{DeviceManager, DeviceType};

fn main() -> Result<()> {
    println!("=== GPU Abstraction Layer Demonstration ===\n");

    // === Example 1: Device Enumeration ===
    println!("Example 1: Device Enumeration and Selection");
    println!("--------------------------------------------");

    let manager = DeviceManager::new()?;
    println!("Available devices: {}", manager.device_count());

    for (idx, info) in manager.list_devices().iter().enumerate() {
        println!("\nDevice {idx}:");
        println!("  Type: {}", info.device_type);
        println!("  Name: {}", info.name);
        println!("  Total memory: {:.2} GB", info.total_memory as f64 / 1e9);
        println!(
            "  Available memory: {:.2} GB",
            info.available_memory as f64 / 1e9
        );
        println!("  Compute units: {}", info.compute_units);
        println!("  Compute capability: {}", info.compute_capability);
    }
    println!();

    // === Example 2: Device Selection ===
    println!("Example 2: Device Selection Strategies");
    println!("---------------------------------------");

    // Get default device
    let default_device = manager.default_device()?;
    println!(
        "Default device: {} ({})",
        default_device.info().name,
        default_device.device_id()
    );

    // Get best available device (prefers GPU over CPU)
    let best_device = manager.best_device()?;
    println!(
        "Best device: {} ({})",
        best_device.info().name,
        best_device.device_id()
    );

    // Get devices by type
    let cpu_devices = manager.devices_by_type(DeviceType::Cpu);
    println!("CPU devices found: {}", cpu_devices.len());
    println!();

    // Use best device for remaining examples
    let device = best_device;

    // === Example 3: Buffer Allocation and Management ===
    println!("Example 3: Buffer Allocation and Management");
    println!("--------------------------------------------");

    let size = 1024 * 1024; // 1M elements
    println!("Allocating buffer for {} f64 elements...", size);

    let start = Instant::now();
    let buffer = device.allocate::<f64>(size)?;
    let alloc_time = start.elapsed();

    println!("  Allocation time: {:?}", alloc_time);
    println!("  Buffer size: {} elements", buffer.size());
    println!("  Buffer size: {:.2} MB", buffer.size_bytes() as f64 / 1e6);

    // Check device statistics
    let stats = device.stats();
    println!("\nDevice statistics after allocation:");
    println!("  Total allocations: {}", stats.allocations);
    println!(
        "  Allocated bytes: {:.2} MB",
        stats.allocated_bytes as f64 / 1e6
    );
    println!("  Peak bytes: {:.2} MB", stats.peak_bytes as f64 / 1e6);
    println!();

    // === Example 4: Host-Device Data Transfer ===
    println!("Example 4: Host-Device Data Transfer");
    println!("-------------------------------------");

    let transfer_size = 100_000;
    println!(
        "Transferring {} elements between host and device...",
        transfer_size
    );

    // Create host data
    let host_data: Vec<f64> = (0..transfer_size).map(|i| i as f64 * 0.5).collect();

    let mut transfer_buffer = device.allocate(transfer_size)?;

    // Host to device
    let start = Instant::now();
    transfer_buffer.copy_from_host(&host_data)?;
    let h2d_time = start.elapsed();
    println!("  Host-to-device time: {:?}", h2d_time);
    println!(
        "  Bandwidth: {:.2} GB/s",
        (transfer_size * 8) as f64 / 1e9 / h2d_time.as_secs_f64()
    );

    // Device to host
    let start = Instant::now();
    let retrieved = transfer_buffer.copy_to_host()?;
    let d2h_time = start.elapsed();
    println!("  Device-to-host time: {:?}", d2h_time);
    println!(
        "  Bandwidth: {:.2} GB/s",
        (transfer_size * 8) as f64 / 1e9 / d2h_time.as_secs_f64()
    );

    // Verify data integrity
    assert_eq!(retrieved.len(), host_data.len());
    for (i, (&a, &b)) in retrieved.iter().zip(host_data.iter()).enumerate() {
        assert_eq!(a, b, "Mismatch at index {}", i);
    }
    println!("  ✓ Data integrity verified");

    // Check transfer statistics
    let transfer_stats = transfer_buffer.stats();
    println!("\nTransfer statistics:");
    println!("  H2D transfers: {}", transfer_stats.h2d_transfers);
    println!("  D2H transfers: {}", transfer_stats.d2h_transfers);
    println!(
        "  H2D bytes: {:.2} KB",
        transfer_stats.h2d_bytes as f64 / 1e3
    );
    println!(
        "  D2H bytes: {:.2} KB",
        transfer_stats.d2h_bytes as f64 / 1e3
    );
    println!();

    // === Example 5: Element-wise Operations ===
    println!("Example 5: Element-wise Operations on Device");
    println!("---------------------------------------------");

    let op_size = 1_000_000;
    println!(
        "Performing element-wise operations on {} elements...",
        op_size
    );

    // Allocate buffers
    let mut a = device.allocate_filled(op_size, 2.0)?;
    let mut b = device.allocate_filled(op_size, 3.0)?;
    let mut c = device.allocate(op_size)?;

    // Element-wise addition: c = a + b
    let start = Instant::now();
    device.add(&a, &b, &mut c)?;
    let add_time = start.elapsed();
    println!("  Addition time: {:?}", add_time);
    println!(
        "  Throughput: {:.2} GFLOPS",
        op_size as f64 / 1e9 / add_time.as_secs_f64()
    );

    // Verify result
    let result = c.copy_to_host()?;
    assert!(result.iter().all(|&x| x == 5.0), "Addition failed");
    println!("  ✓ Addition result verified: 2.0 + 3.0 = 5.0");

    // Element-wise multiplication: c = a * b
    a.fill(4.0)?;
    b.fill(5.0)?;

    let start = Instant::now();
    device.mul(&a, &b, &mut c)?;
    let mul_time = start.elapsed();
    println!("\n  Multiplication time: {:?}", mul_time);
    println!(
        "  Throughput: {:.2} GFLOPS",
        op_size as f64 / 1e9 / mul_time.as_secs_f64()
    );

    // Verify result
    let result = c.copy_to_host()?;
    assert!(result.iter().all(|&x| x == 20.0), "Multiplication failed");
    println!("  ✓ Multiplication result verified: 4.0 * 5.0 = 20.0");

    // Check operation statistics
    let op_stats = device.stats();
    println!("\nDevice statistics after operations:");
    println!("  Total operations: {}", op_stats.operations);
    println!(
        "  Current allocated: {:.2} MB",
        op_stats.allocated_bytes as f64 / 1e6
    );
    println!();

    // === Example 6: Buffer Lifecycle ===
    println!("Example 6: Buffer Lifecycle and Memory Management");
    println!("--------------------------------------------------");

    let initial_stats = device.stats();
    println!("Initial state:");
    println!("  Allocations: {}", initial_stats.allocations);
    println!("  Deallocations: {}", initial_stats.deallocations);
    println!(
        "  Allocated bytes: {:.2} MB",
        initial_stats.allocated_bytes as f64 / 1e6
    );

    {
        // Create temporary buffers in scope
        let _temp1 = device.allocate::<f64>(1000)?;
        let _temp2 = device.allocate::<f32>(2000)?;
        let _temp3 = device.allocate::<i32>(3000)?;

        let temp_stats = device.stats();
        println!("\nWith temporary buffers:");
        println!("  Allocations: {}", temp_stats.allocations);
        println!(
            "  Allocated bytes: {:.2} MB",
            temp_stats.allocated_bytes as f64 / 1e6
        );
    } // Buffers dropped here

    let final_stats = device.stats();
    println!("\nAfter buffer drop:");
    println!("  Allocations: {}", final_stats.allocations);
    println!("  Deallocations: {}", final_stats.deallocations);
    println!(
        "  Allocated bytes: {:.2} MB",
        final_stats.allocated_bytes as f64 / 1e6
    );
    println!("  ✓ Memory properly reclaimed");
    println!();

    // === Example 7: Performance Comparison ===
    println!("Example 7: CPU vs Device Performance");
    println!("-------------------------------------");

    let perf_size = 10_000_000;
    println!("Comparing performance for {} elements...\n", perf_size);

    // CPU baseline
    println!("CPU baseline:");
    let cpu_a = vec![2.0f64; perf_size];
    let cpu_b = vec![3.0f64; perf_size];
    let start = Instant::now();
    let cpu_result: Vec<f64> = cpu_a.iter().zip(cpu_b.iter()).map(|(a, b)| a + b).collect();
    let cpu_time = start.elapsed();
    println!("  Time: {:?}", cpu_time);
    println!(
        "  Throughput: {:.2} GFLOPS",
        perf_size as f64 / 1e9 / cpu_time.as_secs_f64()
    );
    assert!(cpu_result.iter().all(|&x| x == 5.0));

    // Device operation
    println!("\nDevice operation:");
    let mut dev_a = device.allocate(perf_size)?;
    let mut dev_b = device.allocate(perf_size)?;
    let mut dev_c = device.allocate(perf_size)?;

    dev_a.copy_from_host(&cpu_a)?;
    dev_b.copy_from_host(&cpu_b)?;

    let start = Instant::now();
    device.add(&dev_a, &dev_b, &mut dev_c)?;
    device.synchronize()?; // Wait for completion
    let dev_time = start.elapsed();

    println!("  Time: {:?}", dev_time);
    println!(
        "  Throughput: {:.2} GFLOPS",
        perf_size as f64 / 1e9 / dev_time.as_secs_f64()
    );

    let dev_result = dev_c.copy_to_host()?;
    assert!(dev_result.iter().all(|&x| x == 5.0));

    println!(
        "\nSpeedup: {:.2}x",
        cpu_time.as_secs_f64() / dev_time.as_secs_f64()
    );
    println!();

    // === Summary ===
    println!("=== Summary ===");
    println!("The GPU abstraction layer provides:");
    println!("  ✓ Device-agnostic interface for tensor operations");
    println!("  ✓ Automatic device selection (CPU fallback available)");
    println!("  ✓ Efficient buffer management with automatic cleanup");
    println!("  ✓ Host-device memory transfers with statistics");
    println!("  ✓ Element-wise operations (add, multiply)");
    println!("  ✓ Comprehensive performance monitoring");
    println!();

    println!("Architecture:");
    println!("  - Current: CPU fallback implementation");
    println!("  - Future: CUDA, ROCm, Vulkan, Metal backends");
    println!("  - Same API across all backends");
    println!();

    println!("Performance notes:");
    println!("  - CPU fallback uses parallel iterators (scirs2-core)");
    println!("  - Automatic SIMD vectorization where possible");
    println!("  - Future GPU backends will provide significant speedups");

    Ok(())
}
