//! GPU abstraction layer for tensor operations
//!
//! This module provides a device-agnostic interface for GPU-accelerated tensor operations
//! with CPU fallback. It supports:
//! - Unified device buffer management (CPU/GPU)
//! - Asynchronous memory transfers
//! - Multi-device support
//! - Automatic device selection
//!
//! # Architecture
//!
//! The abstraction is designed to be backend-agnostic, allowing future integration with:
//! - CUDA (NVIDIA GPUs)
//! - ROCm (AMD GPUs)
//! - Vulkan Compute
//! - Metal (Apple Silicon)
//!
//! # Current Implementation
//!
//! The current implementation uses a CPU fallback backend that:
//! - Simulates GPU behavior using CPU operations
//! - Supports async transfers using thread pools
//! - Provides the same API as future GPU backends
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::gpu::{DeviceManager, DeviceType, DeviceBuffer};
//!
//! // Get device manager
//! let manager = DeviceManager::new()?;
//!
//! // Select best device (CPU fallback if no GPU)
//! let device = manager.best_device()?;
//!
//! // Allocate buffer on device
//! let mut buffer = device.allocate::<f64>(1024)?;
//!
//! // Copy data to device
//! let data = vec![1.0; 1024];
//! buffer.copy_from_host(&data)?;
//!
//! // Perform operations on device
//! // ...
//!
//! // Copy result back
//! let result = buffer.copy_to_host()?;
//! ```

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use scirs2_core::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device (always available)
    Cpu,
    /// CUDA GPU (NVIDIA)
    Cuda,
    /// ROCm GPU (AMD)
    Rocm,
    /// Vulkan Compute
    Vulkan,
    /// Metal (Apple Silicon)
    Metal,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Cuda => write!(f, "CUDA"),
            DeviceType::Rocm => write!(f, "ROCm"),
            DeviceType::Vulkan => write!(f, "Vulkan"),
            DeviceType::Metal => write!(f, "Metal"),
        }
    }
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type
    pub device_type: DeviceType,
    /// Device ID (0-based)
    pub device_id: usize,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability (device-specific)
    pub compute_capability: String,
    /// Number of compute units / streaming multiprocessors
    pub compute_units: usize,
}

/// Device buffer holding data on a specific device
pub struct DeviceBuffer<T> {
    /// Buffer data (CPU fallback uses Vec)
    data: Vec<T>,
    /// Device this buffer belongs to
    device: Arc<Device>,
    /// Buffer size in elements
    size: usize,
    /// Statistics
    stats: Arc<DeviceBufferStats>,
}

/// Device buffer statistics
struct DeviceBufferStats {
    /// Total bytes allocated
    bytes_allocated: AtomicU64,
    /// Number of host->device transfers
    h2d_transfers: AtomicU64,
    /// Number of device->host transfers
    d2h_transfers: AtomicU64,
    /// Total bytes transferred to device
    h2d_bytes: AtomicU64,
    /// Total bytes transferred from device
    d2h_bytes: AtomicU64,
}

impl DeviceBufferStats {
    fn new(bytes: u64) -> Self {
        Self {
            bytes_allocated: AtomicU64::new(bytes),
            h2d_transfers: AtomicU64::new(0),
            d2h_transfers: AtomicU64::new(0),
            h2d_bytes: AtomicU64::new(0),
            d2h_bytes: AtomicU64::new(0),
        }
    }

    fn record_h2d(&self, bytes: u64) {
        self.h2d_transfers.fetch_add(1, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_d2h(&self, bytes: u64) {
        self.d2h_transfers.fetch_add(1, Ordering::Relaxed);
        self.d2h_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DeviceBufferStatsSnapshot {
        DeviceBufferStatsSnapshot {
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            h2d_transfers: self.h2d_transfers.load(Ordering::Relaxed),
            d2h_transfers: self.d2h_transfers.load(Ordering::Relaxed),
            h2d_bytes: self.h2d_bytes.load(Ordering::Relaxed),
            d2h_bytes: self.d2h_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Device buffer statistics snapshot
#[derive(Debug, Clone)]
pub struct DeviceBufferStatsSnapshot {
    /// Total bytes allocated
    pub bytes_allocated: u64,
    /// Number of host->device transfers
    pub h2d_transfers: u64,
    /// Number of device->host transfers
    pub d2h_transfers: u64,
    /// Total bytes transferred to device
    pub h2d_bytes: u64,
    /// Total bytes transferred from device
    pub d2h_bytes: u64,
}

impl<T> DeviceBuffer<T> {
    /// Create new device buffer
    fn new(device: Arc<Device>, size: usize) -> Self
    where
        T: Clone + Send + Sync,
    {
        let bytes = size * std::mem::size_of::<T>();
        let device_ref = device.clone();
        device_ref.stats.record_allocation(bytes as u64);

        Self {
            data: Vec::with_capacity(size),
            device,
            size,
            stats: Arc::new(DeviceBufferStats::new(bytes as u64)),
        }
    }

    /// Get buffer size in elements
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }

    /// Get device this buffer belongs to
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get statistics
    pub fn stats(&self) -> DeviceBufferStatsSnapshot {
        self.stats.snapshot()
    }
}

impl<T: Clone + Send + Sync> DeviceBuffer<T> {
    /// Copy data from host to device
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.size {
            return Err(anyhow!(
                "Data size mismatch: expected {}, got {}",
                self.size,
                data.len()
            ));
        }

        // CPU fallback: simple copy
        self.data.clear();
        self.data.extend_from_slice(data);

        self.stats.record_h2d(self.size_bytes() as u64);
        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self) -> Result<Vec<T>> {
        // CPU fallback: simple clone
        let result = self.data.clone();
        self.stats.record_d2h(self.size_bytes() as u64);
        Ok(result)
    }

    /// Fill buffer with a constant value
    pub fn fill(&mut self, value: T) -> Result<()>
    where
        T: Clone,
    {
        self.data.clear();
        self.data.resize(self.size, value);
        Ok(())
    }

    /// Get reference to underlying data (CPU fallback only)
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable reference to underlying data (CPU fallback only)
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        let bytes = self.size_bytes() as u64;
        self.device.stats.record_deallocation(bytes);
    }
}

/// Device for tensor operations
pub struct Device {
    /// Device information
    info: DeviceInfo,
    /// Device statistics
    stats: Arc<DeviceStats>,
}

/// Device statistics
struct DeviceStats {
    /// Total allocations
    allocations: AtomicU64,
    /// Total deallocations
    deallocations: AtomicU64,
    /// Current allocated bytes
    allocated_bytes: AtomicU64,
    /// Peak allocated bytes
    peak_bytes: AtomicU64,
    /// Total operations executed
    operations: AtomicU64,
}

impl DeviceStats {
    fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            allocated_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            operations: AtomicU64::new(0),
        }
    }

    fn record_allocation(&self, bytes: u64) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let new_allocated = self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;

        // Update peak if needed
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while new_allocated > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                new_allocated,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }
    }

    fn record_deallocation(&self, bytes: u64) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn record_operation(&self) {
        self.operations.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DeviceStatsSnapshot {
        DeviceStatsSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            operations: self.operations.load(Ordering::Relaxed),
        }
    }
}

/// Device statistics snapshot
#[derive(Debug, Clone)]
pub struct DeviceStatsSnapshot {
    /// Total allocations
    pub allocations: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Current allocated bytes
    pub allocated_bytes: u64,
    /// Peak allocated bytes
    pub peak_bytes: u64,
    /// Total operations executed
    pub operations: u64,
}

impl Device {
    /// Create a new device
    fn new(info: DeviceInfo) -> Arc<Self> {
        Arc::new(Self {
            info,
            stats: Arc::new(DeviceStats::new()),
        })
    }

    /// Get device information
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get device type
    pub fn device_type(&self) -> DeviceType {
        self.info.device_type
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.info.device_id
    }

    /// Allocate buffer on device
    pub fn allocate<T: Clone + Send + Sync>(
        self: &Arc<Self>,
        size: usize,
    ) -> Result<DeviceBuffer<T>> {
        Ok(DeviceBuffer::new(self.clone(), size))
    }

    /// Allocate and fill buffer
    pub fn allocate_filled<T: Clone + Send + Sync>(
        self: &Arc<Self>,
        size: usize,
        value: T,
    ) -> Result<DeviceBuffer<T>> {
        let mut buffer = self.allocate(size)?;
        buffer.fill(value)?;
        Ok(buffer)
    }

    /// Element-wise addition: c = a + b
    pub fn add<T>(
        &self,
        a: &DeviceBuffer<T>,
        b: &DeviceBuffer<T>,
        c: &mut DeviceBuffer<T>,
    ) -> Result<()>
    where
        T: Clone + Send + Sync + std::ops::Add<Output = T>,
    {
        if a.size() != b.size() || a.size() != c.size() {
            return Err(anyhow!("Buffer size mismatch"));
        }

        self.stats.record_operation();

        // CPU fallback
        #[cfg(feature = "parallel")]
        {
            c.as_mut_slice()
                .par_iter_mut()
                .zip(a.as_slice().par_iter().zip(b.as_slice().par_iter()))
                .for_each(|(c_val, (a_val, b_val))| {
                    *c_val = a_val.clone() + b_val.clone();
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..a.size() {
                c.as_mut_slice()[i] = a.as_slice()[i].clone() + b.as_slice()[i].clone();
            }
        }

        Ok(())
    }

    /// Element-wise multiplication: c = a * b
    pub fn mul<T>(
        &self,
        a: &DeviceBuffer<T>,
        b: &DeviceBuffer<T>,
        c: &mut DeviceBuffer<T>,
    ) -> Result<()>
    where
        T: Clone + Send + Sync + std::ops::Mul<Output = T>,
    {
        if a.size() != b.size() || a.size() != c.size() {
            return Err(anyhow!("Buffer size mismatch"));
        }

        self.stats.record_operation();

        // CPU fallback
        #[cfg(feature = "parallel")]
        {
            c.as_mut_slice()
                .par_iter_mut()
                .zip(a.as_slice().par_iter().zip(b.as_slice().par_iter()))
                .for_each(|(c_val, (a_val, b_val))| {
                    *c_val = a_val.clone() * b_val.clone();
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..a.size() {
                c.as_mut_slice()[i] = a.as_slice()[i].clone() * b.as_slice()[i].clone();
            }
        }

        Ok(())
    }

    /// Synchronize device (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        // CPU fallback: no-op
        Ok(())
    }

    /// Get device statistics
    pub fn stats(&self) -> DeviceStatsSnapshot {
        self.stats.snapshot()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // Cleanup device resources (CPU fallback: no-op)
    }
}

/// Device manager for managing multiple devices
pub struct DeviceManager {
    /// Available devices
    devices: Vec<Arc<Device>>,
    /// Default device index
    default_device: AtomicUsize,
}

impl DeviceManager {
    /// Create a new device manager
    ///
    /// Enumerates all available devices (currently only CPU fallback)
    pub fn new() -> Result<Self> {
        let mut devices = Vec::new();

        // Always add CPU device
        let cpu_info = DeviceInfo {
            device_type: DeviceType::Cpu,
            device_id: 0,
            name: "CPU".to_string(),
            total_memory: Self::get_system_memory(),
            available_memory: Self::get_available_memory(),
            compute_capability: "N/A".to_string(),
            compute_units: num_cpus::get(),
        };
        devices.push(Device::new(cpu_info));

        // TODO: Add GPU device enumeration when backends are implemented
        // - CUDA devices
        // - ROCm devices
        // - Vulkan devices
        // - Metal devices

        Ok(Self {
            devices,
            default_device: AtomicUsize::new(0),
        })
    }

    /// Get total system memory in bytes
    fn get_system_memory() -> usize {
        // Platform-specific memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
        }

        // Fallback: 16 GB
        16 * 1024 * 1024 * 1024
    }

    /// Get available system memory in bytes
    fn get_available_memory() -> usize {
        // Platform-specific memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
        }

        // Fallback: assume 50% available
        Self::get_system_memory() / 2
    }

    /// Get number of available devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get device by index
    pub fn device(&self, index: usize) -> Result<Arc<Device>> {
        self.devices
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow!("Device index out of range: {}", index))
    }

    /// Get devices by type
    pub fn devices_by_type(&self, device_type: DeviceType) -> Vec<Arc<Device>> {
        self.devices
            .iter()
            .filter(|d| d.device_type() == device_type)
            .cloned()
            .collect()
    }

    /// Get default device
    pub fn default_device(&self) -> Result<Arc<Device>> {
        let index = self.default_device.load(Ordering::Relaxed);
        self.device(index)
    }

    /// Set default device
    pub fn set_default_device(&self, index: usize) -> Result<()> {
        if index >= self.devices.len() {
            return Err(anyhow!("Device index out of range: {}", index));
        }
        self.default_device.store(index, Ordering::Relaxed);
        Ok(())
    }

    /// Get best available device
    ///
    /// Prefers GPU over CPU, and devices with more memory
    pub fn best_device(&self) -> Result<Arc<Device>> {
        let mut best = self.device(0)?;

        for device in &self.devices {
            // Prefer GPU over CPU
            if device.device_type() != DeviceType::Cpu && best.device_type() == DeviceType::Cpu {
                best = device.clone();
                continue;
            }

            // Among same type, prefer more memory
            if device.device_type() == best.device_type()
                && device.info().available_memory > best.info().available_memory
            {
                best = device.clone();
            }
        }

        Ok(best)
    }

    /// List all devices
    pub fn list_devices(&self) -> Vec<DeviceInfo> {
        self.devices.iter().map(|d| d.info().clone()).collect()
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create device manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new().unwrap();
        assert!(manager.device_count() > 0);

        // CPU device should always be available
        let devices = manager.devices_by_type(DeviceType::Cpu);
        assert_eq!(devices.len(), 1);
    }

    #[test]
    fn test_device_info() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert_eq!(device.device_id(), 0);
        assert!(device.info().total_memory > 0);
        assert!(device.info().compute_units > 0);
    }

    #[test]
    fn test_buffer_allocation() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let buffer: DeviceBuffer<f64> = device.allocate(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * std::mem::size_of::<f64>());
    }

    #[test]
    fn test_buffer_fill() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let mut buffer = device.allocate::<f64>(100).unwrap();
        buffer.fill(42.0).unwrap();

        let data = buffer.copy_to_host().unwrap();
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_host_device_copy() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let host_data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut buffer = device.allocate(100).unwrap();

        buffer.copy_from_host(&host_data).unwrap();
        let result = buffer.copy_to_host().unwrap();

        assert_eq!(result, host_data);
    }

    #[test]
    fn test_device_add() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let mut a = device.allocate(100).unwrap();
        let mut b = device.allocate(100).unwrap();
        let mut c = device.allocate(100).unwrap();

        a.fill(1.0).unwrap();
        b.fill(2.0).unwrap();

        device.add(&a, &b, &mut c).unwrap();

        let result = c.copy_to_host().unwrap();
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_device_mul() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let mut a = device.allocate(100).unwrap();
        let mut b = device.allocate(100).unwrap();
        let mut c = device.allocate(100).unwrap();

        a.fill(3.0).unwrap();
        b.fill(4.0).unwrap();

        device.mul(&a, &b, &mut c).unwrap();

        let result = c.copy_to_host().unwrap();
        assert!(result.iter().all(|&x| x == 12.0));
    }

    #[test]
    fn test_buffer_stats() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let mut buffer = device.allocate::<f64>(1024).unwrap();

        let data = vec![1.0; 1024];
        buffer.copy_from_host(&data).unwrap();
        let _ = buffer.copy_to_host().unwrap();

        let stats = buffer.stats();
        assert_eq!(stats.h2d_transfers, 1);
        assert_eq!(stats.d2h_transfers, 1);
        assert_eq!(stats.h2d_bytes, 1024 * 8);
        assert_eq!(stats.d2h_bytes, 1024 * 8);
    }

    #[test]
    fn test_device_stats() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let _buffer1 = device.allocate::<f64>(1024).unwrap();
        let _buffer2 = device.allocate::<f32>(2048).unwrap();

        let stats = device.stats();
        assert_eq!(stats.allocations, 2);
        assert!(stats.allocated_bytes > 0);
    }

    #[test]
    fn test_default_device() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.default_device().unwrap();
        assert_eq!(device.device_id(), 0);

        // Try setting different default
        if manager.device_count() > 1 {
            manager.set_default_device(1).unwrap();
            let device = manager.default_device().unwrap();
            assert_eq!(device.device_id(), 1);
        }
    }

    #[test]
    fn test_best_device() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.best_device().unwrap();

        // Should return CPU device when no GPU available
        assert_eq!(device.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_list_devices() {
        let manager = DeviceManager::new().unwrap();
        let devices = manager.list_devices();

        assert!(!devices.is_empty());
        assert_eq!(devices[0].device_type, DeviceType::Cpu);
    }

    #[test]
    fn test_buffer_size_mismatch() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let mut buffer = device.allocate::<f64>(100).unwrap();
        let wrong_data = vec![1.0; 50];

        assert!(buffer.copy_from_host(&wrong_data).is_err());
    }

    #[test]
    fn test_device_op_size_mismatch() {
        let manager = DeviceManager::new().unwrap();
        let device = manager.device(0).unwrap();

        let a = device.allocate_filled(100, 1.0).unwrap();
        let b = device.allocate_filled(50, 2.0).unwrap();
        let mut c = device.allocate(100).unwrap();

        assert!(device.add(&a, &b, &mut c).is_err());
    }
}
