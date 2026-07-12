//! Real NVIDIA CUDA dispatch for [`Device::add`](crate::gpu::Device::add) and
//! [`Device::mul`](crate::gpu::Device::mul).
//!
//! This module is compiled **only** under the `cuda-compute` feature. It routes
//! `f32` / `f64` element-wise `c = a + b` and `c = a * b` onto an actual NVIDIA
//! GPU using the pure-Rust [`oxicuda`](https://crates.io/crates/oxicuda-driver)
//! family (driver + memory + BLAS). No CUDA SDK / `nvcc` is needed at build
//! time: the kernels' PTX is generated in pure Rust and JIT-loaded at runtime
//! through `libcuda`.
//!
//! # This is a capability path, NOT a speedup
//!
//! `tenrso-ooc`'s [`DeviceBuffer`](crate::gpu::DeviceBuffer) is **host-backed**
//! (`data: Vec<T>`). Each op therefore performs a full round-trip:
//! host -> device (H2D) copy of both inputs, the kernel, then a device -> host
//! (D2H) copy of the result. For a single element-wise op the PCIe transfer
//! **dominates** and this path is *slower* than the CPU implementation. It
//! exists to make the dispatch **real and correct** (the arithmetic genuinely
//! runs on the GPU), not to accelerate a lone host-buffer op. Keeping data
//! resident on the device across many ops is a future milestone.
//!
//! # Honesty
//!
//! Every oxicuda failure (driver init, context creation, allocation, H2D/D2H
//! copy, kernel launch, stream sync) is propagated as an [`anyhow::Error`]. A
//! requested GPU op that fails returns `Err` — it is **never** silently
//! replaced by a CPU result presented as if the GPU had produced it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use oxicuda_blas::elementwise;
use oxicuda_blas::handle::BlasHandle;
use oxicuda_blas::types::GpuFloat;
use oxicuda_driver::{Context, Device as CudaDevice};
use oxicuda_memory::DeviceBuffer as CudaDeviceBuffer;

/// Which element-wise binary kernel to launch.
#[derive(Clone, Copy)]
enum BinOp {
    /// `c[i] = a[i] + b[i]`
    Add,
    /// `c[i] = a[i] * b[i]`
    Mul,
}

impl BinOp {
    /// Short name for error messages.
    fn as_str(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Mul => "mul",
        }
    }
}

/// A per-thread, initialised CUDA context + BLAS handle for one device.
///
/// The [`BlasHandle`] owns its own `Arc<Context>` internally and keeps a warm
/// compiled-module cache, so reusing it across calls amortises both context
/// creation and PTX JIT.
struct CudaState {
    handle: BlasHandle,
}

thread_local! {
    /// Cache of [`CudaState`] keyed by CUDA device ordinal.
    ///
    /// A `thread_local` is correct here because a CUDA context's "current-ness"
    /// is a per-thread property and [`BlasHandle`] is `Send` but not `Sync`.
    /// Each thread that dispatches an op lazily builds (and then reuses) its own
    /// context/handle for a given device ordinal.
    static CUDA_STATE: RefCell<HashMap<i32, CudaState>> = RefCell::new(HashMap::new());
}

/// Runs `f` with a ready [`BlasHandle`] for `device_ordinal`, building and
/// caching the CUDA context/handle on first use for this thread.
///
/// The handle's context is made current on the calling thread before `f` runs,
/// so subsequent device allocations and copies target the right context even if
/// another context became current in between calls.
fn with_handle<R>(device_ordinal: i32, f: impl FnOnce(&BlasHandle) -> Result<R>) -> Result<R> {
    CUDA_STATE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let state = match cache.entry(device_ordinal) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                oxicuda_driver::init()
                    .map_err(|e| anyhow!("oxicuda: CUDA driver init failed: {e}"))?;
                let device = CudaDevice::get(device_ordinal).map_err(|e| {
                    anyhow!("oxicuda: CUDA device #{device_ordinal} unavailable: {e}")
                })?;
                let context = Arc::new(Context::new(&device).map_err(|e| {
                    anyhow!(
                        "oxicuda: failed to create CUDA context on device #{device_ordinal}: {e}"
                    )
                })?);
                let handle = BlasHandle::new(&context).map_err(|e| {
                    anyhow!(
                        "oxicuda: failed to create BLAS handle on device #{device_ordinal}: {e}"
                    )
                })?;
                slot.insert(CudaState { handle })
            }
        };

        // Bind our context current on this thread before issuing any device
        // work; `Context::new` makes it current on creation, but a later call
        // could have run after another context became current.
        state
            .handle
            .context()
            .set_current()
            .map_err(|e| anyhow!("oxicuda: failed to make CUDA context current: {e}"))?;

        f(&state.handle)
    })
}

/// Executes one element-wise binary op on the GPU: uploads `a` and `b`, launches
/// the kernel, synchronises the launch stream, and copies the result into `c`.
///
/// # Correctness
///
/// The kernel launches on the handle's (non-null) stream, so the stream is
/// synchronised **before** the D2H copy; skipping the sync would read stale
/// device memory.
///
/// # Errors
///
/// Returns `Err` on any driver/allocation/copy/launch/sync failure, or on a
/// length mismatch between the three slices.
fn run_binary<T: GpuFloat>(
    device_ordinal: i32,
    op: BinOp,
    a: &[T],
    b: &[T],
    c: &mut [T],
) -> Result<()> {
    let n = a.len();
    if b.len() != n || c.len() != n {
        return Err(anyhow!(
            "cuda {} dispatch: slice length mismatch (a={}, b={}, c={})",
            op.as_str(),
            n,
            b.len(),
            c.len()
        ));
    }
    if n == 0 {
        return Ok(());
    }
    let n_u32 = u32::try_from(n)
        .map_err(|_| anyhow!("cuda {} dispatch: length {n} exceeds u32", op.as_str()))?;

    with_handle(device_ordinal, |handle| {
        let d_a = CudaDeviceBuffer::from_host(a)
            .map_err(|e| anyhow!("oxicuda: H2D copy of A failed: {e}"))?;
        let d_b = CudaDeviceBuffer::from_host(b)
            .map_err(|e| anyhow!("oxicuda: H2D copy of B failed: {e}"))?;
        let mut d_c = CudaDeviceBuffer::<T>::alloc(n)
            .map_err(|e| anyhow!("oxicuda: device allocation of C failed: {e}"))?;

        let launched = match op {
            BinOp::Add => elementwise::add::<T>(handle, n_u32, &d_a, &d_b, &mut d_c),
            BinOp::Mul => elementwise::mul::<T>(handle, n_u32, &d_a, &d_b, &mut d_c),
        };
        launched.map_err(|e| {
            anyhow!(
                "oxicuda: elementwise {} kernel launch failed: {e}",
                op.as_str()
            )
        })?;

        // The kernel runs asynchronously on the handle's stream; wait for it to
        // complete before copying results back, or D2H would read stale memory.
        handle
            .stream()
            .synchronize()
            .map_err(|e| anyhow!("oxicuda: stream synchronise failed: {e}"))?;

        d_c.copy_to_host(c)
            .map_err(|e| anyhow!("oxicuda: D2H copy of C failed: {e}"))?;
        Ok(())
    })
}

/// GPU element-wise add for `f32`: `c[i] = a[i] + b[i]` on device
/// `device_ordinal`.
pub(crate) fn cuda_elementwise_add_f32(
    device_ordinal: i32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    run_binary(device_ordinal, BinOp::Add, a, b, c)
}

/// GPU element-wise multiply for `f32`: `c[i] = a[i] * b[i]` on device
/// `device_ordinal`.
pub(crate) fn cuda_elementwise_mul_f32(
    device_ordinal: i32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    run_binary(device_ordinal, BinOp::Mul, a, b, c)
}

/// GPU element-wise add for `f64`: `c[i] = a[i] + b[i]` on device
/// `device_ordinal`.
pub(crate) fn cuda_elementwise_add_f64(
    device_ordinal: i32,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) -> Result<()> {
    run_binary(device_ordinal, BinOp::Add, a, b, c)
}

/// GPU element-wise multiply for `f64`: `c[i] = a[i] * b[i]` on device
/// `device_ordinal`.
pub(crate) fn cuda_elementwise_mul_f64(
    device_ordinal: i32,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) -> Result<()> {
    run_binary(device_ordinal, BinOp::Mul, a, b, c)
}
