//! Allocation-profiling harness for `DenseND<T>` core operations.
//!
//! Answers, with real measured numbers rather than guesses: does `reshape`
//! allocate? Does `permute` copy? Does `view()` allocate? How does that compare
//! to doing the same thing directly on the raw `scirs2_core::ndarray_ext` array
//! `DenseND` wraps?
//!
//! This is deliberately *not* a criterion benchmark: criterion measures wall
//! time, and its own bookkeeping (batching, warm-up, statistics) allocates in
//! ways that would pollute an allocation count. Instead this is a plain binary
//! (`harness = false`, no `fn main` from criterion) with a custom counting
//! `#[global_allocator]` that wraps `std::alloc::System`, incrementing atomic
//! counters on every `alloc`/`dealloc`/`realloc`. Each operation is measured by
//! diffing the counters immediately before/after a single call — no external
//! profiler needed.
//!
//! Run with:
//! ```bash
//! cargo bench --bench alloc_profile
//! ```
//! (despite the `cargo bench` invocation, this prints a table and exits — it
//! is not measuring wall-clock time at all).

use scirs2_core::ndarray_ext::{Array, Array2, IxDyn};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tenrso_core::DenseND;

// ============================================================================
// Counting global allocator
// ============================================================================

/// Total number of `alloc` calls observed since process start.
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Total bytes requested across all `alloc` calls since process start.
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Total number of `dealloc` calls observed since process start.
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Total bytes freed across all `dealloc` calls since process start.
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// A `GlobalAlloc` that delegates to `System` while counting every
/// allocation/deallocation. This is the standard pure-Rust way to profile
/// allocations without an external tool (valgrind/heaptrack/DHAT are not
/// always available in CI/sandbox environments).
struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    // `realloc` intentionally left at its default `GlobalAlloc` implementation,
    // which is defined in terms of `self.alloc` + `self.dealloc` — so growth
    // (e.g. `Vec::push` past capacity) is still counted correctly without
    // needing a bespoke override here.
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocator counters at a point in time (allocation side only — see
/// `AllocReport` for why deallocations aren't part of the per-operation
/// report).
#[derive(Clone, Copy)]
struct AllocSnapshot {
    alloc_count: usize,
    alloc_bytes: u64,
}

fn snapshot() -> AllocSnapshot {
    AllocSnapshot {
        alloc_count: ALLOC_COUNT.load(Ordering::SeqCst),
        alloc_bytes: ALLOC_BYTES.load(Ordering::SeqCst),
    }
}

/// Allocation activity attributable to running `f`.
///
/// The primary deliverable is `allocs`/`alloc_bytes` (per the task at hand:
/// "number of allocations and total bytes"). `wall_time` is captured too,
/// as a single-shot (not averaged/statistical) sanity check — it is not a
/// substitute for the criterion measurements in `benches/ndarray_baseline.rs`,
/// but it is enough to confirm, e.g., that a zero-allocation operation is
/// also cheap in wall-clock terms rather than merely allocation-free.
struct AllocReport {
    allocs: usize,
    alloc_bytes: u64,
    wall_time: std::time::Duration,
}

/// Run `f` once, returning its result together with the allocator activity
/// caused strictly by `f` (measured by diffing global counters immediately
/// before and after the call — nothing else runs on this thread in between).
fn measure<F, R>(f: F) -> (R, AllocReport)
where
    F: FnOnce() -> R,
{
    let before = snapshot();
    let start = std::time::Instant::now();
    let result = std::hint::black_box(f());
    let wall_time = start.elapsed();
    let after = snapshot();
    let report = AllocReport {
        allocs: after.alloc_count - before.alloc_count,
        alloc_bytes: after.alloc_bytes - before.alloc_bytes,
        wall_time,
    };
    (result, report)
}

// ============================================================================
// Report table
// ============================================================================

struct Row {
    operation: String,
    /// `None` means "no `DenseND` equivalent exists" (e.g. a zero-copy
    /// consuming API `DenseND` doesn't expose), not "measured as zero".
    dense_allocs: Option<usize>,
    dense_bytes: Option<u64>,
    /// Single-shot wall time; see `AllocReport::wall_time` docs.
    dense_time: Option<std::time::Duration>,
    raw_allocs: usize,
    raw_bytes: u64,
    raw_time: std::time::Duration,
}

fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map_or_else(|| "n/a".to_string(), |x| x.to_string())
}

fn fmt_opt_u64(v: Option<u64>) -> String {
    v.map_or_else(|| "n/a".to_string(), |x| x.to_string())
}

fn fmt_opt_duration(v: Option<std::time::Duration>) -> String {
    v.map_or_else(|| "n/a".to_string(), |d| format!("{:.3?}", d))
}

fn print_table(rows: &[Row]) {
    println!(
        "{:<38} | {:>13} | {:>13} | {:>10} | {:>13} | {:>13} | {:>10}",
        "operation",
        "dense_nd #allocs",
        "dense_nd bytes",
        "dense_nd t",
        "ndarray #allocs",
        "ndarray bytes",
        "ndarray t"
    );
    println!("{}", "-".repeat(38 + (3 + 13) * 3 + (3 + 10) * 2));
    for row in rows {
        println!(
            "{:<38} | {:>13} | {:>13} | {:>10} | {:>13} | {:>13} | {:>10}",
            row.operation,
            fmt_opt_usize(row.dense_allocs),
            fmt_opt_u64(row.dense_bytes),
            fmt_opt_duration(row.dense_time),
            row.raw_allocs,
            row.raw_bytes,
            format!("{:.3?}", row.raw_time)
        );
    }
}

type RawArray = Array<f64, IxDyn>;

const UNFOLD_MODE: usize = 1;
const PERMUTE_AXES: [usize; 3] = [2, 0, 1];

/// Hand-fused single-pass unfold on the raw array (one copy). See
/// `benches/ndarray_baseline.rs` for the full rationale.
fn raw_unfold_single_pass(raw: &RawArray, mode: usize) -> Array2<f64> {
    let shape = raw.shape().to_vec();
    let rows = shape[mode];
    let cols: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_, &s)| s)
        .product();
    let mut perm: Vec<usize> = vec![mode];
    perm.extend((0..mode).chain((mode + 1)..shape.len()));
    let permuted_view = raw.view().permuted_axes(IxDyn(&perm));
    let flat: Vec<f64> = permuted_view.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), flat).expect("shape matches by construction")
}

fn profile_for_size(shape: &[usize], label: &str, rows: &mut Vec<Row>) {
    let total: usize = shape.iter().product();
    let bytes_per_tensor = (total * std::mem::size_of::<f64>()) as u64;
    let flat_data: Vec<f64> = (0..total).map(|i| i as f64).collect();

    // Build the fixtures used by every subsequent measurement. Their
    // construction cost is not attributed to any row below.
    let dense_tensor = DenseND::<f64>::from_vec(flat_data.clone(), shape).expect("valid shape");
    let raw_array = RawArray::from_shape_vec(IxDyn(shape), flat_data.clone()).expect("valid shape");
    let new_shape = [shape[0] * shape[1], shape[2]];

    macro_rules! row {
        ($name:expr, $dense:expr, $raw:expr) => {{
            let (_, dense_report) = measure(|| $dense);
            let (_, raw_report) = measure(|| $raw);
            rows.push(Row {
                operation: format!("{} ({label}^3, {} elems)", $name, total),
                dense_allocs: Some(dense_report.allocs),
                dense_bytes: Some(dense_report.alloc_bytes),
                dense_time: Some(dense_report.wall_time),
                raw_allocs: raw_report.allocs,
                raw_bytes: raw_report.alloc_bytes,
                raw_time: raw_report.wall_time,
            });
        }};
    }

    row!(
        "construction_zeros",
        DenseND::<f64>::zeros(shape),
        RawArray::zeros(IxDyn(shape))
    );
    // The input `Vec` must be cloned to get a fresh, uniquely-owned buffer for
    // each call (both `from_vec` and `from_shape_vec` consume their argument),
    // but that clone is test-harness plumbing, not part of what "from_vec"
    // does. So the clone happens in an explicit setup step *outside* the
    // measured region, exactly like `iter_batched` would in a criterion bench.
    {
        let (_, dense_report) = {
            let data = flat_data.clone();
            measure(|| DenseND::from_vec(data, shape).unwrap())
        };
        let (_, raw_report) = {
            let data = flat_data.clone();
            measure(|| RawArray::from_shape_vec(IxDyn(shape), data).unwrap())
        };
        rows.push(Row {
            operation: format!("construction_from_vec ({label}^3, {total} elems)"),
            dense_allocs: Some(dense_report.allocs),
            dense_bytes: Some(dense_report.alloc_bytes),
            dense_time: Some(dense_report.wall_time),
            raw_allocs: raw_report.allocs,
            raw_bytes: raw_report.alloc_bytes,
            raw_time: raw_report.wall_time,
        });
    }
    row!(
        "reshape",
        dense_tensor.reshape(&new_shape).unwrap(),
        raw_array
            .view()
            .into_shape_with_order(IxDyn(&new_shape))
            .unwrap()
            .to_owned()
    );
    // Zero-copy consuming reshape: measured on an explicit fresh clone so the
    // clone's own allocation doesn't get attributed to the reshape call.
    {
        let owned_for_reshape = raw_array.clone();
        let (_, raw_report) = measure(|| {
            owned_for_reshape
                .into_shape_with_order(IxDyn(&new_shape))
                .unwrap()
        });
        rows.push(Row {
            operation: format!("reshape_zero_copy [ndarray-only] ({label}^3, {total} elems)"),
            dense_allocs: None,
            dense_bytes: None,
            dense_time: None,
            raw_allocs: raw_report.allocs,
            raw_bytes: raw_report.alloc_bytes,
            raw_time: raw_report.wall_time,
        });
    }
    row!(
        "permute",
        dense_tensor.permute(&PERMUTE_AXES).unwrap(),
        raw_array.clone().permuted_axes(IxDyn(&PERMUTE_AXES))
    );
    {
        let (_, raw_report) = measure(|| raw_array.view().permuted_axes(IxDyn(&PERMUTE_AXES)));
        rows.push(Row {
            operation: format!("permute_view [ndarray-only] ({label}^3, {total} elems)"),
            dense_allocs: None,
            dense_bytes: None,
            dense_time: None,
            raw_allocs: raw_report.allocs,
            raw_bytes: raw_report.alloc_bytes,
            raw_time: raw_report.wall_time,
        });
    }
    row!(
        "unfold",
        dense_tensor.unfold(UNFOLD_MODE).unwrap(),
        raw_unfold_single_pass(&raw_array, UNFOLD_MODE)
    );
    row!(
        "add (same shape)",
        &dense_tensor + &dense_tensor,
        &raw_array + &raw_array
    );
    row!(
        "hadamard",
        dense_tensor.hadamard(&dense_tensor).unwrap(),
        &raw_array * &raw_array
    );
    row!("mul_scalar", &dense_tensor * 2.0_f64, &raw_array * 2.0_f64);
    row!("sum", dense_tensor.sum(), raw_array.sum());
    row!("mean", dense_tensor.mean(), raw_array.mean().unwrap());
    row!("view", dense_tensor.view(), raw_array.view());
    {
        let idx = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
        row!("index_get", dense_tensor[&idx[..]], raw_array[IxDyn(&idx)]);
    }

    println!("\n(tensor size for {label}^3: {total} elements = {bytes_per_tensor} bytes of f64)");
}

fn main() {
    let mut rows = Vec::new();

    // Warm up the allocator/runtime once outside of any measured region so
    // one-time lazy-initialization allocations (thread-locals, etc.) don't
    // pollute the first real measurement.
    let warm = DenseND::<f64>::zeros(&[4, 4]);
    std::hint::black_box(&warm);
    drop(warm);

    profile_for_size(&[64, 64, 64], "64", &mut rows);
    profile_for_size(&[256, 256, 256], "256", &mut rows);

    println!();
    print_table(&rows);

    println!(
        "\nTotal process allocations so far: {} allocs, {} bytes; {} deallocs, {} bytes",
        ALLOC_COUNT.load(Ordering::SeqCst),
        ALLOC_BYTES.load(Ordering::SeqCst),
        DEALLOC_COUNT.load(Ordering::SeqCst),
        DEALLOC_BYTES.load(Ordering::SeqCst),
    );
}
