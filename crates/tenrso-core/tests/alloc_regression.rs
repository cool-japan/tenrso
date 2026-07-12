//! Allocation regression tests for `DenseND<T>`'s hot paths.
//!
//! These are guard-rails for four measured performance defects that were fixed
//! in the `dense` module. Every one of them was invisible to a correctness test
//! — the results were always right, they just cost 2x-3x the memory traffic they
//! needed to. So they are pinned here at the level where they are actually
//! observable: **the number of bytes the allocator hands out per operation.**
//!
//! * `&a + &b` (same shape) must allocate the output buffer and *nothing else*.
//!   It used to clone both operands first: 4 allocations / 3x the bytes.
//! * `unfold` must copy the tensor exactly once. It used to copy it twice
//!   (`permute` cloned, then `reshape` re-gathered the non-contiguous clone).
//! * `into_permuted` / `permute_view` must move or borrow the buffer: zero
//!   allocations. (`permute` still copies once — that is its contract.)
//! * `into_reshape` must be genuinely zero-copy for contiguous input, unlike the
//!   borrowing `reshape`, which unconditionally copies.
//!
//! ## How the measurement works
//!
//! A `#[global_allocator]` wraps `System` and counts allocations into
//! **thread-local** counters. Thread-locality matters: the test harness (and
//! nextest, and any rayon pool) allocates on other threads concurrently, and a
//! global atomic counter would fold that noise into our numbers. The counters
//! are `const`-initialized `Cell`s with no destructor, so touching them from
//! inside `alloc` cannot itself allocate or recurse.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use tenrso_core::DenseND;

thread_local! {
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: every method forwards to `System` unchanged; the only added work is
// bumping two thread-local `Cell`s, which cannot allocate (no destructor => no
// lazy TLS registration) and cannot panic (`try_with` swallows the
// during-destruction case).
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    // `realloc` keeps the default `GlobalAlloc` implementation, which is defined
    // in terms of `alloc` + `dealloc`, so growth is counted without an override.
}

fn record(size: usize) {
    let _ = ALLOC_COUNT.try_with(|c| c.set(c.get() + 1));
    let _ = ALLOC_BYTES.try_with(|c| c.set(c.get() + size as u64));
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocation activity caused strictly by `f`, measured on the calling thread.
struct Alloc {
    count: usize,
    bytes: u64,
}

fn measure<F, R>(f: F) -> (R, Alloc)
where
    F: FnOnce() -> R,
{
    let count_before = ALLOC_COUNT.with(Cell::get);
    let bytes_before = ALLOC_BYTES.with(Cell::get);
    let result = std::hint::black_box(f());
    let count_after = ALLOC_COUNT.with(Cell::get);
    let bytes_after = ALLOC_BYTES.with(Cell::get);
    (
        result,
        Alloc {
            count: count_after - count_before,
            bytes: bytes_after - bytes_before,
        },
    )
}

/// `[32, 32, 32]` = 32,768 elements = 262,144 bytes of `f64`.
const SHAPE: [usize; 3] = [32, 32, 32];
const ELEMS: usize = 32 * 32 * 32;
const TENSOR_BYTES: u64 = (ELEMS * std::mem::size_of::<f64>()) as u64;

fn iota() -> DenseND<f64> {
    let data: Vec<f64> = (0..ELEMS).map(|i| i as f64).collect();
    DenseND::from_vec(data, &SHAPE).expect("valid shape")
}

/// Touch the machinery once before measuring, so that any one-off lazy
/// initialization inside `std`/ndarray is not attributed to the operation.
fn warm_up() {
    let t = DenseND::<f64>::zeros(&[2, 2]);
    std::hint::black_box(t.sum());
    std::hint::black_box(&t + &t);
    std::hint::black_box(t.unfold(0).expect("mode 0"));
    std::hint::black_box(t.clone().into_permuted(&[1, 0]).expect("valid"));
    std::hint::black_box(t.clone().into_reshape(&[4]).expect("valid"));
}

// ---------------------------------------------------------------------------
// Bug 1: Add/Sub cloned both operands
// ---------------------------------------------------------------------------

#[test]
fn add_same_shape_allocates_only_the_output_buffer() {
    warm_up();
    let a = iota();
    let b = iota();

    let (sum, alloc) = measure(|| &a + &b);

    assert_eq!(
        alloc.count, 1,
        "same-shape `&a + &b` must allocate exactly the output buffer \
         (it used to clone both operands first: 4 allocations)"
    );
    assert_eq!(
        alloc.bytes, TENSOR_BYTES,
        "same-shape `&a + &b` must allocate exactly one tensor's worth of bytes \
         (it used to allocate 3x that)"
    );
    assert_eq!(sum[&[1, 2, 3]], 2.0 * a[&[1, 2, 3]]);
}

#[test]
fn sub_same_shape_allocates_only_the_output_buffer() {
    warm_up();
    let a = iota();
    let b = iota();

    let (diff, alloc) = measure(|| &a - &b);

    assert_eq!(alloc.count, 1, "same-shape `&a - &b` must allocate once");
    assert_eq!(alloc.bytes, TENSOR_BYTES);
    assert_eq!(diff[&[1, 2, 3]], 0.0);
}

#[test]
fn broadcasting_add_still_works_and_only_materializes_the_broadcast_operand() {
    warm_up();
    let a = iota(); // [32, 32, 32]
    let b = DenseND::<f64>::from_elem(&[32], 1.0); // broadcasts along the last axis

    let (sum, alloc) = measure(|| &a + &b);

    assert_eq!(sum.shape(), &[32, 32, 32]);
    assert_eq!(sum[&[0, 0, 0]], a[&[0, 0, 0]] + 1.0);
    assert_eq!(sum[&[31, 31, 31]], a[&[31, 31, 31]] + 1.0);
    // Exactly two full-size buffers may be allocated here: the materialized
    // broadcast of `b`, and the output. The operand that already has the target
    // shape (`a`) must be *borrowed*, not cloned — cloning it would push this to
    // three. (The `+ 4096` slack covers the handful of small index/shape
    // scratch vectors.)
    assert!(
        alloc.bytes < 2 * TENSOR_BYTES + 4096,
        "a broadcasting add must materialize only the broadcast operand and the \
         output; the same-shape operand must not be cloned. Allocated {} bytes \
         for a {TENSOR_BYTES}-byte tensor",
        alloc.bytes
    );
}

// ---------------------------------------------------------------------------
// Bug 2: unfold copied the tensor twice
// ---------------------------------------------------------------------------

#[test]
fn unfold_copies_the_tensor_exactly_once() {
    warm_up();
    let tensor = iota();

    for mode in 0..3 {
        let (unfolded, alloc) = measure(|| tensor.unfold(mode).expect("mode in range"));

        assert_eq!(unfolded.len(), ELEMS);
        assert!(
            alloc.bytes >= TENSOR_BYTES,
            "mode {mode}: unfold must materialize the output matrix"
        );
        // One full-tensor buffer + a handful of bytes for the permutation vector.
        // Before the fix this was exactly 2x TENSOR_BYTES (permute cloned, then
        // reshape re-gathered the non-contiguous clone).
        assert!(
            alloc.bytes < TENSOR_BYTES + 4096,
            "mode {mode}: unfold must copy the tensor exactly once, but it \
             allocated {} bytes for a {TENSOR_BYTES}-byte tensor",
            alloc.bytes
        );
        assert!(
            alloc.count <= 3,
            "mode {mode}: unfold made {} allocations; expected the output buffer \
             plus at most the small permutation scratch vector",
            alloc.count
        );
    }
}

// ---------------------------------------------------------------------------
// Bug 3: permute clones O(n) where permuted_axes is O(1)
// ---------------------------------------------------------------------------

#[test]
fn permute_view_does_not_allocate() {
    warm_up();
    let tensor = iota();

    let (view, alloc) = measure(|| tensor.permute_view(&[2, 0, 1]).expect("valid permutation"));

    assert_eq!(view.shape(), &[32, 32, 32]);
    assert_eq!(
        alloc.count, 0,
        "permute_view rewrites stride metadata only; it must not allocate"
    );
    assert_eq!(alloc.bytes, 0);
}

#[test]
fn into_permuted_does_not_allocate() {
    warm_up();
    let tensor = iota();
    let expected = tensor[&[1, 2, 3]];

    let (permuted, alloc) = measure(|| tensor.into_permuted(&[2, 0, 1]).expect("valid"));

    assert_eq!(permuted[&[3, 1, 2]], expected);
    assert_eq!(
        alloc.count, 0,
        "into_permuted moves the buffer and rewrites strides; it must not allocate"
    );
    assert_eq!(alloc.bytes, 0);
}

#[test]
fn permute_allocates_exactly_one_buffer() {
    // `permute(&self)` returns an owned, independent tensor, so one O(n) copy is
    // inherent to its contract. What it must *not* do is allocate anything on
    // top of that (the axis-validation scratch buffer is now stack-allocated).
    warm_up();
    let tensor = iota();

    let (permuted, alloc) = measure(|| tensor.permute(&[2, 0, 1]).expect("valid permutation"));

    assert_eq!(permuted.shape(), &[32, 32, 32]);
    assert_eq!(
        alloc.count, 1,
        "permute must allocate exactly the one buffer it copies into"
    );
    assert_eq!(alloc.bytes, TENSOR_BYTES);
}

// ---------------------------------------------------------------------------
// Bug 4: reshape was never zero-copy despite its docstring
// ---------------------------------------------------------------------------

#[test]
fn into_reshape_of_contiguous_tensor_does_not_allocate() {
    warm_up();
    let tensor = iota();

    let (reshaped, alloc) = measure(|| tensor.into_reshape(&[1024, 32]).expect("size preserved"));

    assert_eq!(reshaped.shape(), &[1024, 32]);
    assert_eq!(
        alloc.count, 0,
        "into_reshape of a contiguous tensor rewrites metadata and moves the \
         buffer; it must not allocate"
    );
    assert_eq!(alloc.bytes, 0);
}

#[test]
fn reshape_copies_once_as_documented() {
    // The borrowing `reshape` cannot be zero-copy (it must return an independent
    // owner). Its docstring now says so; this pins the measured cost.
    warm_up();
    let tensor = iota();

    let (reshaped, alloc) = measure(|| tensor.reshape(&[1024, 32]).expect("size preserved"));

    assert_eq!(reshaped.shape(), &[1024, 32]);
    assert_eq!(alloc.count, 1);
    assert_eq!(alloc.bytes, TENSOR_BYTES);
}

#[test]
fn into_reshape_of_non_contiguous_tensor_copies_once() {
    warm_up();
    let permuted = iota().into_permuted(&[2, 0, 1]).expect("valid permutation");
    assert!(!permuted.is_contiguous());

    let (reshaped, alloc) = measure(|| permuted.into_reshape(&[1024, 32]).expect("size preserved"));

    assert_eq!(reshaped.shape(), &[1024, 32]);
    assert_eq!(
        alloc.count, 1,
        "a non-contiguous buffer must be gathered exactly once"
    );
    assert_eq!(alloc.bytes, TENSOR_BYTES);
}

// ---------------------------------------------------------------------------
// Bug 5: sum/mean used a naive serial fold
// ---------------------------------------------------------------------------

#[test]
fn sum_and_mean_do_not_allocate() {
    warm_up();
    let tensor = iota();

    let (total, alloc) = measure(|| tensor.sum());
    assert_eq!(
        alloc.count, 0,
        "sum is a pure reduction; it must not allocate"
    );
    let expected: f64 = (0..ELEMS).map(|i| i as f64).sum();
    assert!((total - expected).abs() < 1e-6);

    let (mean, alloc) = measure(|| tensor.mean());
    assert_eq!(
        alloc.count, 0,
        "mean is a pure reduction; it must not allocate"
    );
    assert!((mean - expected / ELEMS as f64).abs() < 1e-9);
}
