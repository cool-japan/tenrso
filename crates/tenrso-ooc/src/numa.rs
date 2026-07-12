//! NUMA-aware Memory Allocation
//!
//! This module provides NUMA (Non-Uniform Memory Access) aware memory allocation
//! for large tensor buffers on multi-socket systems.
//!
//! # What is real, and what is not
//!
//! Everything here talks to the kernel; nothing is simulated:
//!
//! - **Topology detection** parses `/sys/devices/system/node` (node ids, per-node
//!   `MemTotal`/`MemFree`, CPU lists, and the ACPI SLIT `distance` table).
//! - **Allocation** goes through [`std::alloc::alloc_zeroed`] with a [`Layout`] that
//!   honors the requested alignment, and hands back an owned [`NumaBuffer`] that frees
//!   itself on `Drop`.
//! - **Placement** is applied with `mbind(2)` (Linux). The buffer records whether the
//!   kernel actually accepted the policy ([`NumaBuffer::binding_status`]), and the node
//!   its pages really landed on is read back from the kernel with `get_mempolicy(2)`
//!   ([`NumaBuffer::node`]). [`NumaBuffer::query_kernel_binding`] re-reads the kernel's
//!   own view of the policy in force on a live buffer at any time.
//! - **Local** means local: the calling thread's node is resolved with `getcpu(2)`
//!   (see [`current_numa_node`]), not assumed to be node 0.
//!
//! On platforms without the Linux memory-policy syscalls, allocation still returns real,
//! correctly aligned memory, but the requested policy is **not** applied and the buffer
//! reports [`BindingStatus::Unenforced`]. The policy is never silently claimed to be in
//! force. Such allocations are counted in [`NumaStats::failed_allocations`].
//!
//! # Page granularity
//!
//! `mbind(2)` operates on whole pages, so on Linux a buffer's allocation is padded up to
//! a multiple of the page size and aligned to at least one page. [`NumaBuffer::len`] is
//! the size you asked for; [`NumaBuffer::capacity`] is the padded size that is actually
//! reserved (and bound). This guarantees the pages the policy is applied to belong
//! exclusively to this buffer.
//!
//! # Example
//!
//! ```rust
//! use tenrso_ooc::numa::{NumaAllocator, NumaPolicy};
//!
//! let mut allocator = NumaAllocator::new();
//! println!("NUMA nodes: {}", allocator.get_topology().num_nodes);
//!
//! // 1 MiB of real, 4 KiB-aligned memory on the node the calling thread runs on.
//! let mut buffer = allocator
//!     .allocate(1024 * 1024, 4096, NumaPolicy::Local)
//!     .expect("NUMA allocation failed");
//!
//! buffer.as_mut_slice()[0] = 42;
//! assert_eq!(buffer.as_slice()[0], 42);
//!
//! // Where the kernel says the pages actually are, and whether it is enforcing the policy.
//! println!("node: {:?}, binding: {:?}", buffer.node(), buffer.binding_status());
//!
//! // Memory is released when `buffer` is dropped.
//! ```

use serde::{Deserialize, Serialize};
use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Raw Linux memory-policy syscalls
// ---------------------------------------------------------------------------

/// Thin, honest bindings to the Linux NUMA memory-policy syscalls.
///
/// glibc does not export `mbind`/`get_mempolicy`/`set_mempolicy` as functions (they live
/// in `libnuma`), so these go through `syscall(2)` directly.
#[cfg(target_os = "linux")]
mod sys {
    use std::io;

    /// `MPOL_DEFAULT`: use the system/process default policy for the range.
    pub const MPOL_DEFAULT: libc::c_ulong = 0;
    /// `MPOL_PREFERRED`: allocate on the given node if possible, fall back otherwise.
    pub const MPOL_PREFERRED: libc::c_ulong = 1;
    /// `MPOL_BIND`: allocate *only* from the given node set (strict).
    pub const MPOL_BIND: libc::c_ulong = 2;
    /// `MPOL_INTERLEAVE`: spread the range's pages round-robin over the node set.
    pub const MPOL_INTERLEAVE: libc::c_ulong = 3;

    /// `get_mempolicy` flag: return the node id (rather than the policy mode).
    pub const MPOL_F_NODE: libc::c_ulong = 1 << 0;
    /// `get_mempolicy` flag: query the policy of the range containing `addr`.
    pub const MPOL_F_ADDR: libc::c_ulong = 1 << 1;

    /// `mbind` flag: fail if an existing page cannot satisfy the policy.
    pub const MPOL_MF_STRICT: libc::c_ulong = 1 << 0;
    /// `mbind` flag: migrate already-faulted pages that violate the policy.
    pub const MPOL_MF_MOVE: libc::c_ulong = 1 << 1;

    /// Number of `u64` words in a node mask. 16 words = node ids `0..=1023`, which covers
    /// `MAX_NUMNODES` (`CONFIG_NODES_SHIFT` is at most 10 on all mainstream configs).
    pub const NODE_MASK_WORDS: usize = 16;
    /// Number of bits the kernel is told the node mask holds (`maxnode`).
    pub const NODE_MASK_BITS: libc::c_ulong = (NODE_MASK_WORDS * 64) as libc::c_ulong;

    /// A `maxnode`-sized node mask, as the kernel expects it.
    pub type NodeMask = [u64; NODE_MASK_WORDS];

    /// Build a node mask with one bit set per node id.
    ///
    /// Returns `None` if any node id is too large for the mask (`>= NODE_MASK_BITS`).
    pub fn node_mask(nodes: &[usize]) -> Option<NodeMask> {
        let mut mask: NodeMask = [0; NODE_MASK_WORDS];
        for &node in nodes {
            if node >= NODE_MASK_BITS as usize {
                return None;
            }
            mask[node / 64] |= 1u64 << (node % 64);
        }
        Some(mask)
    }

    /// Decode a node mask returned by the kernel into a list of node ids.
    pub fn nodes_from_mask(mask: &NodeMask) -> Vec<usize> {
        let mut nodes = Vec::new();
        for (word_index, word) in mask.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                nodes.push(word_index * 64 + bit);
                bits &= bits - 1;
            }
        }
        nodes
    }

    /// `mbind(2)`: set the NUMA memory policy for the page range `[addr, addr + len)`.
    ///
    /// # Safety
    ///
    /// `addr` must be page-aligned and `[addr, addr + len)` must be a range the caller
    /// exclusively owns (whole pages), because the kernel rounds `len` up to a page
    /// boundary and applies the policy to every page it touches. Applying a policy to a
    /// page shared with another allocation would change that allocation's placement.
    pub unsafe fn mbind(
        addr: *mut u8,
        len: usize,
        mode: libc::c_ulong,
        nodemask: Option<&NodeMask>,
        flags: libc::c_ulong,
    ) -> io::Result<()> {
        let (mask_ptr, maxnode) = match nodemask {
            Some(mask) => (mask.as_ptr(), NODE_MASK_BITS),
            // MPOL_DEFAULT requires an empty node set: a NULL mask with maxnode = 0.
            None => (std::ptr::null(), 0),
        };

        // SAFETY: `SYS_mbind` takes (start, len, mode, nmask, maxnode, flags). `addr` is
        // page-aligned and owned by the caller (function precondition). `mask_ptr` either
        // points at a live `NodeMask` of exactly `NODE_MASK_WORDS` u64 words -- the kernel
        // reads `BITS_TO_LONGS(maxnode)` = `NODE_MASK_WORDS` words from it, so it never
        // reads out of bounds -- or is NULL with `maxnode == 0`, which the kernel accepts
        // for MPOL_DEFAULT. The syscall only reads through the pointers; it never retains
        // them. Violating this: passing a shorter mask than `maxnode` advertises would let
        // the kernel read past the end of the array.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                addr as *mut libc::c_void,
                len as libc::c_ulong,
                mode,
                mask_ptr,
                maxnode,
                flags,
            )
        };

        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// `get_mempolicy(2)` with `MPOL_F_NODE | MPOL_F_ADDR`: the node the page backing
    /// `addr` actually resides on, straight from the kernel.
    ///
    /// If the page has not been faulted in yet the kernel faults it in (as if the process
    /// had read `addr`) and reports where it placed it.
    ///
    /// # Safety
    ///
    /// `addr` must point into a live, readable mapping owned by this process.
    pub unsafe fn resident_node(addr: *const u8) -> io::Result<usize> {
        let mut node: libc::c_int = -1;

        // SAFETY: `SYS_get_mempolicy` takes (policy, nmask, maxnode, addr, flags). We pass
        // a pointer to a live `c_int` for the result and a NULL node mask with maxnode = 0
        // (permitted: with MPOL_F_NODE the node id is returned through `policy`, and the
        // kernel skips the mask entirely when `nmask` is NULL). `addr` is a live mapping
        // (function precondition). The kernel only writes `node`.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_get_mempolicy,
                &mut node as *mut libc::c_int,
                std::ptr::null_mut::<u64>(),
                0 as libc::c_ulong,
                addr as *const libc::c_void,
                MPOL_F_NODE | MPOL_F_ADDR,
            )
        };

        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        if node < 0 {
            return Err(io::Error::other(format!(
                "get_mempolicy returned a negative node id ({node})"
            )));
        }
        Ok(node as usize)
    }

    /// `get_mempolicy(2)` with `MPOL_F_ADDR`: the policy mode and node set the kernel has
    /// in force for the range containing `addr`.
    ///
    /// # Safety
    ///
    /// `addr` must point into a live mapping owned by this process.
    pub unsafe fn policy_of(addr: *const u8) -> io::Result<(i32, Vec<usize>)> {
        let mut mode: libc::c_int = -1;
        let mut mask: NodeMask = [0; NODE_MASK_WORDS];

        // SAFETY: same syscall as `resident_node`, but asking for the mode and the node
        // mask. `mask` is a live array of exactly `NODE_MASK_WORDS` u64 words and we tell
        // the kernel `maxnode = NODE_MASK_BITS`, so it writes at most `NODE_MASK_WORDS`
        // words -- no out-of-bounds write. `addr` is a live mapping (precondition).
        let rc = unsafe {
            libc::syscall(
                libc::SYS_get_mempolicy,
                &mut mode as *mut libc::c_int,
                mask.as_mut_ptr(),
                NODE_MASK_BITS,
                addr as *const libc::c_void,
                MPOL_F_ADDR,
            )
        };

        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((mode as i32, nodes_from_mask(&mask)))
    }

    /// `getcpu(2)`: the CPU and NUMA node the calling thread is running on *right now*.
    pub fn current_cpu_and_node() -> io::Result<(usize, usize)> {
        let mut cpu: libc::c_uint = 0;
        let mut node: libc::c_uint = 0;

        // SAFETY: `SYS_getcpu` takes (cpu, node, tcache) and writes at most one `c_uint`
        // through each of the first two pointers, both of which point at live locals. The
        // third argument is the unused (since Linux 2.6.24) `tcache` pointer, passed NULL.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_getcpu,
                &mut cpu as *mut libc::c_uint,
                &mut node as *mut libc::c_uint,
                std::ptr::null_mut::<libc::c_void>(),
            )
        };

        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((cpu as usize, node as usize))
    }

    /// The system page size, from `sysconf(_SC_PAGESIZE)`.
    pub fn page_size() -> usize {
        // SAFETY: `sysconf` is thread-safe and `_SC_PAGESIZE` is a valid, always-supported
        // parameter; it takes no pointers. A negative return means "no limit / error",
        // which cannot happen for _SC_PAGESIZE, but we guard anyway.
        let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if value > 0 {
            value as usize
        } else {
            4096
        }
    }
}

/// The system page size (4096 on platforms where it cannot be queried).
fn page_size() -> usize {
    #[cfg(target_os = "linux")]
    {
        sys::page_size()
    }
    #[cfg(not(target_os = "linux"))]
    {
        4096
    }
}

/// Resolve the NUMA node the **calling thread** is currently running on.
///
/// On Linux this is `getcpu(2)`. Returns `None` where the kernel cannot be asked
/// (non-Linux platforms, or a failing syscall).
///
/// Note that a thread can migrate between CPUs at any time, so this is a snapshot: it is
/// the node that was local at the moment of the call.
pub fn current_numa_node() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        sys::current_cpu_and_node().ok().map(|(_cpu, node)| node)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Policies, topology, statistics
// ---------------------------------------------------------------------------

/// NUMA allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NumaPolicy {
    /// Allocate on the same NUMA node as the calling thread (default).
    ///
    /// The node is resolved with [`current_numa_node`] and applied with `MPOL_PREFERRED`,
    /// so the kernel may still fall back to another node under memory pressure.
    #[default]
    Local,

    /// Interleave the buffer's pages round-robin across all NUMA nodes
    /// (`MPOL_INTERLEAVE` over the full node set).
    ///
    /// This distributes memory *bandwidth*, and is the right policy for a large tensor
    /// read by threads on every node. Because the pages of one buffer are spread over
    /// several nodes, [`NumaBuffer::node`] reports only where the *first* page landed.
    Interleaved,

    /// Prefer a specific NUMA node (`MPOL_PREFERRED`).
    ///
    /// If that node does not have enough free memory for the request, the allocator picks
    /// the node with the most free memory instead and counts a
    /// [`NumaStats::failed_allocations`].
    Preferred(usize),

    /// Bind strictly to a specific NUMA node (`MPOL_BIND | MPOL_MF_STRICT`).
    ///
    /// Unlike every other policy this one is *strict*: if the kernel refuses the binding,
    /// the allocation fails with [`NumaError::BindFailed`] rather than silently returning
    /// unbound memory.
    Bind(usize),

    /// Allocate on the node with the most free memory, accounting for the allocations this
    /// allocator already has outstanding (`MPOL_PREFERRED` on the chosen node).
    Balanced,

    /// Apply the system default policy to the range (`MPOL_DEFAULT`).
    Default,
}

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNode {
    /// Node ID
    pub node_id: usize,

    /// Total memory capacity in bytes, from `/sys/devices/system/node/nodeN/meminfo`
    /// (0 if the kernel does not expose it).
    pub total_memory_bytes: usize,

    /// Free memory in bytes, from `/sys/devices/system/node/nodeN/meminfo` at the time
    /// the topology was detected (0 if the kernel does not expose it).
    ///
    /// This is a snapshot: call [`NumaAllocator::refresh_topology`] to re-read it.
    pub free_memory_bytes: usize,

    /// Number of CPUs on this node
    pub num_cpus: usize,

    /// CPU IDs on this node
    pub cpu_ids: Vec<usize>,

    /// Relative distance to other nodes (`node_id -> distance`), from the ACPI SLIT table
    /// exposed at `/sys/devices/system/node/nodeN/distance`. 10 conventionally means
    /// "local".
    ///
    /// Empty when the kernel does not expose a distance table. No distances are invented.
    pub distances: HashMap<usize, usize>,
}

impl NumaNode {
    /// Create a new NUMA node
    pub fn new(node_id: usize) -> Self {
        Self {
            node_id,
            total_memory_bytes: 0,
            free_memory_bytes: 0,
            num_cpus: 0,
            cpu_ids: Vec::new(),
            distances: HashMap::new(),
        }
    }

    /// Memory usage as a fraction (0.0 to 1.0)
    pub fn memory_usage(&self) -> f64 {
        if self.total_memory_bytes == 0 {
            return 0.0;
        }
        let used = self
            .total_memory_bytes
            .saturating_sub(self.free_memory_bytes);
        used as f64 / self.total_memory_bytes as f64
    }
}

/// NUMA topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,

    /// NUMA nodes, sorted by [`NumaNode::node_id`]
    pub nodes: Vec<NumaNode>,

    /// Whether this system actually has more than one NUMA node
    pub numa_available: bool,

    /// Total system memory
    pub total_memory_bytes: usize,

    /// Number of CPUs in the system
    pub num_cpus: usize,
}

impl NumaTopology {
    /// Create a new (empty) topology
    pub fn new() -> Self {
        Self {
            num_nodes: 0,
            nodes: Vec::new(),
            numa_available: false,
            total_memory_bytes: 0,
            num_cpus: 0,
        }
    }

    /// Get a node by its **node id** (not by index: node ids need not be contiguous).
    pub fn get_node(&self, node_id: usize) -> Option<&NumaNode> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }

    /// All node ids, ascending.
    pub fn node_ids(&self) -> Vec<usize> {
        self.nodes.iter().map(|node| node.node_id).collect()
    }

    /// Get the node with the most free memory
    pub fn most_free_node(&self) -> Option<usize> {
        self.nodes
            .iter()
            .max_by_key(|node| node.free_memory_bytes)
            .map(|node| node.node_id)
    }

    /// Get the node with the least memory usage
    pub fn least_used_node(&self) -> Option<usize> {
        self.nodes
            .iter()
            .min_by(|a, b| {
                a.memory_usage()
                    .partial_cmp(&b.memory_usage())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|node| node.node_id)
    }

    /// The node that owns `cpu`, according to the per-node CPU lists read from sysfs.
    pub fn node_for_cpu(&self, cpu: usize) -> Option<usize> {
        self.nodes
            .iter()
            .find(|node| node.cpu_ids.contains(&cpu))
            .map(|node| node.node_id)
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// NUMA allocation statistics.
///
/// Every counter here describes **real allocations** made through a [`NumaAllocator`],
/// except [`NumaStats::local_accesses`] / [`NumaStats::remote_accesses`], which are
/// caller-reported instrumentation (see [`NumaAllocator::record_access`]) -- TenRSo does
/// not intercept individual loads and stores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaStats {
    /// Number of allocations placed on each node, keyed by the node the kernel says the
    /// memory actually landed on. Cumulative: it does not decrease when buffers are freed.
    pub allocations_per_node: HashMap<usize, usize>,

    /// Bytes allocated on each node. Cumulative; see [`NumaAllocator::live_bytes_per_node`]
    /// for the bytes currently outstanding.
    pub bytes_per_node: HashMap<usize, usize>,

    /// Accesses the caller reported as coming from a *different* node than the one the
    /// allocation lives on.
    pub remote_accesses: usize,

    /// Accesses the caller reported as coming from the *same* node as the allocation.
    pub local_accesses: usize,

    /// Allocations whose requested policy could not be honored: a
    /// [`NumaPolicy::Preferred`] node that had to fall back to another node, a buffer the
    /// kernel would not apply the policy to (reported as
    /// [`BindingStatus::Unenforced`]), or an allocation that failed outright.
    pub failed_allocations: usize,
}

impl NumaStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            allocations_per_node: HashMap::new(),
            bytes_per_node: HashMap::new(),
            remote_accesses: 0,
            local_accesses: 0,
            failed_allocations: 0,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, node_id: usize, bytes: usize) {
        *self.allocations_per_node.entry(node_id).or_insert(0) += 1;
        *self.bytes_per_node.entry(node_id).or_insert(0) += bytes;
    }

    /// Record an access
    pub fn record_access(&mut self, is_local: bool) {
        if is_local {
            self.local_accesses += 1;
        } else {
            self.remote_accesses += 1;
        }
    }

    /// Get locality ratio (local_accesses / total_accesses)
    pub fn locality_ratio(&self) -> f64 {
        let total = self.local_accesses + self.remote_accesses;
        if total == 0 {
            return 0.0;
        }
        self.local_accesses as f64 / total as f64
    }

    /// Get total allocations
    pub fn total_allocations(&self) -> usize {
        self.allocations_per_node.values().sum()
    }

    /// Get total bytes allocated
    pub fn total_bytes(&self) -> usize {
        self.bytes_per_node.values().sum()
    }
}

impl Default for NumaStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by [`NumaAllocator::allocate`].
#[derive(Debug, Error)]
pub enum NumaError {
    /// A zero-sized allocation was requested.
    #[error("zero-sized NUMA allocation requested")]
    ZeroSize,

    /// The (size, alignment) pair cannot form a valid [`Layout`].
    #[error(
        "invalid layout: size={size}, alignment={alignment} \
         (alignment must be a non-zero power of two, and the page-padded size must not overflow)"
    )]
    InvalidLayout {
        /// Requested size in bytes.
        size: usize,
        /// Requested alignment in bytes.
        alignment: usize,
    },

    /// The global allocator could not satisfy the request.
    #[error("out of memory: could not allocate {size} bytes with {alignment}-byte alignment")]
    OutOfMemory {
        /// Padded size in bytes that was requested from the global allocator.
        size: usize,
        /// Effective alignment in bytes.
        alignment: usize,
    },

    /// The topology has no nodes at all.
    #[error("no NUMA nodes detected")]
    NoNodes,

    /// A policy named a node this system does not have.
    #[error("NUMA node {requested} does not exist (present nodes: {present:?})")]
    NoSuchNode {
        /// The node id the policy asked for.
        requested: usize,
        /// The node ids this system actually has.
        present: Vec<usize>,
    },

    /// `mbind(2)` refused a strict [`NumaPolicy::Bind`].
    #[error("mbind(2) failed to bind {len} bytes at {addr:#x} to node(s) {nodes:?}: {source}")]
    BindFailed {
        /// Address of the range the kernel refused.
        addr: usize,
        /// Length of the range the kernel refused.
        len: usize,
        /// Node ids that were requested.
        nodes: Vec<usize>,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// `get_mempolicy(2)` failed while reading back the kernel's view of a buffer.
    #[error("get_mempolicy(2) failed: {0}")]
    QueryFailed(#[source] std::io::Error),

    /// This platform has no NUMA memory-policy API.
    #[error("NUMA memory policies are not supported on this platform")]
    Unsupported,
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

/// Whether the kernel is actually enforcing the requested policy on a [`NumaBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingStatus {
    /// `mbind(2)` succeeded: the kernel has the requested policy in force for this range.
    Enforced,

    /// The policy was **not** applied. The buffer is ordinary, correctly aligned memory,
    /// placed wherever the system default policy put it.
    Unenforced(UnenforcedReason),
}

/// Why a NUMA policy could not be applied to a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnenforcedReason {
    /// This platform has no NUMA memory-policy API (anything other than Linux).
    PlatformUnsupported,

    /// The kernel rejected `mbind(2)` -- most commonly `ENOSYS` on a kernel built without
    /// `CONFIG_NUMA`. Carries the raw errno.
    KernelRejected(i32),
}

/// The memory policy the kernel reports for a live buffer: its own view, read back with
/// `get_mempolicy(2)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelBinding {
    /// The policy mode in force on the range.
    pub mode: KernelMode,

    /// The node set the policy applies to (empty for [`KernelMode::Default`]).
    pub nodes: Vec<usize>,

    /// The node the buffer's first page actually resides on.
    pub first_page_node: usize,
}

/// A memory-policy mode as reported by the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelMode {
    /// `MPOL_DEFAULT`
    Default,
    /// `MPOL_PREFERRED`
    Preferred,
    /// `MPOL_BIND`
    Bind,
    /// `MPOL_INTERLEAVE`
    Interleave,
    /// A mode this crate does not model (e.g. `MPOL_LOCAL`, `MPOL_PREFERRED_MANY`).
    Other(i32),
}

impl KernelMode {
    #[cfg(target_os = "linux")]
    fn from_raw(mode: i32) -> Self {
        match mode as libc::c_ulong {
            sys::MPOL_DEFAULT => KernelMode::Default,
            sys::MPOL_PREFERRED => KernelMode::Preferred,
            sys::MPOL_BIND => KernelMode::Bind,
            sys::MPOL_INTERLEAVE => KernelMode::Interleave,
            _ => KernelMode::Other(mode),
        }
    }
}

/// Per-node byte counters shared between a [`NumaAllocator`] and the buffers it handed
/// out, so that a buffer dropped by the caller still updates the allocator's view of what
/// is outstanding.
#[derive(Debug)]
struct LiveBytes {
    per_node: Vec<AtomicUsize>,
}

impl LiveBytes {
    fn new(max_node_id: usize) -> Self {
        Self {
            per_node: (0..=max_node_id).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    fn add(&self, node_id: usize, bytes: usize) {
        if let Some(counter) = self.per_node.get(node_id) {
            counter.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    fn sub(&self, node_id: usize, bytes: usize) {
        if let Some(counter) = self.per_node.get(node_id) {
            // `fetch_update` keeps the counter from wrapping if the same buffer were
            // somehow accounted twice; it saturates at zero instead.
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes))
            });
        }
    }

    fn get(&self, node_id: usize) -> usize {
        self.per_node
            .get(node_id)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
}

/// An owned, page-aligned, NUMA-placed memory buffer.
///
/// The memory is real: it is obtained from the global allocator with a [`Layout`] that
/// honors the requested alignment, it is zero-initialized, and it is freed when the
/// `NumaBuffer` is dropped. On Linux the buffer's pages carry the NUMA policy that was
/// requested (see [`BindingStatus`]), applied with `mbind(2)`.
pub struct NumaBuffer {
    /// Start of the allocation. Always non-null and aligned to `layout.align()`.
    ptr: NonNull<u8>,

    /// The layout the memory was allocated with. On Linux `layout.size()` is the requested
    /// size padded up to whole pages, and `layout.align()` is at least one page.
    layout: Layout,

    /// The size the caller asked for (`<= layout.size()`).
    len: usize,

    /// The policy that was requested for this buffer.
    policy: NumaPolicy,

    /// Node ids the policy resolved to (one node, or every node for
    /// [`NumaPolicy::Interleaved`]; empty for [`NumaPolicy::Default`]).
    requested_nodes: Vec<usize>,

    /// The node the first page really resides on, read back from the kernel.
    /// `None` where the kernel cannot be asked.
    resident_node: Option<usize>,

    /// Whether the kernel accepted the policy.
    binding: BindingStatus,

    /// Live-byte accounting shared with the allocator that produced this buffer.
    live: Option<Arc<LiveBytes>>,

    /// The node the live-byte accounting was charged to.
    charged_node: usize,

    /// Whether `mbind(2)` was asked to migrate pages that already violate the policy
    /// (`MPOL_MF_MOVE`). See [`NumaBuffer::allocate_raw`] for why this is not free.
    migrate: bool,
}

// SAFETY: `NumaBuffer` uniquely owns its allocation -- the `NonNull<u8>` is never aliased
// (there is no `Clone`, and `as_mut_slice` requires `&mut self`), and the memory is plain
// zero-initialized bytes with no thread affinity. Moving it to another thread, or sharing
// `&NumaBuffer` (which only permits reads), is therefore sound, exactly as it is for
// `Box<[u8]>`. The NUMA policy is a property of the pages, not of the owning thread, so it
// survives the move unchanged. Violating this: handing out an aliased raw pointer and
// mutating through it from two threads at once.
unsafe impl Send for NumaBuffer {}
// SAFETY: see the `Send` justification above -- `&NumaBuffer` grants read-only access to
// immutable metadata and to `as_slice`, and `LiveBytes` is atomic.
unsafe impl Sync for NumaBuffer {}

impl NumaBuffer {
    /// Allocate `size_bytes` of zeroed memory with at least `alignment`-byte alignment and
    /// apply `policy` over `nodes`.
    ///
    /// `migrate` asks `mbind(2)` to migrate pages that already exist and violate the policy
    /// (`MPOL_MF_MOVE`). It is *not* free: entering the kernel's migration path costs a
    /// fixed ~80 us per call on this class of machine (it drains every CPU's LRU cache via
    /// an IPI), against ~1.4 us for the same `mbind` without it. The caller
    /// ([`NumaAllocator::allocate`]) therefore only asks for migration when it can actually
    /// change the outcome -- that is, when the system has more than one node. On a
    /// single-node system every page is already on the only node in the policy's node mask,
    /// so migration is provably a no-op and is skipped.
    fn allocate_raw(
        size_bytes: usize,
        alignment: usize,
        policy: NumaPolicy,
        nodes: Vec<usize>,
        migrate: bool,
    ) -> Result<Self, NumaError> {
        if size_bytes == 0 {
            return Err(NumaError::ZeroSize);
        }
        if !alignment.is_power_of_two() {
            return Err(NumaError::InvalidLayout {
                size: size_bytes,
                alignment,
            });
        }

        // `mbind(2)` works on whole pages, so pad the allocation up to a page multiple and
        // align it to at least a page. That way the pages the policy is applied to belong
        // exclusively to this buffer and we cannot disturb a neighbouring allocation that
        // happens to share a page.
        let page = page_size();
        let effective_alignment = alignment.max(page);
        let padded_size =
            size_bytes
                .checked_next_multiple_of(page)
                .ok_or(NumaError::InvalidLayout {
                    size: size_bytes,
                    alignment,
                })?;

        let layout = Layout::from_size_align(padded_size, effective_alignment).map_err(|_| {
            NumaError::InvalidLayout {
                size: size_bytes,
                alignment,
            }
        })?;

        // SAFETY: `layout` has a non-zero size (`padded_size >= page >= 1`, since
        // `size_bytes > 0`), which is `alloc_zeroed`'s precondition. The returned pointer
        // is either null (handled immediately below) or points to `layout.size()` bytes of
        // freshly zeroed, uniquely owned memory aligned to `layout.align()`. We store the
        // exact same `layout` in `self.layout` and pass it back to `dealloc` in `Drop`, as
        // required. Violating this: deallocating with a different layout, or reading the
        // memory before it is zeroed.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };

        let ptr = NonNull::new(raw).ok_or(NumaError::OutOfMemory {
            size: padded_size,
            alignment: effective_alignment,
        })?;

        let mut buffer = Self {
            ptr,
            layout,
            len: size_bytes,
            policy,
            requested_nodes: nodes,
            resident_node: None,
            binding: BindingStatus::Unenforced(UnenforcedReason::PlatformUnsupported),
            live: None,
            charged_node: 0,
            migrate,
        };

        buffer.apply_policy()?;
        buffer.charged_node = buffer
            .resident_node
            .or_else(|| buffer.requested_nodes.first().copied())
            .unwrap_or(0);

        Ok(buffer)
    }

    /// Apply the requested policy to the freshly allocated pages, then read back where the
    /// kernel actually put them.
    #[cfg(target_os = "linux")]
    fn apply_policy(&mut self) -> Result<(), NumaError> {
        // Migrate already-resident pages that violate the policy, but only where that can
        // change anything (see `allocate_raw`: it costs ~80 us, and on a single-node system
        // there is nowhere for a page to move to).
        let move_flag = if self.migrate { sys::MPOL_MF_MOVE } else { 0 };

        let (mode, mask, flags) = match self.policy {
            NumaPolicy::Bind(_) => (
                sys::MPOL_BIND,
                sys::node_mask(&self.requested_nodes),
                // Strict: every page in the range must end up on the requested node --
                // migrating the ones that are not there already -- or the call fails and
                // the allocation fails with it. This is the one policy that guarantees
                // placement rather than merely asking for it.
                sys::MPOL_MF_STRICT | move_flag,
            ),
            NumaPolicy::Interleaved => (
                sys::MPOL_INTERLEAVE,
                sys::node_mask(&self.requested_nodes),
                move_flag,
            ),
            NumaPolicy::Local | NumaPolicy::Preferred(_) | NumaPolicy::Balanced => (
                sys::MPOL_PREFERRED,
                sys::node_mask(&self.requested_nodes),
                move_flag,
            ),
            // MPOL_DEFAULT takes an empty node set. Applying it explicitly is not a no-op:
            // it clears any policy inherited from a VMA the global allocator recycled.
            NumaPolicy::Default => (sys::MPOL_DEFAULT, None, move_flag),
        };

        // A node id too large for the kernel's node mask cannot be a real node.
        if !matches!(self.policy, NumaPolicy::Default) && mask.is_none() {
            return Err(NumaError::NoSuchNode {
                requested: self.requested_nodes.first().copied().unwrap_or(0),
                present: Vec::new(),
            });
        }

        // SAFETY: `self.ptr` is page-aligned (the layout's alignment is at least one page)
        // and `self.layout.size()` is a whole number of pages that this buffer exclusively
        // owns, which is exactly `mbind`'s precondition: the kernel rounds the length up to
        // a page boundary, and every page it touches is ours. `mask`, when `Some`, is a
        // live `NodeMask` for the duration of the call.
        let result = unsafe {
            sys::mbind(
                self.ptr.as_ptr(),
                self.layout.size(),
                mode,
                mask.as_ref(),
                flags,
            )
        };

        match result {
            Ok(()) => {
                self.binding = BindingStatus::Enforced;
            }
            Err(error) => {
                // A strict Bind that the kernel refuses is a hard failure: returning
                // unbound memory to a caller who asked for a strict binding would be a lie.
                if matches!(self.policy, NumaPolicy::Bind(_)) {
                    return Err(NumaError::BindFailed {
                        addr: self.ptr.as_ptr() as usize,
                        len: self.layout.size(),
                        nodes: self.requested_nodes.clone(),
                        source: error,
                    });
                }
                // Every other policy is advisory: keep the (real, usable) memory, but say
                // plainly that the policy is not in force.
                self.binding = BindingStatus::Unenforced(UnenforcedReason::KernelRejected(
                    error.raw_os_error().unwrap_or(0),
                ));
            }
        }

        // Ask the kernel where the memory actually is. With MPOL_F_NODE | MPOL_F_ADDR this
        // faults the first page in and reports the node it was placed on -- the kernel's
        // own view, not our assumption.
        //
        // SAFETY: `self.ptr` points at a live, readable, zero-initialized mapping we own.
        self.resident_node = unsafe { sys::resident_node(self.ptr.as_ptr()) }.ok();

        Ok(())
    }

    /// Fallback for platforms with no NUMA memory-policy API: the memory is real and
    /// correctly aligned, but no policy is applied and none is claimed to be.
    #[cfg(not(target_os = "linux"))]
    fn apply_policy(&mut self) -> Result<(), NumaError> {
        self.binding = BindingStatus::Unenforced(UnenforcedReason::PlatformUnsupported);
        self.resident_node = None;
        Ok(())
    }

    /// The buffer's contents (`len` bytes, zero-initialized at allocation).
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` is non-null, aligned, and owned by this buffer; the first
        // `self.len` bytes are within the `self.layout.size()` bytes we allocated
        // (`len <= layout.size()`) and were zero-initialized by `alloc_zeroed`, so they are
        // all initialized `u8`s. The returned slice borrows `self`, so it cannot outlive
        // the allocation, and `&self` rules out a concurrent `&mut` alias.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The buffer's contents, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`, plus: `&mut self` guarantees this is the only live
        // reference to the allocation, so the returned `&mut [u8]` is unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Raw pointer to the start of the buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Raw mutable pointer to the start of the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// The number of bytes that were requested.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty. A `NumaBuffer` is never empty: a zero-sized request is
    /// rejected with [`NumaError::ZeroSize`].
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of bytes actually reserved: [`NumaBuffer::len`] padded up to whole pages,
    /// because `mbind(2)` binds whole pages.
    pub fn capacity(&self) -> usize {
        self.layout.size()
    }

    /// The alignment the memory really has (at least the requested alignment, and on Linux
    /// at least one page).
    pub fn alignment(&self) -> usize {
        self.layout.align()
    }

    /// The policy this buffer was allocated with.
    pub fn policy(&self) -> NumaPolicy {
        self.policy
    }

    /// The node ids the policy resolved to.
    pub fn requested_nodes(&self) -> &[usize] {
        &self.requested_nodes
    }

    /// The node the buffer's **first page** actually resides on, as reported by the kernel
    /// at allocation time. `None` where the kernel could not be asked.
    ///
    /// For [`NumaPolicy::Interleaved`] the remaining pages are spread over the other nodes;
    /// use [`NumaBuffer::query_kernel_binding`] to see the full node set.
    pub fn node(&self) -> Option<usize> {
        self.resident_node
    }

    /// Whether the kernel is enforcing the requested policy on these pages.
    pub fn binding_status(&self) -> BindingStatus {
        self.binding
    }

    /// Re-read the kernel's own view of the policy in force on this buffer *right now*.
    ///
    /// This is a live `get_mempolicy(2)` query, not a cached copy of what was requested: it
    /// is how you verify the pages really went where they were asked to go.
    pub fn query_kernel_binding(&self) -> Result<KernelBinding, NumaError> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: `self.ptr` points at a live, readable mapping owned by this process
            // for as long as `self` is alive, which is `policy_of`/`resident_node`'s
            // precondition. Both calls only read through the pointer.
            let (mode, nodes) =
                unsafe { sys::policy_of(self.ptr.as_ptr()) }.map_err(NumaError::QueryFailed)?;
            let first_page_node =
                unsafe { sys::resident_node(self.ptr.as_ptr()) }.map_err(NumaError::QueryFailed)?;

            Ok(KernelBinding {
                mode: KernelMode::from_raw(mode),
                nodes,
                first_page_node,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(NumaError::Unsupported)
        }
    }
}

impl std::fmt::Debug for NumaBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NumaBuffer")
            .field("addr", &format_args!("{:#x}", self.ptr.as_ptr() as usize))
            .field("len", &self.len)
            .field("capacity", &self.layout.size())
            .field("alignment", &self.layout.align())
            .field("policy", &self.policy)
            .field("requested_nodes", &self.requested_nodes)
            .field("resident_node", &self.resident_node)
            .field("binding", &self.binding)
            .finish()
    }
}

impl Drop for NumaBuffer {
    fn drop(&mut self) {
        if let Some(live) = &self.live {
            live.sub(self.charged_node, self.len);
        }

        // Hand the pages back to the global allocator with the system default policy rather
        // than with ours still attached: the allocator may recycle this VMA for an
        // unrelated allocation, which would otherwise silently inherit our binding.
        // Best-effort -- there is nothing useful to do if the kernel refuses, and `Drop`
        // must not panic.
        #[cfg(target_os = "linux")]
        {
            if matches!(self.binding, BindingStatus::Enforced)
                && !matches!(self.policy, NumaPolicy::Default)
            {
                // SAFETY: same preconditions as in `apply_policy` -- `self.ptr` is
                // page-aligned and the range is a whole number of pages this buffer still
                // exclusively owns (we have not deallocated yet). MPOL_DEFAULT takes a NULL
                // node mask.
                let _ = unsafe {
                    sys::mbind(
                        self.ptr.as_ptr(),
                        self.layout.size(),
                        sys::MPOL_DEFAULT,
                        None,
                        0,
                    )
                };
            }
        }

        // SAFETY: `self.ptr` came from `alloc_zeroed` with exactly `self.layout` in
        // `allocate_raw` and has not been deallocated (this is `Drop`, which runs once).
        // Deallocating with the same pointer and the same layout is `dealloc`'s
        // precondition. Violating this: a double free, or passing a different layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

// ---------------------------------------------------------------------------
// Allocator
// ---------------------------------------------------------------------------

/// The node set a policy resolved to, plus whether we had to settle for a different node
/// than the caller asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeTarget {
    nodes: Vec<usize>,
    fell_back: bool,
}

/// Resolve a policy to the node set to bind to.
///
/// Pure function of its inputs (`local_node` is passed in rather than queried) so that the
/// multi-node behaviour can be tested deterministically on any machine.
///
/// `outstanding` reports the bytes this allocator already has live on a node, so that a
/// capacity decision reflects allocations we have made ourselves and not just the sysfs
/// snapshot.
fn resolve_target(
    topology: &NumaTopology,
    size_bytes: usize,
    policy: NumaPolicy,
    local_node: Option<usize>,
    outstanding: &dyn Fn(usize) -> usize,
) -> Result<NodeTarget, NumaError> {
    if topology.nodes.is_empty() {
        return Err(NumaError::NoNodes);
    }

    let available = |node_id: usize| -> usize {
        topology
            .get_node(node_id)
            .map(|node| node.free_memory_bytes.saturating_sub(outstanding(node_id)))
            .unwrap_or(0)
    };

    // The node with the most room left, breaking ties towards the lowest node id.
    let most_available = || -> Option<usize> {
        topology
            .nodes
            .iter()
            .map(|node| node.node_id)
            .max_by_key(|&node_id| (available(node_id), std::cmp::Reverse(node_id)))
    };

    let first_node = topology.nodes[0].node_id;

    match policy {
        NumaPolicy::Local => {
            // The node the calling thread runs on -- if the kernel told us one we recognise.
            match local_node.filter(|&node| topology.get_node(node).is_some()) {
                Some(node) => Ok(NodeTarget {
                    nodes: vec![node],
                    fell_back: false,
                }),
                // We could not resolve the caller's node, so this allocation is not local to
                // anything we can name. Place it on the node with the most room and say so.
                None => Ok(NodeTarget {
                    nodes: vec![most_available().unwrap_or(first_node)],
                    fell_back: true,
                }),
            }
        }

        NumaPolicy::Interleaved => Ok(NodeTarget {
            nodes: topology.node_ids(),
            fell_back: false,
        }),

        NumaPolicy::Preferred(node_id) => {
            if topology.get_node(node_id).is_none() {
                return Err(NumaError::NoSuchNode {
                    requested: node_id,
                    present: topology.node_ids(),
                });
            }
            if available(node_id) >= size_bytes {
                Ok(NodeTarget {
                    nodes: vec![node_id],
                    fell_back: false,
                })
            } else {
                let fallback = most_available().unwrap_or(first_node);
                Ok(NodeTarget {
                    nodes: vec![fallback],
                    fell_back: fallback != node_id,
                })
            }
        }

        NumaPolicy::Bind(node_id) => {
            if topology.get_node(node_id).is_none() {
                return Err(NumaError::NoSuchNode {
                    requested: node_id,
                    present: topology.node_ids(),
                });
            }
            Ok(NodeTarget {
                nodes: vec![node_id],
                fell_back: false,
            })
        }

        NumaPolicy::Balanced => Ok(NodeTarget {
            nodes: vec![most_available().unwrap_or(first_node)],
            fell_back: false,
        }),

        NumaPolicy::Default => Ok(NodeTarget {
            nodes: Vec::new(),
            fell_back: false,
        }),
    }
}

/// NUMA-aware allocator.
///
/// Produces real, owned [`NumaBuffer`]s whose pages carry the requested NUMA policy, and
/// keeps [`NumaStats`] about the allocations it actually made.
///
/// Two APIs are offered over the same machinery:
///
/// - [`NumaAllocator::allocate`] returns an owned [`NumaBuffer`] that frees itself on drop.
///   Prefer this.
/// - [`NumaAllocator::allocate_aligned`] keeps the buffer inside the allocator and returns
///   an id for it, for callers that want the allocator to own the memory
///   ([`NumaAllocator::buffer`], [`NumaAllocator::take`], [`NumaAllocator::free`]).
pub struct NumaAllocator {
    /// NUMA topology
    topology: NumaTopology,

    /// Allocation statistics
    stats: NumaStats,

    /// Current policy
    policy: NumaPolicy,

    /// Buffers this allocator owns, by allocation id.
    allocations: HashMap<usize, NumaBuffer>,

    /// Next allocation id
    next_allocation_id: usize,

    /// Bytes currently live on each node, including buffers handed to callers.
    live: Arc<LiveBytes>,
}

impl NumaAllocator {
    /// Create a new NUMA allocator (detects the topology immediately).
    pub fn new() -> Self {
        let topology = detect_topology();
        let max_node_id = topology
            .nodes
            .iter()
            .map(|node| node.node_id)
            .max()
            .unwrap_or(0);

        Self {
            topology,
            stats: NumaStats::new(),
            policy: NumaPolicy::default(),
            allocations: HashMap::new(),
            next_allocation_id: 0,
            live: Arc::new(LiveBytes::new(max_node_id)),
        }
    }

    /// Detect the NUMA topology of this system.
    ///
    /// On Linux this parses `/sys/devices/system/node`. On systems without NUMA (or where
    /// the kernel does not expose it) it reports a single node covering all CPUs.
    pub fn detect_topology(&self) -> NumaTopology {
        detect_topology()
    }

    /// Re-read the per-node free-memory figures from the kernel.
    ///
    /// [`NumaNode::free_memory_bytes`] is a snapshot taken when the topology was detected;
    /// call this to refresh it (for example before a batch of [`NumaPolicy::Balanced`]
    /// allocations).
    pub fn refresh_topology(&mut self) {
        self.topology = detect_topology();
    }

    /// Set allocation policy
    pub fn set_policy(&mut self, policy: NumaPolicy) {
        self.policy = policy;
    }

    /// Get current policy
    pub fn get_policy(&self) -> NumaPolicy {
        self.policy
    }

    /// Get topology
    pub fn get_topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Get statistics
    pub fn get_stats(&self) -> &NumaStats {
        &self.stats
    }

    /// Bytes currently allocated on `node_id` and not yet freed, counting buffers this
    /// allocator handed out to callers as well as the ones it still owns.
    pub fn live_bytes_per_node(&self, node_id: usize) -> usize {
        self.live.get(node_id)
    }

    /// The node the calling thread is running on, as far as this allocator's topology knows.
    ///
    /// Falls back to the sysfs CPU -> node map if the kernel names a node that was not in
    /// the topology, and returns `None` if the node cannot be resolved at all.
    pub fn local_node(&self) -> Option<usize> {
        #[cfg(target_os = "linux")]
        {
            let (cpu, node) = sys::current_cpu_and_node().ok()?;
            if self.topology.get_node(node).is_some() {
                return Some(node);
            }
            self.topology.node_for_cpu(cpu)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// Allocate a real, owned, NUMA-placed buffer of `size_bytes` with at least
    /// `alignment`-byte alignment.
    ///
    /// The buffer is zero-initialized and frees itself when dropped. On Linux the requested
    /// policy is applied to its pages with `mbind(2)`; check
    /// [`NumaBuffer::binding_status`] to see whether the kernel accepted it, and
    /// [`NumaBuffer::node`] for where the pages actually landed.
    ///
    /// # Errors
    ///
    /// - [`NumaError::ZeroSize`] / [`NumaError::InvalidLayout`] for a nonsensical request.
    /// - [`NumaError::NoSuchNode`] if the policy names a node this system does not have.
    /// - [`NumaError::OutOfMemory`] if the global allocator cannot satisfy the request.
    /// - [`NumaError::BindFailed`] if a strict [`NumaPolicy::Bind`] is refused by the
    ///   kernel. Advisory policies do not fail here: they return a usable buffer that
    ///   reports [`BindingStatus::Unenforced`].
    ///
    /// # Cost
    ///
    /// Two syscalls per allocation, independent of size: one `mbind(2)` (~1.4 us) to apply
    /// the policy and one `get_mempolicy(2)` (~1.3 us) to read back where the pages
    /// actually went. `get_mempolicy` faults in the first page, so a lazily-zeroed buffer
    /// commits one page up front.
    ///
    /// On a **multi-node** system `mbind` additionally carries `MPOL_MF_MOVE`, so that a
    /// recycled region whose pages are already on the wrong node is corrected rather than
    /// quietly left in place. Entering the kernel's migration path costs a fixed ~80 us
    /// (it drains every CPU's LRU cache), which is the price of placement actually being
    /// true. On a single-node system that flag cannot change any outcome and is skipped.
    ///
    /// (Timings measured on a contended 8-CPU x86-64 box, 64 KiB allocations, median of 3.)
    pub fn allocate(
        &mut self,
        size_bytes: usize,
        alignment: usize,
        policy: NumaPolicy,
    ) -> Result<NumaBuffer, NumaError> {
        let live = Arc::clone(&self.live);
        let target = resolve_target(
            &self.topology,
            size_bytes,
            policy,
            self.local_node(),
            &|node_id| live.get(node_id),
        );

        let target = match target {
            Ok(target) => target,
            Err(error) => {
                self.stats.failed_allocations += 1;
                return Err(error);
            }
        };

        // Page migration can only ever move a page to a *different* node, so it is worth its
        // (substantial, ~80 us) cost only on a system that has one to move to.
        let migrate = self.topology.num_nodes > 1;

        let mut buffer =
            match NumaBuffer::allocate_raw(size_bytes, alignment, policy, target.nodes, migrate) {
                Ok(buffer) => buffer,
                Err(error) => {
                    self.stats.failed_allocations += 1;
                    return Err(error);
                }
            };

        // The policy was not honored if we had to pick a different node than asked, or if
        // the kernel would not enforce it. Either way the caller can see it in the stats.
        if target.fell_back || matches!(buffer.binding, BindingStatus::Unenforced(_)) {
            self.stats.failed_allocations += 1;
        }

        let node_id = buffer.charged_node;
        self.stats.record_allocation(node_id, size_bytes);
        self.live.add(node_id, size_bytes);
        buffer.live = Some(Arc::clone(&self.live));

        Ok(buffer)
    }

    /// Allocate a buffer that the **allocator** keeps ownership of, and return its id.
    ///
    /// Equivalent to [`NumaAllocator::allocate`] followed by storing the buffer in the
    /// allocator. Reach the memory with [`NumaAllocator::buffer`] /
    /// [`NumaAllocator::buffer_mut`], take ownership of it with [`NumaAllocator::take`], or
    /// release it with [`NumaAllocator::free`]. Any buffer still held is freed when the
    /// allocator is dropped.
    ///
    /// Returns `None` if the allocation failed; use [`NumaAllocator::allocate`] if you need
    /// to know why.
    pub fn allocate_aligned(
        &mut self,
        size_bytes: usize,
        alignment: usize,
        policy: NumaPolicy,
    ) -> Option<usize> {
        let buffer = self.allocate(size_bytes, alignment, policy).ok()?;

        let allocation_id = self.next_allocation_id;
        self.next_allocation_id += 1;
        self.allocations.insert(allocation_id, buffer);

        Some(allocation_id)
    }

    /// Borrow a buffer the allocator owns.
    pub fn buffer(&self, allocation_id: usize) -> Option<&NumaBuffer> {
        self.allocations.get(&allocation_id)
    }

    /// Mutably borrow a buffer the allocator owns.
    pub fn buffer_mut(&mut self, allocation_id: usize) -> Option<&mut NumaBuffer> {
        self.allocations.get_mut(&allocation_id)
    }

    /// Take ownership of a buffer out of the allocator.
    pub fn take(&mut self, allocation_id: usize) -> Option<NumaBuffer> {
        self.allocations.remove(&allocation_id)
    }

    /// Free a buffer the allocator owns, releasing its memory immediately.
    ///
    /// `size_bytes` must be the size the allocation was made with; it is checked against
    /// the buffer's recorded length so that a mismatched accounting bug cannot free the
    /// wrong amount. Returns `false` (and frees nothing) if the id is unknown or the size
    /// does not match.
    pub fn free(&mut self, allocation_id: usize, size_bytes: usize) -> bool {
        match self.allocations.get(&allocation_id) {
            Some(buffer) if buffer.len() == size_bytes => {
                // Dropping the buffer deallocates the memory and updates the live-byte
                // counters.
                self.allocations.remove(&allocation_id);
                true
            }
            _ => false,
        }
    }

    /// The node an allocation's memory really lives on.
    ///
    /// This is the node the kernel reported for the buffer's first page. On a platform
    /// without a NUMA API, where the kernel cannot be asked, it is the node the policy
    /// selected.
    pub fn get_allocation_node(&self, allocation_id: usize) -> Option<usize> {
        let buffer = self.allocations.get(&allocation_id)?;
        buffer
            .node()
            .or_else(|| buffer.requested_nodes().first().copied())
    }

    /// Record that a thread running on `accessing_node` touched an allocation.
    ///
    /// This is **caller-reported** instrumentation: TenRSo does not intercept loads and
    /// stores, so [`NumaStats::locality_ratio`] is only as good as the calls made here. The
    /// allocation it is compared against is real, and so is the node it lives on.
    ///
    /// See [`NumaAllocator::record_access_here`] to report an access from the node the
    /// calling thread is actually running on.
    pub fn record_access(&mut self, allocation_id: usize, accessing_node: usize) {
        if let Some(allocation_node) = self.get_allocation_node(allocation_id) {
            let is_local = allocation_node == accessing_node;
            self.stats.record_access(is_local);
        }
    }

    /// Record an access to an allocation from the node the **calling thread** is running on
    /// right now (resolved with `getcpu(2)`).
    ///
    /// Returns `false` if the allocation is unknown or the current node cannot be resolved,
    /// in which case nothing is recorded.
    pub fn record_access_here(&mut self, allocation_id: usize) -> bool {
        let Some(current) = self.local_node() else {
            return false;
        };
        let Some(allocation_node) = self.get_allocation_node(allocation_id) else {
            return false;
        };
        self.stats.record_access(allocation_node == current);
        true
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = NumaStats::new();
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Topology detection
// ---------------------------------------------------------------------------

/// Detect the NUMA topology of this system.
fn detect_topology() -> NumaTopology {
    #[cfg(target_os = "linux")]
    {
        detect_topology_linux().unwrap_or_else(detect_topology_fallback)
    }
    #[cfg(not(target_os = "linux"))]
    {
        detect_topology_fallback()
    }
}

/// Parse `/sys/devices/system/node`. Returns `None` if the kernel does not expose it.
#[cfg(target_os = "linux")]
fn detect_topology_linux() -> Option<NumaTopology> {
    use std::fs;
    use std::path::Path;

    let node_path = Path::new("/sys/devices/system/node");
    if !node_path.exists() {
        return None;
    }

    let mut nodes = Vec::new();
    for entry in fs::read_dir(node_path).ok()?.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let Some(node_id_str) = name_str.strip_prefix("node") else {
            continue;
        };
        let Ok(node_id) = node_id_str.parse::<usize>() else {
            continue;
        };

        let mut node = NumaNode::new(node_id);

        if let Ok(meminfo) = fs::read_to_string(entry.path().join("meminfo")) {
            let (total, free) = parse_node_meminfo(&meminfo);
            node.total_memory_bytes = total;
            node.free_memory_bytes = free;
        }

        if let Ok(cpulist) = fs::read_to_string(entry.path().join("cpulist")) {
            node.cpu_ids = parse_cpu_list(cpulist.trim());
            node.num_cpus = node.cpu_ids.len();
        }

        // The ACPI SLIT row for this node: one distance per node, in node-id order.
        if let Ok(distance) = fs::read_to_string(entry.path().join("distance")) {
            node.distances = parse_distances(distance.trim());
        }

        nodes.push(node);
    }

    if nodes.is_empty() {
        return None;
    }

    // `read_dir` yields entries in an arbitrary order; node ids must be sorted so that
    // callers iterating `topology.nodes` see them in a stable, predictable order.
    nodes.sort_by_key(|node| node.node_id);

    let mut topology = NumaTopology::new();
    topology.num_nodes = nodes.len();
    topology.numa_available = nodes.len() > 1;
    topology.total_memory_bytes = nodes.iter().map(|node| node.total_memory_bytes).sum();
    topology.num_cpus = nodes.iter().map(|node| node.num_cpus).sum();
    topology.nodes = nodes;

    Some(topology)
}

/// Single-node topology for systems that do not expose NUMA information.
fn detect_topology_fallback() -> NumaTopology {
    let num_cpus = num_cpus::get();

    let mut node = NumaNode::new(0);
    node.num_cpus = num_cpus;
    node.cpu_ids = (0..num_cpus).collect();
    node.distances.insert(0, 10);

    // Memory figures come from the kernel or not at all: an invented capacity would make
    // every capacity-driven policy decision a fiction. Zero means "unknown", and
    // `resolve_target` then simply has no capacity signal to act on.
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            let (total, free) = parse_proc_meminfo(&meminfo);
            node.total_memory_bytes = total;
            node.free_memory_bytes = free;
        }
    }

    let total_memory_bytes = node.total_memory_bytes;
    let num_cpus = node.num_cpus;

    NumaTopology {
        num_nodes: 1,
        nodes: vec![node],
        numa_available: false,
        total_memory_bytes,
        num_cpus,
    }
}

/// Parse `MemTotal` / `MemFree` (in bytes) out of a per-node sysfs `meminfo`.
///
/// Lines look like `Node 0 MemTotal:       46223480 kB`.
#[cfg(target_os = "linux")]
fn parse_node_meminfo(meminfo: &str) -> (usize, usize) {
    let mut total = 0usize;
    let mut free = 0usize;

    for line in meminfo.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        // ["Node", "<id>", "<key>:", "<value>", "kB"]
        if fields.len() < 4 {
            continue;
        }
        let Ok(value_kb) = fields[3].parse::<usize>() else {
            continue;
        };
        match fields[2] {
            "MemTotal:" => total = value_kb.saturating_mul(1024),
            "MemFree:" => free = value_kb.saturating_mul(1024),
            _ => {}
        }
    }

    (total, free)
}

/// Parse `MemTotal` / `MemAvailable` (in bytes) out of `/proc/meminfo`.
#[cfg(target_os = "linux")]
fn parse_proc_meminfo(meminfo: &str) -> (usize, usize) {
    let mut total = 0usize;
    let mut available = 0usize;

    for line in meminfo.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            continue;
        }
        let Ok(value_kb) = fields[1].parse::<usize>() else {
            continue;
        };
        match fields[0] {
            "MemTotal:" => total = value_kb.saturating_mul(1024),
            "MemAvailable:" => available = value_kb.saturating_mul(1024),
            _ => {}
        }
    }

    (total, available)
}

/// Parse a CPU list in sysfs format (e.g. `"0-3,8-11"`).
fn parse_cpu_list(cpulist: &str) -> Vec<usize> {
    let mut cpus = Vec::new();

    for part in cpulist.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }

    cpus
}

/// Parse a sysfs `distance` row (e.g. `"10 21"`) into `node_id -> distance`.
///
/// The i-th value is the distance from this node to node `i`.
fn parse_distances(distance: &str) -> HashMap<usize, usize> {
    distance
        .split_whitespace()
        .enumerate()
        .filter_map(|(node_id, value)| value.parse::<usize>().ok().map(|d| (node_id, d)))
        .collect()
}

#[cfg(test)]
mod tests;
