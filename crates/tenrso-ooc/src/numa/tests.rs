//! Unit and kernel-truth tests for the NUMA allocator.
//!
//! Split out of `numa.rs` to keep that file under the 2000-line source-size policy.

use super::*;

/// A synthetic two-node topology, so that the multi-node logic is exercised
/// deterministically even on a single-node machine.
fn synthetic_two_node_topology() -> NumaTopology {
    let mut node0 = NumaNode::new(0);
    node0.cpu_ids = vec![0, 1, 2, 3];
    node0.num_cpus = 4;
    node0.total_memory_bytes = 16 * 1024 * 1024 * 1024;
    node0.free_memory_bytes = 1024 * 1024; // nearly full
    node0.distances = HashMap::from([(0, 10), (1, 21)]);

    let mut node1 = NumaNode::new(1);
    node1.cpu_ids = vec![4, 5, 6, 7];
    node1.num_cpus = 4;
    node1.total_memory_bytes = 16 * 1024 * 1024 * 1024;
    node1.free_memory_bytes = 8 * 1024 * 1024 * 1024; // lots of room
    node1.distances = HashMap::from([(0, 21), (1, 10)]);

    NumaTopology {
        num_nodes: 2,
        nodes: vec![node0, node1],
        numa_available: true,
        total_memory_bytes: 32 * 1024 * 1024 * 1024,
        num_cpus: 8,
    }
}

fn no_outstanding(_node_id: usize) -> usize {
    0
}

// -- topology -----------------------------------------------------------

#[test]
fn test_numa_allocator_creation() {
    let allocator = NumaAllocator::new();
    assert!(allocator.get_topology().num_nodes > 0);
}

#[test]
fn test_topology_detection() {
    let allocator = NumaAllocator::new();
    let topology = allocator.get_topology();

    assert!(topology.num_nodes > 0);
    assert_eq!(topology.nodes.len(), topology.num_nodes);
    assert!(topology.num_cpus > 0);

    // Nodes must be sorted by id: `read_dir` does not promise an order.
    let ids = topology.node_ids();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);

    // Every node must be reachable by id (not merely by index).
    for id in ids {
        assert_eq!(
            topology.get_node(id).map(|node| node.node_id),
            Some(id),
            "get_node must look up by node id"
        );
    }
}

#[test]
fn test_topology_queries() {
    let allocator = NumaAllocator::new();
    let topology = allocator.get_topology();

    assert!(topology.most_free_node().is_some());
    assert!(topology.least_used_node().is_some());
    assert!(topology.get_node(topology.nodes[0].node_id).is_some());
    assert!(topology.get_node(usize::MAX).is_none());
}

#[test]
fn test_parse_cpu_list() {
    assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
    assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
    assert_eq!(parse_cpu_list("0-1,4-5"), vec![0, 1, 4, 5]);
    assert_eq!(parse_cpu_list("7"), vec![7]);
    assert!(parse_cpu_list("").is_empty());
}

#[test]
fn test_parse_distances() {
    assert_eq!(parse_distances("10"), HashMap::from([(0, 10)]));
    assert_eq!(parse_distances("10 21"), HashMap::from([(0, 10), (1, 21)]));
    assert!(parse_distances("").is_empty());
}

#[test]
#[cfg(target_os = "linux")]
fn test_parse_node_meminfo() {
    let meminfo = "Node 0 MemTotal:       46223480 kB\n\
                   Node 0 MemFree:          505292 kB\n\
                   Node 0 MemUsed:        45718188 kB\n";
    let (total, free) = parse_node_meminfo(meminfo);
    assert_eq!(total, 46_223_480 * 1024);
    assert_eq!(free, 505_292 * 1024);
}

#[test]
#[cfg(target_os = "linux")]
fn test_real_topology_matches_sysfs() {
    // The topology must describe *this* machine, so cross-check it against sysfs.
    let topology = detect_topology();
    if std::path::Path::new("/sys/devices/system/node/node0").exists() {
        let cpulist = std::fs::read_to_string("/sys/devices/system/node/node0/cpulist")
            .expect("node0 cpulist must be readable");
        let expected = parse_cpu_list(cpulist.trim());
        let node0 = topology.get_node(0).expect("node0 must be in the topology");
        assert_eq!(node0.cpu_ids, expected);
        assert!(
            node0.total_memory_bytes > 0,
            "node0 must report real memory from sysfs"
        );
    }
}

#[test]
fn test_node_for_cpu_synthetic() {
    let topology = synthetic_two_node_topology();
    assert_eq!(topology.node_for_cpu(0), Some(0));
    assert_eq!(topology.node_for_cpu(3), Some(0));
    assert_eq!(topology.node_for_cpu(4), Some(1));
    assert_eq!(topology.node_for_cpu(7), Some(1));
    assert_eq!(topology.node_for_cpu(99), None);
}

// -- policy -> node resolution (deterministic, multi-node) ---------------

#[test]
fn test_local_policy_selects_the_calling_threads_node() {
    let topology = synthetic_two_node_topology();

    // A thread on node 1 must get node 1 -- not node 0.
    let target = resolve_target(&topology, 4096, NumaPolicy::Local, Some(1), &no_outstanding)
        .expect("local resolution must succeed");
    assert_eq!(target.nodes, vec![1]);
    assert!(!target.fell_back);

    let target = resolve_target(&topology, 4096, NumaPolicy::Local, Some(0), &no_outstanding)
        .expect("local resolution must succeed");
    assert_eq!(target.nodes, vec![0]);

    // If the caller's node cannot be resolved we must admit it, not pretend it is node 0.
    let target = resolve_target(&topology, 4096, NumaPolicy::Local, None, &no_outstanding)
        .expect("local resolution must succeed");
    assert!(target.fell_back);
    assert_eq!(target.nodes, vec![1], "falls back to the emptiest node");
}

#[test]
fn test_interleaved_policy_covers_every_node() {
    let topology = synthetic_two_node_topology();
    let target = resolve_target(
        &topology,
        4096,
        NumaPolicy::Interleaved,
        Some(0),
        &no_outstanding,
    )
    .expect("interleave resolution must succeed");
    assert_eq!(target.nodes, vec![0, 1]);
}

#[test]
fn test_preferred_policy_falls_back_when_node_is_full() {
    let topology = synthetic_two_node_topology();

    // Node 0 has 1 MiB free: a 512 KiB request fits.
    let target = resolve_target(
        &topology,
        512 * 1024,
        NumaPolicy::Preferred(0),
        Some(0),
        &no_outstanding,
    )
    .expect("preferred resolution must succeed");
    assert_eq!(target.nodes, vec![0]);
    assert!(!target.fell_back);

    // A 2 GiB request does not: it must fall back to node 1 and admit the fallback.
    let target = resolve_target(
        &topology,
        2 * 1024 * 1024 * 1024,
        NumaPolicy::Preferred(0),
        Some(0),
        &no_outstanding,
    )
    .expect("preferred resolution must succeed");
    assert_eq!(target.nodes, vec![1]);
    assert!(target.fell_back);

    // A node that does not exist is an error, not a silent fallback.
    let error = resolve_target(
        &topology,
        4096,
        NumaPolicy::Preferred(7),
        Some(0),
        &no_outstanding,
    )
    .expect_err("a nonexistent node must be rejected");
    assert!(matches!(error, NumaError::NoSuchNode { requested: 7, .. }));
}

#[test]
fn test_outstanding_bytes_change_the_decision() {
    let topology = synthetic_two_node_topology();

    // With nothing outstanding, Balanced picks node 1 (8 GiB free vs 1 MiB).
    let target = resolve_target(
        &topology,
        4096,
        NumaPolicy::Balanced,
        Some(0),
        &no_outstanding,
    )
    .expect("balanced resolution must succeed");
    assert_eq!(target.nodes, vec![1]);

    // Once we have 8 GiB live on node 1, node 0's remaining 1 MiB is the better bet.
    let outstanding = |node_id: usize| -> usize {
        if node_id == 1 {
            8 * 1024 * 1024 * 1024
        } else {
            0
        }
    };
    let target = resolve_target(&topology, 4096, NumaPolicy::Balanced, Some(0), &outstanding)
        .expect("balanced resolution must succeed");
    assert_eq!(
        target.nodes,
        vec![0],
        "Balanced must account for what this allocator already has live"
    );
}

#[test]
fn test_bind_policy_is_strict_about_nodes() {
    let topology = synthetic_two_node_topology();

    let target = resolve_target(
        &topology,
        4096,
        NumaPolicy::Bind(1),
        Some(0),
        &no_outstanding,
    )
    .expect("bind resolution must succeed");
    assert_eq!(target.nodes, vec![1]);

    let error = resolve_target(
        &topology,
        4096,
        NumaPolicy::Bind(9),
        Some(0),
        &no_outstanding,
    )
    .expect_err("binding to a nonexistent node must fail");
    assert!(matches!(
        error,
        NumaError::NoSuchNode {
            requested: 9,
            ref present
        } if present == &vec![0, 1]
    ));
}

#[test]
fn test_default_policy_binds_no_node() {
    let topology = synthetic_two_node_topology();
    let target = resolve_target(
        &topology,
        4096,
        NumaPolicy::Default,
        Some(0),
        &no_outstanding,
    )
    .expect("default resolution must succeed");
    assert!(target.nodes.is_empty());
}

// -- real memory --------------------------------------------------------

#[test]
fn test_allocation_returns_real_usable_memory() {
    let mut allocator = NumaAllocator::new();
    let size = 256 * 1024;

    let mut buffer = allocator
        .allocate(size, 4096, NumaPolicy::Local)
        .expect("allocation must succeed");

    assert_eq!(buffer.len(), size);
    assert!(!buffer.is_empty());
    assert!(buffer.capacity() >= size);
    assert_eq!(
        buffer.capacity() % page_size(),
        0,
        "the reserved range must be a whole number of pages"
    );

    // Zero-initialized...
    assert!(buffer.as_slice().iter().all(|&byte| byte == 0));

    // ...and genuinely writable and readable back. An integer handle cannot do this.
    buffer.as_mut_slice().fill(0xAB);
    buffer.as_mut_slice()[size - 1] = 0xCD;
    assert!(buffer.as_slice()[..size - 1].iter().all(|&b| b == 0xAB));
    assert_eq!(buffer.as_slice()[size - 1], 0xCD);
}

#[test]
fn test_requested_alignment_is_honored() {
    let mut allocator = NumaAllocator::new();

    for alignment in [64usize, 4096, 64 * 1024, 2 * 1024 * 1024] {
        let buffer = allocator
            .allocate(8192, alignment, NumaPolicy::Local)
            .unwrap_or_else(|error| panic!("allocation at align {alignment} failed: {error}"));

        let addr = buffer.as_ptr() as usize;
        assert_eq!(
            addr % alignment,
            0,
            "pointer {addr:#x} is not {alignment}-byte aligned"
        );
        assert!(
            buffer.alignment() >= alignment,
            "effective alignment {} is weaker than the {alignment} requested",
            buffer.alignment()
        );
        // mbind needs page granularity, so the effective alignment is at least a page.
        assert_eq!(addr % page_size(), 0);
    }
}

#[test]
fn test_zero_sized_and_bad_alignment_are_rejected() {
    let mut allocator = NumaAllocator::new();

    assert!(matches!(
        allocator.allocate(0, 4096, NumaPolicy::Local),
        Err(NumaError::ZeroSize)
    ));
    assert!(matches!(
        allocator.allocate(1024, 3, NumaPolicy::Local),
        Err(NumaError::InvalidLayout { .. })
    ));
}

#[test]
fn test_distinct_allocations_do_not_overlap() {
    let mut allocator = NumaAllocator::new();

    let mut first = allocator
        .allocate(64 * 1024, 4096, NumaPolicy::Local)
        .expect("first allocation");
    let mut second = allocator
        .allocate(64 * 1024, 4096, NumaPolicy::Local)
        .expect("second allocation");

    first.as_mut_slice().fill(0x11);
    second.as_mut_slice().fill(0x22);

    assert!(first.as_slice().iter().all(|&b| b == 0x11));
    assert!(second.as_slice().iter().all(|&b| b == 0x22));
}

// -- the kernel's own view ----------------------------------------------

/// The kernel must confirm, for every policy, that the mode and node set we asked for
/// are the ones actually in force on the buffer's pages -- read back with
/// `get_mempolicy(2)`, not from anything this crate cached.
#[test]
#[cfg(target_os = "linux")]
fn test_kernel_confirms_the_policy_it_was_asked_for() {
    let mut allocator = NumaAllocator::new();
    let node_ids = allocator.get_topology().node_ids();
    let first_node = node_ids[0];

    let cases: Vec<(NumaPolicy, KernelMode)> = vec![
        (NumaPolicy::Bind(first_node), KernelMode::Bind),
        (NumaPolicy::Preferred(first_node), KernelMode::Preferred),
        (NumaPolicy::Local, KernelMode::Preferred),
        (NumaPolicy::Balanced, KernelMode::Preferred),
        (NumaPolicy::Interleaved, KernelMode::Interleave),
        (NumaPolicy::Default, KernelMode::Default),
    ];

    for (policy, expected_mode) in cases {
        let buffer = allocator
            .allocate(1024 * 1024, 4096, policy)
            .unwrap_or_else(|error| panic!("{policy:?} allocation failed: {error}"));

        assert_eq!(
            buffer.binding_status(),
            BindingStatus::Enforced,
            "{policy:?}: the kernel must have accepted mbind(2) on this box"
        );

        let kernel = buffer
            .query_kernel_binding()
            .unwrap_or_else(|error| panic!("{policy:?}: get_mempolicy failed: {error}"));

        assert_eq!(
            kernel.mode, expected_mode,
            "{policy:?}: the kernel reports {:?} for this range, not {expected_mode:?}",
            kernel.mode
        );

        if matches!(policy, NumaPolicy::Default) {
            assert!(
                kernel.nodes.is_empty(),
                "MPOL_DEFAULT must not carry a node set"
            );
        } else {
            assert_eq!(
                kernel.nodes,
                buffer.requested_nodes().to_vec(),
                "{policy:?}: the kernel's node set must be the one we requested"
            );
            assert!(
                kernel.nodes.contains(&kernel.first_page_node),
                "{policy:?}: the first page landed on node {} which is outside the \
                 requested set {:?}",
                kernel.first_page_node,
                kernel.nodes
            );
        }

        // The node recorded at allocation time is the kernel's, so it must still agree.
        assert_eq!(buffer.node(), Some(kernel.first_page_node));
    }
}

/// `Interleaved` must be a genuine `MPOL_INTERLEAVE` over *every* node, not a userspace
/// round-robin that pretends to be one.
#[test]
#[cfg(target_os = "linux")]
fn test_interleave_covers_every_node() {
    let mut allocator = NumaAllocator::new();
    let node_ids = allocator.get_topology().node_ids();

    let buffer = allocator
        .allocate(4 * 1024 * 1024, 4096, NumaPolicy::Interleaved)
        .expect("interleaved allocation");
    let kernel = buffer
        .query_kernel_binding()
        .expect("get_mempolicy must succeed");

    assert_eq!(kernel.mode, KernelMode::Interleave);
    assert_eq!(kernel.nodes, node_ids);
}

/// Independent confirmation from a second kernel interface: the buffer must appear in
/// `/proc/self/numa_maps` at its own address, with the policy we asked for.
#[test]
#[cfg(target_os = "linux")]
fn test_numa_maps_shows_the_binding() {
    let mut allocator = NumaAllocator::new();
    let first_node = allocator.get_topology().nodes[0].node_id;

    // 4 MiB is comfortably over glibc's mmap threshold, so this gets its own VMA and
    // therefore its own line in numa_maps.
    let buffer = allocator
        .allocate(4 * 1024 * 1024, 4096, NumaPolicy::Bind(first_node))
        .expect("bound allocation");

    let addr = buffer.as_ptr() as usize;
    let maps = std::fs::read_to_string("/proc/self/numa_maps")
        .expect("/proc/self/numa_maps must be readable");

    let line = maps
        .lines()
        .find(|line| {
            line.split_whitespace()
                .next()
                .and_then(|start| usize::from_str_radix(start, 16).ok())
                == Some(addr)
        })
        .unwrap_or_else(|| panic!("no numa_maps entry starts at our buffer address {addr:#x}"));

    assert!(
        line.contains(&format!("bind:{first_node}")),
        "numa_maps says `{line}`, which does not show the requested bind:{first_node}"
    );
}

/// A bind to a node that does not exist must fail loudly. This also proves the syscall
/// really reaches the kernel: the kernel is the thing rejecting it.
#[test]
fn test_bind_to_nonexistent_node_fails() {
    let mut allocator = NumaAllocator::new();
    let absent = allocator
        .get_topology()
        .node_ids()
        .iter()
        .max()
        .copied()
        .unwrap_or(0)
        + 64;

    let error = allocator
        .allocate(4096, 4096, NumaPolicy::Bind(absent))
        .expect_err("binding to a nonexistent node must fail");
    assert!(matches!(error, NumaError::NoSuchNode { .. }));

    // The failure is counted, not swallowed.
    assert_eq!(allocator.get_stats().failed_allocations, 1);
}

/// The kernel -- not us -- rejects an impossible node mask. If `mbind` were a no-op
/// stub, this would "succeed".
#[test]
#[cfg(target_os = "linux")]
fn test_kernel_rejects_an_impossible_node_mask() {
    let mut allocator = NumaAllocator::new();
    let mut buffer = allocator
        .allocate(64 * 1024, 4096, NumaPolicy::Default)
        .expect("allocation");

    let absurd_node = 900; // valid in the mask ABI, but no machine has this node
    let mask = sys::node_mask(&[absurd_node]).expect("node 900 fits in the mask");
    let capacity = buffer.capacity();

    // SAFETY: the range is exactly the buffer's own page-aligned, whole-page allocation,
    // which is `mbind`'s precondition. The call is expected to fail; the buffer's memory
    // is untouched either way.
    let result = unsafe {
        sys::mbind(
            buffer.as_mut_ptr(),
            capacity,
            sys::MPOL_BIND,
            Some(&mask),
            sys::MPOL_MF_STRICT,
        )
    };

    let error = result.expect_err("the kernel must reject a binding to node 900");
    assert_eq!(
        error.raw_os_error(),
        Some(libc::EINVAL),
        "expected EINVAL from the kernel, got {error}"
    );
}

// -- local resolution ---------------------------------------------------

#[test]
fn test_local_node_is_resolved_from_the_kernel() {
    let allocator = NumaAllocator::new();

    #[cfg(target_os = "linux")]
    {
        let node = allocator
            .local_node()
            .expect("getcpu(2) must resolve the calling thread's node on Linux");
        assert!(
            allocator.get_topology().get_node(node).is_some(),
            "getcpu reported node {node}, which is not in the detected topology"
        );

        // It must agree with the kernel's own answer, and it must be *derived*, not a
        // hardcoded 0: on this single-node box the true answer happens to be 0, so we
        // check the derivation instead.
        let (cpu, kernel_node) = sys::current_cpu_and_node().expect("getcpu");
        assert_eq!(node, kernel_node);
        assert_eq!(
            allocator.get_topology().node_for_cpu(cpu),
            Some(kernel_node),
            "the sysfs cpu->node map must agree with getcpu(2)"
        );
        assert_eq!(current_numa_node(), Some(kernel_node));
    }

    #[cfg(not(target_os = "linux"))]
    {
        assert!(allocator.local_node().is_none());
        let _ = allocator;
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_local_allocation_lands_on_the_current_node() {
    let mut allocator = NumaAllocator::new();
    let current = allocator.local_node().expect("current node");

    let buffer = allocator
        .allocate(1024 * 1024, 4096, NumaPolicy::Local)
        .expect("local allocation");

    assert_eq!(buffer.requested_nodes(), &[current]);
    assert_eq!(
        buffer.node(),
        Some(current),
        "a Local allocation's pages must be on the node the thread is running on"
    );
}

// -- accounting ---------------------------------------------------------

#[test]
fn test_live_bytes_track_real_buffers() {
    let mut allocator = NumaAllocator::new();
    let size = 128 * 1024;

    let first = allocator
        .allocate(size, 4096, NumaPolicy::Local)
        .expect("first");
    let node = first
        .node()
        .or_else(|| first.requested_nodes().first().copied())
        .expect("the buffer must be charged to a node");
    assert_eq!(allocator.live_bytes_per_node(node), size);

    let second = allocator
        .allocate(size, 4096, NumaPolicy::Bind(node))
        .expect("second");
    assert_eq!(allocator.live_bytes_per_node(node), 2 * size);

    // Dropping a buffer the *caller* owns must still decrement the allocator's view.
    drop(second);
    assert_eq!(allocator.live_bytes_per_node(node), size);

    drop(first);
    assert_eq!(allocator.live_bytes_per_node(node), 0);

    // Cumulative stats do not decrease.
    assert_eq!(allocator.get_stats().total_allocations(), 2);
    assert_eq!(allocator.get_stats().total_bytes(), 2 * size);
    assert_eq!(allocator.get_stats().failed_allocations, 0);
}

#[test]
fn test_registry_allocate_and_free() {
    let mut allocator = NumaAllocator::new();
    let size = 64 * 1024;

    let id = allocator
        .allocate_aligned(size, 4096, NumaPolicy::Local)
        .expect("allocation must succeed");

    let node = allocator
        .get_allocation_node(id)
        .expect("the allocation must have a node");
    assert_eq!(allocator.live_bytes_per_node(node), size);

    // The id names a real buffer we can write to.
    let buffer = allocator.buffer_mut(id).expect("buffer must exist");
    buffer.as_mut_slice().fill(0x5A);
    assert!(allocator
        .buffer(id)
        .expect("buffer")
        .as_slice()
        .iter()
        .all(|&byte| byte == 0x5A));

    // A mismatched size frees nothing.
    assert!(!allocator.free(id, size + 1));
    assert!(allocator.buffer(id).is_some());
    assert_eq!(allocator.live_bytes_per_node(node), size);

    // The right size really releases the memory.
    assert!(allocator.free(id, size));
    assert!(allocator.buffer(id).is_none());
    assert_eq!(allocator.live_bytes_per_node(node), 0);

    // Freeing twice is a no-op, not a double free.
    assert!(!allocator.free(id, size));
}

#[test]
fn test_take_transfers_ownership() {
    let mut allocator = NumaAllocator::new();
    let id = allocator
        .allocate_aligned(32 * 1024, 4096, NumaPolicy::Local)
        .expect("allocation");
    let node = allocator.get_allocation_node(id).expect("node");

    let mut buffer = allocator.take(id).expect("take must yield the buffer");
    assert!(allocator.buffer(id).is_none());
    assert_eq!(
        allocator.live_bytes_per_node(node),
        32 * 1024,
        "a taken buffer is still live"
    );

    buffer.as_mut_slice()[0] = 7;
    assert_eq!(buffer.as_slice()[0], 7);

    drop(buffer);
    assert_eq!(allocator.live_bytes_per_node(node), 0);
}

#[test]
fn test_dropping_the_allocator_frees_its_buffers() {
    let live = {
        let mut allocator = NumaAllocator::new();
        allocator
            .allocate_aligned(64 * 1024, 4096, NumaPolicy::Local)
            .expect("allocation");
        Arc::clone(&allocator.live)
    };
    // The allocator is gone; the buffers it owned went with it.
    assert_eq!(live.get(0), 0);
}

#[test]
fn test_access_tracking() {
    let mut allocator = NumaAllocator::new();

    let id = allocator
        .allocate_aligned(4096, 4096, NumaPolicy::Local)
        .expect("allocation");
    let node = allocator.get_allocation_node(id).expect("node");

    allocator.record_access(id, node);
    allocator.record_access(id, node + 1);

    let stats = allocator.get_stats();
    assert_eq!(stats.local_accesses, 1);
    assert_eq!(stats.remote_accesses, 1);
    assert_eq!(stats.locality_ratio(), 0.5);

    // An access reported from the node we are actually running on is local, because the
    // allocation really is on that node.
    #[cfg(target_os = "linux")]
    {
        assert!(allocator.record_access_here(id));
        assert_eq!(allocator.get_stats().local_accesses, 2);
    }

    // An unknown id records nothing.
    allocator.record_access(9999, 0);
    assert_eq!(
        allocator.get_stats().local_accesses + allocator.get_stats().remote_accesses,
        if cfg!(target_os = "linux") { 3 } else { 2 }
    );
}

#[test]
fn test_preferred_fallback_is_counted_as_a_failure() {
    // Drive the accounting through the same path `allocate` uses, with a topology whose
    // preferred node cannot hold the request.
    let topology = synthetic_two_node_topology();
    let target = resolve_target(
        &topology,
        2 * 1024 * 1024 * 1024,
        NumaPolicy::Preferred(0),
        Some(0),
        &no_outstanding,
    )
    .expect("resolution");

    assert!(target.fell_back);

    let mut stats = NumaStats::new();
    if target.fell_back {
        stats.failed_allocations += 1;
    }
    assert_eq!(stats.failed_allocations, 1);
}

#[test]
fn test_numa_stats() {
    let mut stats = NumaStats::new();

    stats.record_allocation(0, 1024);
    stats.record_allocation(1, 2048);
    stats.record_access(true);
    stats.record_access(false);

    assert_eq!(stats.total_allocations(), 2);
    assert_eq!(stats.total_bytes(), 3072);
    assert_eq!(stats.local_accesses, 1);
    assert_eq!(stats.remote_accesses, 1);
    assert_eq!(stats.locality_ratio(), 0.5);
}

#[test]
fn test_reset_stats() {
    let mut allocator = NumaAllocator::new();
    allocator
        .allocate_aligned(4096, 4096, NumaPolicy::Local)
        .expect("allocation");
    assert_eq!(allocator.get_stats().total_allocations(), 1);

    allocator.reset_stats();
    assert_eq!(allocator.get_stats().total_allocations(), 0);
}

#[test]
fn test_policy_accessors() {
    let mut allocator = NumaAllocator::new();
    assert_eq!(allocator.get_policy(), NumaPolicy::Local);
    allocator.set_policy(NumaPolicy::Interleaved);
    assert_eq!(allocator.get_policy(), NumaPolicy::Interleaved);
}

// -- node mask ABI ------------------------------------------------------

#[test]
#[cfg(target_os = "linux")]
fn test_node_mask_round_trip() {
    let mask = sys::node_mask(&[0, 1, 63, 64, 127]).expect("mask must fit");
    assert_eq!(sys::nodes_from_mask(&mask), vec![0, 1, 63, 64, 127]);
    assert_eq!(mask[0], 0b11 | (1u64 << 63));
    assert_eq!(mask[1], 0b1 | (1u64 << 63));

    assert!(sys::node_mask(&[]).is_some());
    assert!(
        sys::node_mask(&[sys::NODE_MASK_BITS as usize]).is_none(),
        "a node id past the end of the mask must be rejected, not silently truncated"
    );
}
