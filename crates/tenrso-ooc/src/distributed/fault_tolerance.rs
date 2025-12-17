//! Fault tolerance and recovery for distributed out-of-core processing.
//!
//! Provides node failure detection, chunk replication, and automatic recovery.

use super::network::NetworkClient;
use super::protocol::{HeartbeatRequest, NodeId};
use super::registry::DistributedRegistry;
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Replication policy for chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationPolicy {
    /// No replication (risky)
    None,
    /// Fixed replication factor
    Fixed(usize),
    /// Dynamic replication based on access patterns
    Dynamic { min: usize, max: usize },
}

impl Default for ReplicationPolicy {
    fn default() -> Self {
        Self::Fixed(3)
    }
}

/// Configuration for fault tolerance.
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Node failure timeout (no heartbeat)
    pub failure_timeout: Duration,
    /// Replication policy
    pub replication_policy: ReplicationPolicy,
    /// Whether to auto-recover failed chunks
    pub auto_recovery: bool,
    /// Maximum recovery attempts per chunk
    pub max_recovery_attempts: usize,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(5),
            failure_timeout: Duration::from_secs(30),
            replication_policy: ReplicationPolicy::default(),
            auto_recovery: true,
            max_recovery_attempts: 3,
        }
    }
}

/// Statistics for fault tolerance.
#[derive(Debug, Clone, Default)]
pub struct FaultToleranceStats {
    /// Total heartbeats sent
    pub heartbeats_sent: u64,
    /// Total heartbeats received
    pub heartbeats_received: u64,
    /// Number of detected node failures
    pub node_failures: u64,
    /// Number of recovered chunks
    pub chunks_recovered: u64,
    /// Number of failed recoveries
    pub recovery_failures: u64,
    /// Current number of healthy nodes
    pub healthy_nodes: usize,
    /// Current number of failed nodes
    pub failed_nodes: usize,
}

/// Node health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHealth {
    /// Node is healthy and responsive
    Healthy,
    /// Node is suspected to be unhealthy
    Suspected,
    /// Node has failed
    Failed,
}

/// Node health information.
#[derive(Debug, Clone)]
struct NodeHealthInfo {
    /// Node ID
    #[allow(dead_code)]
    node_id: NodeId,
    /// Current health status
    status: NodeHealth,
    /// Last heartbeat timestamp
    last_heartbeat: Instant,
    /// Number of consecutive failed heartbeats
    failed_heartbeats: usize,
}

impl NodeHealthInfo {
    fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            status: NodeHealth::Healthy,
            last_heartbeat: Instant::now(),
            failed_heartbeats: 0,
        }
    }
}

/// Fault tolerance manager.
pub struct FaultToleranceManager {
    config: FaultToleranceConfig,
    registry: Arc<DistributedRegistry>,
    network_client: Arc<NetworkClient>,
    node_health: Arc<RwLock<HashMap<NodeId, NodeHealthInfo>>>,
    stats: Arc<FaultToleranceStatsInternal>,
    running: Arc<RwLock<bool>>,
}

struct FaultToleranceStatsInternal {
    heartbeats_sent: AtomicU64,
    heartbeats_received: AtomicU64,
    node_failures: AtomicU64,
    chunks_recovered: AtomicU64,
    recovery_failures: AtomicU64,
}

impl FaultToleranceStatsInternal {
    fn new() -> Self {
        Self {
            heartbeats_sent: AtomicU64::new(0),
            heartbeats_received: AtomicU64::new(0),
            node_failures: AtomicU64::new(0),
            chunks_recovered: AtomicU64::new(0),
            recovery_failures: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, healthy_nodes: usize, failed_nodes: usize) -> FaultToleranceStats {
        FaultToleranceStats {
            heartbeats_sent: self.heartbeats_sent.load(Ordering::Relaxed),
            heartbeats_received: self.heartbeats_received.load(Ordering::Relaxed),
            node_failures: self.node_failures.load(Ordering::Relaxed),
            chunks_recovered: self.chunks_recovered.load(Ordering::Relaxed),
            recovery_failures: self.recovery_failures.load(Ordering::Relaxed),
            healthy_nodes,
            failed_nodes,
        }
    }
}

impl FaultToleranceManager {
    /// Create a new fault tolerance manager.
    pub fn new(
        config: FaultToleranceConfig,
        registry: Arc<DistributedRegistry>,
        network_client: Arc<NetworkClient>,
    ) -> Self {
        Self {
            config,
            registry,
            network_client,
            node_health: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(FaultToleranceStatsInternal::new()),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the fault tolerance manager.
    pub async fn start(&self, local_node_id: NodeId) -> Result<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(anyhow!("Fault tolerance manager already running"));
            }
            *running = true;
        }

        // Initialize node health for all known nodes
        for node_id in self.registry.all_nodes() {
            let mut health = self.node_health.write();
            health.insert(node_id, NodeHealthInfo::new(node_id));
        }

        // Spawn heartbeat task
        let manager = self.clone_for_task();
        tokio::spawn(async move {
            manager.heartbeat_loop(local_node_id).await;
        });

        // Spawn failure detection task
        let manager = self.clone_for_task();
        tokio::spawn(async move {
            manager.failure_detection_loop().await;
        });

        Ok(())
    }

    /// Stop the fault tolerance manager.
    pub fn stop(&self) {
        let mut running = self.running.write();
        *running = false;
    }

    /// Clone for async tasks.
    fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            registry: Arc::clone(&self.registry),
            network_client: Arc::clone(&self.network_client),
            node_health: Arc::clone(&self.node_health),
            stats: Arc::clone(&self.stats),
            running: Arc::clone(&self.running),
        }
    }

    /// Heartbeat loop - send heartbeats to all nodes.
    async fn heartbeat_loop(&self, local_node_id: NodeId) {
        while *self.running.read() {
            // Get all nodes
            let nodes = self.registry.all_nodes();

            for node_id in nodes {
                if node_id == local_node_id {
                    continue; // Skip self
                }

                // Get node info
                if let Some(node_info) = self.registry.get_node_info(node_id) {
                    // Send heartbeat
                    let request = HeartbeatRequest::new(local_node_id);

                    match self
                        .network_client
                        .send_heartbeat(node_info.address, request)
                        .await
                    {
                        Ok(_response) => {
                            // Update health
                            let mut health = self.node_health.write();
                            if let Some(info) = health.get_mut(&node_id) {
                                info.last_heartbeat = Instant::now();
                                info.failed_heartbeats = 0;
                                info.status = NodeHealth::Healthy;
                            }
                            self.stats
                                .heartbeats_received
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            // Failed heartbeat
                            let mut health = self.node_health.write();
                            if let Some(info) = health.get_mut(&node_id) {
                                info.failed_heartbeats += 1;
                                if info.failed_heartbeats >= 2 {
                                    info.status = NodeHealth::Suspected;
                                }
                            }
                        }
                    }

                    self.stats.heartbeats_sent.fetch_add(1, Ordering::Relaxed);
                }
            }

            sleep(self.config.heartbeat_interval).await;
        }
    }

    /// Failure detection loop - detect and handle node failures.
    async fn failure_detection_loop(&self) {
        while *self.running.read() {
            let now = Instant::now();
            let mut failed_nodes = Vec::new();

            {
                let mut health = self.node_health.write();
                for (node_id, info) in health.iter_mut() {
                    if info.status != NodeHealth::Failed {
                        let elapsed = now.duration_since(info.last_heartbeat);
                        if elapsed > self.config.failure_timeout {
                            info.status = NodeHealth::Failed;
                            failed_nodes.push(*node_id);
                            self.stats.node_failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }

            // Handle failed nodes
            for node_id in failed_nodes {
                if let Err(e) = self.handle_node_failure(node_id).await {
                    eprintln!("Error handling node failure {}: {}", node_id, e);
                }
            }

            sleep(self.config.heartbeat_interval).await;
        }
    }

    /// Handle a node failure.
    async fn handle_node_failure(&self, node_id: NodeId) -> Result<()> {
        // Unregister node from registry
        self.registry.unregister_node(node_id)?;

        // If auto-recovery is enabled, trigger recovery
        if self.config.auto_recovery {
            // TODO: Implement chunk re-replication
            // This would involve:
            // 1. Finding all chunks hosted on failed node
            // 2. Checking if they have other replicas
            // 3. Creating new replicas on healthy nodes if needed
        }

        Ok(())
    }

    /// Get node health status.
    pub fn get_node_health(&self, node_id: NodeId) -> Option<NodeHealth> {
        let health = self.node_health.read();
        health.get(&node_id).map(|info| info.status)
    }

    /// Get all healthy nodes.
    pub fn healthy_nodes(&self) -> Vec<NodeId> {
        let health = self.node_health.read();
        health
            .iter()
            .filter(|(_, info)| info.status == NodeHealth::Healthy)
            .map(|(node_id, _)| *node_id)
            .collect()
    }

    /// Get all failed nodes.
    pub fn failed_nodes(&self) -> Vec<NodeId> {
        let health = self.node_health.read();
        health
            .iter()
            .filter(|(_, info)| info.status == NodeHealth::Failed)
            .map(|(node_id, _)| *node_id)
            .collect()
    }

    /// Get current statistics.
    pub fn stats(&self) -> FaultToleranceStats {
        let health = self.node_health.read();
        let healthy = health
            .values()
            .filter(|info| info.status == NodeHealth::Healthy)
            .count();
        let failed = health
            .values()
            .filter(|info| info.status == NodeHealth::Failed)
            .count();

        self.stats.snapshot(healthy, failed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::network::{NetworkClient, NetworkConfig};
    use super::super::registry::{DistributedRegistry, RegistryConfig};
    use super::*;

    #[test]
    fn test_fault_tolerance_config() {
        let config = FaultToleranceConfig::default();
        assert!(config.heartbeat_interval.as_secs() > 0);
        assert!(config.auto_recovery);
    }

    #[test]
    fn test_replication_policy() {
        let policy1 = ReplicationPolicy::None;
        let policy2 = ReplicationPolicy::Fixed(3);
        let policy3 = ReplicationPolicy::Dynamic { min: 2, max: 5 };

        assert_ne!(policy1, policy2);
        assert_ne!(policy2, policy3);
    }

    #[test]
    fn test_node_health_info() {
        let info = NodeHealthInfo::new(NodeId(1));
        assert_eq!(info.status, NodeHealth::Healthy);
        assert_eq!(info.failed_heartbeats, 0);
    }

    #[tokio::test]
    async fn test_fault_tolerance_manager_creation() {
        let config = FaultToleranceConfig::default();
        let registry = Arc::new(DistributedRegistry::new(RegistryConfig::default()));
        let network_client = Arc::new(NetworkClient::new(NetworkConfig::default()));

        let manager = FaultToleranceManager::new(config, registry, network_client);

        let stats = manager.stats();
        assert_eq!(stats.heartbeats_sent, 0);
        assert_eq!(stats.node_failures, 0);
    }

    #[test]
    fn test_node_health_transitions() {
        let mut info = NodeHealthInfo::new(NodeId(1));

        assert_eq!(info.status, NodeHealth::Healthy);

        info.failed_heartbeats = 2;
        info.status = NodeHealth::Suspected;
        assert_eq!(info.status, NodeHealth::Suspected);

        info.status = NodeHealth::Failed;
        assert_eq!(info.status, NodeHealth::Failed);
    }
}
