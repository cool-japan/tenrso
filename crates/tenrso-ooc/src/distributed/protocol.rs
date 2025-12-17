//! Protocol definitions for distributed chunk communication.
//!
//! Uses bincode for efficient binary serialization.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::net::SocketAddr;

/// Unique identifier for a node in the distributed system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Create a new node ID from a socket address hash.
    pub fn from_addr(addr: &SocketAddr) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        addr.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Get the raw u64 value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Information about a node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier
    pub id: NodeId,
    /// Network address
    pub address: SocketAddr,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Available disk space in bytes
    pub available_disk: u64,
    /// Network bandwidth in bytes/sec
    pub bandwidth_bps: u64,
    /// Current load (0.0 - 1.0)
    pub load: f64,
    /// Number of chunks hosted
    pub num_chunks: usize,
}

impl NodeInfo {
    /// Create a new node info.
    pub fn new(id: NodeId, address: SocketAddr) -> Self {
        Self {
            id,
            address,
            available_memory: 0,
            available_disk: 0,
            bandwidth_bps: 0,
            load: 0.0,
            num_chunks: 0,
        }
    }
}

/// Metadata for a chunk in the distributed system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Chunk identifier (hash of shape + dtype + index)
    pub chunk_id: String,
    /// Shape of the chunk
    pub shape: Vec<usize>,
    /// Data type (e.g., "f64", "f32")
    pub dtype: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Whether chunk is compressed
    pub compressed: bool,
    /// Compression codec if compressed
    pub compression_codec: Option<String>,
    /// Checksum for integrity verification
    pub checksum: u64,
    /// Unix timestamp of creation
    pub created_at: u64,
    /// Unix timestamp of last access
    pub last_accessed: u64,
}

impl ChunkMetadata {
    /// Create chunk metadata for f64 data.
    pub fn new_f64(chunk_id: String, shape: Vec<usize>) -> Self {
        let size_bytes = shape.iter().product::<usize>() as u64 * 8; // f64 = 8 bytes
        Self {
            chunk_id,
            shape,
            dtype: "f64".to_string(),
            size_bytes,
            compressed: false,
            compression_codec: None,
            checksum: 0,
            created_at: current_timestamp(),
            last_accessed: current_timestamp(),
        }
    }

    /// Update the checksum from data.
    pub fn compute_checksum(&mut self, data: &[u8]) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        self.checksum = hasher.finish();
    }

    /// Verify checksum matches data.
    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish() == self.checksum
    }
}

/// Location of a chunk in the distributed system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkLocation {
    /// Node hosting the chunk
    pub node_id: NodeId,
    /// Path on the node's filesystem
    pub path: String,
    /// Whether chunk is in memory or on disk
    pub in_memory: bool,
}

/// Message types for distributed communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Request a chunk from a remote node
    ChunkRequest(ChunkRequest),
    /// Response with chunk data
    ChunkResponse(ChunkResponse),
    /// Heartbeat to check node health
    Heartbeat(HeartbeatRequest),
    /// Heartbeat response
    HeartbeatResponse(HeartbeatResponse),
    /// Register a new node
    RegisterNode(NodeInfo),
    /// Acknowledge node registration
    RegisterAck(NodeId),
    /// Replicate chunk to another node
    ReplicateChunk {
        chunk_id: String,
        target_node: NodeId,
    },
    /// Acknowledge chunk replication
    ReplicateAck { chunk_id: String, success: bool },
}

/// Request to fetch a chunk from a remote node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRequest {
    /// ID of the chunk being requested
    pub chunk_id: String,
    /// Node making the request
    pub requester: NodeId,
    /// Whether to prefetch (don't count as access)
    pub is_prefetch: bool,
    /// Request timestamp
    pub timestamp: u64,
}

impl ChunkRequest {
    /// Create a new chunk request.
    pub fn new(chunk_id: String, requester: NodeId) -> Self {
        Self {
            chunk_id,
            requester,
            is_prefetch: false,
            timestamp: current_timestamp(),
        }
    }

    /// Create a prefetch request.
    pub fn prefetch(chunk_id: String, requester: NodeId) -> Self {
        Self {
            chunk_id,
            requester,
            is_prefetch: true,
            timestamp: current_timestamp(),
        }
    }
}

/// Response containing chunk data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResponse {
    /// ID of the chunk
    pub chunk_id: String,
    /// Metadata
    pub metadata: ChunkMetadata,
    /// Raw chunk data (possibly compressed)
    pub data: Vec<u8>,
    /// Whether request succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Response timestamp
    pub timestamp: u64,
}

impl ChunkResponse {
    /// Create a successful response.
    pub fn success(chunk_id: String, metadata: ChunkMetadata, data: Vec<u8>) -> Self {
        Self {
            chunk_id,
            metadata,
            data,
            success: true,
            error: None,
            timestamp: current_timestamp(),
        }
    }

    /// Create a failed response.
    pub fn error(chunk_id: String, error: String) -> Self {
        Self {
            chunk_id,
            metadata: ChunkMetadata::new_f64(String::new(), vec![]),
            data: Vec::new(),
            success: false,
            error: Some(error),
            timestamp: current_timestamp(),
        }
    }
}

/// Heartbeat request to check node health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    /// Node sending the heartbeat
    pub sender: NodeId,
    /// Timestamp
    pub timestamp: u64,
}

impl HeartbeatRequest {
    /// Create a new heartbeat request.
    pub fn new(sender: NodeId) -> Self {
        Self {
            sender,
            timestamp: current_timestamp(),
        }
    }
}

/// Heartbeat response with node status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    /// Node responding
    pub node: NodeId,
    /// Current node info
    pub info: NodeInfo,
    /// Timestamp
    pub timestamp: u64,
}

impl HeartbeatResponse {
    /// Create a new heartbeat response.
    pub fn new(info: NodeInfo) -> Self {
        Self {
            node: info.id,
            info,
            timestamp: current_timestamp(),
        }
    }
}

/// Get current Unix timestamp in seconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_from_addr() {
        let addr1: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:8001".parse().unwrap();

        let id1 = NodeId::from_addr(&addr1);
        let id2 = NodeId::from_addr(&addr2);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_chunk_metadata_checksum() {
        let mut metadata = ChunkMetadata::new_f64("test_chunk".to_string(), vec![10, 20]);
        let data = vec![1u8, 2, 3, 4, 5];

        metadata.compute_checksum(&data);
        assert!(metadata.verify_checksum(&data));

        let bad_data = vec![5u8, 4, 3, 2, 1];
        assert!(!metadata.verify_checksum(&bad_data));
    }

    #[test]
    fn test_chunk_request_serialization() {
        let req = ChunkRequest::new("chunk_123".to_string(), NodeId(42));
        let bincode_config = bincode::config::standard();
        let serialized = bincode::serde::encode_to_vec(&req, bincode_config).unwrap();
        let (deserialized, _): (ChunkRequest, usize) =
            bincode::serde::decode_from_slice(&serialized, bincode_config).unwrap();

        assert_eq!(req.chunk_id, deserialized.chunk_id);
        assert_eq!(req.requester, deserialized.requester);
    }

    #[test]
    fn test_chunk_response_success() {
        let metadata = ChunkMetadata::new_f64("chunk_456".to_string(), vec![5, 5]);
        let resp = ChunkResponse::success("chunk_456".to_string(), metadata, vec![1, 2, 3]);

        assert!(resp.success);
        assert!(resp.error.is_none());
        assert_eq!(resp.data.len(), 3);
    }

    #[test]
    fn test_chunk_response_error() {
        let resp = ChunkResponse::error("chunk_789".to_string(), "Not found".to_string());

        assert!(!resp.success);
        assert!(resp.error.is_some());
        assert_eq!(resp.data.len(), 0);
    }
}
