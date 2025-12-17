//! Network communication layer for distributed chunk transfer.
//!
//! Uses tokio for async I/O and bincode for efficient serialization.

use super::protocol::{
    ChunkRequest, ChunkResponse, HeartbeatRequest, HeartbeatResponse, MessageType,
};
use super::registry::DistributedRegistry;
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

/// Configuration for network communication.
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Local bind address
    pub bind_address: SocketAddr,
    /// Maximum message size in bytes (default 1GB)
    pub max_message_size: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Write timeout
    pub write_timeout: Duration,
    /// Maximum concurrent connections
    pub max_connections: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8000".parse().unwrap(),
            max_message_size: 1024 * 1024 * 1024, // 1GB
            connection_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            max_connections: 100,
        }
    }
}

/// Statistics for network operations.
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Number of messages sent
    pub messages_sent: u64,
    /// Number of messages received
    pub messages_received: u64,
    /// Number of active connections
    pub active_connections: u64,
    /// Number of failed connections
    pub failed_connections: u64,
}

/// Network server for handling incoming chunk requests.
pub struct NetworkServer {
    config: NetworkConfig,
    registry: Arc<DistributedRegistry>,
    stats: Arc<NetworkStatsInternal>,
    chunk_provider: Arc<dyn ChunkProvider>,
}

/// Internal statistics with atomic counters.
struct NetworkStatsInternal {
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    active_connections: AtomicU64,
    failed_connections: AtomicU64,
}

impl NetworkStatsInternal {
    fn new() -> Self {
        Self {
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            failed_connections: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> NetworkStats {
        NetworkStats {
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            failed_connections: self.failed_connections.load(Ordering::Relaxed),
        }
    }
}

/// Trait for providing chunks to the network layer.
pub trait ChunkProvider: Send + Sync {
    /// Load a chunk by ID.
    fn load_chunk(&self, chunk_id: &str) -> Result<Vec<u8>>;
}

impl NetworkServer {
    /// Create a new network server.
    pub fn new(
        config: NetworkConfig,
        registry: Arc<DistributedRegistry>,
        chunk_provider: Arc<dyn ChunkProvider>,
    ) -> Self {
        Self {
            config,
            registry,
            stats: Arc::new(NetworkStatsInternal::new()),
            chunk_provider,
        }
    }

    /// Start the server and listen for incoming connections.
    pub async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(&self.config.bind_address).await?;

        loop {
            let (stream, peer_addr) = listener.accept().await?;

            // Spawn a task to handle this connection
            let registry = Arc::clone(&self.registry);
            let stats = Arc::clone(&self.stats);
            let chunk_provider = Arc::clone(&self.chunk_provider);
            let config = self.config.clone();

            tokio::spawn(async move {
                stats.active_connections.fetch_add(1, Ordering::Relaxed);

                if let Err(e) =
                    Self::handle_connection(stream, registry, stats.clone(), chunk_provider, config)
                        .await
                {
                    eprintln!("Error handling connection from {}: {}", peer_addr, e);
                    stats.failed_connections.fetch_add(1, Ordering::Relaxed);
                }

                stats.active_connections.fetch_sub(1, Ordering::Relaxed);
            });
        }
    }

    /// Handle a single connection.
    async fn handle_connection(
        mut stream: TcpStream,
        registry: Arc<DistributedRegistry>,
        stats: Arc<NetworkStatsInternal>,
        chunk_provider: Arc<dyn ChunkProvider>,
        config: NetworkConfig,
    ) -> Result<()> {
        loop {
            // Read message length (4 bytes)
            let mut len_buf = [0u8; 4];
            match timeout(config.read_timeout, stream.read_exact(&mut len_buf)).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => return Err(anyhow!("Failed to read message length: {}", e)),
                Err(_) => return Err(anyhow!("Read timeout")),
            }

            let len = u32::from_be_bytes(len_buf) as usize;
            if len > config.max_message_size {
                return Err(anyhow!("Message too large: {} bytes", len));
            }

            // Read message data
            let mut data_buf = vec![0u8; len];
            match timeout(config.read_timeout, stream.read_exact(&mut data_buf)).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => return Err(anyhow!("Failed to read message data: {}", e)),
                Err(_) => return Err(anyhow!("Read timeout")),
            }

            stats
                .bytes_received
                .fetch_add((4 + len) as u64, Ordering::Relaxed);
            stats.messages_received.fetch_add(1, Ordering::Relaxed);

            // Deserialize message
            let bincode_config = bincode::config::standard();
            let (message, _): (MessageType, usize) =
                bincode::serde::decode_from_slice(&data_buf, bincode_config)?;

            // Handle message and generate response
            let response = Self::handle_message(message, &registry, &chunk_provider).await?;

            // Serialize response
            let bincode_config = bincode::config::standard();
            let response_data = bincode::serde::encode_to_vec(&response, bincode_config)?;
            let response_len = response_data.len() as u32;

            // Write response length
            match timeout(
                config.write_timeout,
                stream.write_all(&response_len.to_be_bytes()),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(anyhow!("Failed to write response length: {}", e)),
                Err(_) => return Err(anyhow!("Write timeout")),
            }

            // Write response data
            match timeout(config.write_timeout, stream.write_all(&response_data)).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(anyhow!("Failed to write response data: {}", e)),
                Err(_) => return Err(anyhow!("Write timeout")),
            }

            stats
                .bytes_sent
                .fetch_add((4 + response_data.len()) as u64, Ordering::Relaxed);
            stats.messages_sent.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Handle an incoming message.
    async fn handle_message(
        message: MessageType,
        registry: &Arc<DistributedRegistry>,
        chunk_provider: &Arc<dyn ChunkProvider>,
    ) -> Result<MessageType> {
        match message {
            MessageType::ChunkRequest(req) => {
                // Load chunk data
                match chunk_provider.load_chunk(&req.chunk_id) {
                    Ok(data) => {
                        // Get chunk metadata
                        if let Some(placement) = registry.get_chunk_placement(&req.chunk_id) {
                            let response =
                                ChunkResponse::success(req.chunk_id, placement.metadata, data);
                            Ok(MessageType::ChunkResponse(response))
                        } else {
                            let response = ChunkResponse::error(
                                req.chunk_id,
                                "Chunk not found in registry".to_string(),
                            );
                            Ok(MessageType::ChunkResponse(response))
                        }
                    }
                    Err(e) => {
                        let response = ChunkResponse::error(req.chunk_id, e.to_string());
                        Ok(MessageType::ChunkResponse(response))
                    }
                }
            }
            MessageType::Heartbeat(req) => {
                // Respond with node info
                let node_info = registry
                    .get_node_info(req.sender)
                    .ok_or_else(|| anyhow!("Unknown node: {}", req.sender))?;

                let response = HeartbeatResponse::new(node_info);
                Ok(MessageType::HeartbeatResponse(response))
            }
            _ => Err(anyhow!("Unsupported message type")),
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> NetworkStats {
        self.stats.snapshot()
    }
}

/// Network client for sending chunk requests to remote nodes.
pub struct NetworkClient {
    config: NetworkConfig,
    #[allow(dead_code)]
    connections: Arc<RwLock<HashMap<SocketAddr, TcpStream>>>,
    stats: Arc<NetworkStatsInternal>,
}

impl NetworkClient {
    /// Create a new network client.
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(NetworkStatsInternal::new()),
        }
    }

    /// Request a chunk from a remote node.
    pub async fn request_chunk(
        &self,
        addr: SocketAddr,
        request: ChunkRequest,
    ) -> Result<ChunkResponse> {
        let message = MessageType::ChunkRequest(request);
        let response = self.send_message(addr, message).await?;

        match response {
            MessageType::ChunkResponse(resp) => Ok(resp),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Send a heartbeat to a remote node.
    pub async fn send_heartbeat(
        &self,
        addr: SocketAddr,
        request: HeartbeatRequest,
    ) -> Result<HeartbeatResponse> {
        let message = MessageType::Heartbeat(request);
        let response = self.send_message(addr, message).await?;

        match response {
            MessageType::HeartbeatResponse(resp) => Ok(resp),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Send a message to a remote node.
    async fn send_message(&self, addr: SocketAddr, message: MessageType) -> Result<MessageType> {
        // Get or create connection
        let mut stream = self.get_connection(addr).await?;

        // Serialize message
        let bincode_config = bincode::config::standard();
        let data = bincode::serde::encode_to_vec(&message, bincode_config)?;
        let len = data.len() as u32;

        if len as usize > self.config.max_message_size {
            return Err(anyhow!("Message too large: {} bytes", len));
        }

        // Write message length
        timeout(
            self.config.write_timeout,
            stream.write_all(&len.to_be_bytes()),
        )
        .await??;

        // Write message data
        timeout(self.config.write_timeout, stream.write_all(&data)).await??;

        self.stats
            .bytes_sent
            .fetch_add((4 + data.len()) as u64, Ordering::Relaxed);
        self.stats.messages_sent.fetch_add(1, Ordering::Relaxed);

        // Read response length
        let mut len_buf = [0u8; 4];
        timeout(self.config.read_timeout, stream.read_exact(&mut len_buf)).await??;

        let response_len = u32::from_be_bytes(len_buf) as usize;
        if response_len > self.config.max_message_size {
            return Err(anyhow!("Response too large: {} bytes", response_len));
        }

        // Read response data
        let mut response_buf = vec![0u8; response_len];
        timeout(
            self.config.read_timeout,
            stream.read_exact(&mut response_buf),
        )
        .await??;

        self.stats
            .bytes_received
            .fetch_add((4 + response_len) as u64, Ordering::Relaxed);
        self.stats.messages_received.fetch_add(1, Ordering::Relaxed);

        // Deserialize response
        let bincode_config = bincode::config::standard();
        let (response, _): (MessageType, usize) =
            bincode::serde::decode_from_slice(&response_buf, bincode_config)?;
        Ok(response)
    }

    /// Get or create a connection to a remote node.
    async fn get_connection(&self, addr: SocketAddr) -> Result<TcpStream> {
        // Try to connect
        match timeout(self.config.connection_timeout, TcpStream::connect(addr)).await {
            Ok(Ok(stream)) => Ok(stream),
            Ok(Err(e)) => {
                self.stats
                    .failed_connections
                    .fetch_add(1, Ordering::Relaxed);
                Err(anyhow!("Failed to connect to {}: {}", addr, e))
            }
            Err(_) => {
                self.stats
                    .failed_connections
                    .fetch_add(1, Ordering::Relaxed);
                Err(anyhow!("Connection timeout to {}", addr))
            }
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> NetworkStats {
        self.stats.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct MockChunkProvider {
        chunks: HashMap<String, Vec<u8>>,
    }

    #[allow(dead_code)]
    impl MockChunkProvider {
        fn new() -> Self {
            let mut chunks = HashMap::new();
            chunks.insert("test_chunk".to_string(), vec![1, 2, 3, 4, 5]);
            Self { chunks }
        }
    }

    impl ChunkProvider for MockChunkProvider {
        fn load_chunk(&self, chunk_id: &str) -> Result<Vec<u8>> {
            self.chunks
                .get(chunk_id)
                .cloned()
                .ok_or_else(|| anyhow!("Chunk not found: {}", chunk_id))
        }
    }

    #[test]
    fn test_network_config() {
        let config = NetworkConfig::default();
        assert!(config.max_message_size > 0);
        assert!(config.max_connections > 0);
    }

    #[test]
    fn test_network_stats() {
        let stats = NetworkStatsInternal::new();
        stats.bytes_sent.fetch_add(100, Ordering::Relaxed);
        stats.messages_sent.fetch_add(1, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.bytes_sent, 100);
        assert_eq!(snapshot.messages_sent, 1);
    }

    #[tokio::test]
    async fn test_network_client_creation() {
        let config = NetworkConfig::default();
        let client = NetworkClient::new(config);

        let stats = client.stats();
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.messages_received, 0);
    }
}
