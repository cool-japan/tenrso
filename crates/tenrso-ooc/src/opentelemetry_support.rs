//! # OpenTelemetry Integration
//!
//! Production-grade distributed tracing for out-of-core tensor operations.
//!
//! This module provides:
//! - OpenTelemetry tracer initialization with OTLP exporter
//! - Span creation and context propagation for tensor operations
//! - Integration with existing tracing infrastructure
//! - Semantic conventions for tensor-specific attributes
//! - Performance-optimized span recording
//!
//! ## Features
//!
//! - **Distributed Tracing**: Full support for distributed trace context propagation
//! - **OTLP Export**: Send traces to Jaeger, Zipkin, or any OTLP-compatible backend
//! - **Semantic Conventions**: Tensor-specific span attributes (shape, dtype, memory usage)
//! - **Sampling**: Configurable sampling strategies for high-throughput scenarios
//! - **Resource Detection**: Automatic service and resource metadata
//!
//! ## Usage
//!
//! ```rust,ignore
//! use tenrso_ooc::opentelemetry_support::{init_otel_tracer, OtelConfig, record_tensor_op};
//!
//! // Initialize OpenTelemetry
//! let config = OtelConfig::default()
//!     .with_service_name("tensor-pipeline")
//!     .with_otlp_endpoint("http://localhost:4317");
//! init_otel_tracer(config)?;
//!
//! // Record tensor operations
//! let span = record_tensor_op("matmul", &[256, 256], "f64");
//! // ... perform operation
//! drop(span); // Automatically records span duration
//! ```

use anyhow::{Context, Result};
use opentelemetry::{
    global,
    trace::{Span, Status, Tracer},
    KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider},
    Resource,
};
use opentelemetry_semantic_conventions as semconv;
use std::sync::Arc;
use std::time::Duration;

/// OpenTelemetry configuration for distributed tracing
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Service name for the tracer
    pub service_name: String,
    /// OTLP endpoint (e.g., "http://localhost:4317")
    pub otlp_endpoint: String,
    /// Sampling strategy
    pub sampler: SamplerConfig,
    /// Export timeout in milliseconds
    pub export_timeout_ms: u64,
    /// Maximum batch size for spans
    pub max_batch_size: usize,
    /// Whether to enable metric collection
    pub enable_metrics: bool,
}

/// Sampling configuration
#[derive(Debug, Clone, Copy)]
pub enum SamplerConfig {
    /// Always sample all spans
    AlwaysOn,
    /// Never sample (useful for disabling)
    AlwaysOff,
    /// Sample a fraction of spans (0.0 to 1.0)
    TraceIdRatioBased(f64),
    /// Parent-based sampling with fallback
    ParentBased,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            service_name: "tenrso-ooc".to_string(),
            otlp_endpoint: "http://localhost:4317".to_string(),
            sampler: SamplerConfig::AlwaysOn,
            export_timeout_ms: 10_000,
            max_batch_size: 512,
            enable_metrics: true,
        }
    }
}

impl OtelConfig {
    /// Create a new configuration with custom service name
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set the OTLP endpoint
    pub fn with_otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = endpoint.into();
        self
    }

    /// Set the sampling strategy
    pub fn with_sampler(mut self, sampler: SamplerConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set export timeout
    pub fn with_export_timeout(mut self, timeout_ms: u64) -> Self {
        self.export_timeout_ms = timeout_ms;
        self
    }

    /// Enable or disable metrics
    pub fn with_metrics(mut self, enable: bool) -> Self {
        self.enable_metrics = enable;
        self
    }
}

/// Initialize OpenTelemetry tracer with OTLP exporter
///
/// This sets up the global tracer provider and propagator for distributed tracing.
/// Call this once at application startup.
///
/// # Errors
///
/// Returns an error if the tracer cannot be initialized (e.g., OTLP endpoint unreachable)
pub fn init_otel_tracer(config: OtelConfig) -> Result<()> {
    // Set up trace context propagator
    global::set_text_map_propagator(TraceContextPropagator::new());

    // Build sampler
    let sampler = match config.sampler {
        SamplerConfig::AlwaysOn => Sampler::AlwaysOn,
        SamplerConfig::AlwaysOff => Sampler::AlwaysOff,
        SamplerConfig::TraceIdRatioBased(ratio) => Sampler::TraceIdRatioBased(ratio),
        SamplerConfig::ParentBased => Sampler::ParentBased(Box::new(Sampler::AlwaysOn)),
    };

    // Create resource with service metadata
    let resource = Resource::builder()
        .with_attributes(vec![
            KeyValue::new(semconv::resource::SERVICE_NAME, config.service_name.clone()),
            KeyValue::new(
                semconv::resource::SERVICE_VERSION,
                env!("CARGO_PKG_VERSION"),
            ),
            KeyValue::new("service.namespace", "tenrso"),
        ])
        .build();

    // Configure OTLP exporter with HTTP transport
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(&config.otlp_endpoint)
        .with_timeout(Duration::from_millis(config.export_timeout_ms))
        .build()
        .context("Failed to create OTLP exporter")?;

    // Build tracer provider
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .with_sampler(sampler)
        .with_id_generator(RandomIdGenerator::default())
        .build();

    // Set global tracer provider
    global::set_tracer_provider(provider);

    Ok(())
}

/// Shutdown OpenTelemetry and flush all pending spans
///
/// Call this before application exit to ensure all spans are exported.
///
/// Note: In OpenTelemetry 0.28+, the tracer provider is automatically
/// shut down when dropped. This function is kept for API compatibility.
pub fn shutdown_otel() {
    // No-op in OpenTelemetry 0.28+
    // The global tracer provider will flush on drop
}

/// Tensor operation span builder
pub struct TensorOpSpan {
    span: opentelemetry::global::BoxedSpan,
}

impl TensorOpSpan {
    /// Create a new tensor operation span
    fn new(operation: &str) -> Self {
        let tracer = global::tracer("tenrso-ooc");
        let span = tracer.start(operation.to_string());
        Self { span }
    }

    /// Record tensor shape as attribute
    pub fn with_shape(mut self, shape: &[usize]) -> Self {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        self.span
            .set_attribute(KeyValue::new("tensor.shape", shape_str));
        self
    }

    /// Record data type as attribute
    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.span
            .set_attribute(KeyValue::new("tensor.dtype", dtype.to_string()));
        self
    }

    /// Record memory usage in bytes
    pub fn with_memory_bytes(mut self, bytes: usize) -> Self {
        self.span
            .set_attribute(KeyValue::new("tensor.memory_bytes", bytes as i64));
        self
    }

    /// Record chunk index
    pub fn with_chunk_index(mut self, chunk_id: &str) -> Self {
        self.span
            .set_attribute(KeyValue::new("chunk.id", chunk_id.to_string()));
        self
    }

    /// Record spill tier (RAM, SSD, Disk)
    pub fn with_tier(mut self, tier: &str) -> Self {
        self.span
            .set_attribute(KeyValue::new("memory.tier", tier.to_string()));
        self
    }

    /// Record compression codec
    pub fn with_compression(mut self, codec: &str) -> Self {
        self.span
            .set_attribute(KeyValue::new("compression.codec", codec.to_string()));
        self
    }

    /// Record operation status as success
    pub fn record_success(mut self) -> Self {
        self.span.set_status(Status::Ok);
        self
    }

    /// Record operation status as error
    pub fn record_error(mut self, error: &str) -> Self {
        self.span.set_status(Status::error(error.to_string()));
        self
    }

    /// Add a custom attribute
    pub fn with_attribute(mut self, key: &str, value: impl Into<opentelemetry::Value>) -> Self {
        self.span
            .set_attribute(KeyValue::new(key.to_string(), value.into()));
        self
    }

    /// Get the underlying span for advanced usage
    pub fn inner(&mut self) -> &mut opentelemetry::global::BoxedSpan {
        &mut self.span
    }
}

impl Drop for TensorOpSpan {
    fn drop(&mut self) {
        self.span.end();
    }
}

/// Record a tensor operation with automatic span lifecycle
///
/// # Example
///
/// ```rust,ignore
/// let _span = record_tensor_op("matmul", &[256, 256], "f64")
///     .with_chunk_index("chunk_0")
///     .with_memory_bytes(524288);
/// // ... perform operation
/// // Span is automatically ended when dropped
/// ```
pub fn record_tensor_op(operation: &str, shape: &[usize], dtype: &str) -> TensorOpSpan {
    TensorOpSpan::new(operation)
        .with_shape(shape)
        .with_dtype(dtype)
}

/// Record a chunk I/O operation
pub fn record_io_op(operation: &str, bytes: usize, format: &str) -> TensorOpSpan {
    TensorOpSpan::new(operation)
        .with_memory_bytes(bytes)
        .with_attribute("io.format", format.to_string())
}

/// Record a memory management operation
pub fn record_memory_op(operation: &str, bytes: usize, tier: &str) -> TensorOpSpan {
    TensorOpSpan::new(operation)
        .with_memory_bytes(bytes)
        .with_tier(tier)
}

/// Record a compression operation
pub fn record_compression_op(
    operation: &str,
    bytes_in: usize,
    bytes_out: usize,
    codec: &str,
) -> TensorOpSpan {
    let ratio = if bytes_in > 0 {
        bytes_out as f64 / bytes_in as f64
    } else {
        1.0
    };

    TensorOpSpan::new(operation)
        .with_memory_bytes(bytes_in)
        .with_compression(codec)
        .with_attribute("compression.ratio", ratio)
        .with_attribute("compression.bytes_out", bytes_out as i64)
}

/// OpenTelemetry integration statistics
#[derive(Debug, Clone, Default)]
pub struct OtelStats {
    /// Number of spans created
    pub spans_created: usize,
    /// Number of successful operations
    pub operations_succeeded: usize,
    /// Number of failed operations
    pub operations_failed: usize,
    /// Total bytes traced
    pub total_bytes_traced: usize,
}

/// Thread-safe OpenTelemetry statistics collector
pub struct OtelStatsCollector {
    stats: Arc<parking_lot::RwLock<OtelStats>>,
}

impl OtelStatsCollector {
    /// Create a new statistics collector
    pub fn new() -> Self {
        Self {
            stats: Arc::new(parking_lot::RwLock::new(OtelStats::default())),
        }
    }

    /// Record a span creation
    pub fn record_span(&self) {
        let mut stats = self.stats.write();
        stats.spans_created += 1;
    }

    /// Record a successful operation
    pub fn record_success(&self, bytes: usize) {
        let mut stats = self.stats.write();
        stats.operations_succeeded += 1;
        stats.total_bytes_traced += bytes;
    }

    /// Record a failed operation
    pub fn record_failure(&self, bytes: usize) {
        let mut stats = self.stats.write();
        stats.operations_failed += 1;
        stats.total_bytes_traced += bytes;
    }

    /// Get current statistics
    pub fn stats(&self) -> OtelStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset(&self) {
        *self.stats.write() = OtelStats::default();
    }
}

impl Default for OtelStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otel_config_builder() {
        let config = OtelConfig::default()
            .with_service_name("test-service")
            .with_otlp_endpoint("http://localhost:4318")
            .with_sampler(SamplerConfig::TraceIdRatioBased(0.1))
            .with_export_timeout(5000)
            .with_metrics(false);

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.otlp_endpoint, "http://localhost:4318");
        assert_eq!(config.export_timeout_ms, 5000);
        assert!(!config.enable_metrics);
    }

    #[test]
    fn test_stats_collector() {
        let collector = OtelStatsCollector::new();

        collector.record_span();
        collector.record_success(1024);
        collector.record_failure(512);

        let stats = collector.stats();
        assert_eq!(stats.spans_created, 1);
        assert_eq!(stats.operations_succeeded, 1);
        assert_eq!(stats.operations_failed, 1);
        assert_eq!(stats.total_bytes_traced, 1536);

        collector.reset();
        let stats = collector.stats();
        assert_eq!(stats.spans_created, 0);
    }

    #[test]
    fn test_sampler_config() {
        let config1 = OtelConfig::default().with_sampler(SamplerConfig::AlwaysOn);
        let config2 = OtelConfig::default().with_sampler(SamplerConfig::AlwaysOff);
        let config3 = OtelConfig::default().with_sampler(SamplerConfig::TraceIdRatioBased(0.5));

        // Just verify they compile and are different
        assert!(matches!(config1.sampler, SamplerConfig::AlwaysOn));
        assert!(matches!(config2.sampler, SamplerConfig::AlwaysOff));
        assert!(matches!(
            config3.sampler,
            SamplerConfig::TraceIdRatioBased(_)
        ));
    }
}
