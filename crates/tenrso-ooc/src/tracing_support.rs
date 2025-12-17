//! Structured logging and tracing support for out-of-core operations
//!
//! This module provides tracing integration for monitoring and debugging
//! out-of-core tensor operations, memory management, and I/O performance.
//!
//! # Features
//!
//! - **Structured logging**: JSON and pretty-printed formats
//! - **Span instrumentation**: Automatic timing and context tracking
//! - **Event filtering**: Environment-based log level control
//! - **Performance metrics**: Automatic span timing
//! - **Context propagation**: Correlation across async boundaries
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::tracing_support::{init_tracing, TracingConfig};
//!
//! // Initialize tracing at application startup
//! init_tracing(TracingConfig {
//!     format: TracingFormat::Pretty,
//!     filter: "tenrso_ooc=debug,info".to_string(),
//! }).unwrap();
//!
//! // Tracing will now capture all instrumented operations
//! ```
//!
//! # Environment Variables
//!
//! - `RUST_LOG`: Set log level (e.g., `RUST_LOG=debug`)
//! - `TENRSO_LOG_FORMAT`: Set output format (`json` or `pretty`, default: `pretty`)
//!
//! # Usage in Code
//!
//! Functions are automatically instrumented with `#[tracing::instrument]`:
//!
//! ```ignore
//! #[tracing::instrument(skip(tensor), fields(size = tensor.len()))]
//! fn process_chunk(chunk_id: &str, tensor: &DenseND<f64>) -> Result<()> {
//!     tracing::info!("Processing chunk");
//!     // ... operation ...
//!     tracing::debug!(bytes_processed = %tensor.len() * 8, "Chunk processed");
//!     Ok(())
//! }
//! ```

#[cfg(feature = "tracing")]
use anyhow::Result;
#[cfg(feature = "tracing")]
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

/// Tracing output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    /// Pretty-printed human-readable format
    Pretty,
    /// JSON format for structured logging
    Json,
    /// Compact format (single line per event)
    Compact,
}

impl TracingFormat {
    /// Parse from string
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => TracingFormat::Json,
            "compact" => TracingFormat::Compact,
            _ => TracingFormat::Pretty,
        }
    }
}

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Output format
    pub format: TracingFormat,
    /// Filter directive (e.g., "tenrso_ooc=debug,info")
    pub filter: String,
    /// Enable ANSI colors
    pub with_ansi: bool,
    /// Show target module paths
    pub with_target: bool,
    /// Show thread IDs
    pub with_thread_ids: bool,
    /// Show thread names
    pub with_thread_names: bool,
    /// Show file locations
    pub with_file: bool,
    /// Show line numbers
    pub with_line_number: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        // Check environment for format preference
        let format = std::env::var("TENRSO_LOG_FORMAT")
            .map(|s| TracingFormat::parse(&s))
            .unwrap_or(TracingFormat::Pretty);

        // Check environment for filter
        let filter =
            std::env::var("RUST_LOG").unwrap_or_else(|_| "tenrso_ooc=info,warn".to_string());

        Self {
            format,
            filter,
            with_ansi: true,
            with_target: true,
            with_thread_ids: false,
            with_thread_names: false,
            with_file: true,
            with_line_number: true,
        }
    }
}

/// Initialize tracing subscriber with the given configuration
///
/// This should be called once at application startup.
///
/// # Example
///
/// ```ignore
/// use tenrso_ooc::tracing_support::{init_tracing, TracingConfig};
///
/// fn main() -> Result<()> {
///     init_tracing(TracingConfig::default())?;
///     // ... rest of application ...
///     Ok(())
/// }
/// ```
#[cfg(feature = "tracing")]
pub fn init_tracing(config: TracingConfig) -> Result<()> {
    let filter = EnvFilter::try_new(&config.filter)?;

    match config.format {
        TracingFormat::Pretty => {
            let fmt_layer = fmt::layer()
                .pretty()
                .with_ansi(config.with_ansi)
                .with_target(config.with_target)
                .with_thread_ids(config.with_thread_ids)
                .with_thread_names(config.with_thread_names)
                .with_file(config.with_file)
                .with_line_number(config.with_line_number)
                .with_filter(filter);

            tracing_subscriber::registry().with(fmt_layer).init();
        }
        TracingFormat::Json => {
            let fmt_layer = fmt::layer()
                .json()
                .with_target(config.with_target)
                .with_thread_ids(config.with_thread_ids)
                .with_thread_names(config.with_thread_names)
                .with_file(config.with_file)
                .with_line_number(config.with_line_number)
                .with_filter(filter);

            tracing_subscriber::registry().with(fmt_layer).init();
        }
        TracingFormat::Compact => {
            let fmt_layer = fmt::layer()
                .compact()
                .with_ansi(config.with_ansi)
                .with_target(config.with_target)
                .with_thread_ids(config.with_thread_ids)
                .with_thread_names(config.with_thread_names)
                .with_file(config.with_file)
                .with_line_number(config.with_line_number)
                .with_filter(filter);

            tracing_subscriber::registry().with(fmt_layer).init();
        }
    }

    Ok(())
}

/// Stub for when tracing feature is disabled
#[cfg(not(feature = "tracing"))]
pub fn init_tracing(_config: TracingConfig) -> Result<()> {
    Ok(())
}

/// Helper macro for tracing spans with timing
///
/// This creates a span and records its duration when it ends.
///
/// # Example
///
/// ```ignore
/// timed_span!("chunk_processing", chunk_id = "chunk_0", size = 1024);
/// // ... operations ...
/// // Span automatically records duration on drop
/// ```
#[macro_export]
#[cfg(feature = "tracing")]
macro_rules! timed_span {
    ($name:expr) => {
        tracing::info_span!($name)
    };
    ($name:expr, $($field:tt = $value:expr),+ $(,)?) => {
        tracing::info_span!($name, $($field = $value),+)
    };
}

/// Stub for when tracing is disabled
#[macro_export]
#[cfg(not(feature = "tracing"))]
macro_rules! timed_span {
    ($name:expr) => {
        ()
    };
    ($name:expr, $($field:tt = $value:expr),+ $(,)?) => {
        ()
    };
}

/// Helper to record a metric value
#[cfg(feature = "tracing")]
pub fn record_metric(name: &str, value: f64) {
    tracing::info!(metric = name, value = value, "metric_recorded");
}

/// Stub for when tracing is disabled
#[cfg(not(feature = "tracing"))]
pub fn record_metric(_name: &str, _value: f64) {}

/// Helper to record bytes processed
#[cfg(feature = "tracing")]
pub fn record_bytes(operation: &str, bytes: usize) {
    tracing::debug!(
        operation = operation,
        bytes = bytes,
        mb = bytes as f64 / 1024.0 / 1024.0,
        "bytes_processed"
    );
}

/// Stub for when tracing is disabled
#[cfg(not(feature = "tracing"))]
pub fn record_bytes(_operation: &str, _bytes: usize) {}

/// Helper to record I/O operations
#[cfg(feature = "tracing")]
pub fn record_io(operation: &str, path: &str, bytes: usize, duration_ms: u64) {
    tracing::info!(
        operation = operation,
        path = path,
        bytes = bytes,
        duration_ms = duration_ms,
        throughput_mbps = (bytes as f64 / 1024.0 / 1024.0) / (duration_ms as f64 / 1000.0),
        "io_operation"
    );
}

/// Stub for when tracing is disabled
#[cfg(not(feature = "tracing"))]
pub fn record_io(_operation: &str, _path: &str, _bytes: usize, _duration_ms: u64) {}

/// Helper to record memory operations
#[cfg(feature = "tracing")]
pub fn record_memory_op(operation: &str, chunk_id: &str, bytes: usize, tier: &str) {
    tracing::debug!(
        operation = operation,
        chunk_id = chunk_id,
        bytes = bytes,
        tier = tier,
        "memory_operation"
    );
}

/// Stub for when tracing is disabled
#[cfg(not(feature = "tracing"))]
pub fn record_memory_op(_operation: &str, _chunk_id: &str, _bytes: usize, _tier: &str) {}

#[cfg(test)]
#[cfg(feature = "tracing")]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_format_parse() {
        assert_eq!(TracingFormat::parse("json"), TracingFormat::Json);
        assert_eq!(TracingFormat::parse("pretty"), TracingFormat::Pretty);
        assert_eq!(TracingFormat::parse("compact"), TracingFormat::Compact);
        assert_eq!(TracingFormat::parse("unknown"), TracingFormat::Pretty);
    }

    #[test]
    fn test_default_config() {
        let config = TracingConfig::default();
        assert!(config.with_ansi);
        assert!(config.with_target);
    }

    #[test]
    fn test_record_helpers() {
        // These should not panic
        record_metric("test_metric", 42.0);
        record_bytes("test_operation", 1024);
        record_io("read", "/tmp/test.bin", 2048, 100);
        record_memory_op("spill", "chunk_0", 4096, "SSD");
    }
}

#[cfg(test)]
#[cfg(not(feature = "tracing"))]
mod tests {
    use super::*;

    #[test]
    fn test_stubs_work() {
        // These should all be no-ops
        let config = TracingConfig::default();
        let _ = init_tracing(config);
        record_metric("test", 1.0);
        record_bytes("test", 100);
        record_io("test", "path", 100, 10);
        record_memory_op("test", "id", 100, "RAM");
    }
}
