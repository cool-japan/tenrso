# tenrso-ooc

Out-of-core tensor processing with Arrow/Parquet and memory mapping.

## Overview

`tenrso-ooc` enables tensor operations that exceed available RAM:

- **Arrow IPC** - Efficient in-memory tensor serialization
- **Parquet** - Columnar storage for tensor chunks
- **Memory mapping** - Direct file-backed tensor access
- **Streaming execution** - Chunked tensor contractions
- **Spill-to-disk** - Automatic memory management

## Features

- Deterministic chunk graphs
- Lazy loading and prefetching
- Back-pressure handling
- Integration with Arrow ecosystem

## Usage

```toml
[dependencies]
tenrso-ooc = "0.1"

# With all I/O backends
tenrso-ooc = { version = "0.1", features = ["arrow", "parquet", "mmap"] }
```

### Parquet I/O (TODO: M5)

```rust
use tenrso_ooc::ParquetWriter;

// Write large tensor to Parquet
let writer = ParquetWriter::new("tensor.parquet")?;
writer.write_chunked(&large_tensor, chunk_size=1000)?;

// Read chunks on demand
let reader = ParquetReader::open("tensor.parquet")?;
for chunk in reader.chunks() {
    process_chunk(chunk?);
}
```

### Memory-Mapped Tensors (TODO: M5)

```rust
use tenrso_ooc::MmapTensor;

// Memory-map large file
let tensor = MmapTensor::<f32>::open("data.bin", &shape)?;

// Access like regular tensor (lazy loading)
let value = tensor[[100, 200, 300]];
```

## Feature Flags

- `arrow` - Arrow IPC support
- `parquet` - Parquet I/O
- `mmap` - Memory-mapped access

## License

Apache-2.0
