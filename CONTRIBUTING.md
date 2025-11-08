# Contributing to TenRSo

Thank you for your interest in contributing to TenRSo!

## Development Setup

1. **Install Rust**: Use the version specified in `rust-toolchain.toml` (≥ 1.77)
2. **Clone the repository**:
   ```bash
   git clone https://github.com/cool-japan/tenrso.git
   cd tenrso
   ```
3. **Run tests**:
   ```bash
   cargo test --workspace
   ```

## Coding Standards

### General Principles
- **Public APIs must be documented** with examples that compile (`doc(test)`)
- **Prefer `Result<T, Error>` over panics**; use `thiserror` for error types
- **No allocation in tight loops**; prefer pre-allocation & in-place operations
- **Deterministic results** (tie-breaks documented); seeds for stochastic steps
- **No panics in kernels**; all `unsafe` must be bounded & fuzzed

### Code Style
- Run `cargo fmt` before committing
- Ensure `cargo clippy -- -D warnings` passes
- Keep individual source files under 2000 lines (refactor if larger)
- Use the latest stable versions of dependencies

### Testing
- Write unit tests for all public APIs
- Add integration tests for end-to-end workflows
- Include property-based tests for mathematical correctness
- Benchmark performance-critical paths

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
2. **Write tests**: Ensure new code has adequate test coverage
3. **Update docs**: Add/update documentation for public APIs
4. **Run CI locally**:
   ```bash
   cargo fmt --check
   cargo clippy --workspace -- -D warnings
   cargo test --workspace
   cargo doc --workspace --no-deps
   ```
5. **Submit PR**: Provide clear description of changes and motivation
6. **Review process**: PRs require approval from at least one maintainer

## RFC Process

For significant changes (new APIs, architectural decisions, breaking changes):

1. Create a new file: `rfcs/NNN-title.md`
2. Open a PR with the RFC
3. Allow 3-day review window for feedback
4. Address comments and iterate
5. Merge after consensus

## Feature Flags

When adding new features, consider gating them behind feature flags:
- `cpu` (default)
- `simd`
- `gpu`
- `serde`
- `parquet`
- `arrow`
- `ooc`
- `csf` (enable CSF/HiCOO sparse formats)

## Performance Expectations

- Maintain performance budgets outlined in ROADMAP.md
- Add benchmarks for new kernels or algorithms
- Document algorithmic complexity in docstrings

## Communication

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Owner**: @cool-japan
- **Decision log**: `/docs/DECISIONS.md`

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
