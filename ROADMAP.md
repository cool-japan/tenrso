# TenRSo Development Roadmap

## Timeline: 16–20 weeks to v0.1.0

### M0 (Week 0–1): Repo hygiene ✓
- [x] Workspace skeleton, CI (fmt/clippy/test), MSRV pin, Apache-2.0
- [x] `tenrso-core` minimal dense tensor + axis meta + unfold/fold

### M1 (Week 2–4): Kernels (dense)
- [ ] Khatri–Rao/Kronecker/Hadamard + n-mode product + MTTKRP
- [ ] Simple perf tests; correctness property tests

### M2 (Week 5–6): Decompositions v0
- [ ] CP-ALS (dense); Tucker-HOOI (dense); TT-SVD baseline
- [ ] Reconstruction error benchmarks; doc examples

### M3 (Week 7–8): Sparse & Masked
- [ ] COO/CSR/BCSR; SpMM/SpSpMM; masked-einsum (dense+sparse mix)
- [ ] Conversion paths; light statistics for nnz/density

### M4 (Week 9–10): Planner v0
- [ ] Heuristic order search; representation selection; tiling (CPU)
- [ ] `einsum_ex` public API & integration across crates

### M5 (Week 11–12): Out-of-Core v0
- [ ] Arrow/Parquet readers; chunk iteration; mmap support
- [ ] Streaming contraction with deterministic chunks

### M6 (Week 13–16): AD hooks
- [ ] Custom VJP for contraction; CP/Tucker/TT backward
- [ ] Minimal integration demo with Tensorlogic training loop

### Stretch (Week 17–20)
- [ ] Low-rank + sparse mix planning improvements
- [ ] TT ops
- [ ] More robust OoC policies

## Performance Targets

### Contraction (CPU, float32)
- `einsum("bij,bjk->bik")` 1024×1024×1024 dense: ≥ 80% of OpenBLAS baseline (single-socket)
- Masked variant with 90% zeros: ≥ 5× speedup vs dense naive

### CP-ALS
- Synthetic 256×256×256 rank-64: < 2s / 10 iters on 16-core CPU
- Reconstruction error ≤ 1e-3 (seeded)

### Tucker-HOOI
- 512×512×128, ranks [64,64,32]: < 3s / 10 iters
- Error ≤ 1e-3

### TT-SVD
- 32^6 tensor, eps=1e-6: build in < 2s
- Rounding reduces memory ≥ 10×

## SLoC Estimates

Total target: ~50k–75k SLoC including tests/benches

| Crate | Estimate (SLoC) |
|-------|----------------|
| tenrso-core | ~8k |
| tenrso-kernels | ~10k |
| tenrso-decomp | ~12k |
| tenrso-sparse | ~10k |
| tenrso-planner | ~8k |
| tenrso-ooc | ~5k |
| tenrso-exec | ~6k |
| tenrso-ad | ~6k |

## Future Work (post-v0.1.0)

- GPU/ROCm/Metal backends
- Distributed execution
- Advanced sparse formats (BSR, DIA, ELL)
- More decomposition methods (PARAFAC2, NTF variants)
- Integration with external AD frameworks
- Performance profiling dashboard
