# CUDA Token Lookup Benchmark

This benchmark is a small, isolated GPU example for the repository. It compares a CPU lookup loop against a CUDA kernel that performs the same token mapping in parallel.

Why this benchmark?
- It keeps the example small and self-contained.
- It shows a clear CPU vs GPU timing comparison.
- It avoids coupling the GPU code to the main BPE training pipeline.

Build

From this directory:

```bash
make
```

Run

```bash
./token_lookup_benchmark 20000000
```

This uses a large input array and prints:
- CPU time
- GPU time
- speedup ratio
- validation result

Example output

```text
Input size: 20,000,000 elements
CPU time: 112.443 ms
GPU time: 8.721 ms
Speedup: 12.90x
Validation: PASS
```

Notes
- Requires a CUDA-capable GPU and `nvcc` installed.
- The benchmark is intentionally separate from the main C++ tokenizer code.
