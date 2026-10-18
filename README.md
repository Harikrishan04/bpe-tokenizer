# BPE Tokenizer

A compact C++ project for exploring byte-pair encoding, tokenization experiments, and multilingual/BPE research ideas.

This repository includes the main tokenizer pipeline in `src/` plus several research-oriented parts under `part2/` through `part6/`.

Optional CUDA benchmark
- A standalone GPU benchmark is available in `cuda/`.
- It compares CPU vs GPU timing for a simple token-lookup workload and prints the speedup.

Quick build
- CPU project: `make build`
- Optional CUDA benchmark: `make cuda-bench`

See `cuda/README.md` for details.
