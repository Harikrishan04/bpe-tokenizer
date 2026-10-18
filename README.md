# bpe-tokenizer

A C++17 project for byte-pair encoding (BPE) experiments and tokenizer research. The repository contains a core trainer and six research-oriented parts.

Project layout
- src/ — core trainer and CLI (bpe.cpp, main.cpp)
- part2/src/byte_bpe.cpp — Part 2: byte-level BPE tokenizer and robustness evaluation
- part3/src/fairness.cpp — Part 3: fairness and bias analysis
- part4/src/codemixed.cpp — Part 4: code-mixed/script-aware tokenizer evaluation
- part5/src/edge_tokenizer.cpp — Part 5: low-latency edge-device tokenizer benchmarking
- part6/src/explainable.cpp — Part 6: explainability tooling (TODO: details inferred from source)
- cuda/ — optional CUDA micro-benchmark for token lookup acceleration
- Dockerfile — Ubuntu-based container build for the CPU targets
- Makefile — build targets for CPU parts and the optional CUDA benchmark

Build and run
- Build CPU targets:
  make build

- Run the build verification target:
  make test

- Build the optional CUDA benchmark:
  make cuda-bench

  Then compile the CUDA target manually in cuda/ with nvcc, for example:
  cd cuda
  nvcc -O3 -std=c++17 -arch=sm_75 token_lookup_benchmark.cu -o bench
  ./bench [N]

Docker
- The Dockerfile installs build-essential, g++, and make, then runs `make build` and sets the default command to `./bpe_trainer`.
- The Dockerfile builds the CPU targets only; it does not enable CUDA or nvcc in the container.

Notes
- The CUDA benchmark is intentionally kept separate from the core BPE training pipeline so the project remains buildable without a GPU.
- The repository is organized as research components, and the README is intentionally brief and source-driven.