# CUDA Token Lookup Benchmark

This benchmark compares a single-threaded CPU loop against a CUDA kernel that maps a large input array through a fixed 256-entry lookup table.

What the benchmark does
- Creates a large input array of mapped token values.
- Builds a 256-entry lookup table.
- Runs a single-threaded CPU loop to compute the baseline.
- Copies input and lookup data to the GPU.
- Launches a CUDA kernel that performs the same mapping in parallel.
- Validates GPU output against the CPU output.
- Prints CPU time, kernel time, host-to-device time, device-to-host time, GPU end-to-end time, speedup, and validation status.

Build
- Requires a CUDA-capable GPU and nvcc.
- Example build command:

  nvcc -O3 -std=c++17 -arch=sm_75 token_lookup_benchmark.cu -o bench

- Note: `-arch=sm_75` is an example; choose the GPU compute capability appropriate for your device.

Run
- Optional input size N:

  ./bench [N]

- Default size is 20,000,000 elements.

Measured results
Hardware: Google Colab / Tesla T4
Driver: CUDA 13.0 driver
Compiler: nvcc -O3 -arch=sm_75
Date: 2026-10-18

| N      | CPU ms | GPU kernel ms | Speedup | Validation |
|--------|--------|---------------|---------|------------|
| 20M    | 16.175 | 0.756         | 21.4x   | PASS       |
| 1M     | 0.644  | 0.120         | 5.4x    | PASS       |
| 20M    | 15.966 | 0.724         | 22.1x   | PASS       |

Important notes
- GPU times in the table are kernel-only timing; transfer times are not included.
- The CPU baseline is a single-threaded loop.
- The benchmark program prints transfer times (H2D and D2H) separately and also prints a total GPU end-to-end time.
- The lookup table remains a fixed 256-entry mapping; the logic is intentionally simple and deterministic.