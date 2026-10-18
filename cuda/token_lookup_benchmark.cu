#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static constexpr uint32_t kTableSize = 256U;

__global__ void map_tokens_kernel(const uint32_t* input,
                                 uint32_t* output,
                                 const uint32_t* table,
                                 size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t value = input[idx];
        output[idx] = table[value % kTableSize];
    }
}

static void fill_input(std::vector<uint32_t>& input) {
    std::mt19937_64 rng(12345ULL);
    std::uniform_int_distribution<uint32_t> dist(0, kTableSize - 1U);
    for (uint32_t& v : input) {
        v = dist(rng);
    }
}

static void fill_lookup(std::vector<uint32_t>& lookup) {
    for (uint32_t i = 0; i < kTableSize; ++i) {
        lookup[i] = (i * 17U + 11U) ^ 0x9E3779B9U;
    }
}

int main(int argc, char** argv) {
    size_t n = 20'000'000ULL;
    if (argc > 1) {
        char* end = nullptr;
        long long parsed = std::strtoll(argv[1], &end, 10);
        if (end != argv[1] && parsed > 0) {
            n = static_cast<size_t>(parsed);
        }
    }

    std::vector<uint32_t> input(n);
    std::vector<uint32_t> lookup(kTableSize);
    std::vector<uint32_t> cpu_output(n);
    std::vector<uint32_t> gpu_output(n);

    fill_input(input);
    fill_lookup(lookup);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        cpu_output[i] = lookup[input[i] % kTableSize];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    uint32_t* d_input = nullptr;
    uint32_t* d_output = nullptr;
    uint32_t* d_lookup = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_input, n * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc for input failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMalloc(&d_output, n * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc for output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return 1;
    }

    err = cudaMalloc(&d_lookup, kTableSize * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc for lookup table failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output);
        cudaFree(d_input);
        return 1;
    }

    err = cudaMemcpy(d_input, input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy input failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_lookup);
        cudaFree(d_output);
        cudaFree(d_input);
        return 1;
    }

    err = cudaMemcpy(d_lookup, lookup.data(), kTableSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy lookup failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_lookup);
        cudaFree(d_output);
        cudaFree(d_input);
        return 1;
    }

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    map_tokens_kernel<<<blocks, threads>>>(d_input, d_output, d_lookup, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    err = cudaMemcpy(gpu_output.data(), d_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_lookup);
        cudaFree(d_output);
        cudaFree(d_input);
        return 1;
    }

    bool valid = true;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_output[i] != gpu_output[i]) {
            valid = false;
            std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_output[i]
                      << ", GPU=" << gpu_output[i] << std::endl;
            break;
        }
    }

    cudaFree(d_lookup);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Input size: " << n << " elements\n";
    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";

    double speedup = cpu_ms / std::max(gpu_ms, 0.001f);
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Validation: " << (valid ? "PASS" : "FAIL") << "\n";

    return valid ? 0 : 1;
}
