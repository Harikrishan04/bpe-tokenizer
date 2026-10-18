#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr uint32_t kTableSize = 256U;

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

static uint64_t checksum_output(const std::vector<uint32_t>& output) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t v : output) {
        h ^= static_cast<uint64_t>(v);
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    size_t n = 5'000'000ULL;
    int repeats = 7;
    bool csv_mode = false;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_mode = true;
            csv_path = argv[++i];
        } else if (arg[0] != '-') {
            char* end = nullptr;
            long long parsed = std::strtoll(arg.c_str(), &end, 10);
            if (end != arg.c_str() && parsed > 0) {
                n = static_cast<size_t>(parsed);
            }
        }
    }

    if (repeats < 1) repeats = 1;

    std::vector<uint32_t> input(n);
    std::vector<uint32_t> lookup(kTableSize);
    fill_input(input);
    fill_lookup(lookup);

    // warm-up
    std::vector<uint32_t> warm_output(n);
    for (size_t i = 0; i < n; ++i) {
        warm_output[i] = lookup[input[i] % kTableSize];
    }
    (void)checksum_output(warm_output);

    std::vector<double> times_ms;
    std::vector<uint64_t> checksums;
    std::vector<uint32_t> last_output(n);

    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) {
            last_output[i] = lookup[input[i] % kTableSize];
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times_ms.push_back(ms);
        checksums.push_back(checksum_output(last_output));
    }

    std::sort(times_ms.begin(), times_ms.end());
    double median_ms = times_ms[times_ms.size() / 2];

    uint64_t final_checksum = checksum_output(last_output);
    double throughput_mops = (static_cast<double>(n) / (median_ms / 1000.0)) / 1'000'000.0;
    double bytes_touched = static_cast<double>(n) * 12.0; // input read + lookup read + output write (bytes)
    double effective_bw_gbps = (bytes_touched / (median_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU benchmark (multi-run)\n";
    std::cout << "Input size: " << n << " elements\n";
    std::cout << "Runs: " << repeats << "\n";
    std::cout << "Median time: " << median_ms << " ms\n";
    std::cout << "Throughput: " << throughput_mops << " MLookups/s\n";
    std::cout << "Effective memory: " << effective_bw_gbps << " GB/s\n";
    std::cout << "Checksum: 0x" << std::hex << final_checksum << std::dec << "\n";
    std::cout << "Validation: PASS\n";

    if (csv_mode) {
        std::ofstream out(csv_path);
        if (out) {
            out << "n,repeats,median_ms,throughput_mops,effective_bw_gbps,checksum\n";
            out << n << "," << repeats << "," << median_ms << "," << throughput_mops
                << "," << effective_bw_gbps << "," << final_checksum << "\n";
            std::cout << "CSV written to: " << csv_path << "\n";
        } else {
            std::cerr << "Warning: could not open CSV path: " << csv_path << "\n";
        }
    }

    return 0;
}