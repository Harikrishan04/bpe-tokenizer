NVCC ?= nvcc
TARGET_GPU = token_lookup_benchmark
TARGET_CPU = token_lookup_cpu
CUDAFLAGS ?= -O3 -std=c++17
CPPFLAGS ?= -O2 -std=c++17

all: cpu

gpu: $(TARGET_GPU)

$(TARGET_GPU): token_lookup_benchmark.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

cpu: token_lookup_cpu.cpp
	g++ $(CPPFLAGS) token_lookup_cpu.cpp -o $(TARGET_CPU)

run-gpu: gpu
	./$(TARGET_GPU) 20000000

run-cpu: cpu
	./$(TARGET_CPU) 5000000

clean:
	rm -f $(TARGET_GPU) $(TARGET_CPU)

.PHONY: all gpu cpu run-gpu run-cpu clean