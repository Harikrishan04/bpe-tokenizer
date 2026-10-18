NVCC ?= nvcc
TARGET = token_lookup_benchmark
CUDAFLAGS ?= -O3 -std=c++17

all: $(TARGET)

$(TARGET): token_lookup_benchmark.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) 20000000

clean:
	rm -f $(TARGET)

.PHONY: all run clean
