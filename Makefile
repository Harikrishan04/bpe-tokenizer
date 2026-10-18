# Variables (reusable flags)
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Targets (what "make" can do)
build:
	$(CXX) $(CXXFLAGS) -o bpe_trainer src/bpe.cpp src/main.cpp
	cd part2/src && $(CXX) $(CXXFLAGS) -o byte_bpe byte_bpe.cpp
	cd part3/src && $(CXX) $(CXXFLAGS) -o fairness fairness.cpp
	cd part4/src && $(CXX) $(CXXFLAGS) -o codemixed codemixed.cpp
	cd part5/src && $(CXX) $(CXXFLAGS) -o edge_tokenizer edge_tokenizer.cpp
	cd part6/src && $(CXX) $(CXXFLAGS) -o explainable explainable.cpp
	@echo "✅ Build complete"

cuda-bench:
	$(MAKE) -C cuda

# Keep the core repo build predictable while allowing a separate GPU benchmark.
test: build
	@echo "Verifying binaries..."
	@[ -f bpe_trainer ] && echo "✅ bpe_trainer OK" || echo "❌ FAILED"
	@[ -f part2/src/byte_bpe ] && echo "✅ byte_bpe OK" || echo "❌ FAILED"
	@[ -f part3/src/fairness ] && echo "✅ fairness OK" || echo "❌ FAILED"
	@[ -f part4/src/codemixed ] && echo "✅ codemixed OK" || echo "❌ FAILED"
	@[ -f part5/src/edge_tokenizer ] && echo "✅ edge_tokenizer OK" || echo "❌ FAILED"
	@[ -f part6/src/explainable ] && echo "✅ explainable OK" || echo "❌ FAILED"

clean:
	rm -f bpe_trainer part2/src/byte_bpe part3/src/fairness part4/src/codemixed part5/src/edge_tokenizer part6/src/explainable
	$(MAKE) -C cuda clean
	@echo "✅ Cleaned"

.PHONY: build cuda-bench test clean
