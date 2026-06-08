# Variables (reusable flags)
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Targets (what "make" can do)
build:
	$(CXX) $(CXXFLAGS) -o bpe_trainer src/bpe.cpp src/main.cpp
	cd part2/src && $(CXX) $(CXXFLAGS) -o byte_bpe byte_bpe.cpp
	cd part3/src && $(CXX) $(CXXFLAGS) -o fairness fairness.cpp
	@echo "✅ Build complete"

test: build
	@echo "Verifying binaries..."
	@[ -f bpe_trainer ] && echo "✅ bpe_trainer OK" || echo "❌ FAILED"

clean:
	rm -f bpe_trainer part2/src/byte_bpe part3/src/fairness
	@echo "✅ Cleaned"

.PHONY: build test clean
