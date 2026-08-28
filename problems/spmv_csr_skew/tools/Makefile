# Builds the measurement harnesses. `make -C problems/spmv_csr_skew/tools`
NVCC ?= nvcc
ARCH ?= -arch=sm_86

all: bench_cusparse floors

bench_cusparse: bench_cusparse.cu
	$(NVCC) $(ARCH) -O3 -std=c++17 -o $@ $< -lcusparse

floors: floors.cu
	$(NVCC) $(ARCH) -O3 -std=c++17 -o $@ $<

clean:
	rm -f bench_cusparse floors

.PHONY: all clean
