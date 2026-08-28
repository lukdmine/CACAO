// What each ingredient of CSR SpMV costs on its own, so a proposal can be judged
// against a floor instead of against the last iteration.
//
//     make -C problems/spmv_csr_skew/tools floors
//     problems/spmv_csr_skew/tools/floors mawi 18571154 19020160 18571154
//
// Four kernels over the same buffers:
//   stream       col_indices + values read once, no gather, no row structure
//   rowptr       row_offsets read once, nothing else
//   flat gather  the full multiply-accumulate over every nonzero, still with NO
//                row structure -- the arithmetic and the x gather, and nothing else
//   warp-per-row a correct SpMV with one warp per row, the naive decomposition
//
// Measured on an RTX 3090 / CUDA 12.0 (medians, microseconds):
//
//   transient    stream  14.6   rowptr   9.2   flat gather  15.4   warp-per-row    138.2
//   rail4284     stream 104.4   rowptr   8.2   flat gather 139.3   warp-per-row    224.1
//   mawi         stream 174.1   rowptr  88.1   flat gather 234.8   warp-per-row 16 199.7
//
// The small cases carry ~4 us of launch overhead each, so their numbers are not
// additive floors. mawi's are: gathering all 19M nonzeros costs 235 us, reading
// row_offsets costs 88 us, writing y costs about as much again -- so a correct
// kernel has no business being slower than roughly 400-500 us there.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
template<typename T> std::vector<T> rd(const std::string& p, size_t n){
  std::ifstream f(p, std::ios::binary); std::vector<T> v(n);
  f.read((char*)v.data(), n*sizeof(T)); return v; }

__global__ void k_flat(int nnz, const int* col, const float* val, const float* x, float* sink){
  int i = blockIdx.x*blockDim.x + threadIdx.x; float s = 0;
  for (int j = i; j < nnz; j += gridDim.x*blockDim.x) s += val[j]*x[col[j]];
  if (s == 1234.5678f) sink[0] = s;                       // never true; keeps the work
}
__global__ void k_stream(int nnz, const int* col, const float* val, float* sink){
  int i = blockIdx.x*blockDim.x + threadIdx.x; float s = 0;
  for (int j = i; j < nnz; j += gridDim.x*blockDim.x) s += val[j] + col[j];
  if (s == 1234.5678f) sink[0] = s;                       // same streams, no gather
}
__global__ void k_rowptr(int nrows, const int* rp, float* sink){
  int i = blockIdx.x*blockDim.x + threadIdx.x; float s = 0;
  for (int r = i; r < nrows; r += gridDim.x*blockDim.x) s += rp[r+1]-rp[r];
  if (s == 1234.5678f) sink[0] = s;
}
__global__ void k_warprow(int nrows, const int* rp, const int* col, const float* val,
                          const float* x, float* y){
  int g = blockIdx.x*blockDim.x + threadIdx.x, r = g>>5, l = g&31;
  if (r >= nrows) return;
  int b = rp[r], e = rp[r+1]; float s = 0;
  for (int j = b+l; j < e; j += 32) s += val[j]*x[col[j]];
  for (int o = 16; o >= 1; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if (l == 0) y[r] = s;
}
float timeit(void(*launch)(), int reps=10){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  launch(); cudaDeviceSynchronize();
  std::vector<float> t;
  for(int i=0;i<reps;i++){ cudaEventRecord(a); launch(); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); t.push_back(ms*1000);}
  std::sort(t.begin(),t.end()); return t[t.size()/2];
}
int NROWS, NNZ; int *d_rp,*d_col; float *d_val,*d_x,*d_y;
void L_flat(){ k_flat<<<8192,256>>>(NNZ,d_col,d_val,d_x,d_y); }
void L_stream(){ k_stream<<<8192,256>>>(NNZ,d_col,d_val,d_y); }
void L_rowptr(){ k_rowptr<<<8192,256>>>(NROWS,d_rp,d_y); }
void L_warp(){ k_warprow<<<(NROWS*32+255)/256,256>>>(NROWS,d_rp,d_col,d_val,d_x,d_y); }
int main(int argc,char**argv){
  std::string d = "../inputs/", n = argv[1];
  NROWS = atoi(argv[2]); NNZ = atoi(argv[3]); int NCOLS = atoi(argv[4]);
  auto rp = rd<int>(d+n+".rowptr.bin", NROWS+1);
  auto co = rd<int>(d+n+".col.bin", NNZ);
  auto va = rd<float>(d+n+".val.bin", NNZ);
  std::vector<float> x(NCOLS, 1.5f);
  cudaMalloc(&d_rp,4*(NROWS+1)); cudaMalloc(&d_col,4L*NNZ); cudaMalloc(&d_val,4L*NNZ);
  cudaMalloc(&d_x,4L*NCOLS); cudaMalloc(&d_y,4L*std::max(NROWS,1));
  cudaMemcpy(d_rp,rp.data(),4L*(NROWS+1),cudaMemcpyHostToDevice);
  cudaMemcpy(d_col,co.data(),4L*NNZ,cudaMemcpyHostToDevice);
  cudaMemcpy(d_val,va.data(),4L*NNZ,cudaMemcpyHostToDevice);
  cudaMemcpy(d_x,x.data(),4L*NCOLS,cudaMemcpyHostToDevice);
  printf("%-10s  stream(col+val) %8.1f us | rowptr only %8.1f us | flat gather %8.1f us "
         "| warp-per-row %9.1f us\n", n.c_str(),
         timeit(L_stream), timeit(L_rowptr), timeit(L_flat), timeit(L_warp));
  return 0;
}
