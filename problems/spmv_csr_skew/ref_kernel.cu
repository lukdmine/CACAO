// Correctness oracle for CSR SpMV: y = A * x.
//
// One warp per row, strided within the row, shuffle reduction. Deliberately dull: this
// file decides what "right" means, so it is written to be read, not to be fast. KTT runs
// it exactly once per tuning process and caches the output, so its cost does not enter
// any measured number -- including the ~90 ms it spends on mawi's 7.4M-nonzero row, where
// one warp is left grinding after the other 18.5 million have finished. That tail is a
// fair preview of what the tuned kernel must NOT do.
//
// Argument order is the `boundary` from inputs.yaml: runtime scalars first, in
// declaration order, then every buffer in declaration order.

extern "C" __global__ void spmv_csr_reference(const int  num_rows,
                                              const int  num_entries,
                                              const int* __restrict__ row_offsets,
                                              const int* __restrict__ col_indices,
                                              const float* __restrict__ values,
                                              const float* __restrict__ x,
                                              float* __restrict__ y)
{
    (void)num_entries;

    const int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int row  = gid >> 5;
    const int lane = gid & 31;

    // Uniform across the warp (blockDim.x is a multiple of 32), so the whole warp
    // leaves together and the full-mask shuffles below stay legal.
    if (row >= num_rows)
        return;

    const int begin = row_offsets[row];
    const int end   = row_offsets[row + 1];

    float sum = 0.0f;
    for (int j = begin + lane; j < end; j += 32)
        sum += values[j] * x[col_indices[j]];

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);

    if (lane == 0)
        y[row] = sum;
}
