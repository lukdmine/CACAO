// =============================================================================
// Reference implementation of the 1D Moving Average kernel.
//
// For each output element i:
//   out[i] = sum(in[clamp(j, 0, n-1)], j = i-R .. i+R-1) / (2*R)
//
// When the window extends past array boundaries, the first or last element
// is used (clamping), matching the CPU reference implementation.
//
// Problem scalars (available as #define constants from problem.yaml):
//   N      - number of elements in the input/output arrays
//   R      - half the window size (window has 2*R elements)
//
// Vector arguments (function parameters):
//   input  - input float array of length N
//   average- output float array of length N (validated against this)
// =============================================================================

extern "C" __global__ void moving_average_reference(
    const float* input,
    float* average)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= N) return;

    float sum = 0.0f;
    for (int w = -R; w < R; w++) {
        int idx = i + w;
        // Clamp to valid range [0, N-1]
        if (idx < 0)    idx = 0;
        if (idx >= N)   idx = N - 1;
        sum += input[idx];
    }
    average[i] = sum / float(2 * R);
}

