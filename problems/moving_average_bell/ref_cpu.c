/*
 * ============================================================================
 *  Bell-weighted moving average -- CPU reference implementation
 * ============================================================================
 *
 *      out[i] = ( sum_{j=-R..R-1} w_j * in[clamp(i+j)] ) / sum_j w_j
 *
 *  with the weight of tap j given by a bell-shaped window of the normalized
 *  offset d = j/R, which is exactly zero at the ends of the window:
 *
 *      phi(d) = exp(-ALPHA*d*d) - exp(-ALPHA)
 *
 *  a Gaussian shifted down so that it is exactly zero at |d| = 1.  It is
 *  non-negative, single-peaked, and costs one exponential per tap.
 *
 * ============================================================================
 */

#include <math.h>

extern "C" {

static float windowW(float d)
{
    return expf(-ALPHA * d * d) - expf(-ALPHA);
}

/* Pointers only, in inputs.yaml order (in, out); N, R and ALPHA arrive as -D macros. */
void moving_average_reference(const float *in, float *out)
{
    /* The weights depend only on j: compute them once, not once per output element. */
    float  w[2 * R];
    double norm = 0.0;
    for (int j = -R; j < R; j++) {
        w[j + R] = windowW((float)j / (float)R);
        norm += (double)w[j + R];
    }

    for (int i = 0; i < N; i++) {
        float acc = 0.0f;
        for (int j = -R; j < R; j++) {
            int idx = i + j;                /* clamp at both ends of the array */
            if (idx < 0)  idx = 0;
            if (idx >= N) idx = N - 1;
            acc += w[j + R] * in[idx];
        }
        out[i] = (float)(acc / norm);
    }
}

}
