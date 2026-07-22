#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scalars COV_M (observations) and COV_K (variables) arrive as -D macros. */
void covariance_reference(const float* data, float* cov) {
    double* mean = (double*)malloc(sizeof(double) * (size_t)COV_K);
    if (!mean) return;

    for (int j = 0; j < COV_K; ++j) {
        double s = 0.0;
        for (int i = 0; i < COV_M; ++i) s += (double)data[(size_t)i * COV_K + j];
        mean[j] = s / (double)COV_M;
    }

    /* Symmetric: compute the upper triangle and mirror it. */
    for (int i = 0; i < COV_K; ++i) {
        for (int j = i; j < COV_K; ++j) {
            double s = 0.0;
            for (int m = 0; m < COV_M; ++m) {
                const double a = (double)data[(size_t)m * COV_K + i] - mean[i];
                const double b = (double)data[(size_t)m * COV_K + j] - mean[j];
                s += a * b;
            }
            const float v = (float)(s / (double)(COV_M - 1));
            cov[(size_t)i * COV_K + j] = v;
            cov[(size_t)j * COV_K + i] = v;
        }
    }
    free(mean);
}

#ifdef __cplusplus
}
#endif
