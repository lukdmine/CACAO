#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void galaxies_reference(
    float* Ax,
    float* Ay,
    float* Az,
    float* Bx,
    float* By,
    float* Bz,
    float* result
) {
    float diff = 0.0f;
    for (int i = 0; i < N - 1; i++) {
        float tmp = 0.0f;
        for (int j = i + 1; j < N; j++) {
            float da = sqrt((Ax[i] - Ax[j]) * (Ax[i] - Ax[j])
                + (Ay[i] - Ay[j]) * (Ay[i] - Ay[j])
                + (Az[i] - Az[j]) * (Az[i] - Az[j]));
            float db = sqrt((Bx[i] - Bx[j]) * (Bx[i] - Bx[j])
                + (By[i] - By[j]) * (By[i] - By[j])
                + (Bz[i] - Bz[j]) * (Bz[i] - Bz[j]));
            tmp += (da - db) * (da - db);
        }
        diff += tmp;
    }

    result[0] = sqrt(1 / ((float)N * ((float)N - 1)) * diff);
}

#ifdef __cplusplus
}
#endif
