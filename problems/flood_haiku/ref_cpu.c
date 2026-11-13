#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Scalars (MAP_W, MAP_H, LEVEL, SRC_X, SRC_Y) arrive as -D macros; the contract is
// pointer args only, in inputs.yaml `args` order.
void flood_reference(const int* heights, int* flooded) {
    const long n = (long)MAP_W * (long)MAP_H;
    for (long i = 0; i < n; ++i) flooded[i] = 0;

    const long src = (long)SRC_Y * (long)MAP_W + (long)SRC_X;
    if (heights[src] > LEVEL) return;   /* source is dry: nothing floods */

    int* queue = (int*)malloc(sizeof(int) * (size_t)n);
    if (!queue) return;
    long head = 0, tail = 0;

    flooded[src] = 1;
    queue[tail++] = (int)src;

    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    while (head < tail) {
        const long c = (long)queue[head++];
        const int cx = (int)(c % (long)MAP_W);
        const int cy = (int)(c / (long)MAP_W);
        for (int k = 0; k < 4; ++k) {
            const int nx = cx + dx[k];
            const int ny = cy + dy[k];
            if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) continue;
            const long ni = (long)ny * (long)MAP_W + (long)nx;
            if (!flooded[ni] && heights[ni] <= LEVEL) {
                flooded[ni] = 1;
                queue[tail++] = (int)ni;
            }
        }
    }
    free(queue);
}

#ifdef __cplusplus
}
#endif
