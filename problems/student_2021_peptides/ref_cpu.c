// 7 = K, 12 = R, 1 = D, 2 = E
static int is_salt_bridge(const int a1, const int a2) {
    switch (a1) {
        case 7:
            if ((a2 == 1) || (a2 == 2))
                return 1;
            break;
        case 12:
            if ((a2 == 1) || (a2 == 2))
                return 1;
            break;
        case 1:
            if ((a2 == 7) || (a2 == 12))
                return 1;
            break;
        case 2:
            if ((a2 == 7) || (a2 == 12))
                return 1;
            break;
        default:
            break;
    }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

void peptides_reference(
    float* tableA,
    float* tableB,
    int* chains,
    float* primaryScore,
    float* secondaryScore
) {
    // iterate over sequences
    for (int i = 0; i < N; i++) {
        primaryScore[i] = 0.0f;
        secondaryScore[i] = 0.0f;
        // iterate over structure (18 amino acids per chain)
        for (int j = 0; j < 18; j++) {
            // stride-1 sliding window: chain i uses chains[i..i+17]
            primaryScore[i] += tableA[chains[i + j] * 18 + j];
            secondaryScore[i] += tableB[chains[i + j] * 18 + j];
            // cyclic position offsets for salt bridge check
            int idx5 = (j > 4) ? j - 5 : 18 - (5 - j);
            int idx3 = (j > 2) ? j - 3 : 18 - (3 - j);
            int idx2 = (j > 1) ? j - 2 : 18 - (2 - j);
            if ((is_salt_bridge(chains[idx5 + i], chains[j + i]))
            || (is_salt_bridge(chains[idx3 + i], chains[j + i]))
            || (is_salt_bridge(chains[idx2 + i], chains[j + i]))) {
                primaryScore[i] = 0.0f;
                secondaryScore[i] = 0.0f;
                break;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
