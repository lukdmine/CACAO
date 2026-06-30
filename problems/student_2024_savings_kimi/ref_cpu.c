#ifdef __cplusplus
extern "C" {
#endif

void savings_reference(
    int* changes,
    int* account,
    int* sum
) {
    // first period: copy initial deposits
    for (int i = 0; i < CLIENTS; i++)
        account[i] = changes[i];

    // column-wise prefix sum: cumulative balance per customer
    for (int j = 1; j < PERIODS; j++) {
        for (int i = 0; i < CLIENTS; i++) {
            account[j * CLIENTS + i] = account[(j - 1) * CLIENTS + i]
                + changes[j * CLIENTS + i];
        }
    }

    // row-wise reduction: total money per period
    for (int j = 0; j < PERIODS; j++) {
        int s = 0;
        for (int i = 0; i < CLIENTS; i++) {
            s += account[j * CLIENTS + i];
        }
        sum[j] = s;
    }
}

#ifdef __cplusplus
}
#endif
