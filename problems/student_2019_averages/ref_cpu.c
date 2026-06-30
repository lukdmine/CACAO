#ifdef __cplusplus
extern "C" {
#endif

void averages_reference(
    const int* results,
    float* avg_stud,
    float* avg_que
) {
    for (int s = 0; s < STUDENTS; s++) {
        int stud = 0;
        for (int q = 0; q < QUESTIONS; q++) {
            stud += results[s * QUESTIONS + q];
        }
        avg_stud[s] = (float)stud / (float)QUESTIONS;
    }

    for (int q = 0; q < QUESTIONS; q++) {
        int que = 0;
        for (int s = 0; s < STUDENTS; s++) {
            que += results[s * QUESTIONS + q];
        }
        avg_que[q] = (float)que / (float)STUDENTS;
    }
}

#ifdef __cplusplus
}
#endif
