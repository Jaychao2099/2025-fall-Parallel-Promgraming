#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define default_thread_cnt 6
#define default_number_of_tosses 100000000LL

static inline double rand_minus1_to_1(int *seed) {
    return ((double)rand_r(seed) / (double)RAND_MAX) * 2.0 - 1.0;
}

typedef struct {
    int64_t *number_in_circle;
    int64_t number_of_tosses;
    int thread_count;
    int thread_index;
    int seed;
} thread_args_t;

volatile int flag;

void *toss(void *args) {
    int64_t *number_in_circle = ((thread_args_t *)args)->number_in_circle;
    int64_t number_of_tosses = ((thread_args_t *)args)->number_of_tosses;
    int tread_cnt = ((thread_args_t *)args)->thread_count;
    int my_index = ((thread_args_t *)args)->thread_index;
    int my_seed = ((thread_args_t *)args)->seed;

    // printf("my_index = %d\n", my_index);

    int64_t tmp_number = 0LL;
    for (int64_t toss_cnt = 0; toss_cnt < number_of_tosses; toss_cnt++) {
        double x = rand_minus1_to_1(&my_seed);
        double y = rand_minus1_to_1(&my_seed);
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) tmp_number++;
    }
    while (flag != my_index);
    *number_in_circle += tmp_number;  
    flag = (flag+1) % tread_cnt;
}

int main(int argc, char *argv[]) {
    int thread_cnt = default_thread_cnt;
    int64_t number_of_tosses = default_number_of_tosses;

    if (argc > 1) thread_cnt = atoi(argv[1]);
    if (argc > 2) number_of_tosses = atoll(argv[2]);

    srand(time(NULL));
    
    int64_t number_in_circle = 0LL;
    int64_t toss_per_thread = number_of_tosses / thread_cnt;
    int64_t remaining_tosses = number_of_tosses % thread_cnt;

    pthread_t *threads = malloc(thread_cnt * sizeof(pthread_t));
    thread_args_t *args = malloc(thread_cnt * sizeof(thread_args_t));
    flag = 0;
    
    for (int i = 0; i < thread_cnt; i++) {
        int64_t real_toss_per_thread = toss_per_thread + (i < remaining_tosses ? 1 : 0);
        args[i] = (thread_args_t){&number_in_circle, real_toss_per_thread, thread_cnt, i, rand()};
        pthread_create(&threads[i], NULL, toss, &args[i]);
    }

    for (int i = 0; i < thread_cnt; i++) {
        pthread_join(threads[i], NULL);
    }

    double pi_estimate = 4 * number_in_circle / ((double)number_of_tosses);

    printf("%lf\n", pi_estimate);
    // printf("%.15lf\n", (double)3.14159265358979323846);

    free(args);
    free(threads);
    return 0;
}
