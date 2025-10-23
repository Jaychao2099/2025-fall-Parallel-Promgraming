#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define default_thread_cnt 6
#define default_total_tosses 100000000LL

pthread_mutex_t mutex;
int64_t tosses_per_thread;
int64_t number_in_circle;

void *toss(void *my_seed) {
    int64_t tmp_number = 0LL;
    for (int64_t toss_cnt = 0; toss_cnt < tosses_per_thread; toss_cnt++) {
        double x = ((double)rand_r(my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand_r(my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) tmp_number++;
    }
    pthread_mutex_lock(&mutex);
    number_in_circle += tmp_number;
    pthread_mutex_unlock(&mutex);
}

int main(int argc, char *argv[]) {
    int thread_cnt = default_thread_cnt;
    int64_t total_tosses = default_total_tosses;

    if (argc > 1) thread_cnt = atoi(argv[1]);
    if (argc > 2) total_tosses = atoll(argv[2]);

    tosses_per_thread = total_tosses / thread_cnt + 1;

    srand(time(NULL));
    
    number_in_circle = 0LL;

    pthread_t *threads = malloc(thread_cnt * sizeof(pthread_t));
    uint32_t *seed = malloc(thread_cnt * sizeof(uint32_t));
    pthread_mutex_init(&mutex, NULL);
    
    for (int i = 0; i < thread_cnt; i++) {
        seed[i] = rand();
        pthread_create(&threads[i], NULL, toss, &seed[i]);
    }

    for (int i = 0; i < thread_cnt; i++) {
        pthread_join(threads[i], NULL);
    }

    double pi_estimate = 4.0 * number_in_circle / ((double)total_tosses);

    printf("%lf\n", pi_estimate);

    pthread_mutex_destroy(&mutex);
    free(seed);
    free(threads);
    return 0;
}
