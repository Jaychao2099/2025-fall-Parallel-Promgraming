#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define default_thread_cnt 6
#define default_number_of_tosses 100000000LL

pthread_mutex_t mutex;

typedef struct {
    int64_t *number_in_circle;
    int64_t number_of_tosses;
    int seed;
} thread_args_t;

void *toss(void *args) {
    int64_t *number_in_circle = ((thread_args_t *)args)->number_in_circle;
    int64_t number_of_tosses = ((thread_args_t *)args)->number_of_tosses;
    int my_seed = ((thread_args_t *)args)->seed;

    int64_t tmp_number = 0LL;
    for (int64_t toss_cnt = 0; toss_cnt < number_of_tosses; toss_cnt++) {
        double x = ((double)rand_r(&my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand_r(&my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) tmp_number++;
    }
    pthread_mutex_lock(&mutex);
    (*number_in_circle) += tmp_number;
    pthread_mutex_unlock(&mutex);
}

int main(int argc, char *argv[]) {
    int thread_cnt = default_thread_cnt;
    int64_t number_of_tosses = default_number_of_tosses;

    if (argc > 1) thread_cnt = atoi(argv[1]);
    if (argc > 2) number_of_tosses = atoll(argv[2]);

    srand(time(NULL));
    
    int64_t number_in_circle = 0LL;
    int64_t toss_per_thread = number_of_tosses / thread_cnt;

    pthread_t *threads = malloc(thread_cnt * sizeof(pthread_t));
    thread_args_t *args = malloc(thread_cnt * sizeof(thread_args_t));
    pthread_mutex_init(&mutex, NULL);
    
    for (int i = 0; i < thread_cnt; i++) {
        args[i] = (thread_args_t){&number_in_circle, toss_per_thread, rand()};
        pthread_create(&threads[i], NULL, toss, &args[i]);
    }

    for (int i = 0; i < thread_cnt; i++) {
        pthread_join(threads[i], NULL);
    }

    double pi_estimate = 4.0 * number_in_circle / ((double)number_of_tosses);

    printf("%lf\n", pi_estimate);

    pthread_mutex_destroy(&mutex);
    free(args);
    free(threads);
    return 0;
}
