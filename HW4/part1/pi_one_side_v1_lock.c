#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>

#define SIMD_SIZE 4

static inline __attribute__((always_inline)) int64_t toss(int64_t number_of_tosses, uint32_t my_seed) {
    int64_t tmp_number = 0LL;
    double x[SIMD_SIZE], y[SIMD_SIZE];
    double distance_squared[SIMD_SIZE];
    int hit[SIMD_SIZE];

    for (int64_t toss_cnt = 0; toss_cnt < number_of_tosses; toss_cnt += SIMD_SIZE) {
        for (int i = 0; i < SIMD_SIZE; i++) {
            x[i] = ((double)rand_r(&my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
            y[i] = ((double)rand_r(&my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
            distance_squared[i] = x[i] * x[i] + y[i] * y[i];
            hit[i] = (distance_squared[i] <= 1.0);
        }
        for (int i = 0; i < SIMD_SIZE; i++) {
            tmp_number += hit[i];
        }
    }
    return tmp_number;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int64_t number_in_circle = 0LL;
    uint32_t seed = (world_rank + 42069) ^ time(NULL);
    
    if (world_rank == 0)
    {
        // Main
        int64_t *all_result;
        // Use MPI to allocate memory for the target window
        MPI_Alloc_mem((long)(world_size * sizeof(int64_t)), MPI_INFO_NULL, (void *)&all_result);
        for (int i = 0; i < world_size; i++) all_result[i] = 0; // init

        // Create a window. Set the displacement unit to sizeof(int) to simplify the addressing at the originator processes
        // int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)
        MPI_Win_create(all_result, (long)(world_size * sizeof(int64_t)), sizeof(int64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        all_result[0] = toss(tosses / world_size, seed);

        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < world_size; i++) {
            number_in_circle += all_result[i];
        }
        MPI_Free_mem(all_result);
    }
    else
    {
        // Workers
        // Worker processes do not expose memory in the window
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        // Register with the main
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
        int64_t local_result = toss(tosses / world_size, seed);
        MPI_Put(&local_result, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * number_in_circle / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
