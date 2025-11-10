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

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // int64_t number_in_circle = 0LL;
    uint32_t seed = world_rank * time(NULL);

    // TODO: binary tree redunction
    
    int64_t local_result = toss(tosses / world_size, seed);
    for (int i = world_rank, j = 1; j <= (world_size>>1); i >>= 1, j <<= 1) {
        if (i & 0x1) {
            // printf("i'm %d, i'm done, i = %d\n", world_rank, i);
            MPI_Send(&local_result, 1, MPI_LONG_LONG, world_rank-j, 0, MPI_COMM_WORLD);
            break;
        } else {
            // printf("i'm %d, i = %d\n", world_rank, i);
            MPI_Status status;
            int64_t neighbor_result;
            MPI_Recv(&neighbor_result, 1, MPI_LONG_LONG, world_rank+j, 0, MPI_COMM_WORLD, &status);
            local_result += local_result;
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result =  4.0 * local_result / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
