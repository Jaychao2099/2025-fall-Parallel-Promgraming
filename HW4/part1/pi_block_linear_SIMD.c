#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>

int64_t toss(int64_t number_of_tosses, int my_seed) {
    int64_t tmp_number = 0LL;
    for (int64_t toss_cnt = 0; toss_cnt < number_of_tosses; toss_cnt++) {
        double x = ((double)rand_r(&my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
        double y = ((double)rand_r(&my_seed) / (double)RAND_MAX) * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) tmp_number++;
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

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int64_t number_in_circle = 0LL;

    srand(world_rank * time(NULL));

    if (world_rank > 0)
    {
        // TODO: handle workers
        int64_t local_result = toss(tosses / world_size, rand());
        MPI_Send(&local_result, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: main
        MPI_Status status;
        number_in_circle += toss(tosses / world_size, rand());
        for (int i = 1; i < world_size; i++) {
            int64_t local_result;
            MPI_Recv(&local_result, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &status);
            number_in_circle += local_result;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result =  4.0 * number_in_circle / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
