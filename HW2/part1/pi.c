#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define number_of_tosses 10000000

static inline double rand_minus1_to_1() {
    return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

int main() {
    int number_in_circle = 0;
    
    srand(time(NULL));

    for (int64_t toss = 0; toss < number_of_tosses; toss++) {
        double x = rand_minus1_to_1();
        double y = rand_minus1_to_1();
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) number_in_circle++;
    }

    double pi_estimate = 4 * number_in_circle / ((double)number_of_tosses);

    printf("%.15lf\n", pi_estimate);
    // printf("pi = %.15lf\n", (double)3.14159265358979323846);

    return 0;
}
