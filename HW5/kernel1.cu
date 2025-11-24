#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__global__ void mandel_kernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, int maxIterations, int *img)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    float x = lowerX + blockIdx.x * stepX;
    float y = lowerY + blockIdx.y * stepY;
    int idx = blockIdx.y * resX + blockIdx.x;
    // Compute the Mandelbrot iteration count for (x, y)
    int iteration = 0;
    float zx = 0.0f;
    float zy = 0.0f;
    while (zx * zx + zy * zy < 4.0f && iteration < maxIterations) {
        float temp = zx * zx - zy * zy + x;
        zy = 2.0f * zx * zy + y;
        zx = temp;
        iteration++;
    }
    img[idx] = iteration;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    float step_x_d, step_y_d;

    cudaMalloc(&step_x_d, sizeof(float));
    cudaMemcpy(step_x_d, step_x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&step_y_d, sizeof(float));
    cudaMemcpy(step_y_d, step_y, sizeof(float), cudaMemcpyHostToDevice);

    // int end_row = res_y / 2;

    // for (int j = 0; j < end_row; j++) {
    //     for (int i = 0; i < res_x; i++) {
    //         float x = lower_x + (float)i * step_x;
    //         float y = lower_y + (float)j * step_y;

    //         // img[j * res_x + i] = mandel_kernel(x, y, step_x, step_y, res_x, res_y, max_iterations);
    //     }
    // }
}
