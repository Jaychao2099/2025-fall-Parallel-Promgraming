#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__constant__ float c_lower_x;
__constant__ float c_lower_y;
__constant__ float c_step_x;
__constant__ float c_step_y;
__constant__ int c_res_x;
__constant__ int c_res_y;
__constant__ int c_max_iterations;

__global__ 
void mandel_kernel(int * __restrict__ img)
{
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    int *ptr = &img[thisY * c_res_x];

    // Grid-Stride Loop
    int stride_x = gridDim.x * blockDim.x;

    // 每個 thread 負責一條水平線上的多個點
    for (int xx = thisX; xx < c_res_x; xx += stride_x) {
        if (thisY < c_res_y) {
            float x = fmaf(xx   , c_step_x, c_lower_x);
            float y = fmaf(thisY, c_step_y, c_lower_y);
            
            float c_x = x;
            float c_y = y;

            int i;
            for (i = 0; i < c_max_iterations; ++i) {
                float c_x_sq = c_x * c_x;
                float c_y_sq = c_y * c_y;

                if (c_x_sq + c_y_sq > 4.f)
                    break;

                c_y = fmaf(2.0f * c_x, c_y, y);
                c_x = c_x_sq - c_y_sq + x;
            }

            // img[thisY * c_res_x + xx] = i;    // 保留反而更快???
            ptr[xx] = i;
        }
    }
}

// Host front-end function
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int * __restrict__ img,
             int res_x,
             int res_y,
             int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int *d_img = nullptr;
    const size_t size = res_x * res_y * sizeof(int);

    cudaMalloc(&d_img, size);
    
    cudaMemcpyToSymbol(c_lower_x, &lower_x, sizeof(float));
    cudaMemcpyToSymbol(c_lower_y, &lower_y, sizeof(float));
    cudaMemcpyToSymbol(c_step_x, &step_x, sizeof(float));
    cudaMemcpyToSymbol(c_step_y, &step_y, sizeof(float));
    cudaMemcpyToSymbol(c_res_x, &res_x, sizeof(int));
    cudaMemcpyToSymbol(c_res_y, &res_y, sizeof(int));
    cudaMemcpyToSymbol(c_max_iterations, &max_iterations, sizeof(int));

    dim3 blockSize(64, 8);
    dim3 gridSize(res_x / blockSize.x, res_y / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(d_img);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
}