#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__global__ 
void mandel_kernel(float lower_x, float lower_y, 
                   float step_x, float step_y, 
                   int * __restrict__ img, 
                   const unsigned int res_x, const unsigned int res_y, 
                   int max_iterations)
{
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid-Stride Loop
    // 如果圖像中間很慢，所有 SM 都能分到一點中間的任務
    int stride_y = gridDim.y * blockDim.y;

    // 每個 thread 負責一條垂直線上的多個點
    for (int yy = thisY; yy < res_y; yy += stride_y) {
        if (thisX < res_x) {
            float x = fmaf(thisX, step_x, lower_x);
            float y = fmaf(yy   , step_y, lower_y);
            
            float c_x = x;
            float c_y = y;

            int i;
            for (i = 0; i < max_iterations; ++i) {
                float c_x_sq = c_x * c_x;
                float c_y_sq = c_y * c_y;

                if (c_x_sq + c_y_sq > 4.f)
                    break;

                c_y = fmaf(2.0f * c_x, c_y, y);
                c_x = c_x_sq - c_y_sq + x;
            }

            img[yy * res_x + thisX] = i;
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

    dim3 blockSize(8, 48);
    dim3 gridSize(res_x / blockSize.x, res_y / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
}