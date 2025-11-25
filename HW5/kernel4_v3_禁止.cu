#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__device__ __forceinline__ 
void calculate_pixel(float x0, float y0, int* output_ptr, int max_iterations) {

    float y2 = y0 * y0;
    
    // 左邊的大圓
    // (x+1)^2 + y^2 <= 1/16
    // if ((x0 + 1.f) * (x0 + 1.f) + y2 <= 0.0625f) {
    if (fmaf(x0 + 1.f, x0 + 1.f, y2) <= 0.0625f) {
        *output_ptr = max_iterations;
        return;
    }

    // 中間的心臟
    // q = (x - 1/4)^2 + y^2
    // q * (q + (x - 1/4)) <= 1/4 * y^2
    // float q = (x0 - 0.25f) * (x0 - 0.25f) + y2;
    float q = fmaf(x0 - 0.25f, x0 - 0.25f, y2);
    if (q * (q + (x0 - 0.25f)) <= 0.25f * y2) {
        *output_ptr = max_iterations;
        return;
    }
    // --------------------------------------------------

    float tmp_x = x0;
    float tmp_y = y0;

    int i;
    for (i = 0; i < max_iterations; ++i) {
        float tmp_x_sq = tmp_x * tmp_x;
        float tmp_y_sq = tmp_y * tmp_y;

        if (tmp_x_sq + tmp_y_sq > 4.f)
            break;
        
        tmp_y = fmaf(2.0f * tmp_x, tmp_y, y0);
        tmp_x = tmp_x_sq - tmp_y_sq + x0;
    }

    *output_ptr = i;
}

__global__ 
void mandel_kernel_balanced(float lower_x, float lower_y, float step_x, float step_y, int * __restrict__ img, int res_x, int res_y, int max_iterations)
{
    
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Grid-Stride Loop
    // 如果圖像中間很慢，所有 SM 都能分到一點中間的任務，
    // int stride_y = gridDim.y * blockDim.y;

    // // 每個 thread 負責一條垂直線上的多個點
    // for (int y = thisY; y < res_y; y += stride_y) {
    //     if (thisX < res_x) {
    //         float c_x = fmaf(thisX, step_x, lower_x);
    //         float c_y = fmaf(y    , step_y, lower_y);
            
    //         calculate_pixel(c_x, c_y, &img[y * res_x + thisX], max_iterations);
    //     }
    // }
    
    if (thisX >= res_x || thisY >= res_y) return;

    float x = fmaf(thisX, step_x, lower_x);
    float y = fmaf(thisY, step_y, lower_y);

    calculate_pixel(x, y, &img[thisY * res_x + thisX], max_iterations);
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int * __restrict__ img, int res_x, int res_y, int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    static int *d_img = nullptr;
    size_t size = res_x * res_y * sizeof(int);
    
    if (d_img == nullptr) {
        cudaMalloc(&d_img, size);
    }

    dim3 blockSize(8, 8);
    
    dim3 gridSize(res_x / blockSize.x, res_y / blockSize.y);

    mandel_kernel_balanced<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);
}