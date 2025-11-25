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
    // 每個 Thread 處理 4 個像素 (int4 的寬度)
    int tIdX = (blockIdx.x * blockDim.x + threadIdx.x) * 4; 
    // int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdX >= res_x || thisY >= res_y) return;

    // 準備 4 個像素的座標
    float x[4], y[4];
    float c_x[4], c_y[4];
    int iter[4] = {0, 0, 0, 0};
    bool active[4] = {true, true, true, true};
    int active_count = 4;

    float y_tmp = fmaf(thisY, step_y, lower_y);
    
#pragma unroll
    for(int k = 0; k < 4; k++) {
        x[k] = fmaf(tIdX + k, step_x, lower_x);
        y[k] = y_tmp;
        c_x[k] = x[k];
        c_y[k] = y[k];
    }

    int i;
    for (i = 0; i < max_iterations; ++i) {
        if (active_count == 0) break;

#pragma unroll
        for(int k = 0; k < 4; k++) {
            if (active[k]) {
                float c_x_sq = c_x[k] * c_x[k];
                float c_y_sq = c_y[k] * c_y[k];

                if (c_x_sq + c_y_sq > 4.f) {
                    active[k] = false;
                    iter[k] = i;
                    active_count--;
                } else {
                    float temp_x = c_x_sq - c_y_sq + x[k];
                    c_y[k] = fmaf(2.0f * c_x[k], c_y[k], y[k]);
                    c_x[k] = temp_x;
                }
            }
        }
    }
    
#pragma unroll
    for(int k = 0; k < 4; k++) {
        if (active[k]) iter[k] = max_iterations;
    }

    // img[thisY * res_x + thisX] = i;
    int output_idx = (thisY * res_x + tIdX) / 4; 
    reinterpret_cast<int4*>(img)[output_idx] = make_int4(iter[0], iter[1], iter[2], iter[3]);
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

    dim3 blockSize(8, 8);
    dim3 gridSize((res_x / 4) / blockSize.x, res_y / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
}