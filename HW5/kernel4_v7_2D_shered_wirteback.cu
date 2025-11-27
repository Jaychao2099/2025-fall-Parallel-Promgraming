#include <cstdio>
#include <cstdlib>
#include <cuda.h>

// Constant Memory
__constant__ float c_lower_x;
__constant__ float c_lower_y;
__constant__ float c_step_x;
__constant__ float c_step_y;
__constant__ int c_res_x;
__constant__ int c_res_y;
__constant__ int c_max_iterations;

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define TOTAL_THREADS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y)

__global__ 
void mandel_kernel(int * __restrict__ img)
{
    const int res_x = c_res_x;
    const int res_y = c_res_y;

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= res_x || iy >= res_y) return;
    
    // Register Caching
    const float step_x = c_step_x;
    const float step_y = c_step_y;
    const float lower_x = c_lower_x;
    const float lower_y = c_lower_y;
    const int max_iter = c_max_iterations;

    // 1D Shared Memory
    __shared__ int s_cache[TOTAL_THREADS_PER_BLOCK];
    
    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    float x = fmaf(ix, step_x, lower_x);
    float y = fmaf(iy, step_y, lower_y);

    float c_x = x;
    float c_y = y;
    float c_x2 = x * x;
    float c_y2 = y * y;

    int k = 0;

    // Manual Loop Unrolling, 減少 (k < max_iter)的檢查次數
    while (k < max_iter) {
        #define ITER_STEP \
            if (c_x2 + c_y2 > 4.0f) { goto done; } \
            c_y = fmaf(2.0f * c_x, c_y, y); \
            c_x = c_x2 - c_y2 + x; \
            c_x2 = c_x * c_x; \
            c_y2 = c_y * c_y; \
            k++;

        ITER_STEP ITER_STEP ITER_STEP ITER_STEP
        ITER_STEP ITER_STEP ITER_STEP ITER_STEP
        ITER_STEP ITER_STEP ITER_STEP ITER_STEP
        ITER_STEP ITER_STEP ITER_STEP ITER_STEP
        #undef ITER_STEP
    }

done:
    if (k > max_iter) k = max_iter;

    // 寫入 Shared Memory
    s_cache[tid_in_block] = k;
    __syncthreads();

    // 寫回 Global Memory
    int global_idx = iy * res_x + ix;
    img[global_idx] = s_cache[tid_in_block];
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int * __restrict__ img, int res_x, int res_y, int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    // 用 static, 只在第一次呼叫時分配
    static int *d_img = nullptr;
    size_t size = res_x * res_y * sizeof(int);

    if (d_img == nullptr) cudaMalloc(&d_img, size);

    // Constant Memory
    cudaMemcpyToSymbol(c_lower_x, &lower_x, sizeof(float));
    cudaMemcpyToSymbol(c_lower_y, &lower_y, sizeof(float));
    cudaMemcpyToSymbol(c_step_x, &step_x, sizeof(float));
    cudaMemcpyToSymbol(c_step_y, &step_y, sizeof(float));
    cudaMemcpyToSymbol(c_res_x, &res_x, sizeof(int));
    cudaMemcpyToSymbol(c_res_y, &res_y, sizeof(int));
    cudaMemcpyToSymbol(c_max_iterations, &max_iterations, sizeof(int));

    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y); // 8, 8
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x, (res_y + blockSize.y - 1) / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(d_img);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);
}