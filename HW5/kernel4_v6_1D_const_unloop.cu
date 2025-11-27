#include <cstdio>
#include <cstdlib>
#include <cuda.h>

// 使用 Constant Memory 儲存全域參數
// 減少 Kernel 的參數傳遞開銷
__constant__ float c_lower_x;
__constant__ float c_lower_y;
__constant__ float c_step_x;
__constant__ float c_step_y;
__constant__ int c_res_x;
// __constant__ int c_res_y;
__constant__ int c_total_res;
__constant__ int c_max_iterations;

// #define c_res_x 1600
// #define total_res 1920000

__global__ 
void mandel_kernel(int * __restrict__ img)
{
    // 1D index，確保 Memory Coalescing, 是 CUDA 寫入記憶體最快的方式
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= c_total_res) return;
    
    // 直接計算 x, y 坐標, 相比於 2D Block 造成的潛在 Memory 分歧，1D 帶來的存取優勢更大
    // 在 max_iter-bound 的 kernel 中影響極小
    int r_x = idx % c_res_x;
    int r_y = idx / c_res_x;

    float x = fmaf(r_x, c_step_x, c_lower_x);
    float y = fmaf(r_y, c_step_y, c_lower_y);

    float c_x = x;
    float c_y = y;
    
    // 提前保存平方值
    float c_x2 = c_x * c_x;
    float c_y2 = c_y * c_y;

    int k = 0;
    int max_iter = c_max_iterations;

    // Manual Loop Unrolling, 減少 (k < max_iter) 的檢查次數
    while (k < max_iter) {
        #define ITER_STEP \
            if (c_x2 + c_y2 > 4.0f) { goto done; } \
            c_y = fmaf(2.0f * c_x, c_y, y); \
            c_x = c_x2 - c_y2 + x; \
            c_x2 = c_x * c_x; \
            c_y2 = c_y * c_y; \
            k++;

        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        ITER_STEP
        
        #undef ITER_STEP
    }

done:
    if (k > max_iter) k = max_iter;
    img[idx] = k;
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

    // 用 static, 只在第一次呼叫或解析度改變時分配
    static int *d_img = nullptr;

    const size_t total_res = res_x * res_y;
    
    size_t size = total_res * sizeof(int);

    if (d_img == nullptr) {
        cudaMalloc(&d_img, size);
    }

    // 更新 Constant Memory
    cudaMemcpyToSymbol(c_lower_x, &lower_x, sizeof(float));
    cudaMemcpyToSymbol(c_lower_y, &lower_y, sizeof(float));
    cudaMemcpyToSymbol(c_step_x, &step_x, sizeof(float));
    cudaMemcpyToSymbol(c_step_y, &step_y, sizeof(float));
    cudaMemcpyToSymbol(c_res_x, &res_x, sizeof(int));
    // cudaMemcpyToSymbol(c_res_y, &res_y, sizeof(int));
    cudaMemcpyToSymbol(c_total_res, &total_res, sizeof(int));
    cudaMemcpyToSymbol(c_max_iterations, &max_iterations, sizeof(int));

    int blockSize = 512;
    int gridSize = total_res / blockSize;

    mandel_kernel<<<gridSize, blockSize>>>(d_img);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);
}