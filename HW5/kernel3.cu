#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, size_t pitch, int res_x, int res_y, int max_iterations)
{
    // Grid-Stride Loop (Y), 每次跳躍 "整個 Grid 的高度"
    for (int thisY = blockIdx.y * blockDim.y + threadIdx.y; thisY < res_y; thisY += gridDim.y * blockDim.y) {
        int* row = (int*)((char*)img + thisY * pitch);
        
        float y = lower_y + thisY * step_y;
        float c_y = y;

        // Grid-Stride Loop (X), 每次跳躍 "整個 Grid 的寬度"
        for (int thisX = blockIdx.x * blockDim.x + threadIdx.x; thisX < res_x; thisX += gridDim.x * blockDim.x) {
            float x = lower_x + thisX * step_x;
            float c_x = x;

            float tmp_x = c_x;
            float tmp_y = c_y;

            int count;
            for (count = 0; count < max_iterations; ++count) {
                if (tmp_x * tmp_x + tmp_y * tmp_y > 4.f)
                    break;

                float new_x = (tmp_x * tmp_x) - (tmp_y * tmp_y);
                float new_y = 2.f * tmp_x * tmp_y;
                tmp_x = c_x + new_x;
                tmp_y = c_y + new_y;
            }

            row[thisX] = count;
        }
    }
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

    int *d_img;
    size_t pitch;   // how many byte in 1 row in physics
    size_t width_bytes = res_x * sizeof(int);
    cudaMallocPitch(&d_img, &pitch, width_bytes, res_y);  // 有 padding (最後有空白)

    int *pinned_img;
    size_t size = width_bytes * res_y;
    cudaHostAlloc(&pinned_img, size, cudaHostAllocDefault);

    dim3 blockSize(8, 8);
    dim3 gridSize((res_x) / blockSize.x, (res_y) / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, pitch, res_x, res_y, max_iterations);

    cudaMemcpy2D(pinned_img, width_bytes,   // dst, dst pitch
                 d_img, pitch,              // src, src pitch
                 width_bytes,       // 每 row 實際有效寬度
                 res_y, 
                 cudaMemcpyDeviceToHost);

    memcpy(img, pinned_img, size);
    
    cudaFree(d_img);
    cudaFreeHost(pinned_img);
}