#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, size_t pitch, int res_x, int res_y, int max_iterations)
{
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= res_x || thisY >= res_y) return;

    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    float x = lower_x + thisX * step_x;
    float y = lower_y + thisY * step_y;

    float c_x = x;
    float c_y = y;

    int i;
    for (i = 0; i < max_iterations; ++i) {
        if (c_x * c_x + c_y * c_y > 4.f)
            break;

        float new_c_x = (c_x * c_x) - (c_y * c_y);
        float new_c_y = 2.f * c_x * c_y;
        
        c_x = x + new_c_x;
        c_y = y + new_c_y;
    }

    img[thisY * pitch / sizeof(int) + thisX] = i;   // skip y rows (以 Byte 單位)
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
    cudaMallocPitch(&d_img, &pitch, res_x*sizeof(int), res_y);  // 有 padding (最後有空白)

    int *pinned_img;
    size_t size = res_x * res_y * sizeof(int);
    cudaHostAlloc(&pinned_img, size, cudaHostAllocDefault);

    dim3 blockSize(8, 8);
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x, (res_y + blockSize.y - 1) / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, pitch, res_x, res_y, max_iterations);

    cudaMemcpy2D(pinned_img, res_x * sizeof(int),   // dst, dst pitch
                 d_img, pitch,                      // src, src pitch
                 res_x * sizeof(int),       // 每 row 實際有效寬度
                 res_y, 
                 cudaMemcpyDeviceToHost);

    memcpy(img, pinned_img, size);
    
    cudaFree(d_img);
    cudaFreeHost(pinned_img);
}