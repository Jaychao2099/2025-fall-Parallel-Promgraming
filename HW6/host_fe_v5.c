#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static float FILTER_1[49] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,2,0,2,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static float FILTER_2[9] = {0,0,1,0,1,0,0,0,1};
static float FILTER_3[25] = {0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0};

static inline __attribute__((always_inline)) bool is_filter_same(float *a, float *b, int count) {
    for (int i = 0; i < count; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    // 用 static OpenCL 物件，避免重複創建
    static cl_command_queue queue = NULL;
    static cl_kernel k_default = NULL;
    static cl_kernel k_f1 = NULL;
    static cl_kernel k_f2 = NULL;
    static cl_kernel k_f3 = NULL;
    
    static cl_mem input_buffer_d = NULL;
    static cl_mem output_buffer_d = NULL;
    static cl_mem filter_buffer_d = NULL;

    // 紀錄上一次的狀態，用來判斷是否需要重新分配 Buffer
    static int prev_image_width = 0;
    static int prev_image_height = 0;
    static float *prev_input_ptr = NULL;
    static float *prev_output_ptr = NULL;

    cl_int status;
    int image_size = image_width * image_height * sizeof(float);
    int filter_size = filter_width * filter_width * sizeof(float);

    // 初始化 Command Queue, Kernels
    if (queue == NULL) {
        queue = clCreateCommandQueue(*context, *device, 0, &status);
        
        k_default = clCreateKernel(*program, "convolution", &status);
        k_f1 = clCreateKernel(*program, "convolution_f1", &status);
        k_f2 = clCreateKernel(*program, "convolution_f2", &status);
        k_f3 = clCreateKernel(*program, "convolution_f3", &status);
    }

    // 管理 Buffers (CL_MEM_USE_HOST_PTR)
    // 圖片尺寸改變，或者傳入的 Host 指標改變時，才重新創建 Buffer
    bool need_new_buffers = (image_width != prev_image_width) || 
                            (image_height != prev_image_height) || 
                            (input_image != prev_input_ptr) || 
                            (output_image != prev_output_ptr);

    if (need_new_buffers) {
        if (input_buffer_d) clReleaseMemObject(input_buffer_d);
        if (output_buffer_d) clReleaseMemObject(output_buffer_d);
        
        // CL_MEM_USE_HOST_PTR 讓 GPU 直接使用傳入的 input_image/output_image 記憶體
        input_buffer_d = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                                      image_size, input_image, &status);
        output_buffer_d = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
                                       image_size, output_image, &status);
        
        // 更新紀錄
        prev_image_width = image_width;
        prev_image_height = image_height;
        prev_input_ptr = input_image;
        prev_output_ptr = output_image;
    }

    // Filter Buffer
    // 只有當 buffer 不存在時才創建
    if (filter_buffer_d == NULL) {
        // 每次重建 filter buffer (因為它很小，開銷低)
        filter_buffer_d = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filter_size, filter, &status);
    } else {
        // 如果已經存在，釋放後重建
        clReleaseMemObject(filter_buffer_d);
        filter_buffer_d = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filter_size, filter, &status);
    }

    // 選 Kernel
    cl_kernel kernel = k_default;
    bool use_optimized = false;

    if (is_filter_same(filter, FILTER_1, 49)) {
        kernel = k_f1;
        use_optimized = true;
    } else if (is_filter_same(filter, FILTER_2, 9)) {
        kernel = k_f2;
        use_optimized = true;
    } else if (is_filter_same(filter, FILTER_3, 25)) {
        kernel = k_f3;
        use_optimized = true;
    }

    // 設定參數
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&image_height);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&image_width);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&input_buffer_d);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&output_buffer_d);
    if (!use_optimized) {
        clSetKernelArg(kernel, 4, sizeof(int), (void*)&filter_width);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&filter_buffer_d);
        clSetKernelArg(kernel, 6, filter_size, NULL);       // Local memory size argument
    }

    // 執行 Kernel
    size_t localws[2] = {25, 25};
    size_t globalws[2] = {image_width, image_height};
    
    // 不需要 clEnqueueWriteBuffer (input)，因為用了 CL_MEM_USE_HOST_PTR
    clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);

    // 取回結果 (使用 clEnqueueMapBuffer 取代 ReadBuffer)
    // MapBuffer 等待 Kernel 結束並確保 Host 端記憶體可讀
    void *mapped_ptr = clEnqueueMapBuffer(queue, output_buffer_d, CL_TRUE, CL_MAP_READ, 
                                        0, image_size, 0, NULL, NULL, &status);
    
    // 用完必須 Unmap
    clEnqueueUnmapMemObject(queue, output_buffer_d, mapped_ptr, 0, NULL, NULL);
}