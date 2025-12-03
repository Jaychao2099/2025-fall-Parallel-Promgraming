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
    int image_size = image_width * image_height * sizeof(float);
    int filter_size = filter_width * filter_width * sizeof(float);
    // cl_int ciErrNum;

    // comand queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, NULL);

    // buffer
    cl_mem input_buffer_d = clCreateBuffer(*context, CL_MEM_READ_ONLY, image_size, NULL, NULL);
    cl_mem filter_buffer_d = clCreateBuffer(*context, CL_MEM_READ_ONLY, filter_size, NULL, NULL);
    cl_mem output_buffer_d = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, NULL);

    // move to device
    clEnqueueWriteBuffer(queue, filter_buffer_d, CL_FALSE, 0, filter_size, (void *)filter, 0, NULL,  NULL);
    clEnqueueWriteBuffer(queue, input_buffer_d, CL_TRUE, 0, image_size, (void *)input_image, 0, NULL,  NULL);

    char *kernel_name = "convolution";
    bool use_optimized = false;

    if (is_filter_same(filter, FILTER_1, 49)) {
        kernel_name = "convolution_f1";
        use_optimized = true;
    } else if (is_filter_same(filter, FILTER_2, 9)) {
        kernel_name = "convolution_f2";
        use_optimized = true;
    } else if (is_filter_same(filter, FILTER_3, 25)) {
        kernel_name = "convolution_f3";
        use_optimized = true;
    }

    // kernel
    cl_kernel kernel = clCreateKernel(*program, kernel_name, NULL);
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&image_height);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&image_width);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&input_buffer_d);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&output_buffer_d);
    if (!use_optimized) {
        clSetKernelArg(kernel, 4, sizeof(int), (void*)&filter_width);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&filter_buffer_d);
        clSetKernelArg(kernel, 6, filter_size, NULL);   // local
    }

    // run
    // 600 * 400
    size_t localws[2] = {25,25};
    size_t globalws[2] = {image_width, image_height};
    clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);

    // move result to host
    clEnqueueReadBuffer(queue, output_buffer_d, CL_TRUE, 0, image_size,  (void *)output_image, NULL, NULL, NULL);
}
