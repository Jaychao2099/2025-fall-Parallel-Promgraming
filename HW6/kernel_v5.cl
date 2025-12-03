__constant float FILTER_1[49] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,2,0,2,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
__constant float FILTER_2[9] = {0,0,1,0,1,0,0,0,1};
__constant float FILTER_3[25] = {0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0};

__kernel void convolution(
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image,
    int filter_width,
    __global float *filter,
    __local float *filter_local
) {
    const int ix = get_global_id(0); // X 座標 (Column)
    const int iy = get_global_id(1); // Y 座標 (Row)

    int local_id = get_local_id(0) + get_local_id(1) * get_local_size(0);
    int filter_len = filter_width * filter_width;
    if (local_id < filter_len) {
        filter_local[local_id] = filter[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iterate over the rows of the source image
    int halffilter_size = filter_width / 2;
    float sum = 0.0f; // Reset sum for new source pixel
    // Apply the filter to the neighborhood
    int k_start = (-halffilter_size + iy >= 0) ? -halffilter_size : 0;
    int k_end = (halffilter_size + iy < image_height) ? halffilter_size : (halffilter_size + iy - image_height - 1);
    int l_start = (-halffilter_size + ix >= 0) ? -halffilter_size : 0;
    int l_end = (halffilter_size + ix < image_width) ? halffilter_size : (halffilter_size + ix - image_width - 1);

    for (int k = k_start; k <= k_end; k++) {
        for (int l = l_start; l <= l_end; l++) {
            sum += input_image[((iy + k) * image_width) + ix + l] * filter_local[((k + halffilter_size) * filter_width) + l + halffilter_size];
        }
    }
    output_image[(iy * image_width) + ix] = sum;
}

__kernel void convolution_f1(
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    // int halffilter_size = filter_width / 2;
    const int halffilter_size = 3;
    float sum = 0.0f;
    
    int k_start = (-halffilter_size + iy >= 0) ? -halffilter_size : 0;
    int k_end = (halffilter_size + iy < image_height) ? halffilter_size : (halffilter_size + iy - image_height - 1);
    int l_start = (-halffilter_size + ix >= 0) ? -halffilter_size : 0;
    // int l_end = (halffilter_size + ix < image_width) ? halffilter_size : (halffilter_size + ix - image_width - 1);

    for (int k = k_start; k <= k_end; k++) {
        int l = l_start;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_1[((k + halffilter_size) * 7) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_1[((k + halffilter_size) * 7) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_1[((k + halffilter_size) * 7) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_1[((k + halffilter_size) * 7) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_1[((k + halffilter_size) * 7) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_1[((k + halffilter_size) * 7) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_1[((k + halffilter_size) * 7) + l + halffilter_size];
    }
    output_image[(iy * image_width) + ix] = sum;
}

__kernel void convolution_f2(
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    // int halffilter_size = filter_width / 2;
    const int halffilter_size = 1;
    float sum = 0.0f;
    
    int k_start = (-halffilter_size + iy >= 0) ? -halffilter_size : 0;
    int k_end = (halffilter_size + iy < image_height) ? halffilter_size : (halffilter_size + iy - image_height - 1);
    int l_start = (-halffilter_size + ix >= 0) ? -halffilter_size : 0;
    // int l_end = (halffilter_size + ix < image_width) ? halffilter_size : (halffilter_size + ix - image_width - 1);

    for (int k = k_start; k <= k_end; k++) {
        int l = l_start;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_2[((k + halffilter_size) * 3) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_2[((k + halffilter_size) * 3) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_2[((k + halffilter_size) * 3) + l + halffilter_size];
    }
    output_image[(iy * image_width) + ix] = sum;
}

__kernel void convolution_f3(
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    // int halffilter_size = filter_width / 2;
    const int halffilter_size = 2;
    float sum = 0.0f;
    
    int k_start = (-halffilter_size + iy >= 0) ? -halffilter_size : 0;
    int k_end = (halffilter_size + iy < image_height) ? halffilter_size : (halffilter_size + iy - image_height - 1);
    int l_start = (-halffilter_size + ix >= 0) ? -halffilter_size : 0;
    // int l_end = (halffilter_size + ix < image_width) ? halffilter_size : (halffilter_size + ix - image_width - 1);

    for (int k = k_start; k <= k_end; k++) {
        int l = l_start;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_3[((k + halffilter_size) * 5) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_3[((k + halffilter_size) * 5) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_3[((k + halffilter_size) * 5) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_3[((k + halffilter_size) * 5) + l + halffilter_size]; l++;
        sum += input_image[((iy + k) * image_width) + ix + l] * FILTER_3[((k + halffilter_size) * 5) + l + halffilter_size];
    }
    output_image[(iy * image_width) + ix] = sum;
}