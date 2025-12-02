__kernel void convolution(int filter_width,
                          __global float *filter,
                          int image_height,
                          int image_width,
                          __global float *input_image,
                          __global float *output_image) 
{
    const int ix = get_global_id(0); // X 座標 (Column)
    const int iy = get_global_id(1); // Y 座標 (Row)

    // Iterate over the rows of the source image
    int halffilter_size = filter_width / 2;
    float sum = 0.0f; // Reset sum for new source pixel
    // Apply the filter to the neighborhood
    for (int k = -halffilter_size; k <= halffilter_size; k++) {
        for (int l = -halffilter_size; l <= halffilter_size; l++) {
            if (iy + k >= 0 && iy + k < image_height && ix + l >= 0 && ix + l < image_width) {
                sum += input_image[((iy + k) * image_width) + ix + l] * filter[((k + halffilter_size) * filter_width) + l + halffilter_size];
            }
        }
    }
    output_image[(iy * image_width) + ix] = sum;
}
