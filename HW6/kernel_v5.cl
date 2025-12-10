// __constant float FILTER_1[49] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,2,0,2,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
// __constant float FILTER_2[9] = {0,0,1,0,1,0,0,0,1};
// __constant float FILTER_3[25] = {0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0};

#define FILTER_WIDTH_1 7 
#define FILTER_WIDTH_2 3
#define FILTER_WIDTH_3 5
#define HALF_FILTER_1 (FILTER_WIDTH_1 >> 1)
#define HALF_FILTER_2 (FILTER_WIDTH_2 >> 1)
#define HALF_FILTER_3 (FILTER_WIDTH_3 >> 1)
#define TILE_W 25
#define TILE_H 25
// Local Memory 的寬度 = WorkGroup寬度 + 左右Padding
#define SMEM_W_1 (TILE_W + 2 * (HALF_FILTER_1))
#define SMEM_H_1 (TILE_H + 2 * (HALF_FILTER_1))

#define SMEM_W_2 (TILE_W + 2 * (HALF_FILTER_2))
#define SMEM_H_2 (TILE_H + 2 * (HALF_FILTER_2))

#define SMEM_W_3 (TILE_W + 2 * (HALF_FILTER_3))
#define SMEM_H_3 (TILE_H + 2 * (HALF_FILTER_3))

__kernel void convolution(
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image,
    int filter_width,
    __global float *filter,
    __local float *filter_local
) {
    const int halffilter_size = filter_width / 2;
    const int smem_w = TILE_W + 2 * halffilter_size;
    const int smem_h = TILE_H + 2 * halffilter_size;

    // __local float tile[smem_w][smem_h];
    __local float tile[50][50];

    int tx = get_local_id(0); // 0~24
    int ty = get_local_id(1); // 0~24
    const int ix = get_global_id(0); // X 座標 (Column)
    const int iy = get_global_id(1); // Y 座標 (Row)

    int thread_id = ty * TILE_W + tx;           // 0 ~ 624
    int filter_len = filter_width * filter_width;
    if (thread_id < filter_len) {
        filter_local[thread_id] = filter[thread_id];
    }
    // barrier(CLK_LOCAL_MEM_FENCE);
    
    int total_pixels_to_load = smem_h * smem_w; // 31 * 31 = 961
    int threads_per_group = TILE_W * TILE_H;    // 625

    for (int i = thread_id; i < total_pixels_to_load; i += threads_per_group) {
        int l_y = i / smem_w;
        int l_x = i % smem_w;
        
        // Global Memory 座標
        int global_col = (get_group_id(0) * TILE_W) - halffilter_size + l_x;
        int global_row = (get_group_id(1) * TILE_H) - halffilter_size + l_y;

        if (global_row >= 0 && global_row < image_height && global_col >= 0 && global_col < image_width) {
            tile[l_y][l_x] = input_image[global_row * image_width + global_col];
        } else {
            tile[l_y][l_x] = 0.0f; // Padding 0
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    // 已 padding，不用判斷邊界
    for (int k = -halffilter_size; k <= halffilter_size; k++) {
        for (int l = -halffilter_size; l <= halffilter_size; l++) {
            float pixel = tile[ty + halffilter_size + k][tx + halffilter_size + l];
            float f_val = filter_local[(k + halffilter_size) * filter_width + (l + halffilter_size)];
            sum += pixel * f_val;
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
    __local float tile[SMEM_W_1][SMEM_H_1];

    int tx = get_local_id(0); // 0~24
    int ty = get_local_id(1); // 0~24
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    int total_pixels_to_load = SMEM_H_1 * SMEM_W_1; // 31 * 31 = 961
    int thread_id = ty * TILE_W + tx;           // 0 ~ 624
    int threads_per_group = TILE_W * TILE_H;    // 625

    for (int i = thread_id; i < total_pixels_to_load; i += threads_per_group) {
        int l_y = i / SMEM_W_1;
        int l_x = i % SMEM_W_1;
        
        // Global Memory 座標
        int global_col = (get_group_id(0) * TILE_W) - HALF_FILTER_1 + l_x;
        int global_row = (get_group_id(1) * TILE_H) - HALF_FILTER_1 + l_y;

        if (global_row >= 0 && global_row < image_height && global_col >= 0 && global_col < image_width) {
            tile[l_y][l_x] = input_image[global_row * image_width + global_col];
        } else {
            tile[l_y][l_x] = 0.0f; // Padding 0
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    // center
    int c_r = ty + HALF_FILTER_1;
    int c_c = tx + HALF_FILTER_1;
    sum += tile[c_r - 1][c_c - 1];
    sum += tile[c_r - 1][c_c + 1];
    sum += tile[c_r    ][c_c - 1] * 2.0f;
    sum += tile[c_r    ][c_c + 1] * 2.0f;
    sum += tile[c_r + 1][c_c - 1];
    sum += tile[c_r + 1][c_c + 1];

    output_image[(iy * image_width) + ix] = sum;
}

__kernel void convolution_f2(
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image
) {
    __local float tile[SMEM_W_2][SMEM_H_2];

    int tx = get_local_id(0); // 0~24
    int ty = get_local_id(1); // 0~24
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    int total_pixels_to_load = SMEM_H_2 * SMEM_W_2; // 31 * 31 = 961
    int thread_id = ty * TILE_W + tx;           // 0 ~ 624
    int threads_per_group = TILE_W * TILE_H;    // 625

    for (int i = thread_id; i < total_pixels_to_load; i += threads_per_group) {
        int l_y = i / SMEM_W_2;
        int l_x = i % SMEM_W_2;
        
        // 對應到 Global Memory 座標
        int global_col = (get_group_id(0) * TILE_W) - HALF_FILTER_2 + l_x;
        int global_row = (get_group_id(1) * TILE_H) - HALF_FILTER_2 + l_y;

        if (global_row >= 0 && global_row < image_height && global_col >= 0 && global_col < image_width) {
            tile[l_y][l_x] = input_image[global_row * image_width + global_col];
        } else {
            tile[l_y][l_x] = 0.0f; // Padding 0
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float sum = 0.0f;
    // center
    int c_c = tx + HALF_FILTER_2;
    int c_r = ty + HALF_FILTER_2;
    sum += tile[c_r - 1][c_c + 1];
    sum += tile[c_r    ][c_c    ];
    sum += tile[c_r + 1][c_c + 1];

    output_image[(iy * image_width) + ix] = sum;
}

__kernel void convolution_f3(
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image
) {
    __local float tile[SMEM_W_3][SMEM_H_3];

    int tx = get_local_id(0); // 0~24
    int ty = get_local_id(1); // 0~24
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    int total_pixels_to_load = SMEM_H_3 * SMEM_W_3; // 31 * 31 = 961
    int thread_id = ty * TILE_W + tx;           // 0 ~ 624
    int threads_per_group = TILE_W * TILE_H;    // 625

    for (int i = thread_id; i < total_pixels_to_load; i += threads_per_group) {
        int l_y = i / SMEM_W_3;
        int l_x = i % SMEM_W_3;
        
        // 對應到 Global Memory 座標
        int global_col = (get_group_id(0) * TILE_W) - HALF_FILTER_3 + l_x;
        int global_row = (get_group_id(1) * TILE_H) - HALF_FILTER_3 + l_y;

        if (global_row >= 0 && global_row < image_height && global_col >= 0 && global_col < image_width) {
            tile[l_y][l_x] = input_image[global_row * image_width + global_col];
        } else {
            tile[l_y][l_x] = 0.0f; // Padding 0
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    // center
    int c_r = ty + HALF_FILTER_3;
    int c_c = tx + HALF_FILTER_3;
    sum += tile[c_r - 1][c_c - 1];
    sum += tile[c_r - 1][c_c + 1];

    sum += tile[c_r    ][c_c - 1];
    sum += tile[c_r    ][c_c    ];
    sum += tile[c_r    ][c_c + 1];

    sum += tile[c_r + 1][c_c - 1];
    sum += tile[c_r    ][c_c    ];
    sum += tile[c_r + 1][c_c + 1];
    output_image[(iy * image_width) + ix] = sum;
}