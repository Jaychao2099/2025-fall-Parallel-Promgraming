#include <mpi.h>
#include <cstring>
#include <algorithm>
#include <cstdlib>

#pragma GCC optimize("O3", "unroll-loops")

// 全域變數 矩陣維度, 本地資料
static int g_n, g_m, g_l;
static int g_my_rows;
static int g_my_start_row;

// 記憶體對齊到 cache line (64 bytes)
inline int* aligned_alloc_int(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, count * sizeof(int)) != 0) {
        return new int[count];
    }
    return static_cast<int*>(ptr);
}

// 根據矩陣大小選擇最佳 block size
inline void get_optimal_block_sizes(int n, int m, int l, int &bs_i, int &bs_j, int &bs_k) {
    int max_dim = std::max({n, m, l});
    
    if (max_dim <= 500) {
        // 小矩陣：L1 cache 優化
        bs_i = 32;
        bs_j = 64;   // 稍大，因為 B 是 column-major
        bs_k = 128;  // k 方向可以更大
    } else {
        // 大矩陣：L2/L3 cache 優化
        bs_i = 64;
        bs_j = 64;
        bs_k = 256;
    }
}

// 計算每個 rank 應該處理的 row 範圍
inline void compute_row_distribution(int n, int size, int rank, int &start_row, int &num_rows) {
    int rows_per_proc = n / size;
    int remainder = n % size;
    
    if (rank < remainder) {
        num_rows = rows_per_proc + 1;
        start_row = rank * num_rows;
    } else {
        num_rows = rows_per_proc;
        start_row = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
    }
}

// 分配記憶體並分發資料給各個 processes
void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 廣播矩陣維度給所有 processes
    if (rank == 0) {
        g_n = n;
        g_m = m;
        g_l = l;
    }
    MPI_Bcast(&g_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 計算本地應該處理的 row
    compute_row_distribution(g_n, size, rank, g_my_start_row, g_my_rows);
    
    // 分配本地 A 矩陣記憶體
    // 使用對齊的記憶體分配
    *a_mat_ptr = aligned_alloc_int(g_my_rows * g_m);
    
    // Scatterv 參數
    int *sendcounts = nullptr;
    int *displs = nullptr;
    
    if (rank == 0) {
        sendcounts = new int[size];
        displs = new int[size];
        
        for (int i = 0; i < size; ++i) {
            int start_row, num_rows;
            compute_row_distribution(g_n, size, i, start_row, num_rows);
            sendcounts[i] = num_rows * g_m;
            displs[i] = start_row * g_m;
        }
    }
    
    // 分發 A 矩陣的 row
    MPI_Scatterv(a_mat, sendcounts, displs, MPI_INT,
                 *a_mat_ptr, g_my_rows * g_m, MPI_INT,
                 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        delete[] sendcounts;
        delete[] displs;
    }
    
    // 廣播 B 矩陣給所有 processes
    // B 矩陣也使用對齊記憶體
    *b_mat_ptr = aligned_alloc_int(g_m * g_l);
    if (rank == 0) {
        memcpy(*b_mat_ptr, b_mat, g_m * g_l * sizeof(int));
    }
    MPI_Bcast(*b_mat_ptr, g_m * g_l, MPI_INT, 0, MPI_COMM_WORLD);
}

// 內部計算函數
inline void compute_block_kernel(
    const int * __restrict__ a_block,
    const int * __restrict__ b_block,
    int * __restrict__ c_block,
    int rows, int cols, int k_size,
    int a_stride, int b_stride, int c_stride)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int sum = 0;
            // #pragma GCC ivdep  // 沒有迴圈依賴
            for (int k = 0; k < k_size; ++k) {
                sum += a_block[i * a_stride + k] * b_block[j * b_stride + k];
            }
            c_block[i * c_stride + j] += sum;
        }
    }
}

// 每個 process 獨立計算其負責的 row，然後收集結果
void matrix_multiply(
    const int n, const int m, const int l, 
    const int * __restrict__ a_mat, 
    const int * __restrict__ b_mat, 
    int * __restrict__ out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 根據矩陣大小選擇最佳 block size
    int bs_i, bs_j, bs_k;
    get_optimal_block_sizes(g_n, g_m, g_l, bs_i, bs_j, bs_k);
    
    // 分配本地結果矩陣
    // 使用對齊的記憶體存放結果
    int *local_result = aligned_alloc_int(g_my_rows * g_l);
    memset(local_result, 0, g_my_rows * g_l * sizeof(int));
    
    // 三層 tiling
    for (int ii = 0; ii < g_my_rows; ii += bs_i) {
        int i_block_size = std::min(bs_i, g_my_rows - ii);
        
        for (int jj = 0; jj < g_l; jj += bs_j) {
            int j_block_size = std::min(bs_j, g_l - jj);
            
            for (int kk = 0; kk < g_m; kk += bs_k) {
                int k_block_size = std::min(bs_k, g_m - kk);
                
                // 在 block 內進行計算
                compute_block_kernel(
                    &a_mat[ii * g_m + kk],        // A block
                    &b_mat[jj * g_m + kk],        // B block
                    &local_result[ii * g_l + jj], // C block
                    i_block_size,
                    j_block_size,
                    k_block_size,
                    g_m,  // A stride
                    g_m,  // B stride (column-major)
                    g_l   // C stride
                );
            }
        }
    }
    
    // Gatherv 參數
    int *recvcounts = nullptr;
    int *displs = nullptr;
    
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        
        for (int i = 0; i < size; ++i) {
            int start_row, num_rows;
            compute_row_distribution(g_n, size, i, start_row, num_rows);
            recvcounts[i] = num_rows * g_l;
            displs[i] = start_row * g_l;
        }
    }
    
    // 收集結果到 rank 0
    MPI_Gatherv(local_result, g_my_rows * g_l, MPI_INT,
                out_mat, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    
    free(local_result);  // posix_memalign ---> free
    
    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    free(a_mat);  // posix_memalign ---> free
    free(b_mat);
}
