#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h> // 引入 AVX/AVX2 intrinsics

#define default_thread_cnt 8
#define default_number_of_tosses 100000000LL

// AVX 一次處理 4 個 double
#define SIMD_WIDTH 4

typedef struct {
    int64_t number_of_tosses;
    uint64_t seed;
} thread_args_t;


// 向量化的 xorshift* PRNG 狀態
// 每個執行緒需要自己的狀態
typedef struct {
    __m256i state;
} prng_state;

// 初始化 PRNG 狀態
void prng_init(prng_state *s, uint64_t seed) {
    // 簡單地用線性同餘法為 4 個通道產生不同的種子
    uint64_t s0 = seed * 0x2545F4914F6CDD1DULL;
    uint64_t s1 = s0 * 0x2545F4914F6CDD1DULL;
    uint64_t s2 = s1 * 0x2545F4914F6CDD1DULL;
    uint64_t s3 = s2 * 0x2545F4914F6CDD1DULL;
    s->state = _mm256_set_epi64x(s0, s1, s2, s3);
}

// 產生一個包含 4 個隨機 64 位元整數的向量
static inline __m256i rand_vec(prng_state *s) {
    s->state = _mm256_xor_si256(s->state, _mm256_slli_epi64(s->state, 13));
    s->state = _mm256_xor_si256(s->state, _mm256_srli_epi64(s->state, 7));
    s->state = _mm256_xor_si256(s->state, _mm256_slli_epi64(s->state, 17));
    return s->state;
}

void *toss(void *args) {
    int64_t number_of_tosses = ((thread_args_t *)args)->number_of_tosses;
    uint64_t seed = ((thread_args_t *)args)->seed;

    prng_state rng_state;
    prng_init(&rng_state, seed);

    int64_t *local_number_in_circle = malloc(sizeof(int64_t));
    *local_number_in_circle = 0LL;

    // 常數向量
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_two = _mm256_set1_pd(2.0);
    // 2^53，用於將 uint64 轉換為 [0, 1) 的 double
    const __m256d v_scale = _mm256_set1_pd(1.0 / (1ULL << 53));
    const __m256i v_mask_53bit = _mm256_set1_epi64x(0x1FFFFFFFFFFFFFULL);

    int64_t total_tosses_in_vec = (number_of_tosses / SIMD_WIDTH) * SIMD_WIDTH;
    int64_t i;

    __m256d v_sum = _mm256_setzero_pd();

    for (i = 0; i < total_tosses_in_vec; i += SIMD_WIDTH) {
        // 1. 產生 0-1 之間的隨機 double 向量
        __m256i r1_i = _mm256_and_si256(rand_vec(&rng_state), v_mask_53bit);
        __m256i r2_i = _mm256_and_si256(rand_vec(&rng_state), v_mask_53bit);
        __m256d r1_d = _mm256_mul_pd(_mm256_cvtepi64_pd(r1_i), v_scale);
        __m256d r2_d = _mm256_mul_pd(_mm256_cvtepi64_pd(r2_i), v_scale);

        // 2. 轉換到 [-1, 1] 區間
        // x = r1 * 2.0 - 1.0; y = r2 * 2.0 - 1.0
        __m256d x = _mm256_sub_pd(_mm256_mul_pd(r1_d, v_two), v_one);
        __m256d y = _mm256_sub_pd(_mm256_mul_pd(r2_d, v_two), v_one);

        // 3. 計算距離平方 x*x + y*y
        __m256d x2 = _mm256_mul_pd(x, x);
        __m256d y2 = _mm256_mul_pd(y, y);
        __m256d dist_sq = _mm256_add_pd(x2, y2);
        
        // 4. 比較 distance_squared <= 1
        __m256d mask = _mm256_cmp_pd(dist_sq, v_one, _CMP_LE_OQ);

        // 5. 累計結果 (mask 會是 0xFFF... 或 0x000...)
        // 我們可以把它和 1.0 做 AND 運算，得到 1.0 或 0.0，再累加
        __m256d add_val = _mm256_and_pd(mask, v_one);
        v_sum = _mm256_add_pd(v_sum, add_val);
    }
    
    // 將向量 v_sum 中的 4 個 double 加總
    double sum_array[SIMD_WIDTH];
    _mm256_storeu_pd(sum_array, v_sum);
    *local_number_in_circle += (int64_t)sum_array[0] + (int64_t)sum_array[1] + (int64_t)sum_array[2] + (int64_t)sum_array[3];

    // 處理剩餘的不足 4 個的 toss (純量版本)
    for (; i < number_of_tosses; ++i) {
        uint64_t r1_val = _mm_cvtsi128_si64(_mm256_castsi256_si128(rand_vec(&rng_state)));
        uint64_t r2_val = _mm_cvtsi128_si64(_mm256_castsi256_si128(rand_vec(&rng_state)));
        double x_s = (double)(r1_val & *(uint64_t*)(&v_mask_53bit)) * (*((double*)(&v_scale)+i)) * 2.0 - 1.0;
        double y_s = (double)(r2_val & *(uint64_t*)(&v_mask_53bit)) * (*((double*)(&v_scale)+i)) * 2.0 - 1.0;
        if (x_s * x_s + y_s * y_s <= 1.0) {
            (*local_number_in_circle)++;
        }
    }
    
    return (void *)local_number_in_circle;
}

int main(int argc, char *argv[]) {
    // ... main 函數和之前移除 lock 的版本幾乎一樣 ...
    int thread_cnt = default_thread_cnt;
    int64_t number_of_tosses = default_number_of_tosses;

    if (argc > 1) thread_cnt = atoi(argv[1]);
    if (argc > 2) number_of_tosses = atoll(argv[2]);

    srand(time(NULL));

    int64_t number_in_circle = 0LL;
    int64_t toss_per_thread = number_of_tosses / thread_cnt;
    int64_t remaining_tosses = number_of_tosses % thread_cnt;

    pthread_t *threads = malloc(thread_cnt * sizeof(pthread_t));
    thread_args_t *args = malloc(thread_cnt * sizeof(thread_args_t));

    for (int i = 0; i < thread_cnt; i++) {
        int64_t real_toss_per_thread = toss_per_thread + (i < remaining_tosses ? 1 : 0);
        uint64_t seed = ((uint64_t)rand() << 32) | rand() | 1; // 確保種子非 0
        args[i] = (thread_args_t){real_toss_per_thread, seed};
        pthread_create(&threads[i], NULL, toss, &args[i]);
    }

    for (int i = 0; i < thread_cnt; i++) {
        void *ret_val;
        pthread_join(threads[i], &ret_val);
        if (ret_val) {
            number_in_circle += *(int64_t *)ret_val;
            free(ret_val);
        }
    }

    double pi_estimate = 4.0 * number_in_circle / ((double)number_of_tosses);
    printf("%lf\n", pi_estimate);

    free(args);
    free(threads);
    return 0;
}