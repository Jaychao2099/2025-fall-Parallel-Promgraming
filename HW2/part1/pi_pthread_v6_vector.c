#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h> // 引入 AVX/AVX2 intrinsics

#define default_thread_cnt 6
#define default_number_of_tosses 100000000LL

// AVX 一次處理 4 個 double
#define SIMD_WIDTH 4

typedef struct {
    int64_t number_of_tosses;
    uint64_t seed;
} thread_args_t;

// 向量化的 xorshift+ PRNG 狀態
typedef struct {
    __m256i state;
} prng_state;

// 初始化 PRNG 狀態
void prng_init(prng_state *s, uint64_t seed) {
    uint64_t s0 = seed * 0x2545F4914F6CDD1DULL;
    uint64_t s1 = s0 * 0x2545F4914F6CDD1DULL;
    uint64_t s2 = s1 * 0x2545F4914F6CDD1DULL;
    uint64_t s3 = s2 * 0x2545F4914F6CDD1DULL;
    s->state = _mm256_set_epi64x(s3, s2, s1, s0);
}

// 產生一個包含 4 個隨機 64 位元整數的向量 (xorshift+)
static inline __m256i rand_vec(prng_state *s) {
    __m256i old_state = s->state;
    s->state = _mm256_xor_si256(s->state, _mm256_slli_epi64(s->state, 13));
    s->state = _mm256_xor_si256(s->state, _mm256_srli_epi64(s->state, 7));
    s->state = _mm256_xor_si256(s->state, _mm256_slli_epi64(s->state, 17));
    return _mm256_add_epi64(old_state, s->state);
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
    // 用於建構 [1.0, 2.0) double 的位元遮罩 (0x3FF0...0)
    const __m256i v_double_one_bits = _mm256_set1_epi64x(0x3FF0000000000000);
    // 用於取出尾數的 52 位元
    const __m256i v_mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF);

    int64_t total_tosses_in_vec = (number_of_tosses / SIMD_WIDTH) * SIMD_WIDTH;
    int64_t i;
    int64_t local_sum = 0;

    for (i = 0; i < total_tosses_in_vec; i += SIMD_WIDTH) {
        // --- AVX2-compatible random double generation in [-1, 1] ---
        __m256i r1_i = rand_vec(&rng_state);
        __m256i r2_i = rand_vec(&rng_state);
        
        // 取出隨機數的高 52 位元
        r1_i = _mm256_and_si256(r1_i, v_mantissa_mask);
        r2_i = _mm256_and_si256(r2_i, v_mantissa_mask);

        // 將 52 位元尾數與 1.0 的指數部分結合，得到 [1.0, 2.0) 的 double
        __m256d r1_d_1_2 = _mm256_castsi256_pd(_mm256_or_si256(r1_i, v_double_one_bits));
        __m256d r2_d_1_2 = _mm256_castsi256_pd(_mm256_or_si256(r2_i, v_double_one_bits));
        
        // 減 1.0 得到 [0, 1.0)
        __m256d r1_d_0_1 = _mm256_sub_pd(r1_d_1_2, v_one);
        __m256d r2_d_0_1 = _mm256_sub_pd(r2_d_1_2, v_one);

        // 轉換到 [-1, 1] 區間
        __m256d x = _mm256_sub_pd(_mm256_mul_pd(r1_d_0_1, v_two), v_one);
        __m256d y = _mm256_sub_pd(_mm256_mul_pd(r2_d_0_1, v_two), v_one);

        // 計算距離平方 x*x + y*y
        __m256d dist_sq = _mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y));
        
        // 比較 distance_squared <= 1
        __m256d mask = _mm256_cmp_pd(dist_sq, v_one, _CMP_LE_OQ);

        // 從 mask 取得一個 4-bit 的整數 (e.g., 1101b -> 13)
        int mask_bits = _mm256_movemask_pd(mask);
        
        // 使用內建函式計算 bit 數，這比累加向量更有效率
        local_sum += _mm_popcnt_u32(mask_bits);
    }
    *local_number_in_circle = local_sum;
    
    // 處理剩餘的不足 4 個的 toss (純量版本)
    for (; i < number_of_tosses; ++i) {
        uint64_t r1 = ((uint64_t*)&rng_state.state)[0];
        uint64_t r2 = ((uint64_t*)&rng_state.state)[1];
        
        // 簡化版的純量隨機數生成
        double x_s = (double)(r1 & 0x000FFFFFFFFFFFFF) / (double)0x0010000000000000 * 2.0 - 1.0;
        double y_s = (double)(r2 & 0x000FFFFFFFFFFFFF) / (double)0x0010000000000000 * 2.0 - 1.0;

        if (x_s * x_s + y_s * y_s <= 1.0) {
            (*local_number_in_circle)++;
        }
        // 更新狀態避免重複
        prng_init(&rng_state, r1 ^ r2);
    }
    
    return (void *)local_number_in_circle;
}

int main(int argc, char *argv[]) {
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
        uint64_t seed = ((uint64_t)rand() << 32) | rand() | 1;
        args[i] = (thread_args_t){real_toss_per_thread, seed};
        pthread_create(&threads[i], NULL, toss, &args[i]);
    }

    for (int i = 0; i < thread_cnt; i++) {
        void *ret_val;
        pthread_join(threads[i], &ret_val);
        number_in_circle += *(int64_t *)ret_val;
        free(ret_val);
    }

    double pi_estimate = 4.0 * number_in_circle / ((double)number_of_tosses);
    printf("%lf\n", pi_estimate);

    free(args);
    free(threads);
    return 0;
}