#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h> // AVX/AVX2

#define default_thread_cnt 6
#define default_number_of_tosses 100000000LL

// AVX 一次處理 8 個 float
#define SIMD_WIDTH 8

typedef struct {
    int64_t number_of_tosses;
    uint64_t seed;
} thread_args_t;

// 初始化 PRNG 狀態
void prng_init(__m256i *state, uint64_t seed) {
    uint64_t s0 = seed * 0x2545F4914F6CDD1DULL;
    uint64_t s1 = s0 * 0x2545F4914F6CDD1DULL;
    uint64_t s2 = s1 * 0x2545F4914F6CDD1DULL;
    uint64_t s3 = s2 * 0x2545F4914F6CDD1DULL;
    *state = _mm256_set_epi64x(s3, s2, s1, s0);
}

// 產生一個偽隨機 256-bit 向量
static inline __m256i rand_vec(__m256i *state) {
    __m256i old_state = *state;                   // old_x = x;
    *state = _mm256_xor_si256(*state, _mm256_slli_epi64(*state, 13));     // x ^= x << 13;
    *state = _mm256_xor_si256(*state, _mm256_srli_epi64(*state, 7));      // x ^= x >> 7;
    *state = _mm256_xor_si256(*state, _mm256_slli_epi64(*state, 17));     // x ^= x << 17;
    return _mm256_add_epi64(old_state, *state);   // return old_x + x;
}

void *toss(void *args) {
    int64_t number_of_tosses = ((thread_args_t *)args)->number_of_tosses;
    uint64_t seed = ((thread_args_t *)args)->seed;

    // 向量化的 xorshift+ PRNG 狀態
    __m256i rng_state;
    prng_init(&rng_state, seed);

    int64_t *local_number_in_circle = malloc(sizeof(int64_t));      // 在 heap 分配空間
    *local_number_in_circle = 0LL;

    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_two = _mm256_set1_ps(2.0f);

    // 1.0f 的 exp mask
    const __m256i v_float_one_bits = _mm256_set1_epi32(0x3F800000);
    // Mantissa 的 mask
    const __m256i v_mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
    
    int64_t local_sum = 0;

    for (int64_t i = 0; i < number_of_tosses; i += SIMD_WIDTH) {
        // 產生 8 個 32-bit 隨機數 (x, y) (int 型別)
        __m256i x_i = rand_vec(&rng_state);
        __m256i y_i = rand_vec(&rng_state);
        
        // 先取小數部分
        x_i = _mm256_and_si256(x_i, v_mantissa_mask);
        y_i = _mm256_and_si256(y_i, v_mantissa_mask);

        // 將 23 位元尾數與 1.0 的指數部分結合，並重新解釋為 float 型別
        // x = [1.0, 2.0)
        __m256 x_f_1_2 = _mm256_castsi256_ps(_mm256_or_si256(x_i, v_float_one_bits));
        __m256 y_f_1_2 = _mm256_castsi256_ps(_mm256_or_si256(y_i, v_float_one_bits));
        
        // x - 1.0 = [0, 1.0)
        __m256 x_f_0_1 = _mm256_sub_ps(x_f_1_2, v_one);
        __m256 y_f_0_1 = _mm256_sub_ps(y_f_1_2, v_one);

        // (2*x - 1) = [-1, 1)
        __m256 x_f = _mm256_sub_ps(_mm256_mul_ps(x_f_0_1, v_two), v_one);
        __m256 y_f = _mm256_sub_ps(_mm256_mul_ps(y_f_0_1, v_two), v_one);

        // -----------------------------------------------------------

        // distance_squared = x*x + y*y
        __m256 dist_sq = _mm256_add_ps(_mm256_mul_ps(x_f, x_f), _mm256_mul_ps(y_f, y_f));
        
        // if (distance_squared <= 1)
        __m256 mask = _mm256_cmp_ps(dist_sq, v_one, _CMP_LE_OQ);

        // 256-bit mask 轉換為 8-bit mask (每個 float 一個布林值)
        int mask_bits = _mm256_movemask_ps(mask);

        // 加總 1 的個數
        local_sum += __builtin_popcount(mask_bits);
    }
    
    *local_number_in_circle = local_sum;

    // 剩餘部分忽略

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
        uint64_t seed = ((uint64_t)rand() << 32) | rand();
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