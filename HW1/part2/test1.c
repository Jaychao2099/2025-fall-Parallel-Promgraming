#include "test.h"

void test1(float *restrict a, float *restrict b, float *restrict c, int N)
// restrict 向編譯器暗示：在指標存續期間，僅會使用該指標本身或直接衍生值（例如指標 + 1）來存取其所指向的物件
{
    __builtin_assume(N == 1024);    // 向編譯器提供了更多關於程式的資訊 —— 例如此程式始終具有相同的輸入大小 —— 使其能夠執行更多優化。
    a = (float *)__builtin_assume_aligned(a, 32);   // 提示該陣列以 32 byte 為對齊單位，對齊 AVX2 的 256-bit (32 byte) YMM 暫存器。
    b = (float *)__builtin_assume_aligned(b, 32);
    c = (float *)__builtin_assume_aligned(c, 32);

    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < N; j++)
        {
            c[j] = a[j] + b[j];
        }
    }
}
