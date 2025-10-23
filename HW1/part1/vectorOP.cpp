#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_float zero = _pp_vset_float(0.f);
    __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i = 0; i < N - (N % VECTOR_WIDTH); i += VECTOR_WIDTH)
    {

        // All ones
        maskAll = _pp_init_ones();  // {1,1,1,1}

        // All zeros
        maskIsNegative = _pp_init_ones(0);  // {0,0,0,0}

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll); // x = values[i];  // 一次 VECTOR_WIDTH 個

        // Set mask according to predicate
        _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x[i] < 0)

        // Execute instruction using mask ("if" clause)
        _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

        // Inverse maskIsNegative to generate "else" mask
        maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

        // Execute instruction ("else" clause)
        _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll);
    }

    // handle remaining elements
    for (int i = N - N % VECTOR_WIDTH; i < N; i++) {
        float x = values[i];
        output[i] = x < 0 ? -x : x;
    }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //
    __pp_vec_float x;
    __pp_vec_int exp;
    __pp_vec_float result;
    __pp_vec_float max_value = _pp_vset_float(9.999999f);
    
    __pp_vec_int zero = _pp_vset_int(0);
    __pp_vec_int ones = _pp_vset_int(1);
    
    __pp_mask maskAll = _pp_init_ones();  // {1,1,1,1}

    __pp_mask maskIsPositive;
    __pp_mask maskTooLarge, maskNotTooLarge;

    __pp_mask maskCanExp;

    for (int i = 0; i < N - (N % VECTOR_WIDTH); i += VECTOR_WIDTH)
    {
        result = _pp_vset_float(1.0f);
        maskCanExp = _pp_init_ones();

        _pp_vload_float(x, values + i, maskCanExp);
        _pp_vload_int(exp, exponents + i, maskCanExp);
        
        // 篩選 x[i] > 9.999999f 的資料
        _pp_vgt_float(maskTooLarge, result, max_value, maskCanExp);
        maskNotTooLarge = _pp_mask_not(maskTooLarge);

        // 篩選 exp[i] > 0 的資料
        _pp_vgt_int(maskIsPositive, exp, zero, maskCanExp);

        // 綜合篩選
        maskCanExp = _pp_mask_and(maskIsPositive, maskNotTooLarge);

        // while (x[i] <= 9.999999f && exp[i] > 0) {
        while (_pp_cntbits(maskCanExp) > 0) {

            // result[i] *= x[i];
            _pp_vmult_float(result, result, x, maskCanExp);

            // if (result[i] > 9.999999f) { result[i] = 9.999999f; }
            _pp_vgt_float(maskTooLarge, result, max_value, maskCanExp);
            _pp_vset_float(result, 9.999999f, maskTooLarge);
            
            // exp[i]--;
            _pp_vsub_int(exp, exp, ones, maskCanExp);

            _pp_vgt_int(maskIsPositive, exp, zero, maskCanExp);
            maskNotTooLarge = _pp_mask_not(maskTooLarge);
            maskCanExp = _pp_mask_and(maskIsPositive, maskNotTooLarge);
        }
        _pp_vstore_float(output + i, result, maskAll);
    }

    // handle remaining elements
    for (int i = N - N % VECTOR_WIDTH; i < N; i++) {
        float x = values[i];
        int exp = exponents[i];

        float result = 1.0f;
        while (exp > 0) {
            result *= x;
            exp--;
        }
        if (result > 9.999999f) result = 9.999999f;
        output[i] = result;
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //

    __pp_vec_float x;
    __pp_vec_float result = _pp_vset_float(0.0f);
    __pp_mask maskAll = _pp_init_ones();

    // O(N / VECTOR_WIDTH)
    for (int i = 0; i < N - (N % VECTOR_WIDTH); i += VECTOR_WIDTH) {
        _pp_vload_float(x, values + i, maskAll);
        _pp_vadd_float(result, result, x, maskAll);     // result[i] += x[i];
    }
    
    // O(log2(VECTOR_WIDTH))
    for (int i = 0; i < log2f(VECTOR_WIDTH); i++) {
        _pp_hadd_float(result, result);
        _pp_interleave_float(result, result);
    }
    
    return result.value[0];
}