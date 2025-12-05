#include "../util/util.h"
#include <cuda_fp16.h>
#define WARP_NUM 4
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128
// 1x7168 -> 7168x9216
// naive
__global__ void gemv_1x7168_7168x9216(const char *A, const uint32_t *table, const __half *x, float *y, int m, int n) {
    constexpr uint32_t K = 7168;
    constexpr uint32_t K_BLOCKS = K / 64;
    constexpr uint32_t M = 7168;
    constexpr uint32_t half_M = 7168 / 2;
    constexpr uint32_t N = 9216;
    constexpr uint32_t SEQ = 4 * 8 * WARP_SIZE;

    __shared__ half x_smem[SEQ];
    __shared__ uint32_t table_smem[WARP_NUM * K_BLOCKS];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t table_row_start = blockIdx.x * WARP_NUM;
    table += table_row_start * K_BLOCKS;
    for (uint32_t i = threadIdx.x; i < WARP_NUM * K_BLOCKS; i += THREADS_PER_BLOCK) {
        table_smem[i] = table[i];
    }
    __syncthreads();
    float sum = 0;
    for (uint32_t i = 0; i < K / SEQ; i++) {
        for (uint32_t j = 0; j < SEQ / THREADS_PER_BLOCK; j++) {
            uint32_t idx = i * SEQ + j * THREADS_PER_BLOCK + threadIdx.x;
            x_smem[idx - i*SEQ] = x[idx];
        }
        __syncthreads();
        for (uint32_t k = 0; k < SEQ / (8 * WARP_SIZE); k++) {
            uint32_t weight_T_row = blockIdx.x * 4 + warp_id;
            uint32_t weight_T_col = i * SEQ + k * 8 * WARP_SIZE + lane_id * 8;
            uint32_t sss = weight_T_col / 2;
            uint32_t raw = *reinterpret_cast<const uint32_t *>(&A[
                half_M * weight_T_row + sss]);
            uint8_t q0 = raw & 0xF;
            uint8_t q1 = (raw >> 4) & 0xF;
            uint8_t q2 = (raw >> 8) & 0xF;
            uint8_t q3 = (raw >> 12) & 0xF;
            uint8_t q4 = (raw >> 16) & 0xF;
            uint8_t q5 = (raw >> 20) & 0xF;
            uint8_t q6 = (raw >> 24) & 0xF;
            uint8_t q7 = (raw >> 28) & 0xF;
            uint32_t table_col = weight_T_col / 64;
            half table_zp = *reinterpret_cast<half*>(&table_smem[warp_id * K_BLOCKS + table_col]);
            half table_scale = *(reinterpret_cast<half*>(&table_smem[warp_id * K_BLOCKS + table_col])+1);
            float zp    = __half2float(table_zp);
            float scale = __half2float(table_scale);
            float res0= zp + static_cast<float>(q0) * scale;
            float res1= zp + static_cast<float>(q1) * scale;
            float res2= zp + static_cast<float>(q2) * scale;
            float res3= zp + static_cast<float>(q3) * scale;
            float res4= zp + static_cast<float>(q4) * scale;
            float res5= zp + static_cast<float>(q5) * scale;
            float res6= zp + static_cast<float>(q6) * scale;
            float res7= zp + static_cast<float>(q7) * scale;
            sum += res0 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8]);
            sum += res1 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8 + 1]);
            sum += res2 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8 + 2]);
            sum += res3 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8 + 3]);
            sum += res4 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8 + 4]);
            sum += res5 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8 + 5]);
            sum += res6 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8 + 6]);
            sum += res7 * __half2float(x_smem[k * 8 * WARP_SIZE + lane_id * 8 + 7]);
        }
        __syncthreads();
    }
    sum = warp_reduce_sum(sum);
    if (lane_id == 0) {
        y[blockIdx.x * WARP_NUM + warp_id] = sum;
    }
}
