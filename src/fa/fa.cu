#include <cstdint>
#include <cuda_fp16.h>
#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#endif
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>

__global__ void flash_attention_v1() {}

//     y1 y2 y3
//  x1
//  x2
//  x3
__global__ void v1_fwd_kernel_naive(float const *Q, float const *K,
                                    float const *V, int const target_seq_len,
                                    int const src_seq_len, int const d,
                                    int const Tc, int const Tr, int const Bc,
                                    int const Br, float const softmax_scale,
                                    float *l, float *m, float *O) {
    uint32_t kv_offset = (blockIdx.x * gridDim.y * src_seq_len * d) +
                         (blockIdx.y * src_seq_len * d);
    uint32_t q_offset = (blockIdx.x * gridDim.y * target_seq_len * d) +
                        (blockIdx.y * target_seq_len * d);

    uint32_t lm_offset = (blockIdx.x * gridDim.y * target_seq_len) +
                         (blockIdx.y * target_seq_len);

    extern __shared__ float smem[];

    int32_t tile_size = Bc * d;
    float *Qi = smem;
    float *Kj = &smem[tile_size];
    float *Vj = &smem[tile_size * 2];
    float *S = &smem[tile_size * 3];

    for (uint32_t j = 0; j < Tc; j++) {
        for (uint32_t x = 0; x < d; x++) {
            Kj[threadIdx.x * d + x] =
                K[kv_offset + (tile_size * j) + (threadIdx.x * d) + x];
            Vj[threadIdx.x * d + x] =
                V[kv_offset + (tile_size * j) + (threadIdx.x * d) + x];
        }
        __syncthreads();
        for (uint32_t i = 0; i < Tr; i++) {
            if (threadIdx.x < Br) {
                for (uint32_t x = 0; x < d; x++) {
                    Qi[threadIdx.x * d + x] =
                        Q[q_offset + (tile_size * i) + (threadIdx.x * d) + x];
                }
                float row_m_prev = m[lm_offset + (Br * i) + threadIdx.x];
                float row_l_prev = l[lm_offset + (Br * i) + threadIdx.x];

                float row_m = -INFINITY;

                for (int y = 0; y < Bc; y++) {
                    float sum = 0;
                    for (uint32_t x = 0; x < d; x++) {
                        sum += Qi[(threadIdx.x * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(Bc * threadIdx.x) + y] = sum;

                    if (sum > row_m) {
                        row_m = sum;
                    }
                }

                float row_l = 0;
                for (int y = 0; y < Bc; y++) {
                    S[(Bc * threadIdx.x) + y] =
                        __expf(S[(Bc * threadIdx.x) + y] - row_m);
                    row_l += S[(Bc * threadIdx.x) + y];
                }

                float row_m_new = max(row_m_prev, row_m);
                float row_l_new =
                    (__expf(row_m_prev - row_m_new) * row_l_prev) +
                    (__expf(row_m - row_m_new) * row_l);

                for (int x = 0; x < d; x++) {
                    float pv = 0;
                    for (int y = 0; y < Bc; y++) {
                        pv += S[(Bc * threadIdx.x) + y] * Vj[(y * d) + x];
                    }
                    O[q_offset + (tile_size * i) + (threadIdx.x * d) + x] =
                        (1 / row_l_new) *
                        ((row_l_prev * __expf(row_m_prev - row_m_new) *
                          O[q_offset + (tile_size * i) + (threadIdx.x * d) +
                            x]) +
                         (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + (Br * i) + threadIdx.x] = row_m_new;
                l[lm_offset + (Br * i) + threadIdx.x] = row_l_new;
            }
        }
        __syncthreads();
    }
}

// target_seq_len == src_seq_len
// model_dim == v_head_dim
// Br == 64
// split_kv
// 可参考 docs/draw/fa.excalidraw
template <int const QKV_HEADS, int const HEAD_DIM, int const MMA_M, int const MMA_N, int const MMA_K,
          int const STAGE, int const Bc = 64, int const WARP_NUM_SEQLEN_QS = 2,
          int const WARP_NUM_SEQLEN_K = 4>
__global__ void v2_fwd_kernel(half *Q, half *K, half *V, half *O,
                              int QKV_seqlen) {
    // static assertions and constexpr calculations
    static_assert(MMA_M == 16 && MMA_N == 8 && MMA_K == 16);
    constexpr uint32_t Br = 64;
    static_assert(Br % (MMA_M * WARP_NUM_SEQLEN_QS) == 0,
                  "Br must be divisible by MMA_M * WARP_NUM_SEQLEN_QS");
    constexpr uint32_t WARP_ITER_SEQLEN_QS = Br / (MMA_M * WARP_NUM_SEQLEN_QS);

    static_assert(Bc % (MMA_N * WARP_NUM_SEQLEN_K) == 0,
                  "Bc must be divisible by MMA_N * WARP_NUM_SEQLEN_K");
    constexpr uint32_t WARP_ITER_SEQLEN_K = Bc / (MMA_N * WARP_NUM_SEQLEN_K);

    static_assert(HEAD_DIM % (MMA_N * WARP_NUM_SEQLEN_K) == 0,
                  "HEAD_DIM must be divisible by MMA_N * WARP_NUM_SEQLEN_K");
    constexpr uint32_t WARP_ITER_HEAD_DIM_V =
        HEAD_DIM / (MMA_N * WARP_NUM_SEQLEN_K);

    static_assert(HEAD_DIM % MMA_K == 0, "HEAD_DIM must be divisible by MMA_K");
    constexpr uint32_t hidden_K_ITER = HEAD_DIM / MMA_K;

    static_assert(Bc % MMA_K == 0, "Bc must be divisible by MMA_K");
    constexpr uint32_t hidden_Bc_ITER = Bc / MMA_K;

    uint32_t Tc = QKV_seqlen / Bc;

    constexpr uint32_t WARP_SIZE = 32;

    constexpr uint32_t THREAD_SIZE = WARP_NUM_SEQLEN_K * WARP_NUM_SEQLEN_QS * WARP_SIZE; 

    constexpr float scale = 1.0f / sqrt(HEAD_DIM);

    uint32_t QKV_batch_id = blockIdx.z;
    uint32_t QKV_head_id = blockIdx.y;
    uint32_t Q_tile_id = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;

    uint32_t warp_seqlen_qs_id = warp_id % WARP_NUM_SEQLEN_QS;
    uint32_t warp_seqlen_k_id = warp_id / WARP_NUM_SEQLEN_QS;

    constexpr uint32_t QKV_HEAD_SIZE = QKV_seqlen * HEAD_DIM;
    constexpr uint32_t BATCH_SIZE = QKV_HEADS * QKV_HEAD_SIZE;
    


}
