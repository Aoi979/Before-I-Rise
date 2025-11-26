#include <__clang_cuda_runtime_wrapper.h>
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

  uint32_t lm_offset =
      (blockIdx.x * gridDim.y * target_seq_len) + (blockIdx.y * target_seq_len);

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
          S[(Bc * threadIdx.x) + y] = __expf(S[(Bc * threadIdx.x) + y] - row_m);
          row_l += S[(Bc * threadIdx.x) + y];
        }

        float row_m_new = max(row_m_prev, row_m);
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                          (__expf(row_m - row_m_new) * row_l);

        for (int x = 0; x < d; x++) {
          float pv = 0;
          for (int y = 0; y < Bc; y++) {
            pv += S[(Bc * threadIdx.x) + y] * Vj[(y * d) + x];
          }
          O[q_offset + (tile_size * i) + (threadIdx.x * d) + x] =
              (1 / row_l_new) *
              ((row_l_prev * __expf(row_m_prev - row_m_new) *
                O[q_offset + (tile_size * i) + (threadIdx.x * d) + x]) +
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
template <int const QKV_HEADS, int const HEAD_DIM, int const MMA_M,
          int const MMA_N, int const MMA_K, int const STAGE, int const Bc = 64,
          int const WARP_NUM_SEQLEN_QS = 2, int const WARP_NUM_SEQLEN_K = 4>
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

  constexpr uint32_t THREAD_SIZE =
      WARP_NUM_SEQLEN_K * WARP_NUM_SEQLEN_QS * WARP_SIZE;

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

  // Q, K, V, O [seqlen, head_dim]
  uint32_t Q_gmem_offset = QKV_batch_id * BATCH_SIZE +
                           QKV_head_id * QKV_HEAD_SIZE +
                           Q_tile_id * Br * HEAD_DIM;

  uint32_t K_gmem_offset =
      QKV_batch_id * BATCH_SIZE + QKV_head_id * QKV_HEAD_SIZE;

  uint32_t V_gmem_offset = K_gmem_offset;

  uint32_t O_gmem_offset = Q_gmem_offset;

  constexpr uint32_t smem_Q_row_size = (THREAD_SIZE / Br);
  uint32_t load_smem_Q_Br = tid / smem_Q_row_size;
  uint32_t load_smem_Q_d = tid % smem_Q_row_size * (HEAD_DIM / smem_Q_row_size);

  constexpr uint32_t smem_KV_row_size = (THREAD_SIZE / Bc);
  uint32_t load_smem_KV_Bc = tid / smem_KV_row_size;
  uint32_t load_smem_KV_d = tid % smem_KV_row_size * (HEAD_DIM / smem_KV_row_size);

  extern __shared__ half sram[];

  constexpr uint32_t Q_tile_size = Br * HEAD_DIM;
  constexpr uint32_t KV_tile_size = Bc * HEAD_DIM;
  constexpr uint32_t S_tile_size = Br * Bc;

  half *Q_tile_smem = sram;
  half *K_tile_smem = &sram[Q_tile_size];
  half *V_tile_smem = &sram[Q_tile_size + STAGE * KV_tile_size];
  half *S_tile_smem = V_tile_smem + KV_tile_size;

  uint32_t Q_tile_smem_address = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t K_tile_smem_address = __cvta_generic_to_shared(K_tile_smem);
  uint32_t V_tile_smem_address = __cvta_generic_to_shared(V_tile_smem);
  uint32_t S_tile_smem_address = __cvta_generic_to_shared(S_tile_smem);

  float lane_Bc_max_old[WARP_ITER_SEQLEN_QS][2] = -INFINITY;
  float lane_Bc_sum_old[WARP_ITER_SEQLEN_QS][2] = 0.0f;

  __shared__ float Bc_max_new_smem[Br][WARP_NUM_SEQLEN_K];
  __shared__ float Bc_sum_new_smem[Br][WARP_NUM_SEQLEN_K];

  using REG_SIZE_T = uint32_t;

  // 16 * 16 * 2 / (32 * 4) = 4 
  REG_SIZE_T R_Q[WARP_ITER_SEQLEN_QS][4];
  // 16 * 8
  REG_SIZE_T R_K[WARP_ITER_SEQLEN_K][2];
  // 16 * 8
  REG_SIZE_T R_V[WARP_ITER_HEAD_DIM_V][2];
  // 16 * 8
  REG_SIZE_T R_S[WARP_ITER_SEQLEN_QS][WARP_ITER_SEQLEN_K][2] = 0.0f;
  // 16 * 8
  REG_SIZE_T R_O[WARP_ITER_SEQLEN_QS][WARP_ITER_HEAD_DIM_V][2] = 0.0f;
  // 16 * 8
  REG_SIZE_T R_Final[WARP_ITER_SEQLEN_QS][WARP_ITER_HEAD_DIM_V][2] = 0.0f;

  // load Q, gmem to smem, only once
  {
     

  }


}
