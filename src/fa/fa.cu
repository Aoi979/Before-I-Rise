#include <vector>
#ifdef __clang__
#include <__clang_cuda_math.h>
#endif
#ifdef __clang__
#include <__clang_cuda_runtime_wrapper.h>
#endif
#include <cstdint>
#include <cuda_fp16.h>
#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#endif
#include "../util/util.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

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
// STAGE == 1 or 2
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

  float scale = 1.0f / sqrt(float(HEAD_DIM));

  uint32_t QKV_batch_id = blockIdx.z;
  uint32_t QKV_head_id = blockIdx.y;
  uint32_t Q_tile_id = blockIdx.x;
  uint32_t tid = threadIdx.x;
  uint32_t warp_id = tid / WARP_SIZE;
  uint32_t lane_id = tid % WARP_SIZE;

  uint32_t warp_seqlen_qs_id = warp_id % WARP_NUM_SEQLEN_QS;
  uint32_t warp_seqlen_k_id = warp_id / WARP_NUM_SEQLEN_QS;

  uint32_t QKV_HEAD_SIZE = QKV_seqlen * HEAD_DIM;
  uint32_t BATCH_SIZE = QKV_HEADS * QKV_HEAD_SIZE;

  // Q, K, V, O [seqlen, head_dim]
  uint32_t Q_gmem_offset = QKV_batch_id * BATCH_SIZE +
                           QKV_head_id * QKV_HEAD_SIZE +
                           Q_tile_id * Br * HEAD_DIM;

  uint32_t K_gmem_offset =
      QKV_batch_id * BATCH_SIZE + QKV_head_id * QKV_HEAD_SIZE;

  uint32_t V_gmem_offset = K_gmem_offset;

  uint32_t O_gmem_offset = Q_gmem_offset;

  // LDG128 == 8 half
  constexpr uint32_t VECSIZE = 8;
  constexpr uint32_t smem_Q_row_thread_num = HEAD_DIM / VECSIZE;

  uint32_t load_smem_Q_Br = tid / smem_Q_row_thread_num;
  uint32_t load_smem_Q_d = (tid % smem_Q_row_thread_num) * VECSIZE;

  constexpr uint32_t load_smem_Q_stride = (THREAD_SIZE * VECSIZE) / HEAD_DIM;

  constexpr uint32_t smem_KV_row_thread_num = HEAD_DIM / VECSIZE;

  uint32_t load_smem_KV_Bc = tid / smem_KV_row_thread_num;
  uint32_t load_smem_KV_d = (tid % smem_KV_row_thread_num) * VECSIZE;

  constexpr uint32_t load_smem_KV_stride = (THREAD_SIZE * VECSIZE) / HEAD_DIM;

  extern __shared__ half sram[];

  constexpr uint32_t Q_tile_size = Br * HEAD_DIM;
  constexpr uint32_t KV_tile_size = Bc * HEAD_DIM;
  constexpr uint32_t S_tile_size = Br * Bc;

  auto Q_tile_smem = sram;
  auto K_tile_smem = Q_tile_smem + Q_tile_size;
  auto V_tile_smem = K_tile_smem + STAGE * KV_tile_size;

  auto S_tile_smem = V_tile_smem + KV_tile_size;

  uint32_t Q_tile_smem_address = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t K_tile_smem_address = __cvta_generic_to_shared(K_tile_smem);
  uint32_t V_tile_smem_address = __cvta_generic_to_shared(V_tile_smem);
  uint32_t S_tile_smem_address = __cvta_generic_to_shared(S_tile_smem);

  float lane_Bc_max_old[WARP_ITER_SEQLEN_QS][2];
  fill_2D_regs<float, WARP_ITER_SEQLEN_QS, 2>(lane_Bc_max_old, -INFINITY);
  float lane_Bc_sum_old[WARP_ITER_SEQLEN_QS][2] = {};
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
  REG_SIZE_T R_S[WARP_ITER_SEQLEN_QS][WARP_ITER_SEQLEN_K][2] = {};
  // 16 * 8
  REG_SIZE_T R_O[WARP_ITER_SEQLEN_QS][WARP_ITER_HEAD_DIM_V][2] = {};
  // 16 * 8
  REG_SIZE_T R_Final[WARP_ITER_SEQLEN_QS][WARP_ITER_HEAD_DIM_V][2] = {};

  // load Q, gmem to smem, only once
  {
    for (int i = 0; i < Br / load_smem_Q_stride; i++) {
      uint32_t load_gmem_Q =
          Q_gmem_offset + (load_smem_Q_Br + i * load_smem_Q_stride) * HEAD_DIM +
          load_smem_Q_d;
      uint32_t load_smem_Q =
          Q_tile_smem_address +
          ((load_smem_Q_Br + i) * HEAD_DIM + load_smem_Q_d) * sizeof(half);
      CP_ASYNC_CG(load_smem_Q, &Q[load_gmem_Q], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  if constexpr (STAGE > 1) {
    for (int stage = 0; stage < STAGE - 1; stage++) {
      for (int i = 0; i < Bc / load_smem_KV_stride; i++) {
        uint32_t K_Bc_gmem_offset = stage * Bc;
        uint32_t load_gmem_K = K_gmem_offset + K_Bc_gmem_offset * HEAD_DIM +
                               load_smem_KV_Bc * HEAD_DIM +
                               i * load_smem_KV_stride * HEAD_DIM +
                               load_smem_KV_d;
        uint32_t load_smem_K =
            K_tile_smem_address +
            (stage * KV_tile_size + load_smem_KV_Bc * HEAD_DIM +
             i * load_smem_KV_stride * HEAD_DIM + load_smem_KV_d) *
                sizeof(half);
        CP_ASYNC_CG(load_smem_K, &K[load_gmem_K], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }
    // (STAGE - 1) - 1 = STAGE - 2
    // wait Q and 1 K stage
    CP_ASYNC_WAIT_GROUP(STAGE - 2);
    __syncthreads();
  }

  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; tile_K_seqlen++) {
    uint32_t K_smem_select = (tile_K_seqlen % STAGE);
    uint32_t K_smem_select_next = (tile_K_seqlen + (STAGE - 1)) % STAGE;

    if constexpr (STAGE > 1) {
      // load V asynchronously
      {
        for (int i = 0; i < Bc / load_smem_KV_stride; i++) {
          uint32_t V_Bc_gmem_offset = tile_K_seqlen * Bc;
          uint32_t load_gmem_V = V_gmem_offset + V_Bc_gmem_offset * HEAD_DIM +
                                 load_smem_KV_Bc * HEAD_DIM +
                                 i * load_smem_KV_stride * HEAD_DIM +
                                 load_smem_KV_d;
          uint32_t load_smem_V =
              V_tile_smem_address +
              (load_smem_KV_Bc * HEAD_DIM + i * load_smem_KV_stride * HEAD_DIM +
               load_smem_KV_d) *
                  sizeof(half);
          CP_ASYNC_CG(load_smem_V, &V[load_gmem_V], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }

      // NOTE: load next stage (only STGAE == 2) asynchronously
      if ((tile_K_seqlen + 1) < Tc) {
        for (int i = 0; i < Bc / load_smem_KV_stride; i++) {
          uint32_t K_Bc_gmem_offset = (tile_K_seqlen + 1) * Bc;
          uint32_t load_gmem_K = K_gmem_offset + K_Bc_gmem_offset * HEAD_DIM +
                                 load_smem_KV_Bc * HEAD_DIM +
                                 i * load_smem_KV_stride * HEAD_DIM +
                                 load_smem_KV_d;
          uint32_t load_smem_K =
              K_tile_smem_address +
              (K_smem_select_next * KV_tile_size + load_smem_KV_Bc * HEAD_DIM +
               i * load_smem_KV_stride * HEAD_DIM + load_smem_KV_d) *
                  sizeof(half);
          CP_ASYNC_CG(load_smem_K, &K[load_gmem_K], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    } else {
      // stage == 1
      // ..........
      // ..........
    }

    fill_3D_regs<uint32_t, WARP_ITER_SEQLEN_QS, WARP_ITER_SEQLEN_K, 2>(R_S, 0);
    for (int bk_d = 0; bk_d < hidden_K_ITER; bk_d++) {
      // Q | smem2register
      for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
        uint32_t Q_smem_warp_offset =
            (warp_seqlen_qs_id * MMA_M * WARP_NUM_SEQLEN_QS) + i * MMA_M;
        uint32_t lane_Q_smem_Br = Q_smem_warp_offset + lane_id % MMA_M;
        uint32_t lane_Q_smem_d = bk_d * MMA_K + (lane_id / MMA_M) * 8;
        uint32_t lane_Q_smem_address =
            Q_tile_smem_address +
            ((lane_Q_smem_Br * HEAD_DIM) + lane_Q_smem_d) * sizeof(half);
        LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                    lane_Q_smem_address);
      }
      // K | smem2register
      for (int i = 0; i < WARP_ITER_SEQLEN_K; i++) {
        uint32_t K_smem_warp_offset =
            warp_seqlen_qs_id * MMA_N * WARP_ITER_SEQLEN_K + i * MMA_N;
        uint32_t lane_K_smem_Bc = K_smem_warp_offset + lane_id % MMA_N;
        uint32_t lane_K_smem_d = bk_d * MMA_K + (lane_id / MMA_N) * 8;
        uint32_t lane_K_smem_address =
            K_tile_smem_address + (K_smem_select * KV_tile_size +
                                   lane_K_smem_Bc * HEAD_DIM + lane_K_smem_d) *
                                      sizeof(half);
        LDMATRIX_X2(R_K[i][0], R_K[i][1], lane_K_smem_address);
      }
      for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
        for (int j = 0; j < WARP_ITER_SEQLEN_K; j++) {
          HMMA16816(R_S[i][j][0], R_S[i][j][1], R_Q[i][0], R_Q[i][1], R_Q[i][2],
                    R_Q[i][3], R_K[j][0], R_K[j][1], R_S[i][j][0],
                    R_S[i][j][1]);
        }
      }
    }
    __syncthreads();
    float lane_row_max_new[WARP_ITER_SEQLEN_QS][2];
    fill_2D_regs<float, WARP_ITER_SEQLEN_QS, 2>(lane_row_max_new, -INFINITY);
    float lane_row_sum_new[WARP_ITER_SEQLEN_QS][2] = {};
    // warp level reduce max and store to smem
    for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
      for (int j = 0; j < WARP_ITER_SEQLEN_K; j++) {
        float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
        float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));
        float tmp_max_0 = max(t_reg_S_0.x, t_reg_S_0.y);
        float tmp_max_1 = max(t_reg_S_1.x, t_reg_S_1.y);
        lane_row_max_new[i][0] = max(lane_row_max_new[i][0], tmp_max_0);
        lane_row_max_new[i][1] = max(lane_row_max_new[i][1], tmp_max_1);
      }
      lane_row_max_new[i][0] =
          warp_reduce_max<float, 4>(lane_row_max_new[i][0]);
      lane_row_max_new[i][1] =
          warp_reduce_max<float, 4>(lane_row_max_new[i][1]);
      if (lane_id % 4 == 0) {
        Bc_max_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                        i * MMA_M + 0 * 8 + (lane_id / 4)][warp_seqlen_k_id] =
            lane_row_max_new[i][0];
        Bc_max_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                        i * MMA_M + 1 * 8 + (lane_id / 4)][warp_seqlen_k_id] =
            lane_row_max_new[i][1];
      }
    }
    __syncthreads();

    // block level reduce max and load from smem
    // 64(Br) * 4(WARP_NUM_SEQLEN_K) = 256
    float warp_row_max =
        Bc_max_new_smem[tid / WARP_NUM_SEQLEN_K][tid % WARP_NUM_SEQLEN_K];

    float block_row_max = warp_reduce_max<float, 4>(warp_row_max);

    // only Bc_max_new_smem[x][0] is correct
    Bc_max_new_smem[tid / WARP_NUM_SEQLEN_K][tid % WARP_NUM_SEQLEN_K] =
        block_row_max;
    __syncthreads();

    for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
      float Bc_row_max_0 =
          Bc_max_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                          i * MMA_M + 0 * 8 + (lane_id / 4)][0];
      float Bc_row_max_1 =
          Bc_max_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                          i * MMA_M + 1 * 8 + (lane_id / 4)][0];

      float Bc_row_max_old_0 = lane_Bc_max_old[i][0];
      float Bc_row_max_old_1 = lane_Bc_max_old[i][1];

      Bc_row_max_0 = max(Bc_row_max_old_0, Bc_row_max_0);
      Bc_row_max_1 = max(Bc_row_max_old_1, Bc_row_max_1);

      for (int j = 0; j < WARP_ITER_SEQLEN_K; j++) {
        float2 t_reg_S_0 = __half22float2(HALF2(R_S[i][j][0]));
        float2 t_reg_S_1 = __half22float2(HALF2(R_S[i][j][1]));

        t_reg_S_0.x = __expf(__fmaf_rn(t_reg_S_0.x, scale, -Bc_row_max_0));
        t_reg_S_0.y = __expf(__fmaf_rn(t_reg_S_0.y, scale, -Bc_row_max_0));
        t_reg_S_1.x = __expf(__fmaf_rn(t_reg_S_1.x, scale, -Bc_row_max_1));
        t_reg_S_1.y = __expf(__fmaf_rn(t_reg_S_1.y, scale, -Bc_row_max_1));

        lane_row_sum_new[i][0] += t_reg_S_0.x + t_reg_S_0.y;
        lane_row_sum_new[i][1] += t_reg_S_1.x + t_reg_S_1.y;

        HALF2(R_S[i][j][0]) = __float22half2_rn(t_reg_S_0);
        HALF2(R_S[i][j][1]) = __float22half2_rn(t_reg_S_1);
      }

      // warp level reduce sum and store to smem
      lane_row_sum_new[i][0] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[i][0]);
      lane_row_sum_new[i][1] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[i][1]);

      if (lane_id % 4 == 0) {
        Bc_sum_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                        i * MMA_M + 0 * 8 + (lane_id / 4)][warp_seqlen_k_id] =
            lane_row_sum_new[i][0];
        Bc_sum_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                        i * MMA_M + 1 * 8 + (lane_id / 4)][warp_seqlen_k_id] =
            lane_row_sum_new[i][1];
      }
    }
    __syncthreads();

    // block level reduce sum and load from smem
    float warp_row_sum =
        Bc_sum_new_smem[tid / WARP_NUM_SEQLEN_K][tid % WARP_NUM_SEQLEN_K];
    float block_row_sum = warp_reduce_sum<float, 4>(warp_row_sum);
    // only Bc_sum_new_smem[x][0] is correct
    Bc_sum_new_smem[tid / WARP_NUM_SEQLEN_K][tid % WARP_NUM_SEQLEN_K] =
        block_row_sum;
    __syncthreads();

    // S is ready, compute O

    for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
      for (int j = 0; j < WARP_ITER_SEQLEN_K; j++) {
        R_Q[0][0] = R_S[i][j][0];
        R_Q[0][1] = __shfl_sync(0xffffffff, R_S[i][j][0], lane_id + 1, 4);
        R_Q[0][2] = __shfl_sync(0xffffffff, R_S[i][j][0], lane_id + 2, 4);
        R_Q[0][3] = __shfl_sync(0xffffffff, R_S[i][j][0], lane_id + 3, 4);

        R_Q[1][0] = R_S[i][j][1];
        R_Q[1][1] = __shfl_sync(0xffffffff, R_S[i][j][1], lane_id + 1, 4);
        R_Q[1][2] = __shfl_sync(0xffffffff, R_S[i][j][1], lane_id + 2, 4);
        R_Q[1][3] = __shfl_sync(0xffffffff, R_S[i][j][1], lane_id + 3, 4);

        if (lane_id % 4 == 0) {
          uint32_t store_S_smem_Br =
              warp_seqlen_qs_id * (MMA_M * WARP_ITER_SEQLEN_QS) + i * MMA_M +
              lane_id / 4;
          uint32_t store_S_smem_Bc =
              warp_seqlen_k_id * (MMA_N * WARP_ITER_SEQLEN_K) + j * MMA_N;
          uint32_t store_smem_S_address_0 =
              (store_S_smem_Br + 0) * Bc + store_S_smem_Bc;
          uint32_t store_smem_S_address_1 =
              (store_S_smem_Br + 8) * Bc + store_S_smem_Bc;
          LDST128BITS(S_tile_smem[store_smem_S_address_0]) =
              LDST128BITS(R_Q[0][0]);
          LDST128BITS(S_tile_smem[store_smem_S_address_1]) =
              LDST128BITS(R_Q[1][0]);
        }
      }
    }
    __syncthreads();
    if constexpr (STAGE > 1) {
      if (tile_K_seqlen + 1 < Tc) {
        CP_ASYNC_WAIT_GROUP(1);
      } else {
        CP_ASYNC_WAIT_GROUP(0);
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();
    fill_3D_regs<uint32_t, WARP_ITER_SEQLEN_QS, WARP_ITER_HEAD_DIM_V, 2>(R_O,
                                                                         0);
    for (int bk_Bc = 0; bk_Bc < hidden_Bc_ITER; bk_Bc++) {
      for (int i = 0; i < WARP_NUM_SEQLEN_QS; i++) {
        uint32_t S_smem_warp_offset =
            warp_seqlen_qs_id * MMA_M * WARP_ITER_SEQLEN_QS + i * MMA_M;
        uint32_t lane_S_smem_Br = S_smem_warp_offset + lane_id % MMA_M;
        uint32_t lane_S_smem_Bc = bk_Bc * MMA_K + (lane_id / MMA_M) * 8;
        uint32_t lane_S_smem_address =
            S_tile_smem_address +
            (lane_S_smem_Br * Bc + lane_S_smem_Bc) * sizeof(half);
        LDMATRIX_X4(R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3],
                    lane_S_smem_address);
      }
      for (int i = 0; i < WARP_ITER_HEAD_DIM_V; i++) {
        uint32_t V_smem_warp_offset =
            warp_seqlen_k_id * MMA_N * WARP_ITER_HEAD_DIM_V + i * MMA_N;
        uint32_t lane_V_smem_Bc = bk_Bc * MMA_K + lane_id % MMA_K;
        uint32_t lane_V_smem_d = V_smem_warp_offset;
        uint32_t lane_V_smem_address =
            V_tile_smem_address +
            (lane_V_smem_Bc * HEAD_DIM + lane_V_smem_d) * sizeof(half);
        LDMATRIX_X2_T(R_V[i][0], R_V[i][1], lane_V_smem_address);
      }

      for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
        for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
          HMMA16816(R_O[i][j][0], R_O[i][j][1], R_Q[i][0], R_Q[i][1], R_Q[i][2],
                    R_Q[i][3], R_V[j][0], R_V[j][1], R_O[i][j][0],
                    R_O[i][j][1]);
        }
      }
    }
    __syncthreads();
    for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
      float Bc_row_max_0 =
          Bc_max_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                          i * MMA_M + 0 * 8 + (lane_id / 4)][0];
      float Bc_row_max_1 =
          Bc_max_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                          i * MMA_M + 1 * 8 + (lane_id / 4)][0];
      float Bc_row_sum_0 =
          Bc_sum_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                          i * MMA_M + 0 * 8 + (lane_id / 4)][0];
      float Bc_row_sum_1 =
          Bc_sum_new_smem[warp_seqlen_qs_id * (WARP_ITER_SEQLEN_QS * MMA_M) +
                          i * MMA_M + 1 * 8 + (lane_id / 4)][0];
      float Bc_row_max_old_0 = lane_Bc_max_old[i][0];
      float Bc_row_max_old_1 = lane_Bc_max_old[i][1];

      Bc_row_max_0 = max(Bc_row_max_0, Bc_row_max_old_0);
      Bc_row_max_1 = max(Bc_row_max_1, Bc_row_max_old_1);

      Bc_row_max_old_0 = (tile_K_seqlen > 0) ? Bc_row_max_old_0 : Bc_row_max_0;
      Bc_row_max_old_1 = (tile_K_seqlen > 0) ? Bc_row_max_old_1 : Bc_row_max_1;

      float rescale_o_factor_0 = __expf(Bc_row_max_old_0 - Bc_row_max_0);
      float rescale_o_factor_1 = __expf(Bc_row_max_old_1 - Bc_row_max_1);

      for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
        float2 t_reg_O_0 = __half22float2(HALF2(R_O[i][j][0]));
        float2 t_reg_O_1 = __half22float2(HALF2(R_O[i][j][1]));
        float2 t_reg_Final_0 = __half22float2(HALF2(R_Final[i][j][0]));
        float2 t_reg_Final_1 = __half22float2(HALF2(R_Final[i][j][1]));

        t_reg_Final_0.x =
            __fmaf_rn(rescale_o_factor_0, t_reg_Final_0.x, t_reg_O_0.x);
        t_reg_Final_0.y =
            __fmaf_rn(rescale_o_factor_0, t_reg_Final_0.y, t_reg_O_0.y);
        t_reg_Final_1.x =
            __fmaf_rn(rescale_o_factor_1, t_reg_Final_1.x, t_reg_O_1.x);
        t_reg_Final_1.y =
            __fmaf_rn(rescale_o_factor_1, t_reg_Final_1.y, t_reg_O_1.y);
        HALF2(R_Final[i][j][0]) = __float22half2_rn(t_reg_Final_0);
        HALF2(R_Final[i][j][1]) = __float22half2_rn(t_reg_Final_1);
      }
      float Bc_row_sum_old_0 = lane_Bc_sum_old[i][0];
      float Bc_row_sum_old_1 = lane_Bc_sum_old[i][1];

      lane_Bc_sum_old[i][0] =
          __fmaf_rn(rescale_o_factor_0, Bc_row_sum_old_0, Bc_row_sum_0);
      lane_Bc_sum_old[i][1] =
          __fmaf_rn(rescale_o_factor_1, Bc_row_sum_old_1, Bc_row_sum_1);
      lane_Bc_max_old[i][0] = Bc_row_max_0;
      lane_Bc_max_old[i][1] = Bc_row_max_1;
    }
    if constexpr (STAGE > 1) {
      if ((tile_K_seqlen + 1) < Tc) {
        CP_ASYNC_WAIT_GROUP(0);
      }
      __syncthreads();
    }
  }
  __syncthreads();

  for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
    float rescale_factor_0 = __frcp_rn(lane_Bc_sum_old[i][0]);
    float rescale_factor_1 = __frcp_rn(lane_Bc_sum_old[i][1]);
    for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
      float2 t_reg_Final_0 = __half22float2(HALF2(R_Final[i][j][0]));
      float2 t_reg_Final_1 = __half22float2(HALF2(R_Final[i][j][1]));
      t_reg_Final_0.x = rescale_factor_0 * t_reg_Final_0.x;
      t_reg_Final_0.y = rescale_factor_0 * t_reg_Final_0.y;
      t_reg_Final_1.x = rescale_factor_1 * t_reg_Final_1.x;
      t_reg_Final_1.y = rescale_factor_1 * t_reg_Final_1.y;
      HALF2(R_Final[i][j][0]) = __float22half2_rn(t_reg_Final_0);
      HALF2(R_Final[i][j][1]) = __float22half2_rn(t_reg_Final_1);
    }
  }

  for (int i = 0; i < WARP_ITER_SEQLEN_QS; i++) {
    for (int j = 0; j < WARP_ITER_HEAD_DIM_V; j++) {
      R_Q[0][0] = R_Final[i][j][0];
      R_Q[1][0] = R_Final[i][j][1];
      R_Q[0][1] = __shfl_sync((0xffffffff), R_Final[i][j][0], lane_id + 1, 4);
      R_Q[0][2] = __shfl_sync((0xffffffff), R_Final[i][j][0], lane_id + 2, 4);
      R_Q[0][3] = __shfl_sync((0xffffffff), R_Final[i][j][0], lane_id + 3, 4);
      R_Q[1][1] = __shfl_sync((0xffffffff), R_Final[i][j][1], lane_id + 1, 4);
      R_Q[1][2] = __shfl_sync((0xffffffff), R_Final[i][j][1], lane_id + 2, 4);
      R_Q[1][3] = __shfl_sync((0xffffffff), R_Final[i][j][1], lane_id + 3, 4);

      if (lane_id % 4 == 0) {
        uint32_t store_O_gmem_Br =
            warp_seqlen_qs_id * (MMA_M * WARP_ITER_SEQLEN_QS) + i * MMA_M +
            lane_id / 4;
        uint32_t store_O_gmem_d =
            warp_seqlen_k_id * (MMA_N * WARP_ITER_HEAD_DIM_V) + j * MMA_N;
        uint32_t store_gmem_O_address_0 =
            O_gmem_offset + (store_O_gmem_Br + 0) * HEAD_DIM + store_O_gmem_d;
        uint32_t store_gmem_O_address_1 =
            O_gmem_offset + (store_O_gmem_Br + 8) * HEAD_DIM + store_O_gmem_d;
        LDST128BITS(O[store_gmem_O_address_0]) = LDST128BITS(R_Q[0][0]);
        LDST128BITS(O[store_gmem_O_address_1]) = LDST128BITS(R_Q[1][0]);
      }
    }
  }
}


template <int const QKV_HEADS, int const HEAD_DIM, int const MMA_M,
          int const MMA_N, int const MMA_K, int const STAGE, int const Bc = 64,
          int const WARP_NUM_SEQLEN_QS = 2, int const WARP_NUM_SEQLEN_K = 4>
__global__ void flash_attn_v2_split_q(half *Q, half *K, half *V, half *O,
                              int QKV_seqlen) {













                              }
template <int HEAD_DIM, int STAGE>
void launch_flash_attn_mma_stages(std::vector<half> &Q, std::vector<half> &K,
                                  std::vector<half> &V, std::vector<half> &O,
                                  uint32_t seqlen) {
  constexpr uint32_t warp_size = 32;
  constexpr uint32_t MMA_M = 16;
  constexpr uint32_t MMA_K = 16;
  constexpr uint32_t MMA_N = 8;
  constexpr uint32_t WARP_ITER_SEQLEN_QS = 2;
  constexpr uint32_t WARP_ITER_SEQLEN_K = 2;
  constexpr uint32_t WARP_NUM_SEQLEN_K = 4;
  constexpr uint32_t WARP_NUM_SEQLEN_QS = 2;
  constexpr uint32_t Br = 64;
  constexpr uint32_t Bc = 64;
  constexpr uint32_t threads =
      WARP_NUM_SEQLEN_K * WARP_NUM_SEQLEN_QS * warp_size;
  const uint32_t Q_smem_size = Br * HEAD_DIM;
  const uint32_t K_stages_smem_size = Bc * HEAD_DIM * STAGE;
  const uint32_t V_smem_size = Bc * HEAD_DIM;
  const uint32_t S_smem_size = Bc * Br;
  const uint32_t smem_size =
      (Q_smem_size + K_stages_smem_size + V_smem_size + S_smem_size) *
      sizeof(half);
  dim3 block(threads, 1, 1);
  dim3 grid(seqlen / Br, 1, 1);

  half *dQ, *dK, *dV, *dO;
  cudaMalloc(&dQ, Q.size() * sizeof(half));
  cudaMalloc(&dK, K.size() * sizeof(half));
  cudaMalloc(&dV, V.size() * sizeof(half));
  cudaMalloc(&dO, O.size() * sizeof(half));

  cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, K.data(), K.size() * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, V.data(), V.size() * sizeof(half), cudaMemcpyHostToDevice);

  cudaFuncSetAttribute(v2_fwd_kernel<1, HEAD_DIM, MMA_M, MMA_N, MMA_K, STAGE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  v2_fwd_kernel<1, HEAD_DIM, MMA_M, MMA_N, MMA_K, STAGE>
      <<<grid, block, smem_size>>>(dQ, dK, dV, dO, seqlen);

  cudaMemcpy(O.data(), dO, O.size() * sizeof(half), cudaMemcpyDeviceToHost);

  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
}