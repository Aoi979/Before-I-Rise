#include "../util/util.h"
#include <cstdint>
#include <cuda_fp16.h>

template <int const QKV_HEADS, int const HEAD_DIM, int const MMA_M,
          int const MMA_N, int const MMA_K, int const STAGE, int const Bc = 64,
          int const WARP_NUM_SEQLEN_QS = 4, int const WARP_NUM_SEQLEN_K = 1>
__global__ void flash_attn_v2_split_q_tiling(half *Q, half *K, half *V, half *O,
                                             int QKV_seqlen) {
  // static assertions and constexpr calculations
  static_assert(MMA_M == 16 && MMA_N == 8 && MMA_K == 16);
  constexpr uint32_t Br = 64;
  // 1
  static_assert(Br / (MMA_M * WARP_NUM_SEQLEN_QS) == 1,
                "Br must be divisible by MMA_M * WARP_NUM_SEQLEN_QS");
  constexpr uint32_t WARP_ITER_SEQLEN_QS = 1;

  // 8
  static_assert(Bc % (MMA_N * WARP_NUM_SEQLEN_K) == 0,
                "Bc must be divisible by MMA_N * WARP_NUM_SEQLEN_K");
  constexpr uint32_t WARP_ITER_SEQLEN_K = Bc / (MMA_N * WARP_NUM_SEQLEN_K);
  // 8
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

  // (warp_id / WARP_NUM_SEQLEN_QS) == 0
  constexpr uint32_t warp_seqlen_k_id = 0;

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

  constexpr uint32_t VECSIZE = 8;

  constexpr uint32_t smem_Q_row_thread_num = HEAD_DIM / VECSIZE;

  uint32_t load_smem_Q_Br = tid / smem_Q_row_thread_num;
  uint32_t load_smem_Q_d = (tid % smem_Q_row_thread_num) * VECSIZE;

  constexpr uint32_t load_smem_Q_stride = (THREAD_SIZE * VECSIZE) / HEAD_DIM;

  // V_HEAD_DIM == HEAD_DIM
  constexpr uint32_t smem_KV_row_thread_num = HEAD_DIM / VECSIZE;

  uint32_t load_smem_KV_Bc = tid / smem_KV_row_thread_num;
  uint32_t load_smem_KV_d = (tid % smem_KV_row_thread_num) * VECSIZE;

  constexpr uint32_t load_smem_KV_stride = (THREAD_SIZE * VECSIZE) / HEAD_DIM;

  extern __shared__ half sram[];

  constexpr uint32_t Q_tile_size = Br * MMA_K;

  constexpr uint32_t K_tile_size = Bc * MMA_K;

  constexpr uint32_t V_tile_size = MMA_K * HEAD_DIM;

  auto Q_tile_smem = sram;

  auto K_tile_smem = Q_tile_smem + STAGE * Q_tile_size;

  // reuse Q+K smem
  auto V_tile_smem = Q_tile_smem;

  uint32_t Q_tile_smem_address = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t K_tile_smem_address = __cvta_generic_to_shared(K_tile_smem);
  uint32_t V_tile_smem_address = __cvta_generic_to_shared(V_tile_smem);

  float lane_Bc_max_old[WARP_ITER_SEQLEN_QS][2];
  fill_2D_regs<float, WARP_ITER_SEQLEN_QS, 2>(lane_Bc_max_old, -INFINITY);
  float lane_Bc_sum_old[WARP_ITER_SEQLEN_QS][2] = {};

  using REG_SIZE_T = uint32_t;

  // 16 * 16 * 2 / (32 * 4) = 4
  REG_SIZE_T R_Q[4];
  // 16 * 8
  REG_SIZE_T R_K[WARP_ITER_SEQLEN_K][2];
  // 16 * 8
  REG_SIZE_T R_V[WARP_ITER_HEAD_DIM_V][2];
  // 16 * 8
  REG_SIZE_T R_S[WARP_ITER_SEQLEN_K][2] = {};
  // 16 * 8
  REG_SIZE_T R_O[WARP_ITER_HEAD_DIM_V][2] = {};
  // 16 * 8
  REG_SIZE_T R_Final[WARP_ITER_HEAD_DIM_V][2] = {};

  for (uint32_t K_tile_id = 0; K_tile_id < Tc; K_tile_id++) {
    if constexpr (STAGE > 1) {
        for (uint32_t stage = 0; stage < STAGE -1; stage++) {


        }
    }
  }
}