#include <cassert>
#include <cstdint>
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_CONST_FLOAT4(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])

template<int const BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       float const *A, float const *B,
                                       float beta, float *C) {
    uint const cRow = blockIdx.x;
    uint const cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    uint const threadCol = threadIdx.x % BLOCKSIZE;
    uint const threadRow = threadIdx.x / BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
#pragma unroll
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                    Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] =
            alpha * tmp + beta * C[threadRow * N + threadCol];
}

// computing TM*TN elements pre thread
// computing BM*BN elements pre block
// 2048*2048*2048 BM * BN: 128*128  256 threads pre block
template<int const BM = 128, int const BN = 128, int const TM = 8, int const TN = 8, int const bK = 32>
__global__ void sgemm_shared_mem_2d(int M, int N, int K, float alpha,
                                    float const *A, float const *B,
                                    float beta, float *C) {
    float reg_a[TM] = {0.0};
    float reg_b[TN] = {0.0};
    float result[TM * TN] = {0.0};

    constexpr uint32_t threads_per_block = (BM * BN) / (TM * TN);
    constexpr uint32_t threads_per_row = BN / TN;
    constexpr uint32_t load_a_per_thread = (bK * BM) / threads_per_block;
    constexpr uint32_t load_b_per_thread = (bK * BN) / threads_per_block;

    uint32_t thread_idx = threadIdx.y * threads_per_row + threadIdx.x;

    __shared__ float a_smem[bK * BM];
    __shared__ float b_smem[bK * BN];

    uint32_t k_iter = K / bK;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    for (int k = 0; k < k_iter; ++k) {
        for (int a = 0; a < load_a_per_thread; ++a) {
            uint32_t smem_idx = thread_idx + a * threads_per_block;
            uint32_t global_a_idx = (smem_idx % BM) * K + (smem_idx / BM) + k * bK;
            a_smem[smem_idx] = A[global_a_idx];
        }

        for (int b = 0; b < load_b_per_thread; ++b) {
            uint32_t smem_idx = thread_idx + b * threads_per_block;
            uint32_t global_b_idx = (smem_idx / BN) * N + (smem_idx % BN) + k * bK * N;
            b_smem[smem_idx] = B[global_b_idx];
        }

        __syncthreads();


        for (uint32_t dot_product_idx = 0; dot_product_idx < bK; ++dot_product_idx) {
            for (uint32_t i = 0; i < TM; i++) {
                uint32_t row_in_block = threadIdx.y * TM + i;
                reg_a[i] = a_smem[dot_product_idx * BM + row_in_block];
            }

            for (uint32_t i = 0; i < TN; i++) {
                uint32_t col_in_block = threadIdx.x * TN + i;
                reg_b[i] = b_smem[dot_product_idx * BN + col_in_block];
            }

            for (uint32_t reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                for (uint32_t reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                    result[reg_idx_a * TN + reg_idx_b] += reg_a[reg_idx_a] * reg_b[reg_idx_b];
                }
            }
        }
        __syncthreads();
    }
    for (uint32_t reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
        for (uint32_t reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
            C[(threadIdx.y * TM + reg_idx_a) * N + threadIdx.x * TN + reg_idx_b] =
                    alpha * result[reg_idx_a * TN + reg_idx_b] + beta * C[
                        (threadIdx.y * TM + reg_idx_a) * N + threadIdx.x * TN + reg_idx_b];
        }
    }
}

template<int const BM = 128, int const BN = 128, int const TM = 4, int const TN = 4, int const WM = 32, int const WN =
        64, int const bK = 32, int const WM_ITER = 2, int const WN_ITER = 2>
__global__ void sgemm_smem_warp_tiling(int M, int N, int K, float alpha,
                                       float const *A, float const *B,
                                       float beta, float *C) {
    __shared__ float a_smem[bK * BM];
    __shared__ float b_smem[bK * BN];

    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t threads_per_block = ((BM * BN) / (WM * WN)) * WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float reg_a[2 * TM] = {0.0};
    float reg_b[2 * TN] = {0.0};
    float result[4 * TM * TN] = {0.0};
    constexpr uint32_t load_a_per_thread = (bK * BM) / threads_per_block;
    constexpr uint32_t load_b_per_thread = (bK * BN) / threads_per_block;
    constexpr uint32_t warps_per_row = BN / WN;
    uint32_t warp_row = warp_id / warps_per_row;
    uint32_t warp_col = warp_id % warps_per_row;
    constexpr uint32_t inner_N = WN / WN_ITER;
    constexpr uint32_t inner_M = WM / WM_ITER;
    uint32_t inner_row = lane_id / (inner_N / TN);
    uint32_t inner_col = lane_id % (inner_N / TN);
    uint32_t k_iter = K / bK;
    for (int k = 0; k < k_iter; ++k) {
        for (int a = 0; a < load_a_per_thread; ++a) {
            uint32_t smem_idx = threadIdx.x + a * threads_per_block;
            uint32_t global_a_idx = (smem_idx % BM) * K + (smem_idx / BM) + k * bK;
            a_smem[smem_idx] = A[global_a_idx];
        }
        for (int b = 0; b < load_b_per_thread; ++b) {
            uint32_t smem_idx = threadIdx.x + b * threads_per_block;
            uint32_t global_b_idx = (smem_idx / BN) * N + (smem_idx % BN) + k * bK * N;
            b_smem[smem_idx] = B[global_b_idx];
        }

        __syncthreads();
        for (uint32_t dot_product_idx = 0; dot_product_idx < bK; ++dot_product_idx) {
            float *inner_A = &a_smem[warp_row * WM];
            float *inner_B = &b_smem[warp_col * WN];

            for (int a = 0; a < WM_ITER; a++) {
                for (int i = 0; i < TM; i++) {
                    reg_a[a * TM + i] = inner_A[inner_row * TM + i + dot_product_idx * BM + a * inner_M];
                }
            }
            for (int b = 0; b < WN_ITER; b++) {
                for (int i = 0; i < TN; i++) {
                    reg_b[b * TN + i] = inner_B[inner_col * TN + i + dot_product_idx * BN + b * inner_N];
                }
            }

            for (int a = 0; a < WM_ITER; a++) {
                for (int b = 0; b < WN_ITER; b++) {
                    for (int reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                        for (int reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                            result[a * TM * 2 * TN + b * TN + reg_idx_a * 2 * TN + reg_idx_b] += reg_a[
                                a * TM + reg_idx_a] * reg_b[b * TN + reg_idx_b];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    float *final_C = &C[warp_row * WM * N + warp_col * WN];
    for (int a = 0; a < WM_ITER; a++) {
        for (int b = 0; b < WN_ITER; b++) {
            for (uint32_t reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                for (uint32_t reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                    uint32_t row_offset = a * inner_M + inner_row * TM + reg_idx_a;
                    uint32_t col_offset = b * inner_N + inner_col * TN + reg_idx_b;
                    final_C[row_offset * N + col_offset] =
                            result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b]
                            + beta * final_C[row_offset * N + col_offset];
                }
            }
        }
    }
}


template<int const BM = 128, int const BN = 128, int const TM = 4, int const TN = 4, int const WM = 32, int const WN =
        64, int const bK = 32, int const WM_ITER = 2, int const WN_ITER = 2>
__global__ void sgemm_smem_warp_tiling_vec4(int M, int N, int K, float alpha,
                                            float const *A, float const *B,
                                            float beta, float *C) {
    float4 ldg_a;
    __shared__ float a_smem[bK * BM];
    __shared__ float b_smem[bK * BN];
    constexpr uint32_t VEC_SIZE = 4;
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t threads_per_block = ((BM * BN) / (WM * WN)) * WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float reg_a[2 * TM] = {0.0};
    float reg_b[2 * TN] = {0.0};
    float result[4 * TM * TN] = {0.0};
    constexpr uint32_t load_a_per_thread = ((bK * BM) / threads_per_block) / VEC_SIZE;
    constexpr uint32_t load_b_per_thread = ((bK * BN) / threads_per_block) / VEC_SIZE;
    constexpr uint32_t warps_per_row = BN / WN;
    uint32_t warp_row = warp_id / warps_per_row;
    uint32_t warp_col = warp_id % warps_per_row;
    constexpr uint32_t inner_N = WN / WN_ITER;
    constexpr uint32_t inner_M = WM / WM_ITER;
    uint32_t inner_row = lane_id / (inner_N / TN);
    uint32_t inner_col = lane_id % (inner_N / TN);
    uint32_t k_iter = K / bK;
    for (int k = 0; k < k_iter; ++k) {
        for (int a = 0; a < load_a_per_thread; ++a) {
            uint32_t smem_idx = ((threadIdx.x % (bK / VEC_SIZE)) * VEC_SIZE * BM) + (threadIdx.x / (bK / VEC_SIZE)) + (
                                    (a * threads_per_block * VEC_SIZE) / bK);
            uint32_t global_a_idx = (smem_idx % BM) * K + (smem_idx / BM) + k * bK;
            ldg_a = FETCH_CONST_FLOAT4(A[global_a_idx]);
            a_smem[smem_idx] = ldg_a.x;
            a_smem[smem_idx + BM] = ldg_a.y;
            a_smem[smem_idx + 2 * BM] = ldg_a.z;
            a_smem[smem_idx + 3 * BM] = ldg_a.w;
        }
        for (int b = 0; b < load_b_per_thread; ++b) {
            uint32_t smem_idx = threadIdx.x * VEC_SIZE + b * threads_per_block * VEC_SIZE;
            uint32_t global_b_idx = (smem_idx / BN) * N + (smem_idx % BN) + k * bK * N;
            FETCH_FLOAT4(b_smem[smem_idx]) = FETCH_CONST_FLOAT4(B[global_b_idx]);
        }
        __syncthreads();

        for (uint32_t dot_product_idx = 0; dot_product_idx < bK; ++dot_product_idx) {
            float *inner_A = &a_smem[warp_row * WM];
            float *inner_B = &b_smem[warp_col * WN];

            for (int a = 0; a < WM_ITER; a++) {
                for (int i = 0; i < TM / VEC_SIZE; i++) {
                    FETCH_FLOAT4(reg_a[a * TM + i * VEC_SIZE]) = FETCH_FLOAT4(
                        inner_A[inner_row * TM + i * VEC_SIZE + dot_product_idx * BM + a * inner_M]);
                }
            }
            for (int b = 0; b < WN_ITER; b++) {
                for (int i = 0; i < TN / VEC_SIZE; i++) {
                    FETCH_FLOAT4(reg_b[b * TN + i * VEC_SIZE]) = FETCH_FLOAT4(
                        inner_B[inner_col * TN + i * VEC_SIZE + dot_product_idx * BN + b * inner_N]);
                }
            }

            for (int a = 0; a < WM_ITER; a++) {
                for (int b = 0; b < WN_ITER; b++) {
                    for (int reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                        for (int reg_idx_b = 0; reg_idx_b < TN; reg_idx_b++) {
                            result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b] += reg_a[
                                a * TM + reg_idx_a] * reg_b[b * TN + reg_idx_b];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    float *final_C = &C[warp_row * WM * N + warp_col * WN];
    for (int a = 0; a < WM_ITER; a++) {
        for (int b = 0; b < WN_ITER; b++) {
            for (uint32_t reg_idx_a = 0; reg_idx_a < TM; reg_idx_a++) {
                for (uint32_t reg_idx_b = 0; reg_idx_b < TN / VEC_SIZE; reg_idx_b++) {
                    uint32_t row_offset = a * inner_M + inner_row * TM + reg_idx_a;
                    uint32_t col_offset = b * inner_N + inner_col * TN + reg_idx_b * VEC_SIZE;
                    FETCH_FLOAT4(final_C[row_offset * N + col_offset]) =
                            FETCH_FLOAT4(
                                result[a * TM * WN_ITER * TN + b * TN + reg_idx_a * WN_ITER * TN + reg_idx_b * VEC_SIZE
                                ]);
                }
            }
        }
    }
}

namespace cuda::gemm::v1 {
    void bir_Sgemm(int m, int n, int k, float const *alpha, float const *A,
                   float const *B, float const *beta, float *C) {
        constexpr int M = 2048;
        constexpr int N = 2048;
        constexpr int K = 2048;
        constexpr int BM = 128;
        constexpr int BN = 128;
        constexpr int TM = 4;
        constexpr int TN = 4;
        assert(m == M && n == N && k == K);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));

        cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        dim3 grid(N / BN, M / BM);
        dim3 block(256);
        sgemm_smem_warp_tiling_vec4
                <<<grid, block>>>(M, N, K, *alpha, d_A, d_B, *beta, d_C);
        cudaDeviceSynchronize();
        cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
} // namespace cuda::gemm::v1
