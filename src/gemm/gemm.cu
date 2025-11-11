#include <cassert>
#include <cstdint>

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
template<int const BM = 128, int const BN = 128, int const TM = 8, int const TN = 8, int const bK = 8>
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
                    alpha * result[reg_idx_a * TN + reg_idx_b];
        }
    }
}

namespace cuda::gemm::v1 {
    void bir_Sgemm(int m, int n, int k, float const *alpha, float const *A,
                   float const *B, float const *beta, float *C) {
        constexpr int M = 4096;
        constexpr int N = 4096;
        constexpr int K = 4096;
        constexpr int BM = 128;
        constexpr int BN = 128;
        constexpr int TM = 8;
        constexpr int TN = 8;
        assert(m == M && n == N && k == K);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));

        cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        dim3 grid(N / BN, M / BM);
        dim3 block(BN / TN, BM / TM);
        sgemm_shared_mem_2d
                <<<grid, block>>>(M, N, K, *alpha, d_A, d_B, *beta, d_C);
        cudaDeviceSynchronize();
        cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
} // namespace cuda::gemm::v1
