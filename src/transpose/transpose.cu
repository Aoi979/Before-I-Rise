#include <cstdint>

template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void transpose_naive(int M, int N, const float *input, float *output) {
    const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    if (y >= N || x >= M) {
        return;
    }
    output[y * M + x] = input[x * N + y];
}

template<int BLOCK_DIM_X, int BLOCK_DIM_Y, int BM = 32, int BN = 32>
__global__ void transpose_smem_32x32(int M, int N, const float *input, float *output) {

}
