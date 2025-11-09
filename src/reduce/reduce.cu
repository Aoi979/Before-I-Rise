#include <cstdint>

namespace cuda::reduce {
    template<const uint32_t BLOCKSIZE>
    __device__ __forceinline__ float warp_reduce_sum(float sum) {
        if constexpr (BLOCKSIZE >= 32) {
            sum += __shfl_down_sync(0xffffffff, sum, 16);
        }
        if constexpr (BLOCKSIZE >= 16) {
            sum += __shfl_down_sync(0xffffffff, sum, 8);
        }
        if constexpr (BLOCKSIZE >= 8) {
            sum += __shfl_down_sync(0xffffffff, sum, 4);
        }
        if constexpr (BLOCKSIZE >= 4) {
            sum += __shfl_down_sync(0xffffffff, sum, 2);
        }
        if constexpr (BLOCKSIZE >= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, 1);
        }
        return sum;
    }

    template<const uint32_t BLOCKSIZE>
    __global__ void reduce_sum(float *d_in, float *d_out, uint32_t n) {
        constexpr uint32_t WARP_SIZE = 32;
        float sum = 0.0f;
        uint32_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        uint32_t tid = threadIdx.x;
        uint32_t grid_size = blockDim.x * 2 * gridDim.x;

        while (i < n) {
            sum += d_in[i] + d_in[i+BLOCKSIZE];
            i += grid_size;
        }

        __shared__ float warp_sum[WARP_SIZE];
        uint32_t lane_id = threadIdx.x % WARP_SIZE;
        uint32_t warp_id = threadIdx.x / WARP_SIZE;

        sum = warp_reduce_sum<BLOCKSIZE>(sum);

        if (lane_id == 0) {
            warp_sum[warp_id] = sum;
        }
        __syncthreads();

        sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warp_sum[lane_id] : 0.0f;

        if (warp_id == 0) {
            sum = warp_reduce_sum<BLOCKSIZE/WARP_SIZE>(sum);
        }
        if (tid ==0) {
            d_out[blockIdx.x] = sum;
        }
    }
}
