#include <cstdint>
#include <sys/types.h>

namespace cuda::reduce {
    template<uint32_t const ACTIVE_THREADS = 32>
    __device__ __forceinline__ float warp_reduce_sum(float sum) {
        if constexpr (ACTIVE_THREADS >= 32) {
            sum += __shfl_down_sync(0xffffffff, sum, 16);
        }
        if constexpr (ACTIVE_THREADS >= 16) {
            sum += __shfl_down_sync(0xffffffff, sum, 8);
        }
        if constexpr (ACTIVE_THREADS >= 8) {
            sum += __shfl_down_sync(0xffffffff, sum, 4);
        }
        if constexpr (ACTIVE_THREADS >= 4) {
            sum += __shfl_down_sync(0xffffffff, sum, 2);
        }
        if constexpr (ACTIVE_THREADS >= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, 1);
        }
        return sum;
    }

    template<uint32_t const ACTIVE_THREADS = 32>
    __device__ __forceinline__ float warp_reduce_max(float max) {
        if constexpr (ACTIVE_THREADS >= 32) {
            max = fmaxf(max, __shfl_down_sync(0xffffffff, max, 16));
        }
        if constexpr (ACTIVE_THREADS >= 16) {
            max = fmaxf(max, __shfl_down_sync(0xffffffff, max, 8));
        }
        if constexpr (ACTIVE_THREADS >= 8) {
            max = fmaxf(max, __shfl_down_sync(0xffffffff, max, 4));
        }
        if constexpr (ACTIVE_THREADS >= 4) {
            max = fmaxf(max, __shfl_down_sync(0xffffffff, max, 2));
        }
        if constexpr (ACTIVE_THREADS >= 2) {
            max = fmaxf(max, __shfl_down_sync(0xffffffff, max, 1));
        }
        return max;
    }

    namespace unsafe {
        // 约束： Block内warp数量不得超过WARP_SIZE(32)
        // d_out为数组，每个block会需要一个位置写入block
        // reduce的结果，而不是使用atomic_add直接得到结果 a: |0 1 2 ... BLOCKSIZE-1 |
        // BLOCKSIZE ... 2*BLOCKSIZE-1|......|GRIDSIZE ......| thread 0会执行 0 +
        // a[BLOCKSIZE], 每个Block有线程数n,
        // 那么Block负责处理的数据最起码是包含2n个数的区间的求和 并且可以跨步到后续数组
        // 总之这个设计的挺麻烦的,对形状的约束多，不考虑再整这种了
        template<uint32_t const BLOCKSIZE>
        __global__ void reduce_sum(float *d_in, float *d_out, uint32_t n) {
            constexpr uint32_t WARP_SIZE = 32;
            float sum = 0.0f;
            uint32_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
            uint32_t tid = threadIdx.x;
            uint32_t grid_size = blockDim.x * 2 * gridDim.x;

            while (i < n) {
                sum += d_in[i] + d_in[i + BLOCKSIZE];
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
                sum = warp_reduce_sum<BLOCKSIZE / WARP_SIZE>(sum);
            }
            if (tid == 0) {
                d_out[blockIdx.x] = sum;
            }
        }
    } // namespace unsafe

    // grid(N/128),block(128)
    // in: 1xN, out = sum(in)
    template<const uint32_t BLOCKSIZE = 128>
    __global__ void reduce_sum(float *in, float *out, int N) {
        constexpr uint32_t WARP_SIZE = 32;
        uint32_t tid = threadIdx.x;
        int idx = blockIdx.x * BLOCKSIZE + tid;
        constexpr uint32_t WARPS_NUM = (BLOCKSIZE + WARP_SIZE - 1) / WARP_SIZE;
        __shared__ float reduce_smem[WARPS_NUM];
        float sum = (idx < N) ? in[idx] : 0.0f;
        uint32_t warp_id = tid / WARP_SIZE;
        uint32_t lane_id = tid % WARP_SIZE;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            reduce_smem[warp_id] = sum;
        }
        __syncthreads();
        sum = (lane_id < WARPS_NUM) ? reduce_smem[lane_id] : 0.0f;
        if (warp_id == 0) {
            sum = warp_reduce_sum(sum);
        }
        if (tid == 0) {
            atomicAdd(out, sum);
        }
    }

    // grid(N/128), block(128/4)
    // in: 1xN, out = sum(in)
    template<const uint32_t BLOCKSIZE>
    __global__ void reduce_sum_vec4(float *in, float *out, int N) {
    }
} // namespace cuda::reduce
