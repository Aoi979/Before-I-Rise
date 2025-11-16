#pragma once
#include "../../../../opt/cuda/targets/x86_64-linux/include/cuda_fp16.h"

namespace cuda::gemm {
    namespace final {
        void bir_Sgemm(bool transa, bool transb, int m, int n, int k,
                       float const *alpha, float const *A, int lda, float const *B,
                       int ldb, float const *beta, float *C, int ldc);
    }

    namespace v1 {
        void bir_Sgemm(int m, int n, int k, float const *alpha, float const *A,
                       float const *B, float const *beta, float *C);

        void bir_Hgemm(int m, int n, int k, half const *alpha, half const *A,
                       half const *B, half const *beta, float *C);
    }
}
