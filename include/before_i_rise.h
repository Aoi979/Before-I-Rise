namespace cuda::gemm {
namespace final {
void bir_Sgemm(bool transa, bool transb, int m, int n, int k,
               float const *alpha, float const *A, int lda, float const *B,
               int ldb, float const *beta, float *C, int ldc);

}

namespace v1 {
void bir_Sgemm(int m, int n, int k, float const *alpha, float const *A,
               float const *B, float const *beta, float *C);
}
} // namespace cuda::gemm
