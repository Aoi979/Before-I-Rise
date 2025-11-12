#include <iostream>
#include <vector>

#include "../include/before_i_rise.h"
#include <cstdlib>
#include <cmath>


void cpu_sgemm(int M, int N, int K,
               float alpha,
               const float* A,
               const float* B,
               float beta,
               float* C)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}


int main() {
    constexpr int M =4096;
    constexpr int N =4096;
    constexpr int K =4096;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);


    for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    float alpha = 1.0f;
    float beta = 0.0f;

    // 调用 CUDA GEMM（GPU 版本）
    cuda::gemm::v1::bir_Sgemm(M, N, K, &alpha, A.data(), B.data(), &beta, C.data());

    // ----------- 验证前 10 个元素 -----------
    std::cout << "Checking first 10 results vs CPU..." << std::endl;
    bool all_ok = true;
    const float tol = 1e-3f;

    for (int idx = 0; idx < 10; ++idx) {
        int i = 0;     // 只验证第 0 行
        int j = idx;   // 前 10 个元素
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[i * K + k] * B[k * N + j];
        float cpu_val = alpha * sum + beta * 0.0f;
        float gpu_val = C[i * N + j];
        float diff = std::fabs(cpu_val - gpu_val);
        std::cout << "C[" << j << "]: GPU=" << gpu_val
                  << " CPU=" << cpu_val
                  << " diff=" << diff;
        if (diff <= tol)
            std::cout << " ✅\n";
        else {
            std::cout << " ❌\n";
            all_ok = false;
        }
    }

    if (all_ok)
        std::cout << "✅ 前 10 个元素均匹配，GPU 结果正确。" << std::endl;
    else
        std::cout << "❌ 检测到差异，可能存在计算或同步问题。" << std::endl;

    return 0;
}
