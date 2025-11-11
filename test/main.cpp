#include <iostream>
#include <vector>

#include "../include/before_i_rise.h"

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    // 分配矩阵
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    // 初始化矩阵数据，随便用随机数
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    float alpha = 1.0f;
    float beta = 0.0f;

    // 调用你写的 CUDA GEMM
    cuda::gemm::v1::bir_Sgemm(M, N, K, &alpha, A.data(), B.data(), &beta, C.data());

    // 简单验证：输出 C 的前几个元素
    std::cout << "C[0..9]: ";
    for (int i = 0; i < 10; ++i) std::cout << C[i] << " ";
    std::cout << std::endl;

    return 0;
}