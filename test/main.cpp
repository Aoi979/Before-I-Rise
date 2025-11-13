#include <iostream>
#include <vector>
#include <iomanip>
#include "../include/before_i_rise.h"
#include <cstdlib>
#include <cmath>


// CPU版本的GEMM计算
void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}


int main() {
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // 初始化随机数据
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    float alpha = 1.0f;
    float beta = 0.0f;

    // 调用 CUDA GEMM（GPU 版本）
    std::cout << "Running GPU GEMM..." << std::endl;
    cuda::gemm::v1::bir_Sgemm(M, N, K, &alpha, A.data(), B.data(), &beta, C_gpu.data());

    // 调用 CPU GEMM
    std::cout << "Running CPU GEMM..." << std::endl;
    cpu_gemm(M, N, K, alpha, A.data(), B.data(), beta, C_cpu.data());

    // ----------- 验证全部元素 -----------
    std::cout << "验证全部 " << M * N << " 个元素..." << std::endl;

    int error_count = 0;
    int correct_count = 0;
    const float tol = 1e-3f;
    const int max_errors_to_show = 10;
    const int max_correct_to_show = 10; // 显示前几个正确的元素
    double total_diff = 0.0;
    float max_diff = 0.0f;



    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            float gpu_val = C_gpu[idx];
            float cpu_val = C_cpu[idx];
            float diff = std::fabs(cpu_val - gpu_val);

            // 统计信息
            total_diff += diff;
            if (diff > max_diff) max_diff = diff;

            // 错误检查
            if (diff > tol) {
                error_count++;
                if (error_count <= max_errors_to_show) {
                    std::cout << "\n错误位置 C[" << i << "][" << j << "]: "
                            << "GPU=" << std::setprecision(6) << gpu_val
                            << " CPU=" << std::setprecision(6) << cpu_val
                            << " 差值=" << std::setprecision(6) << diff << " ❌";
                }
            } else {
                correct_count++;
                if (correct_count <= max_correct_to_show) {
                    std::cout << "\n正确位置 C[" << i << "][" << j << "]: "
                            << "GPU=" << std::setprecision(6) << gpu_val
                            << " CPU=" << std::setprecision(6) << cpu_val
                            << " 差值=" << std::setprecision(6) << diff << " ✅";
                }
            }
        }
    }


    // 输出结果
    std::cout << "\n === 验证结果 ===" << std::endl;
    std::cout << "矩阵大小: " << M << " × " << N << " (共 " << M * N << " 个元素)" << std::endl;
    std::cout << "容差: " << tol << std::endl;
    std::cout << "正确元素数量: " << correct_count << std::endl;
    std::cout << "错误元素数量: " << error_count << std::endl;
    std::cout << "正确率: " << std::setprecision(4) << (correct_count * 100.0 / (M * N)) << "%" << std::endl;
    std::cout << "平均绝对误差: " << std::setprecision(8) << (total_diff / (M * N)) << std::endl;
    std::cout << "最大绝对误差: " << std::setprecision(8) << max_diff << std::endl;

    if (error_count == 0) {
        std::cout << "✅ 所有 " << M * N << " 个元素完全匹配！GPU 结果100%正确！" << std::endl;
    } else {
        std::cout << "❌ 检测到 " << error_count << " 个错误元素";
        if (error_count > max_errors_to_show) {
            std::cout << " (显示前 " << max_errors_to_show << " 个)";
        }
        std::cout << std::endl;
    }

    return error_count == 0 ? 0 : 1;
}
