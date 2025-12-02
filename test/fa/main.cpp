#include <before_i_rise.h>
#include <iostream>
#include <random>
template <typename T>
struct matrix {
    int rows, cols;
    std::vector<T> data;

    matrix(int r, int c) : rows(r), cols(c), data(r * c) {}

    inline T& operator()(int r, int c) { return data[r * cols + c]; }
    inline const T& operator()(int r, int c) const { return data[r * cols + c]; }

    inline T* ptr() { return data.data(); }
    inline const T* ptr() const { return data.data(); }
};

template <typename T>
void fill_random(matrix<T>& m) {
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> dist(0.f, 1.f);

    for (auto& x : m.data) {
        x = dist(gen);
    }
}

template <typename T>
void validate_results(
    const matrix<T>& gpu,
    const matrix<T>& cpu,
    float tol = 1e-3f
) {
    int errors = 0;
    int correct = 0;
    float max_diff = 0.f;
    double total_diff = 0.0;

    const int M = gpu.rows;
    const int N = gpu.cols;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            float diff = std::fabs(gpu.data[idx] - cpu.data[idx]);

            total_diff += diff;
            max_diff = std::max(max_diff, diff);

            if (diff > tol) errors++;
            else correct++;
        }
    }

    std::cout << "正确: " << correct << "\n";
    std::cout << "错误: " << errors << "\n";
    std::cout << "平均误差: " << total_diff / (M * N) << "\n";
    std::cout << "最大误差: " << max_diff << "\n";

    if (errors == 0) {
        std::cout << "结果完全一致！\n";
    } else {
        std::cout << "有误差。\n";
    }
}



int main(){
    return 0;
}