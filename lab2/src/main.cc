#include <vector>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <arm_neon.h>

const int N = 2048; 

float** alloc_matrix(int n) {
    float** A = new float*[n];
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
    }
    return A;
}

void free_matrix(float** A, int n) {
    for (int i = 0; i < n; i++) {
        delete[] A[i];
    }
    delete[] A;
}

void init_matrix(float** A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) { A[i][j] = 0.0f; }
        A[i][i] = 1.0f;
        for (int j = i + 1; j < n; j++) { A[i][j] = rand() % 100; }
    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) { A[i][j] += A[k][j]; }
        }
    }
}

void copy_matrix(float** src, float** dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

// 串行版本
void gauss_serial(float** A, int n) {
    for (int k = 0; k < n; k++) {
        float vt = A[k][k];
        for (int j = k + 1; j < n; j++) { A[k][j] = A[k][j] / vt; }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) { A[i][j] = A[i][j] - factor * A[k][j]; }
            A[i][k] = 0.0f;
        }
    }
}

// NEON 向量化版本
void gauss_neon(float** A, int n) {
    for (int k = 0; k < n; k++) {
        float vt = A[k][k];
        float inv_vt = 1.0f / vt;
        float32x4_t inv_vt_vec = vdupq_n_f32(inv_vt); 
        
        int j = k + 1;
        for (; j <= n - 4; j += 4) {
            float32x4_t a_kj = vld1q_f32(&A[k][j]); 
            float32x4_t res = vmulq_f32(a_kj, inv_vt_vec); 
            vst1q_f32(&A[k][j], res); 
        }
        for (; j < n; j++) { A[k][j] = A[k][j] * inv_vt; }
        A[k][k] = 1.0f;

        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            float32x4_t factor_vec = vdupq_n_f32(factor); 
            
            j = k + 1;
            for (; j <= n - 4; j += 4) {
                float32x4_t a_ij = vld1q_f32(&A[i][j]); 
                float32x4_t a_kj = vld1q_f32(&A[k][j]); 
                float32x4_t res = vmlsq_f32(a_ij, a_kj, factor_vec); 
                vst1q_f32(&A[i][j], res); 
            }
            for (; j < n; j++) { A[i][j] = A[i][j] - factor * A[k][j]; }
            A[i][k] = 0.0f;
        }
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Matrix Size : " << N << "x" << N << std::endl;

    // 准备数据
    float** A_serial = alloc_matrix(N);
    float** A_neon = alloc_matrix(N);
    init_matrix(A_serial, N);
    copy_matrix(A_serial, A_neon, N); // 确保两个算法跑的数据完全一致

    // 1. 测试串行版本
    auto Start1 = std::chrono::high_resolution_clock::now();
    gauss_serial(A_serial, N);
    auto End1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = End1 - Start1;
    std::cout << "Serial Latency : " << elapsed1.count() << " (ms) " << std::endl;

    // 2. 测试 NEON 版本
    auto Start2 = std::chrono::high_resolution_clock::now();
    gauss_neon(A_neon, N);
    auto End2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = End2 - Start2;
    std::cout << "NEON Latency   : " << elapsed2.count() << " (ms) " << std::endl;

    // 3. 计算加速比
    std::cout << "Speedup        : " << elapsed1.count() / elapsed2.count() << "x" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    free_matrix(A_serial, N);
    free_matrix(A_neon, N);
    return 0;
}