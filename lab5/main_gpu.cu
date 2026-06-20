/**
 * Lab5 GPU 高斯消去 — CUDA 版本
 * 运行: 在 Google Colab 或本地 nvcc 编译
 * Colab 用法: 上传此文件，在 notebook cell 中运行:
 *   !nvcc -O2 main_gpu.cu -o gpu_gauss
 *   !./gpu_gauss
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ==================== GPU Kernels ====================

// Kernel 1a: Division — 多 block 并行归一化，但不写对角线（避免 block 间竞争）
__global__ void division_kernel(float* A, int k, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int col = k + 1 + tid;
    if (col < N) {
        float diag = A[k * N + k];
        if (fabs(diag) > 1e-30f) {
            A[k * N + col] /= diag;
        }
    }
}

// Kernel 1b: 单线程设置对角线为 1（无竞争）
__global__ void set_diag_kernel(float* A, int k, int N) {
    A[k * N + k] = 1.0f;
}

// Kernel 2: Elimination — 每 block 负责一行，块内线程协同消去
__global__ void eliminate_kernel(float* A, int k, int N) {
    int row = k + 1 + blockIdx.x;   // 每个 block 负责一行
    if (row >= N) return;

    float factor = A[row * N + k];
    int col = k + 1 + threadIdx.x;

    // Grid-stride: 线程可能不够，多轮处理
    for (int c = col; c < N; c += blockDim.x) {
        A[row * N + c] -= factor * A[k * N + c];
    }

    // 同步后，仅 thread 0 清空第 k 列
    __syncthreads();
    if (threadIdx.x == 0) {
        A[row * N + k] = 0.0f;
    }
}

// ==================== CPU Serial Basiline ====================

void gauss_serial(float* A, int N) {
    for (int k = 0; k < N; k++) {
        float inv = 1.0f / A[k * N + k];
        for (int j = k + 1; j < N; j++)
            A[k * N + j] *= inv;
        A[k * N + k] = 1.0f;

        for (int i = k + 1; i < N; i++) {
            float factor = A[i * N + k];
            for (int j = k + 1; j < N; j++)
                A[i * N + j] -= factor * A[k * N + j];
            A[i * N + k] = 0.0f;
        }
    }
}

// ==================== Init & Validate ====================

void init_matrix(float* A, int N) {
    srand(1314);
    for (int i = 0; i < N * N; i++) A[i] = 0.0f;
    // 构造严格对角占优矩阵，确保高斯消去不产生 inf/nan
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                A[i * N + j] = (float)(rand() % 100 - 50);  // [-50, 50)
                row_sum += fabs(A[i * N + j]);
            }
        }
        A[i * N + i] = row_sum + 10.0f;  // 对角 > 非对角绝对值之和
    }
}

bool validate(float* A, float* B, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(A[i] - B[i]) > 1e-2) {
            printf("Mismatch at [%d]: CPU=%.6f GPU=%.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

// ==================== GPU Gauss Elimination ====================

float gpu_gauss(float* h_A, int N) {
    float *d_A;
    size_t size = N * N * sizeof(float);

    CHECK(cudaMalloc(&d_A, size));
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    int block_size = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int k = 0; k < N; k++) {
        int remaining = N - k - 1;

        if (remaining > 0) {
            int div_blocks = (remaining + block_size - 1) / block_size;
            division_kernel<<<div_blocks, block_size>>>(d_A, k, N);
            eliminate_kernel<<<remaining, block_size>>>(d_A, k, N);
        }
        // 无论 remaining 是否为 0，都要设对角线为 1（最后一行也需处理）
        set_diag_kernel<<<1, 1>>>(d_A, k, N);

        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));
    cudaFree(d_A);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// ==================== Main ====================

int main() {
    const int N_vals[] = {512, 1024, 2048};
    const int n_count = 3;

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s, CUDA Cores: %d, Clock: %d MHz, Mem: %.1f GB\n\n",
           prop.name, prop.multiProcessorCount * 128,  // rough estimate
           prop.clockRate / 1000, prop.totalGlobalMem / 1e9);

    printf("N,Serial_ms,GPU_ms,Speedup,Valid\n");

    for (int ni = 0; ni < n_count; ni++) {
        int N = N_vals[ni];
        size_t size = N * N * sizeof(float);
        float* h_A = (float*)malloc(size);
        float* h_ref = (float*)malloc(size);

        init_matrix(h_A, N);

        // CPU Serial
        memcpy(h_ref, h_A, size);
        auto t0 = std::chrono::high_resolution_clock::now();
        gauss_serial(h_ref, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // GPU
        float* h_gpu = (float*)malloc(size);
        memcpy(h_gpu, h_A, size);
        float gpu_ms = gpu_gauss(h_gpu, N);

        // Validate
        bool ok = validate(h_ref, h_gpu, N);

        printf("%d,%.2f,%.4f,%.2fx,%s\n",
               N, cpu_ms, gpu_ms, cpu_ms / gpu_ms, ok ? "PASS" : "FAIL");

        free(h_A); free(h_ref); free(h_gpu);
    }

    return 0;
}
