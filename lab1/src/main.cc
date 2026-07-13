/**
 * Lab1 体系结构相关编程
 * 实验1: 矩阵与向量内积 — Cache 优化
 * 实验2: n个数求和 — 超标量优化
 */
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
using namespace std;

// ==================== 实验1: 矩阵与向量内积 ====================

// 平凡算法：逐列计算内积 (j 外层)，每次内层 i 循环访问非连续内存（跨行，stride=N）
void matvec_basic(float** A, float* v, float* result, int N) {
    for (int j = 0; j < N; j++) {
        result[j] = 0.0f;
        for (int i = 0; i < N; i++)
            result[j] += A[i][j] * v[i];  // A[i][j] 跨行访问：stride=N
    }
}

// Cache 优化算法：逐行计算 (i 外层)，利用行优先连续存储优化 Cache
void matvec_cacheopt(float** A, float* v, float* result, int N) {
    for (int j = 0; j < N; j++) result[j] = 0.0f;
    for (int i = 0; i < N; i++) {
        float vi = v[i];
        for (int j = 0; j < N; j++)
            result[j] += A[i][j] * vi;    // A[i][j] 同行连续访问：Cache友好
    }
}

// ==================== 实验2: n个数求和 ====================

// 平凡算法：单链累加
float sum_basic(float* a, int N) {
    float s = 0.0f;
    for (int i = 0; i < N; i++) s += a[i];
    return s;
}

// 超标量优化：两路链式累加（利用指令级并行）
float sum_two_chain(float* a, int N) {
    float s1 = 0.0f, s2 = 0.0f;
    int i = 0;
    for (; i <= N - 2; i += 2) {
        s1 += a[i];
        s2 += a[i + 1];
    }
    for (; i < N; i++) s1 += a[i];
    return s1 + s2;
}

// 超标量优化：四路链式累加
float sum_four_chain(float* a, int N) {
    float s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    int i = 0;
    for (; i <= N - 4; i += 4) {
        s1 += a[i];
        s2 += a[i + 1];
        s3 += a[i + 2];
        s4 += a[i + 3];
    }
    for (; i < N; i++) s1 += a[i];
    return s1 + s2 + s3 + s4;
}

// 递归两两相加（树形归约）
float sum_recursive(float* a, int N) {
    if (N == 1) return a[0];
    if (N == 2) return a[0] + a[1];
    int half = N / 2;
    float* b = new float[half + 1];
    for (int i = 0; i < half; i++)
        b[i] = a[2 * i] + a[2 * i + 1];
    if (N % 2 == 1) b[half] = a[N - 1];
    float res = sum_recursive(b, (N % 2 == 1) ? half + 1 : half);
    delete[] b;
    return res;
}

// ==================== 测量辅助 ====================

template<typename Func>
double measure(Func f, int repeat) {
    auto t0 = chrono::high_resolution_clock::now();
    for (int r = 0; r < repeat; r++) f();
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
}

// ==================== Main ====================

int main() {
    // ---------- 实验1: 矩阵内积 ----------
    cout << "=== 实验1: 矩阵与向量内积 (Cache优化) ===" << endl;
    cout << "N,Basic_ms,CacheOpt_ms,Speedup" << endl;

    int sizes[] = {256, 512, 1024, 2048, 4096};
    for (int ni = 0; ni < 5; ni++) {
        int N = sizes[ni];
        // 分配
        float** A = new float*[N];
        for (int i = 0; i < N; i++) A[i] = new float[N];
        float* v = new float[N];
        float* res = new float[N];
        // 初始化
        for (int i = 0; i < N; i++) {
            v[i] = (float)(i % 100);
            for (int j = 0; j < N; j++)
                A[i][j] = (float)((i + j) % 100);
        }
        int rep = max(1, 1000000 / (N * N));
        double t1 = measure([&]() { matvec_basic(A, v, res, N); }, rep);
        double t2 = measure([&]() { matvec_cacheopt(A, v, res, N); }, rep);
        cout << N << "," << t1 << "," << t2 << "," << t1 / t2 << endl;
        for (int i = 0; i < N; i++) delete[] A[i];
        delete[] A; delete[] v; delete[] res;
    }

    // ---------- 实验2: n个数求和 ----------
    cout << "\n=== 实验2: n个数求和 (超标量优化) ===" << endl;
    cout << "N,Basic_ms,TwoChain_ms,FourChain_ms,Recursive_ms" << endl;

    int Ns[] = {100000, 1000000, 10000000, 50000000};
    for (int ni = 0; ni < 4; ni++) {
        int N = Ns[ni];
        float* a = new float[N];
        for (int i = 0; i < N; i++) a[i] = (float)(i % 1000) / 1000.0f;

        auto t1 = measure([&]() { float r = sum_basic(a, N); }, 20);
        auto t2 = measure([&]() { float r = sum_two_chain(a, N); }, 20);
        auto t3 = measure([&]() { float r = sum_four_chain(a, N); }, 20);
        auto t4 = measure([&]() { float r = sum_recursive(a, N); }, 20);

        cout << N << "," << t1 << "," << t2 << "," << t3 << "," << t4 << endl;
        delete[] a;
    }
    return 0;
}
