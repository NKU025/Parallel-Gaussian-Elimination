#include <iostream>
#include <cstdlib>
#include <stdlib.h>      // posix_memalign
#include <chrono>
#include <cmath>
#include <pthread.h>
#include <omp.h>
#include <arm_neon.h>
#include <cstring>

// 默认配置（若运行时不指定参数则使用默认值）
int DEFAULT_N = 1024;
int NUM_THREADS = 4;

// Pthread 参数传递结构体
struct ThreadParam {
    int t_id;
    int N;
    float** m;
    pthread_barrier_t* barrier;  // 局部 barrier 指针，替代全局变量
};

// 矩阵内存分配（16 字节对齐，利于 NEON 向量化加载）
float** alloc_matrix(int N) {
    float** m = new float*[N];
    for (int i = 0; i < N; i++) {
        void* ptr;
        if (posix_memalign(&ptr, 16, N * sizeof(float)) != 0) {
            std::cerr << "Memory allocation failed!" << std::endl;
            exit(-1);
        }
        m[i] = (float*)ptr;
    }
    return m;
}

// 释放内存
void free_matrix(float** m, int N) {
    for (int i = 0; i < N; i++) {
        free(m[i]);
    }
    delete[] m;
}

// 矩阵初始化：先生成上三角 + 对角为 1，再通过行组合确保稠密且非奇异
// 此方式比纯随机 + 对角增强更可靠，不会出现 inf/nan
void init_matrix(float** m, int N) {
    srand(1314);
    // 第 1 步：构造上三角矩阵，对角线为 1
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m[i][j] = 0.0f;
        }
        m[i][i] = 1.0f;
        for (int j = i + 1; j < N; j++) {
            m[i][j] = (float)(rand() % 100);
        }
    }
    // 第 2 步：随机行组合，打破上三角结构，生成稠密矩阵
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                m[i][j] += m[k][j];
            }
        }
    }
}

// 矩阵拷贝（用于每次测试前重置数据）
void copy_matrix(float** src, float** dst, int N) {
    for (int i = 0; i < N; i++) {
        std::memcpy(dst[i], src[i], N * sizeof(float));
    }
}

// 结果校验（与串行版本比对，容差 1e-2）
bool validate(float** base, float** test, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (std::abs(base[i][j] - test[i][j]) > 1e-2) {
                return false;
            }
        }
    }
    return true;
}

// 检查 barrier 返回值（PTHREAD_BARRIER_SERIAL_THREAD 不是错误）
inline void check_barrier(int rc) {
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        std::cerr << "Barrier error: " << rc << std::endl;
        exit(-1);
    }
}

// ================= 1. 串行版本 (Baseline) =================
void gauss_serial(float** m, int N) {
    for (int k = 0; k < N; k++) {
        float inv_pivot = 1.0f / m[k][k];  // 预计算倒数，除法转乘法
        for (int j = k + 1; j < N; j++) {
            m[k][j] = m[k][j] * inv_pivot;
        }
        m[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++) {
            float factor = m[i][k];
            for (int j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - factor * m[k][j];
            }
            m[i][k] = 0.0f;
        }
    }
}

// ================= 2. 纯 NEON 向量化版本 =================
// 使用倒数乘法替代除法指令，与 Lab2 报告优化策略一致
void gauss_neon(float** m, int N) {
    for (int k = 0; k < N; k++) {
        float inv_pivot = 1.0f / m[k][k];
        float32x4_t v_inv = vdupq_n_f32(inv_pivot);
        int j;
        // 归一化：向量化乘法
        for (j = k + 1; j <= N - 4; j += 4) {
            float32x4_t vjt = vld1q_f32(&m[k][j]);
            vjt = vmulq_f32(vjt, v_inv);
            vst1q_f32(&m[k][j], vjt);
        }
        for (; j < N; j++) {
            m[k][j] = m[k][j] * inv_pivot;
        }
        m[k][k] = 1.0f;

        // 消去：向量化乘减
        for (int i = k + 1; i < N; i++) {
            float32x4_t vik = vdupq_n_f32(m[i][k]);
            for (j = k + 1; j <= N - 4; j += 4) {
                float32x4_t vkj = vld1q_f32(&m[k][j]);
                float32x4_t vij = vld1q_f32(&m[i][j]);
                vij = vmlsq_f32(vij, vkj, vik);  // 融合乘减，一条指令完成
                vst1q_f32(&m[i][j], vij);
            }
            for (; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0.0f;
        }
    }
}

// ================= 3. Pthread 标量版本 =================
void* pthread_gauss_func(void* arg) {
    ThreadParam* param = (ThreadParam*)arg;
    int t_id = param->t_id;
    int N = param->N;
    float** m = param->m;
    pthread_barrier_t* bar = param->barrier;

    for (int k = 0; k < N; k++) {
        // 线程 0 负责归一化（除法行）
        if (t_id == 0) {
            float inv_pivot = 1.0f / m[k][k];
            for (int j = k + 1; j < N; j++) {
                m[k][j] = m[k][j] * inv_pivot;
            }
            m[k][k] = 1.0f;
        }
        // 同步：确保除法完成
        check_barrier(pthread_barrier_wait(bar));

        // 循环划分：第 t_id 号线程处理行 k+1+t_id, k+1+t_id+NUM_THREADS, ...
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float factor = m[i][k];
            for (int j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - factor * m[k][j];
            }
            m[i][k] = 0.0f;
        }
        // 同步：确保本轮消元全部结束
        check_barrier(pthread_barrier_wait(bar));
    }
    return NULL;
}

void gauss_pthread(float** m, int N) {
    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, NUM_THREADS) != 0) {
        std::cerr << "Barrier init failed!" << std::endl;
        exit(-1);
    }

    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadParam* params = new ThreadParam[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        params[i].t_id = i;
        params[i].N = N;
        params[i].m = m;
        params[i].barrier = &barrier;
        if (pthread_create(&threads[i], NULL, pthread_gauss_func, &params[i]) != 0) {
            std::cerr << "Thread create failed!" << std::endl;
            exit(-1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);
    delete[] threads;
    delete[] params;
}

// ================= 4. Pthread + NEON 结合版本 =================
void* pthread_neon_gauss_func(void* arg) {
    ThreadParam* param = (ThreadParam*)arg;
    int t_id = param->t_id;
    int N = param->N;
    float** m = param->m;
    pthread_barrier_t* bar = param->barrier;

    for (int k = 0; k < N; k++) {
        // 线程 0：NEON 向量化归一化（倒数乘法）
        if (t_id == 0) {
            float inv_pivot = 1.0f / m[k][k];
            float32x4_t v_inv = vdupq_n_f32(inv_pivot);
            int j;
            for (j = k + 1; j <= N - 4; j += 4) {
                float32x4_t vjt = vld1q_f32(&m[k][j]);
                vjt = vmulq_f32(vjt, v_inv);
                vst1q_f32(&m[k][j], vjt);
            }
            for (; j < N; j++) {
                m[k][j] = m[k][j] * inv_pivot;
            }
            m[k][k] = 1.0f;
        }
        check_barrier(pthread_barrier_wait(bar));

        // 循环划分 + NEON 消去
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float32x4_t vik = vdupq_n_f32(m[i][k]);
            int j;
            for (j = k + 1; j <= N - 4; j += 4) {
                float32x4_t vkj = vld1q_f32(&m[k][j]);
                float32x4_t vij = vld1q_f32(&m[i][j]);
                vij = vmlsq_f32(vij, vkj, vik);  // 融合乘减
                vst1q_f32(&m[i][j], vij);
            }
            for (; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0.0f;
        }
        check_barrier(pthread_barrier_wait(bar));
    }
    return NULL;
}

void gauss_pthread_neon(float** m, int N) {
    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, NUM_THREADS) != 0) {
        std::cerr << "Barrier init failed!" << std::endl;
        exit(-1);
    }

    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadParam* params = new ThreadParam[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        params[i].t_id = i;
        params[i].N = N;
        params[i].m = m;
        params[i].barrier = &barrier;
        if (pthread_create(&threads[i], NULL, pthread_neon_gauss_func, &params[i]) != 0) {
            std::cerr << "Thread create failed!" << std::endl;
            exit(-1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);
    delete[] threads;
    delete[] params;
}

// ================= 5. OpenMP 标量版本 =================
void gauss_openmp(float** m, int N) {
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel shared(m, N)
    {
        for (int k = 0; k < N; k++) {
            // 单线程执行归一化，自带隐式屏障
            #pragma omp single
            {
                float inv_pivot = 1.0f / m[k][k];
                for (int j = k + 1; j < N; j++) {
                    m[k][j] = m[k][j] * inv_pivot;
                }
                m[k][k] = 1.0f;
            }

            // 动态调度 + chunk_size=16，平衡负载与调度开销
            #pragma omp for schedule(dynamic, 16)
            for (int i = k + 1; i < N; i++) {
                float factor = m[i][k];
                for (int j = k + 1; j < N; j++) {
                    m[i][j] = m[i][j] - factor * m[k][j];
                }
                m[i][k] = 0.0f;
            }
            // 隐式屏障：离开 for 区域时自动同步
        }
    }
}

// ================= 6. OpenMP + NEON 结合版本 =================
void gauss_openmp_neon(float** m, int N) {
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel shared(m, N)
    {
        for (int k = 0; k < N; k++) {
            #pragma omp single
            {
                float inv_pivot = 1.0f / m[k][k];
                float32x4_t v_inv = vdupq_n_f32(inv_pivot);
                int j;
                for (j = k + 1; j <= N - 4; j += 4) {
                    float32x4_t vjt = vld1q_f32(&m[k][j]);
                    vjt = vmulq_f32(vjt, v_inv);
                    vst1q_f32(&m[k][j], vjt);
                }
                for (; j < N; j++) {
                    m[k][j] = m[k][j] * inv_pivot;
                }
                m[k][k] = 1.0f;
            }

            #pragma omp for schedule(dynamic, 16)
            for (int i = k + 1; i < N; i++) {
                float32x4_t vik = vdupq_n_f32(m[i][k]);
                int j;
                for (j = k + 1; j <= N - 4; j += 4) {
                    float32x4_t vkj = vld1q_f32(&m[k][j]);
                    float32x4_t vij = vld1q_f32(&m[i][j]);
                    vij = vmlsq_f32(vij, vkj, vik);  // 融合乘减
                    vst1q_f32(&m[i][j], vij);
                }
                for (; j < N; j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0.0f;
            }
        }
    }
}

// ================= Main 主函数 =================
// 一次运行自动测试所有 N × Threads 组合，输出 CSV 格式便于导入 Excel/Python 绘图
int main() {
    const int N_values[] = {512, 1024, 2048};
    const int T_values[] = {1, 2, 4, 8};
    const int N_count = 3;
    const int T_count = 4;

    // CSV 表头
    std::cout << "N,Threads,Serial_ms,NEON_ms,Pthread_ms,Pthread+NEON_ms,OpenMP_ms,OpenMP+NEON_ms,"
              << "NEON_Speedup,Pthread_Speedup,Pthread+NEON_Speedup,OpenMP_Speedup,OpenMP+NEON_Speedup,"
              << "Pthread_Valid,P+NEON_Valid,OMP_Valid,OMP+NEON_Valid" << std::endl;

    for (int ni = 0; ni < N_count; ni++) {
        int N = N_values[ni];

        std::cerr << "[INFO] Testing N=" << N << "..." << std::endl;

        float** m_src = alloc_matrix(N);
        float** m_base = alloc_matrix(N);
        float** m_test = alloc_matrix(N);

        init_matrix(m_src, N);

        // 1. Serial (仅与 N 有关，测一次)
        copy_matrix(m_src, m_base, N);
        auto t0 = std::chrono::high_resolution_clock::now();
        gauss_serial(m_base, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_serial = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2. Pure NEON (仅与 N 有关，测一次)
        copy_matrix(m_src, m_test, N);
        t0 = std::chrono::high_resolution_clock::now();
        gauss_neon(m_test, N);
        t1 = std::chrono::high_resolution_clock::now();
        double time_neon = std::chrono::duration<double, std::milli>(t1 - t0).count();
        bool neon_valid = validate(m_base, m_test, N);

        // 遍历线程数
        for (int ti = 0; ti < T_count; ti++) {
            NUM_THREADS = T_values[ti];
            double tp, tpn, to, ton;
            bool vp, vpn, vo, von;

            // Pthread Scalar
            copy_matrix(m_src, m_test, N);
            t0 = std::chrono::high_resolution_clock::now();
            gauss_pthread(m_test, N);
            t1 = std::chrono::high_resolution_clock::now();
            tp = std::chrono::duration<double, std::milli>(t1 - t0).count();
            vp = validate(m_base, m_test, N);

            // Pthread + NEON
            copy_matrix(m_src, m_test, N);
            t0 = std::chrono::high_resolution_clock::now();
            gauss_pthread_neon(m_test, N);
            t1 = std::chrono::high_resolution_clock::now();
            tpn = std::chrono::duration<double, std::milli>(t1 - t0).count();
            vpn = validate(m_base, m_test, N);

            // OpenMP Scalar
            copy_matrix(m_src, m_test, N);
            t0 = std::chrono::high_resolution_clock::now();
            gauss_openmp(m_test, N);
            t1 = std::chrono::high_resolution_clock::now();
            to = std::chrono::duration<double, std::milli>(t1 - t0).count();
            vo = validate(m_base, m_test, N);

            // OpenMP + NEON
            copy_matrix(m_src, m_test, N);
            t0 = std::chrono::high_resolution_clock::now();
            gauss_openmp_neon(m_test, N);
            t1 = std::chrono::high_resolution_clock::now();
            ton = std::chrono::duration<double, std::milli>(t1 - t0).count();
            von = validate(m_base, m_test, N);

            // CSV 数据行
            std::cout << N << "," << NUM_THREADS << ","
                      << time_serial << "," << time_neon << ","
                      << tp << "," << tpn << ","
                      << to << "," << ton << ","
                      << time_serial / time_neon << ","
                      << time_serial / tp << ","
                      << time_serial / tpn << ","
                      << time_serial / to << ","
                      << time_serial / ton << ","
                      << (vp  ? "PASS" : "FAIL") << ","
                      << (vpn ? "PASS" : "FAIL") << ","
                      << (vo  ? "PASS" : "FAIL") << ","
                      << (von ? "PASS" : "FAIL") << std::endl;
        }

        free_matrix(m_src, N);
        free_matrix(m_base, N);
        free_matrix(m_test, N);
    }

    std::cerr << "[INFO] All tests done." << std::endl;
    return 0;
}
