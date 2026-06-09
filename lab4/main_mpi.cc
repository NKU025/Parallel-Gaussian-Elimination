#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <arm_neon.h>

// ============== 内存分配 ==============

float* alloc_vec(int n) {
    // 用 calloc 替代 posix_memalign，避免对齐问题
    float* p = (float*)calloc(n, sizeof(float));
    if (!p) { std::cerr << "malloc failed" << std::endl; exit(-1); }
    return p;
}

// ============== 初始化（上三角 + 行组合） ==============

void init_matrix(float* a, int N) {
    srand(1314);
    for (int i = 0; i < N * N; i++) a[i] = 0.0f;
    for (int i = 0; i < N; i++) {
        a[i * N + i] = 1.0f;
        for (int j = i + 1; j < N; j++)
            a[i * N + j] = (float)(rand() % 100);
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                a[i * N + j] += a[k * N + j];
}

// ============== 串行高斯消去（基线） ==============

double gauss_serial(float* a, int N) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < N; k++) {
        float inv = 1.0f / a[k * N + k];
        for (int j = k + 1; j < N; j++)
            a[k * N + j] *= inv;
        a[k * N + k] = 1.0f;

        for (int i = k + 1; i < N; i++) {
            float factor = a[i * N + k];
            for (int j = k + 1; j < N; j++)
                a[i * N + j] -= factor * a[k * N + j];
            a[i * N + k] = 0.0f;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============== 结果验证 ==============

bool validate(float* a, float* b, int N) {
    for (int i = 0; i < N * N; i++) {
        if (std::abs(a[i] - b[i]) > 1e-2) return false;
    }
    return true;
}

// ============== MPI 高斯消去（一维行块划分） ==============

double gauss_mpi(float* local_a, int N, int rank, int size) {
    // 计算行范围
    int base = N / size;
    int rem  = N % size;
    int start = (rank < rem) ? rank * (base + 1) : rank * base + rem;
    int lrows = (rank < rem) ? (base + 1) : base;
    // local_a 大小为 lrows * N

    float* krow = alloc_vec(N);
    double t_start = MPI_Wtime();

    for (int k = 0; k < N; k++) {
        // 确定第 k 行的所有者
        int owner;
        if (k < rem * (base + 1))
            owner = k / (base + 1);
        else
            owner = (k - rem * (base + 1)) / base + rem;

        if (rank == owner) {
            int lk = k - start;                    // 本地行索引
            float inv = 1.0f / local_a[lk * N + k];
            for (int j = k + 1; j < N; j++)
                local_a[lk * N + j] *= inv;
            local_a[lk * N + k] = 1.0f;
            memcpy(krow, &local_a[lk * N], N * sizeof(float));
        }

        MPI_Bcast(krow, N, MPI_FLOAT, owner, MPI_COMM_WORLD);

        // 消去本地所有 >k 的行
        for (int li = 0; li < lrows; li++) {
            int gi = start + li;
            if (gi <= k) continue;
            float factor = local_a[li * N + k];
            for (int j = k + 1; j < N; j++)
                local_a[li * N + j] -= factor * krow[j];
            local_a[li * N + k] = 0.0f;
        }
    }

    double elapsed = MPI_Wtime() - t_start;
    free(krow);
    return elapsed;
}

// ============== 单次测试 ==============

void run_one_test(int N, int rank, int size, double* out_serial, double* out_mpi, bool* out_valid) {
    int base = N / size;
    int rem  = N % size;
    int start = (rank < rem) ? rank * (base + 1) : rank * base + rem;
    int lrows = (rank < rem) ? (base + 1) : base;

    float* local_a = alloc_vec(lrows * N);
    double serial_time = 0;
    float* serial_result = NULL;
    bool valid = false;

    if (rank == 0) {
        float* full = alloc_vec(N * N);
        init_matrix(full, N);
        serial_result = alloc_vec(N * N);
        memcpy(serial_result, full, N * N * sizeof(float));
        serial_time = gauss_serial(serial_result, N);

        for (int p = 1; p < size; p++) {
            int ps = (p < rem) ? p * (base + 1) : p * base + rem;
            int pr = (p < rem) ? (base + 1) : base;
            MPI_Send(&full[ps * N], pr * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
        memcpy(local_a, full, lrows * N * sizeof(float));
        free(full);
    } else {
        MPI_Recv(local_a, lrows * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double mpi_time = gauss_mpi(local_a, N, rank, size);

    if (rank == 0) {
        float* full_result = alloc_vec(N * N);
        memcpy(full_result, local_a, lrows * N * sizeof(float));
        for (int p = 1; p < size; p++) {
            int ps = (p < rem) ? p * (base + 1) : p * base + rem;
            int pr = (p < rem) ? (base + 1) : base;
            MPI_Recv(&full_result[ps * N], pr * N, MPI_FLOAT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        valid = validate(serial_result, full_result, N);
        free(full_result);
    } else {
        MPI_Send(local_a, lrows * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        *out_serial = serial_time;
        *out_mpi = mpi_time * 1000.0;
        *out_valid = valid;
        free(serial_result);
    }
    free(local_a);
}

// ============== Main ==============

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N_vals[] = {512, 1024, 2048};
    const int n_count = 3;

    if (rank == 0) {
        std::cout << "N,Processes,Serial_ms,MPI_ms,Speedup,Valid" << std::endl;
    }

    for (int ni = 0; ni < n_count; ni++) {
        int N = N_vals[ni];
        double s_time, m_time;
        bool ok;

        run_one_test(N, rank, size, &s_time, &m_time, &ok);

        if (rank == 0) {
            std::cout << N << "," << size << ","
                      << s_time << "," << m_time << ","
                      << s_time / m_time << ","
                      << (ok ? "PASS" : "FAIL") << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
