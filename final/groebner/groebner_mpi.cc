/**
 * Gröbner 基高斯消去 — MPI 并行版本（修复版）
 * 编译: mpicxx -O2 groebner_mpi.cc -o groebner_mpi
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <sstream>
#include <mpi.h>
using namespace std;

typedef unsigned int mat_t;
const int mat_L = 32;

void parse_line(const string& line, vector<mat_t>& row, int& header) {
    istringstream iss(line);
    row.clear();
    int val;
    bool first = true;
    while (iss >> val) {
        if (first) { header = val; first = false; }
        row.push_back(val);
    }
}

void load_data(const string& dir, int num_ele, int num_row, int cols,
               vector<vector<mat_t>>& ele_sparse, vector<vector<mat_t>>& row_sparse) {
    ele_sparse.resize(cols);
    for (int i = 1; i <= num_ele; i++) {
        ifstream f(dir + "/" + to_string(i) + ".txt");
        if (!f.is_open()) continue;
        string line; getline(f, line);
        int header; vector<mat_t> r;
        parse_line(line, r, header);
        if (header < cols) ele_sparse[header] = r;
    }
    row_sparse.resize(num_row);
    for (int i = 1; i <= num_row; i++) {
        ifstream f(dir + "/" + to_string(i) + ".txt");
        string line; getline(f, line);
        int header;
        parse_line(line, row_sparse[i-1], header);
    }
}

void sparse_to_bits(const vector<mat_t>& sparse, vector<mat_t>& bits, int num_words) {
    bits.assign(num_words, 0);
    for (mat_t col : sparse)
        if ((int)(col / mat_L) < num_words)
            bits[col / mat_L] |= ((mat_t)1 << (col % mat_L));
}

inline bool is_one(const vector<mat_t>& bits, int col) {
    return bits[col / mat_L] & ((mat_t)1 << (col % mat_L));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string dir = "/home/data/Groebner";
    int cols = 200, num_ele = 100, num_row = 200;
    if (argc > 1) dir = argv[1];
    if (argc > 2) cols = atoi(argv[2]);
    if (argc > 3) num_ele = atoi(argv[3]);
    if (argc > 4) num_row = atoi(argv[4]);

    int num_words = (cols + mat_L - 1) / mat_L + 1;

    // 0 号进程加载，然后广播消元子
    vector<vector<mat_t>> ele_sparse, row_sparse;
    if (rank == 0) load_data(dir, num_ele, num_row, cols, ele_sparse, row_sparse);

    // 构建 + 广播消元子（全部广播，简单粗暴但可靠）
    vector<mat_t> ele_flat(cols * num_words, 0);
    if (rank == 0) {
        for (int j = 0; j < cols; j++)
            if (!ele_sparse[j].empty()) {
                vector<mat_t> bits;
                sparse_to_bits(ele_sparse[j], bits, num_words);
                memcpy(&ele_flat[j * num_words], bits.data(), num_words * sizeof(mat_t));
            }
    }
    MPI_Bcast(ele_flat.data(), cols * num_words, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // 行划分
    int base = num_row / size, rem = num_row % size;
    int start = (rank < rem) ? rank * (base + 1) : rank * base + rem;
    int local_rows = (rank < rem) ? (base + 1) : base;

    // 0 号构建被消元行 + 分发
    vector<mat_t> local_flat(local_rows * num_words, 0);
    if (rank == 0) {
        vector<mat_t> all_rows(num_row * num_words, 0);
        for (int i = 0; i < num_row; i++)
            if (!row_sparse[i].empty()) {
                vector<mat_t> bits;
                sparse_to_bits(row_sparse[i], bits, num_words);
                memcpy(&all_rows[i * num_words], bits.data(), num_words * sizeof(mat_t));
            }
        // 自己的
        memcpy(local_flat.data(), &all_rows[start * num_words], local_rows * num_words * sizeof(mat_t));
        // 发其他人的
        for (int p = 1; p < size; p++) {
            int p_start = (p < rem) ? p * (base + 1) : p * base + rem;
            int p_rows  = (p < rem) ? (base + 1) : base;
            MPI_Send(&all_rows[p_start * num_words], p_rows * num_words, MPI_UNSIGNED, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_flat.data(), local_rows * num_words, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 本地 upgraded 标记
    vector<char> local_upgraded(local_rows, 0);

    double t0 = MPI_Wtime();

    // ===== 主循环 =====
    for (int j = cols - 1; j >= 0; j--) {
        int wj = j / mat_L, bj = j % mat_L;
        bool has_elim = (ele_flat[j * num_words + wj] >> bj) & 1;

        if (has_elim) {
            for (int i = 0; i < local_rows; i++) {
                if (local_upgraded[i]) continue;
                if (local_flat[i * num_words + wj] & ((mat_t)1 << bj)) {
                    for (int w = 0; w < num_words; w++)
                        local_flat[i * num_words + w] ^= ele_flat[j * num_words + w];
                }
            }
        } else {
            // 升格：各进程找候选
            int local_cand = num_row;
            for (int i = 0; i < local_rows; i++) {
                if (local_upgraded[i]) continue;
                if (local_flat[i * num_words + j / mat_L] & ((mat_t)1 << (j % mat_L))) {
                    local_cand = start + i;
                    break;
                }
            }
            int global_min;
            MPI_Allreduce(&local_cand, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            if (global_min < num_row) {
                int owner;
                if (global_min < rem * (base + 1))
                    owner = global_min / (base + 1);
                else
                    owner = (global_min - rem * (base + 1)) / base + rem;

                if (rank == owner) {
                    int li = global_min - start;
                    memcpy(&ele_flat[j * num_words], &local_flat[li * num_words], num_words * sizeof(mat_t));
                    local_upgraded[li] = 1;
                }
                MPI_Bcast(&ele_flat[j * num_words], num_words, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
                j++;  // 重新处理
            }
        }
    }

    double elapsed = MPI_Wtime() - t0;

    if (rank == 0)
        cout << "Groebner MPI: " << (elapsed * 1000.0) << " ms (cols=" << cols << " P=" << size << ")" << endl;

    MPI_Finalize();
    return 0;
}
