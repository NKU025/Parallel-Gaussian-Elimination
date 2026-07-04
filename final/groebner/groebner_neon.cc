/**
 * Gröbner 基高斯消去 — NEON SIMD 加速版本
 * 使用 NEON 128位向量一次处理 4 个 uint32_t（即 128 列）
 * 编译: g++ -O2 groebner_neon.cc -o groebner_neon
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <sstream>
#include <arm_neon.h>
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
        ele_sparse[header] = r;
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
        bits[col / mat_L] |= ((mat_t)1 << (col % mat_L));
}

inline bool is_one(const vector<mat_t>& bits, int col) {
    return bits[col / mat_L] & ((mat_t)1 << (col % mat_L));
}

// NEON 向量化 XOR 消去（一次处理 4 个 uint32_t = 128 列）
void xor_neon(mat_t* row, mat_t* ele, int num_words) {
    int w = 0;
    for (; w <= num_words - 4; w += 4) {
        uint32x4_t vr = vld1q_u32(&row[w]);
        uint32x4_t ve = vld1q_u32(&ele[w]);
        vst1q_u32(&row[w], veorq_u32(vr, ve));
    }
    for (; w < num_words; w++)
        row[w] ^= ele[w];
}

void groebner_neon(vector<vector<mat_t>>& ele_sparse,
                   vector<vector<mat_t>>& row_sparse,
                   int cols, int num_words) {
    // 位向量化
    vector<vector<mat_t>> ele(cols, vector<mat_t>(num_words, 0));
    for (int j = 0; j < cols; j++)
        if (!ele_sparse[j].empty())
            sparse_to_bits(ele_sparse[j], ele[j], num_words);

    int num_rows = row_sparse.size();
    vector<vector<mat_t>> row(num_rows, vector<mat_t>(num_words, 0));
    vector<bool> upgraded(num_rows, false);
    for (int i = 0; i < num_rows; i++)
        sparse_to_bits(row_sparse[i], row[i], num_words);

    for (int j = cols - 1; j >= 0; j--) {
        if (is_one(ele[j], j)) {
            for (int i = 0; i < num_rows; i++) {
                if (upgraded[i]) continue;
                if (is_one(row[i], j))
                    xor_neon(row[i].data(), ele[j].data(), num_words);  // NEON!
            }
        } else {
            for (int i = 0; i < num_rows; i++) {
                if (upgraded[i]) continue;
                if (is_one(row[i], j)) {
                    for (int w = 0; w < num_words; w++)
                        ele[j][w] = row[i][w];
                    upgraded[i] = true;
                    j++;
                    break;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <data_dir> <cols> <num_ele> <num_row>" << endl;
        return 1;
    }
    string dir = argv[1];
    int cols = atoi(argv[2]), num_ele = atoi(argv[3]), num_row = atoi(argv[4]);
    int num_words = (cols + mat_L - 1) / mat_L + 1;

    vector<vector<mat_t>> ele_sparse, row_sparse;
    load_data(dir, num_ele, num_row, cols, ele_sparse, row_sparse);

    auto t0 = chrono::high_resolution_clock::now();
    groebner_neon(ele_sparse, row_sparse, cols, num_words);
    auto t1 = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, std::milli>(t1 - t0).count();
    cout << "Groebner NEON:  " << ms << " ms (cols=" << cols << ")" << endl;
    return 0;
}
