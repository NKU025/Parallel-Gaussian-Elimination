/**
 * Gröbner 基高斯消去 — 串行基线版本
 * GF(2) 上的特殊高斯消去：0/1 矩阵，运算为 XOR（^=）
 * 输入格式：稀疏向量（整数表示 1 元素所在列号，从高到低排列）
 * 编译: g++ -O2 groebner_serial.cc -o groebner_serial
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <sstream>
using namespace std;

typedef unsigned int mat_t;
const int mat_L = 32;  // sizeof(mat_t) * 8 = 32 bits

// 读入一行稀疏格式的数据（如 "9 3 1"），转换为稠密位向量
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

// 从目录读取数据集，返回消元子和被消元行
void load_data(const string& dir, int num_ele, int num_row, int cols,
               vector<vector<mat_t>>& ele_sparse, vector<vector<mat_t>>& row_sparse) {
    ele_sparse.resize(cols);
    // 读消元子
    for (int i = 1; i <= num_ele; i++) {
        ifstream f(dir + "/" + to_string(i) + ".txt");
        if (!f.is_open()) continue;
        string line;
        getline(f, line);
        int header;
        vector<mat_t> r;
        parse_line(line, r, header);
        ele_sparse[header] = r;  // 首项列号决定放在哪一行
    }
    // 读被消元行
    row_sparse.resize(num_row);
    for (int i = 1; i <= num_row; i++) {
        ifstream f(dir + "/" + to_string(i) + ".txt");
        string line;
        getline(f, line);
        int header;
        parse_line(line, row_sparse[i-1], header);
    }
}

// 将稀疏行转换为位向量（每 32 列一个 uint32_t）
void sparse_to_bits(const vector<mat_t>& sparse, vector<mat_t>& bits, int num_words) {
    bits.assign(num_words, 0);
    for (mat_t col : sparse) {
        bits[col / mat_L] |= ((mat_t)1 << (col % mat_L));
    }
}

// 检查位向量中某列是否为 1
inline bool is_one(const vector<mat_t>& bits, int col) {
    return bits[col / mat_L] & ((mat_t)1 << (col % mat_L));
}

// Gröbner 基串行算法
void groebner_serial(const vector<vector<mat_t>>& ele_sparse,
                     const vector<vector<mat_t>>& row_sparse,
                     int cols, int num_words) {
    // 转换为位向量
    vector<vector<mat_t>> ele(cols, vector<mat_t>(num_words, 0));
    for (int j = 0; j < cols; j++) {
        if (!ele_sparse[j].empty())
            sparse_to_bits(ele_sparse[j], ele[j], num_words);
    }

    int num_rows = row_sparse.size();
    vector<vector<mat_t>> row(num_rows, vector<mat_t>(num_words, 0));
    vector<bool> upgraded(num_rows, false);
    for (int i = 0; i < num_rows; i++)
        sparse_to_bits(row_sparse[i], row[i], num_words);

    // 主循环：从高列到低列遍历消元子位置
    for (int j = cols - 1; j >= 0; j--) {
        if (is_one(ele[j], j)) {
            // 存在消元子，对所有被消元行消去
            for (int i = 0; i < num_rows; i++) {
                if (upgraded[i]) continue;
                if (is_one(row[i], j)) {
                    // XOR with eliminator j
                    for (int w = 0; w < num_words; w++)
                        row[i][w] ^= ele[j][w];
                }
            }
        } else {
            // 不存在消元子，找第一个 j 列为 1 的被消元行升格
            for (int i = 0; i < num_rows; i++) {
                if (upgraded[i]) continue;
                if (is_one(row[i], j)) {
                    // 升格
                    for (int w = 0; w < num_words; w++)
                        ele[j][w] = row[i][w];
                    upgraded[i] = true;
                    j++;  // 重新处理当前列
                    break;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <data_dir> [cols] [num_ele] [num_row]" << endl;
        return 1;
    }

    string dir = argv[1];
    int cols = argc > 2 ? atoi(argv[2]) : 130;
    int num_ele = argc > 3 ? atoi(argv[3]) : 10;
    int num_row = argc > 4 ? atoi(argv[4]) : 6;
    int num_words = (cols + mat_L - 1) / mat_L + 1;

    // 加载数据
    vector<vector<mat_t>> ele_sparse, row_sparse;
    load_data(dir, num_ele, num_row, cols, ele_sparse, row_sparse);

    auto t0 = chrono::high_resolution_clock::now();
    groebner_serial(ele_sparse, row_sparse, cols, num_words);
    auto t1 = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, std::milli>(t1 - t0).count();
    cout << "Groebner Serial: " << ms << " ms (cols=" << cols
         << ", ele=" << num_ele << ", rows=" << num_row << ")" << endl;

    return 0;
}
