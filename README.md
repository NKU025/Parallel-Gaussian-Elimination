# Parallel Gaussian Elimination

南开大学 并行程序设计 — 高斯消去算法并行化实验。

## 实验平台

- 华为鲲鹏 920 (ARM AArch64, 8核/节点)
- openEuler OS, GCC -O2 / mpicxx
- PBS 作业调度系统

---

## Lab2: SIMD NEON 向量化

### 关键结果

| N | 串行 (ms) | NEON (ms) | 加速比 |
|---|----------|-----------|--------|
| 512 | 47.03 | 22.41 | 2.10x |
| 1024 | 426.94 | 207.86 | 2.05x |
| 2048 | 3575.58 | 1730.09 | 2.07x |

- NEON 向量化提供稳定 ~2.0x 加速比
- 使用 `vmlsq_f32` 融合乘减指令，除法转乘法优化

## Lab3: Pthread/OpenMP 多线程

### 算法版本

| # | 算法 | 说明 |
|---|------|------|
| 1 | Serial | 串行基线 |
| 2 | Pure NEON | SIMD 向量化（2.0x 加速） |
| 3 | Pthread | Barrier + 循环划分 |
| 4 | Pthread+NEON | Pthread + NEON 结合 |
| 5 | OpenMP | schedule(dynamic) 动态调度 |
| 6 | OpenMP+NEON | OpenMP + NEON（**最佳 9.64x**） |

### 关键结果

- N=2048, 8线程: OpenMP+NEON 达到 **9.64x** 加速比
- OpenMP 整体优于 Pthread（线程池 vs 手动 barrier）
- 小规模（N=512）: 多线程可能负优化，纯 NEON 最合适

---

## Lab4: MPI 多进程

### 算法设计

一维行块划分 + 逐行广播。数据初始化时分发一次，之后每轮仅广播一行（4KB），通信复杂度 $O(N^2)$。

### 关键结果

| N | 2 进程 | 4 进程 |
|---|--------|--------|
| 512 | 1.28x | 1.80x |
| 1024 | 1.50x | 2.27x |
| 2048 | 1.32x | **2.43x** |

- 4 进程 N=2048 最高 **2.43x** 加速比
- 进程数 2→4 加速比几乎翻倍，扩展性良好
- MPI 跨节点通信是主要性能瓶颈

---

## 目录

```
├── lab2/
│   ├── src/main.cc           # Lab2 SIMD NEON 代码
│   ├── scripts/test.sh       # 测试脚本
│   └── report/               # 实验报告
├── lab3/
│   ├── main.cc              # Lab3 完整代码（6算法版本）
│   ├── plot_charts.py       # 图表生成脚本
│   ├── test_output.txt      # 原始测试输出
│   ├── report/main.pdf      # Lab3 实验报告（14页）
│   ├── report/main.tex      # LaTeX 源文件
│   └── figures/             # 9 张性能图表
├── lab4/
│   ├── main_mpi.cc          # Lab4 MPI 代码
│   ├── mpi.sh               # PBS 提交脚本
│   ├── report/main.pdf      # Lab4 实验报告（13页）
│   ├── report/main.tex      # LaTeX 源文件
│   └── figures/             # 3 张性能图表
├── plot_charts.py           # Lab3 图表生成脚本
└── test_output.txt          # Lab3 原始测试输出
```
