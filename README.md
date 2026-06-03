# Parallel Gaussian Elimination

南开大学 并行程序设计 Lab3：Pthread/OpenMP 多线程加速的高斯消去算法。

## 实验平台

- 华为鲲鹏 920 (ARM AArch64, 8核)
- openEuler OS, GCC -O2
- PBS 作业调度系统

## 算法版本

| # | 算法 | 说明 |
|---|------|------|
| 1 | Serial | 串行基线 |
| 2 | Pure NEON | SIMD 向量化（2.0x 加速） |
| 3 | Pthread | Barrier + 循环划分 |
| 4 | Pthread+NEON | Pthread + NEON 结合 |
| 5 | OpenMP | schedule(dynamic) 动态调度 |
| 6 | OpenMP+NEON | OpenMP + NEON（**最佳 9.64x**） |

## 关键结果

- N=2048, 8线程: OpenMP+NEON 达到 **9.64x** 加速比
- OpenMP 整体优于 Pthread（线程池 vs 手动 barrier）
- 小规模（N=512）: 多线程可能负优化，纯 NEON 最合适

## 目录

- `main.cc` — 完整实验代码（6个算法版本）
- `report/main.pdf` — 实验报告（14页）
- `report/main.tex` — LaTeX 源文件
- `plot_charts.py` — 性能图表生成脚本
- `test_output.txt` — 原始测试输出
