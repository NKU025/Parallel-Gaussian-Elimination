import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ============================================================
# 数据：来自 test.o 的 CSV 输出
# ============================================================
N_values = [512, 1024, 2048]
threads = [1, 2, 4, 8]

data = {
    'Serial':      {512: 48.51, 1024: 446.13, 2048: 4556.36},
    'NEON':        {512: 22.07, 1024: 206.28, 2048: 2272.37},
    'Pthread':     {(512,1):52.17,(512,2):99.75,(512,4):87.77,(512,8):70.68,
                    (1024,1):442.47,(1024,2):958.39,(1024,4):573.84,(1024,8):369.22,
                    (2048,1):3878.15,(2048,2):4303.83,(2048,4):2004.75,(2048,8):1107.23},
    'Pthread+NEON':{(512,1):24.16,(512,2):49.01,(512,4):52.68,(512,8):52.44,
                    (1024,1):208.58,(1024,2):364.92,(1024,4):270.66,(1024,8):191.17,
                    (2048,1):2060.84,(2048,2):1621.99,(2048,4):1013.16,(2048,8):1142.32},
    'OpenMP':      {(512,1):49.95,(512,2):30.85,(512,4):39.45,(512,8):23.57,
                    (1024,1):457.58,(1024,2):522.10,(1024,4):324.55,(1024,8):188.61,
                    (2048,1):3772.42,(2048,2):4610.35,(2048,4):1337.43,(2048,8):888.81},
    'OpenMP+NEON': {(512,1):22.82,(512,2):15.41,(512,4):20.58,(512,8):12.88,
                    (1024,1):210.35,(1024,2):234.11,(1024,4):146.45,(1024,8):86.89,
                    (2048,1):2134.19,(2048,2):1241.49,(2048,4):725.22,(2048,8):472.89},
}

# 预计算加速比
def speedup(algo_name, N, t=None):
    """加速比 = Serial / 算法耗时"""
    serial_t = data['Serial'][N]
    if algo_name == 'Serial':
        return 1.0
    elif algo_name == 'NEON':
        return serial_t / data['NEON'][N]
    else:
        return serial_t / data[algo_name][(N, t)]

# ============================================================
# 全局样式
# ============================================================
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'Serial':      '#555555',
    'NEON':        '#2ca02c',
    'Pthread':     '#d62728',
    'Pthread+NEON':'#ff7f0e',
    'OpenMP':      '#1f77b4',
    'OpenMP+NEON': '#9467bd',
}
MARKERS = {
    'Serial':      's',
    'NEON':        'D',
    'Pthread':     'o',
    'Pthread+NEON':'^',
    'OpenMP':      'v',
    'OpenMP+NEON': 'p',
}
LINESTYLE = {
    'Serial':      '--',
    'NEON':        '--',
    'Pthread':     '-',
    'Pthread+NEON':'-.',
    'OpenMP':      '-',
    'OpenMP+NEON':'-.',
}

outdir = 'D:/Workspace/并行程序设计作业/figures'
os.makedirs(outdir, exist_ok=True)

# ============================================================
# 图 1-3: 执行时间 vs 线程数（每个 N 一张）
# ============================================================
for N in N_values:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # 画水平参考线：Serial 和 NEON（不依赖线程数）
    ax.axhline(y=data['Serial'][N], color=COLORS['Serial'],
               linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(y=data['NEON'][N], color=COLORS['NEON'],
               linestyle='--', linewidth=1.2, alpha=0.7)

    # 画四条并行算法的折线
    for algo in ['Pthread', 'Pthread+NEON', 'OpenMP', 'OpenMP+NEON']:
        times = [data[algo][(N, t)] for t in threads]
        ax.plot(threads, times, color=COLORS[algo], marker=MARKERS[algo],
                linewidth=2, markersize=9, linestyle=LINESTYLE[algo], label=algo)

    # 在 Serial 和 NEON 线末端加标签
    x_right = threads[-1] + 0.3
    ax.text(x_right, data['Serial'][N], 'Serial', color=COLORS['Serial'],
            va='center', fontsize=10, fontweight='bold')
    ax.text(x_right, data['NEON'][N], 'NEON', color=COLORS['NEON'],
            va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('线程数', fontsize=13)
    ax.set_ylabel('执行时间 (ms)', fontsize=13)
    ax.set_title(f'不同算法的执行时间对比 (N={N})', fontsize=15, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(loc='best', fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 8.8)

    fig.tight_layout()
    fig.savefig(f'{outdir}/time_N{N}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: time_N{N}.png')

# ============================================================
# 图 4-6: 加速比 vs 线程数（每个 N 一张）
# ============================================================
for N in N_values:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # 水平线：speedup=1.0（串行基准）
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    # NEON 加速比（水平线）
    neon_su = speedup('NEON', N)
    ax.axhline(y=neon_su, color=COLORS['NEON'], linestyle='--',
               linewidth=1.2, alpha=0.7)
    ax.text(threads[-1] + 0.3, neon_su, f'NEON ({neon_su:.2f}x)',
            color=COLORS['NEON'], va='center', fontsize=10, fontweight='bold')

    for algo in ['Pthread', 'Pthread+NEON', 'OpenMP', 'OpenMP+NEON']:
        su = [speedup(algo, N, t) for t in threads]
        ax.plot(threads, su, color=COLORS[algo], marker=MARKERS[algo],
                linewidth=2, markersize=9, linestyle=LINESTYLE[algo], label=algo)

    ax.set_xlabel('线程数', fontsize=13)
    ax.set_ylabel('加速比 (vs Serial)', fontsize=13)
    ax.set_title(f'加速比随线程数变化 (N={N})', fontsize=15, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(loc='best', fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 8.8)

    # 标注最佳加速比
    best_algo = max(['Pthread', 'Pthread+NEON', 'OpenMP', 'OpenMP+NEON'],
                    key=lambda a: max(speedup(a, N, t) for t in threads))
    best_t = max(threads, key=lambda t: speedup(best_algo, N, t))
    best_su = speedup(best_algo, N, best_t)
    ax.annotate(f'Best: {best_algo}+NEON\n{best_su:.2f}x @ {best_t}线程',
                xy=(best_t, best_su), xytext=(best_t + 1.5, best_su - 0.8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    fig.tight_layout()
    fig.savefig(f'{outdir}/speedup_N{N}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: speedup_N{N}.png')

# ============================================================
# 图 7: 综合对比 — 最佳加速比 vs 线程数
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for algo in ['Pthread', 'Pthread+NEON', 'OpenMP', 'OpenMP+NEON']:
    for N in N_values:
        su_values = [speedup(algo, N, t) for t in threads]
        ax.plot(threads, su_values, color=COLORS[algo], marker=MARKERS[algo],
                linewidth=2, markersize=10, linestyle=LINESTYLE[algo],
                label=f'{algo} (N={N})', alpha=0.85)

ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('线程数', fontsize=13)
ax.set_ylabel('加速比 (vs Serial)', fontsize=13)
ax.set_title('所有测试组合加速比总览', fontsize=15, fontweight='bold')
ax.set_xticks(threads)
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{outdir}/speedup_overview.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: speedup_overview.png')

# ============================================================
# 图 8: Pthread vs OpenMP 最佳加速比柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(N_values))
width = 0.2

for i, (algo, offset) in enumerate(zip(
    ['Pthread', 'Pthread+NEON', 'OpenMP', 'OpenMP+NEON'],
    [-0.3, -0.1, 0.1, 0.3])):
    best_su = [max(speedup(algo, N, t) for t in threads) for N in N_values]
    bars = ax.bar(x + offset, best_su, width, color=COLORS[algo], label=algo, edgecolor='white')
    for bar, val in zip(bars, best_su):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('矩阵规模 (N)', fontsize=13)
ax.set_ylabel('最佳加速比', fontsize=13)
ax.set_title('各算法在不同规模下的最佳加速比', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{n}' for n in N_values])
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(f'{outdir}/best_speedup_bar.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: best_speedup_bar.png')

# ============================================================
# 图 9: 扩展性分析 — 加速比/线程数（效率）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for algo in ['Pthread', 'Pthread+NEON', 'OpenMP', 'OpenMP+NEON']:
    for N in N_values:
        eff = [speedup(algo, N, t) / t for t in threads]
        ax.plot(threads, eff, color=COLORS[algo], marker=MARKERS[algo],
                linewidth=2, markersize=9, linestyle=LINESTYLE[algo],
                label=f'{algo} (N={N})', alpha=0.85)

ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='理想效率=1.0')
ax.set_xlabel('线程数', fontsize=13)
ax.set_ylabel('并行效率 (加速比/线程数)', fontsize=13)
ax.set_title('并行扩展效率分析', fontsize=15, fontweight='bold')
ax.set_xticks(threads)
ax.legend(loc='best', fontsize=8, ncol=2, framealpha=0.8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{outdir}/efficiency.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: efficiency.png')

print(f'\nDone! All figures saved to: {outdir}')
