"""
FSNet-v3 可视化: 8图全面对比分析
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

plt.rcParams.update({
    'font.size': 11, 'figure.dpi': 150,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.facecolor': 'white'
})

# 加载结果
with open('results/comprehensive/comprehensive_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

import os
os.makedirs('figures/v3', exist_ok=True)

# 数据准备
order = ['OGD', 'ER', 'FSNet', 'FSNet-Fixed', 'NoMem', 'FSNet-Advanced', 'FSNet-Ultra', 'FSNet-v3']
models, mapes, mses, maes = [], [], [], []
for name in order:
    if name in results:
        models.append(name)
        mapes.append(results[name]['mape'])
        mses.append(results[name]['mse'])
        maes.append(results[name]['mae'])

# ─── 图1: MAPE全排名 (8模型) ───
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#d32f2f' if n == 'FSNet-v3' else '#1976d2' if n == 'FSNet-Ultra' 
          else '#388e3c' if n == 'FSNet-Advanced' else '#78909c' for n in models]
bars = ax.bar(range(len(models)), mapes, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
for i, (bar, v) in enumerate(zip(bars, mapes)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{v:.2f}%', ha='center', va='bottom', fontweight='bold' if models[i]=='FSNet-v3' else 'normal',
            fontsize=11, color='#d32f2f' if models[i]=='FSNet-v3' else 'black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=25, ha='right')
ax.set_ylabel('MAPE (%)')
ax.set_title('Full Model Ranking — MAPE (8 Models on ETTh1)')
ax.annotate('NEW SOTA', xy=(len(models)-1, mapes[-1]), xytext=(len(models)-2.5, mapes[-1]+2),
            arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=2),
            fontsize=14, fontweight='bold', color='#d32f2f')
plt.tight_layout()
plt.savefig('figures/v3/1_v3_mape_ranking.png', bbox_inches='tight')
plt.close()
print("✅ 1_v3_mape_ranking.png")

# ─── 图2: v3 vs Ultra vs Advanced 三代对比 ───
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
labels = ['Advanced', 'Ultra', 'v3']
vals_mape = [results['FSNet-Advanced']['mape'], results['FSNet-Ultra']['mape'], results['FSNet-v3']['mape']]
vals_mse = [results['FSNet-Advanced']['mse'], results['FSNet-Ultra']['mse'], results['FSNet-v3']['mse']]
vals_mae = [results['FSNet-Advanced']['mae'], results['FSNet-Ultra']['mae'], results['FSNet-v3']['mae']]
colors3 = ['#388e3c', '#1976d2', '#d32f2f']

for ax, vals, title, fmt in zip(axes, [vals_mape, vals_mse, vals_mae],
                                  ['MAPE (%)', 'MSE', 'MAE'],
                                  ['{:.2f}%', '{:.6f}', '{:.6f}']):
    bars = ax.bar(labels, vals, color=colors3, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                fmt.format(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)

fig.suptitle('Three Generations Head-to-Head', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/v3/2_v3_three_generations.png', bbox_inches='tight')
plt.close()
print("✅ 2_v3_three_generations.png")

# ─── 图3: v3 相对各方法改进 ───
fig, ax = plt.subplots(figsize=(10, 6))
v3_mape = results['FSNet-v3']['mape']
baselines = ['OGD', 'ER', 'NoMem', 'FSNet', 'FSNet-Fixed', 'FSNet-Advanced', 'FSNet-Ultra']
improvements = []
for name in baselines:
    if name in results:
        imp = (v3_mape - results[name]['mape']) / results[name]['mape'] * 100
        improvements.append(imp)
    else:
        improvements.append(0)
colors_imp = ['#2196f3' if v < 0 else '#f44336' for v in improvements]
bars = ax.barh(baselines, improvements, color=colors_imp, height=0.5)
for bar, v in zip(bars, improvements):
    xpos = bar.get_width() - 4 if v < -10 else bar.get_width() + 0.5
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('MAPE Improvement (%)')
ax.set_title('FSNet-v3 Improvement vs All Baselines')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/v3/3_v3_improvement.png', bbox_inches='tight')
plt.close()
print("✅ 3_v3_improvement.png")

# ─── 图4: MSE+MAE 双面板 (8模型) ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = range(len(models))
colors8 = ['#d32f2f' if n == 'FSNet-v3' else '#1976d2' if n == 'FSNet-Ultra' 
           else '#388e3c' if n == 'FSNet-Advanced' else '#9e9e9e' for n in models]

ax1.bar(x, mses, color=colors8, width=0.6)
for i, v in enumerate(mses):
    ax1.text(i, v*1.02, f'{v:.4f}', ha='center', fontsize=8, rotation=30)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax1.set_title('MSE Comparison', fontweight='bold')
ax1.set_ylabel('MSE')

ax2.bar(x, maes, color=colors8, width=0.6)
for i, v in enumerate(maes):
    ax2.text(i, v*1.02, f'{v:.4f}', ha='center', fontsize=8, rotation=30)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax2.set_title('MAE Comparison', fontweight='bold')
ax2.set_ylabel('MAE')

plt.tight_layout()
plt.savefig('figures/v3/4_v3_mse_mae.png', bbox_inches='tight')
plt.close()
print("✅ 4_v3_mse_mae.png")

# ─── 图5: 性能演进曲线 (4阶段) ───
fig, ax = plt.subplots(figsize=(10, 6))
evolution = [
    ('OGD\n(Baseline)', 14.75),
    ('FSNet\n(Bug)', 10.07),
    ('FSNet-Fixed\n(BugFix)', 10.25),
    ('FSNet-Advanced\n(v1: 4 innovations)', 5.06),
    ('FSNet-Ultra\n(v2: 6 innovations)', 4.81),
    ('FSNet-v3\n(v3: Semantic-Aligned)', 4.21),
]
names, vals = zip(*evolution)
x = range(len(vals))
ax.plot(x, vals, 'o-', color='#1976d2', markersize=10, linewidth=2.5, markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#1976d2')
ax.plot(len(vals)-1, vals[-1], 'o', color='#d32f2f', markersize=14, markeredgewidth=3,
        markerfacecolor='#d32f2f', zorder=5)
for i, (n, v) in enumerate(zip(names, vals)):
    offset = 0.6 if i < 3 else -0.6
    va = 'bottom' if i < 3 else 'top'
    ax.annotate(f'{v:.2f}%', (i, v), textcoords="offset points", xytext=(0, 12 if i < 3 else -15),
                ha='center', fontsize=11, fontweight='bold' if i >= 4 else 'normal',
                color='#d32f2f' if i == len(vals)-1 else 'black')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('MAPE (%)')
ax.set_title('Performance Evolution: From OGD to FSNet-v3', fontweight='bold')
ax.fill_between(x, vals, alpha=0.1, color='#1976d2')
plt.tight_layout()
plt.savefig('figures/v3/5_v3_evolution.png', bbox_inches='tight')
plt.close()
print("✅ 5_v3_evolution.png")

# ─── 图6: 雷达图 (4模型) ───
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ['1/MAPE', '1/MSE', '1/MAE', '1/RMSE']
selected = ['OGD', 'FSNet-Advanced', 'FSNet-Ultra', 'FSNet-v3']
radar_colors = ['#9e9e9e', '#388e3c', '#1976d2', '#d32f2f']

# 归一化 (1/metric, 相对OGD)
ogd = results['OGD']
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for name, color in zip(selected, radar_colors):
    d = results[name]
    values = [
        ogd['mape'] / d['mape'],
        ogd['mse'] / d['mse'],
        ogd['mae'] / d['mae'],
        ogd['mse']**0.5 / d['mse']**0.5
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color, markersize=6)
    ax.fill(angles, values, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_title('Radar Chart — Relative Performance (vs OGD=1.0)', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.savefig('figures/v3/6_v3_radar.png', bbox_inches='tight')
plt.close()
print("✅ 6_v3_radar.png")

# ─── 图7: 创新对比表 (Advanced vs Ultra vs v3) ───
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

rows = [
    ['Dimension', 'Advanced (v1)', 'Ultra (v2)', 'v3 (Semantic)'],
    ['Calibration\nGranularity', 'w=[C_out,K]\n(channels identical)', 'w=[C_out,K]\n(channels identical)', 'w=[C_out,C_in,K]\n(full-tensor) OK'],
    ['Memory\nStructure', 'Single-head M=32', '4-head split-q M=64\n(semantic break)', '4-group channel M=32\n(semantic intact) OK'],
    ['Temperature\nStrategy', 'Shift->temp UP\n(wrong direction)', 'Per-head [0.2,0.8]', 'Shift->temp DOWN\n(sharper) OK'],
    ['Controller', '1-layer Linear+SiLU', '2-layer Gated Residual\n(overcomplicated)', '1-layer GradNorm-Gate\n(moderate) OK'],
    ['Gradient\nInput', '1st moment only', '1st+2nd moment concat\n(noisy in OL)', '1st moment + norm scalar\n(lightweight) OK'],
    ['Regularization', 'Dropout(0.1)', 'Dropout(0.1)', 'tanh clamp + Dropout\n+ residual init OK'],
    ['MAPE', '5.06%', '4.81%', '4.21% BEST'],
    ['MSE', '0.0131', '0.0085', '0.0064 BEST'],
]

table = ax.table(cellText=rows, cellLoc='center', loc='center', 
                 colWidths=[0.15, 0.28, 0.28, 0.28])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

# 样式
for i in range(len(rows)):
    for j in range(4):
        cell = table[i, j]
        if i == 0:
            cell.set_facecolor('#37474f')
            cell.set_text_props(color='white', fontweight='bold')
        elif j == 3 and i > 0:
            cell.set_facecolor('#ffebee')
        elif i in [7, 8]:
            cell.set_facecolor('#e8f5e9' if j == 3 else '#f5f5f5')
            cell.set_text_props(fontweight='bold')

ax.set_title('Architecture Innovation Comparison — Three Generations', 
             fontsize=14, fontweight='bold', pad=20)
plt.savefig('figures/v3/7_v3_innovation_table.png', bbox_inches='tight')
plt.close()
print("✅ 7_v3_innovation_table.png")

# ─── 图8: 最终排行榜 ───
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

header = ['Rank', 'Model', 'MAPE (%)', 'MSE', 'MAE', 'vs OGD', 'Time (s)']
rows_lb = []
sorted_results = sorted(results.items(), key=lambda x: x[1]['mape'])
ogd_mape = results['OGD']['mape']
for i, (name, data) in enumerate(sorted_results, 1):
    imp = f"-{(1 - data['mape']/ogd_mape)*100:.1f}%" if name != 'OGD' else '—'
    rows_lb.append([
        f'#{i}', name, f"{data['mape']:.2f}", f"{data['mse']:.6f}",
        f"{data['mae']:.6f}", imp, f"{data['total_time']:.0f}"
    ])

table = ax.table(cellText=rows_lb, colLabels=header, cellLoc='center', loc='center',
                 colWidths=[0.06, 0.17, 0.1, 0.14, 0.14, 0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

for j in range(7):
    table[0, j].set_facecolor('#1565c0')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Highlight #1
for j in range(7):
    table[1, j].set_facecolor('#fff9c4')
    table[1, j].set_text_props(fontweight='bold')

ax.set_title('Final Leaderboard — 8 Models on ETTh1 (features=S, 2 epochs)',
             fontsize=14, fontweight='bold', pad=20)
plt.savefig('figures/v3/8_v3_leaderboard.png', bbox_inches='tight')
plt.close()
print("✅ 8_v3_leaderboard.png")

print("\n" + "="*60)
print("  全部 8 张可视化已生成到 figures/v3/")
print("="*60)
