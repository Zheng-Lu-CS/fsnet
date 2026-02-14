"""
FSNet-Ultra vs FSNet-Advanced 全面对比可视化
7 模型排名 + Ultra 突破性分析
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

out = 'figures/ultra/'
os.makedirs(out, exist_ok=True)

# ── 加载数据 ──
with open('results/comprehensive/comprehensive_results.json', 'r', encoding='utf-8') as f:
    comp = json.load(f)

data = {}
for k, r in comp.items():
    label = k if k != 'FSNet' else 'FSNet-Bug'
    data[label] = {
        'MSE': r['mse'], 'MAE': r['mae'], 'RMSE': r['rmse'],
        'MAPE': r['mape'], 'MSPE': r.get('mspe', 0),
        'total_time': r['total_time']
    }

order = ['OGD', 'ER', 'NoMem', 'FSNet-Bug', 'FSNet-Fixed',
         'FSNet-Advanced', 'FSNet-Ultra']
methods = [m for m in order if m in data]
N = len(methods)

pal = {
    'OGD': '#e74c3c', 'ER': '#e67e22', 'NoMem': '#f1c40f',
    'FSNet-Bug': '#95a5a6', 'FSNet-Fixed': '#3498db',
    'FSNet-Advanced': '#2ecc71', 'FSNet-Ultra': '#9b59b6',
}
bc = [pal.get(m, '#7f8c8d') for m in methods]

# ============================================================
# 1. MAPE 全排名
# ============================================================
fig, ax = plt.subplots(figsize=(15, 7))
mape = [data[m]['MAPE'] for m in methods]
bars = ax.bar(range(N), mape, color=bc, alpha=0.88, width=0.6,
              edgecolor='white', linewidth=1.5)
bi = int(np.argmin(mape))
bars[bi].set_edgecolor('#8e44ad'); bars[bi].set_linewidth(3)

for i, (b, v) in enumerate(zip(bars, mape)):
    txt = f'{v:.2f}%'
    if i == bi: txt += '  NEW SOTA'
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.12, txt,
            ha='center', va='bottom', fontsize=11,
            fontweight='bold', color='#8e44ad' if i == bi else '#2c3e50')

# 标注两次创新
if 'FSNet-Advanced' in data and 'FSNet-Ultra' in data:
    adv_i = methods.index('FSNet-Advanced')
    ult_i = methods.index('FSNet-Ultra')
    imp = (data['FSNet-Advanced']['MAPE'] - data['FSNet-Ultra']['MAPE']) / data['FSNet-Advanced']['MAPE'] * 100
    ax.annotate(f'Ultra vs Advanced\n-{imp:.1f}%',
                xy=(ult_i, mape[ult_i]),
                xytext=(ult_i - 1.5, mape[ult_i] + 3),
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2),
                fontsize=10, color='#8e44ad', fontweight='bold')

ax.set_xticks(range(N)); ax.set_xticklabels(methods, fontsize=10, rotation=20)
ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold')
ax.set_title('7 Models MAPE Ranking — FSNet-Ultra NEW SOTA', fontsize=15, fontweight='bold', pad=18)
ax.set_ylim(0, max(mape)*1.25); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(out, '1_mape_ranking.png'), dpi=300, bbox_inches='tight')
print('>> 1_mape_ranking.png'); plt.close()

# ============================================================
# 2. Advanced vs Ultra 直接对比 (grouped bars)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metrics = ['MAPE', 'MSE', 'MAE', 'RMSE']
pair = ['FSNet-Advanced', 'FSNet-Ultra']
pair_c = [pal['FSNet-Advanced'], pal['FSNet-Ultra']]

for ax, met in zip(axes, metrics):
    vals = [data[m][met] for m in pair]
    bs = ax.bar(range(2), vals, color=pair_c, alpha=0.85, width=0.5)
    for b, v in zip(bs, vals):
        fmt = f'{v:.2f}%' if met == 'MAPE' else f'{v:.6f}'
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), fmt,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(2)); ax.set_xticklabels(pair, fontsize=9)
    imp = (vals[0] - vals[1]) / vals[0] * 100
    ax.set_title(f'{met}  (Ultra: -{imp:.1f}%)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('FSNet-Advanced vs FSNet-Ultra — Head-to-Head', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out, '2_advanced_vs_ultra.png'), dpi=300, bbox_inches='tight')
print('>> 2_advanced_vs_ultra.png'); plt.close()

# ============================================================
# 3. 改进瀑布图 — Ultra vs all
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
ultra_mape = data['FSNet-Ultra']['MAPE']
others = [m for m in methods if m != 'FSNet-Ultra']
imp_v = [(data[m]['MAPE'] - ultra_mape) / data[m]['MAPE'] * 100 for m in others]
ic = [pal.get(m, '#7f8c8d') for m in others]

bars = ax.bar(range(len(others)), imp_v, color=ic, alpha=0.85, width=0.55, edgecolor='white', lw=1.5)
for b, v in zip(bars, imp_v):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xticks(range(len(others))); ax.set_xticklabels(others, fontsize=11, rotation=15)
ax.set_ylabel('MAPE Reduction (%)', fontsize=13, fontweight='bold')
ax.set_title(f'FSNet-Ultra (MAPE={ultra_mape:.2f}%) — Improvement Over All Baselines',
             fontsize=14, fontweight='bold', pad=15)
ax.axhline(y=50, color='red', ls='--', lw=1.5, alpha=0.6)
ax.text(len(others)-0.5, 51, '50% line', fontsize=9, color='red')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(out, '3_ultra_improvement.png'), dpi=300, bbox_inches='tight')
print('>> 3_ultra_improvement.png'); plt.close()

# ============================================================
# 4. 雷达图 — OGD / NoMem / Advanced / Ultra
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
radar_m = ['MSE', 'MAE', 'RMSE', 'MAPE']
angles = np.linspace(0, 2*np.pi, len(radar_m), endpoint=False).tolist()
angles += angles[:1]

key4 = ['OGD', 'NoMem', 'FSNet-Advanced', 'FSNet-Ultra']
key4 = [m for m in key4 if m in data]
maxv = {rm: max(data[m][rm] for m in key4) for rm in radar_m}

for method in key4:
    vals = [data[method][rm] for rm in radar_m]
    norm = [1 - v/maxv[rm] if maxv[rm] > 0 else 1 for v, rm in zip(vals, radar_m)]
    norm += norm[:1]
    ax.plot(angles, norm, 'o-', lw=2.5, label=method, color=pal.get(method))
    ax.fill(angles, norm, alpha=0.08, color=pal.get(method))

ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_m, fontsize=13)
ax.set_ylim(0, 1)
ax.set_title('Radar: OGD / NoMem / Advanced / Ultra\n(outer = better)',
             fontsize=13, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(out, '4_radar_ultra.png'), dpi=300, bbox_inches='tight')
print('>> 4_radar_ultra.png'); plt.close()

# ============================================================
# 5. 创新演进 — 三代架构指标下降曲线
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
stages = ['OGD\n(Baseline)', 'NoMem\n(Adapter)', 'FSNet-Bug\n(Original)',
          'FSNet-Fixed\n(Bug Fix)', 'Advanced\n(4 Innovations)', 'Ultra\n(6 New)']
stage_methods = ['OGD', 'NoMem', 'FSNet-Bug', 'FSNet-Fixed', 'FSNet-Advanced', 'FSNet-Ultra']
stage_methods = [m for m in stage_methods if m in data]
stages = stages[:len(stage_methods)]

mape_line = [data[m]['MAPE'] for m in stage_methods]
mse_line = [data[m]['MSE'] * 1000 for m in stage_methods]  # scale for visibility
colors_line = [pal.get(m) for m in stage_methods]

ax.plot(range(len(stages)), mape_line, 'o-', lw=3, color='#e74c3c', markersize=10, label='MAPE (%)')
ax.fill_between(range(len(stages)), mape_line, alpha=0.1, color='#e74c3c')

for i, (v, c) in enumerate(zip(mape_line, colors_line)):
    ax.scatter(i, v, s=200, c=c, zorder=5, edgecolors='white', linewidth=2)
    ax.text(i, v + 0.4, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')

ax2 = ax.twinx()
ax2.plot(range(len(stages)), mse_line, 's--', lw=2, color='#3498db', markersize=8, label='MSE (x1000)')
ax2.set_ylabel('MSE (x1000)', fontsize=12, color='#3498db')
for i, v in enumerate(mse_line):
    ax2.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9, color='#3498db')

ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold', color='#e74c3c')
ax.set_title('Architecture Evolution — Three Generations of Innovation',
             fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=11)
ax2.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(out, '5_evolution_curve.png'), dpi=300, bbox_inches='tight')
print('>> 5_evolution_curve.png'); plt.close()

# ============================================================
# 6. 创新点对比表 — Advanced vs Ultra
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')

headers = ['Dimension', 'FSNet-Advanced', 'FSNet-Ultra', 'Ultra Gain']
rows = [
    headers,
    ['Memory Attention', 'Single-Head (Top-3)', 'Multi-Head 4-Head (Top-3/head)', 'Multi-perspective retrieval'],
    ['Controller', 'Linear+SiLU+Dropout', '2x Gated Residual Block', 'Deeper + non-linear gating'],
    ['Memory Capacity', '32 slots', '64 slots (2x)', '2x pattern coverage'],
    ['Gradient Signal', '1st-moment EMA only', '1st + 2nd-moment EMA', 'Richer shift detection'],
    ['Trigger Threshold', 'Fixed cos<-0.75', 'Adaptive self-tuning', 'Auto-calibrated sensitivity'],
    ['Regularization', 'Dropout(0.1)', 'Dropout + Diversity Penalty', 'Prevents memory collapse'],
    ['Hidden Dim', 'nh=64', 'nh=96 (+50%)', 'More expressive'],
    ['MAPE', '5.06%', '4.81%', '-4.9% (relative)'],
    ['MSE', '0.013064', '0.008535', '-34.6%'],
    ['MAE', '0.069066', '0.061516', '-10.9%'],
]

tab = ax.table(cellText=rows, cellLoc='center', loc='center',
               colWidths=[0.18, 0.26, 0.28, 0.24])
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.5)

for j in range(4):
    tab[(0, j)].set_facecolor('#2c3e50')
    tab[(0, j)].set_text_props(weight='bold', color='white', fontsize=11)

for i in range(1, len(rows)):
    for j in range(4):
        cell = tab[(i, j)]
        if i >= 8:  # metric rows
            cell.set_facecolor('#f0e6ff')
            cell.set_text_props(fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f8f9fa')

ax.set_title('FSNet-Advanced vs FSNet-Ultra — Innovation Comparison',
             fontsize=16, fontweight='bold', pad=30)
plt.savefig(os.path.join(out, '6_innovation_comparison.png'), dpi=300, bbox_inches='tight')
print('>> 6_innovation_comparison.png'); plt.close()

# ============================================================
# 7. 最终排行榜
# ============================================================
fig, ax = plt.subplots(figsize=(18, 9))
ax.axis('off')

ogd_mape = data['OGD']['MAPE']
headers = ['Rank', 'Model', 'MAPE(%)', 'MSE', 'MAE', 'RMSE', 'vs OGD', 'Category']
sorted_m = sorted(methods, key=lambda m: data[m]['MAPE'])
tdata = [headers]
for rank, m in enumerate(sorted_m, 1):
    d = data[m]
    vs = f"-{(ogd_mape - d['MAPE'])/ogd_mape*100:.1f}%" if m != 'OGD' else 'baseline'
    cat = ('Baseline' if m in ['OGD','ER'] else
           'Bug ver.' if m == 'FSNet-Bug' else
           'Fix ver.' if m in ['NoMem','FSNet-Fixed'] else
           'Innovation-v1' if m == 'FSNet-Advanced' else
           'Innovation-v2')
    tdata.append([f'#{rank}', m, f'{d["MAPE"]:.2f}', f'{d["MSE"]:.6f}',
                  f'{d["MAE"]:.6f}', f'{d["RMSE"]:.6f}', vs, cat])

tab = ax.table(cellText=tdata, cellLoc='center', loc='center',
               colWidths=[0.06, 0.15, 0.09, 0.12, 0.12, 0.12, 0.1, 0.14])
tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1, 2.8)

for j in range(len(headers)):
    tab[(0,j)].set_facecolor('#2c3e50')
    tab[(0,j)].set_text_props(weight='bold', color='white', fontsize=12)

for i in range(1, len(tdata)):
    mname = tdata[i][1]
    for j in range(len(headers)):
        cell = tab[(i,j)]
        if mname == 'FSNet-Ultra':
            cell.set_facecolor('#e8daef')
            if j == 1:
                cell.set_text_props(weight='bold', color='#8e44ad', fontsize=12)
        elif mname == 'FSNet-Advanced':
            cell.set_facecolor('#d5f5e3')
        elif i % 2 == 0:
            cell.set_facecolor('#f8f9fa')

ax.set_title('Final Leaderboard — 7 Models on ETTh1', fontsize=16, fontweight='bold', pad=35)
plt.savefig(os.path.join(out, '7_final_leaderboard.png'), dpi=300, bbox_inches='tight')
print('>> 7_final_leaderboard.png'); plt.close()

# ── 打印总结 ──
print(f"\n{'='*80}")
print("  FINAL RESULTS SUMMARY")
print(f"{'='*80}")
for i, m in enumerate(sorted_m, 1):
    d = data[m]
    s = ' ⭐ NEW SOTA' if i == 1 else ''
    print(f"  #{i} {m:<18} MAPE={d['MAPE']:>6.2f}%  MSE={d['MSE']:.6f}  MAE={d['MAE']:.6f}{s}")

adv = data['FSNet-Advanced']
ult = data['FSNet-Ultra']
print(f"\n  Ultra vs Advanced:")
print(f"    MAPE: {adv['MAPE']:.2f}% -> {ult['MAPE']:.2f}%  ({(adv['MAPE']-ult['MAPE'])/adv['MAPE']*100:+.1f}%)")
print(f"    MSE:  {adv['MSE']:.6f} -> {ult['MSE']:.6f}  ({(adv['MSE']-ult['MSE'])/adv['MSE']*100:+.1f}%)")
print(f"    MAE:  {adv['MAE']:.6f} -> {ult['MAE']:.6f}  ({(adv['MAE']-ult['MAE'])/adv['MAE']*100:+.1f}%)")

print(f"\n  7 charts saved to: {out}")
print(f"{'='*80}")
