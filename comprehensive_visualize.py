"""
全面对比可视化：NoMem vs FSNet原始 vs FSNet-Fixed
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = 'figures/comprehensive/'
os.makedirs(output_dir, exist_ok=True)

# 加载之前的消融实验结果
with open('results/ablation/ablation_results.json', 'r', encoding='utf-8') as f:
    ablation = json.load(f)

# 加载综合对比结果
comp_file = 'results/comprehensive/comprehensive_results.json'
if os.path.exists(comp_file):
    with open(comp_file, 'r', encoding='utf-8') as f:
        comp = json.load(f)
else:
    comp = {}

# 构建统一数据结构
all_data = {}

# 从ablation结果中提取
for key in ['OGD', 'ER', 'NoMem', 'FSNet']:
    if key in ablation:
        m = ablation[key]['metrics']
        t = ablation[key]['time']
        label = key if key != 'FSNet' else 'FSNet-Bug'
        all_data[label] = {
            'MSE': m['MSE'], 'MAE': m['MAE'], 'RMSE': m['RMSE'],
            'MAPE': m['MAPE'], 'MSPE': m['MSPE'],
            'total_time': t['total'],
            'train_time': t['train'], 'test_time': t['test']
        }

# 从综合实验结果中提取
if 'FSNet-Fixed' in comp:
    r = comp['FSNet-Fixed']
    all_data['FSNet-Fixed'] = {
        'MSE': r['mse'], 'MAE': r['mae'], 'RMSE': r['rmse'],
        'MAPE': r['mape'], 'MSPE': r.get('mspe', 0),
        'total_time': r['total_time'],
        'train_time': r.get('train_time', 0), 'test_time': r.get('test_time', 0)
    }

print("\n" + "="*80)
print("可用数据：", list(all_data.keys()))
print("="*80)

# 颜色方案
colors = {
    'OGD': '#e74c3c',
    'ER': '#3498db',
    'NoMem': '#f39c12',
    'FSNet-Bug': '#95a5a6',
    'FSNet-Fixed': '#2ecc71',
}

methods = list(all_data.keys())
n_methods = len(methods)

# ==============================================================
# 图1: MAPE对比（核心指标）
# ==============================================================
fig, ax = plt.subplots(figsize=(12, 7))

mape_vals = [all_data[m]['MAPE'] for m in methods]
bar_colors = [colors.get(m, '#7f8c8d') for m in methods]

bars = ax.bar(range(n_methods), mape_vals, color=bar_colors, alpha=0.85, width=0.6,
              edgecolor='white', linewidth=1.5)

# 标注最佳
best_idx = np.argmin(mape_vals)
bars[best_idx].set_edgecolor('#e67e22')
bars[best_idx].set_linewidth(3)

# 数值标签
for i, (bar, val) in enumerate(zip(bars, mape_vals)):
    label = f'{val:.2f}%'
    if i == best_idx:
        label += ' ⭐'
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
           label, ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xticks(range(n_methods))
ax.set_xticklabels(methods, fontsize=11, rotation=15)
ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold')
ax.set_title('各方法MAPE对比\n(越低越好)', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(mape_vals) * 1.2)

# 添加改进标注
if 'NoMem' in all_data and 'FSNet-Bug' in all_data:
    nomem_mape = all_data['NoMem']['MAPE']
    bug_mape = all_data['FSNet-Bug']['MAPE']
    ax.annotate('Bug导致\nMemory失效',
               xy=(methods.index('FSNet-Bug'), bug_mape),
               xytext=(methods.index('FSNet-Bug') + 0.5, bug_mape + 1.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_mape_comparison.png'), dpi=300, bbox_inches='tight')
print(">> 保存: 1_mape_comparison.png")
plt.close()

# ==============================================================
# 图2: MSE+MAE双指标对比
# ==============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# MSE
mse_vals = [all_data[m]['MSE'] for m in methods]
bars1 = ax1.bar(range(n_methods), mse_vals, color=bar_colors, alpha=0.85, width=0.6)
for i, (bar, val) in enumerate(zip(bars1, mse_vals)):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)
ax1.set_xticks(range(n_methods))
ax1.set_xticklabels(methods, fontsize=10, rotation=15)
ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
ax1.set_title('MSE对比 (越低越好)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# MAE
mae_vals = [all_data[m]['MAE'] for m in methods]
bars2 = ax2.bar(range(n_methods), mae_vals, color=bar_colors, alpha=0.85, width=0.6)
for i, (bar, val) in enumerate(zip(bars2, mae_vals)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)
ax2.set_xticks(range(n_methods))
ax2.set_xticklabels(methods, fontsize=10, rotation=15)
ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax2.set_title('MAE对比 (越低越好)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_mse_mae_comparison.png'), dpi=300, bbox_inches='tight')
print(">> 保存: 2_mse_mae_comparison.png")
plt.close()

# ==============================================================
# 图3: 改进幅度分析（相对OGD基线）
# ==============================================================
fig, ax = plt.subplots(figsize=(14, 7))

ogd_mape = all_data['OGD']['MAPE']
improvements = {}
for m in methods:
    if m != 'OGD':
        imp = (ogd_mape - all_data[m]['MAPE']) / ogd_mape * 100
        improvements[m] = imp

imp_methods = list(improvements.keys())
imp_vals = list(improvements.values())
imp_colors = [colors.get(m, '#7f8c8d') for m in imp_methods]

bars = ax.barh(imp_methods, imp_vals, color=imp_colors, alpha=0.85, height=0.5)

for bar, val in zip(bars, imp_vals):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
           f'+{val:.1f}%', ha='left', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('相对OGD基线的MAPE改进 (%)', fontsize=12, fontweight='bold')
ax.set_title('各方法相对OGD基线的改进幅度', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_improvement_vs_baseline.png'), dpi=300, bbox_inches='tight')
print(">> 保存: 3_improvement_vs_baseline.png")
plt.close()

# ==============================================================
# 图4: 雷达图
# ==============================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

radar_metrics = ['MSE', 'MAE', 'RMSE', 'MAPE']
categories = radar_metrics + radar_metrics[:1]  # 闭合

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

# 选择关键方法
key_methods = ['OGD', 'NoMem', 'FSNet-Bug']
if 'FSNet-Fixed' in all_data:
    key_methods.append('FSNet-Fixed')

for method in key_methods:
    vals = [all_data[method][m] for m in radar_metrics]
    # 归一化（统一到最大值为1）
    max_vals = [max(all_data[m2][m] for m2 in key_methods) for m in radar_metrics]
    norm_vals = [1 - v/mv if mv > 0 else 1 for v, mv in zip(vals, max_vals)]
    norm_vals += norm_vals[:1]  # 闭合
    
    ax.plot(angles, norm_vals, 'o-', linewidth=2, label=method,
           color=colors.get(method, '#7f8c8d'))
    ax.fill(angles, norm_vals, alpha=0.1, color=colors.get(method, '#7f8c8d'))

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=12)
ax.set_ylim(0, 1)
ax.set_title('综合性能雷达图\n(外圈=更好)', fontsize=14, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_radar_comparison.png'), dpi=300, bbox_inches='tight')
print(">> 保存: 4_radar_comparison.png")
plt.close()

# ==============================================================
# 图5: Bug分析 - FSNet-Bug vs FSNet-Fixed vs NoMem
# ==============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

compare_methods = ['NoMem', 'FSNet-Bug']
if 'FSNet-Fixed' in all_data:
    compare_methods.append('FSNet-Fixed')

for ax_idx, metric in enumerate(['MAPE', 'MSE', 'MAE']):
    vals = [all_data[m][metric] for m in compare_methods]
    bars = axes[ax_idx].bar(range(len(compare_methods)), vals,
                           color=[colors.get(m, '#7f8c8d') for m in compare_methods],
                           alpha=0.85, width=0.5)
    
    # 标注最佳
    best_i = np.argmin(vals)
    bars[best_i].set_edgecolor('#e67e22')
    bars[best_i].set_linewidth(3)
    
    for bar, val in zip(bars, vals):
        fmt = '{:.2f}%' if metric == 'MAPE' else '{:.4f}'
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         fmt.format(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[ax_idx].set_xticks(range(len(compare_methods)))
    axes[ax_idx].set_xticklabels(compare_methods, fontsize=10)
    axes[ax_idx].set_title(f'{metric}对比', fontsize=13, fontweight='bold')
    axes[ax_idx].grid(True, alpha=0.3, axis='y')

fig.suptitle('Bug修复效果分析：Memory机制恢复验证', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_bug_fix_analysis.png'), dpi=300, bbox_inches='tight')
print(">> 保存: 5_bug_fix_analysis.png")
plt.close()

# ==============================================================
# 图6: 综合总结表格
# ==============================================================
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

# 表头
headers = ['模型', 'MAPE(%)', 'MSE', 'MAE', 'RMSE', 'vs OGD改进', '总时间(s)']

# 数据行
table_data = [headers]
for m in methods:
    d = all_data[m]
    if m == 'OGD':
        imp = '-'
    else:
        imp = f'+{(ogd_mape - d["MAPE"]) / ogd_mape * 100:.1f}%'
    
    table_data.append([
        m, f'{d["MAPE"]:.2f}', f'{d["MSE"]:.6f}', f'{d["MAE"]:.6f}',
        f'{d["RMSE"]:.6f}', imp, f'{d["total_time"]:.0f}'
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.14, 0.11, 0.13, 0.13, 0.13, 0.13, 0.11])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 表头样式
for j in range(len(headers)):
    cell = table[(0, j)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# 数据行样式
for i in range(1, len(table_data)):
    for j in range(len(headers)):
        cell = table[(i, j)]
        # 交替行颜色
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        
        # 最佳行高亮
        method_name = table_data[i][0]
        mape_val = all_data[method_name]['MAPE']
        best_mape = min(d['MAPE'] for d in all_data.values())
        if mape_val == best_mape:
            cell.set_facecolor('#d5f5e3')
            if j == 0:
                cell.set_text_props(weight='bold', color='#27ae60')

ax.set_title('全面性能对比总表', fontsize=16, fontweight='bold', pad=30)

plt.savefig(os.path.join(output_dir, '6_summary_table.png'), dpi=300, bbox_inches='tight')
print(">> 保存: 6_summary_table.png")
plt.close()

# ==============================================================
# 图7: 创新架构设计概览
# ==============================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 标题
ax.text(0.5, 0.97, 'FSNet架构优化路线图', fontsize=18, fontweight='bold',
        ha='center', va='top', transform=ax.transAxes)

# 三个阶段
stages = [
    {
        'title': '阶段1: Bug修复',
        'color': '#e74c3c',
        'items': [
            'fw_chunks使用idx而非v加权',
            'Memory检索与注意力无关',
            '修复: old_q = (W[:,idx]*v).sum()',
            '效果: Memory机制恢复工作',
        ],
        'x': 0.05, 'width': 0.28
    },
    {
        'title': '阶段2: 理论分析',
        'color': '#3498db',
        'items': [
            'Chunk按flatten非通道分块',
            '融合系数固定不自适应',
            'Memory写入缺乏归一化',
            '检索策略需要优化',
        ],
        'x': 0.36, 'width': 0.28
    },
    {
        'title': '阶段3: 创新架构',
        'color': '#2ecc71',
        'items': [
            '结构对齐Chunk(按通道)',
            '自适应融合系数(可学习)',
            'Top-3检索+动态温度',
            'Controller Dropout正则化',
        ],
        'x': 0.67, 'width': 0.28
    }
]

for stage in stages:
    # 框
    rect = plt.Rectangle((stage['x'], 0.15), stage['width'], 0.70,
                         linewidth=2, edgecolor=stage['color'],
                         facecolor=stage['color'], alpha=0.1)
    ax.add_patch(rect)
    
    # 标题
    ax.text(stage['x'] + stage['width']/2, 0.82, stage['title'],
           fontsize=14, fontweight='bold', ha='center', color=stage['color'])
    
    # 内容
    for i, item in enumerate(stage['items']):
        ax.text(stage['x'] + 0.02, 0.72 - i*0.12, f'• {item}',
               fontsize=10, va='top')

# 箭头连接
for i in range(2):
    ax.annotate('', xy=(stages[i+1]['x'] - 0.01, 0.5),
               xytext=(stages[i]['x'] + stages[i]['width'] + 0.01, 0.5),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=3))

# 底部总结
ax.text(0.5, 0.05, 
        '核心发现: 论文实现Bug导致Memory失效 → 修复后验证Memory有效性 → 提出结构对齐创新方案',
        fontsize=12, ha='center', va='bottom', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.8))

plt.savefig(os.path.join(output_dir, '7_architecture_roadmap.png'), dpi=300, bbox_inches='tight')
print(">> 保存: 7_architecture_roadmap.png")
plt.close()

# ==============================================================
# 打印总结
# ==============================================================
print("\n" + "="*80)
print("全面对比结果总结")
print("="*80)

for m in methods:
    d = all_data[m]
    imp = ''
    if m != 'OGD':
        imp = f'(vs OGD: +{(ogd_mape - d["MAPE"])/ogd_mape*100:.1f}%)'
    print(f"  {m:<15} MAPE={d['MAPE']:.2f}%  MSE={d['MSE']:.6f}  MAE={d['MAE']:.6f}  {imp}")

# FSNet-Bug vs FSNet-Fixed对比
if 'FSNet-Fixed' in all_data:
    bug = all_data['FSNet-Bug']
    fixed = all_data['FSNet-Fixed']
    nomem = all_data['NoMem']
    
    print(f"\n📊 Bug修复效果分析:")
    print(f"  FSNet-Bug  → FSNet-Fixed: MAPE {bug['MAPE']:.2f}% → {fixed['MAPE']:.2f}%")
    fix_imp = (bug['MAPE'] - fixed['MAPE']) / bug['MAPE'] * 100
    print(f"  修复后MAPE改进: {fix_imp:+.2f}%")
    
    vs_nomem = (nomem['MAPE'] - fixed['MAPE']) / nomem['MAPE'] * 100
    print(f"\n📊 FSNet-Fixed vs NoMem:")
    print(f"  NoMem: MAPE={nomem['MAPE']:.2f}%  FSNet-Fixed: MAPE={fixed['MAPE']:.2f}%")
    print(f"  改进: {vs_nomem:+.2f}%")
    
    if fixed['MAPE'] < nomem['MAPE']:
        print(f"\n  ✅ FSNet-Fixed成功超越NoMem!")
        print(f"  ✅ 证明Bug修复使Memory机制恢复工作")
    elif fixed['MAPE'] < bug['MAPE']:
        print(f"\n  ✅ Bug修复带来性能提升")
        print(f"  💡 但还未超越NoMem，需要更多训练")

print(f"\n✅ 共生成7张可视化图表到: {output_dir}")
print("="*80)
