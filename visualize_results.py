"""
可视化分析脚本 - 生成论文级别图表
运行方式: python fsnet/visualize_results.py
前置条件: 先运行ablation_study.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

# 配置中文字体（Windows系统）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("="*70)
print("可视化分析 - 生成论文级别图表")
print("="*70)

# 加载实验结果
result_dir = './results/ablation/'
json_path = f"{result_dir}ablation_results.json"
np_path = f"{result_dir}ablation_predictions.npz"

if not os.path.exists(json_path):
    print(f"❌ 错误: 找不到实验结果文件 {json_path}")
    print("   请先运行: python fsnet/ablation_study.py")
    exit(1)

# 加载数据
with open(json_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

predictions = np.load(np_path)

print(f"✅ 加载数据成功")
print(f"   方法数量: {len(results)}")
print(f"   预测数组: {list(predictions.keys())}")

# 创建输出目录
fig_dir = './figures/'
os.makedirs(fig_dir, exist_ok=True)

# 颜色方案
colors = {
    'OGD': '#e74c3c',
    'ER': '#3498db',
    'NoMem': '#f39c12',
    'FSNet': '#2ecc71'
}

print("\n" + "="*70)
print("图1: 预测曲线对比（前200个时间步）")
print("="*70)

# 图1: 预测曲线对比
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('消融实验: 预测曲线对比 (前200步)', fontsize=16, fontweight='bold')

methods = ['OGD', 'ER', 'NoMem', 'FSNet']
for idx, method in enumerate(methods):
    ax = axes[idx // 2, idx % 2]
    
    # 获取数据（只显示前200个预测步）
    preds = predictions[f'{method}_preds'][:200].flatten()
    trues = predictions[f'{method}_trues'][:200].flatten()
    
    # 绘制
    ax.plot(trues, label='真实值', color='black', linewidth=2, alpha=0.7)
    ax.plot(preds, label='预测值', color=colors[method], linewidth=1.5, linestyle='--')
    
    # 计算局部误差
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues)**2)
    
    # 标题和标签
    ax.set_title(f'{results[method]["name"]}\nMAE={mae:.4f}, MSE={mse:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('值', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1_path = f'{fig_dir}1_prediction_curves.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"✅ 保存图1: {fig1_path}")
plt.close()

print("\n" + "="*70)
print("图2: 性能指标对比柱状图")
print("="*70)

# 图2: 性能指标对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('消融实验: 性能指标对比', fontsize=16, fontweight='bold')

metrics_to_plot = [
    ('MSE', 'Mean Squared Error'),
    ('MAE', 'Mean Absolute Error'),
    ('MAPE', 'Mean Absolute Percentage Error (%)'),
    ('RMSE', 'Root Mean Squared Error')
]

for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    # 提取数据
    values = [results[m]['metrics'][metric_key] for m in methods]
    bars = ax.bar(methods, values, color=[colors[m] for m in methods], 
                   edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}' if val < 100 else f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 标题和标签
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('值', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 标记最佳值
    best_idx = np.argmin(values)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
fig2_path = f'{fig_dir}2_metrics_comparison.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"✅ 保存图2: {fig2_path}")
plt.close()

print("\n" + "="*70)
print("图3: 改进百分比雷达图")
print("="*70)

# 图3: 改进百分比（相比OGD baseline）
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_title('FSNet各组件贡献分析（相比OGD baseline）', fontsize=14, fontweight='bold', pad=20)

# 计算改进百分比
baseline = results['OGD']['metrics']
improvements = {
    'ER': {},
    'NoMem': {},
    'FSNet': {}
}

for method in ['ER', 'NoMem', 'FSNet']:
    for metric in ['MSE', 'MAE', 'RMSE', 'MAPE']:
        baseline_val = baseline[metric]
        method_val = results[method]['metrics'][metric]
        improvement = (baseline_val - method_val) / baseline_val * 100
        improvements[method][metric] = improvement

# 雷达图
metrics_labels = ['MSE', 'MAE', 'RMSE', 'MAPE']
angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合

ax = plt.subplot(111, projection='polar')

for method in ['ER', 'NoMem', 'FSNet']:
    values = [improvements[method][m] for m in metrics_labels]
    values += values[:1]  # 闭合
    
    ax.plot(angles, values, 'o-', linewidth=2, label=results[method]['name'],
            color=colors[method])
    ax.fill(angles, values, alpha=0.15, color=colors[method])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_labels, fontsize=12)
ax.set_ylim(0, max([max(improvements[m].values()) for m in improvements]) * 1.1)
ax.set_ylabel('改进百分比 (%)', fontsize=11)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True)

fig3_path = f'{fig_dir}3_improvement_radar.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"✅ 保存图3: {fig3_path}")
plt.close()

print("\n" + "="*70)
print("图4: 误差分布箱线图")
print("="*70)

# 图4: 误差分布
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_title('预测误差分布对比', fontsize=14, fontweight='bold')

errors_data = []
labels = []

for method in methods:
    preds = predictions[f'{method}_preds'].flatten()
    trues = predictions[f'{method}_trues'].flatten()
    errors = np.abs(preds - trues)
    errors_data.append(errors)
    labels.append(f"{method}\n(MAE={np.mean(errors):.4f})")

bp = ax.boxplot(errors_data, labels=labels, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

# 上色
for patch, method in zip(bp['boxes'], methods):
    patch.set_facecolor(colors[method])
    patch.set_alpha(0.6)

ax.set_ylabel('绝对误差', fontsize=12)
ax.set_xlabel('方法', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)

fig4_path = f'{fig_dir}4_error_distribution.png'
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
print(f"✅ 保存图4: {fig4_path}")
plt.close()

print("\n" + "="*70)
print("图5: 时间效率对比")
print("="*70)

# 图5: 时间效率
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('计算时间对比', fontsize=14, fontweight='bold')

# 训练时间
train_times = [results[m]['time']['train'] for m in methods]
bars1 = ax1.bar(methods, train_times, color=[colors[m] for m in methods],
                edgecolor='black', linewidth=1.5)
for bar, val in zip(bars1, train_times):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_title('训练时间', fontsize=12)
ax1.set_ylabel('秒', fontsize=10)
ax1.grid(True, axis='y', alpha=0.3)

# 测试时间
test_times = [results[m]['time']['test'] for m in methods]
bars2 = ax2.bar(methods, test_times, color=[colors[m] for m in methods],
                edgecolor='black', linewidth=1.5)
for bar, val in zip(bars2, test_times):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_title('测试时间', fontsize=12)
ax2.set_ylabel('秒', fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig5_path = f'{fig_dir}5_time_comparison.png'
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
print(f"✅ 保存图5: {fig5_path}")
plt.close()

print("\n" + "="*70)
print("图6: 逐步改进趋势图")
print("="*70)

# 图6: 逐步改进趋势
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

metrics_trend = ['MSE', 'MAE', 'RMSE', 'MAPE']
x = np.arange(len(metrics_trend))
width = 0.2

for i, method in enumerate(methods):
    values = [results[method]['metrics'][m] for m in metrics_trend]
    # 归一化到0-1（方便比较）
    max_vals = [max([results[m]['metrics'][metric] for m in methods]) 
                for metric in metrics_trend]
    normalized = [v/mv for v, mv in zip(values, max_vals)]
    
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, normalized, width, label=results[method]['name'],
                   color=colors[method], edgecolor='black', linewidth=1)

ax.set_xlabel('指标', fontsize=12)
ax.set_ylabel('归一化值（越小越好）', fontsize=12)
ax.set_title('各方法性能全景对比（归一化）', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_trend)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 1.1)

fig6_path = f'{fig_dir}6_normalized_comparison.png'
plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
print(f"✅ 保存图6: {fig6_path}")
plt.close()

print("\n" + "="*70)
print("📊 可视化完成! 生成图表汇总:")
print("="*70)
print(f"1. {fig1_path}")
print(f"2. {fig2_path}")
print(f"3. {fig3_path}")
print(f"4. {fig4_path}")
print(f"5. {fig5_path}")
print(f"6. {fig6_path}")
print("="*70)
print("✅ 所有图表已保存到 ./figures/ 目录")
print("💡 建议: 这些图表可直接用于论文、报告、GitHub README")
