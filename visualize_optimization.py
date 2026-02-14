"""
FSNet优化结果可视化
对比NoMem与修复Bug后的FSNet性能
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置中文字体和负号显示
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')  # 使用默认样式
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_results():
    """加载实验结果"""
    results_file = 'results/optimization/optimization_results.json'
    
    if not os.path.exists(results_file):
        print(f"错误: 找不到结果文件 {results_file}")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def create_comparison_plots(results):
    """创建对比图表"""
    
    # 创建输出目录
    output_dir = 'figures/optimization/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    nomem = results.get('NoMem', {})
    fsnet = results.get('FSNet_Fixed', {})
    
    if not nomem or not fsnet:
        print("错误: 缺少实验结果数据")
        return
    
    # 1. 性能指标对比条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['mse', 'mae', 'rmse', 'mape']
    metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE (%)']
    x = np.arange(len(metrics))
    width = 0.35
    
    nomem_vals = [nomem.get(m, 0) for m in metrics]
    fsnet_vals = [fsnet.get(m, 0) for m in metrics]
    
    bars1 = ax.bar(x - width/2, nomem_vals, width, label='NoMem (只有Adapter)', color='#f39c12', alpha=0.8)
    bars2 = ax.bar(x + width/2, fsnet_vals, width, label='FSNet-Fixed (修复后)', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('性能指标', fontsize=12, fontweight='bold')
    ax.set_ylabel('数值', fontsize=12, fontweight='bold')
    ax.set_title('NoMem vs FSNet-Fixed 性能对比', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}' if height < 1 else f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 1_metrics_comparison.png")
    plt.close()
    
    # 2. 改进幅度分析
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = []
    for metric in metrics:
        nomem_val = nomem.get(metric, 0)
        fsnet_val = fsnet.get(metric, 0)
        if nomem_val != 0:
            improvement = (nomem_val - fsnet_val) / nomem_val * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.barh(metric_names, improvements, color=colors, alpha=0.8)
    
    ax.set_xlabel('改进幅度 (%)', fontsize=12, fontweight='bold')
    ax.set_title('FSNet-Fixed相比NoMem的改进幅度\n(正值=更好)', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{imp:+.2f}%',
               ha='left' if imp > 0 else 'right',
               va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_improvement_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 2_improvement_analysis.png")
    plt.close()
    
    # 3. 综合雷达图对比
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 归一化指标（越小越好，所以取倒数）
    categories = ['MSE', 'MAE', 'RMSE', 'MAPE']
    
    # 归一化到0-1范围（越小越好转换为越大越好）
    def normalize_metric(nomem_val, fsnet_val):
        max_val = max(nomem_val, fsnet_val)
        if max_val == 0:
            return 1.0, 1.0
        # 转换为"越大越好"：1 - (value / max)
        return 1 - nomem_val/max_val, 1 - fsnet_val/max_val
    
    nomem_radar = []
    fsnet_radar = []
    for metric in metrics:
        n_norm, f_norm = normalize_metric(nomem[metric], fsnet[metric])
        nomem_radar.append(n_norm)
        fsnet_radar.append(f_norm)
    
    # 完成闭合
    nomem_radar += nomem_radar[:1]
    fsnet_radar += fsnet_radar[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, nomem_radar, 'o-', linewidth=2, label='NoMem', color='#f39c12')
    ax.fill(angles, nomem_radar, alpha=0.25, color='#f39c12')
    
    ax.plot(angles, fsnet_radar, 'o-', linewidth=2, label='FSNet-Fixed', color='#2ecc71')
    ax.fill(angles, fsnet_radar, alpha=0.25, color='#2ecc71')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('综合性能雷达图\n(外圈=更好)', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_radar_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 3_radar_comparison.png")
    plt.close()
    
    # 4. 训练时间对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_metrics = ['train_time', 'test_time', 'ol_time']
    time_names = ['训练时间', '测试时间', '在线学习时间']
    x = np.arange(len(time_metrics))
    
    nomem_times = [nomem.get(m, 0) for m in time_metrics]
    fsnet_times = [fsnet.get(m, 0) for m in time_metrics]
    
    bars1 = ax.bar(x - width/2, nomem_times, width, label='NoMem', color='#f39c12', alpha=0.8)
    bars2 = ax.bar(x + width/2, fsnet_times, width, label='FSNet-Fixed', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('阶段', fontsize=12, fontweight='bold')
    ax.set_ylabel('时间 (秒)', fontsize=12, fontweight='bold')
    ax.set_title('训练与测试时间对比', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(time_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_time_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 4_time_comparison.png")
    plt.close()
    
    # 5. 综合对比表格图
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = [
        ['指标', 'NoMem', 'FSNet-Fixed', '改进幅度', '结论'],
    ]
    
    for metric, name in zip(metrics, metric_names):
        n_val = nomem[metric]
        f_val = fsnet[metric]
        imp = (n_val - f_val) / n_val * 100 if n_val != 0 else 0
        conclusion = '✓ 更好' if imp > 0 else '✗ 更差'
        
        table_data.append([
            name,
            f"{n_val:.6f}" if n_val < 1 else f"{n_val:.2f}",
            f"{f_val:.6f}" if f_val < 1 else f"{f_val:.2f}",
            f"{imp:+.2f}%",
            conclusion
        ])
    
    # 添加时间信息
    table_data.append(['总时间(s)', f"{nomem['total_time']:.1f}", 
                      f"{fsnet['total_time']:.1f}", 
                      f"{(nomem['total_time'] - fsnet['total_time']):.1f}s",
                      '时间差异'])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.25, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # 设置表头样式
    for j in range(5):
        cell = table[(0, j)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # 设置数据行样式
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            
            # 结论列特殊颜色
            if j == 4 and i < len(table_data) - 1:
                if '更好' in table_data[i][j]:
                    cell.set_text_props(color='#27ae60', weight='bold')
                elif '更差' in table_data[i][j]:
                    cell.set_text_props(color='#e74c3c', weight='bold')
    
    plt.title('NoMem vs FSNet-Fixed 详细对比表', fontsize=16, fontweight='bold', pad=30)
    plt.savefig(os.path.join(output_dir, '5_detailed_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 5_detailed_table.png")
    plt.close()
    
    print(f"\n✅ 所有可视化图表已生成到: {output_dir}")

def print_summary(results):
    """打印结果摘要"""
    nomem = results.get('NoMem', {})
    fsnet = results.get('FSNet_Fixed', {})
    
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)
    
    print(f"\n【NoMem - Adapter Only】")
    print(f"  MSE:  {nomem.get('mse', 0):.6f}")
    print(f"  MAE:  {nomem.get('mae', 0):.6f}")
    print(f"  MAPE: {nomem.get('mape', 0):.2f}%")
    print(f"  时间: {nomem.get('total_time', 0):.2f}s")
    
    print(f"\n【FSNet-Fixed - Adapter + Memory (修复后)】")
    print(f"  MSE:  {fsnet.get('mse', 0):.6f}")
    print(f"  MAE:  {fsnet.get('mae', 0):.6f}")
    print(f"  MAPE: {fsnet.get('mape', 0):.2f}%")
    print(f"  时间: {fsnet.get('total_time', 0):.2f}s")
    
    # 计算改进
    mse_imp = (nomem.get('mse', 0) - fsnet.get('mse', 0)) / nomem.get('mse', 1) * 100
    mae_imp = (nomem.get('mae', 0) - fsnet.get('mae', 0)) / nomem.get('mae', 1) * 100
    mape_imp = (nomem.get('mape', 0) - fsnet.get('mape', 0)) / nomem.get('mape', 1) * 100
    
    print(f"\n【改进幅度】")
    print(f"  MSE:  {mse_imp:+.2f}%")
    print(f"  MAE:  {mae_imp:+.2f}%")
    print(f"  MAPE: {mape_imp:+.2f}%")
    
    print(f"\n【核心发现】")
    if mape_imp > 0:
        print(f"  ✅ 修复后的FSNet超越NoMem!")
        print(f"  ✅ Bug修复使Memory机制恢复正常工作")
        print(f"  ✅ 验证了Associative Memory的有效性")
    else:
        print(f"  ⚠️ FSNet仍未超越NoMem")
        print(f"  💡 可能需要更多训练轮次来体现Memory优势")
    
    print("\n" + "="*80)

def main():
    print("\n" + "="*80)
    print("FSNet优化结果可视化")
    print("="*80 + "\n")
    
    # 加载结果
    results = load_results()
    if results is None:
        print("❌ 无法加载实验结果，请先运行 optimization_experiment.py")
        return
    
    # 打印摘要
    print_summary(results)
    
    # 生成可视化图表
    print("\n正在生成可视化图表...")
    create_comparison_plots(results)
    
    print("\n✅ 可视化完成!")

if __name__ == '__main__':
    main()
