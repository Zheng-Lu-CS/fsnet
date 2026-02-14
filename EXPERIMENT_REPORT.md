# FSNet项目实验报告

**学生**: [你的名字]  
**日期**: 2026年2月12日  
**项目**: Fast & Slow Learning for Online Time Series Forecasting  

---

## 📌 项目概述

本项目复现并分析了FSNet模型——一个结合快速适应（Adapter）和慢速学习（Associative Memory）的在线时序预测框架。通过消融实验验证了各组件的有效性。

---

## 🎯 实验目标

1. 理解FSNet核心机制（Adapter + Memory）
2. 完成消融实验，验证各组件贡献
3. 可视化分析性能差异
4. 发现模型优化方向

---

## 📊 实验设置

### 数据集
- **名称**: ETTh1 (Electricity Transformer Temperature)
- **模式**: 单变量 (S)
- **序列长度**: 48步输入 → 12步预测
- **特征**: OT (Oil Temperature)

### 模型配置
- **训练轮数**: 2 epochs（快速验证）
- **批大小**: 8
- **学习率**: 0.0001
- **优化器**: AdamW
- **设备**: CPU

### 对比方法
1. **OGD**: 标准在线梯度下降（Baseline）
2. **ER**: 经验回放 (Experience Replay)
3. **NoMem**: FSNet无Memory（只有Adapter）
4. **FSNet**: 完整模型（Adapter + Memory）

---

## 📈 实验结果

### 性能指标对比

| 方法 | MAPE (%)↓ | MSE↓ | MAE↓ | RMSE↓ |
|------|-----------|------|------|-------|
| OGD | 14.75 | 0.0635 | 0.177 | 0.205 |
| ER | 13.53 | 0.0531 | 0.166 | 0.194 |
| **NoMem** | **9.73** | **0.0319** | **0.123** | **0.145** |
| FSNet | 10.07 | 0.0387 | 0.131 | 0.156 |

**评级标准**: MAPE < 10% = 优秀(A+), 10-20% = 良好(A)

### 改进百分比（相比OGD Baseline）

| 方法 | MAPE改进 | MSE改进 | MAE改进 |
|------|----------|---------|---------|
| ER | +8.3% | +16.3% | +6.1% |
| NoMem | **+34.0%** | **+49.8%** | **+30.4%** |
| FSNet | +31.7% | +39.0% | +25.8% |

### 计算时间

| 方法 | 训练时间 | 测试时间 | 总时间 |
|------|---------|---------|--------|
| OGD | 42s | 271s | 313s |
| ER | 48s | 1147s | 1195s |
| NoMem | 124s | 984s | 1108s |
| FSNet | 101s | 1703s | 1804s |

---

## 🔍 关键发现

### 发现1: Adapter机制贡献巨大 ⭐⭐⭐

**现象**: NoMem相比OGD，所有指标都有显著提升：
- MAPE: 14.75% → 9.73% (改进34%)
- MSE: 0.0635 → 0.0319 (改进50%)

**解释**: Adapter通过梯度信息动态调整卷积参数，实现了快速适应新数据分布的能力。

**代码位置**: `models/ts2vec/fsnet_.py` 第85-130行

### 发现2: Memory的作用需要进一步研究 ⭐⭐

**现象**: 完整FSNet的性能略低于NoMem：
- MAPE: 10.07% vs 9.73%
- MSE: 0.0387 vs 0.0319

**可能原因**:
1. **训练不充分**: 只有2 epochs，Memory需要更长时间积累有效模式
2. **数据集特点**: ETTh1单变量数据较简单，Adapter已足够
3. **超参数未调优**: Memory的top-k=2、融合权重τ=0.75可能不optimal

**改进方向**:
- 增加训练轮数到5-10 epochs
- 调优Memory超参数（k, τ）
- 在更复杂数据集上实验（多变量、Weather、Traffic）

### 发现3: 所有方法都超越Baseline ⭐

**现象**: 即使是最弱的ER，也比OGD提升了8.3%

**意义**: 验证了在线学习和快速适应机制对时序预测的重要性

---

## 📉 可视化分析

生成了6张高质量图表（见 `figures/` 目录）：

1. **预测曲线对比**: 直观展示各方法的预测精度
2. **性能指标柱状图**: MSE/MAE/RMSE/MAPE全方位对比
3. **改进雷达图**: 相比Baseline的提升幅度
4. **误差分布箱线图**: 各方法的预测稳定性
5. **时间对比**: 训练/测试效率分析
6. **归一化对比**: 统一标准下的综合性能

**关键观察**:
- NoMem的预测曲线最平滑，峰值捕捉最准确
- FSNet的误差方差比NoMem略大（可能因Memory引入noise）
- ER的测试时间最长（buffer检索开销）

---

## 💡 核心技术理解

### Adapter机制（快速适应）

**原理**:
```
梯度信息 → MLP控制器 → 生成校准参数[w, b, f]
动态卷积 = f × Conv(x, weight×w, bias×b)
```

**作用**: 根据当前任务难度动态调整模型参数，实现快速适应

### Associative Memory（慢速积累）

**原理**:
```
查询向量q = concat([w, b, f])
记忆检索: q @ MemoryMatrix → top-k相似模式
融合: new_params = τ×adaptive + (1-τ)×memory
```

**作用**: 存储和检索历史模式，利用长期知识

---

## 🚀 后续优化方向

### 短期（1周内）

1. **增加训练轮数**: 2 epochs → 5-10 epochs
2. **超参数搜索**: 网格搜索Memory的k和τ
3. **GPU加速**: 修改`device_config.py`启用CUDA，速度提升10-50x

### 中期（1个月内）

1. **多数据集实验**: Weather, Traffic, ECL
2. **多变量模式**: ETTh1 (M) 7个特征全用
3. **架构优化**: 
   - 多头注意力记忆（已实现代码在 `models/improvements/`）
   - 动态Adapter（根据任务难度调整校准强度）

### 长期（研究方向）

1. **理论分析**: 为什么Memory在某些情况下反而降低性能？
2. **自适应机制**: 自动决定何时使用Memory
3. **可解释性**: 可视化Adapter的校准过程和Memory的检索模式

---

## 📁 项目成果

### 代码文件

```
fsnet/
├── myexp.py                    # 快速实验入口
├── ablation_study.py           # 消融实验脚本
├── visualize_results.py        # 可视化分析
├── architecture_optimization.py # 架构优化
├── CODE_STRUCTURE.md           # 代码结构文档
├── exp/                        # 4个实验类（OGD, ER, NoMem, FSNet）
├── models/ts2vec/              # 核心模型实现
│   ├── fsnet.py               # TSEncoder主模型
│   └── fsnet_.py              # Adapter+Memory核心（必读！）
└── models/improvements/        # 优化模块（多头注意力等）
```

### 实验结果

```
results/ablation/
├── ablation_results.json       # 性能指标
└── ablation_predictions.npz    # 预测数组（用于可视化）

figures/
├── 1_prediction_curves.png     # 预测曲线对比
├── 2_metrics_comparison.png    # 性能指标柱状图
├── 3_improvement_radar.png     # 改进百分比雷达图
├── 4_error_distribution.png    # 误差分布箱线图
├── 5_time_comparison.png       # 计算时间对比
└── 6_normalized_comparison.png # 归一化性能对比
```

### 文档

- **CODE_STRUCTURE.md**: 代码结构速通指南（30分钟理解全项目）
- **FIXES_SUMMARY.md**: 调试过程中遇到的5个关键bug及解决方案
- **ACTION_PLAN.md**: 4小时项目攻略（分钟级执行清单）
- **README_CN.md**: 完整的中文项目说明

---

## 🎓 学习收获

1. **技术能力**:
   - 掌握了在线学习和快速适应机制
   - 理解了Meta-Learning中的Adapter设计
   - 学会了消融实验的设计和分析方法

2. **工程能力**:
   - 调试大型深度学习项目（解决5个关键bug）
   - 模块化代码设计（统一设备管理、配置管理）
   - 科研级可视化（matplotlib、论文级图表）

3. **科研思维**:
   - 提出"为什么Memory反而降低性能"的研究问题
   - 设计对照实验验证假设
   - 批判性思考现有方法的局限性

---

## 📝 总结

本项目成功复现了FSNet模型，并通过消融实验发现了一个有趣的现象：**Adapter机制单独使用已经能取得很好的效果（MAPE 9.73%），而Memory的加入反而略微降低了性能**。这一发现为后续研究提供了方向：

1. 需要更长的训练时间让Memory积累有效模式
2. 需要在更复杂的数据集上验证Memory的价值
3. 需要优化Memory的超参数和检索机制

整体而言，项目达到了预期目标：
- ✅ 理解了FSNet核心原理
- ✅ 完成了消融实验和可视化
- ✅ 发现了模型优化方向
- ✅ 具备了向GitHub和论文投稿的基础

---

## 📚 参考资料

1. **原始论文**: [Learning Fast and Slow for Online Time Series Forecasting](https://arxiv.org/abs/2202.11672)
2. **数据集**: [ETDataset](https://github.com/zhouhaoyi/ETDataset)
3. **代码仓库**: [本项目GitHub链接]

---

**报告完成时间**: 2026年2月12日  
**实验总耗时**: ~4小时（消融实验1h + 其他3h）  
**代码行数**: ~3000行（核心模型）+ ~1000行（实验脚本）
