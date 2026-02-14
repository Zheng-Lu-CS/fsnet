# FSNet-v3 技术报告：语义对齐架构创新

## 概述

FSNet-v3 是基于对 Advanced (v1) 和 Ultra (v2) 的深度代码审查后设计的第三代优化架构。不同于前两代的"堆叠创新"，v3 的核心思路是**语义一致性**——确保每个组件的数学行为与其设计意图完全对齐。

**核心结果**: MAPE 4.21% → 相对 OGD 基线改进 71.5%

| 版本 | MAPE | MSE | MAE | vs Ultra |
|------|------|-----|-----|----------|
| Advanced (v1) | 5.06% | 0.0131 | 0.0691 | — |
| Ultra (v2) | 4.81% | 0.0085 | 0.0615 | — |
| **v3 (Semantic)** | **4.21%** | **0.0064** | **0.0547** | **MAPE -12.5%, MSE -24.8%** |

---

## 设计动机：8大架构缺陷分析

v3 的每项创新都对应 v1/v2 中的一个具体缺陷。

### 缺陷 1: 温度方向错误 (Advanced)
**现象**: 分布漂移越大 → temp 越高 → softmax 越平滑 → 检索越分散  
**问题**: 漂移大时应该更聚焦地检索最匹配的历史模式，而非取平均  

### 缺陷 2: 校准粒度不足 (Advanced/Ultra)
**现象**: `w = [C_out, K]`，对所有输入通道施加相同的核调制  
**问题**: 不同输入通道携带不同语义，应有独立的校准权重  

### 缺陷 3: q向量随意切分 (Ultra)
**现象**: 将 q 拼接向量等分为 4 个 head，每个 head 跨越 w/b/f 边界  
**问题**: 同一个 head 内部混入了不同类型参数的片段，语义不完整  

### 缺陷 4: 门控过复杂 (Ultra)
**现象**: 2 层 GatedResidualBlock，每层有 2 个 Linear + Gate + Residual  
**问题**: 在线学习只有极少样本更新，过多参数容易过拟合  

### 缺陷 5: "二阶矩"在 OL 下不可靠 (Ultra)
**现象**: `EMA(grad²)` 作为"不确定性信号"输入 controller  
**问题**: OL 每步仅 1 个样本的梯度，平方后方差极大，无法可靠估计  

### 缺陷 6: 正则化不足 (Ultra)
**现象**: Ultra 新增 12K+ 参数，仅靠 Dropout(0.1)  
**问题**: OL 学习窗口极短，需要更强的输出约束  

---

## 6 项语义对齐创新

### 创新 ①: 分解式全张量校准 (Factored Full-Tensor Calibration)

**对应缺陷**: #2 (校准粒度不足)

```
w_cin = calib_cin(h)    → [C_out, C_in]  (通道重要性)
w_k   = calib_k(h)      → [C_out, K]     (核位置调制)
w_full = (1 + w_cin)⊗(1 + w_k) → [C_out, C_in, K]  (外积组合)
```

**关键设计**: 
- 使用 `1 + calib` 残差形式 → 零初始化时等于恒等映射
- 分解参数化 → 参数量 = C_out×C_in + C_out×K (远小于 C_out×C_in×K)
- 每个输出通道对每个输入通道有独立的调制系数

### 创新 ②: 通道分组语义Memory (Channel-Grouped Semantic Memory)

**对应缺陷**: #3 (q向量随意切分)

```
C_out = 64 → 分为 4 组，每组 16 个通道
每组独立 Memory Bank: [group_q_dim, M]
group_q = [w_cin_group, w_k_group, b_group, f_group]
```

**vs Ultra的多头Memory**:
- Ultra: q = [w▪b▪f] → 拼接后强行切4份 → head可能跨越w和b的边界
- v3: 按C_out通道自然分组 → 每组的q包含该组通道的完整校准信息
- 语义完整性: 每个Memory头只负责自己通道组的模式记忆

### 创新 ③: 反向温度缩放 (Inverse Temperature Scaling)

**对应缺陷**: #1 (温度方向错误)

```
temp = base_temp / (1 + shift_magnitude)
```

- 漂移大 → temp 低 → softmax 更尖锐 → 检索更聚焦
- 漂移小 → temp 高 → softmax 更平滑 → 检索更平均
- 这与直觉一致: 异常时应精确匹配历史模式

### 创新 ④: 梯度范数条件门控 (Gradient-Norm Conditioned Gate)

**对应缺陷**: #4 (门控过复杂) + #5 (二阶矩不可靠)

```
h = controller(grads)             # [C_out, nh]
grad_norm = ||grads||₂ per C_out  # [C_out, 1]  ← 标量信号
z = σ(W_gate · [h, grad_norm])    # [C_out, nh]  单层门

out = (1-z) · h + z · transform(h)
```

**设计理由**:
- 用梯度范数(标量)替代梯度二阶矩(向量) → 鲁棒、轻量
- 单层门替代 2 层 GatedResidual → 参数减半
- grad_norm 大 → 不确定性高 → 门趋向保守

### 创新 ⑤: tanh 限幅 + 残差初始化 (Output Clamping + Residual Init)

**对应缺陷**: #6 (正则化不足)

```python
w_cin = tanh(calib_cin(h)) * scale  # 输出范围 [-scale, scale]
w_k   = tanh(calib_k(h)) * scale
b_adj = tanh(calib_b(h)) * scale
f_adj = tanh(calib_f(h)) * scale
```

- tanh 确保校准输出有界，不会爆炸
- 所有校准头零初始化 → 初始状态 = 不做任何校准 (恒等)
- 较大的校准值需要更强的梯度信号来达到 → 自然正则化

### 创新 ⑥: 单步 EMA 融合 (Single-Step EMA Blend)

**对应缺陷**: Advanced 的两步 EMA 冗余

```python
# Update EMA buffer (no grad)
q_ema = γ * q_ema + (1-γ) * q.detach()
# Blend (gradient flows through q)
q = α * q + (1-α) * q_ema.detach()
```

- 一步更新 + 一步融合，结构清晰
- 梯度只通过当前 q 流动，不会传播到历史 EMA

---

## 架构参数对比

| 参数 | Advanced | Ultra | v3 |
|------|----------|-------|----|
| nh (隐藏维度) | 64 | 96 | 64 |
| Memory 槽数 M | 32 | 64 | 32 |
| Memory 结构 | 单头 | 4头(切q) | 4组(切通道) |
| Top-k 检索 | 3 | 3 | 3 |
| Controller 层数 | 1 | 2(含门控) | 1+门 |
| 梯度输入维度 | C_in×K | 2×C_in×K | C_in×K+1 |
| 校准参数 | [C_out,K] | [C_out,K] | [C_out,C_in]+[C_out,K] |
| 正则化 | Dropout | Dropout | tanh+Dropout+zero-init |

---

## 实验结果

### 8模型完整排行

| # | Model | MAPE(%) | MSE | MAE | vs OGD |
|---|-------|---------|-----|-----|--------|
| 1 | **FSNet-v3** | **4.21** | **0.0064** | **0.0547** | **-71.5%** |
| 2 | FSNet-Ultra | 4.81 | 0.0085 | 0.0615 | -67.4% |
| 3 | FSNet-Advanced | 5.06 | 0.0131 | 0.0691 | -65.7% |
| 4 | NoMem | 9.73 | 0.0319 | 0.1226 | -34.0% |
| 5 | FSNet (Bug) | 10.07 | 0.0387 | 0.1310 | -31.7% |
| 6 | FSNet-Fixed | 10.25 | 0.0390 | 0.1291 | -30.5% |
| 7 | ER | 13.53 | 0.0531 | 0.1661 | -8.3% |
| 8 | OGD | 14.75 | 0.0635 | 0.1766 | — |

### v3 vs Ultra 逐项改进

| 指标 | Ultra | v3 | 相对改进 |
|------|-------|----|---------|
| MAPE | 4.81% | 4.21% | -12.5% |
| MSE | 0.0085 | 0.0064 | -24.8% |
| MAE | 0.0615 | 0.0547 | -11.2% |
| vali_loss | 0.1089 | 0.0934 | -14.2% |

### 关键洞察

1. **语义一致性 > 参数堆叠**: v3 参数量比 Ultra 更少，但性能更好
2. **全张量校准是最大贡献**: `[C_out,C_in,K]` 让每个参数都可独立调节
3. **通道分组Memory消除了语义断裂**: 检索结果总是语义完整的校准模式
4. **less is more**: 简化的单层门控比复杂的双层门控更适合OL场景

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `models/ts2vec/fsnet_v3.py` | v3 模型实现 (~320行) |
| `train_v3.py` | v3 训练+测试脚本 |
| `visualize_v3.py` | v3 可视化 (8张图) |
| `ANSWER.md` | 8大疑问深度分析 |

---

*Report generated after FSNet-v3 achieving MAPE=4.21% on ETTh1 (features='S', 2 epochs)*
