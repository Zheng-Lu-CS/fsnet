"""
FSNet-Ultra: 第二轮深度创新架构 — 冲击SOTA

相对 FSNet-Advanced 的 6 项全新创新：
  ① Multi-Head Memory Attention  — 4头并行记忆检索
  ② Gated Residual Controller    — 深层门控残差controller
  ③ Memory Bank Expansion (M=64) — 2倍记忆容量
  ④ Gradient 2nd-Moment Tracking — 一阶+二阶梯度统计
  ⑤ Adaptive Trigger Threshold   — 可学习的漂移检测阈
  ⑥ Memory Diversity Penalty     — 软正交正则化防止槽坍缩

与 FSNet-Advanced 创新对比:
  Advanced: 结构对齐chunk | 自适应tau | Top-3检索 | Dropout
  Ultra:    多头Memory   | 门控Controller | 2倍记忆 | 梯度二阶矩 | 自适应阈值 | 多样性惩罚
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


def normalize_col(W):
    """按列归一化"""
    return F.normalize(W, dim=0)


# ===========================================================================
#  Ultra 创新组件
# ===========================================================================

class GatedResidualBlock(nn.Module):
    """创新②: 门控残差块 — 比简单 Linear+SiLU 更强的非线性建模
    
    Gate 机制类似 GRU：
        z = σ(Wz·x + Uz·h)    # update gate
        h̃ = SiLU(Wr·(z⊙h) + Wx·x)
        h_out = (1-z)·h + z·h̃
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """x: [..., d_model]"""
        h = F.silu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        # gate
        z = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
        out = (1 - z) * x + z * h
        return self.norm(out)


class MultiHeadMemoryAttention(nn.Module):
    """创新①: 多头记忆注意力
    
    将 q 向量拆成 H 个子空间，每个头独立检索 Memory Bank，
    最终 concat 并投影回原空间。类似 Multi-Head Attention，
    但 K=V=Memory Bank (固定/缓慢更新)。
    
    优势：
    - 不同头关注不同粒度/方面的历史模式
    - 增加记忆检索的多样性
    - 防止单一注意力退化
    """
    def __init__(self, q_dim, M=64, n_heads=4, top_k=3):
        super().__init__()
        assert q_dim % n_heads == 0, f"q_dim {q_dim} must be divisible by n_heads {n_heads}"
        self.q_dim = q_dim
        self.n_heads = n_heads
        self.head_dim = q_dim // n_heads
        self.M = M
        self.top_k = top_k
        
        # 每个头拥有独立的 Memory Bank
        # W_heads: [n_heads, head_dim, M]
        self.W = nn.Parameter(torch.empty(n_heads, self.head_dim, M), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data.view(-1, M))
        self.W.data = normalize_col(self.W.data)
        
        # 动态温度 (每个头独立)
        self.temp_proj = nn.Linear(q_dim, n_heads)
        
        # 输出投影
        self.out_proj = nn.Linear(q_dim, q_dim)
    
    def retrieve(self, q, tau_adaptive):
        """
        q: [q_dim]  — detached
        tau_adaptive: scalar
        returns: blended [q_dim]
        """
        q_heads = q.view(self.n_heads, self.head_dim) ? # [H, d_head]
        
        # 动态温度: [H] in [0.2, 0.8]
        temps = torch.sigmoid(self.temp_proj(q)) * 0.6 + 0.2  # [H]
        
        retrieved_heads = []
        topk_indices_all = []
        
        for h in range(self.n_heads):
            # 单头注意力
            att = q_heads[h] @ self.W[h]  # [M]
            att = F.softmax(att / (temps[h] + 1e-8), dim=0)
            
            k = min(self.top_k, self.M)
            v, idx = torch.topk(att, k)
            topk_indices_all.append((h, idx, v))
            
            # 加权检索
            retrieved = (self.W[h, :, idx] * v).sum(dim=1)  # [d_head]
            retrieved_heads.append(retrieved)
        
        # concat + project
        retrieved_cat = torch.cat(retrieved_heads, dim=0)  # [q_dim]
        retrieved_out = self.out_proj(retrieved_cat)
        
        return retrieved_out, topk_indices_all
    
    def write(self, q_det, topk_indices_all, tau):
        """写回记忆 (无梯度)"""
        q_heads = q_det.view(self.n_heads, self.head_dim)
        with torch.no_grad():
            for h, idx, v in topk_indices_all:
                for j in idx:
                    self.W.data[h, :, j] = F.normalize(
                        tau * self.W.data[h, :, j] + (1 - tau) * q_heads[h],?
                        dim=0
                    )
    
    def diversity_penalty(self):
        """创新⑥: 记忆多样性惩罚 — 鼓励不同槽存储不同模式
        
        计算 Memory Bank 内部的平均余弦相似度，作为正则项。
        相似度越高 → penalty越大 → 推动Memory多样化。
        """
        penalty = 0.0
        for h in range(self.n_heads):
            # [d_head, M] → 列归一化
            W_norm = F.normalize(self.W[h].detach(), dim=0)  # [d_head, M]
            # 槽间余弦相似度矩阵
            sim = W_norm.T @ W_norm  # [M, M]
            # 取上三角 (排除对角线)
            mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
            penalty += sim[mask].mean()
        return penalty / self.n_heads


# ===========================================================================
#  SamePadConvUltra — 核心卷积层
# ===========================================================================

class SamePadConvUltra(nn.Module):
    """
    FSNet-Ultra 核心卷积层
    
    6项创新集成：
    ① MultiHeadMemoryAttention (4头)
    ② GatedResidualController (2层门控)
    ③ Memory M=64
    ④ 梯度一阶+二阶矩统计
    ⑤ 自适应漂移触发阈值
    ⑥ Memory多样性惩罚
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, 
                 groups=1, gamma=0.9, n_heads=4, M=64, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        
        # --- 基础卷积 ---
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = out_channels * in_channels * kernel_size
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
        # --- 结构对齐Chunk (沿用Advanced理念) ---
        self.n_chunks = out_channels
        self.chunk_in_d = in_channels * kernel_size
        
        # --- 创新④: 梯度二阶矩追踪 ---
        self.register_buffer('grads', torch.zeros(self.dim))       # EMA一阶矩
        self.register_buffer('grads_sq', torch.zeros(self.dim))    # EMA二阶矩 (新增!)
        self.register_buffer('f_grads', torch.zeros(self.dim))  ?   # 快速梯度EMA
        
        # --- 创新②: 门控残差Controller (2层) ---
        nh = 96  # 增大隐藏维度 (64→96)
        # 输入: chunk_in_d + chunk_in_d (二阶矩拼接)
        self.input_proj = nn.Linear(self.chunk_in_d * 2, nh)  # 同时接收一阶+二阶
        self.gated_block1 = GatedResidualBlock(nh, dropout=0.1)
        self.gated_block2 = GatedResidualBlock(nh, dropout=0.1)
        
        # 校准头
        self.calib_w = nn.Linear(nh, kernel_size)
        self.calib_b = nn.Linear(nh, 1)
        self.calib_f = nn.Linear(nh, 1)
        
        # 自适应融合系数
        self.tau_net = nn.Sequential(
            nn.Linear(nh, 1),
            nn.Sigmoid()
        )
        
        # --- 创新①: 多头记忆注意力 ---
        q_dim = out_channels * (kernel_size + 2)
        # 确保 q_dim 能被 n_heads 整除
        self.n_heads = n_heads
        # 如果不整除，pad到最近的倍数
        self.q_dim_padded = ((q_dim + n_heads - 1) // n_heads) * n_heads
        self.q_dim_raw = q_dim
        if self.q_dim_padded != q_dim:
            self.q_pad = nn.Linear(q_dim, self.q_dim_padded)
            self.q_unpad = nn.Linear(self.q_dim_padded, q_dim)
        else:
            self.q_pad = None
            self.q_unpad = None
        
        self.memory = MultiHeadMemoryAttention(
            q_dim=self.q_dim_padded, M=M, n_heads=n_heads, top_k=3
        )
        self.register_buffer('q_ema', torch.zeros(q_dim))
        
        # --- 创新⑤: 自适应触发阈值 ---
        self.register_buffer('trigger_threshold', torch.tensor(-0.75))
        self.trigger_adapt_rate = 0.01  # 阈值调整速率
        
        # 触发器
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.sq_gamma = 0.99  # 二阶矩衰减更慢
        
    def ctrl_params(self):
        """所有可训练Controller参数"""
        modules = [
            self.input_proj, self.gated_block1, self.gated_block2,
            self.calib_w, self.calib_b, self.calib_f, self.tau_net,
            self.memory.temp_proj, self.memory.out_proj
        ]
        if self.q_pad is not None:
            modules.extend([self.q_pad, self.q_unpad])
        for m in modules:
            for p in m.parameters():
                yield p
    
    def store_grad(self):
        """存储梯度 + 检测分布漂移 (含自适应阈值)"""
        grad = self.conv.weight.grad.data.clone()
        grad_flat = F.normalize(grad.reshape(-1), dim=0)
        
        # 创新④: 同时更新一阶矩和二阶矩
        self.f_grads.mul_(self.f_gamma).add_(grad_flat, alpha=1 - self.f_gamma)
        self.grads_sq.mul_(self.sq_gamma).add_(grad_flat.pow(2), alpha=1 - self.sq_gamma)
        
        if not self.training:
            e = self.cos(self.f_grads, self.grads)
            
            # 创新⑤: 使用自适应阈值
            if e < self.trigger_threshold:
                self.trigger = 1
            
            # 自适应调整阈值: 如果频繁触发则提高阈值，反之降低
            with torch.no_grad():
                if self.trigger == 1:
                    # 触发了 → 稍微提高阈值 (更难触发)，防止过度频繁
                    self.trigger_threshold.add_(self.trigger_adapt_rate)
                    self.trigger_threshold.clamp_(max=-0.3)
                else:
                    # 未触发 → 稍微降低阈值 (更容易触发)
                    self.trigger_threshold.sub_(self.trigger_adapt_rate * 0.1)
                    self.trigger_threshold.clamp_(min=-0.95)
        
        self.grads.mul_(self.gamma).add_(grad_flat, alpha=1 - self.gamma)
    
    def fw_chunks(self):
        """前向: 梯度 → Controller → 校准参数 (含Memory检索)"""
        # 重组梯度为通道视角
        g1 = self.grads.view(self.out_channels, self.in_channels, self.kernel_size)
        g1 = g1.view(self.out_channels, -1)  # [C_out, C_in*K]
        
        # 创新④: 二阶矩也reshape
        g2 = self.grads_sq.view(self.out_channels, self.in_channels, self.kernel_size)
        g2 = g2.view(self.out_channels, -1)  # [C_out, C_in*K]
        
        # 拼接一阶+二阶信息作为Controller输入
        g_cat = torch.cat([g1, g2], dim=-1)  # [C_out, 2*C_in*K]
        
        # 创新②: 门控残差Controller
        h = F.silu(self.input_proj(g_cat))   # [C_out, nh]
        h = self.gated_block1(h)             # [C_out, nh]
        h = self.gated_block2(h)             # [C_out, nh]

        # 校准参数
        w = self.calib_w(h)             # [C_out, K]
        b = self.calib_b(h).squeeze(-1) # [C_out]
        f = self.calib_f(h).squeeze(-1) # [C_out]
        
        # 自适应tau
        tau_adaptive = self.tau_net(h).mean() * 0.5 + 0.5  # [0.5, 1.0]
        
        # 构建q向量
        q = torch.cat([w.reshape(-1), b, f], dim=0)  # [q_dim_raw]
        
        # EMA (detached)
        with torch.no_grad():
            self.q_ema.mul_(self.f_gamma).add_(q.detach(), alpha=1 - self.f_gamma)
        q = 0.5 * q + 0.5 * self.q_ema.detach()
        
        # --- 创新①: 多头Memory检索 ---
        if self.trigger == 1:
            self.trigger = 0
            q_det = q.detach()
            
            # pad if needed
            if self.q_pad is not None:
                q_mem = self.q_pad(q_det)
            else:
                q_mem = q_det
            
            # 多头检索
            retrieved, topk_info = self.memory.retrieve(q_mem, tau_adaptive.item())
            
            # unpad
            if self.q_unpad is not None:
                retrieved = self.q_unpad(retrieved)
            
            # 用自适应系数融合
            q = tau_adaptive * q + (1 - tau_adaptive) * retrieved.detach()
            
            # 写回Memory
            if self.q_pad is not None:
                self.memory.write(self.q_pad(q_det), topk_info, tau_adaptive.item())
            else:
                self.memory.write(q_det, topk_info, tau_adaptive.item())
        
        # 还原 w, b, f
        w_size = self.out_channels * self.kernel_size
        w = q[:w_size].view(self.out_channels, self.kernel_size)
        b = q[w_size:w_size + self.out_channels]
        f = q[w_size + self.out_channels:]
        
        w = w.unsqueeze(1)             # [C_out, 1, K]
        f = f.unsqueeze(0).unsqueeze(-1)  # [1, C_out, 1]
        return w, b, f
    
    def forward(self, x):
        w, b, f = self.fw_chunks()
        cw = self.conv.weight * w
        out = F.conv1d(x, cw, padding=self.padding,
                       dilation=self.dilation, bias=self.bias * b)
        out = f * out
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out
    
    def representation(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


# ===========================================================================
#  Ultra ConvBlock & Encoder
# ===========================================================================

class ConvBlockUltra(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False, gamma=0.9, n_heads=4, M=64, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.conv1 = SamePadConvUltra(
            in_channels, out_channels, kernel_size,
            dilation=dilation, gamma=gamma, n_heads=n_heads, M=M, device=self.device
        )
        self.conv2 = SamePadConvUltra(
            out_channels, out_channels, kernel_size,
            dilation=dilation, gamma=gamma, n_heads=n_heads, M=M, device=self.device
        )
        self.projector = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels or final else None
    
    def ctrl_params(self):
        return chain(self.conv1.ctrl_params(), self.conv2.ctrl_params())
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoderUltra(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9,
                 n_heads=4, M=64, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.net = nn.Sequential(*[
            ConvBlockUltra(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1),
                gamma=gamma, n_heads=n_heads, M=M, device=self.device
            )
            for i in range(len(channels))
        ])
    
    def ctrl_params(self):
        ctrl = [layer.ctrl_params() for layer in self.net]
        for p in chain(*ctrl):
            yield p
    
    def forward(self, x):
        return self.net(x)
    
    def diversity_loss(self):
        """聚合所有层的Memory多样性惩罚"""
        total = 0.0
        count = 0
        for block in self.net:
            for conv in [block.conv1, block.conv2]:
                total += conv.memory.diversity_penalty()
                count += 1
        return total / max(count, 1)
