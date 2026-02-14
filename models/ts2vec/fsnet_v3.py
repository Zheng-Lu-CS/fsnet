"""
FSNet-v3: 语义对齐架构 — Semantically-Aligned Calibration

基于对 Advanced/Ultra 的深度审查，针对性修复 6 个架构缺陷：

  ① 分解式全参数调制 — w=[C_out,C_in,K] (修复: 通道无差别调制)
  ② 通道分组语义Memory — 每组通道独立Memory Bank (修复: 多头语义割裂)
  ③ 反向温度缩放 — 漂移越大,检索越聚焦 (修复: temp方向)
  ④ 梯度范数条件化Gate — 1层门控+grad_norm信号 (修复: 二阶矩噪声+过复杂)
  ⑤ 校准输出范围约束 — tanh限幅,残差初始化 (修复: 正则不足)
  ⑥ 简洁EMA — 单步blend (修复: 冗余两步)

设计原则: 语义对齐 + 适度复杂度 + 理论可解释
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


def normalize_col(W):
    return F.normalize(W, dim=0)


# ===========================================================================
#  v3 核心组件
# ===========================================================================

class ChannelGroupedMemory(nn.Module):
    """创新②: 通道分组语义Memory
    
    不切割 q 向量, 而是按输出通道分组。
    每组拥有独立 Memory Bank, 存储/检索该组通道的完整校准模式。
    
    vs Ultra多头: Ultra按拼接顺序切q → 语义割裂
    vs v3分组:   按通道分组, 每组q包含完整的 [w_cin, w_k, b, f] → 语义完整
    """
    def __init__(self, n_groups, group_q_dim, M=32, top_k=3):
        super().__init__()
        self.n_groups = n_groups
        self.group_q_dim = group_q_dim
        self.M = M
        self.top_k = top_k
        
        # 每组独立 Memory Bank: [n_groups, group_q_dim, M]
        self.W = nn.Parameter(
            torch.empty(n_groups, group_q_dim, M), requires_grad=False
        )
        nn.init.xavier_uniform_(self.W.data.view(-1, M))
        # 按 q_dim 维度归一化 (每个槽是单位向量)
        for g in range(n_groups):
            self.W.data[g] = normalize_col(self.W.data[g])
        
        # 创新③: 每组可学习温度基线 (logit空间)
        self.temp_logit = nn.Parameter(torch.zeros(n_groups))
    
    def retrieve(self, q_groups, shift_magnitude=0.0):
        """
        q_groups: [n_groups, group_q_dim] — detached
        shift_magnitude: float ≥ 0, 漂移强度
        returns: retrieved [n_groups, group_q_dim], topk_info
        """
        retrieved_list = []
        topk_info = []
        
        for g in range(self.n_groups):
            q = q_groups[g]  # [group_q_dim]
            
            # 创新③: 反向温度 — 漂移越大 temp 越低, 检索越聚焦
            base_temp = torch.sigmoid(self.temp_logit[g]) * 0.6 + 0.2  # [0.2, 0.8]
            temp = base_temp / (1.0 + shift_magnitude)  # shift大 → temp低 → 更尖锐
            
            # 注意力
            att = q @ self.W[g]  # [M]
            att = F.softmax(att / (temp + 1e-8), dim=0)
            
            k = min(self.top_k, self.M)
            v, idx = torch.topk(att, k)
            topk_info.append((g, idx, v))
            
            # 加权检索
            retrieved = (self.W[g, :, idx] * v).sum(dim=1)  # [group_q_dim]
            retrieved_list.append(retrieved)
        
        return torch.stack(retrieved_list), topk_info  # [n_groups, group_q_dim]
    
    def write(self, q_groups_det, topk_info, tau):
        """写回Memory (无梯度)"""
        with torch.no_grad():
            for g, idx, v in topk_info:
                for j in idx:
                    self.W.data[g, :, j] = F.normalize(
                        tau * self.W.data[g, :, j] + (1 - tau) * q_groups_det[g],
                        dim=0
                    )


# ===========================================================================
#  SamePadConvV3 — 核心卷积层
# ===========================================================================

class SamePadConvV3(nn.Module):
    """
    FSNet-v3 语义对齐卷积层
    
    6项针对性改进:
    ① 分解式全参数调制 w = (1+w_cin)⊗(1+w_k) → [C_out, C_in, K]
    ② 通道分组Memory (4组, 每组完整语义)
    ③ 反向温度 (漂移大→temp低→聚焦)
    ④ Gradient-norm条件化单层Gate
    ⑤ tanh限幅 + 残差初始化
    ⑥ 单步EMA
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, gamma=0.9, n_groups=4, M=32, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        
        # --- 基础卷积 ---
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = out_channels * in_channels * kernel_size
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
        # --- 结构对齐 Chunk (按 out_channels) ---
        self.n_chunks = out_channels
        self.chunk_in_d = in_channels * kernel_size
        
        # --- 梯度缓存 ---
        self.register_buffer('grads', torch.zeros(self.dim))
        self.register_buffer('f_grads', torch.zeros(self.dim))
        
        # --- 创新④: Gradient-norm 条件化单层 Gate ---
        nh = 64
        self.controller = nn.Sequential(
            nn.Linear(self.chunk_in_d, nh),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        # Gate: 输入 [nh + 1], 其中 +1 是梯度范数
        self.gate_proj = nn.Linear(nh + 1, nh)
        self.gate_transform = nn.Sequential(
            nn.Linear(nh, nh),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.gate_norm = nn.LayerNorm(nh)
        
        # --- 创新①: 分解式全参数校准头 ---
        self.calib_k = nn.Linear(nh, kernel_size)      # [C_out, K]
        self.calib_cin = nn.Linear(nh, in_channels)     # [C_out, C_in] — 新增!
        self.calib_b = nn.Linear(nh, 1)                 # [C_out, 1]
        self.calib_f = nn.Linear(nh, 1)                 # [C_out, 1]
        
        # 创新⑤: 校准头零初始化 (初始校准=0 → w=1+0=1, 恒等映射)
        nn.init.zeros_(self.calib_k.weight)
        nn.init.zeros_(self.calib_k.bias)
        nn.init.zeros_(self.calib_cin.weight)
        nn.init.zeros_(self.calib_cin.bias)
        nn.init.zeros_(self.calib_b.weight)
        nn.init.zeros_(self.calib_b.bias)
        nn.init.zeros_(self.calib_f.weight)
        nn.init.zeros_(self.calib_f.bias)
        
        # 自适应融合系数 tau
        self.tau_net = nn.Sequential(
            nn.Linear(nh, 1),
            nn.Sigmoid()
        )
        
        # --- 创新②: 通道分组Memory ---
        # 确保 C_out 能被 n_groups 整除
        self.n_groups = min(n_groups, out_channels)
        while out_channels % self.n_groups != 0:
            self.n_groups -= 1
        self.channels_per_group = out_channels // self.n_groups
        
        # 每组的q包含: w_cin[cpg, C_in] + w_k[cpg, K] + b[cpg] + f[cpg]
        cpg = self.channels_per_group
        self.group_q_dim = cpg * (in_channels + kernel_size + 2)
        
        self.memory = ChannelGroupedMemory(
            n_groups=self.n_groups,
            group_q_dim=self.group_q_dim,
            M=M, top_k=3
        )
        
        # 全局 q 的 EMA
        total_q_dim = out_channels * (in_channels + kernel_size + 2)
        self.register_buffer('q_ema', torch.zeros(total_q_dim))
        self.total_q_dim = total_q_dim
        
        # --- 触发器 (含自适应阈值) ---
        self.register_buffer('trigger_threshold', torch.tensor(-0.75))
        self.trigger_adapt_rate = 0.01
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.shift_magnitude = 0.0  # 记录当前漂移强度
        self.gamma = gamma
        self.f_gamma = 0.3
        
        # 创新⑤: 校准范围参数
        self.calib_scale = 2.0  # tanh * scale, 限制在 [-2, 2]
    
    def ctrl_params(self):
        """可训练的Controller参数"""
        modules = [
            self.controller[0],  # Linear inside Sequential
            self.gate_proj, self.gate_transform[0], self.gate_norm,
            self.calib_k, self.calib_cin, self.calib_b, self.calib_f,
            self.tau_net,
        ]
        for m in modules:
            for p in m.parameters():
                yield p
        # Memory的可学习温度
        yield self.memory.temp_logit
    
    def store_grad(self):
        """存储梯度 + 检测漂移 (含自适应阈值)"""
        grad = self.conv.weight.grad.data.clone()
        grad_flat = F.normalize(grad.reshape(-1), dim=0)
        
        self.f_grads.mul_(self.f_gamma).add_(grad_flat, alpha=1 - self.f_gamma)
        
        if not self.training:
            e = self.cos(self.f_grads, self.grads)
            
            if e < self.trigger_threshold:
                self.trigger = 1
                self.shift_magnitude = abs(e.item())  # 记录漂移强度
            
            # 自适应调整阈值
            with torch.no_grad():
                if self.trigger == 1:
                    self.trigger_threshold.add_(self.trigger_adapt_rate)
                    self.trigger_threshold.clamp_(max=-0.3)
                else:
                    self.trigger_threshold.sub_(self.trigger_adapt_rate * 0.1)
                    self.trigger_threshold.clamp_(min=-0.95)
        
        self.grads.mul_(self.gamma).add_(grad_flat, alpha=1 - self.gamma)
    
    def fw_chunks(self):
        """前向: 梯度 → Gate → 校准参数 → Memory"""
        # === 结构对齐: 按输出通道重组梯度 ===
        g = self.grads.view(self.out_channels, self.in_channels, self.kernel_size)
        g = g.view(self.out_channels, -1)  # [C_out, C_in*K]
        
        # === Controller ===
        h = self.controller(g)  # [C_out, nh]
        
        # === 创新④: Gradient-norm 条件化 Gate ===
        # 每个输出通道的梯度范数 (不确定性信号)
        grad_norm = g.norm(dim=-1, keepdim=True)  # [C_out, 1]
        gate_input = torch.cat([h, grad_norm], dim=-1)  # [C_out, nh+1]
        z = torch.sigmoid(self.gate_proj(gate_input))    # [C_out, nh]
        h_trans = self.gate_transform(h)                  # [C_out, nh]
        h = self.gate_norm((1 - z) * h + z * h_trans)     # 门控残差
        
        # === 创新①: 分解式全参数校准 ===
        # 创新⑤: tanh限幅, 范围 [-scale, +scale]
        w_cin_raw = torch.tanh(self.calib_cin(h)) * self.calib_scale  # [C_out, C_in]
        w_k_raw = torch.tanh(self.calib_k(h)) * self.calib_scale      # [C_out, K]
        b_raw = torch.tanh(self.calib_b(h)).squeeze(-1) * self.calib_scale  # [C_out]
        f_raw = torch.tanh(self.calib_f(h)).squeeze(-1) * self.calib_scale  # [C_out]
        
        # 自适应 tau
        tau_adaptive = self.tau_net(h).mean() * 0.5 + 0.5  # [0.5, 1.0]
        
        # === 构建全局 q 向量 (用于Memory) ===
        q = torch.cat([w_cin_raw.reshape(-1), w_k_raw.reshape(-1), b_raw, f_raw])
        # [C_out*C_in + C_out*K + C_out + C_out = C_out*(C_in+K+2)]
        
        # === 创新⑥: 单步 EMA blend ===
        with torch.no_grad():
            self.q_ema.mul_(self.f_gamma).add_(q.detach(), alpha=1 - self.f_gamma)
        q = 0.7 * q + 0.3 * self.q_ema.detach()
        
        # === 创新②: 通道分组 Memory 检索 ===
        if self.trigger == 1:
            self.trigger = 0
            q_det = q.detach()
            
            # 将 q 按通道分组 — 语义完整!
            cpg = self.channels_per_group
            q_groups = []
            offset = 0
            for g_idx in range(self.n_groups):
                start = g_idx * cpg
                end = start + cpg
                # 每组: [w_cin[cpg,C_in], w_k[cpg,K], b[cpg], f[cpg]]
                cin_start = start * self.in_channels
                cin_end = end * self.in_channels
                k_offset = self.out_channels * self.in_channels
                k_start = k_offset + start * self.kernel_size
                k_end = k_offset + end * self.kernel_size
                bf_offset = k_offset + self.out_channels * self.kernel_size
                
                group_q_parts = [
                    q_det[cin_start:cin_end],           # w_cin for this group
                    q_det[k_start:k_end],               # w_k for this group  
                    q_det[bf_offset + start : bf_offset + end],  # b
                    q_det[bf_offset + self.out_channels + start : 
                          bf_offset + self.out_channels + end],  # f
                ]
                q_groups.append(torch.cat(group_q_parts))
            
            q_groups_tensor = torch.stack(q_groups)  # [n_groups, group_q_dim]
            
            # 检索 (带反向温度)
            retrieved, topk_info = self.memory.retrieve(
                q_groups_tensor, shift_magnitude=self.shift_magnitude
            )
            
            # 重组 retrieved 回全局 q 形状
            retrieved_global = torch.zeros_like(q_det)
            for g_idx in range(self.n_groups):
                start = g_idx * cpg
                end = start + cpg
                r = retrieved[g_idx]
                r_offset = 0
                
                # w_cin
                cin_start = start * self.in_channels
                cin_end = end * self.in_channels
                sz = cin_end - cin_start
                retrieved_global[cin_start:cin_end] = r[r_offset:r_offset+sz]
                r_offset += sz
                
                # w_k
                k_offset_g = self.out_channels * self.in_channels
                k_start = k_offset_g + start * self.kernel_size
                k_end = k_offset_g + end * self.kernel_size
                sz = k_end - k_start
                retrieved_global[k_start:k_end] = r[r_offset:r_offset+sz]
                r_offset += sz
                
                # b
                bf_offset = k_offset_g + self.out_channels * self.kernel_size
                retrieved_global[bf_offset + start : bf_offset + end] = r[r_offset:r_offset+cpg]
                r_offset += cpg
                
                # f
                retrieved_global[bf_offset + self.out_channels + start :
                                bf_offset + self.out_channels + end] = r[r_offset:r_offset+cpg]
            
            # 使用自适应tau融合
            q = tau_adaptive * q + (1 - tau_adaptive) * retrieved_global.detach()
            
            # 写回 Memory (用 tau_adaptive)
            self.memory.write(q_groups_tensor, topk_info, tau_adaptive.item())
        
        # === 还原校准参数 ===
        cin_total = self.out_channels * self.in_channels
        k_total = self.out_channels * self.kernel_size
        
        w_cin = q[:cin_total].view(self.out_channels, self.in_channels)
        w_k = q[cin_total:cin_total + k_total].view(self.out_channels, self.kernel_size)
        b = q[cin_total + k_total : cin_total + k_total + self.out_channels]
        f = q[cin_total + k_total + self.out_channels :]
        
        # 创新①: 分解式全参数调制 — 残差形式
        # w_full = (1 + w_cin) ⊗ (1 + w_k) → [C_out, C_in, K]
        w_full = (1 + w_cin).unsqueeze(-1) * (1 + w_k).unsqueeze(1)
        
        # b 和 f 也用残差形式
        b_calib = 1 + b   # [C_out]
        f_calib = (1 + f).unsqueeze(0).unsqueeze(-1)  # [1, C_out, 1]
        
        return w_full, b_calib, f_calib
    
    def forward(self, x):
        w, b, f = self.fw_chunks()
        # w: [C_out, C_in, K] — 全参数调制!
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
#  v3 ConvBlock & Encoder
# ===========================================================================

class ConvBlockV3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False, gamma=0.9, n_groups=4, M=32, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.conv1 = SamePadConvV3(
            in_channels, out_channels, kernel_size,
            dilation=dilation, gamma=gamma, n_groups=n_groups, M=M,
            device=self.device
        )
        self.conv2 = SamePadConvV3(
            out_channels, out_channels, kernel_size,
            dilation=dilation, gamma=gamma, n_groups=n_groups, M=M,
            device=self.device
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


class DilatedConvEncoderV3(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9,
                 n_groups=4, M=32, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.net = nn.Sequential(*[
            ConvBlockV3(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1),
                gamma=gamma, n_groups=n_groups, M=M, device=self.device
            )
            for i in range(len(channels))
        ])
    
    def ctrl_params(self):
        ctrl = [layer.ctrl_params() for layer in self.net]
        for p in chain(*ctrl):
            yield p
    
    def forward(self, x):
        return self.net(x)
