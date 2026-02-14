"""
FSNet Advanced: 创新架构优化版本

核心改进：
1. 结构对齐的Chunk机制（按通道而非flatten顺序）
2. 改进的Memory检索策略（温度缩放+Top-k加权）
3. 多尺度梯度聚合
4. 自适应融合系数
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

def normalize(W):
    W_norm = torch.norm(W)
    W_norm = torch.relu(W_norm - 1) + 1
    W = W / W_norm
    return W

class SamePadConvAdvanced(nn.Module):
    """
    创新点：
    1. 结构对齐Chunk：每个chunk对应一个输出通道
    2. 动态温度：根据梯度变化自适应调整
    3. 改进的Memory更新策略
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, 
                 gamma=0.9, temp=0.5, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        
        # 卷积层
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
            groups=groups, bias=False
        )
        self.bias = nn.Parameter(torch.zeros([out_channels]), requires_grad=True)
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # 参数维度
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = out_channels * in_channels * kernel_size
        
        # 创新1: 结构对齐的Chunk - 每个chunk对应一个输出通道
        self.n_chunks = out_channels  # 改进：按输出通道分块
        self.chunk_in_d = in_channels * kernel_size  # 每个chunk的输入维度
        
        # 梯度缓存（按通道组织）
        self.register_buffer('grads', torch.zeros(self.dim))
        self.register_buffer('f_grads', torch.zeros(self.dim))
        
        # Controller: 为每个通道生成校准参数
        nh = 64
        self.controller = nn.Sequential(
            nn.Linear(self.chunk_in_d, nh),
            nn.SiLU(),
            nn.Dropout(0.1)  # 添加dropout提升泛化
        )
        
        # 校准头
        self.calib_w = nn.Linear(nh, kernel_size)  # 每个通道生成K个权重
        self.calib_b = nn.Linear(nh, 1)  # 每个通道的bias校准
        self.calib_f = nn.Linear(nh, 1)  # 每个通道的feature校准
        
        # 创新2: 自适应融合系数
        self.tau_learner = nn.Sequential(
            nn.Linear(nh, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出[0,1]
        )
        
        # Memory结构
        M = 32  # 内存槽数量
        self.q_dim = out_channels * (kernel_size + 2)  # q向量维度
        self.register_buffer('q_ema', torch.zeros(self.q_dim))
        self.W = nn.Parameter(torch.empty(self.q_dim, M), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = F.normalize(self.W.data, dim=0)
        
        # 创新3: 动态温度
        self.base_temp = temp
        self.register_buffer('temp', torch.tensor(temp))
        
        # 触发器相关
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.tau = 0.75
        
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def ctrl_params(self):
        """返回可训练的controller参数"""
        c_iter = chain(
            self.controller.parameters(),
            self.calib_w.parameters(),
            self.calib_b.parameters(),
            self.calib_f.parameters(),
            self.tau_learner.parameters()
        )
        for p in c_iter:
            yield p
    
    def store_grad(self):
        """存储梯度并检测分布偏移"""
        grad = self.conv.weight.grad.data.clone()
        grad = F.normalize(grad.reshape(-1), dim=0)
        
        # 更新快慢梯度EMA
        self.f_grads.mul_(self.f_gamma).add_(grad, alpha=1 - self.f_gamma)
        
        if not self.training:
            # 检测分布偏移
            e = self.cos(self.f_grads, self.grads)
            if e < -self.tau:
                self.trigger = 1
                # 创新: 动态调整温度
                self.temp = self.base_temp * (1 + abs(e))
        
        self.grads.mul_(self.gamma).add_(grad, alpha=1 - self.gamma)
    
    def fw_chunks(self):
        """
        前向传播：生成校准参数
        
        创新点：
        1. 结构对齐：按输出通道重组梯度
        2. 自适应融合系数
        3. 改进的Memory检索
        """
        # 重组梯度：[C_out, C_in, K] -> [C_out, C_in*K]
        grads_reshaped = self.grads.view(self.out_channels, self.in_channels, self.kernel_size)
        grads_reshaped = grads_reshaped.view(self.out_channels, -1)  # [C_out, C_in*K]
        
        # 通过controller处理每个通道
        rep = self.controller(grads_reshaped)  # [C_out, nh]
        
        # 生成校准参数
        w = self.calib_w(rep)  # [C_out, K]
        b = self.calib_b(rep).squeeze(-1)  # [C_out]
        f = self.calib_f(rep).squeeze(-1)  # [C_out]
        
        # 创新: 自适应融合系数
        tau_adaptive = self.tau_learner(rep).mean() * 0.5 + 0.5  # [0.5, 1.0]
        
        # 构建q向量
        q = torch.cat([w.reshape(-1), b, f], dim=0)  # [q_dim]
        
        # EMA更新 (detach to avoid backward graph issues)
        with torch.no_grad():
            self.q_ema.mul_(self.f_gamma).add_(q.detach(), alpha=1 - self.f_gamma)
        q_ema = self.q_ema.detach()
        # blend current q with EMA
        q = 0.5 * q + 0.5 * q_ema
        
        # Memory检索（触发时）
        if self.trigger == 1:
            self.trigger = 0
            
            # 计算注意力分数（使用动态温度, detach q for memory ops）
            q_det = q.detach()
            att = q_det @ self.W  # [M]
            att = F.softmax(att / self.temp, dim=0)
            
            # Top-k检索
            k = min(3, self.W.shape[1])  # 增加到3个槽
            v, idx = torch.topk(att, k)
            
            # 正确的加权检索
            old_q = (self.W[:, idx] * v).sum(dim=1)  # [q_dim]
            
            # 使用自适应系数融合
            q = tau_adaptive * q + (1 - tau_adaptive) * old_q.detach()
            
            # Memory更新：只更新top-k槽 (no grad)
            with torch.no_grad():
                for j in idx:
                    self.W.data[:, j] = F.normalize(
                        tau_adaptive * self.W.data[:, j] + (1 - tau_adaptive) * q_det,
                        dim=0
                    )
        
        # 分解q回w, b, f
        w_size = self.out_channels * self.kernel_size
        w = q[:w_size].view(self.out_channels, self.kernel_size)
        b = q[w_size:w_size + self.out_channels]
        f = q[w_size + self.out_channels:]
        
        # 重塑为卷积需要的形状
        w = w.unsqueeze(1)  # [C_out, 1, K] - broadcasts with [C_out, C_in, K]
        f = f.unsqueeze(0).unsqueeze(-1)  # [1, C_out, 1] - broadcasts with [B, C_out, T]
        
        return w, b, f
    
    def forward(self, x):
        """前向传播"""
        w, b, f = self.fw_chunks()
        # w: [1, C_out, 1, K], b: [C_out], f: [1, C_out, 1]
        
        # 校准卷积权重: conv.weight [C_out, C_in, K] * w [C_out, 1, K]
        cw = self.conv.weight * w  # broadcast: [C_out, C_in, K]
        
        # 卷积
        out = F.conv1d(x, cw, padding=self.padding,
                      dilation=self.dilation, bias=self.bias * b)
        
        # 特征校准: f [1, C_out, 1] * out [B, C_out, T]
        out = f * out
        
        # 去除多余的padding
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        
        return out
    
    def representation(self, x):
        """纯表示学习（无校准）"""
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class ConvBlockAdvanced(nn.Module):
    """改进的卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, 
                 final=False, gamma=0.9, temp=0.5, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        
        self.conv1 = SamePadConvAdvanced(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, gamma=gamma, temp=temp, device=self.device
        )
        self.conv2 = SamePadConvAdvanced(
            out_channels, out_channels, kernel_size,
            dilation=dilation, gamma=gamma, temp=temp, device=self.device
        )
        
        self.projector = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels or final else None
    
    def ctrl_params(self):
        """返回controller参数"""
        c_iter = chain(
            self.conv1.ctrl_params(),
            self.conv2.ctrl_params()
        )
        return c_iter
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoderAdvanced(nn.Module):
    """改进的扩张卷积编码器"""
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9, 
                 temp=0.5, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        
        self.net = nn.Sequential(*[
            ConvBlockAdvanced(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1),
                gamma=gamma,
                temp=temp,
                device=self.device
            )
            for i in range(len(channels))
        ])
    
    def ctrl_params(self):
        """返回所有controller参数"""
        ctrl = []
        for layer in self.net:
            ctrl.append(layer.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p
    
    def forward(self, x):
        return self.net(x)
