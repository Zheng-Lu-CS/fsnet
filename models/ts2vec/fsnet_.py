import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb

from itertools import chain

def normalize(W):
    W_norm = torch.norm(W)
    W_norm = torch.relu(W_norm - 1) + 1
    W = W/ W_norm
    return W

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, gamma=0.9, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')  # 默认CPU
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=False
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]),requires_grad=True)
        self.padding=padding
        self.dilation = dilation
        self.kernel_size= kernel_size
        
        self.grad_dim, self.shape = [], []
        for p in self.conv.parameters():
            self.grad_dim.append(p.numel())
            self.shape.append(p.size())
        self.dim = sum(self.grad_dim)
        self.in_channels = in_channels
        self.out_features= out_channels

        self.n_chunks = in_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(in_channels*kernel_size// self.n_chunks)
        
        self.grads = torch.Tensor(sum(self.grad_dim)).fill_(0)
        self.f_grads = torch.Tensor(sum(self.grad_dim)).fill_(0)
        nh=64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, out_channels//in_channels)
        self.calib_f = nn.Linear(nh, out_channels//in_channels)
        dim = self.n_chunks * (self.chunk_out_d + 2 * out_channels // in_channels)
        self.W = nn.Parameter(torch.empty(dim, 32), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = normalize(self.W.data)
        
        
        #self.calib_w = torch.nn.Parameter(torch.ones(out_channels, in_channels,1), requires_grad = True)
        #self.calib_b = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad = True)
        #self.calib_f = torch.nn.Parameter(torch.ones(1,out_channels,1), requires_grad = True)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.tau = 0.75
    def ctrl_params(self):
        c_iter = chain(self.controller.parameters(), self.calib_w.parameters(), 
                self.calib_b.parameters(), self.calib_f.parameters())
        for p in c_iter:
            yield p

    def store_grad(self):
        #print('storing grad')
        grad = self.conv.weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        self.f_grads = self.f_gamma * self.f_grads + (1-self.f_gamma) * grad
        if not self.training: #这里是不是有问题啊，应该是在训练的时候就开启注意力计算吧
            e = self.cos(self.f_grads, self.grads)
            
            if e < -self.tau:
                self.trigger = 1
        self.grads = self.gamma * self.grads + (1-self.gamma) * grad
        
    def fw_chunks(self):
        x = self.grads.view(self.n_chunks, -1)
        rep = self.controller(x)
        w = self.calib_w(rep)
        b = self.calib_b(rep)
        f = self.calib_f(rep)
        q = torch.cat([w.view(-1), b.view(-1), f.view(-1)])
        if not hasattr(self, 'q_ema'):
            setattr(self, 'q_ema', torch.zeros(*q.size()).float().to(self.device))  
        else:
            self.q_ema = self.f_gamma * self.q_ema.detach() + (1-self.f_gamma)*q
            q = self.q_ema
        if self.trigger == 1:
            dim = w.size(0)
            self.trigger = 0
            # read - compute attention scores (detach q for memory ops)
            q_detached = q.detach()
            att = q_detached @ self.W
            att = F.softmax(att/0.5, dim=0)
            
            # retrieve top-k memory slots
            v, idx = torch.topk(att, 2)
            
            # BUG FIX: Use attention weights v instead of idx
            # Correct weighted retrieval from memory
            old_q = (self.W[:, idx] * v).sum(dim=1)  # [q_dim]
            
            # Blend current q with retrieved memory (detached old_q to avoid graph issues)
            q_blended = self.tau * q + (1 - self.tau) * old_q.detach()
            
            # write back to memory (update selected slots, fully detached)
            with torch.no_grad():
                for j in idx:
                    self.W.data[:, j] = normalize(
                        self.tau * self.W.data[:, j] + (1 - self.tau) * q_detached
                    )
            
            # use blended q for parameter calibration
            q = q_blended
            
            # split q back to w, b, f
            nw, nb, nf = w.numel(), b.numel(), f.numel()
            q_w = q[:nw].view(w.size())
            q_b = q[nw:nw+nb].view(b.size())
            q_f = q[nw+nb:].view(f.size())
            
            # blend with adapter outputs
            w = self.tau * w + (1-self.tau) * q_w
            b = self.tau * b + (1-self.tau) * q_b
            f = self.tau * f + (1-self.tau) * q_f
            
        f = f.view(-1).unsqueeze(0).unsqueeze(2)
       
        return w.unsqueeze(0) ,b.view(-1),f

    def forward(self, x):
        w,b,f = self.fw_chunks()
        d0, d1 = self.conv.weight.shape[1:]
        
        cw = self.conv.weight * w
        try:
            conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation, bias = self.bias * b)
            out =  f * conv_out
        except Exception as e:
            print(f'Warning: Exception in SamePadConv forward: {e}')
            raise  # Re-raise the exception
        return out

    def representation(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

    def _forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
# class SamePadConvBetter(nn.Module):
#     def __init__(self, C_in, C_out, K, dilation=1, groups=1, M=32, nh=64,
#                  gamma=0.9, f_gamma=0.3, tau=0.75, temp=0.5):
#         super().__init__()
#         assert groups == 1, "先把 groups=1 跑通；要支持 groups 需要重新定义chunk逻辑"

#         self.C_in, self.C_out, self.K = C_in, C_out, K
#         self.dilation = dilation
#         self.gamma = gamma
#         self.f_gamma = f_gamma
#         self.tau = tau
#         self.temp = temp

#         # conv本体（不使用padding，改用forward里显式pad）
#         self.conv = nn.Conv1d(C_in, C_out, K, padding=0, dilation=dilation, groups=groups, bias=False)
#         self.bias = nn.Parameter(torch.zeros(C_out))

#         # dim / chunk
#         self.dim = C_out * C_in * K
#         self.n_chunks = C_in
#         self.chunk_in_d = C_out * K     # = dim // C_in (groups=1)

#         # 梯度EMA缓存
#         self.register_buffer("grads", torch.zeros(self.dim))
#         self.register_buffer("f_grads", torch.zeros(self.dim))

#         # controller + heads
#         self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
#         self.calib_w = nn.Linear(nh, K)

#         # 关键改动：b,f 直接输出 C_out（无整除假设）
#         self.calib_b = nn.Linear(nh, C_out)
#         self.calib_f = nn.Linear(nh, C_out)

#         # q & memory
#         self.q_dim = C_in * K + 2 * C_out
#         self.register_buffer("q_ema", torch.zeros(self.q_dim))
#         self.W = nn.Parameter(torch.empty(self.q_dim, M), requires_grad=False)
#         nn.init.xavier_uniform_(self.W.data)
#         self.W.data = F.normalize(self.W.data, dim=0)

#         # trigger相关
#         self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
#         self.trigger = 0

#     def _same_pad_1d(self, x):
#         rf = (self.K - 1) * self.dilation + 1
#         total = rf - 1
#         left = total // 2
#         right = total - left
#         return F.pad(x, (left, right))

#     @torch.no_grad()
#     def store_grad(self):
#         g = self.conv.weight.grad
#         if g is None:
#             return
#         g = F.normalize(g.reshape(-1), dim=0)
#         self.f_grads.mul_(self.f_gamma).add_(g, alpha=1 - self.f_gamma)

#         # 你原版是 eval 时才触发；这里保留同逻辑
#         if not self.training:
#             e = self.cos(self.f_grads, self.grads)
#             if e < -self.tau:
#                 self.trigger = 1

#         self.grads.mul_(self.gamma).add_(g, alpha=1 - self.gamma)

#     def fw_chunks(self):
#         # 1) grads -> rep -> w
#         x = self.grads.view(self.C_in, self.chunk_in_d)          # [C_in, C_out*K]
#         rep = self.controller(x)                                 # [C_in, nh]
#         w = self.calib_w(rep)                                     # [C_in, K]

#         # 2) 聚合rep -> 生成 b,f（无整除假设）
#         g = rep.mean(dim=0)                                       # [nh]
#         b = self.calib_b(g)                                       # [C_out]
#         f = self.calib_f(g)                                       # [C_out]

#         # 3) q + ema
#         q = torch.cat([w.reshape(-1), b, f], dim=0)               # [q_dim]
#         self.q_ema.mul_(self.f_gamma).add_(q, alpha=1 - self.f_gamma)
#         q = self.q_ema

#         # 4) trigger -> retrieve + write + fuse on q
#         if self.trigger == 1:
#             self.trigger = 0
#             att = F.softmax((q @ self.W) / self.temp, dim=0)      # [M]
#             v, idx = torch.topk(att, k=2)

#             old_q = (self.W[:, idx] * v).sum(dim=1)               # [q_dim]
#             q = self.tau * q + (1 - self.tau) * old_q

#             # write back only selected slots
#             for j in idx:
#                 self.W.data[:, j] = F.normalize(
#                     self.tau * self.W.data[:, j] + (1 - self.tau) * q,
#                     dim=0
#                 )

#         # 5) slice back
#         w_flat = q[: self.C_in * self.K]
#         b = q[self.C_in * self.K : self.C_in * self.K + self.C_out]
#         f = q[self.C_in * self.K + self.C_out :]

#         w = w_flat.view(1, self.C_in, self.K)                     # [1,C_in,K]
#         b = b.view(self.C_out)                                     # [C_out]
#         f = f.view(1, self.C_out, 1)                               # [1,C_out,1]
#         return w, b, f

#     def forward(self, x):
#         # x: [B,C_in,T]
#         w, b, f = self.fw_chunks()

#         # 动态权重
#         cw = self.conv.weight * w                                 # [C_out,C_in,K] broadcast
#         x_pad = self._same_pad_1d(x)
#         y = F.conv1d(x_pad, cw, bias=self.bias * b, dilation=self.dilation, padding=0)
#         return f * y

#     def representation(self, x):
#         # 如果你希望“纯卷积表示”也严格same-length，就走同样pad逻辑
#         x_pad = self._same_pad_1d(x)
#         y = F.conv1d(x_pad, self.conv.weight, bias=self.bias, dilation=self.dilation, padding=0)
#         return y
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, gamma=0.9, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma, device=self.device)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma, device=self.device)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def ctrl_params(self):  
        c_iter = chain(self.conv1.controller.parameters(), self.conv1.calib_w.parameters(), 
                self.conv1.calib_b.parameters(), self.conv1.calib_f.parameters(),
                self.conv2.controller.parameters(), self.conv2.calib_w.parameters(), 
                self.conv2.calib_b.parameters(), self.conv2.calib_f.parameters())

        return c_iter 
       


    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1), gamma=gamma, device=self.device
            )
            for i in range(len(channels))
        ])
    def ctrl_params(self):
        ctrl = []
        for l in self.net:
            ctrl.append(l.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p
    def forward(self, x):
        return self.net(x)
