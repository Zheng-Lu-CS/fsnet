"""
FSNet Advanced实验类：使用创新架构的实验配置
"""
import os
import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic
import numpy as np
import time

class Exp_FSNetAdvanced(Exp_Basic):
    """
    使用创新架构的FSNet实验类
    
    创新点：
    1. 结构对齐的Chunk机制
    2. 自适应融合系数
    3. 改进的Memory检索策略
    
    注：该类继承Exp_Basic，使用其_build_model方法
    仅在模型文件中替换为advanced版本
    """
    
    def __init__(self, args):
        # 标记使用advanced架构
        self.use_advanced = True
        super(Exp_FSNetAdvanced, self).__init__(args)
    
    def train(self, setting):
        """训练模型"""
        print(f"\n[FSNet-Advanced] 使用创新架构训练")
        print("创新点：")
        print("  1. 结构对齐Chunk（按通道分块）")
        print("  2. 自适应融合系数")
        print("  3. 改进Memory检索（Top-3加权）")
        print("  4. 动态温度缩放\n")
        
        return super().train(setting)
    
    def online_learning_eval(self, setting, load=False):
        """在线学习评估"""
        print(f"\n[FSNet-Advanced] 在线学习评估（创新架构）")
        return super().online_learning_eval(setting, load)
