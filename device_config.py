"""
全局设备配置文件
====================================
在这里修改USE_CUDA就能全局切换CPU/GPU

用法:
    from device_config import get_device
    device = get_device()
"""

import torch
import os

# ============================================
# 🔧 全局设备配置 - 只需要改这一个地方！
# ============================================
USE_CUDA = False  # ✅ 改为 True 启用GPU，False 使用CPU

# ============================================
# 自动设备选择逻辑（不需要修改）
# ============================================
def get_device(force_cpu=None):
    """
    获取全局设备
    
    参数:
        force_cpu: 强制使用CPU（覆盖全局配置）
    
    返回:
        torch.device: CPU或CUDA设备
    """
    # 强制CPU模式
    if force_cpu is not None and force_cpu:
        return torch.device('cpu')
    
    # 使用全局配置
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'[GPU] 使用GPU: {torch.cuda.get_device_name(0)}')
        print(f'      显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        device = torch.device('cpu')
        if USE_CUDA and not torch.cuda.is_available():
            print('[WARNING] 配置了USE_CUDA=True但CUDA不可用，回退到CPU')
        else:
            print('[CPU] 使用CPU')
    
    return device

def set_device_env():
    """设置CUDA相关环境变量"""
    if not USE_CUDA:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('[CONFIG] 已禁用CUDA（设置CUDA_VISIBLE_DEVICES="")')

# ============================================
# 便捷函数
# ============================================
def is_cuda_available():
    """检查CUDA是否可用"""
    return torch.cuda.is_available()

def get_device_info():
    """获取设备信息"""
    device = get_device()
    info = {
        'device': str(device),
        'use_cuda': USE_CUDA,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if device.type == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['gpu_count'] = torch.cuda.device_count()
    
    return info

def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    print('\n' + '='*50)
    print('[DEVICE] 设备配置')
    print('='*50)
    print(f'当前设备: {info["device"]}')
    print(f'配置USE_CUDA: {info["use_cuda"]}')
    print(f'CUDA可用: {info["cuda_available"]}')
    
    if 'gpu_name' in info:
        print(f'GPU型号: {info["gpu_name"]}')
        print(f'GPU显存: {info["gpu_memory_gb"]:.2f} GB')
        print(f'GPU数量: {info["gpu_count"]}')
    print('='*50 + '\n')

# ============================================
# 快速测试
# ============================================
if __name__ == '__main__':
    print_device_info()
    
    # 测试张量创建
    device = get_device()
    x = torch.randn(3, 3).to(device)
    print(f'\n测试张量: {x.device}')
    print('[OK] 设备配置工作正常！')
