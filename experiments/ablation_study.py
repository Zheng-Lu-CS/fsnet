"""
消融实验脚本 - 对比4个方法性能
运行方式: python fsnet/ablation_study.py
预计耗时: 1小时（每个方法15分钟）
"""

import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_fsnet import Exp_TS2VecSupervised as Exp_FSNet
from exp.exp_ogd import Exp_TS2VecSupervised as Exp_OGD
from exp.exp_er import Exp_TS2VecSupervised as Exp_ER
from exp.exp_nomem import Exp_TS2VecSupervised as Exp_NoMem
from utils.tools import dotdict
import device_config

print("="*70)
print("消融实验 - 对比FSNet及其变体")
print("="*70)
print(f"设备: {device_config.get_device()}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# 统一配置（保持和myexp.py一致）
base_args = {
    # 基础配置
    'model': 'fs',
    'data': 'ETTh1',
    'root_path': './fsnet/data/',
    'data_path': 'ETTh1.csv',
    'features': 'S',  # 单变量模式
    'target': 'OT',
    'freq': 'h',
    'checkpoints': './fsnet/checkpoints/',
    
    # 序列长度
    'seq_len': 48,
    'label_len': 24,
    'pred_len': 12,
    
    # 模型维度
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'dropout': 0.05,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'factor': 5,
    'distil': True,
    'mix': True,
    'attn': 'prob',
    'moving_avg': 25,
    
    # 数据处理
    'cols': None,
    'inverse': False,
    'do_predict': False,
    'test_flop': False,
    'devices': '0',
    
    # 优化器
    'num_workers': 0,
    'itr': 1,
    'train_epochs': 2,  # 快速实验，只跑2轮
    'batch_size': 8,
    'patience': 3,
    'learning_rate': 0.0001,
    'des': 'ablation',
    'loss': 'mse',
    'lradj': 'type1',
    'use_amp': False,
    
    # FSNet特定参数
    'olr': 0.001,
    'n_inner': 1,
    'opt': 'adamw',
    'hiddens': [64],
    'kernel_size': 3,
    'gpu': 0,
    'use_gpu': False,  # CPU模式
    'use_multi_gpu': False,
    
    # 在线学习参数
    'online_learning': 'full',  # 'none', 'full', or 'regressor'
    'finetune': False,
    'finetune_model_seed': 0,
    'repr_dims': 64,
    'max_train_length': 201,
    'method': 'fsnet',
    'test_bsz': 1,
    'detail_freq': 'h',
}

# 定义4个实验方法
experiments = {
    'OGD': {
        'name': 'Online Gradient Descent（标准在线学习）',
        'exp_class': Exp_OGD,
        'desc': '基础baseline，无任何快速适应机制',
        'color': '#e74c3c'
    },
    'ER': {
        'name': 'Experience Replay（经验回放）',
        'exp_class': Exp_ER,
        'desc': '使用buffer存储历史样本重放',
        'color': '#3498db'
    },
    'NoMem': {
        'name': 'FSNet-NoMemory（只有Adapter）',
        'exp_class': Exp_NoMem,
        'desc': '移除关联记忆，验证Memory的贡献',
        'color': '#f39c12'
    },
    'FSNet': {
        'name': 'FSNet（完整模型）',
        'exp_class': Exp_FSNet,
        'desc': 'Adapter + Associative Memory',
        'color': '#2ecc71'
    }
}

# 存储结果
results = {}

# 运行每个实验
for method_key, method_info in experiments.items():
    print(f"\n{'='*70}")
    print(f"📊 方法 {list(experiments.keys()).index(method_key)+1}/4: {method_key}")
    print(f"{'='*70}")
    print(f"名称: {method_info['name']}")
    print(f"说明: {method_info['desc']}")
    print("-"*70)
    
    # 创建实验对象
    args = dotdict(base_args.copy())
    args.des = method_key
    
    setting = f"{args.data}_{args.features}_{args.seq_len}_{args.pred_len}_{method_key}"
    
    exp = method_info['exp_class'](args)
    
    # 训练
    print(f"\n⏳ 开始训练...")
    train_start = time.time()
    exp.train(setting)
    train_time = time.time() - train_start
    
    # 测试
    print(f"\n⏳ 开始测试...")
    test_start = time.time()
    metrics, mae_array, mse_array, preds, trues = exp.test(setting)
    test_time = time.time() - test_start
    
    # 解析指标
    mae, mse, rmse, mape, mspe, _ = metrics
    
    # 存储结果
    results[method_key] = {
        'name': method_info['name'],
        'desc': method_info['desc'],
        'metrics': {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape * 100),  # 转为百分比
            'MSPE': float(mspe)
        },
        'time': {
            'train': train_time,
            'test': test_time,
            'total': train_time + test_time
        },
        'predictions': preds,  # 保存用于后续可视化
        'ground_truth': trues
    }
    
    # 打印结果
    print(f"\n✅ {method_key} 完成!")
    print(f"   MSE:  {mse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAPE: {mape*100:.2f}%")
    print(f"   训练时间: {train_time:.1f}s")
    print(f"   测试时间: {test_time:.1f}s")

# 保存完整结果
result_dir = './results/ablation/'
os.makedirs(result_dir, exist_ok=True)

# 保存JSON格式（不含预测数组）
results_json = {
    k: {
        'name': v['name'],
        'desc': v['desc'],
        'metrics': v['metrics'],
        'time': v['time']
    } for k, v in results.items()
}

json_path = f"{result_dir}ablation_results.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)

# 保存NumPy格式（含预测数组，用于可视化）
np_path = f"{result_dir}ablation_predictions.npz"
np.savez(
    np_path,
    **{f'{k}_preds': v['predictions'] for k, v in results.items()},
    **{f'{k}_trues': v['ground_truth'] for k, v in results.items()}
)

print(f"\n{'='*70}")
print("📊 消融实验结果汇总")
print(f"{'='*70}")
print(f"{'方法':<15} {'MSE':<10} {'MAE':<10} {'MAPE':<10} {'训练时间':<12} {'测试时间':<12}")
print("-"*70)

for method_key in ['OGD', 'ER', 'NoMem', 'FSNet']:
    r = results[method_key]
    print(f"{method_key:<15} "
          f"{r['metrics']['MSE']:<10.6f} "
          f"{r['metrics']['MAE']:<10.6f} "
          f"{r['metrics']['MAPE']:<9.2f}% "
          f"{r['time']['train']:<11.1f}s "
          f"{r['time']['test']:<11.1f}s")

print("="*70)

# 计算改进百分比
baseline_mse = results['OGD']['metrics']['MSE']
fsnet_mse = results['FSNet']['metrics']['MSE']
improvement = (baseline_mse - fsnet_mse) / baseline_mse * 100

print(f"\n🎯 FSNet相比OGD改进: {improvement:.2f}%")
print(f"📁 结果已保存:")
print(f"   - JSON: {json_path}")
print(f"   - NumPy: {np_path}")
print(f"\n⏱️  总耗时: {sum(r['time']['total'] for r in results.values()):.1f}s")
print(f"✅ 消融实验完成!")
