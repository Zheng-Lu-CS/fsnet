"""
FSNet-Advanced 训练实验
通过monkey-patch替换DilatedConvEncoder为Advanced版本
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import time
import json
import numpy as np
import argparse
from datetime import datetime

# Monkey-patch: 用Advanced版本替换原始DilatedConvEncoder
from models.ts2vec import fsnet_advanced
from models.ts2vec import fsnet_ as original_fsnet
from models.ts2vec import fsnet as fsnet_module

# 保存原始类
OriginalDilatedConvEncoder = original_fsnet.DilatedConvEncoder
OriginalSamePadConv = original_fsnet.SamePadConv

# 替换
original_fsnet.DilatedConvEncoder = fsnet_advanced.DilatedConvEncoderAdvanced
original_fsnet.SamePadConv = fsnet_advanced.SamePadConvAdvanced

print("✅ Monkey-patch: DilatedConvEncoder → DilatedConvEncoderAdvanced")
print("✅ Monkey-patch: SamePadConv → SamePadConvAdvanced")

from exp.exp_fsnet import Exp_TS2VecSupervised

def get_args():
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument('--root_path', type=str, default='./fsnet/data/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')
    
    # 模型
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    
    # 训练
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    
    # 在线学习
    parser.add_argument('--online_learning', type=str, default='full')
    
    # FSNet参数
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n_inner', type=int, default=3)
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--adapt_lr', type=float, default=0.005)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--dw', type=float, default=0.01)
    parser.add_argument('--hiddens', type=int, default=64)
    
    # FSNet-Advanced特有
    parser.add_argument('--temp', type=float, default=0.5, help='Memory温度参数')
    
    # 其他
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoints', type=str, default='./fsnet/checkpoints/fsnet_advanced/')
    parser.add_argument('--des', type=str, default='advanced')
    
    args = parser.parse_args([])
    args.use_gpu = False
    
    # 补充exp_fsnet.py需要的所有参数
    args.finetune = False
    args.finetune_model_seed = 0
    args.use_amp = False
    args.test_bsz = 1
    args.detail_freq = 'h'
    args.inverse = False
    args.cols = None
    
    return args


def main():
    print("="*80)
    print("FSNet-Advanced 创新模型训练")
    print("="*80)
    
    args = get_args()
    os.makedirs(args.checkpoints, exist_ok=True)
    
    setting = f'{args.data}_{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}'
    
    # 创建实验
    exp = Exp_TS2VecSupervised(args)
    
    # === 训练 ===
    print("\n[FSNet-Advanced] 开始训练...")
    t0 = time.time()
    exp.train(setting)
    train_time = time.time() - t0
    print(f"[FSNet-Advanced] 训练完成: {train_time:.1f}s")
    
    # === 测试 ===
    print("\n[FSNet-Advanced] 开始测试 (含在线学习)...")
    t1 = time.time()
    metrics, mae_array, mse_array, preds, trues = exp.test(setting)
    test_time = time.time() - t1
    
    mae, mse, rmse, mape, mspe, _ = metrics
    
    print(f"\n{'='*60}")
    print(f"FSNet-Advanced 测试结果:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape*100:.2f}%")
    print(f"  Train: {train_time:.1f}s  Test: {test_time:.1f}s  Total: {train_time+test_time:.1f}s")
    print(f"{'='*60}")
    
    # === 保存结果 ===
    result = {
        'FSNet-Advanced': {
            'method': 'FSNet-Advanced',
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape * 100),
            'mspe': float(mspe),
            'train_time': train_time,
            'test_time': test_time,
            'ol_time': 0,
            'total_time': train_time + test_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # 更新到综合结果
    comp_file = 'results/comprehensive/comprehensive_results.json'
    if os.path.exists(comp_file):
        with open(comp_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = {}
    
    existing.update(result)
    
    os.makedirs('results/comprehensive', exist_ok=True)
    with open(comp_file, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已更新到: {comp_file}")
    
    # === 全面对比 ===
    print(f"\n{'='*80}")
    print("全面性能对比")
    print(f"{'='*80}")
    print(f"{'模型':<20} {'MAPE(%)':<10} {'MSE':<12} {'MAE':<12} {'时间(s)':<10}")
    print("-" * 64)
    for k, v in existing.items():
        print(f"{k:<20} {v['mape']:<10.2f} {v['mse']:<12.6f} {v['mae']:<12.6f} {v['total_time']:<10.0f}")
    
    # 排名
    sorted_methods = sorted(existing.items(), key=lambda x: x[1]['mape'])
    print(f"\n🏆 MAPE排名:")
    for i, (name, data) in enumerate(sorted_methods, 1):
        marker = " ⭐" if name == 'FSNet-Advanced' else ""
        print(f"  {i}. {name}: {data['mape']:.2f}%{marker}")


if __name__ == '__main__':
    main()
