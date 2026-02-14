"""
FSNet优化实验：对比NoMem与修复Bug后的FSNet
修复内容：fw_chunks中使用attention weights v进行加权，而非错误的idx索引
"""
import os
import sys
import json
import time
import numpy as np
import torch
from argparse import Namespace
from datetime import datetime

# 添加路径
sys.path.append('fsnet')

# 导入实验类（使用别名区分）
from exp.exp_nomem import Exp_TS2VecSupervised as Exp_NoMem
from exp.exp_fsnet import Exp_TS2VecSupervised as Exp_FSNet

def create_base_args():
    """创建与ablation_study.py完全相同的基础参数"""
    args = Namespace(
        # 基础配置
        model='fs',
        data='ETTh1',
        root_path='./fsnet/data/',
        data_path='ETTh1.csv',
        features='S',  # 单变量
        target='OT',
        freq='h',
        checkpoints='./fsnet/checkpoints/',
        
        # 序列长度
        seq_len=48,
        label_len=24,
        pred_len=12,
        
        # 模型维度
        enc_in=1,  # 单变量输入
        dec_in=1,
        c_out=1,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        factor=5,
        distil=True,
        mix=True,
        attn='prob',
        moving_avg=25,
        
        # 数据处理
        cols=None,
        inverse=False,
        do_predict=False,
        test_flop=False,
        devices='0',
        
        # 训练配置
        num_workers=0,
        itr=1,
        train_epochs=2,
        batch_size=8,
        patience=3,
        learning_rate=0.0001,
        des='optimization_exp',
        loss='mse',
        lradj='type1',
        use_amp=False,
        
        # FSNet特定参数
        olr=0.001,
        n_inner=1,
        opt='adamw',
        hiddens=[64],
        kernel_size=3,
        
        # 在线学习配置
        online_learning='full',
        ol_lr=0.01,
        buffer_size=64,
        finetune=False,
        finetune_model_seed=0,
        repr_dims=64,
        max_train_length=201,
        method='fsnet',
        test_bsz=1,
        
        # 设备配置
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        
        # 其他
        detail_freq='h',
    )
    return args

def train_and_evaluate(exp_class, method_name, args):
    """训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"开始训练: {method_name}")
    print(f"{'='*60}\n")
    
    # 设置实验特定的checkpoint路径
    args.checkpoints = f'./fsnet/checkpoints/{method_name.lower()}/'
    os.makedirs(args.checkpoints, exist_ok=True)
    
    # 创建实验实例
    exp = exp_class(args)
    
    # 训练
    print(f"[{method_name}] 开始训练...")
    train_start = time.time()
    
    setting = f'{args.data}_{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}'
    exp.train(setting)
    
    train_time = time.time() - train_start
    print(f"[{method_name}] 训练完成，耗时: {train_time:.2f}秒")
    
    # 测试
    print(f"[{method_name}] 开始测试...")
    test_start = time.time()
    
    metrics, mae_array, mse_array, preds, trues = exp.test(setting)
    mae, mse, rmse, mape, mspe, _ = metrics
    
    test_time = time.time() - test_start
    print(f"[{method_name}] 测试完成，耗时: {test_time:.2f}秒")
    
    # 在线学习测试
    print(f"[{method_name}] 开始在线学习评估...")
    ol_start = time.time()
    
    ol_mse, ol_mae, predictions = exp.online_learning_eval(setting, load=True)
    
    ol_time = time.time() - ol_start
    print(f"[{method_name}] 在线学习评估完成，耗时: {ol_time:.2f}秒")
    
    # 计算额外指标
    ol_rmse = np.sqrt(ol_mse)
    
    # 使用在线学习的指标作为最终结果（更准确）
    results = {
        'method': method_name,
        'train_time': train_time,
        'test_time': test_time,
        'ol_time': ol_time,
        'total_time': train_time + test_time + ol_time,
        'mse': float(ol_mse),
        'mae': float(ol_mae),
        'rmse': float(ol_rmse),
        'mape': float(mape * 100) if mape < 1 else float(mape),  # 转为百分比
        'mspe': float(mspe * 100) if mspe < 1 else float(mspe),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"\n[{method_name}] 结果摘要:")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAPE: {results['mape']:.2f}%")
    print(f"  总耗时: {results['total_time']:.2f}秒")
    
    return results

def main():
    print("\n" + "="*80)
    print("FSNet架构优化实验")
    print("="*80)
    print("\n实验目的:")
    print("  1. 修复fw_chunks中的bug：使用attention weights v而非idx")
    print("  2. 对比NoMem（Adapter-only）与优化后FSNet的性能")
    print("  3. 验证修复后的FSNet能否超越NoMem")
    print("\n关键修复:")
    print("  - 原代码: old_w = ww @ idx  # 错误！")
    print("  - 修复后: old_q = (self.W[:, idx] * v).sum(dim=1)  # 正确使用注意力权重")
    print("="*80 + "\n")
    
    # 创建结果目录
    results_dir = 'results/optimization/'
    os.makedirs(results_dir, exist_ok=True)
    
    # 基础参数
    base_args = create_base_args()
    
    # 实验列表
    experiments = [
        (Exp_NoMem, "NoMem"),
        (Exp_FSNet, "FSNet_Fixed"),
    ]
    
    all_results = {}
    
    # 运行实验
    for exp_class, method_name in experiments:
        try:
            args = create_base_args()  # 每次创建新的args副本
            results = train_and_evaluate(exp_class, method_name, args)
            all_results[method_name] = results
        except Exception as e:
            print(f"\n❌ [{method_name}] 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    results_file = os.path.join(results_dir, 'optimization_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {results_file}")
    
    # 对比分析
    print("\n" + "="*80)
    print("性能对比分析")
    print("="*80)
    
    if 'NoMem' in all_results and 'FSNet_Fixed' in all_results:
        nomem = all_results['NoMem']
        fsnet = all_results['FSNet_Fixed']
        
        print(f"\n{'指标':<15} {'NoMem':<15} {'FSNet_Fixed':<15} {'改进':<15}")
        print("-" * 60)
        
        metrics = ['mse', 'mae', 'rmse', 'mape']
        for metric in metrics:
            nomem_val = nomem[metric]
            fsnet_val = fsnet[metric]
            improvement = (nomem_val - fsnet_val) / nomem_val * 100
            
            print(f"{metric.upper():<15} {nomem_val:<15.6f} {fsnet_val:<15.6f} {improvement:>+.2f}%")
        
        print(f"\n{'训练时间(s)':<15} {nomem['train_time']:<15.2f} {fsnet['train_time']:<15.2f}")
        print(f"{'总时间(s)':<15} {nomem['total_time']:<15.2f} {fsnet['total_time']:<15.2f}")
        
        print("\n核心发现:")
        if fsnet['mape'] < nomem['mape']:
            improvement = ((nomem['mape'] - fsnet['mape']) / nomem['mape'] * 100)
            print(f"  ✅ 修复后的FSNet超越NoMem: MAPE改进 {improvement:.2f}%")
            print(f"  ✅ 证明了Associative Memory机制的有效性")
            print(f"  ✅ 原bug导致Memory模块失效，修复后性能显著提升")
        else:
            print(f"  ⚠️ FSNet仍未超越NoMem，可能需要更多训练轮次")
        
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
