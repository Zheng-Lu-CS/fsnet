"""
轻量化CPU训练脚本
用于快速验证代码和调试
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from exp.exp_fsnet import Exp_TS2VecSupervised

# 强制CPU模式
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def main():
    parser = argparse.ArgumentParser(description='FSNet Lightweight Training')
    
    # ============ 数据配置 ============
    parser.add_argument('--data', type=str, default='ETTh1', 
                        help='dataset type')
    parser.add_argument('--root_path', type=str, default='./fsnet/data/', 
                        help='root path of data')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', 
                        help='data file')
    parser.add_argument('--features', type=str, default='S',  # ✅ 改为'S'单变量模式
                        help='M:multivariate, S:univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature')
    parser.add_argument('--freq', type=str, default='h',
                        help='h:hourly')
    parser.add_argument('--checkpoints', type=str, default='./fsnet/checkpoints/', 
                        help='checkpoint location')
    
    # ============ 轻量化配置 ============
    parser.add_argument('--seq_len', type=int, default=48,  # 从96->48
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24,  # 从48->24
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=12,  # 从96->12
                        help='prediction sequence length')
    
    # ============ 模型配置 ============
    parser.add_argument('--enc_in', type=int, default=1,  # ✅ 改为1（单变量）
                        help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1,  # ✅ 改为1
                        help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1,  # ✅ 改为1
                        help='output size')
    parser.add_argument('--d_model', type=int, default=64,  # 从512->64
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4,  # 从8->4
                        help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1,  # 从2->1
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128,  # 从2048->128
                        help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5,
                        help='probsparse attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='dropout')
    parser.add_argument('--attn', type=str, default='prob',
                        help='attention type')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention')
    parser.add_argument('--do_predict', action='store_true',
                        help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false',
                        help='use mix attention')
    parser.add_argument('--cols', type=str, nargs='+',
                        help='file list')
    parser.add_argument('--num_workers', type=int, default=0,  # CPU模式设为0
                        help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1,  # 只跑1次
                        help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=2,  # 从10->2
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8,  # 从32->8
                        help='batch size')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--loss', type=str, default='mse',
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training')
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data')
    parser.add_argument('--use_gpu', type=bool, default=False,  # 强制False
                        help='use gpu')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0',
                        help='device ids')
    parser.add_argument('--test_flop', action='store_true',
                        help='test flops')
    
    # ============ FSNet特有参数 ============
    parser.add_argument('--repr_dims', type=int, default=64,  # 从320->64
                        help='representation dimensions')
    parser.add_argument('--max_train_length', type=int, default=201,
                        help='max training length')
    parser.add_argument('--method', type=str, default='fsnet',
                        help='method name')
    parser.add_argument('--online_learning', type=str, default='full',
                        help='full/regressor/none')
    parser.add_argument('--test_bsz', type=int, default=1,
                        help='test batch size')
    parser.add_argument('--n_inner', type=int, default=1,
                        help='inner loop updates')
    parser.add_argument('--opt', type=str, default='adam',
                        help='inner loop updates')
    parser.add_argument('--finetune', type=bool, default=False,
                        help='inner loop updates')
    # detail_freq
    parser.add_argument('--detail_freq', type=str, default='h',
                        help='inner loop updates')
    args = parser.parse_args()
    args.use_gpu = False  # 再次确保CPU模式
    
    print('=' * 50)
    print('FSNet Lightweight Training')
    print('=' * 50)
    print(f'Dataset: {args.data}')
    print(f'Seq Length: {args.seq_len} -> Predict: {args.pred_len}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Epochs: {args.train_epochs}')
    print(f'Device: CPU')
    print(f'Model Params: d_model={args.d_model}, repr_dims={args.repr_dims}')
    print(f'Mode: {"Single Variable" if args.features == "S" else "Multi Variable"}')
    print(f'Target: {args.target if args.features == "S" else "All features"}')
    print(f'Features: enc_in={args.enc_in}, c_out={args.c_out}')
    print('=' * 50)
    
    # 创建实验设置
    setting = f'{args.data}_{args.method}_sl{args.seq_len}_pl{args.pred_len}'
    
    # 初始化实验
    exp = Exp_TS2VecSupervised(args)
    
    # 训练
    print('\n>>> 开始训练...')
    exp.train(setting)
    
    # 测试
    print('\n>>> 开始测试...')
    metrics, mae_array, mse_array, preds, trues = exp.test(setting)
    
    # metrics = [mae, mse, rmse, mape, mspe, exp_time]
    mae, mse, rmse, mape, mspe, test_time = metrics
    
    print('\n' + '=' * 50)
    print('Training Complete!')
    print('=' * 50)
    print(f'Final MSE:  {mse:.6f}')
    print(f'Final MAE:  {mae:.6f}')
    print(f'Final RMSE: {rmse:.6f}')
    print(f'Final MAPE: {mape:.6f}')
    print(f'Test Time:  {test_time:.2f}s')
    print('=' * 50)
    print(f'\nPredictions saved: {preds.shape}')
    print(f'Model checkpoint: ./fsnet/checkpoints/{setting}/')
    print('\nExperiment completed successfully!')
    print('=' * 50)

if __name__ == '__main__':
    main()