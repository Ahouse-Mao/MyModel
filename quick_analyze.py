#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速分析日志文件中的实验结果
"""

import os
import re
import pandas as pd
from datetime import datetime

def parse_log_file(log_path):
    """解析单个日志文件"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取关键参数
        lr_match = re.search(r'Learning Rate:\s+([\d\.e-]+)', content)
        dm_match = re.search(r'Model Dimension:\s+(\d+)', content)
        nh_match = re.search(r'Attention Heads:\s+(\d+)', content)
        el_match = re.search(r'Encoder Layers:\s+(\d+)', content)
        
        # 提取模型参数量
        params_match = re.search(r'generator parameters:\s+(\d+)', content)
        
        # 提取性能指标
        mae_match = re.search(r'mae:([\d\.]+)', content)
        mse_match = re.search(r'mse:([\d\.]+)', content)
        train_time_match = re.search(r'train_time:([\d\.]+)', content)
        test_time_match = re.search(r'test_time:([\d\.]+)', content)
        
        if all([lr_match, dm_match, nh_match, el_match, params_match, mae_match, mse_match]):
            return {
                'log_file': os.path.basename(log_path),
                'learning_rate': float(lr_match.group(1)),
                'd_model': int(dm_match.group(1)),
                'n_heads': int(nh_match.group(1)),
                'e_layers': int(el_match.group(1)),
                'parameters': int(params_match.group(1)),
                'mae': float(mae_match.group(1)),
                'mse': float(mse_match.group(1)),
                'train_time': float(train_time_match.group(1)) if train_time_match else 0,
                'test_time': float(test_time_match.group(1)) if test_time_match else 0,
            }
    except Exception as e:
        print(f"解析日志文件 {log_path} 时出错: {e}")
    
    return None

def main():
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        print("日志目录不存在")
        return
    
    results = []
    
    # 解析所有日志文件
    for log_file in os.listdir(logs_dir):
        if log_file.endswith('.log'):
            log_path = os.path.join(logs_dir, log_file)
            result = parse_log_file(log_path)
            if result:
                results.append(result)
    
    if not results:
        print("未找到有效的日志文件或解析失败")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(['d_model', 'n_heads', 'e_layers'])
    
    print("="*80)
    print("实验结果汇总")
    print("="*80)
    print(f"{'d_model':<8} {'n_heads':<8} {'e_layers':<8} {'参数量(M)':<12} {'MAE':<8} {'MSE':<8} {'训练时间(s)':<12}")
    print("-"*80)
    
    for _, row in df.iterrows():
        params_m = row['parameters'] / 1e6
        print(f"{row['d_model']:<8} {row['n_heads']:<8} {row['e_layers']:<8} "
              f"{params_m:<12.2f} {row['mae']:<8.4f} {row['mse']:<8.4f} {row['train_time']:<12.1f}")
    
    print("="*80)
    
    # 找到最佳配置
    best_mae = df.loc[df['mae'].idxmin()]
    best_mse = df.loc[df['mse'].idxmin()]
    min_params = df.loc[df['parameters'].idxmin()]
    
    print("\n关键发现:")
    print(f"🏆 最低MAE: d_model={best_mae['d_model']}, n_heads={best_mae['n_heads']}, "
          f"e_layers={best_mae['e_layers']}, MAE={best_mae['mae']:.4f}")
    
    print(f"🏆 最低MSE: d_model={best_mse['d_model']}, n_heads={best_mse['n_heads']}, "
          f"e_layers={best_mse['e_layers']}, MSE={best_mse['mse']:.4f}")
    
    print(f"💡 最少参数: d_model={min_params['d_model']}, n_heads={min_params['n_heads']}, "
          f"e_layers={min_params['e_layers']}, 参数量={min_params['parameters']/1e6:.2f}M")
    
    # 计算参数减少比例（假设原始模型232M参数）
    original_params = 232.95e6
    print(f"\n参数减少情况:")
    for _, row in df.iterrows():
        reduction = (original_params - row['parameters']) / original_params * 100
        print(f"d_model={row['d_model']}: 参数减少 {reduction:.1f}% "
              f"({original_params/1e6:.1f}M → {row['parameters']/1e6:.1f}M)")
    
    print("="*80)

if __name__ == "__main__":
    main()
