#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(log_file_path):
    """解析日志文件，提取关键信息"""
    result = {
        'filename': os.path.basename(log_file_path),
        'learning_rate': None,
        'd_model': None,
        'n_heads': None,
        'e_layers': None,
        'seq_len': None,
        'pred_len': None,
        'parameters': None,
        'train_loss': None,
        'val_loss': None,
        'test_loss': None,
        'best_epoch': None,
        'experiment_time': None
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取关键参数
        lr_match = re.search(r'Learning Rate:\s+([\d\.e-]+)', content)
        if lr_match:
            result['learning_rate'] = float(lr_match.group(1))
        
        dm_match = re.search(r'Model Dimension:\s+(\d+)', content)
        if dm_match:
            result['d_model'] = int(dm_match.group(1))
        
        nh_match = re.search(r'Attention Heads:\s+(\d+)', content)
        if nh_match:
            result['n_heads'] = int(nh_match.group(1))
        
        el_match = re.search(r'Encoder Layers:\s+(\d+)', content)
        if el_match:
            result['e_layers'] = int(el_match.group(1))
        
        sl_match = re.search(r'Sequence Length:\s+(\d+)', content)
        if sl_match:
            result['seq_len'] = int(sl_match.group(1))
        
        pl_match = re.search(r'Prediction Length:\s*(\d+)', content)
        if pl_match:
            result['pred_len'] = int(pl_match.group(1))
        
        # 提取参数量
        params_match = re.search(r'###### generator parameters: (\d+)', content)
        if params_match:
            result['parameters'] = int(params_match.group(1))
        
        # 提取最终损失值
        test_loss_matches = re.findall(r'Test Loss: ([\d\.]+)', content)
        if test_loss_matches:
            result['test_loss'] = float(test_loss_matches[-1])
        
        val_loss_matches = re.findall(r'Vali Loss: ([\d\.]+)', content)
        if val_loss_matches:
            result['val_loss'] = float(val_loss_matches[-1])
        
        train_loss_matches = re.findall(r'Train Loss: ([\d\.]+)', content)
        if train_loss_matches:
            result['train_loss'] = float(train_loss_matches[-1])
        
        # 提取保存模型的epoch（最佳epoch）
        save_model_matches = re.findall(r'Validation loss decreased.*Saving model', content)
        if save_model_matches:
            result['best_epoch'] = len(save_model_matches)
        
    except Exception as e:
        print(f"解析文件 {log_file_path} 时出错: {e}")
    
    return result

def analyze_logs():
    """分析所有日志文件"""
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        print("未找到logs目录！")
        return
    
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    if not log_files:
        print("logs目录中没有找到日志文件！")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    results = []
    for log_file in log_files:
        log_path = os.path.join(logs_dir, log_file)
        result = parse_log_file(log_path)
        if result['parameters'] is not None:  # 只包含有效的结果
            results.append(result)
    
    if not results:
        print("没有找到有效的实验结果！")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存完整结果
    output_file = 'experiment_results.csv'
    df.to_csv(output_file, index=False)
    print(f"完整结果已保存到 {output_file}")
    
    # 显示统计信息
    print("\n" + "="*80)
    print("实验结果统计")
    print("="*80)
    
    print("\n参数量统计:")
    if 'parameters' in df.columns:
        print(f"最小参数量: {df['parameters'].min():,}")
        print(f"最大参数量: {df['parameters'].max():,}")
        print(f"平均参数量: {df['parameters'].mean():,.0f}")
    
    print("\n测试损失统计:")
    if 'test_loss' in df.columns and df['test_loss'].notna().any():
        print(f"最小测试损失: {df['test_loss'].min():.6f}")
        print(f"最大测试损失: {df['test_loss'].max():.6f}")
        print(f"平均测试损失: {df['test_loss'].mean():.6f}")
        
        # 找到最佳模型
        best_idx = df['test_loss'].idxmin()
        best_model = df.iloc[best_idx]
        print("\n最佳模型配置:")
        print(f"  文件名: {best_model['filename']}")
        print(f"  测试损失: {best_model['test_loss']:.6f}")
        print(f"  参数量: {int(best_model['parameters']):,}")
        print(f"  学习率: {best_model['learning_rate']}")
        print(f"  模型维度: {best_model['d_model']}")
        print(f"  注意力头数: {best_model['n_heads']}")
        print(f"  编码器层数: {best_model['e_layers']}")
    
    # 创建可视化
    create_visualizations(df)
    
    return df

def create_visualizations(df):
    """创建可视化图表"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MTSN_slim 实验结果分析', fontsize=16, fontweight='bold')
    
    # 参数量 vs 测试损失
    if 'parameters' in df.columns and 'test_loss' in df.columns:
        axes[0, 0].scatter(df['parameters'], df['test_loss'], alpha=0.7, s=50)
        axes[0, 0].set_xlabel('参数量')
        axes[0, 0].set_ylabel('测试损失')
        axes[0, 0].set_title('参数量 vs 测试损失')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 学习率 vs 测试损失
    if 'learning_rate' in df.columns and 'test_loss' in df.columns:
        axes[0, 1].scatter(df['learning_rate'], df['test_loss'], alpha=0.7, s=50)
        axes[0, 1].set_xlabel('学习率')
        axes[0, 1].set_ylabel('测试损失')
        axes[0, 1].set_title('学习率 vs 测试损失')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # d_model vs 测试损失
    if 'd_model' in df.columns and 'test_loss' in df.columns:
        df_grouped = df.groupby('d_model')['test_loss'].mean().reset_index()
        axes[1, 0].bar(df_grouped['d_model'], df_grouped['test_loss'], alpha=0.7)
        axes[1, 0].set_xlabel('模型维度 (d_model)')
        axes[1, 0].set_ylabel('平均测试损失')
        axes[1, 0].set_title('模型维度 vs 平均测试损失')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 参数量分布
    if 'parameters' in df.columns:
        axes[1, 1].hist(df['parameters'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('参数量')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('参数量分布')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_analysis.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为 experiment_analysis.png")
    plt.show()

def generate_report(df):
    """生成分析报告"""
    report_file = 'experiment_report.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# MTSN_slim 实验结果报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验概述\n\n")
        f.write(f"- 总实验数: {len(df)}\n")
        
        if 'parameters' in df.columns:
            f.write(f"- 参数量范围: {df['parameters'].min():,} - {df['parameters'].max():,}\n")
        
        if 'test_loss' in df.columns and df['test_loss'].notna().any():
            f.write(f"- 测试损失范围: {df['test_loss'].min():.6f} - {df['test_loss'].max():.6f}\n")
        
        f.write("\n## 最佳模型\n\n")
        if 'test_loss' in df.columns and df['test_loss'].notna().any():
            best_idx = df['test_loss'].idxmin()
            best_model = df.iloc[best_idx]
            f.write(f"- 测试损失: {best_model['test_loss']:.6f}\n")
            f.write(f"- 参数量: {int(best_model['parameters']):,}\n")
            f.write(f"- 学习率: {best_model['learning_rate']}\n")
            f.write(f"- 模型维度: {best_model['d_model']}\n")
            f.write(f"- 注意力头数: {best_model['n_heads']}\n")
            f.write(f"- 编码器层数: {best_model['e_layers']}\n")
        
        f.write("\n## 参数影响分析\n\n")
        
        # 各参数对性能的影响
        for param in ['d_model', 'n_heads', 'e_layers', 'learning_rate']:
            if param in df.columns and 'test_loss' in df.columns:
                grouped = df.groupby(param)['test_loss'].agg(['mean', 'std', 'count']).reset_index()
                f.write(f"### {param} 影响\n\n")
                f.write("| 值 | 平均测试损失 | 标准差 | 实验次数 |\n")
                f.write("|---|---|---|---|\n")
                for _, row in grouped.iterrows():
                    f.write(f"| {row[param]} | {row['mean']:.6f} | {row['std']:.6f} | {row['count']} |\n")
                f.write("\n")
    
    print(f"详细报告已保存为 {report_file}")

if __name__ == "__main__":
    print("开始分析实验结果...")
    df = analyze_logs()
    
    if df is not None and len(df) > 0:
        generate_report(df)
        print("\n分析完成！")
        print("- 完整结果: experiment_results.csv")
        print("- 可视化图表: experiment_analysis.png") 
        print("- 详细报告: experiment_report.md")
    else:
        print("没有找到有效的实验结果进行分析。")
