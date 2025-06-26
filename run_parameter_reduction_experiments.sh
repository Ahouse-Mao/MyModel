#!/bin/bash

# MTSN_slim 参数量减少实验脚本
# 专门用于测试减少参数量来缓解过拟合的效果

echo "=========================================="
echo "开始 MTSN_slim 参数量减少实验"
echo "=========================================="

# 创建实验结果目录
mkdir -p experiments_results
mkdir -p logs

# 定义减少参数量的实验配置
# 实验1: 原始配置（作为基准）
echo "实验1: 原始配置（基准）"
echo "参数: d_model=16, n_heads=8, e_layers=2, ffn_ratio=4"
echo "=========================================="
python run_MTSN_slim.py \
    --d_model 16 \
    --n_heads 8 \
    --e_layers 2 \
    --ffn_ratio 4 \
    --num_blocks 1 \
    --train_epochs 15 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des "baseline"

# 实验2: 减少模型维度
echo ""
echo "实验2: 减少模型维度"
echo "参数: d_model=8, n_heads=4, e_layers=2, ffn_ratio=4"
echo "=========================================="
python run_MTSN_slim.py \
    --d_model 8 \
    --n_heads 4 \
    --e_layers 2 \
    --ffn_ratio 4 \
    --num_blocks 1 \
    --train_epochs 15 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des "reduce_dim"

# 实验3: 减少层数
echo ""
echo "实验3: 减少层数"
echo "参数: d_model=16, n_heads=8, e_layers=1, ffn_ratio=4"
echo "=========================================="
python run_MTSN_slim.py \
    --d_model 16 \
    --n_heads 8 \
    --e_layers 1 \
    --ffn_ratio 4 \
    --num_blocks 1 \
    --train_epochs 15 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des "reduce_layers"

# 实验4: 减少FFN比例
echo ""
echo "实验4: 减少FFN比例"
echo "参数: d_model=16, n_heads=8, e_layers=2, ffn_ratio=2"
echo "=========================================="
python run_MTSN_slim.py \
    --d_model 16 \
    --n_heads 8 \
    --e_layers 2 \
    --ffn_ratio 2 \
    --num_blocks 1 \
    --train_epochs 15 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des "reduce_ffn"

# 实验5: 综合减少（最小配置）
echo ""
echo "实验5: 综合减少（最小配置）"
echo "参数: d_model=8, n_heads=4, e_layers=1, ffn_ratio=2"
echo "=========================================="
python run_MTSN_slim.py \
    --d_model 8 \
    --n_heads 4 \
    --e_layers 1 \
    --ffn_ratio 2 \
    --num_blocks 1 \
    --train_epochs 15 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des "minimal"

# 实验6: 增加正则化的小模型
echo ""
echo "实验6: 增加正则化的小模型"
echo "参数: d_model=8, n_heads=4, e_layers=1, 增加dropout"
echo "=========================================="
python run_MTSN_slim.py \
    --d_model 8 \
    --n_heads 4 \
    --e_layers 1 \
    --ffn_ratio 2 \
    --num_blocks 1 \
    --dropout 0.3 \
    --head_dropout 0.2 \
    --train_epochs 15 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des "minimal_regularized"

echo ""
echo "=========================================="
echo "参数量减少实验完成！"
echo "日志文件保存在 logs/ 目录下"
echo "使用 'python analyze_results.py' 来分析实验结果"
echo "=========================================="
