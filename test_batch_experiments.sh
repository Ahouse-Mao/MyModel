#!/bin/bash

# 快速测试批量实验功能
echo "=========================================="
echo "测试批量实验系统"
echo "=========================================="

# 创建目录
mkdir -p logs

# 定义测试参数（少量参数以快速验证）
learning_rates=(0.0001)
d_models=(8 16)
n_heads=(4)
e_layers=(1)
seq_lens=(168)
pred_lens=(168)

# 计算实验数量
total_experiments=$((${#learning_rates[@]} * ${#d_models[@]} * ${#n_heads[@]} * ${#e_layers[@]} * ${#seq_lens[@]} * ${#pred_lens[@]}))
current_experiment=0

echo "总共运行 $total_experiments 个测试实验"
echo ""

# 循环运行实验
for lr in "${learning_rates[@]}"; do
    for dm in "${d_models[@]}"; do
        for nh in "${n_heads[@]}"; do
            for el in "${e_layers[@]}"; do
                for sl in "${seq_lens[@]}"; do
                    for pl in "${pred_lens[@]}"; do
                        current_experiment=$((current_experiment + 1))
                        
                        echo "=========================================="
                        echo "测试实验 $current_experiment/$total_experiments"
                        echo "参数: lr=$lr, d_model=$dm, n_heads=$nh, e_layers=$el"
                        echo "=========================================="
                        
                        # 检查d_model能否被n_heads整除
                        if [ $((dm % nh)) -ne 0 ]; then
                            echo "跳过: d_model($dm)不能被n_heads($nh)整除"
                            continue
                        fi
                        
                        # 运行快速测试（只训练1个epoch）
                        python run_MTSN_slim.py \
                            --learning_rate $lr \
                            --d_model $dm \
                            --n_heads $nh \
                            --e_layers $el \
                            --seq_len $sl \
                            --pred_len $pl \
                            --train_epochs 1 \
                            --patience 1 \
                            --des "test_batch_${current_experiment}"
                        
                        if [ $? -eq 0 ]; then
                            echo "✅ 测试实验 $current_experiment 成功"
                        else
                            echo "❌ 测试实验 $current_experiment 失败"
                        fi
                        
                        echo ""
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "批量实验测试完成！"
echo "检查 logs/ 目录中的日志文件"
echo "=========================================="
