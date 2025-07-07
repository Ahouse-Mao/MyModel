#!/bin/bash

# MTSN_slim 批量实验脚本
# 用于测试不同参数组合的效果，每个实验都会生成独立的日志文件

echo "=========================================="
echo "开始 MTSN_slim 批量参数实验"
echo "=========================================="

# 创建实验结果目录
mkdir -p experiments_results
mkdir -p logs

# 定义要测试的参数数组
learning_rates=(0.00005)
d_models=(16)
n_heads=(8)
e_layers=(3)
seq_lens=(336)
pred_lens=(92)

# 计算总实验数量
total_experiments=$((${#learning_rates[@]} * ${#d_models[@]} * ${#n_heads[@]} * ${#e_layers[@]} * ${#seq_lens[@]} * ${#pred_lens[@]}))
current_experiment=0

echo "总共需要运行 $total_experiments 个实验"
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
                        echo "实验 $current_experiment/$total_experiments"
                        echo "参数: lr=$lr, d_model=$dm, n_heads=$nh, e_layers=$el, seq_len=$sl, pred_len=$pl"
                        echo "=========================================="
                        
                        # 检查d_model能否被n_heads整除
                        if [ $((dm % nh)) -ne 0 ]; then
                            echo "跳过: d_model($dm)不能被n_heads($nh)整除"
                            echo ""
                            continue
                        fi
                        
                        # 记录实验开始时间
                        start_time=$(date '+%Y-%m-%d %H:%M:%S')
                        echo "实验开始时间: $start_time"
                        
                        # 运行Python脚本（注意：日志记录现在由Python脚本内部处理）
                        python run_MTSN_slim.py \
                            --learning_rate $lr \
                            --d_model $dm \
                            --n_heads $nh \
                            --e_layers $el \
                            --seq_len $sl \
                            --pred_len $pl \
                            --train_epochs 10 \
                            --patience 3 \
                            --batch_size 16 \
                            --des "batch_exp_${current_experiment}"
                        
                        # 记录实验结果
                        exit_code=$?
                        end_time=$(date '+%Y-%m-%d %H:%M:%S')
                        
                        if [ $exit_code -eq 0 ]; then
                            echo "✅ 实验 $current_experiment 完成"
                            echo "结束时间: $end_time"
                        else
                            echo "❌ 实验 $current_experiment 失败 (退出码: $exit_code)"
                            echo "结束时间: $end_time"
                        fi
                        
                        echo ""
                        
                        # 清理GPU内存（如果需要）
                        sleep 2
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "所有实验完成！"
echo "日志文件保存在 logs/ 目录下"
echo "可以运行 'python analyze_results.py' 来分析实验结果"
echo "=========================================="
