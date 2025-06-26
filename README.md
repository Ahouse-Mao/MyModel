# MTSN_slim 实验自动化系统

本项目提供了一个完整的自动化实验系统，用于MTSN_slim模型的参数调优和过拟合问题解决。

## 功能特性

### 🔧 日志记录功能
- 自动保存所有终端输出到日志文件
- 日志文件名包含关键参数信息，便于识别
- 在日志开头显著显示关键实验参数

### 📊 批量实验支持
- 自动化批量参数测试
- 支持learning_rate, d_model, n_heads, e_layers, seq_len, pred_len等参数的组合测试
- 智能参数验证（如d_model能否被n_heads整除）

### 📈 结果分析工具
- 自动解析日志文件提取关键信息
- 生成可视化图表分析参数影响
- 导出CSV格式的完整结果
- 生成Markdown格式的详细报告

## 文件说明

### 核心文件
- `run_MTSN_slim.py` - 增强的主训练脚本，包含日志功能
- `run_experiments.sh` - 全面的批量实验脚本
- `run_parameter_reduction_experiments.sh` - 专门用于减少参数量的实验脚本
- `analyze_results.py` - 实验结果分析脚本

### 输出文件
- `logs/` - 日志文件目录
- `experiment_results.csv` - 实验结果汇总
- `experiment_analysis.png` - 可视化分析图表
- `experiment_report.md` - 详细分析报告

## 使用方法

### 1. 解决过拟合问题 - 推荐使用参数减少实验

```bash
# 运行专门的参数减少实验（推荐）
./run_parameter_reduction_experiments.sh
```

这个脚本会测试以下配置来减少参数量：
- **基准配置**: d_model=16, n_heads=8, e_layers=2
- **减少维度**: d_model=8, n_heads=4, e_layers=2  
- **减少层数**: d_model=16, n_heads=8, e_layers=1
- **减少FFN**: ffn_ratio=2
- **最小配置**: d_model=8, n_heads=4, e_layers=1
- **增加正则化**: dropout=0.3, head_dropout=0.2

### 2. 全面参数搜索

```bash
# 运行全面的参数组合实验
./run_experiments.sh
```

### 3. 单次实验（带日志）

```bash
# 运行单次实验，所有输出会自动保存到日志文件
python run_MTSN_slim.py --d_model 8 --n_heads 4 --e_layers 1 --learning_rate 0.0001
```

### 4. 分析实验结果

```bash
# 分析所有实验结果
python analyze_results.py
```

## 日志文件命名规则

日志文件按以下格式命名：
```
logs/MTSN_slim_lr{learning_rate}_dm{d_model}_nh{n_heads}_el{e_layers}_sl{seq_len}_pl{pred_len}_{timestamp}.log
```

例如：
```
logs/MTSN_slim_lr0.0001_dm8_nh4_el1_sl168_pl168_20241220_143052.log
```

## 参数量减少策略

针对当前232,945,901个参数的过拟合问题，推荐以下策略：

### 🎯 优先级1：减少模型维度
```bash
--d_model 8 --n_heads 4  # 从d_model=16, n_heads=16减少
```

### 🎯 优先级2：减少层数
```bash
--e_layers 1  # 从e_layers=3减少到1
```

### 🎯 优先级3：减少FFN比例
```bash
--ffn_ratio 2  # 从ffn_ratio=8减少到2
```

### 🎯 优先级4：增加正则化
```bash
--dropout 0.3 --head_dropout 0.2  # 增加dropout
```

## 参数量减少效果（实测）

根据实际测试结果：

- **原始配置**: 232.9M参数 (d_model=16/32, n_heads=16, e_layers=3)
- **优化配置1**: 54.3M参数 (d_model=8, n_heads=4, e_layers=1) - **减少76.7%**
- **优化配置2**: 111.2M参数 (d_model=16, n_heads=4, e_layers=1) - **减少52.3%**

### 性能对比：
- **d_model=8**: MAE=0.3488, MSE=0.2566, 训练时间=53秒/epoch
- **d_model=16**: MAE=0.3186, MSE=0.2187, 训练时间=89秒/epoch

**推荐**: 使用d_model=8配置，在大幅减少参数量的同时保持合理的性能。

## 实验结果监控

### 关键指标
1. **参数量**: 目标是大幅减少到百万以下
2. **测试损失**: 监控过拟合程度
3. **训练/验证损失差距**: 评估过拟合严重程度
4. **收敛速度**: 观察模型训练效率

### 日志中关键信息
```
================================================================================
         MTSN_slim 实验开始
================================================================================
关键实验参数:
  Learning Rate:    0.0001
  Model Dimension:  8
  Attention Heads:  4
  Encoder Layers:   1
  Sequence Length:  168
  Prediction Length:168
================================================================================
```

## 故障排除

### 常见问题
1. **d_model不能被n_heads整除**: 脚本会自动跳过无效组合
2. **内存不足**: 减少batch_size或使用更小的模型配置
3. **GPU内存不足**: 进一步减少参数量或使用CPU训练

### 调试技巧
- 检查logs目录中的最新日志文件
- 使用`analyze_results.py`快速查看所有实验结果
- 单独运行失败的配置进行调试

## 下一步建议

1. 首先运行 `run_parameter_reduction_experiments.sh`
2. 使用 `analyze_results.py` 分析结果
3. 选择参数量和性能平衡最好的配置
4. 针对最佳配置进行fine-tuning

祝实验顺利！🚀
