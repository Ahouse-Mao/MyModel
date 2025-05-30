import torch
from typing import Optional, Tuple, Union

class FeatureAblation:
    def __init__(self, model):
        self.model = model
        
    def attribute(
        self,
        inputs: torch.Tensor,  # 输入形状: (B,M,D,N)
        attr_mode: str = "mode1",  # ["mode1", "mode2", "mode3"]
        perturbations_per_eval: int = 1,
        target: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        实现三种模式的特征归因计算
        """
        # 验证输入维度
        if inputs.dim() != 4:
            raise ValueError(f"输入必须是四维张量，当前形状: {inputs.shape}")
        
        B, M, D, N = inputs.shape
        
        # 构建特征掩码
        feature_mask = self._build_feature_mask(inputs, attr_mode)
        
        # 初始化归因结果
        attributions = torch.zeros_like(inputs)
        
        # 获取特征分组数量
        total_features = self._get_total_features(feature_mask)
        
        # 基础预测
        base_output = self.model(inputs)
        
        # 分批处理特征消融
        for i in range(0, total_features, perturbations_per_eval):
            current_features = min(perturbations_per_eval, total_features - i)
            
            # 构建消融输入
            ablated_inputs = self._construct_ablated_input(
                inputs, feature_mask, baseline, i, i+current_features
            )
            
            # 前向计算
            ablated_output = self.model(ablated_inputs)
            
            # 计算差异
            diff = base_output - ablated_output
            
            # 聚合差异到对应特征
            attributions = self._aggregate_attributions(
                attributions, diff, feature_mask, i, current_features
            )
        
        # 根据模式返回不同维度结果
        if attr_mode == "mode3":
            return attributions.mean(dim=[2, 3])  # (B,M)
        elif attr_mode == "mode2":
            return attributions.mean(dim=2)      # (B,M,N)
        else:
            return attributions                   # (B,M,D,N)

    def _build_feature_mask(
        self, 
        inputs: torch.Tensor, 
        mode: str
    ) -> torch.Tensor:
        """
        构建不同模式的特征掩码
        """
        B, M, D, N = inputs.shape
        
        if mode == "mode3":
            # 每个变量一个特征组
            return torch.arange(M).repeat_interleave(D*N).view(M, D, N)
        elif mode == "mode2":
            # 每个变量+patch一个特征组
            return torch.arange(M*N).repeat_interleave(D).view(M, N, D).permute(0, 2, 1)
        else:
            # 每个独立特征 (M*D*N)
            return torch.arange(M*D*N).view(M, D, N)

    def _construct_ablated_input(
        self,
        inputs: torch.Tensor,
        feature_mask: torch.Tensor,
        baseline: Optional[torch.Tensor],
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        构造消融输入
        """
        ablated = inputs.clone()
        
        # 获取要消融的特征索引
        mask_slice = feature_mask.unsqueeze(0).expand(inputs.shape)
        
        # 对指定特征进行消融
        for feat_idx in range(start, end):
            ablated[mask_slice == feat_idx] = 0 if baseline is None else baseline
            
        return ablated

    def _get_total_features(self, feature_mask: torch.Tensor) -> int:
        """获取总特征数"""
        return feature_mask.max() + 1

    def _aggregate_attributions(
        self,
        attributions: torch.Tensor,
        diff: torch.Tensor,
        feature_mask: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        将差异结果聚合到对应的特征位置
        """
        result = attributions.clone()
        
        # 扩展特征掩码以匹配输入形状
        expanded_mask = feature_mask.unsqueeze(0).expand(result.shape)
        
        # 将差异值分配到对应的特征位置
        for feat_idx in range(start, end):
            result[expanded_mask == feat_idx] = diff.view(-1)[feat_idx]
            
        return result