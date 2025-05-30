import torch
from typing import Optional, Callable

class SimpleFeatureAblation:
    """
    简化版特征消融分析类,用于计算模型中各个特征的重要性。
    支持三种不同的消融模式:
    1. 模式1:计算每个变量、每个通道、每个patch的重要性,输出形状为(B,M,D,N)
    2. 模式2:计算每个变量、每个patch的重要性(通道整体),输出形状为(B,M,N)
    3. 模式3:计算每个变量的重要性,输出形状为(B,M)
    """
    def __init__(self, forward_func: Callable):
        """
        初始化特征消融分析类
        
        Args:
            forward_func: 模型的前向传播函数,接收形状为(B,M,D,N)的输入并返回预测结果
        """
        self.forward_func = forward_func
    
    def compute_patch_importance(self, 
                               inputs: torch.Tensor, 
                               baselines: Optional[torch.Tensor] = None,
                               attr_mode: int = 1) -> torch.Tensor:
        """
        计算每个patch的重要性分数(逐个patch消融)
        
        Args:
            inputs: 输入数据,形状为(B,M,D,N),B是batch_size,M是变量数,
                   D是每个变量的通道个数,N是每个变量在每个通道上的patch数量
            baselines: 基线参考值,如果为None则使用全零张量
            attr_mode: 消融模式,1=每个变量每个通道每个patch,2=每个变量每个patch,3=每个变量
            
        Returns:
            重要性分数,根据attr_mode不同,形状分别为(B,M,D,N)、(B,M,N)或(B,M)
        """
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        
        # 获取原始输出
        with torch.no_grad():
            original_output = self.forward_func(inputs)
            
        B, M, D, N = inputs.shape
        
        # 根据不同模式创建不同形状的结果张量
        if attr_mode == 1:
            # 模式1:每个变量、每个通道、每个patch的重要性
            attributions = torch.zeros_like(inputs)
            
            # 遍历每个变量、每个通道、每个patch
            for m in range(M):
                for d in range(D):
                    for n in range(N):
                        # 创建掩码
                        mask = torch.zeros_like(inputs, dtype=torch.bool)
                        mask[:, m, d, n] = True
                        
                        # 创建修改后的输入
                        modified_input = inputs.clone()
                        modified_input[mask] = baselines[mask]
                        
                        # 计算修改后的输出
                        with torch.no_grad():
                            modified_output = self.forward_func(modified_input)
                        
                        # 计算重要性分数 (原始输出 - 修改后的输出)
                        diff = original_output - modified_output
                        
                        # 对于每个batch样本,将差异分配给对应的特征
                        for b in range(B):
                            attributions[b, m, d, n] = diff[b]
        
        elif attr_mode == 2:
            # 模式2:每个变量、每个patch的重要性(通道整体)
            attributions = torch.zeros((B, M, N), device=inputs.device)
            
            # 遍历每个变量、每个patch
            for m in range(M):
                for n in range(N):
                    # 创建掩码
                    mask = torch.zeros_like(inputs, dtype=torch.bool)
                    mask[:, m, :, n] = True
                    
                    # 创建修改后的输入
                    modified_input = inputs.clone()
                    modified_input[mask] = baselines[mask]
                    
                    # 计算修改后的输出
                    with torch.no_grad():
                        modified_output = self.forward_func(modified_input)
                    
                    # 计算重要性分数
                    diff = original_output - modified_output
                    
                    # 对于每个batch样本,将差异分配给对应的特征
                    for b in range(B):
                        attributions[b, m, n] = diff[b]
        
        else:  # attr_mode == 3
            # 模式3:每个变量的重要性
            attributions = torch.zeros((B, M), device=inputs.device)
            
            # 遍历每个变量
            for m in range(M):
                # 创建掩码
                mask = torch.zeros_like(inputs, dtype=torch.bool)
                mask[:, m, :, :] = True
                
                # 创建修改后的输入
                modified_input = inputs.clone()
                modified_input[mask] = baselines[mask]
                
                # 计算修改后的输出
                with torch.no_grad():
                    modified_output = self.forward_func(modified_input)
                
                # 计算重要性分数
                # TODO: 这里的diff计算可能需要改，如何评价
                diff = original_output - modified_output
                
                # 对于每个batch样本,将差异分配给对应的特征
                for b in range(B):
                    attributions[b, m] = diff[b]
        
        return attributions
    
    def compute_patch_importance_efficient(self,
                                         inputs: torch.Tensor,
                                         baselines: Optional[torch.Tensor] = None,
                                         attr_mode: int = 1,
                                         perturbations_per_eval: int = 1) -> torch.Tensor:
        """
        计算每个patch的重要性分数(批量处理,更高效)
        
        Args:
            inputs: 输入数据,形状为(B,M,D,N),B是batch_size,M是变量数,
                   D是每个变量的通道个数,N是每个变量在每个通道上的patch数量
            baselines: 基线参考值,如果为None则使用全零张量
            attr_mode: 消融模式,1=每个变量每个通道每个patch,2=每个变量每个patch,3=每个变量
            perturbations_per_eval: 每次评估的扰动数量,增大此值可提高计算效率
            
        Returns:
            重要性分数,根据attr_mode不同,形状分别为(B,M,D,N)、(B,M,N)或(B,M)
        """
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        
        # 获取原始输出
        with torch.no_grad():
            original_output = self.forward_func(inputs)
            
        B, M, D, N = inputs.shape
        
        # 根据不同模式创建不同形状的结果张量和处理逻辑
        if attr_mode == 1:
            # 模式1:每个变量、每个通道、每个patch的重要性
            attributions = torch.zeros_like(inputs)
            total_features = M * D * N
            
            # 分批处理特征消融
            for i in range(0, total_features, perturbations_per_eval):
                current_features = min(perturbations_per_eval, total_features - i)
                
                # 创建多个修改后的输入副本
                modified_inputs_batch = []
                for j in range(current_features):
                    feature_idx = i + j
                    if feature_idx >= total_features:
                        break
                        
                    # 计算变量、通道、patch索引
                    m = feature_idx // (D * N)
                    d = (feature_idx % (D * N)) // N
                    n = feature_idx % N
                    
                    # 创建掩码
                    mask = torch.zeros_like(inputs, dtype=torch.bool)
                    mask[:, m, d, n] = True
                    
                    # 创建修改后的输入
                    modified_input = inputs.clone()
                    modified_input[mask] = baselines[mask]
                    modified_inputs_batch.append(modified_input)
                
                # 批量计算修改后的输出
                modified_inputs_batch = torch.cat([input.unsqueeze(0) for input in modified_inputs_batch], dim=0)
                with torch.no_grad():
                    modified_outputs_batch = self.forward_func(modified_inputs_batch.reshape(-1, M, D, N))
                
                # 计算每个特征的重要性分数
                for j in range(current_features):
                    feature_idx = i + j
                    if feature_idx >= total_features:
                        break
                        
                    # 计算变量、通道、patch索引
                    m = feature_idx // (D * N)
                    d = (feature_idx % (D * N)) // N
                    n = feature_idx % N
                    
                    # 计算差异
                    diff = original_output - modified_outputs_batch[j]
                    
                    # 对于每个batch样本,将差异分配给对应的特征
                    for b in range(B):
                        attributions[b, m, d, n] = diff[b]
        
        elif attr_mode == 2:
            # 模式2:每个变量、每个patch的重要性(通道整体)
            attributions = torch.zeros((B, M, N), device=inputs.device)
            total_features = M * N
            
            # 分批处理特征消融
            for i in range(0, total_features, perturbations_per_eval):
                current_features = min(perturbations_per_eval, total_features - i)
                
                # 创建多个修改后的输入副本
                modified_inputs_batch = []
                for j in range(current_features):
                    feature_idx = i + j
                    if feature_idx >= total_features:
                        break
                        
                    # 计算变量、patch索引
                    m = feature_idx // N
                    n = feature_idx % N
                    
                    # 创建掩码
                    mask = torch.zeros_like(inputs, dtype=torch.bool)
                    mask[:, m, :, n] = True
                    
                    # 创建修改后的输入
                    modified_input = inputs.clone()
                    modified_input[mask] = baselines[mask]
                    modified_inputs_batch.append(modified_input)
                
                # 批量计算修改后的输出
                modified_inputs_batch = torch.cat([input.unsqueeze(0) for input in modified_inputs_batch], dim=0)
                with torch.no_grad():
                    modified_outputs_batch = self.forward_func(modified_inputs_batch.reshape(-1, M, D, N))
                
                # 计算每个特征的重要性分数
                for j in range(current_features):
                    feature_idx = i + j
                    if feature_idx >= total_features:
                        break
                        
                    # 计算变量、patch索引
                    m = feature_idx // N
                    n = feature_idx % N
                    
                    # 计算差异
                    diff = original_output - modified_outputs_batch[j]
                    
                    # 对于每个batch样本,将差异分配给对应的特征
                    for b in range(B):
                        attributions[b, m, n] = diff[b]
        
        else:  # attr_mode == 3
            # 模式3:每个变量的重要性
            attributions = torch.zeros((B, M), device=inputs.device)
            total_features = M
            
            # 分批处理特征消融
            for i in range(0, total_features, perturbations_per_eval):
                current_features = min(perturbations_per_eval, total_features - i)
                
                # 创建多个修改后的输入副本
                modified_inputs_batch = []
                for j in range(current_features):
                    feature_idx = i + j
                    if feature_idx >= total_features:
                        break
                        
                    # 计算变量索引
                    m = feature_idx
                    
                    # 创建掩码
                    mask = torch.zeros_like(inputs, dtype=torch.bool)
                    mask[:, m, :, :] = True
                    
                    # 创建修改后的输入
                    modified_input = inputs.clone()
                    modified_input[mask] = baselines[mask]
                    modified_inputs_batch.append(modified_input)
                
                # 批量计算修改后的输出
                modified_inputs_batch = torch.cat([input.unsqueeze(0) for input in modified_inputs_batch], dim=0)
                with torch.no_grad():
                    modified_outputs_batch = self.forward_func(modified_inputs_batch.reshape(-1, M, D, N))
                
                # 计算每个特征的重要性分数
                for j in range(current_features):
                    feature_idx = i + j
                    if feature_idx >= total_features:
                        break
                        
                    # 计算变量索引
                    m = feature_idx
                    
                    # 计算差异
                    diff = original_output - modified_outputs_batch[j]
                    
                    # 对于每个batch样本,将差异分配给对应的特征
                    for b in range(B):
                        attributions[b, m] = diff[b]
        
        return attributions
    
    def attribute(self,
                 inputs: torch.Tensor,
                 baselines: Optional[torch.Tensor] = None,
                 attr_mode: int = 1,
                 efficient: bool = True,
                 perturbations_per_eval: int = 1) -> torch.Tensor:
        """
        计算特征重要性分数的主方法
        
        Args:
            inputs: 输入数据,形状为(B,M,D,N),B是batch_size,M是变量数,
                   D是每个变量的通道个数,N是每个变量在每个通道上的patch数量
            baselines: 基线参考值,如果为None则使用全零张量
            attr_mode: 消融模式,1=每个变量每个通道每个patch,2=每个变量每个patch,3=每个变量
            efficient: 是否使用高效实现,True使用批量处理,False逐个处理
            perturbations_per_eval: 每次评估的扰动数量,增大此值可提高计算效率
            
        Returns:
            重要性分数,根据attr_mode不同,形状分别为(B,M,D,N)、(B,M,N)或(B,M)
        """
        # 验证输入维度
        if inputs.dim() != 4:
            raise ValueError(f"输入必须是四维张量,当前形状: {inputs.shape}")
            
        if efficient:
            return self.compute_patch_importance_efficient(inputs, baselines, attr_mode, perturbations_per_eval)
        else:
            return self.compute_patch_importance(inputs, baselines, attr_mode)