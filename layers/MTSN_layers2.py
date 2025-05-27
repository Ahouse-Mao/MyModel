import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable
from torch import Tensor

class Flatten_Head(nn.Module):
    """
    独立模式适用于变量差异大的情况(例如温度和压力传感器等),适用于多任务学习
    共享模式适用于变量相关性高,适用于单任务多变量预测
    变量数过多建议共享模式,变量数少可采用独立模式

    5.21
    新增了新增长时序和短时序的拼接模式
    
    """
    # TODO：如果要独立处理，能否在这里进行并行化处理，提升处理速度？
    def __init__(self, individual, n_vars, nf, target_window, combie_mode, head_dropout=0):
        super(Flatten_Head, self).__init__()

        self.individual = individual # 是否独立处理每个变量
        self.n_vars = n_vars # 变量总数
        self.combie_mode = combie_mode # 拼接模式

        if self.individual: # 独立处理模式
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars): # 创建每个变量的处理链
                self.flattens.append(nn.Flatten(start_dim=-2)) # 把通道数和patchnum两个维度进行合并
                self.linears.append(nn.Linear(nf, target_window)) # 把特征维度映射到目标窗口大小
                self.dropouts.append(nn.Dropout(head_dropout)) # dropout层
        else: # 共享处理模式
            self.flatten = nn.Flatten(start_dim=-2) # 把通道数和patchnum两个维度进行合并
            self.linear = nn.Linear(nf, target_window) # 把特征维度映射到目标窗口大小
            self.dropout = nn.Dropout(head_dropout) # dropout层

    def forward(self, x, x_long):  # x: [bs x nvars x d_model x patch_num]
        # 新增长时序和短时序的拼接模式
        if self.combie_mode == 'concat':
            x = torch.concat((x, x_long), dim=-1) # 把长序列和短序列拼接在一起
        elif self.combie_mode == 'add':
            x = x + x_long # 把长序列和短序列相加在一起

        if self.individual: 
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    # 封装一个conv1d函数
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    # 返回一个BatchNorm1d的函数
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
        # 类似的保持特征图不变的padding
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels)) # name表示能让子模块以result.bn的形式被调用
    return result

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7): # small_kernel_merged小核是否合并参数，训练/推理模式
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size 
        self.small_kernel = small_kernel

        padding = kernel_size // 2 # 保持特征图尺寸不变的对称填充，可以修改
        if small_kernel_merged: # 推理模式，使用等效卷积
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else: # 训练模式，构建多分支结构, bn函数只是为了构建2个分支更加便捷
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False) # 主分支，大核卷积+BN
            if small_kernel is not None: # 辅助分支，小核卷积+BN
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'): # hasattr()函数用于判断对象是否包含对应的属性
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs) # 并行结构的特征融合

        return out


class Block(nn.Module):
    """
    借鉴自ModernTCN核心部分
    """
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel
    def forward(self,x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        x = self.dw(x) # depth wise卷积，各变量各通道间独立建模
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x)) # convffn1对通道间关系进行建模
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x)) # convffn2对变量间关系进行建模
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    """
    Stage类主要是对Block块进行封装, 每个Stage包含多个Block块, 每个Stage对应不同尺度的卷积
    """
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x
    

def load_balancing_loss_func(
        gate_logits: torch.Tensor, # 传入的gate_logits可以是一个tensor，也可以是一个tuple或者list
        # 如果是tensor则要求形状为(batch_size*seq_len, num_experts)
        top_k: int,
        num_experts: int = None,
) -> torch.Tensor:

    if gate_logits is None: # 检查gate_logits是否有效，无效则返回0.0
        return 0.0

    # Concat the logits from all layers
    # compute_device = gate_logits[0].device

    # concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1) # 最后一维应用softmax

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1) # 选取top_k个专家

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts) # 将top_k个专家的索引转换为one-hot编码

    # 计算tok_k每次选择中选择每个专家的路由概率
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # 综合top_k次选择中每个专家的路由概率，由routing_weights的均值得到，给出了门控单元给予每个专家的概率
    router_prob_per_expert = torch.mean(routing_weights, dim=0) 

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0)) # 最终损失的计算通过广播机制实现

    return overall_loss * num_experts


class FeatureAblation:
    def __init__(self, forward_func: Callable):
        self.forward_fuc = forward_func
        self.use_weights = False

    def attribute(
            self,
            inputs: Tensor,
            baselines: Optional[Tensor] = None, # 基线参考值
            feature_mask: Optional[Tensor] = None, # 特征掩码，指定哪些特征需扰动，默认每个patch为独立特征组
            perturbations_per_eval: int = 1, # 每次前向传播扰动的特征数
    ) -> Tensor:
        
        if baselines is None: # 如果没有提供基线参考值，则使用全零张量
            baselines = torch.zeros_like(inputs)
        
        # 初始未扰动输出
        initial_output = self.forward_fuc(inputs)
        attributions = torch.zeros_like(inputs)

        # 生成特征掩码(默认每个patch为独立的特征组)
        if feature_mask is None:
            B, M, D, N = inputs.shape
            feature_mask = torch.arrange(N).expand(B, M, D, -1).to(inputs.device) # 每个patch作为独立特征

        # 遍历每个特征组
        unique_features = torch.unique(feature_mask)
        for feat in unique_features:
            mask = (feature_mask == feat)
            # 扰动输入：替换为基线值
            modified_input = inputs * (~mask) + baselines * mask
            modified_output = self.forward_fuc(modified_input)
            # 归因值 = 初始输出 - 扰动输出
            diff = (initial_output - modified_output).unsqueeze(-1) # 扩展维度以广播
            attributions += diff * mask # 累加到对应位置

        return attributions

