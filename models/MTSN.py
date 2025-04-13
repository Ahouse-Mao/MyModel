import torch
from torch import nn
from layers.MTSN_long import MTSN_long
from layers.MTSN_short import MTSN_short
import torch.nn.functional as F
from layers.RevIN import RevIN

# TODO：MTSNMoeSparseExpertsLayer的forward部分有待修改内容

class MTSNMoeSparseExpertsLayer(nn.Module):
    'top_k, num_experts, hidden_size'
    def __init__(self, configs, block_name,
                 context_window, patch_len, stride, padding_patch, # patch参数
                 seq_R, freq, c_in, c_out, period, # short的参数
                 target_window, # flatten层参数
                 max_seq_len=1024, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
                 norm='BatchNorm', attn_dropout=0., dropout= 0, act='gelu', key_padding_mask='auto',
                 padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                 pe='zeros', learn_pe=True, verbose=False, fc_dropout=0., # TSTiEncoder参数
                 pretrain_head=False, head_type='flatten', individual=False, head_dropout=0, # flatten层的参数
                 revin=True, affine=True, subtract_last=False, **kwargs):
        super(MTSNMoeSparseExpertsLayer, self).__init__()
        self.comfigs = configs
        self.top_k = configs.top_k
        self.moe_num_experts = configs.num_experts
        self.hidden_size = configs.hidden_size
        self.norm_topk_prob = False # 是否对topk权重进行归一化

        self.gates = nn.Linear(self.hidden_size, self.num_experts)
        if block_name == "MTSN_long":
            self.experts = nn.ModuleList(
                [MTSN_long(
                    context_window, patch_len, stride, padding_patch, # patch参数
                    c_in,
                    target_window, # flatten层参数
                    max_seq_len=1024, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
                    norm='BatchNorm', attn_dropout=0., dropout= 0, act='gelu', key_padding_mask='auto',
                    padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                    pe='zeros', learn_pe=True, verbose=False, fc_dropout=0., # TSTiEncoder参数
                    pretrain_head=False, head_type='flatten', individual=False, head_dropout=0, # flatten层的参数
                    revin=True, affine=True, subtract_last=False, **kwargs # Revin归一化参数
                ) for _ in range(self.num_experts)]
            )
            self.shared_expert = MTSN_long(
                    context_window, patch_len, stride, padding_patch, # patch参数
                    c_in,
                    target_window, # flatten层参数
                    max_seq_len=1024, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
                    norm='BatchNorm', attn_dropout=0., dropout= 0, act='gelu', key_padding_mask='auto',
                    padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                    pe='zeros', learn_pe=True, verbose=False, fc_dropout=0., # TSTiEncoder参数
                    pretrain_head=False, head_type='flatten', individual=False, head_dropout=0, # flatten层的参数
                    revin=True, affine=True, subtract_last=False, **kwargs # Revin归一化参数
                )
        elif block_name == "MTSN_short":
            self.experts = nn.ModuleList(
                [MTSN_short(
                    seq_R, freq, c_in, c_out, period
                )for _ in range(self.num_experts)]
            )
            self.shared_expert = MTSN_short(seq_R, freq, c_in, c_out, period)

        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape # hidden_states: [batch_size, sequence_length, hidden_dim]
        hidden_states = hidden_states.view(-1, hidden_dim) # [batch_size*sequence_length, hidden_dim]
        router_logits = self.gates(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, detype=torch.float) # [batch_size*sequence_length, num_experts]
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts, dim=-1) # [batch_size*sequence_length, top_k]
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # 转回输入时的数据类型
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size*sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 使用one-hot来编码被选择的专家
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # num_classes指定one-hot编码类别数，即专家总数。one-hot函数为selected_experts中的每一个索引，生成一个长度为num_experts的向量，对应索引位置为1，其余位置为0
        # 得到一个(batch_size * sequence_length, top_k, self.num_experts)的one-hot向量
        # 每个(i, j)位置的向量表示第i个输入的第j个选定专家的one-hot编码
        # permute(2, 1, 0)将one-hot向量从(batch_size * sequence_length, top_k, self.num_experts)
        # 变为(self.num_experts, top_k, batch_size * sequence_length)
        # 把num_experts放到第0维度，方便后续按专家索引循环处理

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx] # 从expert候选里选择一个expert模型，只对选中的专家进行前向传播计算
            idx, top_x = torch.where(expert_mask[expert_idx]) # expert_mask[expert_idx]选中对应expert的数据，torch.where(condition) 返回满足条件（值为 1）的所有位置的坐标。
            # idx(num_selected,), 表示该专家在top_k中的位置，范围[0, top_k-1];
            # top_x(num_selected,)表示该专家的输入索引，范围[0, batch_size*sequence_length-1]
            
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            # hidden_states[top_x]把hidden_states中对应top_x索引的元素取出来，得到一个(num_selected, hidden_dim)的向量
            # hidden_states[None, top_x]加入了None后，相当于在第0维增加了一个维度，变成(1, num_selected, hidden_dim), None等价于unsqueeze(0)
            # reshape(-1, hidden_dim)重新展平为二维向量，形状是(num_selected, hidden_dim)

            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] # 对选中的专家进行前向传播计算, 并乘以routing_weights
            # top_x对应输入位置索引，idx对应在top_k中的位置索引, routing_weights[top_x, idx, None]返回的是(num_selected, 1)的向量，None增加了一个维度，便于广播

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.detype)) # 将计算结果加到final_hidden_states中，top_x对应输入位置索引，current_hidden_states对应计算结果
            # 0指定了进行加法操作的维度，top_x指定了加法操作的位置索引，current_hidden_states指定了加法操作的值
            # TODO：这里要注意在进行加法操作时注意专家计算回来的结果的形状，patchTST有可能会改变形状
        
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # TODO：这里的展平操作可能有待修改
        return final_hidden_states, router_logits




class Flatten_Head(nn.Module):
    """
    独立模式适用于变量差异大的情况(例如温度和压力传感器等),适用于多任务学习
    共享模式适用于变量相关性高,适用于单任务多变量预测
    变量数过多建议共享模式,变量数少可采用独立模式
    """
    # TODO：如果要独立处理，能否在这里进行并行化处理，提升处理速度？
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(Flatten_Head, self).__init__()

        self.individual = individual # 是否独立处理每个变量
        self.n_vars = n_vars # 变量总数
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

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
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


class MTSN(nn.Module):
    """
    MTSN主结构
    大概要完成以下任务：
    1.对于数据的处理任务
    2.对于长时序短时序的模型的初始化
    3.整合2个时序任务的输出and计算最终的输出
    4.计算MOE的损失函数
    5.对变量间相关性进行建模

    新加入了参数num_downsample
    """
    
    def __init__(self,
                dims, patch_size, patch_stride,  downsample_ratio, # stem和downsample参数
                num_blocks, ffn_ratio, large_size, small_size, dw_dims, nvars, # block参数
                seq_len=512, individual=False, target_window=96, head_dropout=0.1, # head参数
                num_downsample = 3, # stem和downsample参数
                small_kernel_merged=False, backbone_dropout=0.1, # block参数
                c_in=7, revin=True, affine=True, subtract_last=False): # RevIN参数
        
        super(MTSN, self).__init__()

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # 定义stem and downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential( # stem是数据第一次输入时进入处理的层
            nn.Conv1d(1, dims, kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims)
        )
        self.downsample_layers.append(stem)
        for i in range(num_downsample):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims),
                nn.Conv1d(dims, dims, kernel_size=downsample_ratio, stride=downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # 定义backbone
        self.num_stage = num_blocks
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks, large_size, small_size, dmodel=dims,
                          dw_model=dw_dims, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)

        # 定义flattenhead
        patch_num = seq_len // patch_stride

        self.n_vars = c_in
        self.individual = individual
        d_model = dims
        if patch_num % pow(downsample_ratio,(self.num_stage - 1)) == 0:
            self.head_nf = d_model * patch_num // pow(downsample_ratio,(self.num_stage - 1))
        else:
            self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1))+1)    
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
    
    def forward(self, x):
        # 1.进入先进行RevIN
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)
        
        # 2.准备进入embedding层及多尺度卷积层
        B,M,L = x.shape # B是batch size，M是变量个数，L是序列长度

        x.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = x.shape # D是每个变量的通道个数，N是分patch后的patch个数
            x = x.reshape(B * M, D, N)
            if i==0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride # pad_len的确定只是为了保证切分patch到最后时，万一剩余的序列长度<patch_stidee,则会损失数据，而pad后则不会损失
                    pad = x[:,:,-1:].repeat(1,1,pad_len)
                    x = torch.cat([x,pad],dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
            x = self.downsample_layers[i](x)
            # downsample_layers里只有第一个是stem层，内含一个conv1d(把channel从1变为dims(d_model))和一个batchNorm1d
            # 接下来的另外几层都是batchNorm1d和一个conv1d(把channel从d_model变为d_model)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        
        # 3.进入FlattenHead层
        x = self.head(x)

        # 4.反归一化
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0, 2, 1)
        return x


    

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs

        self.MTSN = MTSN()





