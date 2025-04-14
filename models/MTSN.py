import torch
from torch import nn
from layers.MTSN_long import MTSN_long
from layers.MTSN_short import MTSN_short
import torch.nn.functional as F
from layers.MTSN_layers2 import LayerNorm
from layers.MTSN_layers2 import Stage
from layers.MTSN_layers2 import Flatten_Head
from layers.RevIN import RevIN



class MTSNMoeSparseExpertsLayer(nn.Module):
    """这里暂时只考虑Long的MOE"""
    def __init__(self, patch_num, 
                 patch_len,
                 c_in,
                 top_k, num_experts, hidden_size,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
                 attn_dropout=0., dropout= 0, act='gelu',
                 res_attention=True, pre_norm=False, store_attn=False,
                ):
        super(MTSNMoeSparseExpertsLayer, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.norm_topk_prob = False # 是否对topk权重进行归一化
        

        self.gates = nn.Linear(self.hidden_size, self.num_experts)
        self.experts = nn.ModuleList(
            [MTSN_long(
                patch_len, patch_num, # patch参数
                c_in,  
                n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
                attn_dropout=0., dropout= 0, act='gelu',
                res_attention=True, pre_norm=False, store_attn=False,
            ) for _ in range(self.num_experts)]
        )
        self.shared_expert = MTSN_long(
                patch_len, patch_num, # patch参数
                c_in,  
                n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
                attn_dropout=0., dropout= 0, act='gelu',
                res_attention=True, pre_norm=False, store_attn=False,
        )

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
                top_k, num_experts, # moe参数
                dims, patch_size, patch_stride,  downsample_ratio, # stem和downsample参数
                num_blocks, ffn_ratio, large_size, small_size, dw_dims, nvars, # block参数
                seq_len, individual=False, target_window=96, head_dropout=0.1, # head参数
                num_downsample = 3, # stem和downsample参数
                small_kernel_merged=False, backbone_dropout=0.1, # block参数
                revin=True, affine=True, subtract_last=False): # RevIN参数
        self.nvars = nvars # 变量个数
        self.patch_size = patch_size
        self.target_window = target_window


        
        super(MTSN, self).__init__()

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(nvars, affine=affine, subtract_last=subtract_last)

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

        # 定义backbone_conv
        self.num_stage = num_blocks
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks, large_size, small_size, dmodel=dims,
                          dw_model=dw_dims, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)  

        # 定义MTSN_LongMOE
        self.patch_num = (seq_len + patch_size - patch_stride - patch_size) // patch_stride + 1 # 修改了一下patch_num的计算方式
        self.LongMOE = MTSNMoeSparseExpertsLayer( patch_num=self.patch_num, patch_len=patch_size, c_in=nvars,
                                                 top_k=top_k, num_experts=num_experts, hidden_size=dims*nvars) # 修改hidden_size的计算方式，从dims变为dims*nvars

        # 定义flattenhead
        self.n_vars = nvars
        self.individual = individual
        d_model = dims
        if self.patch_num % pow(downsample_ratio,(self.num_stage - 1)) == 0:
            self.head_nf = d_model * self.patch_num // pow(downsample_ratio,(self.num_stage - 1))
        else:
            self.head_nf = d_model * (self.patch_num // pow(downsample_ratio, (self.num_stage - 1))+1)    
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

        x = x.unsqueeze(-2)
        x_trans = torch.empty(0)
        for i in range(self.num_stage):
            B, M, D, N = x.shape # D是每个变量的通道个数，N是分patch后的patch个数
            x = x.reshape(B * M, D, N)
            if i==0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride # pad_len的确定只是为了保证切分patch到最后时，万一剩余的序列长度<patch_stride,则会损失数据，而pad后则不会损失
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
            
            if i ==0:
                x_trans = x
        
        x_long = self.LongMOE(x_trans)
        
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

        # 这下面是ModernTCN的参数
        self.dims = configs.enc_in

        # 下采样相关
        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio

        # RevIN归一化相关
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        # 模型输入输出相关
        self.c_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.target_window = configs.pred_len

        # 模型内部维度转换参数
        self.dims = configs.d_model
        self.dw_dims = configs.d_model
        self.num_blocks = configs.num_blocks

        # 卷积相关参数设置
        self.large_size = configs.large_size
        self.small_size = configs.small_size

        self.individual = configs.individual

        self.small_kernel_merged = configs.small_kernel_merged
        self.dropout = configs.dropout

        # patch参数
        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride

        # MOE专家模型参数
        self.top_k = configs.top_k
        self.num_experts = configs.num_experts
        self.ffn_ratio = configs.ffn_ratio

        # head参数
        self.head_dropout = configs.head_dropout

        self.model = MTSN(top_k=self.top_k, num_experts=self.num_experts, ffn_ratio=self.ffn_ratio,
                          dims=self.dims, patch_size=self.patch_size, patch_stride=self.patch_stride, downsample_ratio=self.downsample_ratio,
                          num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dw_dims=self.dw_dims, nvars=self.c_in,
                          seq_len=self.seq_len, individual=self.individual, target_window=self.target_window
        )
    
    def forward(self, x , x_mark, dec_inp, batch_y_mark):

        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)

        return x





