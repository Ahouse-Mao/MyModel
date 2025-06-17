import torch
from torch import nn

import torch.nn.functional as F
from layers.RevIN import RevIN

from layers.MTSN_slim_backbone import Stage, PatchTST_backbone

class MTSN_slim(nn.Module):
    def __init__(self,
                configs,
                top_k, num_experts, use_moe,# moe参数
                n_layers, n_heads, d_ff, # transformer参数
                TST_dropout, fc_dropout, head_dropout,
                dims, patch_size, patch_stride,  downsample_ratio, # stem和downsample参数
                num_blocks, ffn_ratio, nvars, # block参数
                seq_len, individual=False, target_window=96, # head参数
                num_downsample = 3, # stem和downsample参数
                small_kernel_merged=False, backbone_dropout=0.1, # block参数
                revin=True, affine=True, subtract_last=False,# RevIN参数
                #TST 参数
                d_k=None, d_v=None, norm='BatchNorm', attn_dropout=0., 
                act='gelu', res_attention=False, pre_norm=False, store_attn=False,
                pe='zeros', learn_pe=True, pretrain_head=False, head_type='flatten',
                **kwargs
                ): 
        self.nvars = nvars # 变量个数
        self.patch_size = patch_size
        self.target_window = target_window
        self.top_k = top_k
        self.num_experts = num_experts
        self.use_moe = use_moe

        super(MTSN_slim, self).__init__()

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(nvars, affine=affine, subtract_last=subtract_last)

        # 定义Transformer层
        self.Trans_model = PatchTST_backbone(use_moe = self.use_moe, configs=configs, c_in=dims, context_window = seq_len,
                                  target_window=target_window, patch_len=patch_size, stride=patch_stride, 
                                  n_layers=n_layers, d_model=d_ff,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=TST_dropout, act=act,
                                  res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  revin = True, affine = True, subtract_last = False,
                                  pe=pe, learn_pe=learn_pe, **kwargs)

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
            layer = Stage(ffn_ratio=ffn_ratio, num_blocks=num_blocks, dmodel=dims, 
                          nvars=nvars, drop=backbone_dropout)
            self.stages.append(layer)

        # 定义Flattenhead
        patch_num = ((seq_len + patch_size - patch_stride - patch_size) // patch_stride + 1) / self.num_stage # 除以 num_stage 是因为每个stage都处理一次patch
        self.head_nf = int(d_ff * patch_num)
        self.n_vars = nvars
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, dims, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        
        
    def forward(self, x):
        # 1.进入先进行RevIN
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)
        
        # 2.进入patch和embedding
        x = x.unsqueeze(-2)
        all_router_logits = 0
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
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

            # 3.进入Transformer
            if i == 0:
                if self.use_moe:
                    x, all_router_logits = self.Trans_model(x)
                else:
                    x = self.Trans_model(x)
            # 4.进入backbone_conv
            x = self.stages[i](x)

            # 5.进入Flattenhead
        z = self.head(x)

        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)

        if self.use_moe:
            return z, all_router_logits
        else:
            return z
        
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
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
        self.use_moe = configs.use_moe
        self.top_k = configs.top_k
        self.num_experts = configs.num_experts
        self.ffn_ratio = configs.ffn_ratio

        # head参数
        self.head_dropout = configs.head_dropout
        self.combie_mode = configs.combie_mode

        # Transformer参数
        self.n_layers = 3
        self.n_heads = 16
        self.d_ff = 32 # 注意这个参数和configs.d_model等价，涉及修改d_model时该参数一并修改

        self.TST_dropout = configs.dropout
        self.fc_dropout = configs.fc_dropout
        self.head_dropout = configs.head_dropout

        self.model = MTSN_slim(top_k=self.top_k, num_experts=self.num_experts, ffn_ratio=self.ffn_ratio,
                          dims=self.dims, patch_size=self.patch_size, patch_stride=self.patch_stride, downsample_ratio=self.downsample_ratio,
                          num_blocks=self.num_blocks, nvars=self.c_in,
                          seq_len=self.seq_len, individual=self.individual, target_window=self.target_window,
                          n_layers= self.n_layers, n_heads=self.n_heads, d_ff=self.d_ff, use_moe=self.use_moe, combie_mode=self.combie_mode,
                           configs=configs,
                          head_dropout= self.head_dropout, TST_dropout=self.dropout, fc_dropout=self.fc_dropout,
                          revin=self.revin, allfine=self.affine, subtract_last=self.subtract_last,
        )
    
    def forward(self, x , x_mark, dec_inp, batch_y_mark, feat_mask_id=None, fm=None):

        x = x.permute(0, 2, 1)
        if self.use_moe:
            x, all_router_logits = self.model(x)
        else:
            x = self.model(x)
        x = x.permute(0, 2, 1)

        if self.use_moe:
            return x, all_router_logits
        else:
            return x