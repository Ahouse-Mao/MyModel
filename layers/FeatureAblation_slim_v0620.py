import torch
import torch.nn as nn
from torch.nn import functional as F

class FeatureAblation_slim(nn.Module):
    def __init__(self, FA_experts, FA_top_k, n_vars):
        super(FeatureAblation_slim, self).__init__()
        self.num_epxerts = FA_experts # 专家个数影响变量的分组
        self.top_k = FA_top_k # 最终选择
        self.n_vars = n_vars # 变量个数

        