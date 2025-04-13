import torch.nn as nn
import torch



class MTSN_short(nn.Module):
    def __init__(self, seq_R, freq, c_in, c_out, period):
        super(MTSN_short, self).__init__()
        
        self.seq_R = seq_R
        self.freq = freq
        self.c_out = c_out
        self.period = period

        if freq == 't':
            dim_time = 5
        elif freq == 'h':
            dim_time = 4
        elif freq == 'd':
            dim_time = 3
        
        self.fc_row = nn.Conv1d(
            in_channels = c_in * (1 + dim_time), # 输入数据会与时间特征拼接，所以通道数会变多
            out_channels = c_in * c_out, # 输出通道，即卷积核数量
            kernel_size = period, # 这个是卷积核大小
            stride = 1, groups = c_in # group = c_in，表示每个输入通道的卷积核都是独立的
        )

        self.fc_col = nn.Conv1d(
            in_channels = c_in * c_out, 
            out_channels = c_in * c_out,
            kernel_size = seq_R, 
            stride = 1, groups = c_in 
        )

    def forward(self, x, x_mark):
        B, R, C, c_in, _ = x.shape
        c_time = x_mark.shape[-1]
        x_input = torch.cat([x, x_mark], dim=-1)
        out = self.fc_row(x_input.permute(0, 1, 3, 4, 2).reshape(
            B * R, c_in * (1 + c_time), C)).reshape(B, R, c_in * self.c_out)
        out = self.fc_col(out.permute(0, 2, 1)).reshape(
            B, c_in, 1, self.c_out).repeat(1, 1, self.period, 1)
        return out.permute(0, 2, 1, 3)



