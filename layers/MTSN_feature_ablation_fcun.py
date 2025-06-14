import torch
import torch.nn.functional as F

class genertate_feature_ablation_mask():
    def __init__(self, args, train_loader_len):
        self.mode = args.feature_ablation_mode
        self.batch_size = args.batch_size
        self.n_vars = args.enc_in
        self.patch_num = (args.seq_len + args.patch_size - args.patch_stride - args.patch_size) // args.patch_stride + 1 

        self.batch_number = 0
        self.use_feature_ablation_mode = args.use_feature_ablation_mode

        self.caculate_mode = args.caculate_mode
        if self.mode == 1:    
            self.abla_range = self.n_vars
            self.scores_tensor = torch.zeros(train_loader_len, self.batch_size, self.n_vars) # 存储数据，存储在类里，避免外部的使用
        elif self.mode == 2:
            self.abla_range = self.n_vars*self.patch_num
            self.scores_tensor = torch.zeros(train_loader_len, self.batch_size, self.n_vars, self.patch_num) # 存储数据，存储在类里，避免外部的使用) 
        self.train_mode =False

    '''
    这里开始是feature ablation scores的获取过程
    '''
    def mask_data(self, data, index):
        idx_1 = index // self.n_vars
        if self.mode == 2:
            idx_2 = index % self.n_vars
        
        if self.mode == 1:
            data[:, idx_1, :, :] = 0
        elif self.mode == 2:
            data[:, idx_1, :, idx_2] = 0
    
        return data
        
    def get_scores_tensor_part(self):
        # 生成容纳一个batch的数据的tensor
        if self.mode == 1:
            scores_tensor = torch.zeros(self.batch_size*10, self.n_vars)
        elif self.mode == 2:
            scores_tensor = torch.zeros(self.batch_size, self.n_vars, self.patch_num)
        return scores_tensor

    def diff_caculation(self, ablation_scores_part, data1, data2, j):
        # 这个函数负责计算相似度和存储数据
        if self.mode == 1:
            # 计算结果形状应当是(batch_size, 1)
            if self.caculate_mode == 1: # 使用余弦相似度计算呢
                data1 = data1.permute(0, 2, 1)
                data2 = data2.permute(0, 2, 1)
                data1_flat = data1.reshape(data1.size(0), -1) # 把形状从(B, Seq, M)变成(B, Seq*M)，方便计算余弦相似度
                data2_flat = data2.reshape(data2.size(0), -1)
                sim = F.cosine_similarity(data1_flat, data2_flat, dim=1) # 计算结果应为(B,)
            elif self.caculate_mode == 2: # 使用MSE计算
                pass # TODO:未实现
        
            ablation_scores_part[:, j] = sim
        elif self.mode == 2:
            pass # TODO:未实现
        return ablation_scores_part

    def add_part(self, ablation_scores_part, index):
        # 这个函数负责把一个batch的数据添加到scores_tensor中
        idx_1 = index // self.n_vars
        if self.mode == 1:
            for i in range(9, -1, -1):
                self.scores_tensor[idx_1 - i, :, :] = ablation_scores_part[(9 - i)*16:(9 - i)*16 + 16, :]

        if self.mode == 2: # TODO:mode2没写完
            idx_2 = index % self.n_vars
    
    '''
    这里开始是结合scores的训练过程
    '''
    def scores_norm(self, ):
        # 这个函数负责把scores_tensor中的数据归一化
        if self.mode == 1:
            self.scores_tensor = F.softmax(self.scores_tensor, dim=2)
    
    def update_batch_number(self, index):
        # 这个函数负责更新batch_number
        self.batch_numbe = index
    
    def feature_weighting(self, data):
        # 这个函数负责根据scores_tensor中的数据来进行feature ablation
        # data是一个batch的数据,形状为(batch_size, n_vars, seq_len)
        # index是当前的index, 从dataloader传入，表示这是第几个batch
        if self.mode == 1:
            if self.use_feature_ablation_mode == 'feature_weighting':
                # 这个模式下, 采用特征加权的方式
                data = data * self.scores_tensor[self.batch_number, :, :] # 形状为(batch_size, n_vars, seq_len)
        return data
    
    # def loss_weighting(self, ):

    '''
    change mode函数用于修改模式,在生成/使用feature ablation间切换
    '''
    def change_train_mode_0(self,):
        self.train_mode = False
    
    def change_train_mode_1(self,):
        self.train_mode = True

    def tensor_is_empty(self, tensor):
        return tensor.numel() == 0

        
        

