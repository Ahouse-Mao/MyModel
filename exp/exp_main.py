import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TPGN, iTransformer, ModernTCN, PatchTST, MTSN
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv
from utils.metrics import metric

warnings.filterwarnings('ignore')

"""
主实验模块
"""

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # 定义一个字典，将模型名称映射到对应的模型类
        model_dict = {
            'TPGN': TPGN,
            'iTransformer': iTransformer,
            'ModernTCN': ModernTCN,
            'PatchTST': PatchTST,
            'MTSN': MTSN
            #'TransformerMOE': TransformerMOE,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # 打印模型参数数量
        print('###### generator parameters:', sum(param.numel() for param in model.parameters()))
        # 返回构建好的模型
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 创建一个均方误差损失函数（MSELoss）的实例
        criterion = nn.MSELoss()
        return criterion

    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'MTSN' in self.args.model:
                    if self.args.use_moe:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer() # 调函数实现
        criterion = self._select_criterion() # 调函数实现

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_time_all_list = [] # 记录每个epoch的运行时间

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            # torch.autograd.set_detect_anomaly(True)
            epoch_time_start = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() # decoder没法知道未来序列的真实值，因此需要基于前label_len个真实值来预测未来序列，未来pre_len用0代替
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) # 当前的参属下label_len是0，所以dec_inp就是batch_y

                moe_loss = 0.0
                # encoder - decoder
                if 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'MTSN' in self.args.model:
                    if self.args.use_moe:
                        outputs, moe_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y) + moe_loss*self.args.moe_loss_weight
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    # with torch.autograd.detect_anomaly():
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    model_optim.step()
            
            epoch_time_end = time.time()
            epoch_time_all = epoch_time_end - epoch_time_start

            epoch_time_all_list.append(epoch_time_all)

            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time_all))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        epoch_time_all_avg = np.mean(epoch_time_all_list)

        return self.model, epoch_time_all_avg

    def test(self, setting, train_time, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' # 创建结果保存目录
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            test_time_start = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() 
                # dim0、dim2和batch_y一致，dim1等于后self.args.pred_len长度的全0矩阵，并将其转换为浮点数。
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) 
                # dim0、dim2和batch_y一致，dim1等于batch_y的前self.args.label_len列和全0矩阵的拼接，并将其转换为浮点数。
                # dec_inp将历史数据与未来的输入拼接在一起，作为解码器的输入。
                # encoder - decoder
                if 'TST' in self.args.model:
                    # 如果模型类型中包含'Linear'或'TST'，则直接调用模型的前向传播函数
                    outputs = self.model(batch_x)
                elif 'MTSN' in self.args.model:
                    if self.args.use_moe:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        # 如果输出注意力权重，则调用模型的前向传播函数，并获取输出和注意力权重
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # 否则，只调用模型的前向传播函数，并获取输出
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0 # f_dim=1时一般模型处理的时多变量数据，=0时一般模型处理的是单变量数据

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) # 选择后pred_len个时间步的输出，并且根据f_dim选择对应的特征维度
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy() # outputs是模型预测的输出，batch_y是真实的目标值

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))
            test_time_end = time.time()
        test_time = test_time_end - test_time_start
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mae:{}, mse:{}, train_time:{}, test_time:{}'.format(mae, mse, train_time, test_time))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, train_time:{}, test_time:{}'.format(mae, mse, train_time, test_time))
        f.write('\n')
        f.write('\n')
        f.close()
        
        if self.args.save_npy:
            np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        # time.sleep(100)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'MTSN' in self.args.model:
                    if self.args.use_moe:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        
        return
