import argparse
import os
import sys
sys.path.append('/home/wms/South/TPGN-main/models')
sys.path.append('/home/wms/South/TPGN-main/layers')
sys.path.append('/home/wms/South/TPGN-main/utils')
from utils.Mypydebug import show_shape

show_shape()

import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.str2bool import str2bool

current_directory = os.getcwd()
print(current_directory)

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description=
    'Models for Long-range Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--task_id', type=str, default='test', help='task id')
parser.add_argument('--model', type=str, default='MTSN',
    help='model name, options: [TPGN, iTransformer, TimeMixer, FITS, ModernTCN, PDF, \
    WITRAN, CrossGNN, FourierGNN, Basisformer, \
    MICN, TimesNet, PatchTST, DLinear, NLinear, Linear, \
    FiLM, FEDformer, Pyraformer, Autoformer, Informer, Transformer, MTSN]')

# MTSN config
parser.add_argument('--num_blocks', type=int, default=2, help='number of blocks')
parser.add_argument('--top_k', type=int, default=2, help='top k')
parser.add_argument('--num_experts', type=int, default=4, help='number of experts')
parser.add_argument('--moe_loss_weight', type=float, default=0.4, help='moe loss weight')


# parser.add_argument('--patch_stride', type=int, default=4, help='patch stride')

# data loader
parser.add_argument('--data', type=str, default='electricity', help='dataset type')
parser.add_argument('--root_path', type=str, default='/home/wms/South/TPGN/Dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--features', type=str, default='MS', # 预测模式，S表示单变量预测单变量，M表示多变量预测多变量，MS表示多变量预测单变量
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, ' 
                            'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                            'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length, no use for TPGN')
parser.add_argument('--pred_len', type=int, default=168, help='prediction sequence length')

# model define 注意如果要进行多变量预测之类的需要在这里修改输入输出
parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
parser.add_argument('--enc_in', type=int, default=321, help='encoder input size') # 
parser.add_argument('--dec_in', type=int, default=321, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads, no use for TPGMNN')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers, no use for TPGMNN')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers, , no use for TPGMNN')
parser.add_argument('--d_ff', type=int, default=0, help='dimension of fcn, no use for TPGMNN')
parser.add_argument('--moving_avg', default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=25, help='train epochs')
parser.add_argument('--batch_size', type=int, default=20, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type4', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU  
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

# For PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0') # 是否使用双分支建模
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# For iTransformer
parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')

# For ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=8, help='ffn_ratio')

parser.add_argument('--patch_size', type=int, default=8, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=4, help='the patch stride')

parser.add_argument('--large_size', nargs='+',type=int, default=51, help='big kernel size')
parser.add_argument('--small_size', nargs='+',type=int, default=5, help='small kernel size for structral reparam')

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')

# For TPGN
parser.add_argument('--TPGN_period', type=int, default=24, help='TPGN_period')
parser.add_argument('--norm', type=int, default=0, help='norm')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.model == 'MICN':
    model_flag = args.model + '-' + args.mode
    decomp_kernel = []  # kernel of decomposition operation 
    isometric_kernel = []  # kernel of isometric convolution
    for ii in args.conv_kernel:
        if ii%2 == 0:   # the kernel of decomposition operation must be odd
            decomp_kernel.append(ii+1)
            isometric_kernel.append((args.seq_len + args.pred_len+ii) // ii) 
        else:
            decomp_kernel.append(ii)
            isometric_kernel.append((args.seq_len + args.pred_len+ii-1) // ii) 
    args.isometric_kernel = isometric_kernel  # kernel of isometric convolution
    args.decomp_kernel = decomp_kernel   # kernel of decomposition operation 
    print("isometric_kernel", isometric_kernel)
    print("decomp_kernel", decomp_kernel)
else:
    model_flag = args.model
print(model_flag)

eval_loss = []
test_loss = []

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_id,
            model_flag,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)
        
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        _, epoch_time_all_avg = exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, epoch_time_all_avg)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        
        torch.cuda.empty_cache()
        
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    model_flag,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)
    
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    
    torch.cuda.empty_cache()