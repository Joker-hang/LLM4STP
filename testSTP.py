from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.GPT4TS import GPT4TS
from models.GPT4STP_Geohash import GPT4STP
from models.DLinear import DLinear
from einops import rearrange
import torch
import torch.nn as nn
from torch import optim

from data_provider.dataloader_nba import NBADataset, seq_collate
from data_provider.dataloader_Geo import TrajectoryDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from models.geohash import geohash_encoding
import warnings
import numpy as np

from utils.dict_as_object import DictAsObject
import random

warnings.filterwarnings('ignore')

fix_seed = 40
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
args = DictAsObject(config)
ades = []
fdes = []
fds = []

if __name__ == '__main__':

    for ii in range(args.itr):

        args.data_path = 'CFD'

        setting = '{}_pl{}_dm{}_nh{}_ps{}_gl{}_df{}_stride{}_itr{}_{}'.format(args.model_id, args.pred_len,
                                                                              args.d_model, args.n_heads,
                                                                              args.patch_size, args.gpt_layers,
                                                                              args.d_ff, args.stride, ii,
                                                                              args.data_path)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        writer = SummaryWriter(log_dir=path)

        test_set = TrajectoryDataset(
            'dataset/' + args.data_path + '/test/',
            'temp_save_dir/' + args.data_path + '/test/',
            load_dataset=args.load_dataset,
            obs_len=args.seq_len,
            pred_len=args.pred_len,
            skip=1)

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size // 2,
            shuffle=True,
            collate_fn=seq_collate,
            pin_memory=True)

        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        print("device: ", device)

        time_now = time.time()

        # model = GPT4TS(args, device)
        model = GPT4STP(args, device)
        # mse, mae = test(model, test_data, test_loader, args, device, ii)

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)

        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()

                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))


            criterion = SMAPE()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        ade, fde, fd = test(model, test_set, test_loader, args, device, ii)
        ades.append(ade)
        fdes.append(fde)
        fds.append(fd)

    print("ade = {:.4f}".format(np.mean(ades)))
    print("fde = {:.4f}".format(np.mean(fdes)))
    print("fd  = {:.4f}".format(np.mean(fds)))
