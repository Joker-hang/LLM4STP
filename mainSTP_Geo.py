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
mses = []
maes = []

if __name__ == '__main__':

    for ii in range(args.itr):

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
            batch_size=args.batch_size // 4,
            shuffle=True,
            collate_fn=seq_collate,
            pin_memory=True)

        is_train = True
        if is_train:
            """ dataloader """

            train_set = TrajectoryDataset(
                'dataset/' + args.data_path + '/train/',
                'temp_save_dir/' + args.data_path + '/train/',
                load_dataset=args.load_dataset,
                obs_len=args.seq_len,
                pred_len=args.pred_len,
                skip=1)

            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=seq_collate,
                pin_memory=True)

            train_steps = len(train_loader)

            valid_set = TrajectoryDataset(
                'dataset/' + args.data_path + '/val/',
                'temp_save_dir/' + args.data_path + '/val/',
                load_dataset=args.load_dataset,
                obs_len=args.seq_len,
                pred_len=args.pred_len,
                skip=1)
            vali_loader = DataLoader(
                valid_set,
                batch_size=args.batch_size // 4,
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
        if is_train:
            for epoch in range(args.train_epochs):

                iter_count = 0
                train_loss = []
                epoch_time = time.time()
                for i, batch in tqdm(enumerate(train_loader)):

                    iter_count += 1
                    batch_x = batch['past_traj'].permute(0, 2, 1)
                    batch_y = batch['future_traj'].permute(0, 2, 1)

                    time1 = time.time()

                    geohash_codeing = geohash_encoding(batch_x).to(device=device).float()
                    geohash_codeing = rearrange(geohash_codeing, 'b l p c -> b l (p c)')

                    # time2 = time.time()
                    # print(time2 - time1)

                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    # 生成高斯分布图
                    total_gaussian = test_set.generate_gaussian_map(
                        nodes_current=batch_x,
                        grid_size=args.grid_size,
                        sigma_x=args.sigmaX,
                        sigma_y=args.sigmaY,
                    )  # [num_vehicles, 224, 224]

                    # time3 = time.time()
                    # print(time3 - time2)

                    model_optim.zero_grad()

                    total_gaussian = total_gaussian.to(device)

                    outputs = model(batch_x, total_gaussian, geohash_codeing)

                    outputs = outputs[:, -args.pred_len:, :]
                    batch_y = batch_y[:, -args.pred_len:, :].to(device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f} | speed: {3:.4f}s/iter; left time: {4:.4f}s".format(
                                i + 1, epoch + 1, loss.item(), speed, left_time))

                        # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    loss.backward()
                    model_optim.step()

                # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                train_loss = np.average(train_loss)
                writer.add_scalar('loss/train', train_loss, epoch)

                vali_loss = vali(model, valid_set, vali_loader, criterion, args, device, ii)
                # test_loss = vali(model, test_set, test_loader, criterion, args, device, ii)
                # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
                #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                writer.add_scalar('loss/val', vali_loss, epoch)

                if args.cos:
                    scheduler.step()
                    # print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(model_optim, epoch + 1, args)
                early_stopping(vali_loss, model, path)

                # ade, fde, mse, mae = test(model, test_set, test_loader, args, device, ii)

                # print("ade_mean = {:.4f}".format(np.mean(ade)))
                # print("fde_mean = {:.4f}".format(np.mean(fde)))
                # print("mse_mean = {:.4f}".format(np.mean(mse)))
                # print("mae_mean = {:.4f}".format(np.mean(mae)))

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        ade, fde, mse, mae = test(model, test_set, test_loader, args, device, ii)
        ades.append(ade)
        fdes.append(fde)
        mses.append(mse)
        maes.append(mae)

    ades = np.array(ades)
    fdes = np.array(fdes)
    print("ade_mean = {:.4f}".format(np.mean(ades)))
    print("fde_mean = {:.4f}".format(np.mean(fdes)))
    print("mse_mean = {:.4f}".format(np.mean(mses)))
    print("mae_mean = {:.4f}".format(np.mean(maes)))
