from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from plugin.Plugin.model import Plugin
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
    series_decomp_multi,
)
from utils.buffer import Buffer
from utils.detector import STEPD
import math
import numpy as np
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
import pdb
import numpy as np
from einops import rearrange
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import defaultdict
import os
import time

import warnings
warnings.filterwarnings('ignore')


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]


class AdaptiveDLinear(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.backbone = net(configs, device)

    def forward(self, x, x_mark_enc=None, x_mark_dec=None):
        return self.backbone(x, x_mark_enc, x_mark_dec)

    def toggle_adaptation_mode(self, enable=True):
        if not enable:
            for param in self.parameters():
                param.requires_grad = True
            return

        for param in self.parameters():
            param.requires_grad = False

        if getattr(self.backbone, 'plugin', None) is not None:
            for param in self.backbone.plugin.MLP.parameters():
                param.requires_grad = True

        if self.backbone.individual:
            for layer in self.backbone.Linear_Seasonal:
                for param in layer.parameters():
                    param.requires_grad = True
            for layer in self.backbone.Linear_Trend:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            for param in self.backbone.Linear_Seasonal.parameters():
                param.requires_grad = True
            for param in self.backbone.Linear_Trend.parameters():
                param.requires_grad = True

class net(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs, device):
        super(net, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.flag = getattr(configs, 'flag', None)
        self.plugin = None
        if self.flag == 'Plugin':
            channel = getattr(configs, 'c_out', configs.enc_in)
            self.plugin = Plugin(configs, channel)

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.device = device
        self.to(device)

    def forward(self, x, x_mark_enc=None, x_mark_dec=None):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x_out = seasonal_output + trend_output
        pred = x_out.permute(0, 2, 1)  # to [Batch, Output length, Channel]

        if self.plugin is not None and x_mark_enc is not None and x_mark_dec is not None:
            pred = self.plugin(x.clone(), x_mark_enc.clone(), pred, x_mark_dec[:, -self.pred_len :, :].clone())

        return pred

class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        if getattr(args, 'flag', None) != 'Plugin':
            args.flag = 'Plugin'
        self.model = AdaptiveDLinear(args, device=self.device)
        self.sleep_interval = max(1, getattr(args, 'sleep_interval', 4))
        detector_window = max(4, getattr(args, 'residual_window', self.sleep_interval))
        buffer_capacity = max(self.sleep_interval, getattr(args, 'glaff_buffer_size', 64))
        self.buffer = Buffer(buffer_capacity, self.device, mode='fifo')
        self.memory_min_batches = max(2, self.sleep_interval)
        self.adapt_steps = max(1, getattr(args, 'sleep_epochs', 1))
        self.adapt_lr = getattr(args, 'glaff_ft_lr', args.learning_rate)
        self.detector = STEPD(new_window_size=detector_window, alpha_w=args.alpha_w, alpha_d=args.alpha_d)
        self._last_online_info = {'drift': False, 'stat': float('nan')}

        if args.finetune:
            raise NotImplementedError('Fine-tuning is not supported for the GLAFF-D3A DLinear experiment')

    def _buffer_size(self):
        if not hasattr(self.buffer, 'examples'):
            return 0
        return min(self.buffer.num_seen_examples, self.buffer.examples.shape[0])

    def _buffer_clear(self):
        if hasattr(self.buffer, 'examples'):
            self.buffer.empty()

    def _buffer_add(self, batch_x, batch_y, batch_x_mark, outputs):
        combined_x = torch.cat([batch_x, batch_x_mark], dim=-1).detach().to(self.device)
        self.buffer.add_data(
            examples=combined_x,
            labels=batch_y.detach().to(self.device),
            logits=outputs.detach().to(self.device),
        )

    def _buffer_sample(self, n_batches):
        size = self._buffer_size()
        if size == 0:
            return None
        n_batches = max(1, min(size, n_batches))
        samples = self.buffer.get_data(n_batches)
        if len(samples) < 3:
            return None
        combined_x, batch_y, logits = samples
        enc_in = getattr(self.args, 'enc_in', None)
        if enc_in is None:
            return None
        batch_x = combined_x[..., :enc_in]
        batch_x_mark = combined_x[..., enc_in:]
        return batch_x, batch_x_mark, batch_y, logits

    def _current_theta(self):
        if not hasattr(self, 'detector') or self.detector is None:
            return float('nan')
        history = getattr(self.detector, 'data', None)
        window = getattr(self.detector, 'new_window_size', None)
        if history is None or window is None or len(history) < window:
            return float('nan')
        recent_window = history[-window:]
        overall = history
        std_dev = np.std(overall)
        if std_dev < 1e-8:
            return 0.0
        n = len(overall)
        theta = (np.mean(recent_window) - np.mean(overall)) / (std_dev / math.sqrt(n))
        return float(theta)


    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 3

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                self.opt.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()
        if self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False

        preds = []
        trues = []
        start = time.time()
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]
        if self.online != 'none':
            self.detector.reset()
            self._buffer_clear()
            self._last_online_info = {'drift': False, 'stat': float('nan')}

        progress = tqdm(test_loader, desc='Test', total=len(test_loader))
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(progress):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

            info = getattr(self, '_last_online_info', None)
            stat = info.get('stat', float('nan')) if info else float('nan')
            drift = info.get('drift', False) if info else False

            progress.set_postfix({
                'MAE': f'{mae:.4f}',
                'MSE': f'{mse:.4f}',
                'RMSE': f'{rmse:.4f}',
                'MAPE': f'{mape:.4f}',
                'MSPE': f'{mspe:.4f}',
                'theta': 'nan' if math.isnan(stat) else f'{stat:.2f}',
                'drift': drift,
            })

        progress.close()

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode == 'test':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return rearrange(outputs, 'b t d -> b (t d)'), rearrange(batch_y, 'b t d -> b (t d)')
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        true = rearrange(batch_y[:, -self.args.pred_len:, :], 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        online_enabled = self.online != 'none'
        if online_enabled and not hasattr(self, 'opt'):
            self.opt = self._select_optimizer()

        for _ in range(self.n_inner):
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

            outputs = rearrange(outputs, 'b t d -> b (t d)')
            loss = criterion(outputs, true)

            if online_enabled:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

        drift_flag = False
        theta_stat = float('nan')
        detection_enabled = self.sleep_interval > 1 or getattr(self.args, 'online_adjust', 0) > 0
        if detection_enabled:
            self.detector.add_data(loss.item(), batch_x)
            self._buffer_add(batch_x.detach(), batch_y.detach(), batch_x_mark.detach(), outputs.detach())
            status, _ = self.detector.run_test()
            theta_stat = self._current_theta()
            drift_ready = (
                status == 1
                and self._buffer_size() >= self.sleep_interval
            )

            if (drift_ready or self.detector.cnt >= 1000) and self.sleep_interval > 1:
                drift_flag = True
                self._lite_adaptation(criterion)
                self.detector.reset()
                self._buffer_clear()

        self._last_online_info = {
            'loss': loss.item(),
            'drift': drift_flag,
            'stat': theta_stat,
        }

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs.detach(), rearrange(batch_y, 'b t d -> b (t d)')

    def _lite_adaptation(self, criterion):
        samples = self._buffer_sample(self.sleep_interval)
        if samples is None:
            return
        batch_x, batch_x_mark, batch_y, _ = samples
        if batch_x.shape[0] == 0:
            return

        f_dim = -1 if self.args.features == 'MS' else 0
        target = rearrange(batch_y[:, -self.args.pred_len:, f_dim:], 'b t d -> b (t d)')

        self.model.toggle_adaptation_mode(enable=True)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.adapt_lr
        )

        self.model.train()
        adapt_losses = []
        for _ in range(self.adapt_steps):
            preds = self.model(batch_x, batch_x_mark, None)
            preds = rearrange(preds[:, -self.args.pred_len:, f_dim:], 'b t d -> b (t d)')
            loss = criterion(preds, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adapt_losses.append(loss.item())

        if adapt_losses:
            avg_loss = sum(adapt_losses) / len(adapt_losses)
            print(
                f"\n[GLAFF-D3A] Lite adaptation triggered: batches={batch_x.shape[0]}, "
                f"steps={len(adapt_losses)}, loss={avg_loss:.6f}"
            )
        else:
            print('\n[GLAFF-D3A] Lite adaptation skipped: no optimization steps executed.')

        self.model.toggle_adaptation_mode(enable=False)
        self.model.eval()
