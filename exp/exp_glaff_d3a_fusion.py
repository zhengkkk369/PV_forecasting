import os
import time
from collections import deque
from typing import Dict, Type

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred
from einops import rearrange
from exp.exp_basic import Exp_Basic
from exp.exp_cross_former import Exp_TS2VecSupervised as ExpCrossFormer
from exp.exp_dlinear import Exp_TS2VecSupervised as ExpDLinear
from exp.exp_fedformer import Exp_TS2VecSupervised as ExpFedformer
from exp.exp_patch import Exp_TS2VecSupervised as ExpPatch
from models.ts2vec.fsnet import TSEncoder
from utils.buffer import Buffer
from utils.metrics import cumavg, metric
from utils.tools import EarlyStopping, adjust_learning_rate


class GlobalLocalFusion(nn.Module):
    """GLAFF-style dual branch with residual fusion."""

    def __init__(self, args, device: torch.device):
        super().__init__()
        self.device = device
        # global branch leverages timestamp features
        self.global_encoder = TSEncoder(
            input_dims=args.enc_in + args.time_features,
            output_dims=320,
            hidden_dims=64,
            depth=10,
        ).to(self.device)
        self.global_head = nn.Linear(320, args.c_out * args.pred_len).to(self.device)

        # local branch focuses on recent dynamics
        self.local_encoder = TSEncoder(
            input_dims=args.enc_in,
            output_dims=320,
            hidden_dims=64,
            depth=10,
        ).to(self.device)
        self.local_head = nn.Linear(320, args.c_out * args.pred_len).to(self.device)

        # fusion weight derived from temporal context
        gate_hidden = max(16, args.time_features)
        self.fusion_gate = nn.Sequential(
            nn.Linear(args.time_features, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def freeze_global(self):
        for p in self.global_encoder.parameters():
            p.requires_grad = False
        for p in self.global_head.parameters():
            p.requires_grad = False

    def local_parameters(self):
        for p in list(self.local_encoder.parameters()) + list(self.local_head.parameters()):
            if p.requires_grad:
                yield p

    def forward(self, x, x_mark):
        """Return fused prediction and branch outputs.

        Args:
            x: [B, L, enc_in]
            x_mark: [B, L, time_features]
        """
        global_input = torch.cat([x, x_mark], dim=-1)
        global_rep = self.global_encoder(global_input, mask='all_true')[:, -1]
        global_pred = self.global_head(global_rep)

        local_rep = self.local_encoder(x, mask='all_true')[:, -1]
        local_residual = self.local_head(local_rep)

        gate = self.fusion_gate(x_mark.mean(dim=1))
        fused = global_pred + (1 - gate) * local_residual
        return fused, global_pred, local_residual


def energy_distance(data_ref: torch.Tensor, data_new: torch.Tensor) -> torch.Tensor:
    ref_new = torch.cdist(data_new, data_ref, p=2).mean()
    new_new = torch.cdist(data_new, data_new, p=2).mean()
    ref_ref = torch.cdist(data_ref, data_ref, p=2).mean()
    return 2 * ref_new - new_new - ref_ref


class ExpGLAFF_TS2Vec(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = GlobalLocalFusion(args, device=self.device)
        self.buffer = Buffer(args.glaff_buffer_size, self.device, mode='fifo')
        self.residual_window = deque(maxlen=args.residual_window)
        self.base_sigma = args.residual_base_sigma

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
        Data = data_dict_.get(self.args.data, Dataset_Custom)
        timeenc = 2

        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args.test_bsz
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
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
            cols=args.cols,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        return nn.MSELoss()

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

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.opt.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                loss = (
                    criterion(pred[0], true)
                    + criterion(pred[1], true)
                    + criterion(pred[2], true)
                )
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.opt.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = 0.0

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali'
            )
            weight = 1.0 / len(pred)
            fused = sum(pred) * weight
            loss = criterion(fused.detach().cpu(), true.detach().cpu())
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

        preds, trues = [], []
        start = time.time()
        maes, mses, rmses, mapes, mspes = [], [], [], [], []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test'
            )
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(
                pred.detach().cpu().numpy(), true.detach().cpu().numpy()
            )
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)

        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode == 'test' and self.online != 'none':
            return self._online_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()
        fused, g_pred, l_pred = self.model(x, batch_x_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
        target = rearrange(batch_y, 'b t d -> b (t d)')
        return [fused, g_pred, l_pred], target

    def _online_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        criterion = self._select_criterion()
        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        fused, g_pred, l_pred = self.model(x, batch_x_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        target_seq = batch_y[:, -self.args.pred_len :, f_dim:]
        target = rearrange(target_seq, 'b t d -> b (t d)')

        residual = (target - fused).detach().flatten().mean().item()
        window_res = (target - fused).detach().flatten()
        self.residual_window.append(window_res)
        self.buffer.add_data(examples=torch.cat([x, batch_x_mark], dim=-1), labels=target_seq, logits=fused.detach())

        if self._detect_virtual_drift():
            print('Virtual drift warning triggered.')

        if self._detect_real_drift():
            print('Real drift detected, fine-tuning local branch.')
            self._fine_tune_local_branch()

        loss = criterion(fused, target) + criterion(g_pred, target) + criterion(l_pred, target)
        if self.online != 'none':
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return fused.detach(), target.detach()

    def _detect_virtual_drift(self) -> bool:
        if not hasattr(self.buffer, 'examples') or self.buffer.num_seen_examples < self.args.virtual_min_samples:
            return False
        examples = self.buffer.examples[: self.buffer.num_seen_examples]
        if examples.shape[0] < 4:
            return False
        mid = examples.shape[0] // 2
        ref = examples[:mid, :, : self.args.enc_in].reshape(mid, -1)
        new = examples[mid:, :, : self.args.enc_in].reshape(examples.shape[0] - mid, -1)
        dist = energy_distance(ref, new)
        return dist.item() > self.args.energy_threshold

    def _detect_real_drift(self) -> bool:
        if len(self.residual_window) == 0:
            return False
        res_tensor = torch.cat(list(self.residual_window))
        mu_res = res_tensor.mean().item()
        sigma_res = res_tensor.std(unbiased=False).item()
        drift_by_mu = abs(mu_res) > self.args.residual_mu_thresh * self.base_sigma
        drift_by_sigma = sigma_res > self.args.residual_sigma_thresh * self.base_sigma
        return drift_by_mu or drift_by_sigma

    def _fine_tune_local_branch(self):
        if not hasattr(self.buffer, 'examples') or self.buffer.is_empty():
            return
        self.model.freeze_global()
        examples, labels, _ = self.buffer.get_all_data()
        x = examples[:, :, : self.args.enc_in]
        x_mark = examples[:, :, self.args.enc_in : self.args.enc_in + self.args.time_features]
        targets = rearrange(labels[:, -self.args.pred_len :, :], 'b t d -> b (t d)')

        dataset = TensorDataset(x, x_mark, targets)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        opt = optim.AdamW(self.model.local_parameters(), lr=self.args.glaff_ft_lr)
        criterion = self._select_criterion()
        self.model.train()
        for _ in range(self.args.glaff_ft_epochs):
            for batch_x, batch_mark, batch_target in loader:
                opt.zero_grad()
                fused, g_pred, l_pred = self.model(batch_x, batch_mark)
                loss = criterion(l_pred, batch_target) + criterion(fused, batch_target)
                loss.backward()
                opt.step()
        self.model.eval()
        torch.cuda.empty_cache()


_BACKBONE_DISPATCH: Dict[str, Type[Exp_Basic]] = {
    'ts2vec': ExpGLAFF_TS2Vec,
    'dlinear': ExpDLinear,
    'patchtst': ExpPatch,
    'fedformer': ExpFedformer,
    'autoformer': ExpFedformer,  # share the same experiment runner
    'informer': ExpFedformer,
    'crossformer': ExpCrossFormer,
}


def _resolve_backbone(name: str) -> str:
    """Normalize backbone names for dispatching."""

    return str(name).lower()


class Exp_TS2VecSupervised:
    """Dispatch to the proper experiment class per backbone for GLAFF fusion."""

    def __init__(self, args):
        backbone = _resolve_backbone(getattr(args, 'glaff_backbone', 'ts2vec'))
        if backbone not in _BACKBONE_DISPATCH:
            raise ValueError(
                f"Unsupported glaff_backbone '{backbone}'. Available: {sorted(_BACKBONE_DISPATCH.keys())}"
            )
        self._exp = _BACKBONE_DISPATCH[backbone](args)

    def __getattr__(self, name):
        return getattr(self._exp, name)

    def __repr__(self):
        return f"Exp_TS2VecSupervised(dispatch={self._exp!r})"