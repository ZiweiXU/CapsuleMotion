import logging
import os
import os.path as osp
from typing import Dict
import random

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch import nn
from torch.utils.data import DataLoader

from lib.mcae.spatial import full_pose
from lib.mcae.eval import knn_classify
from lib.xutilities.engine import TVTMachinePrototype, tqdm_commons
from lib.xutilities.utils import (count_params, init, myself,
                                  worker_init_fn_seed, apply_loss_weights,
                                  unlock_gpu)


# ------------------------------- #
# Additional CLI Parameters
# ------------------------------- #
def params(p):
    p.add_argument('--root', default='data')
    p.add_argument('--config_path', default='config/mmnist_cddc.txt')
    p.add_argument('--ds_name', default='NTURGBD')
    p.add_argument('--model_params', type=str, default='{}')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lr_decay', '--lr-decay', type=float, default=0.1)
    p.add_argument('--opt', type=str, default='RMSprop')
    p.add_argument('--opt_params', type=str, default='{}')
    p.add_argument('--ds_params', type=str, default='{}')
    p.add_argument('--lrsch_params', type=str, default='{}')
    p.add_argument('--gclipv', type=float, default=10)
    p.add_argument('--batch-size', '--batch_size', type=int, default=128)
    p.add_argument('--forward_batch_size', type=int, default=32)
    p.add_argument('--test-batch-size', '--test_batch_size', type=int,
        default=128)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--loss_weights', type=str, default='{}')
    p.add_argument('--acc_name', type=str, default='acc')

    p.add_argument('--num_workers', type=int, default=min(os.cpu_count(), 4))

    p.add_argument('--plot_cm', action='store_true', default=False)
    p.add_argument('--amp', type=int, default=0)
    p.add_argument('--debug_val', '--debug-val',
                   action='store_true', default=False)
    p.add_argument('--debug_runs', type=int, default=5)
    p.add_argument('--model', type=str, default='skmcae')
    return p


def aug_data(seq, label):
    """Augment data for contrastive learning.
    The augmented data (and labels) will be appended to the end of the batch.
    """

    def jitter(x):
        B, _, _, nJ, _ = x.shape
        return x + torch.randn_like(x) * (torch.zeros(B, 1, 1, nJ, 1).uniform_(-1, 1) > 0.1).type(x.dtype)
    
    def rotate(x):
        B, d, T, nJ, nB = x.shape
        if d == 3:
            persp = random.choice([[0,1], [1,2], [0,2]])
        else:
            persp = [0,1]
        tx, ty = torch.zeros(B, 1, 1, 1, 2).repeat(1, T, nJ, nB, 1).split(1, -1)
        theta = torch.zeros((B, 1))\
            .uniform_(-30/360*2*3.14159, 30/360*2*3.14159)\
                .reshape(B, 1, 1, 1, -1).repeat(1, T, nJ, nB, 1)
        st, ct = torch.sin(theta), torch.cos(theta)

        trans = torch.cat([ct, -st, tx, st, ct, ty], -1).reshape(B*T*nJ*nB, 6)
        x2 = x.permute(0,2,3,4,1).reshape(B*T*nJ*nB, d)
        x2[:, persp] = (full_pose(trans) @ torch.cat([x2[:, persp]\
            .reshape(B*T*nJ*nB, -1), torch.ones(B*T*nJ*nB, 1)], -1).unsqueeze(-1))\
                [..., :-1, :].reshape(B*T*nJ*nB, 2)
        x2 = x2.reshape(B, T, nJ, nB, d).permute(0,4,1,2,3)
        return x2
    
    def joint_mask(x):
        B, _, _, nJ, _ = x.shape
        return x * (torch.zeros(B, 1, 1, nJ, 1).uniform_(0, 1) < 0.8).type(x.dtype)
    
    def smooth(x):
        B, d, T, nJ, nB = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(B*nJ*nB, d, T)
        x_ = torch.cat([x[..., 0:1], x, x[..., -1:]], -1).reshape(B*nJ*nB*d, 1, T+2)
        kernel = torch.Tensor([1/3, 1/3, 1/3])[None, None, :]
        x_ = F.conv1d(x_, kernel).reshape(B, nJ, nB, d, T)
        return x_.permute(0, 3, 4, 1, 2)
        
    # B, d, T, nJ, nB = x.shape
    B, nJ = seq.shape[0], seq.shape[3]

    indices = list(range(B))
    if nJ > 1:
        aug_funcs = random.choices([jitter, rotate, joint_mask, smooth], k=4)
    else:
        aug_funcs = random.choices([rotate, smooth], k=2)
    N = len(aug_funcs)
    indice_sets = [indices[i::N] for i in range(N)]
    label_shuffled = torch.cat([label[i].clone() for i in indice_sets], 0)
    ori_sets = [seq[i, ...].clone() for i in indice_sets]
    aug_sets = [aug(i.clone()) for aug, i in zip(aug_funcs, ori_sets)]

    aug_seq = torch.cat(aug_sets, 0)

    return aug_seq, label_shuffled


def aug_data_frame(img, label):
    """Augment data for contrastive learning.
    The augmented data (and labels) will be appended to the end of the batch.
    """
    raise NotImplementedError


class TVTMachine(TVTMachinePrototype):
    def __init__(self, args):
        super(TVTMachine, self).__init__(args)
        if not self.args.test_only:
            self.copy_source(main_file=osp.basename(__file__))
        if bool(int(self.model_params['contrastive'])):
            assert self.args.forward_batch_size == self.args.batch_size * 2
    
    def create_dataloader(self) -> Dict[str, DataLoader]:
        
        if self.model_params.get('input_type', 'coors') == 'frames':
            raise NotImplementedError
        else:
            if 'nturgbd' in self.args.ds_name:
                from lib.dataset.nturgbd import NTURGBD
                ds_class = NTURGBD
                ds_root = osp.join(self.args.root, self.args.ds_name, 'processed')
            elif self.args.ds_name == 'nwucla':
                from lib.dataset.nwucla import NWUCLA
                ds_class = NWUCLA
                ds_root = osp.join(self.args.root, self.args.ds_name)
            elif 't20' in self.args.ds_name:
                from lib.dataset.t20 import T20_Traj
                ds_class = T20_Traj
                ds_root = osp.join(self.args.root, self.args.ds_name)
            train_dataset = ds_class(
                root=ds_root, split='train', debug=self.debug_mode, **self.dataset_params)
            test_dataset = ds_class(
                root=ds_root, split='test', debug=self.debug_mode, **self.dataset_params)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn_seed(self.args), drop_last=True)

        test_dataloader  = DataLoader(
            test_dataset, batch_size=self.args.test_batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn_seed(self.args), drop_last=False)

        return {
            'train': train_dataloader,
            'test': test_dataloader
        }
    
    def create_model(self):
        if 'mcae' in self.args.model:
            from lib.mcae.mp import MCAE_MP as Model
        elif self.args.model == 'baseline':
            from lib.baseline.sk_baseline_model import SkBaseline as Model
        model = Model(**self.model_params).to(self.device)
        logging.getLogger(myself()).info(
            f'Number of params: {count_params(model)}.')
        return model
    
    def create_loss_fn(self):
        if 'mcae' in self.args.model:
            from lib.mcae.mp import loss_fn
        elif self.args.model == 'baseline':
            from lib.baseline.sk_baseline_model import loss_fn
        return loss_fn
    
    def train_epoch(self):
        return super().train_epoch()
    
    def train_batch(self, data):
        X, y = data

        if bool(int(self.model_params['contrastive'])):
            X1, y1 = aug_data(X, y)
            X2, y2 = aug_data(X, y)
            X, y = torch.cat([X1, X2], 0), torch.cat([y1, y2], 0)
        
        X = X.to(self.device)
        y, cy = y.to(self.device).long(), None
        
        self.optimizer.zero_grad()

        # ####### Gradient Accumulation for Smaller Batches ####### #
        splits = self._gradient_accumulate_splits(len(X))

        loss_values = []
        with torch.autograd.set_detect_anomaly(self.debug_mode):
            for j in range(splits):
                left = j * self.args.forward_batch_size
                right = left + self.args.forward_batch_size
                Xb, yb = X[left:right], y[left:right]
                cyb = cy[left:right] if cy is not None else None

                with autocast(self.amp):
                    outputs = self.model(Xb)
                    losses, _ = self.loss_fn(
                        Xb, outputs, loss_weights=self.loss_weights, y=yb, cy=cyb)

                if self.global_step % 50 == 0:
                    for k in losses.keys():
                        self.summary_writer.add_scalar(
                            f'loss/{k}', losses[k].item(),
                            global_step=self.global_step)

                loss, losses = apply_loss_weights(losses, self.loss_weights)
                loss_values.append(loss.item())

                if loss != loss:
                    raise ValueError('Loss goes to nan.')
            
                if self.amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
            if self.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        unlock_gpu()
        del outputs, loss, losses

        return loss_values
    
    def valtest(self, phase, data_loader, topk=1):
        self.model = self.model.to(self.device)
        self.model.eval()
        t = tqdm(data_loader, disable=(not self.args.pbar), **tqdm_commons)

        traj_feats = []

        preds_t = []
        labels_t = []
        
        loss_values = []

        tested_total = 0
        with torch.no_grad():
            for batch_id, data in enumerate(t):
                X, y = data

                tested_total += X.shape[0]
                X = X.to(self.device)
                outputs = self.model(X)
            
                y, cy = y.to(self.device).long(), None
                
                traj_feats.append(outputs["sk_pres"])
                traj_lgts = outputs['sk_lgts']
                preds_t.append(traj_lgts.argmax(-1))
                labels_t.append(y.long())

                losses, _ = self.loss_fn(
                    X, outputs, loss_weights=self.loss_weights, y=y, cy=cy, is_valtest=True)
                loss, _ = apply_loss_weights(losses, self.loss_weights)
                loss_values.append(loss.item())

                if self.debug_mode and batch_id == self.args.debug_runs:
                    break

        loss = np.mean(loss_values)

        results = {}
        
        labels_t = torch.cat(labels_t, 0).cpu()
        traj_feats = torch.cat(traj_feats, 0).cpu()

        results['tacc_km'] = 0

        preds_t = torch.cat(preds_t, 0).reshape(labels_t.shape).cpu()
        results['tacc'] = float((preds_t == labels_t).sum()) / labels_t.shape[0]
        results['tacc_km'] = results['tacc']

        # 1nn classifier test is for final test time only
        if self.args.test_only or ('ucla' in self.args.ds_name):
            train_feats = []
            train_labels = []

            tmp_dl = DataLoader(
                self.dataloaders['train'].dataset, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.num_workers, pin_memory=True,
                worker_init_fn=worker_init_fn_seed(self.args), drop_last=False)

            tt = tqdm(tmp_dl, disable=(not self.args.pbar), **tqdm_commons)
            with torch.no_grad():
                for batch_id, data in enumerate(tt):
                    X, y = data
                    X = X.to(self.device)
                    y, cy = y.to(self.device).long(), None
                    outputs = self.model(X)
                    train_feats.append(outputs["sk_pres"])
                    train_labels.append(y.long())
            
            train_feats = torch.cat(train_feats, 0).cpu()
            train_labels = torch.cat(train_labels, 0).cpu()
            results['tacc_1nn'] = knn_classify(train_feats, traj_feats, train_labels, labels_t)
        
        if self.model.supervised:
            results['acc'] = results['tacc']
        else:
            results['acc'] = results['tacc_km']
        
        self.summary_writer.add_scalar(
            f'acc/{phase}/tacc_km', results['tacc_km'],
            global_step=self.global_step)
        self.summary_writer.add_scalar(
            f'acc/{phase}/tacc', results['tacc'],
            global_step=self.global_step)

        self.summary_writer.flush()

        test_info = {'phase': phase, 'test_loss': loss}
        test_info.update(results)

        torch.cuda.empty_cache()
            
        return test_info
    
    def run(self):
        return super().run()


if __name__ == '__main__':
    args = init(user_param=params, pytorch_deterministic=True)
    TVTMachine(args).run()
