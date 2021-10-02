from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .. import xutilities
from .math_ops import cosine_distance
from .model import MCAE, loss_fn as mcae_loss


class MCAE_MP(xutilities.nn.Module):
    """
    Multi-point extention MCAE-MP to MCAE. 
    In practice, if we get MCAE by setting the perspectives to `[[0,1]]`.

    ----------------------------------------------------------------------------
    ### Args

    ### Attributes

    ### Inputs

    ### Outputs

    """
    def __init__(
        self, input_type='coors', in_channels=3, num_joint=25, num_body=2, maxlen=300,
        graph_name='nturgbd', dropout=0, num_classes=60,
        num_sni_temp=16, len_sni_temp=8, num_sni_param=5,
        gcn_out_channels=64,
        num_seg_temp=16, num_seg_param=1+6, contrastive=False, supervised=False,
        constrain_scale=1, constrain_trans=1.5, out_mode='linear',
        mcae_enable_segment=True,
        mcae_segenc_lstm_hidden=128, mcae_segenc_fc_hidden=32,
        run_mode='2D',
        perspectives=[[0,1], [1,2], [0,2]]):
        super(MCAE_MP, self).__init__()

        self.input_type = input_type
        self.in_channels = in_channels
        self.num_joint = num_joint
        self.num_body = num_body
        self.maxlen = maxlen

        self.graph_name = graph_name
        self.dropout = dropout
        self.num_classes = num_classes

        self.num_sni_temp = num_sni_temp
        self.len_sni_temp = len_sni_temp
        self.num_sni_param = num_sni_param

        self.gcn_out_channels = gcn_out_channels
        
        self.num_seg_temp = num_seg_temp
        self.num_seg_param = num_seg_param

        self.contrastive = contrastive
        self.supervised = supervised

        self.constrain_scale = constrain_scale
        self.constrain_trans = constrain_trans

        self.out_mode = out_mode
        self.perspectives = perspectives
        self.run_mode = run_mode
    
        num_presps = len(self.perspectives)

        if run_mode == '2D':
            self.mcae = nn.ModuleList(
                [
                    MCAE(input_type=input_type, max_len=maxlen, coor_dim=2,
                        num_sni_temp=num_sni_temp, len_sni_temp=len_sni_temp, num_sni_param=5,
                        num_seg_temp=num_seg_temp, num_seg_param=7, num_classes=num_classes,
                        contrastive=contrastive, supervised=supervised,
                        constrain_scale=constrain_scale, constrain_trans=constrain_trans,
                        segenc_lstm_hidden=mcae_segenc_lstm_hidden,
                        segenc_fc_hidden=mcae_segenc_fc_hidden,
                        enable_segment=mcae_enable_segment)
                ])
        
            if out_mode == 'linear':
                if mcae_enable_segment:
                    self.sk_cls = nn.Linear(num_joint*num_presps*num_seg_temp, num_classes)
                else:
                    self.sk_cls = nn.Linear(num_joint*num_presps*num_sni_temp, num_classes)
            else:
                raise NotImplementedError('Unsupported output mode "{out_mode}".')
        elif run_mode == '3D':
            self.mcae = MCAE(
                input_type=input_type, max_len=maxlen, coor_dim=3,
                num_sni_temp=num_sni_temp, len_sni_temp=len_sni_temp, num_sni_param=8,
                num_seg_temp=num_seg_temp, num_seg_param=13, num_classes=num_classes,
                contrastive=contrastive, supervised=supervised,
                constrain_scale=constrain_scale, constrain_trans=constrain_trans,
                segenc_lstm_hidden=mcae_segenc_lstm_hidden,
                segenc_fc_hidden=mcae_segenc_fc_hidden,
                enable_segment=mcae_enable_segment
            )
            if out_mode == 'linear':
                if mcae_enable_segment:
                    self.sk_cls = nn.Linear(num_joint*num_seg_temp, num_classes)
                else:
                    self.sk_cls = nn.Linear(num_joint*num_sni_temp, num_classes)
            else:
                raise NotImplementedError('Unsupported output mode "{out_mode}".')

    def forward(self, x):
        B, d, T, nJ, nB = x.shape
        results = dict.fromkeys(['sk_lgts'], None)
        x = x.permute(0, 4, 3, 2, 1).reshape(B*nB*nJ, T, d)  # (B, nB, nJ, T, d)


        if self.run_mode == '2D':
            xs = [x[..., p] for p in self.perspectives]
            mcae_results = []
            for i in range(len(self.perspectives)):
                mcae_results.append(self.mcae[0](xs[i]))
            results['mcae'] = mcae_results

            seg_pres = torch.cat([i['seg_pres'] for i in results['mcae']], -1)\
                .reshape(B, nB, nJ, -1)  # (B, nB, nJ, 3*nSegT)
            
            results['sk_pres'] = seg_pres.mean(1).reshape(B, -1)
        
        elif self.run_mode == '3D':
            mcae_result = self.mcae(x)
            results['mcae_3d'] = mcae_result

            seg_pres = mcae_result['seg_pres'].reshape(B, nB, nJ, -1)  # (B, nB, nJ, nSegT)
            results['sk_pres'] = seg_pres.mean(1).reshape(B, -1)
        
        if self.supervised:
            results['sk_lgts'] = self.sk_cls(results['sk_pres'])
        else:
            results['sk_lgts'] = self.sk_cls(results['sk_pres'].clone().detach())
        
        return results


def loss_fn(x, results, is_valtest=False, **kwargs):
    default_lsw = dict.fromkeys(
        [
            'sni', 'seg', 'cont', 'con', 'snireg',
            'segreg', 'reg', 'skcon', 'skcls'
        ], 1.0)
    loss_weights = kwargs.get('loss_weights', default_lsw)
    losses = {}
    
    mcae_losses = []
    sk_pres = results['sk_pres']
    sk_lgts = results['sk_lgts']
    sk_y = kwargs.get('y', None)

    if 'mcae' in results.keys():
        mcae_results = results['mcae']
        for r in mcae_results:
            mcae_losses.append(
                mcae_loss(r['x'], r, loss_weights=loss_weights, is_valtest=is_valtest))
        
        for key in loss_weights.keys():
            losses[key] = 0
            if key in mcae_losses[0][0].keys():
                for i in range(len(mcae_results)):
                    losses[key] += mcae_losses[i][0][key]
            else:
                losses.pop(key)

    elif 'mcae_3d' in results.keys():
        r = results['mcae_3d']
        mcae_loss_ = mcae_loss(r['x'], r, loss_weights=loss_weights, is_valtest=is_valtest)[0]
        for key in loss_weights.keys():
            losses[key] = 0
            if key in mcae_loss_.keys():
                losses[key] += mcae_loss_[key]
            else:
                losses.pop(key)
    
    if loss_weights.get('skcon', 0) > 0 and not is_valtest:
        B = sk_pres.shape[0]
        _L = int(B/2)
        tau = 0.1
        trj_pres = sk_pres.reshape(B, -1)
        ori, aug = trj_pres.split(_L, 0)
        dist_grid = 1 - cosine_distance(ori, aug)
        
        dist_grid_exp = torch.exp(dist_grid/tau)
        losses['skcon'] = -torch.log(
            torch.diag(dist_grid_exp) / dist_grid_exp.sum(1)).mean()

    if loss_weights.get('skcls', 0) > 0:
        losses['skcls'] = F.nll_loss(F.log_softmax(sk_lgts, -1), sk_y)
    
    return losses, default_lsw

if __name__ == '__main__':
    device = 'cuda'
    net = MCAE_MP(
        maxlen=64, out_mode='linear', len_sni_temp=8, run_mode='2D').to(device)
    x = torch.randn(4, 3, 64, 25, 2).to(device)
    y = torch.randint(0, 60, (4,)).to(device)
    y_ = net(x)
    losses = loss_fn(None, y_, y=y)
    import ipdb; ipdb.set_trace()
