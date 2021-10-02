import importlib
from typing import Dict

import numpy as np
import torch
from torch import nn, Tensor
from torch._C import Value
from torch.nn import functional as F

from .. import xutilities
from ..mcae.math_ops import cosine_distance
from ..mcae.model import MCAE, loss_fn as mcae_loss
from ..mcae.blocks import BatchLinear


class LSTMBaseline(nn.Module):
    """
    Description goes here.

    ----------------------------------------------------------------------------
    ### Args

    ### Attributes

    ### Inputs
        x: (B, T, d)
    ### Outputs

    """
    def __init__(self, out_feat_dim=64, **kwargs):
        super(LSTMBaseline, self).__init__()
        self.encoder = nn.LSTM(**kwargs)
        self.fc = nn.Linear(2*kwargs['hidden_size'], out_feat_dim)

    def forward(self, x):
        _, (out, _) = self.encoder(x)
        out = out.permute(1, 0, 2).reshape(x.shape[0], -1)
        out = F.leaky_relu(self.fc(out))
        return out


class Conv1DBaseline(nn.Module):
    """
    Description goes here.

    ----------------------------------------------------------------------------
    ### Args

    ### Attributes

    ### Inputs
        x: (B, T, d)
    ### Outputs

    """
    def __init__(self, maxlen=32, input_dim=2, out_channel=32):
        super(Conv1DBaseline, self).__init__()
        layers = []
        layers_num = int(np.log2(maxlen))-1
        base_channel = 48
        layers = [
            nn.Conv1d(input_dim, base_channel, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
        ]
        for i in range(layers_num-1):
            layers += [
                nn.Conv1d(base_channel, base_channel*2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(base_channel*2),
                nn.LeakyReLU(),
            ]
            base_channel *= 2
        layers += [nn.Conv1d(base_channel, out_channel, 4, 2, 1, bias=False)]
        self.conv_trunk = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.conv_trunk(x).squeeze(-1)
        return out


class SkBaseline(xutilities.nn.Module):
    """
    Description goes here.

    ----------------------------------------------------------------------------
    ### Args

    ### Attributes

    ### Inputs

    ### Outputs

    """
    def __init__(
        self, input_type='coors', in_channels=2, num_joint=1, num_body=1, maxlen=300,
        graph_name='nturgbd', dropout=0, num_classes=20, feat_dim=512,
        contrastive=False, supervised=False, 
        baseline='cnn'):
        super(SkBaseline, self).__init__()

        self.input_type = input_type
        self.in_channels = in_channels
        self.num_joint = num_joint
        self.num_body = num_body
        self.maxlen = maxlen

        self.graph_name = graph_name
        self.dropout = dropout
        self.num_classes = num_classes

        self.contrastive = contrastive
        self.supervised = supervised
        self.baseline = baseline

        if baseline == 'lstm':
            self.encoder = LSTMBaseline(out_feat_dim=feat_dim, 
                hidden_size=256, input_size=in_channels, 
                bidirectional=True, batch_first=True)
        elif baseline == 'cnn':
            self.encoder = Conv1DBaseline(
                maxlen=maxlen, input_dim=in_channels, out_channel=feat_dim)
        else:
            raise ValueError(f'Baseline \"{baseline}\" is not implemented.')
        self.sk_cls = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        B, d, T, nJ, nB = x.shape
        # requires: sk_pres (feats), sk_lgts (logits)
        results = dict.fromkeys(['sk_pres', 'sk_lgts'], None)
        x = x.permute(0, 4, 3, 2, 1).reshape(B*nB*nJ, T, d).squeeze()  # (B, T, d)

        results['sk_pres'] = self.encoder(x)
        
        if self.supervised:
            results['sk_lgts'] = self.sk_cls(results['sk_pres'])
        else:
            results['sk_lgts'] = self.sk_cls(results['sk_pres'].clone().detach())
        
        return results


def loss_fn(x, results, is_valtest=False, **kwargs):
    default_lsw = dict.fromkeys(['con', 'cls'], 1.0)
    loss_weights = kwargs.get('loss_weights', default_lsw)
    losses = {}
    
    sk_pres = results['sk_pres']
    sk_lgts = results['sk_lgts']
    sk_y = kwargs.get('y', None)
    
    if loss_weights.get('con', 0) > 0 and not is_valtest:
        B = sk_pres.shape[0]
        _L = int(B/2)
        tau = 0.1
        trj_pres = sk_pres.reshape(B, -1)
        ori, aug = trj_pres.split(_L, 0)
        dist_grid = 1 - cosine_distance(ori, aug)
        
        dist_grid_exp = torch.exp(dist_grid/tau)
        losses['con'] = -torch.log(
            torch.diag(dist_grid_exp) / dist_grid_exp.sum(1)).mean()

    if loss_weights.get('cls', 0) > 0:
        losses['cls'] = F.nll_loss(F.log_softmax(sk_lgts, -1), sk_y)
    
    return losses, default_lsw

if __name__ == '__main__':
    device = 'cuda'
    net = SkBaseline(maxlen=32, baseline='cnn').to(device)
    print(xutilities.utils.count_params(net))
    x = torch.randn(4, 2, 32, 1, 1).to(device)
    y = torch.randint(0, 20, (4,)).to(device)
    y_ = net(x)
    losses = loss_fn(None, y_, y=y)
    import ipdb; ipdb.set_trace()
