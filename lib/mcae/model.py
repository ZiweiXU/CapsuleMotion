from lib.mcae.spatial import compact_pose
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict

from ..xutilities.nn import Module
from .math_ops import cosine_distance
from .endec import (
    ContentEncoder, ContentDecoder, ObjectLocator, RecurrentFeaturePooler,
    SnippetEncoder, SnippetDecoder, SegmentEncoder, SegmentDecoder)
from .spatial import image_to_object, object_to_image, constrain_pose


class MCAE(Module):
    """The base model definition for MCAE. 
    It takes a sequence of coordinates as input.
    
    MCAE has two different autoencoding mechanisms:
    1. The snippet autoencoder.
    2. The segment autoencoder.

    For 2D trajectories, (num_sni_param, num_seg_param) = (5, 1+6).
    num_sni_param corresponds to (pres, scale, tx, ty, theta).

    For 3D trajectories, (num_sni_param, num_seg_param) = (8, 1+12).
    num_sni_param corresponds to (pres, scale, tx, ty, tz, alpha, beta, gamma).

    ----------------------------------------------------------------------------
    ### Args

    ### Attributes

    ### Inputs

    ### Outputs

    """
    def __init__(self,
        coor_dim=2, content_dim=32, max_len=32, frame_channel=1, frame_dim=128,
        input_type='coors', num_classes=20,
        num_sni_temp=32, len_sni_temp=8, num_sni_param=5,
        num_seg_temp=16, num_seg_param=1+6, contrastive=False, supervised=False,
        constrain_scale=None, constrain_trans=None, enable_segment=True,
        segenc_lstm_hidden=128, segenc_fc_hidden=32):
        super(MCAE, self).__init__()
        self.coor_dim = coor_dim
        self.content_dim = content_dim
        self.max_len = max_len
        self.frame_channel = frame_channel
        self.frame_dim = frame_dim
        self.input_type = input_type
        self.num_classes = num_classes
        self.enable_segment = enable_segment

        assert (coor_dim, num_sni_param, num_seg_param) in (
            (2, 5, 7), (3, 8, 13)
        )

        if not self.enable_segment:
            assert len_sni_temp == max_len

        if not constrain_scale:
            self.constrain_scale = 1 if input_type == 'coors' else 4
        else:
            self.constrain_scale = constrain_scale
        if not constrain_trans:
            self.constrain_trans = 1.5 if input_type == 'coors' else 1.5
        else:
            self.constrain_trans = constrain_trans

        self.num_sni_temp = num_sni_temp
        self.len_sni_temp = len_sni_temp
        self.num_sni_param = num_sni_param
        
        self.num_seg_temp = num_seg_temp
        self.num_seg_param = num_seg_param

        self.contrastive = contrastive
        self.supervised = supervised
        
        self.sni_enc = SnippetEncoder(coor_dim=coor_dim,
            num_sni_temp=num_sni_temp, len_sni_temp=len_sni_temp,
            num_sni_param=num_sni_param, max_len=max_len)
        self.sni_dec = SnippetDecoder(
            num_sni_temp=num_sni_temp, len_sni_temp=len_sni_temp,
            max_len=max_len,
            warping_dim=coor_dim+1,
            constrain_scale=self.constrain_scale,
            constrain_trans=self.constrain_trans)
        
        if self.enable_segment:
            self.seg_enc = SegmentEncoder(num_seg_temp=num_seg_temp,
                num_seg_param=num_seg_param, num_sni_temp=num_sni_temp,
                len_sni_temp=len_sni_temp, max_len=max_len,
                lstm_hidden=segenc_lstm_hidden, 
                fc_hidden=segenc_fc_hidden,
                warping_dim=coor_dim+1)
            self.seg_dec = SegmentDecoder(num_seg_temp=num_seg_temp,
                num_seg_param=num_seg_param, num_sni_temp=num_sni_temp,
                len_sni_temp=len_sni_temp, max_len=max_len,
                constrain_scale=self.constrain_scale,
                constrain_trans=self.constrain_trans, warping_dim=coor_dim+1)
            
            self.seg_cls = nn.Linear(num_seg_temp, num_classes)
        else:
            self.seg_cls = nn.Linear(num_sni_temp, num_classes)
    
    def coor_endec(self, x):
        results = {}
        
        sni_params, _ = self.sni_enc(x)
        x_sni = self.sni_dec(sni_params)

        sni_pres, sni_trans = self.sni_dec.transform_params(sni_params)
        sni_params_trans = torch.cat([sni_pres[..., None], sni_trans], -1)
        
        if self.enable_segment:
            seg_params, _ = self.seg_enc(
                sni_params_trans, self.sni_dec.templates.clone().detach())
            seg_pres, pose_pres, pose = self.seg_dec(seg_params)
            x_seg = self.sni_dec.get_decoded_coors(pose_pres, pose, detach_template=True)

            sni_params_recons = torch.cat(
                [pose_pres[..., None], compact_pose(pose)], -1)
        else:
            seg_params = None
            seg_pres = sni_pres.flatten(1)
            x_seg = x_sni
            sni_params_recons = None

        if self.supervised:
            seg_lgts = self.seg_cls(seg_pres)
        else:
            seg_lgts = self.seg_cls(seg_pres.clone().detach())

        if self.enable_segment:
            results['raw_seg_temp'] = self.seg_dec._get_raw_seg_temp(
                self.sni_dec.templates.clone().detach())
        else:
            results['raw_seg_temp'] = None
        results['x_sni'] = x_sni[..., 1:]  # separate scale
        results['x_sni_scale'] = x_sni[..., 0:1]
        results['sni_pres'] = sni_pres
        results['sni_params'] = sni_params

        results['x_seg'] = x_seg[..., 1:]  # separate scale
        results['x_seg_scale'] = x_seg[..., 0:1]
        results['seg_params'] = seg_params
        results['seg_pres'] = seg_pres
        results['seg_lgts'] = seg_lgts

        results['sni_params_trans'] = sni_params_trans
        results['sni_params_recons'] = sni_params_recons

        results['len_sni_temp'] = self.len_sni_temp
        return results


    def forward(self, x):
        results = {}
        
        if self.input_type == 'coors':
            B, T, d = x.shape
            assert (T, d) == (self.max_len, self.coor_dim)
            coors = x
            results.update(self.coor_endec(coors))
        else:
            raise NotImplementedError
        
        results['x'] = x
            
        return results


def loss_fn(x, results, is_valtest=False, **kwargs):
    """
    Keys for loss weight:
    - sni: snippet reconstruction loss
    - seg: segment reconstruction loss
    - cont: smooth regularization
    - reg: sparsity regularization
    - con: constrastive loss
    - cls: auxilliary classification loss
    """
    default_lsw = dict.fromkeys(
        [
            'sni', 'seg', 'cont', 'reg', 'con', 'cls'
        ], 1.0)
    loss_weights = kwargs.get('loss_weights', default_lsw)
    losses = {}

    x_sni, sni_pres, seg_pres = results['x_sni'], results['sni_pres'], results['seg_pres']
    x_seg = results['x_seg']
    sni_params = results['sni_params']
    seg_params = results['seg_params']
    sni_params_trans = results['sni_params_trans']
    sni_params_recons = results['sni_params_recons']
    
    seg_lgts = results['seg_lgts']
    y = kwargs.get('y', None)
    lSnT = results['len_sni_temp']

    if loss_weights.get('sni', 0) > 0:
        losses['sni'] = ((x - x_sni) ** 2).sum((1,2)).mean(0)

    if loss_weights.get('seg', 0) > 0:
        # losses['seg'] = ((x - x_seg) ** 2).sum((1,2)).mean(0)
        losses['seg'] = ((sni_params_trans - sni_params_recons) ** 2).sum((1,2,3)).mean(0)
    
    if loss_weights.get('cls', 0) > 0:
        losses['cls'] = F.nll_loss(F.log_softmax(seg_lgts, dim=-1), y)

    if loss_weights.get('cont', 0) > 0:
        losses['cont'] = 0
        losses['cont'] += ((x_sni[:, lSnT-1:-1:lSnT, :] - x_sni[:, lSnT::lSnT, :]) ** 2).sum((-1,-2)).mean()
        losses['cont'] += ((x_seg[:, lSnT-1:-1:lSnT, :] - x_seg[:, lSnT::lSnT, :]) ** 2).sum((-1,-2)).mean()
    
    if loss_weights.get('reg', 0) > 0:
        losses['reg'] = 0.5*((seg_pres) ** 2).sum(-1).mean()
    
    if loss_weights.get('con', 0) > 0 and seg_pres is not None and not is_valtest:
        B = seg_pres.shape[0]
        _L = int(B/2)
        tau = 0.1
        trj_pres = seg_pres.reshape(B, -1)
        ori, aug = trj_pres.split(_L, 0)
        dist_grid = 1 - cosine_distance(ori, aug)
        
        dist_grid_exp = torch.exp(dist_grid/tau)
        losses['con'] = -torch.log(
            torch.diag(dist_grid_exp) / dist_grid_exp.sum(1)).mean()

    return losses, default_lsw

if __name__ == '__main__':
    net = MCAE(
        contrastive=True, supervised=True, input_type='coors', 
        len_sni_temp=8, enable_segment=True, 
        num_sni_temp=16, num_sni_param=5, num_seg_param=7, coor_dim=2)
    x = torch.Tensor(24, 32, 2).normal_()
    y = torch.randint(0, 20, (24,))
    cy = torch.randint(0, 10, (24,))
    results = net(x)
    
    loss = loss_fn(x, results, y=y, cy=cy)

    import ipdb ; ipdb.set_trace()
