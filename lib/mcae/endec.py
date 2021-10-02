import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .spatial import full_pose, constrain_pose, compact_pose
from .spatial import fixed_template_initializer, fixed_template_initializer_3d
from .blocks import BatchLinear

class RecurrentFeaturePooler(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RecurrentFeaturePooler, self).__init__()
        self.cvar_dim = kwargs.get('cvar_dim', 32)
        self.lstm_pooler = nn.LSTM(
            self.cvar_dim, self.cvar_dim, 1, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(self.cvar_dim*2)
        self.output_layer = nn.Linear(self.cvar_dim*2, self.cvar_dim)

    def forward(self, x):
        """
        Args:
            x : Tensor of shape (B, L, T, cvar_dim)
        
        Returns:
            Tensor of shape (B, L, cvar_dim)
        """
        B, L, T, _ = x.shape
        x = x.reshape(B*L, T, -1)
        # h = self.lstm_pooler(x)[1][0].permute(1, 0, 2).reshape(B*L, -1)
        h = self.lstm_pooler(x)[0][:, -1, :]
        # h = self.bn(h)
        # h = F.leaky_relu(h)
        out = self.output_layer(h)
        out = out.reshape(B, L, -1)
        return out

class ContentEncoder(nn.Module):
    """ContentEncoder
    ### Args

    ### Inputs

    ### Outputs

    """
    def __init__(
        self, frame_channels=1, cvar_dim=32, c_dim=32, intm_channels=8):
        super(ContentEncoder, self).__init__()
        self.frame_channels = frame_channels
        self.cvar_dim = cvar_dim
        self.c_dim = c_dim
        self.intm_channels = intm_channels

        layers_num = int(np.log2(self.c_dim))-1
        base_channel = self.intm_channels
        layers = [
            nn.Conv2d(self.frame_channels, base_channel, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(layers_num-1):
            layers += [
                nn.Conv2d(base_channel, base_channel*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channel*2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            base_channel *= 2
        layers += [nn.Conv2d(base_channel, self.cvar_dim, 4, 2, 1, bias=False)]
        self.conv_trunk = nn.Sequential(*layers)
        self.scale_predictor = nn.Linear(self.cvar_dim, self.cvar_dim)

    def forward(self, x):
        """
        Args:
            x : Tensor of shape (B, frame_channels, c_dim, c_dim)
        
        Returns:
            z : Tensor of shape (B, cvar_dim)
        """
        assert x.shape[2] == self.c_dim
        B, C, H, W = x.shape

        x = self.conv_trunk(x)
        z = x.squeeze(3).squeeze(2)
        scale = torch.exp(self.scale_predictor(z))
        return z, scale


class ContentDecoder(nn.Module):
    """ContentDecoder
    ### Args

    ### Inputs

    ### Outputs

    """
    def __init__(
        self, frame_channels=1, cvar_dim=32, c_dim=32, intm_channels=8):
        super(ContentDecoder, self).__init__()
        self.frame_channels = frame_channels
        self.cvar_dim = cvar_dim
        self.c_dim = c_dim
        self.intm_channels = intm_channels

        layers_num = int(np.log2(self.c_dim))-1
        base_channel = self.intm_channels * (2 ** (layers_num-1))
        layers = [
            nn.ConvTranspose2d(self.cvar_dim, base_channel, 4, 2, 1),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True)
        ]
        for i in range(layers_num-1):
            layers += [
                nn.ConvTranspose2d(base_channel, base_channel // 2,
                    4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channel//2),
                nn.ReLU(inplace=True),
            ]
            base_channel //= 2
        layers += [
            nn.ConvTranspose2d(base_channel, self.frame_channels,
                4, 2, 1, bias=False),
            nn.Sigmoid()
        ]
        self.trans_conv_trunk = nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z : Tensor of shape (B, L, cvar_dim) or (B, L, cvar_dim, 1, 1)
        
        Returns:
            x : Tensor of shape (B, frame_channels, c_dim, c_dim)
        """
        if len(z.size()) == 3:
            z = z.reshape(*z.shape, 1, 1)
        B, L = z.shape[:2]
        x = self.trans_conv_trunk(z.reshape(B*L, *z.shape[-3:]))
        x = x.reshape(B, L, *x.shape[-3:])
        return x


class ObjectLocator(nn.Module):
    """ObjectLocalizer
    ### Args

    ### Inputs
    - x: Tensor of shape `(B, T, C, H, W)`, the input frames

    ### Outputs
    - Tensor of shape `(B, T, d)` where typically `d=2`. Object coordinates.

    """
    def __init__(self,
        frame_dim=128, frame_feat_channels=64, lstm_hidden_dim=128, max_len=64,
        coor_dim=3, frame_layer_num=1):
        super(ObjectLocator, self).__init__()
        self.frame_dim = frame_dim
        self.frame_feat_channels = frame_feat_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.frame_layer_num = frame_layer_num
        self.max_len = max_len
        self.coor_dim = coor_dim

        self.frame_cnn = ContentEncoder(
            c_dim=frame_dim,
            cvar_dim=frame_feat_channels
            )  # this is different from content encoder
        self.frame_lstm = nn.LSTM(
            frame_feat_channels, lstm_hidden_dim*frame_layer_num,
            bidirectional=True, batch_first=True)
        self.coor_out = nn.Linear(2*lstm_hidden_dim, coor_dim, bias=False)

    def forward(self, x):
        B, T, _, H, W = x.shape
        L = self.frame_layer_num
        results = {}
        assert H == W == self.frame_dim
        assert T == self.max_len
        frame_cnn_feat, _ = self.frame_cnn(x.reshape(B*T, *x.shape[2:]))
        frame_cnn_feat = frame_cnn_feat.reshape(B, T, -1)

        hiddens, (h_o, _) = self.frame_lstm(frame_cnn_feat.reshape(B, T, -1))
        out = self.coor_out(hiddens.reshape(B*T, -1)).reshape(B, T, -1)
        return out


class SnippetEncoder(nn.Module):
    """SnippetEncoder:
    Convert a sequence of coordinates to a sequence of
    snippet parameter groups. The number of groups is calculated as
    `T // len_snippet`.
    
    ---
    ### Args

    ### Attributes

    ### Inputs
    - x: Tensor of shape `(B, T, d)`, where `d` is the dimension of coordinates.

    ### Outputs
    - sni_params_mu: 
        Tensor of shape `(B, nSnG, nSnT, nSnP)`
    - sni_params_sigma:
        Tensor of shape `(B, nSnG, nSnT, nSnT)`

    """
    def __init__(self,
        num_sni_temp=16, len_sni_temp=8, intm_channels=8, max_len=64,
        num_sni_param=5, num_object=1, coor_dim=2):
        super(SnippetEncoder, self).__init__()
        self.num_sni_temp = num_sni_temp
        self.len_sni_temp = len_sni_temp
        self.intm_channels = intm_channels
        self.max_len = max_len
        self.num_sni_param = num_sni_param
        self.num_object = num_object
        self.coor_dim = coor_dim

        assert max_len % len_sni_temp == 0
        self.sni_temp_group_num = max_len // len_sni_temp

        # TODO: experimental max_len -> len_sni_temp
        # _num_layers = int(np.log2(max_len)) - 1
        # _num_layers = int(np.log2(len_sni_temp)) - 1
        # TODO: limited f_conv
        _num_layers = int(np.log2(8)) - 1

        _base_channels = intm_channels
        _layers = [
            nn.Conv1d(coor_dim, _base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(_base_channels),
            nn.LeakyReLU()]
        for _ in range(_num_layers):
            _layers += [
                nn.Conv1d(
                    _base_channels, _base_channels*2, 4, 2, 1, bias=False),
                nn.BatchNorm1d(_base_channels*2),
                nn.LeakyReLU()]
            _base_channels *= 2
        
        if len_sni_temp > 8:
            _layers += [nn.AdaptiveAvgPool1d(1)]
        self.sni_feat_extactor = nn.Sequential(*_layers)

        # self.mu_gen = nn.Conv1d(
        #     _base_channels, self.sni_temp_group_num*num_sni_temp*num_sni_param,
        #     1, 1, 0)
        self.mu_gen = nn.Conv1d(
            _base_channels, num_sni_temp*num_sni_param,
            1, 1, 0)

    def forward(self, x):
        B, _, _ = x.shape
        nSnT, lSnT, nSnG, nSnP = self.num_sni_temp, self.len_sni_temp, \
            self.sni_temp_group_num, self.num_sni_param
        
        x_ = torch.cat(x.split(lSnT, 1), 0)
        
        sni_feat = self.sni_feat_extactor(x_.transpose(1, 2))
        sni_params_mu = self.mu_gen(sni_feat).reshape(nSnG, B, nSnT, nSnP).transpose(0, 1)

        # sni_params_mu = self.mu_gen(sni_feat).reshape(B, nSnG, nSnT, nSnP)
        return sni_params_mu, None


class SnippetDecoder(nn.Module):
    """SnippetDecoder:
    Convert a set of snippet parameters to a sequence of warping parameters.
    ---
    ### Args

    ### Attributes
    - templates: shaped (nSnT, lSnT, warping_dim=3)

    ### Inputs
    - x: shaped (B, nSnG, nSnT, nSnP)

    ### Outputs
    - coors: shaped (B, T, d) where typically `d=3` for (scale, tx, ty) at last
        dim.

    """
    def __init__(self,
        num_sni_temp=16, len_sni_temp=8, warping_dim=3, max_len=64,
        constrain_scale=1, constrain_trans=1.5):
        super(SnippetDecoder, self).__init__()
        self.num_sni_temp = num_sni_temp
        self.num_sni_group = max_len // len_sni_temp
        self.warping_dim = warping_dim
        self.constrain_scale = constrain_scale
        self.constrain_trans = constrain_trans

        if self.warping_dim == 3:
            _template_data = fixed_template_initializer(
                num_sni_temp=num_sni_temp, len_sni_temp=len_sni_temp, scale_factor=1)
        else:
            _template_data = fixed_template_initializer_3d(
                num_sni_temp=num_sni_temp, len_sni_temp=len_sni_temp, scale_factor=1)
        assert _template_data.shape == (num_sni_temp, len_sni_temp, warping_dim)
        self.templates = nn.Parameter(_template_data)

    def forward(self, x):
        B, nSnG, nSnT, nSnP = x.shape
        pres, trans = self.transform_params(x)
        dim = 6 if self.warping_dim == 3 else 12
        sw_geo = full_pose(trans.reshape(-1, dim))\
            .reshape(B, nSnG, -1, self.warping_dim, self.warping_dim)
        return self.get_decoded_coors(pres, sw_geo)

    def transform_params(self, x):
        """Transform a set of transformation parameters to 
        geometrically meaningful form.
        """
        if x.shape[-1] == 8:
            return self.transform_params_3d(x)

        pres, sxy, tx, ty, theta = x.split(1, -1)

        if self.training:
            pres = pres + torch.zeros_like(pres).uniform_(-2, 2)
        pres = torch.sigmoid(pres).squeeze(-1)

        # # spatially warping templates
        # _temp = torch.cat([sxy, tx, ty], dim=-1)
        sxy = 2 * torch.sigmoid(sxy)
        theta = theta * 2 * math.pi
        st, ct = torch.sin(theta), torch.cos(theta)
        _temp = torch.cat([sxy*ct, -sxy*st, tx, sxy*st, sxy*ct, ty], -1)
        _temp = constrain_pose(
            _temp, scale=self.constrain_scale, trans_range=self.constrain_trans)
        return pres, _temp
    
    def transform_params_3d(self, x):
        pres, s, tx, ty, tz, alpha, beta, gamma = x.split(1, -1)

        if self.training:
            pres = pres + torch.zeros_like(pres).uniform_(-2, 2)
        pres = torch.sigmoid(pres).squeeze(-1)

        # # spatially warping templates
        # _temp = torch.cat([sxy, tx, ty], dim=-1)
        s = 2 * torch.sigmoid(s)
        alpha, beta, gamma = [i * 2 * math.pi for i in (alpha, beta, gamma)]
        sa, ca = torch.sin(alpha), torch.cos(alpha)
        sb, cb = torch.sin(beta), torch.cos(beta)
        sg, cg = torch.sin(gamma), torch.cos(gamma)

        a11, a12, a13 = ca*cb, (ca*sb*sg-sa*cg), (ca*sb*cg+sa*sg)
        # a14 = tx*a11+ty*a12+tz*a13
        a21, a22, a23 = sa*cb, (sa*sb*sg+ca*cg), (sa*sb*cg-ca*sg)
        # a24 = tx*a21+ty*a22+tz*a23
        a31, a32, a33 = -sb, cb*sg, cb*cg
        # a34 = tx*a31+ty*a32+tz*a33

        _temp = torch.cat([
            s*a11, s*a12, s*a13, tx, s*a21, s*a22, s*a23, ty, s*a31, s*a32, s*a33, tz], -1)
        _temp = constrain_pose(
            _temp, scale=self.constrain_scale, trans_range=self.constrain_trans)
        return pres, _temp

    def get_decoded_coors(self, pres, sw_geo, detach_template=False):
        if self.warping_dim not in [3, 4]: 
            raise NotImplementedError("warping_dim!=3,4 is not implemented.")

        B, nSnG = pres.shape[0], self.num_sni_group
        
        templates = self.templates.clone().detach() if detach_template else self.templates
        if self.warping_dim == 3:
            ss, xs, ys = templates.split(1, -1)
            _temp = torch.cat([xs, ys, torch.ones_like(xs)], -1)
            nxys = (sw_geo.unsqueeze(-3) @ _temp[None, None, :, :, :, None])\
                .squeeze(-1)[..., :2]
        elif self.warping_dim == 4:
            ss, xs, ys, zs = templates.split(1, -1)
            _temp = torch.cat([xs, ys, zs, torch.ones_like(xs)], -1)
            nxys = (sw_geo.unsqueeze(-3) @ _temp[None, None, :, :, :, None])\
                .squeeze(-1)[..., :3]
        sw_templates = torch.cat(
            [ss[None, None, ...].expand(B, nSnG, -1, -1, -1), nxys], -1)

        weighted_template = torch.einsum('bgn,bgnli->bgli', pres, sw_templates)
        weighted_template = weighted_template / pres.sum(2)[..., None, None]
        coors = weighted_template.reshape(B, -1, self.warping_dim)

        return coors


class SegmentEncoder(nn.Module):
    """
    Take a set of snippet parameter groups, and convert it to a set of segment
    parameters.
    
    The segment parameters models the whole input sequence, by looking at the
    groups of snippet parameters. 
    A segment template is described in terms of all the snippet templates by
    learnable attribute `seg_sni_rel`. 
    This attribute describes how snippet templates are transformed to a
    segment template. 
    This is similar to the fixed (but learnable) "part-object" relationship
    presented in SCAE.

    A segment parameter describes how a specific segment is transformed into
    the actual sequence that is being viewed.
    This is similar to the predicted "object-viewer" relationship.
    A segment parameter consists of the following:
    - A weight (or presence) scalar
    - The "object-viewer" relationship, in the form of (sxy, tx, ty, theta)
    
    ----------------------------------------------------------------------------
    ### Args

    ### Inputs
    - sni_params: shaped (B, nSnG, nSnT, nSnP???)
    - sni_temp: shaped (nSnT, lSnT, d) where typically d=3

    ### Outputs
    - seg_feat: shape (B, nSeT, nSeP)

    """
    def __init__(self,
        num_sni_temp=16, len_sni_temp=8, num_seg_temp=48,
        max_len=64, warping_dim=3, num_seg_param=1+6,
        lstm_hidden=128, fc_hidden=32):
        super(SegmentEncoder, self).__init__()
        self.num_seg_temp = num_seg_temp
        self.num_sni_group = max_len // len_sni_temp
        self.num_seg_param = num_seg_param

        num_sni_param_trans = 7 if warping_dim == 3 else 13
        
        self.sni_seq_encoder = nn.LSTM(
            num_sni_temp*(num_sni_param_trans+len_sni_temp*warping_dim),
            lstm_hidden, batch_first=True, bidirectional=True)

        self.seg_param_gen_1 = nn.Linear(lstm_hidden*2, num_seg_temp*fc_hidden)
        self.seg_param_gen_2 = BatchLinear(
            fc_hidden, num_seg_param, indep_dims=[num_seg_temp])
        # self.seg_param_v_gen = BatchLinear(
        #     32, num_seg_param, indep_dims=[num_seg_temp])
        

    def forward(self, sni_params, sni_temp):
        B, nSnG, nSnT, nSnP = sni_params.shape
        nSeT = self.num_seg_temp
        _, (ho, _) = self.sni_seq_encoder(
            torch.cat(
                [
                    sni_params.reshape(B, nSnG, nSnT, nSnP),
                    sni_temp.reshape(1, 1, nSnT, -1).expand(B, nSnG, -1, -1)
                ], -1).reshape(B, nSnG, -1),
        )
        _temp = F.relu(
            self.seg_param_gen_1(F.relu(ho.permute(1, 0, 2).reshape(B, -1))))\
                .reshape(B, nSeT, -1)
        seg_params = self.seg_param_gen_2(_temp)
        # seg_params_v = self.seg_param_v_gen(_temp)
        return seg_params, None


class SegmentDecoder(nn.Module):
    """Decode segment parameters into snippet parameters. The output can be
    directly fed into a snippet decoder to recover the original input sequence.

    ----------------------------------------------------------------------------
    ### Args

    ### Attributes
    - seg_sni_rel

    ### Inputs
    - seg_params: shaped (B, nSeT, nSeP)

    ### Outputs
    - weighted_pres: shaped (B, nSnG, nSnT)
    - pose: shaped (B, nSnG, nSnT, 6)

    """
    def __init__(self,
        num_sni_temp=16, len_sni_temp=8, num_seg_temp=48, num_seg_param=1+6,
        max_len=64, warping_dim=3, constrain_scale=1, constrain_trans=1.5):
        super(SegmentDecoder, self).__init__()
        self.num_seg_temp = num_seg_temp
        self.num_seg_param = num_seg_param
        self.num_sni_group = max_len // len_sni_temp
        self.num_sni_temp = num_sni_temp
        self.warping_dim = warping_dim
        self.constrain_scale = constrain_scale
        self.constrain_trans = constrain_trans

        _limit = np.sqrt(6 / (self.num_sni_group + num_seg_param))
        if self.warping_dim == 3:
            _base = torch.zeros(
                self.num_sni_group, num_seg_temp, num_sni_temp, 5)\
                        .uniform_(-_limit, _limit)
        else:
            _base = torch.zeros(
                self.num_sni_group, num_seg_temp, num_sni_temp, 8)\
                        .uniform_(-_limit, _limit)
            _base[..., 1] = 0
        self.seg_sni_rel = nn.Parameter(_base)
    
    def _get_raw_seg_temp(self, sni_temp):
        """Returns the raw segment template, without transformed by segment 
        parameters.
        """
        nSeT, nSeP = self.num_seg_temp, self.num_seg_param
        nSnG, nSnT = self.num_sni_group, self.num_sni_temp
        lSnT = sni_temp.shape[1]

        if self.warping_dim == 3:
            op_pres, op_sxy, op_tx, op_ty, op_theta = self.seg_sni_rel.split(1, -1)
            op_pres = torch.sigmoid(op_pres)
            op_sxy = 2 * torch.sigmoid(op_sxy)
            op_theta = op_theta * 2 * math.pi
            op_st, op_ct = torch.sin(op_theta), torch.cos(op_theta)
            op = torch.cat(
                [op_sxy*op_ct, -op_sxy*op_st, op_tx, op_sxy*op_st, op_sxy*op_ct, op_ty], -1)
            op = constrain_pose(op)
            op = full_pose(op.reshape(-1, 6)).reshape(nSnG, nSeT, nSnT, self.warping_dim, self.warping_dim)
        elif self.warping_dim == 4:
            op_pres, s, tx, ty, tz, alpha, beta, gamma = self.seg_sni_rel.split(1, -1)
            s = 2 * torch.sigmoid(s)
            alpha, beta, gamma = [i * 2 * math.pi for i in (alpha, beta, gamma)]
            sa, ca = torch.sin(alpha), torch.cos(alpha)
            sb, cb = torch.sin(beta), torch.cos(beta)
            sg, cg = torch.sin(gamma), torch.cos(gamma)

            a11, a12, a13 = ca*cb, (ca*sb*sg-sa*cg), (ca*sb*cg+sa*sg)
            # a14 = tx*a11+ty*a12+tz*a13
            a21, a22, a23 = sa*cb, (sa*sb*sg+ca*cg), (sa*sb*cg-ca*sg)
            # a24 = tx*a21+ty*a22+tz*a23
            a31, a32, a33 = -sb, cb*sg, cb*cg
            # a34 = tx*a31+ty*a32+tz*a33

            op = torch.cat([
                s*a11, s*a12, s*a13, tx, s*a21, s*a22, s*a23, ty, s*a31, s*a32, s*a33, tz], -1)
            op = constrain_pose(op)
            op = full_pose(op.reshape(-1, 12)).reshape(nSnG, nSeT, nSnT, self.warping_dim, self.warping_dim)

        pose = op

        pose = full_pose(
            constrain_pose(
                        compact_pose(pose.reshape(-1, self.warping_dim, self.warping_dim)), 
                        scale=1, 
                        trans_range=1.5)
                    ).reshape(*pose.shape)
        sni_temps = sni_temp.clone().detach()[..., 1:]
        sni_temps = torch.cat([sni_temps, torch.ones(*sni_temps.shape[:-1], 1).to(sni_temps.device)], -1)
        full_sni_temp_trans = pose.unsqueeze(-3) @ sni_temps[None, None, ..., None]
        weighted_sni_temp = (
            (op_pres[None, ..., None, None] * full_sni_temp_trans).sum(3, keepdim=True) 
            / op_pres[None, ..., None, None].sum(3, keepdim=True)
        ).squeeze().permute(1,0,2,3).reshape(nSeT, nSnG*lSnT, self.warping_dim)
        if self.warping_dim == 3:
            weighted_sni_temp = weighted_sni_temp[..., [2,0,1]].detach()
        else:
            weighted_sni_temp = weighted_sni_temp[..., [3,0,1,2]].detach()
        return weighted_sni_temp

    def forward(self, seg_params):
        B, nSeT, nSeP = seg_params.shape
        nSnG, nSnT = self.num_sni_group, self.num_sni_temp
        nSeT = self.num_seg_temp

        if self.warping_dim == 3:
            seg_pres, seg_warp_params = seg_params.split([1, 6], -1)
            seg_warp_params = constrain_pose(seg_warp_params)
            ov = full_pose(
                seg_warp_params.reshape(-1, 6)).reshape(B, 1, nSeT, 1, self.warping_dim, self.warping_dim)
            
            op_pres, op_sxy, op_tx, op_ty, op_theta = self.seg_sni_rel.split(1, -1)
            op_sxy = 2 * torch.sigmoid(op_sxy)
            op_theta = op_theta * 2 * math.pi
            op_st, op_ct = torch.sin(op_theta), torch.cos(op_theta)
            op = torch.cat(
                [op_sxy*op_ct, -op_sxy*op_st, op_tx, op_sxy*op_st, op_sxy*op_ct, op_ty], -1)
            op = constrain_pose(op)
            op = full_pose(op.reshape(-1, 6)).reshape(1, nSnG, nSeT, nSnT, self.warping_dim, self.warping_dim)
        elif self.warping_dim == 4:
            seg_pres, seg_warp_params = seg_params.split([1, 12], -1)
            seg_warp_params = constrain_pose(seg_warp_params)
            ov = full_pose(
                seg_warp_params.reshape(-1, 12)).reshape(B, 1, nSeT, 1, self.warping_dim, self.warping_dim)
            
            op_pres, s, tx, ty, tz, alpha, beta, gamma = self.seg_sni_rel.split(1, -1)
            s = 2 * torch.sigmoid(s)
            alpha, beta, gamma = [i * 2 * math.pi for i in (alpha, beta, gamma)]
            sa, ca = torch.sin(alpha), torch.cos(alpha)
            sb, cb = torch.sin(beta), torch.cos(beta)
            sg, cg = torch.sin(gamma), torch.cos(gamma)

            a11, a12, a13 = ca*cb, (ca*sb*sg-sa*cg), (ca*sb*cg+sa*sg)
            # a14 = tx*a11+ty*a12+tz*a13
            a21, a22, a23 = sa*cb, (sa*sb*sg+ca*cg), (sa*sb*cg-ca*sg)
            # a24 = tx*a21+ty*a22+tz*a23
            a31, a32, a33 = -sb, cb*sg, cb*cg
            # a34 = tx*a31+ty*a32+tz*a33

            op = torch.cat([
                s*a11, s*a12, s*a13, tx, s*a21, s*a22, s*a23, ty, s*a31, s*a32, s*a33, tz], -1)
            op = constrain_pose(op)
            op = full_pose(op.reshape(-1, 12)).reshape(nSnG, nSeT, nSnT, self.warping_dim, self.warping_dim)
        
        # TODO: experimental: commented by ov-after-weighting
        unweighted_pose = (ov @ op)  # shaped (B, nSnG, nSeT, nSnT, 3, 3)
        # unweighted_pose = op

        if self.training:
            seg_pres = seg_pres + torch.zeros_like(seg_pres).uniform_(-2, 2)
            op_pres = op_pres + torch.zeros_like(op_pres).uniform_(-2, 2)
        
        seg_pres = torch.sigmoid(seg_pres).squeeze(-1)  # shaped (B, nSeT)
        op_pres = torch.sigmoid(op_pres).squeeze(-1)  # shaped (nSnG, nSeT, nSnT)

        # b: batch, g: nSeT, r: nSnG, l: nSnT
        pose_pres = torch.einsum('bg,rgl->brl', seg_pres, op_pres)
        pose_pres = pose_pres / seg_pres.sum(-1)[:, None, None]
        pose = torch.einsum(
            'bg,brglij->brlij', seg_pres, unweighted_pose) \
            / seg_pres.sum(-1)[:, None, None, None, None]

        # TODO: experimental: added by ov-after-weighting
        # pose = ov.squeeze(3) @ pose
        pose = full_pose(
            constrain_pose(
                compact_pose(pose.reshape(-1, self.warping_dim, self.warping_dim)), 
                scale=self.constrain_scale, 
                trans_range=self.constrain_trans)
            ).reshape(*pose.shape)

        return seg_pres, pose_pres, pose


if __name__ == '__main__':
    sni_enc = SnippetEncoder()
    sni_dec = SnippetDecoder()
    seg_enc = SegmentEncoder()
    seg_dec = SegmentDecoder()

    x = torch.Tensor(24, 64, 2).normal_()
    sni_params = sni_enc(x)
    x_ = sni_dec(sni_params[0])
    seg_params = seg_enc(sni_params[0], sni_dec.templates)
    pose_pres, pose = seg_dec(seg_params)
    x__ = sni_dec.get_decoded_coors(pose_pres, pose)
    import ipdb; ipdb.set_trace()
