import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.nn import functional as F

from .tensor_ops import init_truncated_normal


def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation, inplace=False))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x

class BatchLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim,
        indep_dims=[1],
        use_bias=True
    ):
        """Builds the module.

        The input to this module will be a tensor shaped
            [t1, t2, ..., i1, i2, ..., in_dim]. The weight of this module will
            be [i1, i2, ..., out_dim, in_dim], which means weights along axis
            [i1, i2, ...] will not be shared.
        
        Args:
            n_units: int, number of output dimensions.
            indep_dims: sorted list of ints [i1, i2, ...].
                Weights along these axis are not shared.
            use_bias: boolean.
        """
        super(BatchLinear, self).__init__()
        # self._tile_dims = sorted(tile_dims)
        self._use_bias = use_bias
        self._indep_dims = indep_dims

        self.weight = nn.Parameter(
                        Tensor(*(indep_dims + [out_dim, in_dim])))
        init_truncated_normal(self.weight, fan_in=in_dim)

        if use_bias:
            self.bias = nn.init.zeros_(
                nn.Parameter(Tensor(*(indep_dims + [out_dim]))))
        else:
            self.bias = Tensor(torch.zeros(out_dim))

    def forward(self, x):
        """
        Args:
            x: tensor of shape [t1, t2, ..., i1, i2, ..., in_dim].

        Returns:
            Tensor of shape [t1, t2, ..., i1, i2, ..., out_dim].
        """
        return (x.unsqueeze(-2) @ self.weight.transpose(-1, -2)).squeeze(-2)\
                + self.bias.to(x.device)


class BatchMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        n_hiddens,
        activation=F.relu,
        activate_final=False,
        use_bias=True,
        tile_dims=[0],
        indep_dims=[32]
    ):
        super(BatchMLP, self).__init__()
        self._in_dim = in_dim
        self._n_hiddens = n_hiddens
        self._activation = activation
        self._activate_final = activate_final
        self._use_bias = use_bias
        self._tile_dims = tile_dims
        self._indep_dims = indep_dims

        self.interim_layers = nn.ModuleList(
            [BatchLinear(
                in_dim, n_hiddens[0],
                indep_dims=self._indep_dims, use_bias=True)]
            +
            [BatchLinear(
                n_hidden, n_hidden,
                indep_dims=self._indep_dims, use_bias=True)
                for n_hidden in self._n_hiddens[1:-1]]
        )
        self.final_layer = BatchLinear(
            self._n_hiddens[-2], self._n_hiddens[-1],
            indep_dims=self._indep_dims, use_bias=self._use_bias
        )

    def forward(self, x):
        h = x
        for layer in self.interim_layers:
            h = self._activation(layer(h))
        h = self.final_layer(h)
        if self._activate_final:
            h = self._activation(h)
        
        return h
