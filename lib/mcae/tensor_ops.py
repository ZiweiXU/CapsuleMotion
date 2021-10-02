import math

import torch
from torch import Tensor
from torch.distributions import Distribution as D
from torch.nn.init import _calculate_fan_in_and_fan_out


# https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
def _truncated_normal(tensor, mean=0, std=1, thres=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < thres) & (tmp > -thres)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_truncated_normal(param, stdv=None, fan_in=None):
    if stdv is None:
        if fan_in is None:
            fan_in, _ = _calculate_fan_in_and_fan_out(param)
        stdv = 1 / math.sqrt(fan_in)
    with torch.no_grad():
        _truncated_normal(param.data, std=stdv)
        return param


def gather_3rd(params, indices):
    """Special case of tf.gather_nd where indices.shape[-1] == 3
    
    Check https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/gather_nd
        for details.

    Args:
        params: Tensor, shaped [B, d_1, ..., d_i], di >= 2
        indices: LongTensor, shaped [B, i_1, ...,i_{k-1}, 3]
    Returns:
        Tensor, of shape indices.shape[:-1] + params.shape[indices.shape[-1]:]
    """
    return params[indices[..., 0], indices[..., 1], indices[..., 2]]


def add_noise(tensor, noise_type='uniform', noise_scale=1.0):
    """Adds noise to tensors."""
    if noise_type == 'uniform':
        # noise = tf.random.uniform(
        #     tensor.shape,
        #     minval=-.5, maxval=.5
        # ) * self._noise_scale
        noise = \
            Tensor(tensor.shape).uniform_(-.5, .5) * noise_scale

    elif noise_type == 'logistic':
        # pdf = tfd.Logistic(0., self._noise_scale)
        pdf = D.LogisticNormal(0., noise_scale)
        noise = pdf.sample(tensor.shape)

    elif not noise_type:
        noise = 0.

    else:
        raise ValueError(
            'Invalid noise type: "{}".'.format(noise_type)
        )

    return tensor + noise.to(tensor.device)
