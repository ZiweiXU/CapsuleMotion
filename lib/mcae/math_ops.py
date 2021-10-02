import math

import torch
from torch.nn import functional as F


def relu1(x):
    return F.relu6(x * 6.) / 6.


def flat_reduce(tensor, reduce_type='sum', final_reduce='mean'):
    """Flattens the tensor and reduces it."""

    def _reduce(tensor, how, *args):
        return getattr(torch, how)(tensor, *args)  # pylint:disable=not-callable

    tensor = tensor.reshape(tensor.shape[0], -1)
    tensor = _reduce(tensor, reduce_type, -1)
    if final_reduce is not None:
        tensor = _reduce(tensor, final_reduce)

    return tensor


def safe_log(tensor, eps=1e-16):
    # is_zero = tf.less(tensor, eps)
    # tensor = tf.where(is_zero, tf.ones_like(tensor), tensor)
    # tensor = tf.where(is_zero, tf.zeros_like(tensor) - 1e8, tf.log(tensor))
    is_zero = tensor < eps
    tensor = torch.where(is_zero, torch.ones_like(tensor), tensor)
    tensor = torch.where(is_zero, torch.zeros_like(tensor) - 1e8, tensor.log())
    return tensor


def safe_ce(labels, probs, dim=-1):
    return torch.mean(-torch.sum(labels * safe_log(probs), dim=dim))


def normalize(tensor, dim):
    return tensor / (torch.sum(tensor, dim, keepdim=True) + 1e-8)


def geometric_transform(pose_tensor, similarity=False, nonlinear=True,
                                                as_matrix=False):
    """Convers paramer tensor into an affine or similarity transform.

    Args:
        pose_tensor: [..., 6] tensor.
        similarity: bool.
        nonlinear: bool; applies nonlinearities to pose params if True.
        as_matrix: bool; convers the transform to a matrix if True.

    Returns:
        [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor.
    """

    scale_x, scale_y, theta, shear, trans_x, trans_y = torch.split(
        pose_tensor, 1, -1)

    if nonlinear:
        scale_x, scale_y = (
            torch.sigmoid(i) + 1e-2 for i in (scale_x, scale_y))

        trans_x, trans_y, shear = (
                torch.tanh(i * 5.) for i in (trans_x, trans_y, shear))

        theta = theta * (2. * math.pi)

    else:
        scale_x, scale_y = (abs(i) + 1e-2 for i in (scale_x, scale_y))

    c, s = torch.cos(theta), torch.sin(theta)

    if similarity:
        scale = scale_x
        pose = [scale * c, -scale * s, trans_x, scale * s, scale * c, trans_y]

    else:
        pose = [
                scale_x * c + shear * scale_y * s,
                -scale_x * s + shear * scale_y * c,
                trans_x,
                scale_y * s,
                scale_y * c,
                trans_y
        ]

    pose = torch.cat(pose, -1)

    # convert to a matrix
    if as_matrix:
        # shape = pose.shape[:-1].as_list()
        shape = list(pose.shape[:-1])
        shape += [2, 3]
        pose = torch.reshape(pose, shape)
        zeros = torch.zeros_like(pose[Ellipsis, :1, 0])
        last = torch.stack([zeros, zeros, zeros + 1], -1)
        pose = torch.cat([pose, last], -2)

    return pose


def geometric_transform_3d(pose_tensor, similarity=False, nonlinear=True,
                                                as_matrix=False):
    pose = pose_tensor

    s_x, s_y, s_z, \
        alpha, beta, gamma, \
            sh_x, sh_y, sh_z, \
                t_x, t_y, t_z = torch.split(pose_tensor, 1, -1)

    if nonlinear:
        s_x, s_y, s_z = (
            torch.sigmoid(i) + 1e-2 for i in (s_x, s_y, s_z))

        t_x, t_y, t_z, sh_x, sh_y, sh_z = (
                torch.tanh(i * 5.)
                for i in (t_x, t_y, t_z, sh_x, sh_y, sh_z)
        )

        alpha, beta, gamma = (
            theta * (2. * math.pi) for theta in (alpha, beta, gamma))

    else:
        s_x, s_y, s_z = (
            abs(i) + 1e-2 for i in (s_x, s_y, s_z))

    calpha, salpha = torch.cos(alpha), torch.sin(alpha)
    cbeta, sbeta = torch.cos(beta), torch.sin(beta)
    cgamma, sgamma = torch.cos(gamma), torch.sin(gamma)
    
    r11 = calpha*cbeta
    r12 = calpha*sbeta*sgamma-salpha*cgamma
    r13 = calpha*sbeta*cgamma+salpha*sbeta
    r21 = salpha*cbeta
    r22 = salpha*sbeta*sgamma+calpha*cbeta
    r23 = salpha*sbeta*cgamma-calpha*sgamma
    r31 = -sbeta
    r32 = cbeta*sgamma
    r33 = cbeta*cgamma

    if similarity:
        scale = s_x
        # pose = [scale * c, -scale * s, trans_x, scale * s, scale * c, trans_y]
        pose = [
            scale*r11, scale*r12, scale*r13, t_x,
            scale*r21, scale*r22, scale*r23, t_y,
            scale*r31, scale*r32, scale*r33, t_y,
        ]

    else:
        pose = [
            s_x*r11+sh_x*s_x*r21, s_x*r12+sh_x*s_x*r22, s_x*r13+sh_x*s_x*r23, t_x,
            s_y*r21+sh_y*s_y*r31, s_y*r22+sh_y*s_y*r32, s_y*r23+sh_y*s_y*r33, t_y,
            sh_z*s_z*r11+s_z*r31, sh_z*s_z*r12+s_z*r32, sh_z*s_z*r13+s_z*r33, t_z
        ]

    pose = torch.cat(pose, -1)

    if as_matrix:
        # shape = pose.shape[:-1].as_list()
        shape = list(pose.shape[:-1])
        shape += [3, 4]
        pose = torch.reshape(pose, shape)
        zeros = torch.zeros_like(pose[Ellipsis, :1, 0])
        last = torch.stack([zeros, zeros, zeros, zeros + 1], -1)
        pose = torch.cat([pose, last], -2)
    
    return pose

# https://github.com/pytorch/pytorch/issues/11202
def cosine_distance(x1, x2=None, eps=1e-8):
    assert len(x1.shape) == len(x2.shape) == 2
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
