"""
Credit: DDPAE and SCAE code.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.autograd import Variable


def fixed_template_initializer(
  num_sni_temp=32, len_sni_temp=8, scale_factor=1.0, mode='2d'):
  if mode == '3d':
    return fixed_template_initializer_3d(
      num_sni_temp=num_sni_temp, len_sni_temp=len_sni_temp, scale_factor=scale_factor)
  W = torch.zeros(num_sni_temp, 3)
  scale = scale_factor
  trans_delta = scale / len_sni_temp
  torch.nn.init.uniform_(W, a=-trans_delta, b=trans_delta)
  W = full_pose(constrain_pose(W, scale=scale, trans_range=trans_delta))

  x0 = torch.zeros(num_sni_temp, 3, 1)
  torch.nn.init.uniform_(x0, a=-0.5/len_sni_temp, b=0.5/len_sni_temp)
  x0[..., 2, 0] = 1

  xs = [x0]
  for _ in range(len_sni_temp-1):
      xs.append(W @ xs[-1])
  xs = torch.stack(xs, 1).squeeze(-1)
  _zero_padding = torch.Tensor([0.])[None, None, :]\
    .repeat(num_sni_temp, len_sni_temp, 1)
  xs = torch.cat(
      [
          _zero_padding,
          xs[:, :, :2]
      ], -1)
  return xs


def fixed_template_initializer_3d(
  num_sni_temp=32, len_sni_temp=8, scale_factor=1.0):
  W = torch.zeros(num_sni_temp, 12)
  scale = scale_factor
  trans_delta = scale / len_sni_temp
  torch.nn.init.uniform_(W, a=-trans_delta, b=trans_delta)
  W = full_pose(constrain_pose(W, scale=scale, trans_range=trans_delta))

  x0 = torch.zeros(num_sni_temp, 4, 1)
  torch.nn.init.uniform_(x0, a=-0.5/len_sni_temp, b=0.5/len_sni_temp)
  x0[..., 3, 0] = 1

  xs = [x0]
  for _ in range(len_sni_temp-1):
      xs.append(W @ xs[-1])
  xs = torch.stack(xs, 1).squeeze(-1)
  _zero_padding = torch.Tensor([0.])[None, None, :]\
    .repeat(num_sni_temp, len_sni_temp, 1)
  xs = torch.cat(
      [
          _zero_padding,
          xs[:, :, :3]
      ], -1)
  return xs

def fixed_template_initializer1(
  num_sni_temp=32, len_sni_temp=8, scale_factor=1.0):

  step = scale_factor / len_sni_temp
  t0 = torch.zeros((num_sni_temp, 1)).uniform_(0, 2*3.1416)
  ts = torch.zeros((num_sni_temp, len_sni_temp-1)).uniform_(-3.1416/12, 3.1416/12)
  ts = torch.cat([t0, t0.expand_as(ts) + ts], -1)
  dx = step * torch.cat([torch.cos(ts)[..., None], torch.sin(ts)[..., None]], -1)
  x0 = torch.zeros(num_sni_temp, 1, 2)
  xs = [x0]
  for i in range(1, len_sni_temp):
    xs.append(xs[-1] + dx[:, i:i+1, :])
  xs = torch.cat(xs, -2)
  # xs = torch.zeros((num_sni_temp, len_sni_temp-1, 2))
  # xs = torch.cat([x0, x0.expand_as(dx) + dx], -2)[:, :-1, :]

  _zero_padding = torch.Tensor([0.])[None, None, :]\
    .repeat(num_sni_temp, len_sni_temp, 1)
  xs = torch.cat([_zero_padding, xs], -1)
  return xs


def expand_pose(pose):
  '''
  param pose: N x 3
  Takes 3-dimensional vectors, and massages them into 2x3 affine transformation matrices:
  [s,x,y] -> [[s,0,x],
              [0,s,y]]
  '''
  n = pose.size(0)
  device = pose.device
  expansion_indices = Variable(torch.LongTensor([1, 0, 2, 0, 1, 3]).to(device), requires_grad=False)
  zeros = Variable(torch.zeros(n, 1).to(device), requires_grad=False)
  out = torch.cat([zeros, pose], dim=1)
  return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)


def full_pose(pose):
    """Expand (B, 6) or (B, 3) pose to (B, 3, 3) transformation matrix."""
    if pose.shape[-1] == 3:
      pose = expand_pose(pose.reshape(-1, 3)).reshape(*pose.shape[:-1], 6)
    
    if pose.shape[-1] == 12:
      return full_pose_3d(pose)
    
    shape = list(pose.shape[:-1])
    shape += [2, 3]
    pose = torch.reshape(pose, shape)
    zeros = torch.zeros_like(pose[Ellipsis, :1, 0])
    last = torch.stack([zeros, zeros, zeros + 1], -1)
    full_pose = torch.cat([pose, last], -2)
    return full_pose


def full_pose_3d(pose):
  """Expand (B, 12) pose to (B, 4, 4) transformation matrix."""
    
  shape = list(pose.shape[:-1])
  shape += [3, 4]
  pose = torch.reshape(pose, shape)
  zeros = torch.zeros_like(pose[..., :1, 0])
  last = torch.stack([zeros, zeros, zeros, zeros + 1], -1)
  full_pose = torch.cat([pose, last], -2)
  return full_pose


def compact_pose(full_pose):
  """Squeeze (B, 3, 3) to (B, 6) compact pose."""
  if full_pose.shape[-1] == 4:
    return compact_pose_3d(full_pose)
  return full_pose[..., :-1, :].reshape(*full_pose.shape[:-2], 2*3)


def compact_pose_3d(full_pose):
  """Squeeze (B, 4, 4) to (B, 12) compact pose."""
  return full_pose[..., :-1, :].reshape(*full_pose.shape[:-2], 3*4)


def constrain_pose(pose, scale=1.0, trans_range=1.5):
    '''
    Constrain the value of the pose vectors.
    
    Args:
      pose :
        Tensor of shape (..., 3) [s, x, y] or (..., 6) [sx, kx, tx, ky, sy, ty]
      scale : Float.
    
    Returns:
      pose : Tensor of shape (..., 3) or (..., 6)
    '''
    # Makes training faster.
    pose_dim = pose.shape[-1]
    assert pose_dim in [3, 6, 12]
    if pose_dim == 12:
      return constrain_pose_3d(pose, scale=scale, trans_range=trans_range)
    if pose_dim == 3:
      # scale = torch.clamp(pose[..., :1], scale - 1, scale + 1)
      # xy = torch.tanh(pose[..., 1:]) * (scale - 0.5)
      scales, trans_x, trans_y = torch.split(pose, 1, -1)
      # soft contraint between (scale-1, scale+1)
      scales = (2*torch.sigmoid(scales) + scale - 1)
      # trans_x, trans_y = (
      #         trans_range * torch.tanh(i) for i in (trans_x, trans_y))
      trans_x, trans_y = (
              i.clamp(-trans_range, trans_range) for i in (trans_x, trans_y))
      pose = torch.cat([scales, trans_x, trans_y], dim=-1)
    elif pose_dim == 6:
      scale_x, shear_x, trans_x, shear_y, scale_y, trans_y = \
          torch.split(pose, 1, -1)
      # scale_x = torch.clamp(pose[..., 0:1], scale - 1, scale + 1)
      # scale_y = torch.clamp(pose[..., 4:5], scale - 1, scale + 1)
      # trans_x = torch.tanh(pose[..., 2:3]) * (scale - 0.5)
      # trans_y = torch.tanh(pose[..., 5:6]) * (scale - 0.5)
      # shear_x = torch.tanh(pose[..., 1:2]) * (scale - 0.5)
      # shear_y = torch.tanh(pose[..., 3:4]) * (scale - 0.5)
      scale_x, scale_y = (
              2*torch.sigmoid(i) + scale - 1 for i in (scale_x, scale_y))
      # trans_x, trans_y, shear_x, shear_y = (
      #         trans_range * torch.tanh(i) for i in (trans_x, trans_y, shear_x, shear_y))
      trans_x, trans_y, shear_x, shear_y = (
              i.clamp(-trans_range, trans_range) for i in (trans_x, trans_y, shear_x, shear_y))
      pose = torch.cat(
          [scale_x, shear_x, trans_x, shear_y, scale_y, trans_y], dim=-1)
    return pose


def constrain_pose_3d(pose, scale=1.0, trans_range=1.5):
    '''
    Constrain the value of the pose vectors.
    
    Args:
      pose :
        Tensor of shape (..., 12) 
          [a11, a12, a13, tx, a21, a22, a23, ty, a31, a32, a33, tz]
      scale : Float.
    
    Returns:
      pose : Tensor of shape (..., 12)
    '''
    # Makes training faster.
    pose_dim = pose.shape[-1]
    assert pose_dim in [12]
    a11, a12, a13, tx, a21, a22, a23, ty, a31, a32, a33, tz = \
        torch.split(pose, 1, -1)
    a11, a22, a33 = (
            2*torch.sigmoid(i) + scale - 1 for i in (a11, a22, a33))
    a12, a13, tx, a21, a23, ty, a31, a32, tz = (
            i.clamp(-trans_range, trans_range) for i in 
            (a12, a13, tx, a21, a23, ty, a31, a32, tz))
    pose = torch.cat(
        [a11, a12, a13, tx, a21, a22, a23, ty, a31, a32, a33, tz], dim=-1)
    return pose


def geometric_transform(pose_tensor, similarity=True, nonlinear=False,
                                                as_matrix=False):
    """Convers paramer tensor into an affine or similarity transform.

    Args:
        pose_tensor: [..., 3] or [..., 6] tensor.
        similarity: bool.
        nonlinear: bool; applies nonlinearities to pose params if True.
        as_matrix: bool; convers the transform to a matrix if True.

    Returns:
        [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor.
    """

    if pose_tensor.shape[-1] == 3:
      pose_tensor = expand_pose(pose_tensor.reshape(-1, 3))\
        .reshape(*pose_tensor.shape[:-1], 6)

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
        pose = full_pose(pose)

    return pose


def pose_inv(pose):
  '''
  param pose: N x 3
  [s,x,y] -> [1/s,-x/s,-y/s]
  '''
  N, _ = pose.size()
  ones = Variable(torch.ones(N, 1).to(pose.device), requires_grad=False)
  out = torch.cat([ones, -pose[:, 1:]], dim=1)
  out = out / pose[:, 0:1]
  return out


def pose_inv_full(pose):
  '''
  param pose: N x 6
  Inverse the 2x3 transformer matrix.
  '''
  N, _ = pose.size()
  b = pose.view(N, 2, 3)[:, :, 2:]
  # A^{-1}
  # Calculate determinant
  determinant = (pose[:, 0] * pose[:, 4] - pose[:, 1] * pose[:, 3] + 1e-8).view(N, 1)
  indices = torch.LongTensor([4, 1, 3, 0]).to(pose.device)
  scale = torch.Tensor([1, -1, -1, 1]).to(pose.device)
  A_inv = torch.index_select(pose, 1, indices) * scale / determinant
  A_inv = A_inv.view(N, 2, 2)
  # b' = - A^{-1} b
  b_inv = - A_inv.matmul(b).view(N, 2, 1)
  transformer_inv = torch.cat([A_inv, b_inv], dim=2)
  return transformer_inv


def image_to_object(images, pose, object_size):
  '''Inverse pose, crop and transform image patches.
  
  Args:
    images : Tensor of shape (B, T, C, H, W).
    pose : Tensor of shape (B, L, T, 6) or (B, L, T, 3).
  '''
  B, L, T, pose_size = pose.shape
  C, H, W = images.size()[-3:]
  images = images.unsqueeze(1).repeat(1, L, 1, 1, 1, 1).view(B*L*T, C, H, W)
  if pose_size == 3:
      transformer_inv = expand_pose(
          pose_inv(pose.reshape(B*L*T, pose_size)))
  elif pose_size == 6:
      transformer_inv = pose_inv_full(
          pose.reshape(B*L*T, pose_size))
  grid = F.affine_grid(transformer_inv,
                       torch.Size((B*L*T, C, object_size, object_size)),
                       align_corners=False)
  obj = F.grid_sample(images, grid, align_corners=False)
  obj = obj.reshape(B, L, T, C, object_size, object_size)
  return obj


def object_to_image(objects, pose, image_size, visualize=False):
  '''Warp object to the target location in image.
  
  Args:
    images : Tensor of shape (B, L, T, C, H, W).
    pose : Tensor of shape (B, L, T, 6) or (B, L, T, 3).
  '''
  B, L, T, pose_size = pose.shape
  _, _, _, C, H, W = objects.size()
  objects = objects.reshape(B*L*T, C, H, W)

  if pose_size == 3:
    transformer = expand_pose(pose.reshape(B*L*T, pose_size))
  elif pose_size == 6:
    transformer = pose.reshape(B*L*T, 2, 3)
  grid = F.affine_grid(transformer,
                       torch.Size((B*L*T, C, image_size, image_size)),
                       align_corners=False)

  components = F.grid_sample(objects, grid, align_corners=False)
  components = components.reshape(B, L, T, C, image_size, image_size)
  if visualize:
    with torch.no_grad():
      center_dist = ((grid - torch.Tensor([0., 0.]).to(grid.device))**2).sum(-1)
      center_inds = center_dist.reshape(B, L, T, -1).argmin(-1)
      rows, cols = center_inds // image_size, center_inds % image_size
      for b in range(B):
        for l in range(L):
          for t in range(T):
            components[b, l, t, :, rows[b, l, :], cols[b, l, :]] = 1

  return components


def calculate_positions(pose):
  '''
  Get the center x, y of the spatial transformer.
  '''
  N, pose_size = pose.size()
  assert pose_size == 3, 'Only implemented pose_size == 3'
  # s, x, y
  s = pose[:, 0]
  xt = pose[:, 1]
  yt = pose[:, 2]
  x = (- xt / s + 1) / 2
  y = (- yt / s + 1) / 2
  return torch.stack([x, y], dim=1)


def bounding_box(z_where, x_size):
  """This doesn't take into account interpolation, but it's close
  enough to be usable."""
  s, x, y = z_where
  w = x_size / s
  h = x_size / s
  xtrans = -x / s * x_size / 2.
  ytrans = -y / s * x_size / 2.
  x = (x_size - w) / 2 + xtrans  # origin is top left
  y = (x_size - h) / 2 + ytrans
  return (x, y), w, h


def draw_components(images, pose):
  '''
  Draw bounding box for the given pose.
  images: size (N x C x H x W), range [0, 1]
  pose: N x 3
  '''
  images = (images.cpu().numpy() * 255).astype(np.uint8) # [0, 255]
  pose = pose.cpu().numpy()
  N, C, H, W = images.shape
  for i in range(N):
    if C == 1:
      img = images[i][0]
    else:
      img = images[i].transpose((1, 2, 0))
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    (x, y), w, h = bounding_box(pose[i], H)
    draw.rectangle([x, y, x + w, y + h], outline=128)
    new_img = np.array(img)
    new_img[0, ...] = 255  # Add line
    new_img[-1, ...] = 255  # Add line
    if C == 1:
      new_img = new_img[np.newaxis, :, :]
    else:
      new_img = new_img.transpose((2, 0, 1))
    images[i] = new_img

  # Back to torch tensor
  images = torch.FloatTensor(images / 255)
  return images
