
import numpy as np
from os import path as osp

from torch.utils.data import Dataset

from .t20_traj_generator import t20_traj as t20_traj_generator


class T20_Traj(Dataset):
    """A wrapper of tMNIST_traj which provides options for training."""
    def __init__(self,
        root='data', split='train', transform=None, target_transform=None,
        debug=False, seq_len=32, canvas_size=128, rescale_factor=None,
        obj_type=None, visualize=False, ds_len=10000, preload=False, output_mode='coors'):
        super(T20_Traj, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.debug = debug
        self.seq_len = seq_len
        self.canvas_size = canvas_size
        self.rescale_factor = rescale_factor
        self.obj_type = obj_type
        self.preload = preload
        self.output_mode = output_mode

        self.len = ds_len

        # 'X': [?, 32, 128, 128], 'y' [?], 'count' [()]
        if split in ['train', 'val']:
            bank_size = 16 if self.debug else 32
            self.data = t20_traj_generator(
                bank_size=bank_size, dataset_size=self.len,
                video_size=self.canvas_size, seq_len=seq_len, device='cpu')
        elif split in ['test']:
            if output_mode == 'coors':
                self.data = np.load(osp.join(root, str(seq_len), 'val_x.npy'))
                self.label = np.load(osp.join(root, str(seq_len), 'val_y.npy'))
            else:
                raise NotImplementedError

    def __len__(self):
        return self.len
    
    def _normalize(self, x):
        frame_center = np.array(
            [self.canvas_size/2, self.canvas_size/2])
        return (x - frame_center) / frame_center

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, dtarget)
                where target is index of the trajectory class and dtarget the
                index of digit class.
        """
        if self.output_mode == 'coors':
            if self.split in ['train', 'val']:
                traj, target = self.data[index]
                traj = self._normalize(traj)
                traj = np.array(traj, dtype=np.single)
                target = target.long()
            elif self.split in ['test']:
                traj = np.array(self.data[index], dtype=np.single)
                target = self.label[index]

            traj = traj.T[..., None, None]  # compatible with (B,d,T,nB,nJ) format
            return traj, target
        else:
            raise NotImplementedError
