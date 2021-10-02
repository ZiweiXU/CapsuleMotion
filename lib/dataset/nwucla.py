import pickle as pk
import numpy as np
import os.path as osp

from torch.utils.data import Dataset

ACTION_NAMES=[
    'pick up with one hand', 'pick up with two hand', 'drop trash', 
    'walk around', 'sit down', 'stand up', 'donning', 'doffing', 'throw', 
    'carry']

class NWUCLA(Dataset):
    def __init__(
        self, root='./data/', split='train',
        version='60', protocol='xsub', enable_val=False, debug=False,
        feat_precomputed=False, feat_suffix='_tpooled4', mmap_mode=None):
        
        super(NWUCLA, self).__init__()
        self.split = split
        self.version = version
        self.protocol = protocol
        self.feat_precomputed = feat_precomputed
        
        # train/val share the same collection of data
        split_ = 'train' if split in ['train', 'val'] else split
        self.data = np.load(
            osp.join(root, f'{split_}_data.npy'),
            mmap_mode=mmap_mode)
        if feat_precomputed:
            self.precomputed_feat = np.load(
                osp.join(root, f'{split_}_feat{feat_suffix}.npy'),
                mmap_mode=mmap_mode)
        self.label = pk.load(
            open(osp.join(root, f'{split_}_label.pkl'), 'rb'))
        self.num_classes = len(set(self.label))
        
        if split in ['train', 'val'] and enable_val:
            self.index = np.load(
                osp.join(root, f'{split}_idx.npy'))
        else:
            self.index = list(range(self.data.shape[0]))
        self.len = len(self.index)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        r_index = self.index[index]
        if self.feat_precomputed:
            ske_data = np.array(self.precomputed_feat[r_index, ...], dtype=np.single)
        else:
            ske_data = np.array(self.data[r_index, ...])
        ske_label = self.label[r_index]
        return ske_data, ske_label


if __name__ == '__main__':
    ds = NWUCLA(root='./data/nwucla/')
    data = ds.__getitem__(100)
    import ipdb; ipdb.set_trace()
