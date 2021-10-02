import pickle as pk
import numpy as np
# import os
import os.path as osp

# import torch
# from torch import Tensor, nn
# from torch.nn import functional as F
from torch.utils.data import Dataset
# from torchvision import datasets, transforms


JOINT_NAMES = [    
    'base of the spine', 'middle of the spine', 'neck', 'head', 'left shoulder',
    'left elbow', 'left wrist', 'left hand', 'right shoulder', 'right elbow',
    'right wrist', 'right hand', 'left hip', 'left knee', 'left ankle',
    'left foot', 'right hip', 'right knee', 'right ankle', 'right foot',
    'spine', 'tip of the left hand', 'left thumb', 'tip of the right hand',
    'right thumb'
]

ACTION_NAMES = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person’s stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person’s ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}


class NTURGBD(Dataset):
    def __init__(
        self, root='./data/', split='train',
        version='60', protocol='xsub', enable_val=False, debug=False,
        feat_precomputed=False, feat_suffix='_tpooled4', mmap_mode=None):
        
        super(NTURGBD, self).__init__()
        self.split = split
        self.version = version
        self.protocol = protocol
        self.feat_precomputed = feat_precomputed
        
        # train/val share the same collection of data
        split_ = 'train' if split in ['train', 'val'] else split
        self.data = np.load(
            osp.join(root, version, protocol, f'{split_}_data.npy'),
            mmap_mode=mmap_mode)
        if feat_precomputed:
            self.precomputed_feat = np.load(
                osp.join(root, version, protocol, f'{split_}_feat{feat_suffix}.npy'),
                mmap_mode=mmap_mode)
        self.label = pk.load(
            open(osp.join(root, version, protocol, f'{split_}_label.pkl'), 'rb'))
        self.num_classes = len(set(self.label[1]))
        
        if split in ['train', 'val'] and enable_val:
            self.index = np.load(
                osp.join(root, version, protocol, f'{split}_idx.npy'))
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
        ske_label = self.label[1][r_index]
        return ske_data, ske_label


if __name__ == '__main__':
    ds = NTURGBD(root='./data/nturgbd/processed/')
    data = ds.__getitem__(100)
    import ipdb; ipdb.set_trace()
