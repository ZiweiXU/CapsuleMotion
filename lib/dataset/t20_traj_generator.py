import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.distributions.uniform import Uniform
import math
from PIL import Image
import numpy as np
import warnings

from tqdm import tqdm
import time
import os
import pickle


def plot(traj, name):
    save_path = os.path.join('outputs/v3/trajs', name + '.png')
    x = traj[0,:,0].cpu()
    y = traj[0,:,1].cpu()
    plt.plot(x,y)
    plt.axis('scaled')
    plt.savefig(save_path)
    plt.close('all')
    pass

class trajectory_generator(object):
    """
    trajectory generator
    """

    def __init__(self, traj_min, traj_max, max_skewness, seq_len, batch_size, device):
        """
        Args:
            traj_min (int): minimum value the trajectory can take (for both x and y axis)
            traj_max (int): maximum value the trajectory can take (for both x and y axis)
            max_skewness (float): control the skewness of the trajectories
                        actual skewness will be sampled from (1, max_skewness)
                        e.g., the trajectory template takes 50*50 pixels
                        0.8 skewness means that height and width are 50/0.8 and 50*0.8 respectively,
                        or height is 50*0.8 and width is 50/0.8, which happens at random
            seq_len (int): we use 32
        Returns:
            trajectory of shape [seq_len, 2], coordinates are within [traj_min, traj_max]
        """
        print("initialize Dataloader...")

        # we consider 20 trajectory patterns
        self.trajectories = ['triangle', 'rectangle', 'pentagon', 'hexagon', 'astroid',
                            'circle', 'hippopede', 'lemniscate', 'heart', 'spiral',
                            'sine', 'abs_sine', 'tanh', 'line', 'parabola',
                            'bell', 'semicubical_parabola', 'cubic_1', 'cubic_2', 'cubic_3']
        # among 20 traj patterns, 11 are open trajectories
        self.open_trajs = ['spiral',
                            'sine', 'abs_sine', 'tanh', 'line', 'parabola',
                            'bell', 'semicubical_parabola', 'cubic_1', 'cubic_2', 'cubic_3']
        
        self.traj_min = traj_min
        self.traj_max = traj_max
        self.max_skewness = max_skewness
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = torch.device(device)

        # dummpy variables for internal processing
        self.N = seq_len * 20 # number of interpolating points
        self.idxs = torch.arange(0,seq_len, device=device).float().unsqueeze(axis=-1)
        self.idxs_diag = torch.tensor([[[i,0],[0,i]] for i in range(seq_len)], device=device).float()
        self.angles = torch.arange(0, 2*math.pi, 0.02*math.pi, device=device).float()
        self.idxs_diag_N = torch.tensor([[[i,0],[0,i]] for i in range(self.N)], device=device).float()

        self.init_templates() # initialize the trajectory templates
        print("initialized!")
    
    def init_templates(self):
        """
        initialize the trajectory templates
        trajectory templates are hand-crafted
        each template is of shape [1, self.N, 2]
        save to self.templates as a dictionary
        """
        self.templates = {}

        # triangle template
        triangle = torch.empty([1, 3 * self.N, 2], device=self.device)
        triangle[:, :self.N, 0] = torch.linspace(0, 50, self.N, device=self.device)
        triangle[:, :self.N, 1] = 0.
        triangle[:, self.N:2*self.N, 0] = torch.linspace(50, 25, self.N, device=self.device)
        triangle[:, self.N:2*self.N, 1] = torch.linspace(0, 50, self.N, device=self.device)
        triangle[:, 2*self.N:, 0] = torch.linspace(25, 0, self.N, device=self.device)
        triangle[:, 2*self.N:, 1] = torch.linspace(50, 0, self.N, device=self.device)
        triangle = self.equal_arc_len_split(triangle, self.N)
        self.templates['triangle'] = triangle

        # rectangle template
        rectangle = torch.empty([1, 4 * self.N, 2], device=self.device)
        rectangle[:, :self.N, 0] = torch.linspace(0, 50, self.N, device=self.device)
        rectangle[:, :self.N, 1] = 0.
        rectangle[:, self.N:2*self.N,0] = 50.
        rectangle[:, self.N:2*self.N,1] = torch.linspace(0, 50, self.N, device=self.device)
        rectangle[:, 2*self.N:3*self.N,0] = torch.linspace(50, 0, self.N, device=self.device)
        rectangle[:, 2*self.N:3*self.N,1] = 50.
        rectangle[:, 3*self.N:,0] = 0.
        rectangle[:, 3*self.N:,1] = torch.linspace(50, 0, self.N, device=self.device)
        rectangle = self.equal_arc_len_split(rectangle, self.N)
        self.templates['rectangle'] = rectangle

        # pentagon template
        # r = 27.63
        # [0, 0], [32.48, 0], [42.51, 30.89], [16.24, 49.98], [-10.03, 30.89]
        pentagon = torch.empty([1, 5 * self.N, 2], device=self.device)
        pentagon[:, :self.N, 0] = torch.linspace(0, 32.48, self.N, device=self.device)
        pentagon[:, :self.N, 1] = 0.
        pentagon[:, self.N:2*self.N, 0] = torch.linspace(32.48, 42.51, self.N, device=self.device)
        pentagon[:, self.N:2*self.N, 1] = torch.linspace(0, 30.89, self.N, device=self.device)
        pentagon[:, 2*self.N:3*self.N, 0] = torch.linspace(42.51, 16.24, self.N, device=self.device)
        pentagon[:, 2*self.N:3*self.N, 1] = torch.linspace(30.89, 49.98, self.N, device=self.device)
        pentagon[:, 3*self.N:4*self.N, 0] = torch.linspace(16.24, -10.03, self.N, device=self.device)
        pentagon[:, 3*self.N:4*self.N, 1] = torch.linspace(49.98, 30.89, self.N, device=self.device)
        pentagon[:, 4*self.N:, 0] = torch.linspace(-10.03, 0, self.N, device=self.device)
        pentagon[:, 4*self.N:, 1] = torch.linspace(30.89, 0, self.N, device=self.device)
        pentagon = self.equal_arc_len_split(pentagon, self.N)
        self.templates['pentagon'] = pentagon

        # hexagon template
        # r = 25
        # [0, 0], [25, 0], [37.5, 21.65], [25, 43.3], [0, 43.3], [-12.5, 21.65]
        hexagon = torch.empty([1, 6 * self.N, 2], device=self.device)
        hexagon[:, :self.N, 0] = torch.linspace(0, 25, self.N, device=self.device)
        hexagon[:, :self.N, 1] = 0.
        hexagon[:, self.N:2*self.N, 0] = torch.linspace(25, 37.5, self.N, device=self.device)
        hexagon[:, self.N:2*self.N, 1] = torch.linspace(0, 21.65, self.N, device=self.device)
        hexagon[:, 2*self.N:3*self.N, 0] = torch.linspace(37.5, 25, self.N, device=self.device)
        hexagon[:, 2*self.N:3*self.N, 1] = torch.linspace(21.65, 43.3, self.N, device=self.device)
        hexagon[:, 3*self.N:4*self.N, 0] = torch.linspace(25, 0, self.N, device=self.device)
        hexagon[:, 3*self.N:4*self.N, 1] = 43.3
        hexagon[:, 4*self.N:5*self.N, 0] = torch.linspace(0, -12.5, self.N, device=self.device)
        hexagon[:, 4*self.N:5*self.N, 1] = torch.linspace(43.3, 21.65, self.N, device=self.device)
        hexagon[:, 5*self.N:, 0] = torch.linspace(-12.5, 0, self.N, device=self.device)
        hexagon[:, 5*self.N:, 1] = torch.linspace(21.65, 0, self.N, device=self.device)
        hexagon = self.equal_arc_len_split(hexagon, self.N)
        self.templates['hexagon'] = hexagon

        # astroid template
        # x = 25 * cos(t)**3
        # y = 25 * sin(t)**3
        # t in [0,2pi]
        ts = torch.linspace(0, 2*math.pi, self.N, device=self.device)
        astroid = torch.empty([1, self.N, 2], device=self.device)
        astroid[:,:,0] = 25 * torch.cos(ts)**3
        astroid[:,:,1] = 25 * torch.sin(ts)**3
        self.templates['astroid'] = astroid

        # circle template
        # x = 25 * sin(t)
        # y = 25 * cos(t)
        # t in [0,2pi]
        ts = torch.linspace(0, 2*math.pi, self.N, device=self.device)
        circle = torch.empty([1, self.N, 2], device=self.device)
        circle[:,:,0] = 25 * torch.sin(ts)
        circle[:,:,1] = 25 * torch.cos(ts)
        self.templates['circle'] = circle

        # hippopede template
        # r**2 = 4 * 9.5 * (10-9.5*sin(t)**2)
        # x = a * sqrt(4 * 9.5 * (10-9.5*sin(t)**2)) * cos(t), a = 1.25
        # y = b * sqrt(4 * 9.5 * (10-9.5*sin(t)**2)) * sin(t), b = 2.5
        # t in [0, 2pi]
        ts = torch.linspace(0, 2*math.pi, self.N, device=self.device)
        hippopede = torch.empty([1, self.N, 2], device=self.device)
        hippopede[:,:,0] = 1.25 * torch.sqrt(4 * 9.5 * (10 - 9.5 * torch.sin(ts)**2)) * torch.cos(ts)
        hippopede[:,:,1] = 2.5 * torch.sqrt(4 * 9.5 * (10 - 9.5 * torch.sin(ts)**2)) * torch.sin(ts)
        self.templates['hippopede'] = hippopede

        # lemniscate template
        # x = 25 * sin(t)
        # y = 50 * sin(t) * cos(t)
        # t in [0, 2pi]
        ts = torch.linspace(0, 2*math.pi, self.N, device=self.device)
        lemniscate = torch.empty([1, self.N, 2], device=self.device)
        lemniscate[:,:,0] = 25 * torch.sin(ts)
        lemniscate[:,:,1] = 50 * torch.sin(ts) * torch.cos(ts)
        self.templates['lemniscate'] = lemniscate

        # heart template
        # x = a * sin(t)**3, a = 25
        # y = b * [13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)], b = 1.7287
        # t in [0, 2pi]
        ts = torch.linspace(0, 2*math.pi, self.N, device=self.device)
        heart = torch.empty([1, self.N, 2], device=self.device)
        heart[:,:,0] = 25 * torch.sin(ts)**3
        heart[:,:,1] = 1.7287 * (13*torch.cos(ts) - 5*torch.cos(2*ts) - 2*torch.cos(3*ts) - torch.cos(4*ts))
        self.templates['heart'] = heart

        # Archimedes' Spiral template
        # x = (25-1.5t)*cos(t)
        # y = (25-1.5t)*sin(t)
        # t in [0,4pi]
        ts = torch.linspace(0, 4*math.pi, self.N, device=self.device)
        spiral = torch.empty([1, self.N, 2], device=self.device)
        spiral[:,:,0] = (25-1.5*ts) * torch.cos(ts)
        spiral[:,:,1] = (25-1.5*ts) * torch.sin(ts)
        self.templates['spiral'] = spiral

        # sine template
        # x = 7.9577 * t
        # y = 25 * sin(t)
        # t in [0, 2pi]
        ts = torch.linspace(0, 2*math.pi, self.N, device=self.device)
        sine = torch.empty([1, self.N, 2], device=self.device)
        sine[:,:,0] = 7.9677 * ts
        sine[:,:,1] = 25 * torch.sin(ts)
        self.templates['sine'] = sine

        # abs sine template
        # x = 7.9577 * t
        # y = 50 * abs(sin(t))
        # t in [0, 2pi]
        ts = torch.linspace(0, 2*math.pi, self.N, device=self.device)
        abs_sine = torch.empty([1, self.N, 2], device=self.device)
        abs_sine[:,:,0] = 7.9677 * ts
        abs_sine[:,:,1] = 50 * torch.abs(torch.sin(ts))
        self.templates['abs_sine'] = abs_sine

        # tanh template
        # x = 6.25 * t
        # y =25 * tanh(t)
        # t in [-4,4]
        ts = torch.linspace(-4, 4, self.N, device=self.device)
        tanh = torch.empty([1, self.N, 2], device=self.device)
        tanh[:,:,0] = 6.25 * ts
        tanh[:,:,1] = 25 * torch.tanh(ts)
        self.templates['tanh'] = tanh

        # line template
        # x = t
        # y = t
        # t in [0, 50]
        line = torch.arange(0, self.N, device=self.device).float().unsqueeze(axis=-1).unsqueeze(0).repeat(1,1,2)
        line = (line / (self.N - 1) * 50)
        self.templates['line'] = line

        # parabola template
        # x = a * t, a = 25
        # y = b * t**2, b = 50
        # t in [-1, 1]
        ts = torch.linspace(-1, 1, self.N, device=self.device)
        parabola = torch.empty([1, self.N, 2], device=self.device)
        parabola[:, :, 0] = 25 * ts
        parabola[:, :, 1] = 50 * ts**2
        self.templates['parabola'] = parabola

        # bell template
        # x**3 - y**2 + 2 = 0
        # No.7 from http://www.milefoot.com/math/planecurves/cubics.htm
        # x = a * t, a = 17.4830
        # y = +/- b * sqrt(t**3 + 2), b = 10.1255
        # t in [1.6, -1.25992], for negative y
        # t in [-1.25992, 1.6], for positive y
        ts_1 = torch.linspace(1.6, -1.25992, self.N, device=self.device)
        ts_2 = torch.linspace(-1.25992, 1.6, self.N, device=self.device)
        bell = torch.empty([1, 2*self.N, 2], device=self.device)
        bell[:,:self.N,0] = 17.4830 * ts_1
        bell[:,:self.N,1] = 10.1255 * torch.sqrt(ts_1**3 + 2)
        bell[:,self.N:,0] = 17.4830 * ts_2
        bell[:,self.N:,1] = -10.1255 * torch.sqrt(ts_2**3 + 2)
        bell = self.equal_arc_len_split(bell, self.N)
        self.templates['bell'] = bell

        # semicubical parabola template
        # x**3 - y**2 = 0
        # No.6 from http://www.milefoot.com/math/planecurves/cubics.htm
        # x = a * t**2, a = 50
        # y = b * t**3, b = 25
        # i in [-1, 1]
        ts = torch.linspace(-1, 1, self.N, device=self.device)
        semicubical_parabola = torch.empty([1, self.N, 2], device=self.device)
        semicubical_parabola[:, :, 0] = 50 * ts**2
        semicubical_parabola[:, :, 1] = 25 * ts**3
        self.templates['semicubical_parabola'] = semicubical_parabola

        # cubic_1 template
        # x**3 - y**2 -3*x +2 = 0
        # No.8 from http://www.milefoot.com/math/planecurves/cubics.htm
        # x = a * t, a = 12.5
        # y = +/- b * sqrt(t**3 - 3*t + 2), b = 12.5
        # t in [2, 1] for positive y
        # t in [1, -2] for negative y
        # t in [-2, 1] for positive y
        # t in [1, 2] for negative y
        ts_1 = torch.linspace(2, 1, self.N, device=self.device)
        ts_2 = torch.linspace(1, -2, self.N, device=self.device)
        ts_3 = torch.linspace(-2, 1, self.N, device=self.device)
        ts_4 = torch.linspace(1, 2, self.N, device=self.device)
        cubic_1 = torch.empty([1, 4*self.N, 2], device=self.device)
        cubic_1[:, :self.N, 0] = 12.5 * ts_1
        cubic_1[:, :self.N, 1] = 12.5 * torch.sqrt(ts_1**3 - 3 * ts_1 + 2)
        cubic_1[:, self.N:2*self.N, 0] = 12.5 * ts_2
        cubic_1[:, self.N:2*self.N, 1] = -12.5 * torch.sqrt(ts_2**3 - 3 * ts_2 + 2)
        cubic_1[:, 2*self.N:3*self.N, 0] = 12.5 * ts_3
        cubic_1[:, 2*self.N:3*self.N, 1] = 12.5 * torch.sqrt(ts_3**3 - 3 * ts_3 + 2)
        cubic_1[:, 3*self.N:4*self.N, 0] = 12.5 * ts_4
        cubic_1[:, 3*self.N:4*self.N, 1] = -12.5 * torch.sqrt(ts_4**3 - 3 * ts_4 + 2)
        cubic_1 = self.equal_arc_len_split(cubic_1, self.N)
        self.templates['cubic_1'] = cubic_1


        # cubic_2 template
        # No.16 of http://www.milefoot.com/math/planecurves/cubics.htm
        # x = a * (-t**2 + 6*t-1)/(t**2+1), a = 8.3333
        # y = b * t, b = 2.0833
        # t in [-12,12]
        ts = torch.linspace(-12, 12, self.N, device=self.device)
        cubic_2 = torch.empty([1, self.N, 2], device=self.device)
        cubic_2[:,:,0] = 8.3333 * (-ts**2 + 6 * ts - 1) / (ts**2 + 1)
        cubic_2[:,:,1] = 2.0833 * ts
        self.templates['cubic_2'] = cubic_2

        # cubic_3 template
        # No.17 of http://www.milefoot.com/math/planecurves/cubics.htm
        # x = a * t, a = 10
        # Eq.1: y = b * (2 - sqrt(-t**4 + t**3 + 6*t**2)) / (t + 1), b = 7.1429
        # Eq.2: y = b * (sqrt(-t**4 + t**3 + 6*t**2) + 2) / (t + 1), b = 7.1429
        # t in [-1.9595, -2] Eq.2
        # t in [-2, 0] Eq.1
        # t in [0, 3] Eq.2
        # t in [3, 0] Eq.1
        # t in [0, -0.31428] Eq.2
        ts_1 = torch.linspace(-1.9595, -2, self.N, device=self.device)
        ts_2 = torch.linspace(-2, 0, self.N, device=self.device)
        ts_3 = torch.linspace(0, 3, self.N, device=self.device)
        ts_4 = torch.linspace(3, 0, self.N, device=self.device)
        ts_5 = torch.linspace(0, -0.31428, self.N, device=self.device)
        cubic_3 = torch.empty([1, 5*self.N, 2], device=self.device)
        cubic_3[:, :self.N, 0] = 10 * ts_1
        cubic_3[:, :self.N, 1] = 7.1429 * (torch.sqrt(-ts_1**4 + ts_1**3 + 6*ts_1**2) + 2) / (ts_1 + 1)
        cubic_3[:, self.N:2*self.N, 0] = 10 * ts_2
        cubic_3[:, self.N:2*self.N, 1] = 7.1429 * (2 - torch.sqrt(-ts_2**4 + ts_2**3 + 6*ts_2**2)) / (ts_2 + 1)
        cubic_3[:, 2*self.N:3*self.N, 0] = 10 * ts_3
        cubic_3[:, 2*self.N:3*self.N, 1] = 7.1429 * (torch.sqrt(-ts_3**4 + ts_3**3 + 6*ts_3**2) + 2) / (ts_3 + 1)
        cubic_3[:, 3*self.N:4*self.N, 0] = 10 * ts_4
        cubic_3[:, 3*self.N:4*self.N, 1] = 7.1429 * (2 - torch.sqrt(-ts_4**4 + ts_4**3 + 6*ts_4**2)) / (ts_4 + 1)
        cubic_3[:, 4*self.N:, 0] = 10 * ts_5
        cubic_3[:, 4*self.N:, 1] = 7.1429 * (torch.sqrt(-ts_5**4 + ts_5**3 + 6*ts_5**2) + 2) / (ts_5 + 1)
        cubic_3 = self.equal_arc_len_split(cubic_3, self.N)
        self.templates['cubic_3'] = cubic_3


    def rotate(self, origin, delta, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.

        Args:
            origin ([float, float]):
            delta ([float, float]): the relative coordinate to the origin
            angle (float):
        Returns:
            rotated point ([float, float]):
        """

        out = torch.empty(delta.shape, device=self.device)
        out[:,0] = origin[0] + math.cos(angle) * delta[:,0] - math.sin(angle) * delta[:,1]
        out[:,1] = origin[1] + math.sin(angle) * delta[:,0] + math.cos(angle) * delta[:,1]
        return out
    
    def rotate_trajs(self, origins, trajs, angles):
        """
        Rotate a batch of trajectories counterclockwise by a list of angles around a batch of origins.
        The angles should be given in radians.

        Args:
            origins: of shape [B, 2]
            trajs: of shape [B, seq_len, 2]
            angles: of shape [N]

        Returns:
            rotated trajectories: of shape [B, N, seq_len, 2]
        """
        B = origins.shape[0]
        N = angles.shape[0]

        out = torch.empty([B, N, self.seq_len, 2], device=self.device)
        out[:,:,:,0] = (origins[:,0].reshape([-1,1,1])
                + torch.cos(angles).unsqueeze(0).unsqueeze(-1).repeat(B,1,self.seq_len) 
                * trajs[:,:,0].unsqueeze(1).repeat(1,N,1)
                - torch.sin(angles).unsqueeze(0).unsqueeze(-1).repeat(B,1,self.seq_len) 
                * trajs[:,:,1].unsqueeze(1).repeat(1,N,1)
                )
        out[:,:,:,1] = (origins[:,1].reshape([-1,1,1])
                + torch.sin(angles).unsqueeze(0).unsqueeze(-1).repeat(B,1,self.seq_len) 
                * trajs[:,:,0].unsqueeze(1).repeat(1,N,1)
                + torch.cos(angles).unsqueeze(0).unsqueeze(-1).repeat(B,1,self.seq_len) 
                * trajs[:,:,1].unsqueeze(1).repeat(1,N,1)
                )
        return out
    
    def random_position(self, batch_size=1):
        """
        returns a random position within [traj_min, traj_max]
        """
        return torch.randint(self.traj_min, self.traj_max, (batch_size,2), device=self.device).float()
    
    def sample_valid_position_angle(self, curves):
        """
        sample a valid initial position and a valid angle
        Valid means the initial position and starting angle need to make sure that
        the trajectories are inside the [traj_min, traj_max].
        """
        N, L, _ = curves.shape

        init_points = self.random_position(N)
        curves = self.rotate_trajs(init_points, 1.1*curves, self.angles)

        temp = (curves >= self.traj_min).float() * (curves <= self.traj_max).float()
        valid_angles = torch.prod(temp.reshape([N, self.angles.shape[0], -1]), dim=-1).bool()
        indicator = valid_angles.max(dim=-1).values

        feasible_init_points = torch.empty([indicator.sum(), 2], device=self.device)
        feasible_angles = torch.empty([indicator.sum()], device=self.device)

        feasible_list = np.where(np.array(indicator.cpu())==True)[0]
        for i in range(len(feasible_list)):
            choices = np.where(np.array(valid_angles[feasible_list[i]].cpu())==True)[0]
            idx = np.random.choice(choices)
            feasible_init_points[i] = init_points[feasible_list[i]]
            feasible_angles[i] = self.angles[idx]
        
        return feasible_init_points, feasible_angles, indicator



    def bisection(self, curves, init_points, angles):
        """
        bisectional search to find the max enlargement coefficient
        """
        N = angles.shape[0]

        lower = torch.empty([N], device=self.device).fill_(1)
        upper = torch.empty([N], device=self.device).fill_(8)
        alpha = torch.empty([N], device=self.device).fill_(4)

        k = 0
        while k <= 8:
            k += 1
            for i in range(N):
                traj = self.rotate(init_points[i], alpha[i]*curves[i], angles[i])
                if (traj >= self.traj_min).all() and (traj <= self.traj_max).all():
                    lower[i] = alpha[i]
                else:
                    upper[i] = alpha[i]
                alpha[i] = (lower[i] + upper[i])/2
        return lower
        
    
    def equal_arc_len_split(self, curve_in, num_splits):
        """
        Split curve_in into equal-arc-length segments

        Args:
            curve_in: batch of trajectories, of shape [N, L, 2], 
                    N is batch size, 
                    L is the sequence length, must be much greater than num_splits
            num_splits: 

        Return:
            curve_out: batch of trajectories, of shape [N, num_splits, 2]
                    for each trajectory [num_splits, 2], the points split the curve into equal-arc-length segments
        """
        N, L, _ = curve_in.shape
        in_distance = (curve_in - torch.cat((curve_in[:,0].unsqueeze(1), curve_in),1)[:,:-1]).norm(dim=-1)
        for i in range(1,L):
            in_distance[:,i] = in_distance[:,i] + in_distance[:,i-1]
        arc_len = in_distance[:,-1]
        # in_distance is the approximated curve distance from current point to intial point
        # arc_len is the total curve length

        # compute target_len
        target_len = torch.empty([N, num_splits], device=self.device)
        for i in range(N):
            target_len[i] = torch.linspace(0, arc_len[i], num_splits)
        
        # curve_out is given by the points closest to target_len
        idxs = (in_distance.unsqueeze(1).repeat(1,num_splits,1) - target_len.unsqueeze(2)).abs().argmin(dim=-1)
        curve_out = torch.empty([N, num_splits, 2], device=self.device)
        for i in range(N):
            curve_out[i] = curve_in[i,idxs[i]]

        return curve_out
    

    def generate(self, traj):
        """
        generate trajectories in following steps:
            1. generate a curve
            2. sample initial position and initial angle (might fail due to bad init position)
            3. find the max enlargement coefficient
            4. sample a random enlargement coefficient
            5. transform curve with initial position and initial angle, and enlargement coefficient
        """

        if traj not in self.trajectories:
            raise AssertionError(traj + ' is not defined!')

        curves = self.templates[traj].repeat(self.batch_size, 1, 1)

        # random skewness
        skew = torch.empty([self.batch_size, 1], device=self.device).uniform_(1, self.max_skewness)
        rand_num = torch.empty([self.batch_size]).uniform_(0, 2)
        indicator = rand_num > 1
        curves[indicator, :, 0] = curves[indicator, :, 0] * skew[indicator]
        curves[indicator, :, 1] = curves[indicator, :, 1] / skew[indicator]
        indicator = rand_num <= 1
        curves[indicator, :, 0] = curves[indicator, :, 0] / skew[indicator]
        curves[indicator, :, 1] = curves[indicator, :, 1] * skew[indicator]

        # if open trajectories
        # introduce some randomness by:
        #       1. start randomly from either ends
        #       2. start and end points are randomly selected from 5% of points at the ends
        if traj in self.open_trajs:
            indicator = torch.empty([self.batch_size]).uniform_(0, 2) > 1
            start_points = torch.randint(0, int(0.05*self.N), (self.batch_size,))
            end_points = torch.randint(int(0.95*self.N), self.N, (self.batch_size,))
            for i in range(self.batch_size):
                if indicator[i].item() is True:
                    curves[i] = torch.flip(curves[i], (0,))
                curves[i, :start_points[i], :] = curves[i, start_points[i]]
                curves[i, end_points[i]:, :] = curves[i, end_points[i]]
        else:
        # if closed trajectories
        # start from a random point
        # randomly clock or counter-clockwise
            indicator = torch.empty([self.batch_size]).uniform_(0, 2) > 1
            start_points = torch.randint(0, self.N, (self.batch_size,))
            for i in range(self.batch_size):
                if indicator[i].item() is True:
                    curves[i] = torch.flip(curves[i], (0,))
                temp = curves[i].clone()
                curves[i, :(self.N-start_points[i]), :] = curves[i, start_points[i]:, :].clone()
                curves[i, (self.N-start_points[i]):, :] = temp[:start_points[i]]
        
        curves = self.equal_arc_len_split(curves, self.seq_len)

        # deduct the coordinate of the starting points so start at the origin
        for i in range(self.batch_size):
            curves[i] = curves[i] - curves[i,0,:].unsqueeze(0)

        # sample valid points and angles
        # sampling might fail for some curves, due to bad init points
        # indicator is True if a valid init_point and angle pair is generated, False otherwise
        init_points, angles, indicator = self.sample_valid_position_angle(curves)
        # discard unsuccessful curves
        curves = curves[indicator]

        # find the maximum enlargement coefficient
        coef_max = self.bisection(curves, init_points, angles)
        # sample coefs within the range
        coef = torch.tensor([ np.random.uniform(1, maximum) for maximum in coef_max.tolist()])

        # enlarge and rotate trajectories
        trajectories = torch.empty(curves.shape, device=self.device)
        for i in range(curves.shape[0]):
            trajectories[i] = self.rotate(init_points[i], coef[i] * curves[i], angles[i])

        return trajectories


class t20_traj(torch.utils.data.DataLoader):
    def __init__(self, bank_size, dataset_size, device, max_skewness=1.4, seq_len=32,
                        video_size=128, out_of_view_margin=0):
        """
        generate training data on-the-fly

        Args:
            bank_size: the DataLoader maintains a bank of of trajectories and returns one each time.
                        When the bank is used up, the DataLoader samples bank_size of new trajectories.
                        This for more efficiently generating trajectories.
                        It is recommended to set bank_size as a multiple of batch size.
                        The actual bank size is bank_size * 20 (number of trajectory patterns).
            dataset_size: only for the __len__ function, data is generated on-the-fly
            device: the generator and the generated data will be on the device
            max_skewness (float): control the skewness of the trajectories
                        actual skewness will be sampled from (1, max_skewness)
                        e.g., original templates are 50*50 pixels
                        0.8 skewness means that height and width are 50/0.8 and 50*0.8 respectively,
                        or height is 50*0.8 and width is 50/0.8, which happens at random
            seq_len: length of the generated video
            video_size:
            out_of_view_margin:
        
        """

        self.trajectories = ['triangle', 'rectangle', 'pentagon', 'hexagon', 'astroid',
                            'circle', 'hippopede', 'lemniscate', 'heart', 'spiral',
                            'sine', 'abs_sine', 'tanh', 'line', 'parabola',
                            'bell', 'semicubical_parabola', 'cubic_1', 'cubic_2', 'cubic_3']
        self.N_traj = len(self.trajectories)

        self.bank_size = bank_size
        self.dataset_size = dataset_size
        self.seq_len = seq_len
        self.video_size = video_size
        self.device = device
    
        self.trajectory_G = trajectory_generator(-out_of_view_margin, 
                                                video_size+out_of_view_margin, 
                                                max_skewness,
                                                seq_len, 
                                                bank_size, 
                                                device)

        self.sample_bank()
    
    def __len__(self):
        return self.dataset_size
    
    def sample_bank(self):
        # self.bank_idx counts how many trajectories from self.traj_bank have been returned
        # reset to zero every time
        self.bank_idx = 0

        # This part is hacky but works fine
        self.traj_labels = torch.empty([self.bank_size * self.N_traj], device=self.device)
        self.traj_bank = torch.empty([self.bank_size * self.N_traj, self.seq_len, 2], device=self.device)
        for i in range(self.N_traj):
            traj_name = self.trajectories[i]
            trajs = self.trajectory_G.generate(traj_name)
            if trajs.shape[0] < self.bank_size:
                N = self.bank_size - trajs.shape[0]
                temp = self.trajectory_G.generate(traj_name)
                if temp.shape[0] >= N:
                    trajs = torch.cat((trajs, temp[:N]), 0)
                else:
                    trajs = torch.cat((trajs, temp), 0)
                    N = self.bank_size - trajs.shape[0]
                    temp = self.trajectory_G.generate(traj_name)
                    if temp.shape[0] >= N:
                        trajs = torch.cat((trajs, temp[:N]), 0)
                        warnings.warn('sampling {} failure rate > 0.66! It is not efficient.'.format(traj_name))
                    else:
                        raise ValueError('sampling {} failure rate > 0.75! It is not efficient. \
                                            Please optimizing the code for sampling this trajectory.'.format(traj_name))
            
            self.traj_bank[i*self.bank_size:(i+1)*self.bank_size] = trajs
            self.traj_labels[i*self.bank_size:(i+1)*self.bank_size] = i

        # shuffle
        idx = np.arange(self.bank_size * self.N_traj)
        np.random.shuffle(idx)
        self.traj_labels = self.traj_labels[idx]
        self.traj_bank = self.traj_bank[idx]
    
    def __getitem__(self, idx):
        
        # retrive trajectories from traj_bank
        # when used up, sample a new traj_bank

        if self.bank_idx >= self.bank_size * self.N_traj:
            self.sample_bank()

        traj = self.traj_bank[self.bank_idx]
        traj_label  = self.traj_labels[self.bank_idx]

        self.bank_idx += 1

        return traj, traj_label

if __name__ == '__main__':

    dataset = t20_traj(
        bank_size=1024, 
        dataset_size=99999, 
        video_size=128, # generate trajectories whose x and y coordinates are within [0,video_size]
        device='cpu'
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256)

    for idx, (trajs, traj_labels) in enumerate(dataloader):
        # trajs are of shape [batch size, sequence length, 2]
        # traj_labels [batch size]
        pass
        import ipdb; ipdb.set_trace()
    pass
        