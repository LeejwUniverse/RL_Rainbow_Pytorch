import torch
import torch.nn as nn
import torch.nn.functional as F
"""
학습 속도문제로 제외. 엄밀한 제어를 위해선 사용!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
torch.manual_seed(7777)
torch.cuda.manual_seed(7777)
torch.cuda.manual_seed_all(7777) # if use multi-GPU
import time
class Q(nn.Module):
    def __init__(self, state_space, action_space, atom_size, vmin, vmax):
        super(Q, self).__init__()
        self.action_space = action_space
        self.atom_size = atom_size
        self.vmin = vmin
        self.vmax = vmax
        self.delta_z = (vmax - vmin) / (atom_size - 1)

        self.fc1 = nn.Linear(state_space,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_space * atom_size)
        self.register_buffer("z", torch.arange(self.vmin, self.vmax + self.delta_z, self.delta_z))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        atoms = self.fc3(x).view(-1, self.action_space, self.atom_size)
        p_dist = F.softmax(atoms, dim=-1)
        Q_values = torch.sum(self.z * p_dist, dim=2)

        return Q_values, p_dist