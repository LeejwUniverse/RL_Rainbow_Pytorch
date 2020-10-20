import torch
import torch.nn as nn

"""
학습 속도문제로 제외. 엄밀한 제어를 위해선 사용!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
torch.manual_seed(7777)
torch.cuda.manual_seed(7777)
torch.cuda.manual_seed_all(7777) # if use multi-GPU

class Q(nn.Module):
    def __init__(self, state_space, action_space):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_space,64)
        self.fc2 = nn.Linear(64,32)
        self.fc_v = nn.Linear(32,1)
        self.fc_a = nn.Linear(32,action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        V_values = self.fc_v(x)
        Advantages = self.fc_a(x)

        Q_values = V_values + (Advantages - Advantages.mean(dim=-1,keepdim=True))
        return Q_values
