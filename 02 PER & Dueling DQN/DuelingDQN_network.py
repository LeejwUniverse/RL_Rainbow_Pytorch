import torch
import torch.nn as nn

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
