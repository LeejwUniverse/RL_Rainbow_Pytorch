import torch
import torch.nn as nn

class Q(nn.Module):
    def __init__(self, state_space, action_space):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_space,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        Q_values = self.fc3(x)
        
        return Q_values