import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

import numpy as np
import time
from collections import deque
import random

import DuelingDQN_network
"""
학습 속도문제로 제외. 엄밀한 제어를 위해선 사용!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
torch.manual_seed(7777)
torch.cuda.manual_seed(7777)
torch.cuda.manual_seed_all(7777) # if use multi-GPU
np.random.seed(7777)
random.seed(7777)

class double_dqn():
    def __init__(self, state_space, action_space, multi_step):
        self.Q_net = DuelingDQN_network.Q(state_space, action_space) ## behavior network.
        self.Q_target_net = DuelingDQN_network.Q(state_space, action_space) ## target network.
        self.Q_target_net.load_state_dict(self.Q_net.state_dict()) ## 초기에 값은 weight로 초기화 하고 시작함. Q_net의 파라미터를 복사함.

        self.learning_rate = 0.0005
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=self.learning_rate) ## optimizer 아담 사용.

        self.action_space = action_space
        self.gamma = 0.99 ## discount reward를 위한 감마.
        self.tau = 0.001 ## soft target update에 사용되는 타우.
        
        self.multi_step = multi_step
    
    def action_policy(self, state):
        Q_values = self.Q_net(state)
        return Q_values.argmax().item()
        
    def soft_update(self): ## DDPG에서 사용된 soft target update 방식.
        for param, target_param in zip(self.Q_net.parameters(), self.Q_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, random_mini_batch):
        
        # data 분배
        mini_batch = random_mini_batch ## 그냥 이름 줄인용.
        
        obs = np.vstack(mini_batch[:, 0]) ## 1-step TD와 Multi-step TD 모두 obs와 actions은 공통으로 사용됨.
        actions = list(mini_batch[:, 1]) 
        rewards = list(mini_batch[:, 2])
        next_obs = np.vstack(mini_batch[:, 3])
        masks = list(mini_batch[:, 4])
        
        # tensor.
        obs = torch.Tensor(obs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.Tensor(rewards) 
        next_obs = torch.Tensor(next_obs)
        masks = torch.Tensor(masks)
        
        # get Q-value
        Q_values = self.Q_net(obs) ## 계속 학습중인 Q NN에서 예상되는 action의 q_value를 얻어온다.
        q_values = Q_values.gather(1, actions).view(-1) ## 각 obs별로 실제 선택된 action들의 q value를 얻어온다. view 해준 이유는 shape을 맞추기 위해.
        
        # get target
        Q_target_values = self.Q_target_net(next_obs)
        next_q_action = self.Q_net(next_obs).max(1)[1].unsqueeze(1)## 실제 발생된 next_state를 넣어, 가장 큰 Q value를 가진 action을 선택한다.
        target_q_value = Q_target_values.gather(1,next_q_action).view(-1)## max함수를 사용하면 [0]에는 value 값이들어 가 있고 [1]에는 index값이 들어가 있다.
        Y = rewards + masks * self.gamma * target_q_value ## 죽었다면, next가 없으므로 얻어진 reward만 추린다.
        
        # loss 정의 
        MSE = torch.nn.MSELoss() ## mean squear error 사용.
        loss = MSE(q_values, Y.detach()) ## target은 단순히 주기적으로 업데이트해 네트워크를 유지시키므로, parameter가 미분되선 안된다. 그래서 detach() 해줌.
        
        # backward 시작!
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # soft target update
        self.soft_update()        