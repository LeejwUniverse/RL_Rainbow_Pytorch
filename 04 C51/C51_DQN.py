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

import C51_network
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

class c51_dqn():
    def __init__(self, state_space, action_space, multi_step, batch_size, atom_size, vmin, vmax):
        self.Q_net = C51_network.Q(state_space, action_space, atom_size, vmin, vmax) ## behavior network.
        self.Q_target_net = C51_network.Q(state_space, action_space, atom_size, vmin, vmax) ## target network.
        self.Q_target_net.load_state_dict(self.Q_net.state_dict()) ## 초기에 값은 weight로 초기화 하고 시작함. Q_net의 파라미터를 복사함.

        self.learning_rate = 0.0005
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=self.learning_rate) ## optimizer 아담 사용.

        self.action_space = action_space
        self.gamma = 0.99 ## discount reward를 위한 감마.
        self.tau = 0.001 ## soft target update에 사용되는 타우.
        self.epsilon = 1 ## 초기 epsilon 1부터 시작.
        self.epsilon_decay = 0.00001 ## epsilon 감쇠 값.
        
        self.multi_step = multi_step
        self.batch_size = batch_size
        self.atom_size = atom_size
        self.vmin = vmin
        self.vmax = vmax
        self.delta_z = (vmax - vmin) / (atom_size - 1)
        self.z = torch.arange(self.vmin, self.vmax + self.delta_z, self.delta_z)
     
    def print_eps(self): ## epsilon 값 출력을 위한 함수.
        return self.epsilon

    def action_policy(self, state):
        Q_values, _ = self.Q_net(state)
        if np.random.rand() <= self.epsilon: ## 0~1의 균일분포 표준정규분포 난수를 생성. 정해준 epsilon 값보다 작은 random 값이 나오면 
            action = random.randrange(self.action_space) ## action을 random하게 선택합니다.
            return action
        else: ## epsilon 값보다 크면, 학습된 Q_player NN 에서 얻어진 Q_values 값중 가장 큰 action을 선택합니다.
            return Q_values.argmax().item()
    
    def soft_update(self): ## DDPG에서 사용된 soft target update 방식.
        for param, target_param in zip(self.Q_net.parameters(), self.Q_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, random_mini_batch):
        self.epsilon -= self.epsilon_decay ## 학습해 감에 따라 epilon에 의존하기보단 학습된 정책에 의존되게.
        self.epsilon = max(self.epsilon, 0.05) ## 그래도 가끔씩 새로운 exploration을 위해 최소 0.05은 주기.
        
        # data 분배
        mini_batch = random_mini_batch ## 그냥 이름 줄인용.
        
        obs = np.vstack(mini_batch[:, 0]) ## 1-step TD와 Multi-step TD 모두 obs와 actions은 공통으로 사용됨.
        actions = list(mini_batch[:, 1]) 
        rewards = list(mini_batch[:, 2])
        next_obs = np.vstack(mini_batch[:, 3])
        masks = list(mini_batch[:, 4])
        
        # tensor.
        obs = torch.Tensor(obs)
        actions = torch.LongTensor(actions)
        rewards = torch.Tensor(rewards).unsqueeze(-1)
        next_obs = torch.Tensor(next_obs)
        masks = torch.Tensor(masks).unsqueeze(-1)

        # get p_distribution.
        _, p_dist = self.Q_net(obs) ## observation에 해당하는 p_distribution을 얻는다.

        log_p_dist = torch.log(p_dist[range(self.batch_size), actions]) ## 실제 선택된 action의 p_distribution만 선택하고 log 취한다.
    
        with torch.no_grad():            
        # get target q_distribution.
            Q_target_values, q_dist = self.Q_target_net(next_obs) ## next_obervation에 해당하는 q_distribution을 얻는다.
            next_actions = Q_target_values.max(1)[1] ## 가장 큰 target Q value를 가진 next_action을 선택한다. (위와 다른게 next는 수집해 저장하지 않아서.)
            q_dist = q_dist[range(self.batch_size), next_actions] ## 실제 선택된 next_action의 _distribution만 선택한다.
            
            m_dist = torch.zeros(self.batch_size, self.atom_size)
            
            T_z = rewards + masks * (self.gamma ** self.multi_step) * self.z
            T_z = T_z.clamp(min=self.vmin, max=self.vmax)
    
            b = (T_z - self.vmin) / self.delta_z
    
            l = b.floor().long()
            u = b.ceil().long()
            un_mask = l != u

            u_bj = (u.float() - b)
            bj_l = (b - l.float())

            for i in range(self.batch_size):
                m_dist[i, l[i,un_mask[i]]] +=  (q_dist[i] * u_bj[i])[un_mask[i]]
                m_dist[i, u[i,un_mask[i]]] +=  (q_dist[i] * bj_l[i])[un_mask[i]]

        # loss 정의
        loss = -(m_dist * log_p_dist).sum(1).mean()
        
        # backward 시작!
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # soft target update
        self.soft_update()        
