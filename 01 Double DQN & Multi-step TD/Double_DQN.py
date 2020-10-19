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

import DQN_network

class double_dqn():
    def __init__(self, state_space, action_space, multi_step):
        self.Q_net = DQN_network.Q(state_space, action_space) ## behavior network.
        self.Q_target_net = DQN_network.Q(state_space, action_space) ## target network.
        self.Q_target_net.load_state_dict(self.Q_net.state_dict()) ## 초기에 값은 weight로 초기화 하고 시작함. Q_net의 파라미터를 복사함.

        self.learning_rate = 0.0005
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=self.learning_rate) ## optimizer 아담 사용.

        self.action_space = action_space
        self.gamma = 0.99 ## discount reward를 위한 감마.
        self.tau = 0.001 ## soft target update에 사용되는 타우.
        self.epsilon = 1 ## 초기 epsilon 1부터 시작.
        self.epsilon_decay = 0.00001 ## epsilon 감쇠 값.
        
        self.multi_step = multi_step

    def print_eps(self): ## epsilon 값 출력을 위한 함수.
        return self.epsilon

    def action_policy(self, state):
        Q_values = self.Q_net(state)
        if np.random.rand() <= self.epsilon: ## 0~1의 균일분포 표준정규분포 난수를 생성. 정해준 epsilon 값보다 작은 random 값이 나오면 
            action = random.randrange(self.action_space) ## action을 random하게 선택합니다.
            return action
        else: ## epsilon 값보다 크면, 학습된 Q_player NN 에서 얻어진 Q_values 값중 가장 큰 action을 선택합니다.
            return Q_values.argmax().item()
    
    def soft_update(self): ## DDPG에서 사용된 soft target update 방식.
        for param, target_param in zip(self.Q_net.parameters(), self.Q_target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, random_mini_batch, random_mini_batch_next, index, buffer):
        self.epsilon -= self.epsilon_decay ## 학습해 감에 따라 epilon에 의존하기보단 학습된 정책에 의존되게.
        self.epsilon = max(self.epsilon, 0.05) ## 그래도 가끔씩 새로운 exploration을 위해 최소 0.05은 주기.
        
        # data 분배
        mini_batch = random_mini_batch ## 그냥 이름 줄인용.
        
        obs = np.vstack(mini_batch[:, 0]) ## 1-step TD와 Multi-step TD 모두 obs와 actions은 공통으로 사용됨.
        actions = list(mini_batch[:, 1]) 

        # tensor.
        if self.multi_step == 1:
            rewards = list(mini_batch[:, 2])
            next_obs = np.vstack(mini_batch[:, 3])
            masks = list(mini_batch[:, 4])

            obs = torch.Tensor(obs)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.Tensor(rewards) 
            next_obs = torch.Tensor(next_obs)
            masks = torch.Tensor(masks)
        # multi step용 tensor.
        else:           
            rewards = list(buffer[:, 2])
            masks = list(buffer[:,4])

            next_index = torch.LongTensor(index) + self.multi_step ## multi step중 마지막에 해당하는 index.
            obs = torch.Tensor(obs)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.Tensor(rewards)
            next_obs = torch.Tensor(np.vstack(random_mini_batch_next[:, 0]))
            masks = torch.Tensor(masks)

        # get Q-value
        Q_values = self.Q_net(obs) ## 계속 학습중인 Q NN에서 예상되는 action의 q_value를 얻어온다.
        q_values = Q_values.gather(1, actions).view(-1) ## 각 obs별로 실제 선택된 action들의 q value를 얻어온다. view 해준 이유는 shape을 맞추기 위해.
        
        # get target
        if self.multi_step == 1: ## TD(0) and 1-step TD
            Q_target_values = self.Q_target_net(next_obs)
            next_q_action = self.Q_net(next_obs).max(1)[1].unsqueeze(1)## 실제 발생된 next_state를 넣어, 가장 높은 Q-value를 가진 action을 선택한다.
                                                                       ## unsqueeze는 차원을 맞추기 위해 사용해, 상황에 따라 적절히 바꿔야함.
            target_q_value = Q_target_values.gather(1,next_q_action).view(-1)## max함수를 사용하면 [0]에는 value 값이들어 가 있고 [1]에는 index값이 들어가 있다.
            Y = rewards + masks * self.gamma * target_q_value ## 죽었다면, next가 없으므로 얻어진 reward만 추린다.
        else: ## multi-step TD
            Q_target_values = self.Q_target_net(next_obs)
            next_q_action = self.Q_net(next_obs).max(1)[1].unsqueeze(1)
            target_q_value = Q_target_values.gather(1,next_q_action).view(-1)## max함수를 사용하면 [0]에는 value 값이들어 가 있고 [1]에는 index값이 들어가 있다.
            for j in range(0, self.multi_step): ## dynamic programming 방식으로 맨뒤인 Target 값부터 역순으로 계산됨.
                target_q_value = rewards[next_index - (j + 1)] + masks[next_index - j] * self.gamma * target_q_value
            Y = target_q_value
        # loss 정의 
        MSE = torch.nn.MSELoss() ## mean squear error 사용.
        loss = MSE(q_values, Y.detach()) ## target은 단순히 주기적으로 업데이트해 네트워크를 유지시키므로, parameter가 미분되선 안된다. 그래서 detach() 해줌.
        
        # backward 시작!
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # soft target update
        self.soft_update()        
