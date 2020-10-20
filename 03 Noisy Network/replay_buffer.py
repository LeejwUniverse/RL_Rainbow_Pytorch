import numpy as np
import time
from collections import deque
import random

np.random.seed(7777)
random.seed(7777)

class replay_buffer():
    def __init__(self, buffer_size, multi_step, batch_size):
        self.buffer_size = buffer_size
        self.multi_step = multi_step
        self.batch_size = batch_size
        
        self.buffer_index = 0
        self.replay_buffer = [0 for i in range(buffer_size)]
        self.multi_step_buffer = deque(maxlen=multi_step)

        self.gamma = 0.99
        self.flag = False
        self.multi_cnt = 0
        self.cnt = 0

    def store(self, data):
        if self.buffer_index == self.buffer_size:
            self.buffer_index = 0
        
        if self.multi_step != 1:
            self.multi_step_buffer.append(data)
            self.multi_cnt += 1
            if self.flag == False and self.multi_cnt == self.multi_step: # if not fully filled with all steps.
                self.flag = True
            if self.flag == False:
                return

            all_data = np.array(self.multi_step_buffer)
            rewards = all_data[:,2]
            masks = all_data[:,4]
            Return_sum = 0
            for i in reversed(range(self.multi_step)):
                Return_sum = rewards[i] + masks[i] * self.gamma * Return_sum
            obs = all_data[0,0]
            action = all_data[0,1]
            next_obs = all_data[-1,3]
            mask = all_data[-1,4]
            data = (obs, action, Return_sum, next_obs, mask)

        self.replay_buffer[self.buffer_index] = data # append data at current replay buffer index.
        self.cnt += 1
        self.cnt = self.cnt if self.buffer_size > self.cnt else self.buffer_size
        self.buffer_index += 1 # count current replay buffer index
    
    def make_batch(self):
        index = random.sample(range(self.cnt), self.batch_size) ## 현재 buffer size 기준으로 batch만큼 sampling함. data 기준 0부터 시작이라 1빼줘야함.
        index = np.array(index)
        random_mini_batch = np.array([self.replay_buffer[x] for x in index])
        
        return random_mini_batch