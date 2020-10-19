import numpy as np
import time
from collections import deque
import random

class replay_buffer():
    def __init__(self, buffer_size, multi_step, batch_size):
        self.buffer = deque(maxlen=buffer_size) ## 실제 저장되는 buffer 선언. deque활용.
        self.buffer_size = buffer_size
        self.multi_step = multi_step
        self.batch_size = batch_size
        self.cnt = 0 ## buffer에 저장된 데이터 갯수.

    def store(self, data):
        self.buffer.append(data)
        if self.cnt != self.buffer_size: ## max buffer size까지 세어줌.
            self.cnt+=1

    def make_batch(self):
        if self.multi_step == 1: ## 1-step TD
            index = random.sample(range(self.cnt - self.multi_step), self.batch_size) ## 현재 buffer size 기준으로 batch만큼 sampling함. data 기준 0부터 시작이라 1빼줘야함.
            index = np.array(index)
            random_mini_batch = np.array([self.buffer[x] for x in index])
            random_mini_batch_next = 0
        else: ## multi-step TD
            index = random.sample(range(self.cnt - self.multi_step), self.batch_size) ## 현재 buffer size 기준으로 batch만큼 sampling함.
            index = np.array(index) ## index로 쓰기위해 np.array활용. list로는 적용 안됨.
            random_mini_batch = np.array([self.buffer[x] for x in index]) ## TD시작지점 index에 해당하는 데이터, batch size가 32라 매우 빠름.
            random_mini_batch_next = np.array([self.buffer[x] for x in (index + self.multi_step)]) ## TD 끝지점 index에 해당하는 데이터.
        
        return random_mini_batch, random_mini_batch_next, index, np.array(self.buffer)