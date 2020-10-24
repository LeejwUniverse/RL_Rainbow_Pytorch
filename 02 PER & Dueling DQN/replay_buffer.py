import numpy as np
import time
from collections import deque
import random
import Tree

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

        """ per """
        self.max_priority = 1.0
        self.beta = 0.4
        self.beta_constant = 0.00001

        self.sum_tree = Tree.sum_tree(buffer_size)
        self.min_tree = Tree.min_tree(buffer_size)

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

        self.sum_tree.add_data(self.max_priority)
        self.min_tree.add_data(self.max_priority)
        self.replay_buffer[self.buffer_index] = data # append data at current replay buffer index.

        self.cnt += 1
        self.cnt = self.cnt if self.buffer_size > self.cnt else self.buffer_size
        self.buffer_index += 1 # count current replay buffer index
    
    def make_batch(self):

        mini_batch = []
        w_batch = []
        idx_batch = []
        sum_p = self.sum_tree.sum_all_priority()
        N = self.buffer_size
        min_p = self.min_tree.min_p() / sum_p # getting max weight.
        max_w = np.power(N * min_p, -self.beta) # 가장 작은 priority가 가장 큰 max weight가 된다.

        K = sum_p/self.batch_size

        for j in range(self.batch_size):
            seg1 = K * j
            seg2 = K * (j + 1)

            sampling_num = random.uniform(seg1, seg2)
            priority, tree_idx, replay_buffer_idx = self.sum_tree.search(sampling_num)
            mini_batch.append(self.replay_buffer[replay_buffer_idx])
            
            p_j = priority / sum_p
            w_j = np.power(p_j * N, -self.beta) / max_w
            w_batch.append(w_j)
            idx_batch.append(tree_idx)
        self.beta = min(1.0, self.beta + self.beta_constant)
        return np.array(mini_batch), np.array(w_batch), idx_batch
    
    def update_priority(self, priority, index):
        self.sum_tree.update_priority(priority, index)
        self.min_tree.update_priority(priority, index)

    def update_max_priority(self, priority):
        self.max_priority = max(self.max_priority, priority)