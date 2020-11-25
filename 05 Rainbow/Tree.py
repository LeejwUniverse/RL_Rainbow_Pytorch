import numpy as np
from collections import deque
np.random.seed(7777)

class sum_tree():
    def __init__(self, buffer_size):
        self.tree_index = buffer_size - 1 # define sum_tree leaf node index.
        self.array_tree = [0 for i in range((buffer_size * 2) - 1)] # set sum_tree size (double of buffer size)
        self.buffer_size = buffer_size

    def update_tree(self, index):
        # index is a starting leaf node point.
        while True:
            index = (index - 1)//2 # parent node index.
            left = (index * 2) + 1 # left child node inex.
            right = (index * 2) + 2 # right child node index
            self.array_tree[index] = self.array_tree[left] + self.array_tree[right] # sum both child node.
            if index == 0: ## if index is a root node.
                break

    def add_data(self, priority):
        if self.tree_index == (self.buffer_size * 2) - 1: # if sum tree index achive last index.
            self.tree_index = self.buffer_size - 1 # change frist leaf node index.

        self.array_tree[self.tree_index] = priority # append priority at current sum_tree leaf node index.

        self.update_tree(self.tree_index) # update sum_tree node. propagate from leaf node to root node.

        self.tree_index += 1 # count current sum_tree index
    
    def search(self, num):
        index = 0 # always start from root index.
        while True:
            left = (index * 2) + 1
            right = (index * 2) + 2
            if num <= self.array_tree[left]: # if child left node is over current value. 
                index = left                # go to the left direction.
            else:
                num -= self.array_tree[left] # if child left node is under current value.
                index = right               # go to the right direction.
            if index >= self.buffer_size - 1: # if current node is leaf node, break!
                break
        priority = self.array_tree[index]
        tree_idx = index
        replay_buffer_idx = index - (self.buffer_size - 1)
        
        return priority, tree_idx, replay_buffer_idx # return real index in replay buffer.

    def update_priority(self, p, index):
        self.array_tree[index] = p
        self.update_tree(index)

    def sum_all_priority(self):
        return self.array_tree[0]

class min_tree():
    def __init__(self, buffer_size):
        self.tree_index = buffer_size - 1 # define min_tree leaf node index.
        self.array_tree = [1 for i in range((buffer_size * 2) - 1)] # set min_tree size (double of buffer size)
        self.buffer_size = buffer_size

    def update_tree(self, index):
        # index is a starting leaf node point.
        while True:
            index = (index - 1)//2 # parent node index.
            left = (index * 2) + 1 # left child node inex.
            right = (index * 2) + 2 # right child node index
            if self.array_tree[left] > self.array_tree[right]: # if a right child node is smaller than left.
                self.array_tree[index] = self.array_tree[right]
            else:
                self.array_tree[index] = self.array_tree[left]
            if index == 0: ## if index is a root node.
                break

    def add_data(self, priority):
        if self.tree_index == (self.buffer_size * 2) - 1: # if min tree index achive last index.
            self.tree_index = self.buffer_size - 1 # change frist leaf node index.
        
        self.array_tree[self.tree_index] = priority # append priority at current min_tree leaf node index.
        self.update_tree(self.tree_index) # update min_tree node. propagate from leaf node to root node.
        self.tree_index += 1 # count current min_tree index

    def update_priority(self, p, index):
        self.array_tree[index] = p
        self.update_tree(index)

    def min_p(self):
        return self.array_tree[0]