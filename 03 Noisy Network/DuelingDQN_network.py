import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import pickle
"""
학습 속도문제로 제외. 엄밀한 제어를 위해선 사용!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
torch.manual_seed(7777)
torch.cuda.manual_seed(7777)
torch.cuda.manual_seed_all(7777) # if use multi-GPU

class Factorised_noisy_layer(nn.Module):
    def __init__(self, input_p, output_q, sigma_zero=0.5):
        super(Factorised_noisy_layer, self).__init__()
        init_mu = 1 / math.sqrt(input_p) # mu의 초기 값을 위함. mu[-1/sqrt(p),+1/sqrt(p)]
        init_sigma = sigma_zero / math.sqrt(input_p) # sigma의 초기값을 위함. sigma_0 / sqrt(p) input_p값이 torch.Tensor type이 아니라 파이썬 라이브러리 sqrt씀.
        
        self.register_buffer("eps_i",torch.zeros(1, input_p)) # Gaussian variable, dimension is R^p
        self.register_buffer("eps_j",torch.zeros(output_q, 1)) # Gaussian variables, dimension is R^q
        # why use register_buffer? when if you want to use some parameters for designing layer and don't include both eps parameters in backpropagate process. 
        
        # initialize parameter.
        self.sigma_W = nn.Parameter(torch.full((output_q,input_p), init_sigma)) # (q,p) 행렬 사이즈에 맞게 초기 sigma 값으로 행렬을 채워준다.
        self.sigma_B = nn.Parameter(torch.full((output_q,), init_sigma)) # (q) 행렬 사이즈에 맞게 초기 sigma 값으로 채워준다.

        self.mu_W = nn.Parameter(torch.empty(output_q,input_p)) # 파라미터 값이 빈 행렬을 만들어 준다.
        self.mu_B = nn.Parameter(torch.empty(output_q))
        self.mu_W.data.uniform_(-init_mu, init_mu) # 빈 행렬에 논문에서 말한 초기화 값을 uniform에서 sampling해 할당한다.
        self.mu_B.data.uniform_(-init_mu, init_mu)

    def forward(self, x):
        self.eps_i.normal_() # assign epsilon_i from normal distribution. register_buffer에 등록된 parameter를 매번 normal 즉 gaussion분포에서 가져옴.
        self.eps_j.normal_() # assign epsilon_j.
                             # factorised는 epsilon_i와 epsilon_j로 나눠진, 값을 이용해 nosiy를 만들기 위한 epsilon으로 사용한다!
        eps_W = torch.mul(self.function_f(self.eps_i), self.function_f(self.eps_j)) # dimension is R^q*p / epsilon W = f(eps_i)f(eps_j)
        eps_B = self.function_f(self.eps_j.squeeze(-1)) # dimension is R^q / epsilon B = f(eps_j)

        no_Bias = self.mu_B + self.sigma_B * eps_B # calculate noisy_Bias.
        no_Wieght = self.mu_W + self.sigma_W * eps_W # calculate noisy_Wieght
        
        return F.linear(x, no_Wieght, no_Bias) # y = no_W * X + no_B

    def function_f(self, eps): # factorised에서 사용되는 f 함수.
        sign = torch.sign(eps) # 부호함수로, 0보다 크면 1, 0과 같으면 0, 0보다 작으면 -1을 반환하는 함수이다.
        eps_abs = eps.abs() # 함수에 들어온 eps 값에 절대값으로 바꿔준다.
        f_x = sign * eps_abs.sqrt() # f(x) = sign(x) * sqrt(abs(x))로 완성해준다.
        return f_x

class Independent_noisy_layer(nn.Module):
    def __init__(self, input_p, output_q):
        super(Factorised_noisy_layer, self).__init__()
        init_mu = math.sqrt(3 / input_p) # mu의 초기 값을 위함. mu[-sqrt(3/p),+sqrt(3/p)]
        init_sigma = 0.017 # sigma를 위한 초기값. factorised와 다르게, sigma_0를 따로 쓰지 않음.
        self.register_buffer("eps_W",torch.zeros(output_q, input_p)) # Gaussian variable, dimension is R^p
        self.register_buffer("eps_B",torch.zeros(output_q)) # Gaussian variables, dimension is R^q
        # why use register_buffer? when if you want to use some parameters for designing layer and don't include both eps parameters in backpropagate process. 
        
        # initialize parameter.
        self.sigma_W = nn.Parameter(torch.full((output_q,input_p), init_sigma)) # (q,p) 행렬 사이즈에 맞게 초기 sigma 값으로 행렬을 채워준다.
        self.sigma_B = nn.Parameter(torch.full((output_q,), init_sigma)) # (q) 행렬 사이즈에 맞게 초기 sigma 값으로 행렬을 채워준다.

        self.mu_W = nn.Parameter(torch.empty(output_q, input_p)) # 파라미터 값이 빈 행렬을 만들어 준다.
        self.mu_B = nn.Parameter(torch.empty(output_q))
        self.mu_W.data.uniform_(-init_mu, init_mu) # 빈 행렬에 논문에서 말한 초기화 값을 uniform에서 sampling해 할당한다.
        self.mu_B.data.uniform_(-init_mu, init_mu)

    def forward(self, x):
        self.eps_W.normal_() # assign weight epsilon from normal distribution. register_buffer에 등록된 parameter를 매번 normal 즉 gaussion분포에서 가져옴.
        self.eps_B.normal_() # assign bias epsilon.
        
        # Independent에서는 행렬 사이즈 만큼 바로 할당함.

        eps_W = self.eps_W # dimension is R^q*p
        eps_B = self.eps_B # dimension is R^q

        no_Bias = self.mu_B + self.sigma_B * eps_B # calculate noisy_Bias.
        no_Wieght = self.mu_W + self.sigma_W * eps_W # calculate noisy_Wieght
        
        return F.linear(x, no_Wieght, no_Bias) # y = no_W * X + no_B

class Q(nn.Module):
    def __init__(self, state_space, action_space):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_space,64)
        self.fc2 = nn.Linear(64,32)
        self.fc_v = Factorised_noisy_layer(32, 1)
        self.fc_a = Factorised_noisy_layer(32, action_space)
        
        #self.fc_v = nn.Linear(32,1)
        #self.fc_a = nn.Linear(32,action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        V_values = self.fc_v(x)
        Advantages = self.fc_a(x)
     
        Q_values = V_values + (Advantages - Advantages.mean(dim=-1,keepdim=True))

        return Q_values
