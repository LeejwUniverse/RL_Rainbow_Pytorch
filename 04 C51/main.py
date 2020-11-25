import gym
import matplotlib.pyplot as plt
import C51_DQN
import replay_buffer as buf
import torch
import pickle
"""
학습 속도문제로 제외. 엄밀한 제어를 위해선 사용!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
torch.manual_seed(7777)
torch.cuda.manual_seed(7777)
torch.cuda.manual_seed_all(7777) # if use multi-GPU

def main():
    initial_exploration = 2000 ## 2000개의 memory가 쌓이고 나서 학습 시작!
    print_interval = 100 ## 몇 episode 마다 log 출력할건지.
    
    maximum_steps = 300 ## infinite task라 일정 시점에서 게임을 종료 시켜줘야함.
    episode = 3000 # episode 정해주기.

    action_space = 2
    state_space = 4
    env = gym.make('CartPole-v1')

    multi_step_size = 2 # 1 = 1 step-TD and TD(0), 2 = 2 step-TD, 3 = 3 step-TD
    atom_size = 51
    vmin = -10
    vmax = 10

    buffer_size = 100000 # replay buffer_size
    batch_size = 32 # batch size

    C51 = C51_DQN.c51_dqn(state_space, action_space, multi_step_size, batch_size, atom_size, vmin, vmax)

    replay_buffer = buf.replay_buffer(buffer_size, multi_step_size, batch_size)
    step = 0 ## 총 step을 계산하기 위한 step.
    score = 0
    show_score = []

    for epi in range(episode):
        obs = env.reset() ## 환경 초기화
        for i in range(maximum_steps):

            action = C51.action_policy(torch.Tensor(obs))
            next_obs, reward, done, _ = env.step(action) ## _ 가 원래 info 정보를 가지고 있는데, 학습에 필요치 않음.

            mask = 0 if done else 1 ## 게임이 종료됬으면, done이 1이면 mask =0 그냥 생존유무 표시용.

            replay_buffer.store((obs, action, reward, next_obs, mask)) # repaly buffer에 data 저장.
            
            obs = next_obs ## current state를 이제 next_state로 변경
            score += reward ## reward 갱신.
            step += 1
            
            if step > initial_exploration: ## 초기 exploration step을 넘어서면 학습 시작.
                random_mini_batch = replay_buffer.make_batch() # random batch sampling.
                C51.train(random_mini_batch)

            if done: ## 죽었다면 게임 초기화를 위한 반복문 탈출
                break
                
        if epi % print_interval == 0 and epi != 0:
            show_score.append(score/print_interval) ## reward score 저장.
            print('episode: ', epi,' step: ', step,' epsilon: ', C51.print_eps(),' score: ', score/print_interval) # log 출력.
            score = 0
            with open('2step_C51.p', 'wb') as file:
                pickle.dump(show_score, file)

    env.close()
if __name__ == '__main__':
    main()