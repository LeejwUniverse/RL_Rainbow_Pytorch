
## Double DQN & N-step TD test result.

![test result](https://github.com/LeejwUniverse/RL_Rainbow/blob/master/01%20Double%20DQN%20%26%20Multi-step%20TD/image/Figure_0.png)

## Version2 Double DQN & N-step TD test result.

![test result](https://github.com/LeejwUniverse/RL_Rainbow/blob/master/01%20Double%20DQN%20%26%20Multi-step%20TD/image/Figure_1_ver2.png)

## ETC..
* 저번 실험에서는 운이 너무 좋았나 보다..
* Multi-step TD를 뒤에서 나올 PER와 기타 알고리즘들에 결합하기 쉽도록 개선했다.
* replay buffer에서 flag를 사용한 코드가 있는데, 매우 많은 step을 가는 강화학습 특성상 혹시 모를 정수 에러를 방지하고자 했다.
