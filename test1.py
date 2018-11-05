#-----------student model-------------------------
import numpy as np
from collections import deque
import  random
import tensorflow as tf
from dqn import DuelingDQN

class simStudent():
    #能力值和耐力值，sigmod函数
    #lectute 该门门课重要性 list (暂时只做1门课)
    def sigmoid(self,ability):
        tmp=np.dot(self.lecture,ability)
        return 1 / (1 + np.exp(-(tmp+1)))

    def __init__(self,lecture):
        self.lecture=lecture
        self.now_state = np.array([])
        self.days=0#天数
        self.problem_count=0#题目数
        self.patient=0#耐心
        self.cor_count=0#正确题目数
        self.over=False#练习周期是否完结
    def reset(self):
        tmp=[]
        for j in range(len(self.lecture)):
            tmp.append(random.triangular(-1,1))
        tmp.append(self.sigmoid(tmp))
        self.now_state =np.array(tmp)
        self.patient=self.now_state[-1]
        return self.now_state.copy()

    def do_math(self,action):
        #耐心低和能力低都容易做错
        true_prob=6*(self.now_state[0]+1)/2+0.5
        if true_prob<action:
            return random.uniform(0,1)<0.2
        if true_prob>=action and self.now_state[1]>=0.25:
            return random.uniform(0,1) < 0.7
        if true_prob>=action and self.now_state[1]<0.25:
            return random.uniform(0,1) < 0.3


    def step(self,action):
        #一共6个难度等级的题目，一天不超过6个,做错消耗更多的耐心,做难题奖励更大
        self.problem_count+=1
        done=False
        if self.problem_count>5:
            done=True
        if self.do_math(action):
            self.cor_count+=1
            self.now_state[0] += 0.06*action/6
            self.now_state[1] = self.now_state[1] * 0.95
        else:
            self.now_state[0] += 0.01
            self.now_state[1]= self.now_state[1] * 0.7
        return (self.now_state, 0.65*(self.now_state[0]+1)/2+0.35*(self.now_state[1]), done)

    def check(self):
        #检查训练周期是否结束
        if self.problem_count>=5:
            self.days+=1
            self.problem_count=0
            if self.cor_count>=3:
                self.patient*=self.patient*((self.cor_count/5)+0.6)
                self.now_state[1]=self.patient
            self.cor_count=0
        if self.days>=9:
            self.over=True


def run():
    step = 0
    for episode in range(300):
        student=simStudent([1])
        observation = student.reset()

        while True:
            action = RL.choose_action(observation)
            observation_,reward,done = student.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                student.check()
            if student.over:
                break
            step += 1

if __name__ == '__main__':
    env = simStudent([1])
    RL = DuelingDQN(6, 2,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    run()

    #例子
    student1=simStudent([1])
    obs=student1.reset()
    while not student1.over:
        print('state:',student1.now_state)
        act=RL.choose_action(obs)
        student1.step(act)
        obs=student1.now_state
        print('action :',act)
        student1.check()
        print('day no.:',student1.days)
