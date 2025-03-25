from collections import defaultdict
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from common.gridworld import GridWorld
from common.print_value import print_value

class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []
        
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        '''
        실제로 수행한 행동과 보상을 기록
        '''
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()
        
    def eval(self):
        '''
        self.memory를 따라 각 state에서 얻은 return을 계산
        '''
        G = 0# init
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] +=1
            self.V[state] += (G - self.V[state]) / self.cnts[state]

if __name__ == "__main__":
    env = GridWorld()
    agent = RandomAgent()
    
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.eval()
                break
            state = next_state
    print_value(agent.V, env)