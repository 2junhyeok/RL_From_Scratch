from collections import defaultdict
import sys
import os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from common.gridworld import GridWorld
from common.print_value import print_value

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        
    def get_action(self, state):
        action_probs = self.pi[state]# action probs per state
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V
        
        self.V[state] += (target - self.V[state]) * self.alpha

if __name__ == "__main__":
    env = GridWorld()
    agent = TdAgent()
    
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.eval(state, reward, next_state, done)
            if done:
                break
            state = next_state
    print_value(agent.V, env)