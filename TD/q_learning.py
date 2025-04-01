from collections import defaultdict
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from common.gridworld import GridWorld
from common.utils import greedy_probs
from common.print_policy import print_q_max

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        
    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p = probs)
    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
        
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)

if __name__ == "__main__":
    env =GridWorld()
    agent = QLearningAgent()
    
    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    print_q_max(agent.Q, env)