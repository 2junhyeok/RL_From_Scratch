import numpy as np
from models.SARSA import SARSA

class QLearning(SARSA):
    def __init__(self, actions, agent_indicator=10):
        super().__init__(actions, agent_indicator)
        
    def update(self, state, action, reward, next_state, next_action):
        state = self._convert_state(state)
        next_state = self._convert_state(next_state)
        _ = next_action
        q_value = self.q_values[state][action]
        
        next_q_value = np.max(self.q_values[next_state])# max
        td_error = reward + self.gamma * next_q_value - q_value
        self.q_values[state][action] = q_value + self.alpha*td_error
        return