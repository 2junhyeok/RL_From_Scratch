from collections import defaultdict

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []
    
    def get_action(self, state):
        pass
    
    def add(self, state, action, reward):
        pass
    
    def reset(self):
        self.memory.clear()
    
    def update(self):
        pass
    
def greedy_probs(Q, state, action_size=4):
    pass
