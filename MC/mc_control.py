from collections import defaultdict

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        #self.cnts = defaultdict(lambda: 0)
        self.memory = []
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.ns())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()
    
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            n = (state, action)
            #self.cnts[n] += 1# state, action 등장 횟수
            #self.Q[n] += (G - self.Q[n]) / self.cnts[n]# action value
            #self.pi[state] = greedy_probs(self.Q, state)
            self.Q[n] += (G - self.Q[key]) / self.cnts[key]
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)
            
    
def greedy_probs(Q, state, epsilon = 0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 -epsilon)
    return action_probs
