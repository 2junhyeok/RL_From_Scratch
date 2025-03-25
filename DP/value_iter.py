from collections import defaultdict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
from policy_iter import greedy_policy
from common.gridworld import GridWorld
from common.print_policy import print_policy


def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state]=0
            continue
        
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        
        V[state] = max(action_values)
    return V

def value_iter(V, env, gamma, threshold = 0.001, is_render = False):
    while True:
        if is_render:
            env.render_v(V)
        
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)
        
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold:
            break
    return V

if __name__=="__main__":
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9
    
    V = value_iter(V, env, gamma)
    
    pi = greedy_policy(V, env, gamma)
    print_policy(pi)