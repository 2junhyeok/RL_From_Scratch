from collections import defaultdict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from common.gridworld import GridWorld
from common.print_policy import print_policy
from DP.policy_eval import policy_eval

def argmax(d):
    '''
    딕셔너리 d에서 가장 큰 value를 찾아 하나의 key를 반환
    Args:
        d (dict): dictionary
    '''
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V, env, gamma):
    '''
    value function V를 사용하여 greedy하게 선택한 정책을 반환한다.
    Args:
        V (defaultdict): value function
        env (GridWorld): environment
        gamma (float): discount rate
    '''
    pi = {}
    
    for state in env.states():
        action_value = {}
        
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_value[action] = value
            
        max_action = argmax(action_value)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0# max_action이 결정적이 되도록 확률 분포를 생성한다.
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold=0.001, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)
    
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        
        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        
        pi = new_pi
    return pi

if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
    print_policy(pi)