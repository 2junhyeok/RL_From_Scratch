from collections import defaultdict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from common.gridworld import GridWorld

def eval_onestep(pi, V, env, gamma=0.9):
    '''
    1. 모든 state에 순차적으로 접근한다.
    2. 해당 state에서 action distribution을 가져온다.
    3. action transition function(env.state())으로 다음 state를 얻는다.
    4. 다음 state 정보를 통해 new_V를 갱신한다.
    Args:
        pi (defaultdict): policy 
        V (defaultdict): value function
        env (GridWorld): environment
        gamma (float): discount rate
    '''
    for state in env.states():
        if state == env.goal_state:
            V[state]=0
            continue
        
        action_probs = pi[state]
        new_V = 0
        
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        
        V[state] = new_V
    
    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    '''
    eval_onestep()을 반복 호출하여 갱신 변화량의 최댓값이 0.001보다 작아지면 중단한다.
    '''
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)
        
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold:
            break
    
    return V

if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)
    
    V = policy_eval(pi, V, env, gamma)

    print(V)