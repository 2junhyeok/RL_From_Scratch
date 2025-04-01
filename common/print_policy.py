import numpy as np

def print_policy(pi):
    actions_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    height = max(state[0] for state in pi.keys()) + 1
    width = max(state[1] for state in pi.keys()) + 1
    
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    for state, actions in pi.items():
        for action, prob in actions.items():
            if prob == 1.0:
                grid[state[0]][state[1]] = actions_map[action]
    
    for row in grid:
        print('|' + '|'.join(f'{cell:^3}' for cell in row) + '|')

def print_q_max(qs, env):
    actions_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    height = env.height
    width = env.width

    grid_policy = [[' ' for _ in range(width)] for _ in range(height)]
    grid_value = [[0.00 for _ in range(width)] for _ in range(height)]

    for (state,action),q in qs.items():
        if grid_value[state[0]][state[1]] <= q:
            grid_value[state[0]][state[1]] = q
            grid_policy[state[0]][state[1]] = actions_map[action]
        else:
            pass
    for row in grid_policy:
        print('|' + '|'.join(f'{cell:^3}' for cell in row) + '|')