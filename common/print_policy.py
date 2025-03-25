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