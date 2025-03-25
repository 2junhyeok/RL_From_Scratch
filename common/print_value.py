def print_value(V, env):
    height = env.height
    width = env.width
    
    grid = [[0.00 for _ in range(width)] for _ in range(height)]
    
    for state, value in V.items():
        grid[state[0]][state[1]] = value  # 값 저장
    
    for row in grid:
        print('|' + '|'.join(f'{cell:^6.2f}' for cell in row) + '|')