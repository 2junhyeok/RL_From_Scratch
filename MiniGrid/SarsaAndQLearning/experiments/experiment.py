import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import gym
from gym import spaces
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
import matplotlib.pyplot as plt

# 경량화된 관측 래퍼
class FlatObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, max_env_steps=50):
        super().__init__(env)
        self.unwrapped.max_steps = max_env_steps
        size = (env.width - 2) * (env.height - 2) * 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(size,), dtype='uint8')

    def observation(self, obs):
        env = self.unwrapped
        grid = env.grid.encode()
        grid[env.agent_pos[0], env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'], COLOR_TO_IDX['red'], env.agent_dir
        ])
        flat = grid[1:-1, 1:-1].ravel()
        return flat

    def render(self, *args, **kwargs):
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)

# 환경 생성
def make_env(env_name):
    env = gym.make(env_name)
    return FlatObsWrapper(env)

# 실험 실행
def run_experiment(args):
    env = make_env(args.env)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.model == 'sarsa':
        from models.SARSA import SARSA as Agent
    else:
        from models.Q_Learning import QLearning as Agent

    agent = Agent(3, agent_indicator=10)# left, right, forward
    agent.alpha, agent.gamma, agent.epsilon = args.alpha, args.gamma, args.epsilon

    q_means, rewards, successes = [], [], []
    wins = 0

    for ep in range(args.episodes):
        obs = env.reset()
        action = agent.act(obs)
        total = 0
        done = False

        while not done:
            nxt, r, done, _ = env.step(action)
            nxt_action = agent.act(nxt)
            agent.update(obs, action, r, nxt, nxt_action)
            total += r
            obs, action = nxt, nxt_action

        q_means.append(np.mean([max(q) for q in agent.q_values.values()]))
        rewards.append(total)
        if total > 0:
            wins += 1
        successes.append(wins / (ep + 1))

        if (ep + 1) % args.log_interval == 0:
            avg_r = np.mean(rewards[-args.log_interval:])
            print(f"Ep {ep+1}/{args.episodes} - avg_reward: {avg_r:.3f}")
            print("State keys in agent.q_values:", list(agent.q_values.keys())[:5], "...")
    
    q_values = {s:np.round(q, 5).tolist() for s, q in agent.q_values.items()}

    q_list = []
    for s, q in q_values.items():
        for a, v in enumerate(q):
            q_list.append({'state': s, 'action': a, 'q_value': v})
    df_q = pd.DataFrame(q_list)
    
    df = pd.DataFrame({
        'q_mean': q_means,
        'reward': rewards,
        'success_rate': successes
    })

    return df, df_q, agent

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 모든 state-action Q-값 히트맵 시각화 함수
def visualize_q_heatmap(df_q, args, save_path=None):
    # 환경 크기 결정 (MiniGrid-Empty-6x6-v0 에서 숫자 추출)
    env_size = int(args.env.split('-')[2].split('x')[0])
    grid_size = env_size - 2  # 테두리 제외

    action_labels = ['Left', 'Right', 'Forward']

    unique_states = sorted(df_q['state'].unique(), reverse=False)
    
    # 상태 레이블 생성
    state_labels = []
    for state in unique_states:
        if isinstance(state, int):
            pos = state // 3
            x, y = pos % grid_size, pos // grid_size
            state_labels.append(f"({x},{y})")

    # Q-value 행렬 생성 (행: 상태, 열: 액션)
    q_matrix = np.zeros((len(unique_states), 3))

    # 상태-액션 쌍에 대해 Q-value 채우기
    for i, state in enumerate(unique_states):
        state_rows = df_q[df_q['state'] == state]
        for _, row in state_rows.iterrows():
            action = int(row['action'])
            if action < 3:  # 0, 1, 2 액션만 사용
                q_matrix[i, action] = row['q_value']

    # 액션 인덱스도 내림차순으로 정렬
    q_matrix = q_matrix[:, ::-1]
    action_labels_desc = action_labels[::-1]

    # 히트맵 시각화
    plt.figure(figsize=(10, 12))
    ax = sns.heatmap(q_matrix, annot=True, fmt='.3f', cmap='viridis',
                     xticklabels=action_labels_desc, yticklabels=state_labels)
    plt.title('Q-values for All States and Actions (Sorted)')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(f"{save_path}/q_heatmap_all_states_actions.png")
    else:
        plt.show()


def visualize_results(df, df_q, args):
    os.makedirs(args.save_path, exist_ok=True)

    plot_training_results(df, save_path=f"{args.save_path}/training_results.png")
    visualize_q_heatmap(df_q, args, save_path=args.save_path)

    df_q.to_csv(f"{args.save_path}/q_values.csv", index=False)

    print(f"시각화 결과가 {args.save_path} 디렉토리에 저장되었습니다.")

def plot_training_results(df, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(df['q_mean'], 'b-')
    axes[0].set_ylabel('Average Q-Value')
    axes[0].set_title('Q-Value Progression')
    axes[0].grid(True)


    smoothed_rewards = moving_average(df['reward'], window_size=50)
    axes[1].plot(smoothed_rewards, 'r-')
    axes[1].set_ylabel('Smoothed Reward')
    axes[1].set_title('Smoothed Reward per Episode')
    axes[1].grid(True)
    
    axes[2].plot(df['success_rate'], 'g-')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_xlabel('Episode')
    axes[2].set_title('Success Rate')
    axes[2].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# 메인 실행 코드
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sarsa','q_learning'], default='q_learning')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-6x6-v0')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-path', type=str, default='results')
    args = parser.parse_args()

    print(f"Model={args.model}, alpha={args.alpha}, gamma={args.gamma}, eps={args.epsilon}")
    results, df_q, agent = run_experiment(args)
    visualize_results(results, df_q, args)