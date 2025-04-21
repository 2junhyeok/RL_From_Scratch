import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import gym_minigrid
from gym import spaces
from collections import defaultdict

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
        # 에이전트 정보를 격자에 삽입
        grid[env.agent_pos[0], env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'], COLOR_TO_IDX['red'], env.agent_dir
        ])
        # 테두리 제거 및 일차원 변환
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
    if args.model == 'sarsa':
        from models.SARSA import SARSA as Agent
    else:
        from models.Q_Learning import QLearning as Agent

    agent = Agent(action_size=4, agent_indicator=10)
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

        # 로그
        q_means.append(np.mean([max(q) for q in agent.q_values.values()]))
        rewards.append(total)
        if total > 0:
            wins += 1
        successes.append(wins / (ep + 1))

        if (ep + 1) % args.log_interval == 0:
            avg_r = np.mean(rewards[-args.log_interval:])
            print(f"Ep {ep+1}/{args.episodes} - avg_reward: {avg_r:.3f}")

    df = pd.DataFrame({
        'q_mean': q_means,
        'reward': rewards,
        'success_rate': successes
    })
    return df, agent

# Q-table 화살표 시각화 및 CSV 저장
def visualize_q_table(agent, grid_size, params, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))
    cmap = plt.cm.Blues

    for state, qs in agent.q_values.items():
        row, col = divmod(state, grid_size - 2)
        x, y = col + 0.5, grid_size - row - 1.5
        mx = max(qs) if qs else 1e-6
        for a, q in enumerate(qs):
            dx, dy = {0:(0,0.3),1:(0,-0.3),2:(-0.3,0),3:(0.3,0)}[a]
            norm = q / mx if mx > 0 else 0
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                     fc=cmap(norm), ec=cmap(norm))
    # 그리드 설정
    ax.set_xticks(range(grid_size+1)); ax.set_yticks(range(grid_size+1))
    ax.grid(True)
    ax.set_title(f"Q Arrows ({params})")
    path = os.path.join(out_dir, f"q_arrows_{params}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved Q-arrow plot: {path}")

    # CSV
    rows = []
    for state, qs in agent.q_values.items():
        r, c = divmod(state, grid_size - 2)
        rows.append([r, c] + qs)
    cols = ['row','col','up','down','left','right']
    pd.DataFrame(rows, columns=cols).to_csv(
        os.path.join(out_dir, f"q_table_{params}.csv"), index=False)
    print("Saved Q-table CSV.")

# 결과 시각화 및 저장
def visualize_results(df, agent, args):
    out = args.save_path
    os.makedirs(out, exist_ok=True)
    params = f"{args.model}_a{args.alpha}_g{args.gamma}_e{args.epsilon}"

    # Q-mean, reward, success rate
    for col, title in [('q_mean','Mean Q-value'),('reward','Episode Reward'),('success_rate','Success Rate')]:
        plt.figure(figsize=(8,4))
        plt.plot(df[col], label=title)
        plt.xlabel('Episode'); plt.ylabel(title)
        plt.title(f"{title} ({params})")
        plt.grid(); plt.legend(); plt.tight_layout()
        plt.savefig(f"{out}/{params}_{col}.png"); plt.close()

    # Q-table arrows & CSV
    visualize_q_table(agent, grid_size=6, params=params, out_dir=out)
    df.to_csv(f"{out}/{params}_results.csv", index=False)
    print("All results saved.")

# 엔트리 포인트
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
    results, agent = run_experiment(args)
    visualize_results(results, agent, args)
