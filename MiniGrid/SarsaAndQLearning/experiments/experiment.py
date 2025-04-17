import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import gym
import gym_minigrid
from gym import spaces
from gym.wrappers import Monitor
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.SARSA import SARSA
from models.Q_Learning import QLearning


max_env_steps = 50

class FlatObsWrapper(gym.core.ObservationWrapper):
    """Fully observable gridworld returning a flat grid encoding."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=((self.env.width-2) * (self.env.height-2) * 3,),
            dtype='uint8'
        )
        self.unwrapped.max_steps = max_env_steps

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        full_grid = full_grid[1:-1, 1:-1]
        
        flattened_grid = full_grid.ravel()
        return flattened_grid
    
    def render(self, *args, **kwargs):
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)

def gen_wrapped_env(env_name):
    try:
        env = gym.make(env_name)
        env = FlatObsWrapper(env)
        print(f"환경 생성 성공: {env_name}, 관측 공간: {env.observation_space}")
        return env
    except Exception as e:
        print(f"환경 생성 실패: {e}")
        sys.exit(1)
        
def run_experiment(args):
    env = gen_wrapped_env(args.env)
    obs = env.reset()
    if args.model == 'sarsa':
        agent = SARSA(4, agent_indicator=10)
    elif args.model == 'q_learning':
        agent = QLearning(4, agent_indicator=10)

    agent.alpha = args.alpha
    agent.gamma = args.gamma
    agent.epsilon = args.epsilon
    
    # 결과 저장 변수
    q_logs = []
    episode_rewards = []
    success_rates = []
    successes = 0
    
    for episode in range(args.episodes):
        done = False
        
        action = agent.act(obs)
        total_reward = 0
        while not done:
            
            next_obs, reward, done, _ = env.step(action)
            
            next_action = agent.act(next_obs)
            agent.update(obs, action, reward, next_obs, next_action)
            
            total_reward += reward
            obs = next_obs
            action = next_action
            
        q_mean = np.mean([np.max(q) for q in agent.q_values.values()])
        q_logs.append(q_mean)# q_mean
        
        episode_rewards.append(total_reward)
        
        if total_reward > 0:
            successes += 1
            
        success_rate = successes / (episode + 1)
        success_rates.append(success_rate)
        
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode+1}/{args.episodes} - reward avg: {np.mean(episode_rewards[-args.log_interval:]):.3f}")

    results = pd.DataFrame({
        'q_mean': q_logs,
        'episode_reward': episode_rewards,
        'success_rate': success_rates
    })
    
    return results, agent

def visualize_results(results, agent, args):
    os.makedirs(args.save_path, exist_ok=True)

    model_params = f"{args.model}_a{args.alpha}_g{args.gamma}_e{args.epsilon}"

    # Q-value convergence plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['q_mean'], label="Q-value mean")
    plt.plot(results['q_mean'].cumsum() / (pd.Series(np.arange(results.shape[0])) + 1), 
             label="Cumulative average Q-value")
    plt.title(f"{args.model.upper()} Convergence Visualization (α={args.alpha}, γ={args.gamma}, ε={args.epsilon})")
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_path}/q_convergence_{model_params}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Episode reward plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['success_rate'], label="Success Rate")
    plt.plot(results['episode_reward'], label="Episode Reward", linewidth=2)
    plt.title(f"{args.model.upper()} Reward per Episode (α={args.alpha}, γ={args.gamma}, ε={args.epsilon})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_path}/episode_rewards_{model_params}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Success rate plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['success_rate'], label="Success Rate")
    plt.title(f"{args.model.upper()} Success Rate (α={args.alpha}, γ={args.gamma}, ε={args.epsilon})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_path}/success_rate_{model_params}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Q-value table heatmap (top 10 states)
    plt.figure(figsize=(12, 8))

    # Select top 10 states (based on max Q-value)
    states = list(agent.q_values.keys())
    if len(states) > 10:
        max_q_per_state = [np.max(agent.q_values[s]) for s in states]
        top_states_idx = np.argsort(max_q_per_state)[-10:]
        top_states = [states[i] for i in top_states_idx]
    else:
        top_states = states

    # Prepare Q-table data
    q_table_data = []
    for s in top_states:
        q_table_data.append(agent.q_values[s])

    sns.heatmap(q_table_data, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=[f"Action {i}" for i in range(len(agent.q_values[states[0]]))],
                yticklabels=[f"State {s}" for s in top_states])
    plt.title(f"{args.model.upper()} Q-value Table for Top States")
    plt.tight_layout()
    plt.savefig(f"{args.save_path}/q_table_heatmap_{model_params}.png", dpi=300, bbox_inches='tight')
    plt.close()

    
    results.to_csv(f"{args.save_path}/{model_params}_results.csv", index=False)
    print("저장완료")

def main():
    parser = argparse.ArgumentParser(description='강화학습 알고리즘 실험')
    
    parser.add_argument('--model', type=str, default='q_learning', choices=['sarsa', 'q_learning'],
                        help='sarsa or q_learning')
    
    parser.add_argument('--alpha', type=float, default=0.01, help='(default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.9, help='(default: 0.9)')
    parser.add_argument('--epsilon', type=float, default=0.2, help='(default: 0.2)')
    
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-6x6-v0', help='학습 환경')
    parser.add_argument('--episodes', type=int, default=1000, help='(default: 1000)')
    parser.add_argument('--log-interval', type=int, default=100, help='(default: 100)')
    
    parser.add_argument('--save-path', type=str, default='results', help='저장 경로 (default: results)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"model: {args.model}, alpha: {args.alpha}, gamma: {args.gamma}, epsilon: {args.epsilon}")
    print(f"env: {args.env}, episode: {args.episodes}")
    
    results, agent = run_experiment(args)
    
    print(f"time : {time.time() - start_time:.2f} second")
    
    visualize_results(results, agent, args)

if __name__ == "__main__":
    main()