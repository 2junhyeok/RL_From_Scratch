import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import argparse
import torch

import src.DQN as DQN
import src.Dueling_DQN as DuelingDQN
from src.Preprocess import ImageEnv


def evaluate(n_evals=5, agent=None):
    eval_env = gym.make('CarRacing-v3', continuous=False)
    eval_env = ImageEnv(eval_env)
    
    scores = 0
    for i in range(n_evals):
        (s, _), done, ret = eval_env.reset(), False, 0
        while not done:
            a = agent.act(s, training=False)
            s_prime, r, terminated, truncated, info = eval_env.step(a)
            s = s_prime
            ret += r
            done = terminated or truncated
        scores += ret
    return np.round(scores / n_evals, 4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eps-decay", type=str, default="normal")
    parser.add_argument("--buffer", type=int, default=int(1e5))
    parser.add_argument("--model", type=str, default="DQN")
    args = parser.parse_args()
    
    if args.eps_decay == "fast":
        epsilon_decay = 0.99
    elif args.eps_decay == "slow":
        epsilon_decay = 0.9999
    else:
        epsilon_decay = 0.995
    
    
    env = gym.make('CarRacing-v3', continuous=False)
    env = ImageEnv(env)

    max_steps = int(1e5)
    eval_interval = 100
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    
    if args.model == "DQN":
        model = DQN.DQN
    if args.model == "DuelingDQN":
        model = DuelingDQN.DuelingDQN
    
    agent = model(state_dim, 
                    action_dim, 
                    gamma=args.gamma, 
                    epsilon_decay=epsilon_decay,
                    buffer_size=args.buffer)
    
    history = {"Step":[], "AvgReturn":[]}
    
    (s, _) = env.reset()
    
    while True:
        a = agent.act(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        result = agent.process((s, a, r, s_prime, terminated))
        
        s = s_prime
        
        if terminated or truncated:
            s, _ = env.reset()
            
        if agent.total_steps % eval_interval == 0:
            ret = evaluate(agent=agent)
            history["Step"].append(agent.total_steps)
            history["AvgReturn"].append(ret)
            

            print(f"[step {agent.total_steps}] AvgReturn: {ret:.3f}")
            
            os.makedirs("logs", exist_ok=True)
            os.makedirs("checkpoints", exist_ok=True)
            plt.figure(figsize=(8, 5))
            plt.plot(history['Step'], history['AvgReturn'], 'r-')
            plt.xlabel('Step')
            plt.ylabel('AvgReturn')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(f"logs/plot_step_{epsilon_decay}_{args.gamma}.png")
            plt.close()
            if len(history["AvgReturn"]) <= 2:
                torch.save(agent.network.state_dict(),
                        f'checkpoints/dqn_{args.gamma}_{epsilon_decay}_step:{agent.total_steps}_R:{history["AvgReturn"][-1]}.pt')
            else:
                if history["AvgReturn"][-1] > max(history["AvgReturn"][:-2]):
                    torch.save(agent.network.state_dict(),
                            f'checkpoints/dqn_{args.gamma}_{epsilon_decay}_step:{agent.total_steps}_R:{history["AvgReturn"][-1]}.pt')

        if agent.total_steps > max_steps:
            break