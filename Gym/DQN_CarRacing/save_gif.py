import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os

from src.Preprocess import ImageEnv
import src.DQN as DQN
import src.Dueling_DQN as DuelingDQN

def load_agent(model_path, model_type, state_dim, action_dim, gamma, epsilon_decay, buffer_size, device='cuda'):
    if model_type == "DQN":
        agent = DQN.DQN(state_dim, action_dim, gamma=gamma, epsilon_decay=epsilon_decay, buffer_size=buffer_size)
    elif model_type == "DuelingDQN":
        agent = DuelingDQN.DuelingDQN(state_dim, action_dim, gamma=gamma, epsilon_decay=epsilon_decay, buffer_size=buffer_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    agent.network.load_state_dict(torch.load(model_path, map_location=device))
    agent.network.eval()
    return agent

def generate_gif(agent, save_path='logs/car_racing_result.gif'):
    eval_env = gym.make('CarRacing-v3', continuous=False, render_mode='rgb_array')
    eval_env = ImageEnv(eval_env)

    frames = []
    (s, _), done, ret = eval_env.reset(), False, 0

    while not done:
        frames.append(eval_env.render())
        a = agent.act(s, training=False)
        s, r, terminated, truncated, _ = eval_env.step(a)
        ret += r
        done = terminated or truncated

    print(f"Total return: {ret:.2f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure()
    ims = [ [plt.imshow(frame, animated=True)] for frame in frames ]

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    ani.save(save_path, writer='pillow')
    print(f"GIF saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="DQN", choices=["DQN", "DuelingDQN"])
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eps-decay", type=str, default="normal", choices=["fast", "normal", "slow"])
    parser.add_argument("--buffer", type=int, default=int(1e5))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="logs/car_racing_result.gif")
    args = parser.parse_args()

    # Epsilon decay 설정
    if args.eps_decay == "fast":
        epsilon_decay = 0.99
    elif args.eps_decay == "slow":
        epsilon_decay = 0.9999
    else:
        epsilon_decay = 0.995

    env = gym.make('CarRacing-v3', continuous=False)
    env = ImageEnv(env)
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    agent = load_agent(
        model_path=args.model_path,
        model_type=args.model_type,
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        epsilon_decay=epsilon_decay,
        buffer_size=args.buffer,
        device=args.device
    )

    generate_gif(agent, save_path=args.output)
