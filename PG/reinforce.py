import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from simple_pg import Agent


class ReinforceAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def update(self):
        self.pi.cleargrads()
        
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma*G
            loss += -F.log(prob)*G
        
        loss.backward()
        self.optimizer.update()
        self.memory = []

def main():
    episodes = 3000
    env = gym.make("CartPole-v0")
    agent = ReinforceAgent()
    reward_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)# pi(A_0|S_0)
            next_state, reward, done, info = env.step(action)

            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()

        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode :{}, total reward : {:.4f}".format(episode, total_reward))
if __name__ == "__main__":
    main()