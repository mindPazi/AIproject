import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from DQN import Agent

ENV = "BipedalWalker-v3"

env = gym.make(ENV)
agent = Agent(env)
model_params, rewards, losses, elapsed_times = agent.learn()

# Save the trained model parameters
torch.save(model_params, 'dqn_model.pth')

window_size = 100
# Function to calculate the moving average using np.mean
def moving_average(data, window_size=window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        window_average = np.mean(window)
        moving_averages.append(window_average)
    return moving_averages

plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Total Reward')
if len(rewards) >= window_size:
    plt.plot(range(window_size, len(rewards) + 1), moving_average(rewards), label='Moving Average (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title(f'Reward Evolution (DQN)')
plt.legend()
plt.savefig(f'Rewards Evolutions DQN.png')

# Plot losses
plt.figure(figsize=(12, 6))
plt.plot(losses, label=f'Losses (DQN)')
plt.xlabel('Episode')
plt.ylabel('Loss (MSE)')
plt.title(f'Loss Value per Episode')
plt.legend()
plt.savefig(f'Losses DQN.png')

# Plot elapsed times
plt.figure(figsize=(12, 6))
plt.plot(elapsed_times, label=f'Elapsed Time (DQN)')
plt.xlabel('Episode')
plt.ylabel('Elapsed Time (s)')
plt.title(f'Elapsed Time per Episode')
plt.legend()
plt.savefig(f'Elapsed Times DQN.png')
