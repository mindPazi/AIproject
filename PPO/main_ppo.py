import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PPO import FeedForwardNN, PPO

env = gym.make("BipedalWalker-v3")

hyperparameters = {
    'lr': 1e-3,
    'gamma': 0.99,
    'n_updates_per_iteration': 10,
    'timesteps_per_batch': 6000,
    'max_timesteps_per_episode': 2000,
}

model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
rewards, actor_losses, critic_losses, elapsed_times = model.learn(total_timesteps=1e7)

#plotta i rewards e gli elapsed times delle iterazioni (ora no sbatti copia da altro codice)

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
plt.title(f'Reward Evolution (PPO)')
plt.legend()
plt.savefig(f'Rewards Evolutions PPO.png')

# Plot actor losses
plt.figure(figsize=(12, 6))
plt.plot(actor_losses, label=f'Actor Losses (PPO)')
plt.xlabel('Iteration')
plt.ylabel('Actor Loss')
plt.title(f'Loss Value of the Actor per Iteration')
plt.legend()
plt.savefig(f'Actor Losses PPO.png')

# Plot critic losses
plt.figure(figsize=(12, 6))
plt.plot(critic_losses, label=f'Critic Losses (PPO)')
plt.xlabel('Iteration')
plt.ylabel('Critic Loss')
plt.title(f'Loss Value of the Critic per Iteration')
plt.legend()
plt.savefig(f'Critic Losses PPO.png')

# Plot elapsed times
plt.figure(figsize=(12, 6))
plt.plot(elapsed_times, label=f'Elapsed Time (PPO)')
plt.xlabel('Iteration')
plt.ylabel('Elapsed Time (s)')
plt.title(f'Elapsed Time per Iteration')
plt.legend()
plt.savefig(f'Elapsed Times PPO.png')
