import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import dill
from qtable import QTableLearning

# Initialize the Bipedal Walker environment
env = gym.make('BipedalWalker-v3')#, render_mode="human")

# Get the observation and action space
obs = env.observation_space
act = env.action_space

print(f"Observation Space: {obs}")
print(f"Action Space: {act}")

window_size = 100

# Function to calculate the moving average using np.mean
def moving_average(data, window_size=window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        window_average = np.mean(window)
        moving_averages.append(window_average)
    return moving_averages

mode = 'uniform'

if mode == 'uniform':
    q_learning = QTableLearning(env, mode=mode)  # Initialize Q-learning with uniform discretization
elif mode == 'range':
    q_learning = QTableLearning(env, obs_buckets=1, mode=mode)  # Initialize Q-learning with range-based discretization
if mode == 'uniform' or mode == 'range':
    file_name = f'qtable_{mode}.dill'

    qtable, rewards, elapsed_times = q_learning.learn()

    # Save the Q-table
    with open(f'qtable_{mode}.dill', 'wb') as f:
        dill.dump(qtable, f)

    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Total Reward')
    if len(rewards) >= window_size:
        plt.plot(range(window_size, len(rewards) + 1), moving_average(rewards), label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Reward Evolution ({mode} discretization)')
    plt.legend()
    plt.savefig(f'Rewards Evolutions {mode}.png')

    # Plot elapsed times
    plt.figure(figsize=(12, 6))
    plt.plot(elapsed_times, label=f'Elapsed Time ({mode})')
    plt.xlabel('Episode')
    plt.ylabel('Elapsed Time (s)')
    plt.title(f'Elapsed Time per Episode ({mode} discretization)')
    plt.legend()
    plt.savefig(f'Elapsed Times {mode}.png')
