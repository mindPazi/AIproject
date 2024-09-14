import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from PPO import FeedForwardNN

"""
    This file is used only to evaluate our trained policy/actor after
    training in main.py with ppo.py. I wrote this file to demonstrate
    that our trained policy exists independently of our learning algorithm,
    which resides in ppo.py. Thus, we can test our trained policy without
    relying on ppo.py.
"""

def _log_summary(ep_len, ep_ret, ep_num):
    """
        Print to stdout what we've logged so far in the most recent episode.

        Parameters:
            None

        Return:
            None
    """
    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def eval_policy(policy, env, episodes, device, render=False):
    rewards = []
    lengths = []
    """
        Evaluate a trained policy on a given environment.
        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
            episodes - The number of episodes to run
            render - Whether we should render our episodes. False by default.

        Return:
            rewards, episodic lengths
        
    """
    # Rollout with the policy and environment, and log each episode's data
    for ep_num in range(1, episodes + 1):
        obs = env.reset()[0]
        done = False
        render_freq = 10

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0            # episodic length
        ep_ret = 0            # episodic return

        while not done:
            t += 1

            # Render environment if specified, off by default
            if render and t % render_freq == 0:
                env.render()

            # Query deterministic action from policy and run it
            with torch.no_grad():
                action = policy(torch.tensor(obs, dtype=torch.float).to(device)).cpu().detach().numpy()
            obs, rew, done, _, _ = env.step(action)

            # Sum all episodic rewards as we go along
            ep_ret += rew

        # Track episodic length
        ep_len = t
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
        rewards.append(ep_ret)
        lengths.append(ep_len)

    return rewards, lengths


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("BipedalWalker-v3", render_mode='human')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
actor_model = torch.load("ppo_actor.pth", map_location=device)

# Build our policy the same way we build our actor model in PPO
policy = FeedForwardNN(obs_dim, act_dim).to(device)
# Load in the actor model saved by the PPO algorithm
policy.load_state_dict(actor_model)
policy.eval()

episodes = 100

# Evaluate our policy with a separate module, eval_policy, to demonstrate
# that once we are done training the model/policy with ppo.py, we no longer need
# ppo.py since it only contains the training algorithm. The model/policy itself exists
# independently as a binary file that can be loaded in with torch.
rewards, lengths = eval_policy(policy=policy, env=env, episodes=episodes, device=device, render=False)

plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title(f'Rewards of PPO (Testing with Loaded Network)')
plt.legend()
plt.savefig(f'Rewards PPO.png')

plt.figure(figsize=(12, 6))
plt.plot(lengths, label='Episodic Length', color='orange')
plt.xlabel('Episode')
plt.ylabel('Episodic Length')
plt.title(f'Episodic Lengths of PPO (Testing with Loaded Network)')
plt.legend()
plt.savefig(f'Episodic Lengths PPO.png')
