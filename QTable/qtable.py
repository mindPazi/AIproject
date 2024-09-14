import numpy as np
import random
from collections import defaultdict
import time

#hyperparameters
OBS_BUCKETS = 9
ACT_BUCKETS = 9
EPISODES = 10000
REWARD_THRESHOLD = -200
GAMMA = 0.99
ALPHA = 0.01
EPSILON_INIT = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01

class QTableLearning:
    def __init__(self, env, episodes=EPISODES, gamma=GAMMA, alpha=ALPHA, epsilon_init=EPSILON_INIT, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                 obs_buckets=OBS_BUCKETS, act_buckets=ACT_BUCKETS, reward_threshold=REWARD_THRESHOLD, mode='uniform', render=False):
        """
        Initialize the Q-learning agent.

        Parameters:
        env (gym.Env): The environment to be used.
        obs_buckets (int): Number of discrete buckets per dimension in the observation space in uniform mode. Buckets per unity in range mode.
        act_buckets (int): Number of discrete actions per dimension in the action space.
        episodes (int): Number of episodes for training.
        gamma (float): Discount factor.
        alpha (float): Learning rate.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        epsilon_min (float): Minimum value for epsilon.
        reward_threshold (float): Threshold for total reward to be considered a success.
        mode (str): Discretization mode, 'uniform' or 'range'.
        render (bool): Whether to render the environment.
        """
        self.env = env
        # discretization parameters
        self.obs_buckets = obs_buckets
        self.act_buckets = act_buckets

        # qtable initialization
        self.qtable = defaultdict(lambda: np.zeros(tuple([self.act_buckets] * self.env.action_space.shape[0])))

        # hyperparameters
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reward_threshold = reward_threshold

        # select observation space discretization mode
        # in range mode, obs_buckets is the number of buckets per unit
        self.mode = mode

        self.render = render
        self.render_interval = 10

    def discretizeState(self, state):
        """
        Discretize the continuous observation state into discrete buckets based on the mode.

        Parameters:
        state (np.array): Continuous observation state.

        Returns:
        tuple: Discretized state.
        """
        if self.mode == 'uniform':
            discrete_state = np.round((state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) * (self.obs_buckets - 1)).astype(int)
        elif self.mode == 'range':
            discrete_state = []
            for i, val in enumerate(state):
                # compute the number of buckets for each dimension
                bucket_size = (self.env.observation_space.high[i] - self.env.observation_space.low[i]) * self.obs_buckets
                # make sure that there is at least one bucket for each dimension
                bucket_size = max(1, int(bucket_size))
                # discretize the dimension
                discrete_val = round((val - self.env.observation_space.low[i]) / (self.env.observation_space.high[i] - self.env.observation_space.low[i]) * (bucket_size - 1))
                discrete_state.append(discrete_val)
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Supported modes are 'uniform' and 'range'.")

        return tuple(discrete_state)

    def discretizeAction(self, action):
        """
        Discretize the continuous action into discrete buckets.

        Parameters:
        action (np.array): Continuous action.

        Returns:
        tuple: Discretized action.
        """
        discrete_action = np.round((action - self.env.action_space.low) / (self.env.action_space.high - self.env.action_space.low) * (self.act_buckets - 1)).astype(int)
        return tuple(discrete_action)

    def undiscretizeAction(self, action):
        """
        Convert a discrete action back into a continuous action.

        Parameters:
        action (tuple): Discretized action.

        Returns:
        tuple: Continuous action.
        """
        action = (np.array(action) / (self.act_buckets - 1)) * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low
        return tuple(action)

    def epsilonGreedyStrategy(self, state):
        """
        Choose an action using the epsilon-greedy strategy.

        Parameters:
        state (tuple): Current discretized state.

        Returns:
        np.array: Chosen action.
        """
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            action = np.random.randint(0, self.act_buckets, size=self.env.action_space.shape)
        else:
            # Exploitation: choose the action with the highest Q value
            q_values = self.qtable[state]
            flat_best_action_index = np.argmax(q_values)
            action = np.array(np.unravel_index(flat_best_action_index, q_values.shape))
        return action

    def updateQTable(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.

        Parameters:
        state (tuple): Current discretized state.
        action (tuple): Discretized action taken.
        reward (float): Reward received.
        next_state (tuple): Next discretized state.
        """
        q_sa = self.qtable[state][action]
        max_next_value = np.max(self.qtable[next_state]) if next_state in self.qtable else 0
        new_value = q_sa * (1 - self.alpha) + self.alpha * (reward + self.gamma * max_next_value)
        self.qtable[state][action] = new_value

    def learn(self):
        """
        Train the Q-learning agent over the specified number of episodes.
        """
        rewards = []  # List to store total rewards for each episode
        elapsed_times = []

        for episode in range(1, self.episodes + 1):
            total_reward = 0
            steps_taken = 0
            start_time = time.time()

            init = self.env.reset()[0]
            state = self.discretizeState(init)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            while total_reward > self.reward_threshold:
                action = self.epsilonGreedyStrategy(state)
                continuous_action = self.undiscretizeAction(action)
                next_state, reward, done, _, _ = self.env.step(continuous_action)

                if self.render and steps_taken % self.render_interval == 0:
                    self.env.render()

                next_state = self.discretizeState(next_state)
                action = tuple(action)
                self.updateQTable(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                steps_taken += 1

                if done:
                    break

            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            rewards.append(total_reward)  # Store total reward for this episode
            print(f"Episode {episode}/{self.episodes}, Total Reward: {total_reward}, Elapsed Time: {elapsed_time}")

        self.env.close()
        max_reward = max(rewards)  # Calculate the maximum reward
        print(f"Maximum Reward: {max_reward}")
        return self.qtable, rewards, elapsed_times
