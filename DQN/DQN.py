import numpy as np
import torch
import random
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
#agent
ACT_BUCKETS = 11
EPISODES = 1000
REWARD_THRESHOLD = -200
GAMMA = 0.99
ALPHA = 0.01
EPSILON_INIT = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
NORMALIZE = True

#experience replay
BATCH_SIZE = 64
MEM_SIZE = 1000000

#neural network
HIDDEN_SIZE = 512
LR = 1e-3

# Experience Replay
class ExperienceReplay:
    def __init__(self, buffer_size, batch_size=BATCH_SIZE):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def store_transition(self, state, action, reward, new_state, done):
        self.buffer.append((state, action, reward, new_state, done))

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        # stack: turns a list of tensors into a tensor with a higher dimension
        states = torch.stack(states).to(DEVICE)
        next_states = torch.stack(next_states).to(DEVICE)

        # tensor: converts a list of values into a tensor
        actions = torch.tensor(actions).to(DEVICE)
        rewards = torch.tensor(rewards).float().to(DEVICE)
        dones = torch.tensor(dones).short().to(DEVICE)

        return states, actions, rewards, next_states, dones

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Normalizer:
    def __init__(self, num_inputs):
        self.mean = np.zeros(num_inputs)
        self.m2 = np.zeros(num_inputs)
        self.count = 0

    # Welford's online algorithm for update using unbiased variance
    # more info: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def update(self, x):
        self.count += 1
        old_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.count
        self.m2 += (x - old_mean) * (x - self.mean)

    def normalize(self, x):
        eps = 1e-10
        mean = torch.tensor(self.mean).float().to(DEVICE)
        if self.count > 1:
            variance = self.m2 / (self.count - 1)
        else:
            variance = np.zeros_like(self.m2)
        stdev = torch.tensor(np.sqrt(variance) + eps).float().to(DEVICE)
        x = (x - mean) / (stdev)
        return x

# Agent
class Agent:
    def __init__(self, env, episodes=EPISODES, gamma=GAMMA, alpha=ALPHA, epsilon_init=EPSILON_INIT, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                 experience_replay_size=MEM_SIZE, act_buckets=ACT_BUCKETS, reward_threshold=REWARD_THRESHOLD, normalize=NORMALIZE, lr=LR, render=False):
        self.env = env
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = ExperienceReplay(experience_replay_size)
        self.action_buckets = act_buckets
        self.reward_threshold = reward_threshold
        self.render = render
        self.render_interval = 10

        self.model = QNetwork(env.observation_space.shape[0], self.action_buckets**env.action_space.shape[0]).to(DEVICE)
        # train the NN every "learning_frequency" steps
        self.learning_frequency = 1
        # weight_decay is the L2 regularization parameter in Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.normalize = normalize
        # dynamic normalization computing mean and variance based on observations
        if normalize:
            self.normalizer = Normalizer(env.observation_space.shape[0])

    def discretize_action(self, action):
        discrete_action = np.round((action - self.env.action_space.low) / (self.env.action_space.high - self.env.action_space.low) * (self.action_buckets - 1)).astype(int)
        return tuple(discrete_action)

    def undiscretize_action(self, discrete_action):
        action = (discrete_action / (self.action_buckets - 1)) * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low
        return tuple(action)

    def store(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        if len(self.memory) > BATCH_SIZE:
            self.learn()

    def updateDQN(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        if self.normalize:
            states = self.normalizer.normalize(states)
            next_states = self.normalizer.normalize(next_states)
        q_eval = self.model(states)
        q_next = self.model(next_states)

        # takes the q_value corresponding to the chosen action (for each sample)
        q_eval_actions = q_eval.gather(1, actions.unsqueeze(1)).squeeze(1)

        q_target = q_eval_actions * (1 - self.alpha) + self.alpha * (rewards + self.gamma * q_next.max(1)[0] * (1 - dones))

        loss = F.mse_loss(q_eval_actions, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            discrete_action = np.random.randint(0, self.action_buckets**self.env.action_space.shape[0])
            return discrete_action
        else:
            with torch.no_grad():
                state = torch.tensor(state).float().to(DEVICE)
                if self.normalize:
                    state = self.normalizer.normalize(state)
                q_values = self.model(state)
                discrete_action = q_values.argmax().item()

                return discrete_action

    def learn(self):
        rewards = []
        elapsed_times = []
        losses = []

        for episode in range(1, self.episodes + 1):
            total_reward = 0
            steps_taken = 0
            start_time = time.time()
            episode_losses = []

            observation = self.env.reset()[0]
            if self.normalize:
                self.normalizer.update(observation)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            while total_reward > self.reward_threshold:
                # action is now a number between 0 and act_buckets ^ action_space_size
                action = self.choose_action(observation)
                # map the number to a 4-dimensional array
                discrete_action = np.array(np.unravel_index(action, [self.action_buckets] * self.env.action_space.shape[0]))
                # extract the corresponding continuous action
                continuous_action = self.undiscretize_action(discrete_action)

                next_observation, reward, done, _, _ = self.env.step(continuous_action)
                if self.normalize:
                    self.normalizer.update(next_observation)

                self.memory.store_transition(torch.tensor(observation).float().to(DEVICE), torch.tensor(action).long().to(DEVICE),
                                             reward, torch.tensor(next_observation).float().to(DEVICE), done)

                if steps_taken % self.learning_frequency == 0 and len(self.memory) > self.memory.batch_size:
                    current_loss = self.updateDQN()
                    episode_losses.append(current_loss)

                if self.render and steps_taken % self.render_interval == 0:
                    self.env.render()

                total_reward += reward
                observation = next_observation
                steps_taken += 1

                if done:
                    break

            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            losses.append(np.mean(episode_losses))
            rewards.append(total_reward)  # Store total reward for this episode
            print(f"Episode {episode}/{self.episodes}, Total Reward: {total_reward}, Elapsed Time: {elapsed_time}")

        self.env.close()
        max_reward = max(rewards)  # Calculate the maximum reward
        print(f"Maximum Reward: {max_reward}")
        return self.model.state_dict(), rewards, losses, elapsed_times
