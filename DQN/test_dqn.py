import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQN import QNetwork

# Definizione dell'ambiente
ENV = "BipedalWalker-v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 100
NORMALIZE = True
REWARD_THRESHOLD = -1000
STEPS_THRESHOLD = 10000

class Normalizer:
    def __init__(self, num_inputs):
        self.mean = np.zeros(num_inputs)
        self.m2 = np.zeros(num_inputs)
        self.count = 0

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


action_buckets = 11

# Carica i parametri del modello salvato
model_params = torch.load('dqn_model.pth', map_location=DEVICE)

# Crea l'ambiente
env = gym.make(ENV, render_mode = 'human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Crea il modello e carica i parametri
model = QNetwork(state_dim, action_buckets ** action_dim).to(DEVICE)
model.load_state_dict(model_params)
model.eval()  # Imposta il modello in modalità valutazione

def test_DQN(dqn, env, episodes = EPISODES, normalize = NORMALIZE, reward_threshold = REWARD_THRESHOLD, steps_threshold = STEPS_THRESHOLD):
    rewards = []
    render_freq = 10
    if normalize:
        normalizer = Normalizer(env.observation_space.shape[0])
    for i in range(episodes):
        observation = env.reset()[0]
        if normalize:
            normalizer.update(observation)
        total_reward = 0
        done = False
        steps = 0

        while not done and total_reward > reward_threshold and steps < steps_threshold:
            with torch.no_grad():  # Disabilita il calcolo dei gradienti durante l'inferenza
                # Preprocessa l'osservazione se necessario
                state = torch.tensor(observation).float().to(DEVICE)
                if normalize:
                    state = normalizer.normalize(state)
                q_values = dqn(state)

                # Scegli l'azione con il massimo Q-value
                flat_discrete_action = q_values.argmax().item()
                discrete_action = np.array(np.unravel_index(flat_discrete_action, [action_buckets] * env.action_space.shape[0]))
                action = (discrete_action / (action_buckets - 1)) * (env.action_space.high - env.action_space.low) + env.action_space.low

            # Esegui l'azione nell'ambiente
            next_observation, reward, done, _, _ = env.step(action)
            if normalize:
                normalizer.update(next_observation)
            total_reward += reward
            observation = next_observation

            # Visualizza l'ambiente (commenta questa linea se non vuoi visualizzare l'ambiente)
            if steps % render_freq == 0:
                env.render()
            steps += 1
        rewards.append(total_reward)

        print(f"Episode {i + 1}, Total Reward: {total_reward}")

    env.close()  # Assicurati di chiudere l'ambiente
    print("Testing completed.")
    return rewards

# Esegui il test
rewards = test_DQN(model, env)


plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title(f'Rewards of DQN (Testing with Loaded network)')
plt.legend()
plt.savefig(f'Rewards DQN.png')
