"""Double DQN baseline용 agent 뼈대."""

from collections import deque
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.ddqn import DDQN
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    BUFFER_SIZE,
    GAMMA,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    TARGET_SYNC_INTERVAL,
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.online_net = DDQN(input_shape=state_shape)
        self.target_net = DDQN(input_shape=state_shape)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(capacity=BUFFER_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.batch_size = BATCH_SIZE
        self.num_actions = num_actions
        self.target_sync_interval = TARGET_SYNC_INTERVAL
        self.update_count = 0
        
    def _state_to_numpy(self, state):
        return np.array(state, dtype=np.float32)

    def select_action(self, state, training=True):
        state_array = self._state_to_numpy(state)
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)
        
        if training and random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
        
        return action


    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array([self._state_to_numpy(state) for state in states], dtype=np.float32),
            dtype=torch.float32,
        )
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(
            np.array(
                [self._state_to_numpy(next_state) for next_state in next_states],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        current_q = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1

        if self.update_count % self.target_sync_interval == 0:
            self.sync_target_network()

        return loss.item()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
