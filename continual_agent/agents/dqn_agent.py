"""
DQN agent for the GenomicsToolEnv.

Implements a standard Deep Q-Network with:
  - Experience replay buffer
  - Target network (hard update every C steps)
  - Epsilon-greedy exploration with linear decay
  - Configurable hidden dimensions

The policy network weights are exposed so that EWC can compute the
Fisher information matrix after each task.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class DQNConfig:
    state_dim: int = 14
    action_dim: int = 11
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 3000      # steps over which epsilon decays
    buffer_size: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 200  # hard update every N steps


class QNetwork(nn.Module):
    """Two-hidden-layer Q-network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def get_recent(self, n: int) -> list:
        return list(self.buffer)[-n:]


class DQNAgent:
    """DQN agent with epsilon-greedy exploration."""

    def __init__(self, config: DQNConfig | None = None, device: str = "cpu") -> None:
        self.cfg = config or DQNConfig()
        self.device = torch.device(device)

        self.policy_net = QNetwork(
            self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dim
        ).to(self.device)
        self.target_net = QNetwork(
            self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)
        self.steps_done: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        decay = self.cfg.epsilon_decay
        progress = min(self.steps_done / decay, 1.0)
        return self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * (
            1.0 - progress
        )

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.cfg.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(state_t).argmax(dim=1).item())

    def greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(state_t).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def update(self, ewc_penalty_fn=None) -> float | None:
        if len(self.buffer) < self.cfg.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.cfg.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.cfg.gamma * next_q * (1 - dones_t)

        td_loss = F.smooth_l1_loss(q_values, target_q)

        ewc_loss = ewc_penalty_fn(self.policy_net) if ewc_penalty_fn else torch.tensor(0.0)
        loss = td_loss + ewc_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())
