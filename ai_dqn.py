"""
ai_dqn.py
CAV Platooning Project – Deep Q-Network (DQN) Controller

Implements the full DQN algorithm as specified in SDD §4.2.3 and §5.1:

    Architecture : Linear(4,64) → ReLU → Linear(64,64) → ReLU
                   → Linear(64,32) → ReLU → Linear(32,3)
    Algorithm    : DQN with Experience Replay + Target Network
    Optimizer    : Adam  (lr = 0.001)
    Loss         : Huber Loss (smooth_l1)
    Actions      : 0=ACCELERATE, 1=MAINTAIN, 2=BRAKE

Switch to this by setting  ai.mode: "dqn"  in config.yaml.
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

from ai_interface import AbstractAIModel


# ── Experience tuple stored in the replay buffer ──────────────────────────────
Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "next_state", "done"]
)


# ─────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Feedforward neural network that maps state → Q-values for each action.

    Architecture (from SDD §5.1):
        Input  : 4  neurons  (normalised state vector)
        Hidden1: 64 neurons  ReLU
        Hidden2: 64 neurons  ReLU
        Hidden3: 32 neurons  ReLU
        Output :  3 neurons  Linear  (Q values for ACCEL / MAINTAIN / BRAKE)
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 3,
                 hidden: List[int] = None):
        super().__init__()
        if hidden is None:
            hidden = [64, 64, 32]

        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # Xavier initialisation for stable early training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Fixed-size circular buffer storing Experience tuples.
    Oldest entries are silently discarded when full (FIFO).
    """

    def __init__(self, capacity: int = 10_000, seed: int = 42):
        self.memory   = deque(maxlen=capacity)
        self.capacity = capacity
        random.seed(seed)

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append(Experience(
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

    @property
    def is_ready(self) -> bool:
        """True once the buffer has enough samples to start training."""
        return len(self.memory) >= max(64, self.capacity // 100)


# ─────────────────────────────────────────────────────────────────────────────
class DQN_Solver(AbstractAIModel):
    """
    Full DQN implementation with:
        - Online network   (selects actions)
        - Target network   (computes stable Bellman targets)
        - Experience replay buffer
        - ε-greedy exploration with exponential decay

    Parameters
    ----------
    state_dim        : int    Size of state vector (4).
    action_dim       : int    Number of discrete actions (3).
    config           : dict   ai: section from config.yaml.
    device           : str    'cuda' | 'cpu' | 'auto'
    """

    def __init__(self,
                 state_dim:  int  = 4,
                 action_dim: int  = 3,
                 config:     dict = None,
                 device:     str  = "auto"):

        if not _TORCH:
            raise ImportError(
                "PyTorch not installed.\n"
                "Run:  pip install torch\n"
                "Then re-launch the simulation."
            )

        cfg = config or {}

        # ── Hyperparameters (from config.yaml ai: section) ────────────────────
        self.lr               = float(cfg.get("learning_rate",       0.001))
        self.gamma            = float(cfg.get("gamma",               0.95))
        self.epsilon          = float(cfg.get("epsilon_start",       1.0))
        self.epsilon_min      = float(cfg.get("epsilon_end",         0.01))
        self.epsilon_decay    = float(cfg.get("epsilon_decay",       0.995))
        self.batch_size       = int(  cfg.get("batch_size",          32))
        self.target_update    = int(  cfg.get("target_update_freq",  100))
        self.buffer_size      = int(  cfg.get("replay_buffer_size",  10_000))
        hidden_layers         = list( cfg.get("hidden_layers",       [64, 64, 32]))

        self.state_dim  = state_dim
        self.action_dim = action_dim

        # ── Device ────────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO]   DQN device: {self.device}")

        # ── Networks ──────────────────────────────────────────────────────────
        self.online = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()   # target never trained directly

        # ── Optimizer & loss ──────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr)

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.memory = ReplayBuffer(capacity=self.buffer_size)

        # ── Counters ──────────────────────────────────────────────────────────
        self.steps_done   = 0
        self.episodes_done = 0
        self.losses: List[float] = []

    # ── AbstractAIModel interface ─────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection.

        With probability ε  → random action (exploration).
        With probability 1-ε → argmax Q(state, ·) (exploitation).

        Parameters
        ----------
        state : np.ndarray shape (4,) — RAW (un-normalised) state vector.

        Returns
        -------
        int  0=ACCELERATE, 1=MAINTAIN, 2=BRAKE
        """
        # Normalise before feeding to network
        norm_state = self.normalise_state(state)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        self.online.eval()
        with torch.no_grad():
            t = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            q_values = self.online(t)
        self.online.train()

        return int(q_values.argmax(dim=1).item())

    def save_model(self, path: str) -> None:
        """Save online network weights + training state to a .pth file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_state_dict":  self.online.state_dict(),
            "target_state_dict":  self.target.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "epsilon":            self.epsilon,
            "steps_done":         self.steps_done,
            "episodes_done":      self.episodes_done,
        }, path)
        print(f"[INFO]   Model saved → {Path(path).name}")

    def load_model(self, path: str) -> None:
        """Restore network weights and training state from a .pth file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon       = checkpoint.get("epsilon",       self.epsilon_min)
        self.steps_done    = checkpoint.get("steps_done",    0)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        print(f"[INFO]   Model loaded ← {Path(path).name}  "
              f"(ep={self.episodes_done}, ε={self.epsilon:.3f})")

    # ── Training ──────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done) -> None:
        """Push one transition into the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Sample a mini-batch and perform one Bellman update.

        Returns
        -------
        float  Huber loss value for this step (0.0 if buffer not ready).
        """
        if not self.memory.is_ready:
            return 0.0

        batch = self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(
                        np.array([self.normalise_state(e.state) for e in batch])
                      ).to(self.device)
        actions     = torch.LongTensor(
                        [e.action for e in batch]
                      ).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(
                        [e.reward for e in batch]
                      ).to(self.device)
        next_states = torch.FloatTensor(
                        np.array([self.normalise_state(e.next_state) for e in batch])
                      ).to(self.device)
        dones       = torch.FloatTensor(
                        [float(e.done) for e in batch]
                      ).to(self.device)

        # ── Current Q values from online network ──────────────────────────────
        q_current = self.online(states).gather(1, actions).squeeze(1)

        # ── Bellman target from target network ────────────────────────────────
        with torch.no_grad():
            q_next   = self.target(next_states).max(1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        # ── Huber loss + backprop ─────────────────────────────────────────────
        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1

        # Sync target network periodically
        if self.steps_done % self.target_update == 0:
            self.update_target_network()

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def update_target_network(self) -> None:
        """Hard copy online → target weights."""
        self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
        self.episodes_done += 1

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def mean_loss(self) -> float:
        """Mean loss over the last 100 training steps."""
        recent = self.losses[-100:] if self.losses else [0.0]
        return float(np.mean(recent))

    def __repr__(self) -> str:
        return (f"DQN_Solver(ε={self.epsilon:.3f}, "
                f"steps={self.steps_done}, "
                f"buffer={len(self.memory)}/{self.buffer_size}, "
                f"loss={self.mean_loss:.4f})")
