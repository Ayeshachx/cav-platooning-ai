"""
ai_interface.py
CAV Platooning Project – Abstract AI Model Interface

All AI controllers (rule-based, DQN, PPO, etc.) must inherit from
AbstractAIModel and implement its three abstract methods.

This satisfies SRS NFR-10 (pluggable AI interface) and SDD §4.3.
Swapping algorithms requires zero changes in agent.py or main.py.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class AbstractAIModel(ABC):
    """
    Abstract base class for all CAV speed-control AI models.

    The predict() method is called every simulation step for each follower
    vehicle. It receives a normalised state vector and returns a discrete
    action index:

        0 = ACCELERATE  (+speed_delta m/s)
        1 = MAINTAIN    (no change)
        2 = BRAKE       (-speed_delta m/s)
    """

    # Action index constants — use these everywhere for readability
    ACTION_ACCELERATE = 0
    ACTION_MAINTAIN   = 1
    ACTION_BRAKE      = 2
    ACTION_NAMES      = {0: "ACCELERATE", 1: "MAINTAIN", 2: "BRAKE"}

    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """
        Choose an action given the current state.

        Parameters
        ----------
        state : np.ndarray, shape (4,), dtype float32
            Normalised state vector:
            [gap_distance, relative_speed, ego_speed, leader_speed]
            All values in range [0, 1] after normalisation.

        Returns
        -------
        int
            Action index: 0 (ACCELERATE), 1 (MAINTAIN), or 2 (BRAKE).
        """

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Persist the model's learned parameters to disk.

        Parameters
        ----------
        path : str
            File path to write to (e.g. 'models/cav_model_ep100.pth').
        """

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Restore the model's learned parameters from disk.

        Parameters
        ----------
        path : str
            File path to read from.
        """

    def normalise_state(self, raw_state: np.ndarray, max_gap: float = 50.0,
                        max_speed: float = 33.33) -> np.ndarray:
        """
        Normalise a raw state vector to [0, 1] range.

        Parameters
        ----------
        raw_state : np.ndarray  shape (4,)
            [gap_m, relative_speed_mps, ego_speed_mps, leader_speed_mps]
        max_gap   : float  Maximum expected gap (metres) for normalisation.
        max_speed : float  Maximum allowed speed (m/s) for normalisation.

        Returns
        -------
        np.ndarray  shape (4,)  dtype float32, all values clipped to [0, 1].
        """
        gap, rel_v, ego_v, lead_v = raw_state

        norm = np.array([
            np.clip(gap  / max_gap,                          0.0, 1.0),
            np.clip((rel_v + max_speed) / (2.0 * max_speed), 0.0, 1.0),
            np.clip(ego_v  / max_speed,                      0.0, 1.0),
            np.clip(lead_v / max_speed,                      0.0, 1.0),
        ], dtype=np.float32)

        return norm
