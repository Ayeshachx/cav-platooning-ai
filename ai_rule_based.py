"""
ai_rule_based.py
CAV Platooning Project – Rule-Based Proportional Gap Controller

This is the PLACEHOLDER controller referenced in SDD §4.2.3 (AbstractAIModel).
It uses a simple proportional (P) controller to maintain a target gap:

    gap_error    = gap_to_leader - target_gap
    correction   = Kp * gap_error
    target_speed = leader_speed + correction  (clamped to [0, max_speed])

    If target_speed > ego_speed  → ACCELERATE
    If target_speed < ego_speed  → BRAKE
    Else                         → MAINTAIN

Once you switch to DQN (ai.mode = "dqn"), this file is no longer called
but remains in the project as a baseline / comparison controller.

No external dependencies beyond NumPy.
"""

import json
import numpy as np
from pathlib import Path
from ai_interface import AbstractAIModel


class RuleBasedController(AbstractAIModel):
    """
    Proportional gap-keeping controller.

    Parameters
    ----------
    target_gap     : float  Desired gap to maintain in metres (default 10.0 m).
    kp             : float  Proportional gain (default 0.5).
    max_correction : float  Maximum speed correction per step in m/s (default 3.0).
    max_speed      : float  Vehicle speed cap in m/s (default 33.33).
    speed_delta    : float  Speed step per action in m/s (default 2.0).
    """

    def __init__(self,
                 target_gap:     float = 10.0,
                 kp:             float = 0.5,
                 max_correction: float = 3.0,
                 max_speed:      float = 33.33,
                 speed_delta:    float = 2.0):

        self.target_gap     = target_gap
        self.kp             = kp
        self.max_correction = max_correction
        self.max_speed      = max_speed
        self.speed_delta    = speed_delta

        # Diagnostics
        self._last_gap_error  = 0.0
        self._last_correction = 0.0

    # ── AbstractAIModel interface ─────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """
        Compute action from raw (un-normalised) state vector.

        Parameters
        ----------
        state : np.ndarray shape (4,)
            [gap_m, relative_speed_mps, ego_speed_mps, leader_speed_mps]
            NOTE: rule-based uses raw values, not normalised.

        Returns
        -------
        int  0=ACCELERATE, 1=MAINTAIN, 2=BRAKE
        """
        gap, rel_speed, ego_speed, leader_speed = float(state[0]), float(state[1]), \
                                                  float(state[2]), float(state[3])

        # ── No leader detected ────────────────────────────────────────────────
        # gap == 0 or very large means no leader in sensor range.
        # Leader vehicle: just cruise at target speed.
        if gap <= 0 or gap > 200.0:
            if ego_speed < self.max_speed - self.speed_delta:
                return self.ACTION_ACCELERATE
            elif ego_speed > self.max_speed + self.speed_delta:
                return self.ACTION_BRAKE
            return self.ACTION_MAINTAIN

        # ── Proportional gap controller ───────────────────────────────────────
        gap_error  = gap - self.target_gap                          # positive → too far
        correction = np.clip(self.kp * gap_error,
                             -self.max_correction, self.max_correction)

        # Target speed = leader speed + correction
        target_speed = np.clip(leader_speed + correction, 0.0, self.max_speed)

        # Store for diagnostics
        self._last_gap_error  = gap_error
        self._last_correction = correction

        # ── Map continuous target to discrete action ──────────────────────────
        diff = target_speed - ego_speed

        if diff > self.speed_delta * 0.3:
            return self.ACTION_ACCELERATE
        elif diff < -self.speed_delta * 0.3:
            return self.ACTION_BRAKE
        else:
            return self.ACTION_MAINTAIN

    def save_model(self, path: str) -> None:
        """Save controller parameters to a JSON file."""
        params = {
            "controller": "RuleBasedController",
            "target_gap":     self.target_gap,
            "kp":             self.kp,
            "max_correction": self.max_correction,
            "max_speed":      self.max_speed,
            "speed_delta":    self.speed_delta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def load_model(self, path: str) -> None:
        """Load controller parameters from a JSON file."""
        with open(path, "r") as f:
            params = json.load(f)
        self.target_gap     = params.get("target_gap",     self.target_gap)
        self.kp             = params.get("kp",             self.kp)
        self.max_correction = params.get("max_correction", self.max_correction)
        self.max_speed      = params.get("max_speed",      self.max_speed)
        self.speed_delta    = params.get("speed_delta",    self.speed_delta)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def compute_target_speed(self, gap: float, leader_speed: float) -> float:
        """
        Public helper: returns the raw target speed (m/s) without discretising.
        Useful for smooth traci.vehicle.setSpeed() calls.
        """
        if gap <= 0 or gap > 200.0:
            return self.max_speed

        gap_error    = gap - self.target_gap
        correction   = np.clip(self.kp * gap_error,
                               -self.max_correction, self.max_correction)
        target_speed = np.clip(leader_speed + correction, 0.0, self.max_speed)
        return float(target_speed)

    def __repr__(self) -> str:
        return (f"RuleBasedController(target_gap={self.target_gap}m, "
                f"kp={self.kp}, last_err={self._last_gap_error:.2f}m)")
