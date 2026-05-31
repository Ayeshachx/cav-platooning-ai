"""
agent.py
CAV Platooning Project – VehicleAgent Class

Represents one CAV in the platoon.  Each step it:
    1. Reads its own state from SUMO via TraCI (get_state)
    2. Passes state to the AI model for a decision   (decide)
    3. Applies the chosen action as a speed command  (take_action)
    4. Computes its reward signal                    (get_reward)

The leader vehicle (CAV_0) cruises at a fixed target speed;
followers run the AI/rule-based controller.

SDD references: §4.2.2 (VehicleAgent), §3.1 (data structures),
                §5.4.1 (pseudocode).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ai_interface import AbstractAIModel


# ── Reward zone boundaries (metres) — from SRS §4.2.6 ────────────────────────
DANGER_MAX  = 2.0
CAUTION_MAX = 5.0
TARGET_MIN  = 5.0
TARGET_MAX  = 15.0
WARNING_MAX = 30.0


@dataclass
class StepData:
    """Snapshot of one agent's state at one simulation step."""
    step:            int
    vehicle_id:      str
    x:               float
    y:               float
    speed:           float          # m/s
    gap_to_leader:   float          # metres  (0.0 if no leader)
    relative_speed:  float          # m/s    (ego − leader)
    ai_action:       int            # 0/1/2 or -1 (leader / override)
    target_speed:    float          # m/s  speed sent to SUMO
    safety_override: bool
    reward:          float
    fuel_consumption:float          # mg/s (raw SUMO value)


class VehicleAgent:
    """
    Represents one CAV platoon vehicle.

    Parameters
    ----------
    vehicle_id   : str   SUMO vehicle ID  (e.g. 'CAV_0').
    ai_model     : AbstractAIModel instance  (rule-based or DQN).
    role         : 'leader' | 'follower'
    config       : dict  Subsection of parsed config.yaml.
    """

    def __init__(self,
                 vehicle_id: str,
                 ai_model:   AbstractAIModel,
                 role:       str,
                 config:     dict):

        self.vehicle_id = vehicle_id
        self.ai_model   = ai_model
        self.role       = role          # 'leader' or 'follower'

        # Config shortcuts
        self.max_speed      = config["simulation"]["max_speed"]
        self.speed_delta    = config["ai"]["speed_delta"]
        self.target_speed   = config["simulation"]["target_speed"]
        self.reward_cfg     = config["reward"]

        # Live state (updated each step by update_state)
        self.speed:          float = 0.0
        self.position:       tuple = (0.0, 0.0)
        self.gap_to_leader:  float = 0.0
        self.relative_speed: float = 0.0
        self.leader_speed:   float = 0.0
        self.fuel:           float = 0.0

        # Episode accumulators
        self.total_reward:    float = 0.0
        self.step_count:      int   = 0
        self.override_count:  int   = 0

        # Last action taken
        self._last_action:       int   = AbstractAIModel.ACTION_MAINTAIN
        self._last_target_speed: float = 0.0
        self._safety_override:   bool  = False

    # ── State ─────────────────────────────────────────────────────────────────

    def update_state(self, traci) -> None:
        """
        Query TraCI for all sensor readings and update internal state.
        Called at the START of each simulation step.
        """
        try:
            self.speed    = traci.vehicle.getSpeed(self.vehicle_id)
            self.position = traci.vehicle.getPosition(self.vehicle_id)
            self.fuel     = traci.vehicle.getFuelConsumption(self.vehicle_id)

            # Leader detection: look up to 100 m ahead
            leader_info = traci.vehicle.getLeader(self.vehicle_id, 100.0)

            if leader_info is not None:
                leader_id, raw_gap = leader_info
                # SUMO returns gap between front bumpers; ensure ≥ 0
                self.gap_to_leader = max(0.0, raw_gap)
                try:
                    self.leader_speed   = traci.vehicle.getSpeed(leader_id)
                    self.relative_speed = self.speed - self.leader_speed
                except Exception:
                    self.leader_speed   = self.speed   # assume matching
                    self.relative_speed = 0.0
            else:
                # No leader in range — treat as free-flow
                self.gap_to_leader  = 0.0   # sentinel: no leader
                self.leader_speed   = self.speed
                self.relative_speed = 0.0

        except Exception as e:
            # Vehicle may not be in network yet or already departed
            pass

    def get_raw_state(self) -> np.ndarray:
        """
        Return the raw (un-normalised) state vector shape (4,).
        [gap_m, relative_speed_mps, ego_speed_mps, leader_speed_mps]
        """
        return np.array([
            self.gap_to_leader,
            self.relative_speed,
            self.speed,
            self.leader_speed,
        ], dtype=np.float32)

    # ── Decision & Action ─────────────────────────────────────────────────────

    def decide_and_act(self, traci,
                       safety_override_active: bool,
                       override_target_speed:  float = 0.0) -> int:
        """
        Compute and apply a speed command for this step.

        If safety_override_active is True, the SafetyModule has already
        called setSpeed(0) — we just record the override and return.

        Returns
        -------
        int  Action index taken (−1 if safety override or leader cruising).
        """
        self._safety_override = safety_override_active

        if safety_override_active:
            self._last_action       = -1
            self._last_target_speed = override_target_speed
            return -1

        # ── Leader: cruise at constant target speed ───────────────────────────
        if self.role == "leader":
            target = self.target_speed
            self._apply_speed(traci, target)
            self._last_action       = -1          # leader doesn't use AI
            self._last_target_speed = target
            return -1

        # ── Follower: rule-based or AI decides ───────────────────────────────
        raw_state = self.get_raw_state()
        action    = self.ai_model.predict(raw_state)

        # If using rule-based, compute smooth continuous target speed directly
        if hasattr(self.ai_model, "compute_target_speed") and self.gap_to_leader > 0:
            target = self.ai_model.compute_target_speed(
                self.gap_to_leader, self.leader_speed
            )
        else:
            target = self._action_to_speed(action)

        self._apply_speed(traci, target)
        self._last_action       = action
        self._last_target_speed = target
        return action

    def _action_to_speed(self, action: int) -> float:
        """Map discrete action index to a target speed."""
        if action == AbstractAIModel.ACTION_ACCELERATE:
            return min(self.speed + self.speed_delta, self.max_speed)
        elif action == AbstractAIModel.ACTION_BRAKE:
            return max(self.speed - self.speed_delta, 0.0)
        else:   # MAINTAIN
            return self.speed

    def _apply_speed(self, traci, target_speed: float) -> None:
        """Send speed command to SUMO via TraCI."""
        try:
            traci.vehicle.setSpeed(self.vehicle_id,
                                   float(np.clip(target_speed, 0.0, self.max_speed)))
        except Exception:
            pass

    # ── Reward ────────────────────────────────────────────────────────────────

    def compute_reward(self, safety_override: bool) -> float:
        """
        Calculate reward for the current step.

        Reward zones (gap in metres):
            danger   < 2 m   → −100  (set externally by SafetyModule)
            caution  2–5 m   → linear penalty
            target   5–15 m  → positive bonus
            warning  15–30 m → small penalty
            far      > 30 m  → large penalty

        Returns
        -------
        float  Reward value for this step.
        """
        r = self.reward_cfg

        if safety_override:
            # Penalty already applied by SafetyModule; don't double-count
            return float(r["override_penalty"]) if "override_penalty" in r \
                   else -100.0

        # Leader has no gap-based reward
        if self.role == "leader":
            return 0.0

        gap = self.gap_to_leader

        # No leader detected → neutral
        if gap <= 0.0:
            return 0.0

        # ── Gap-zone reward ───────────────────────────────────────────────────
        if gap < DANGER_MAX:
            reward = float(r.get("caution_penalty", -5.0)) * 2   # near-danger
        elif gap < CAUTION_MAX:
            # Linear penalty: worst at gap=2, zero at gap=5
            frac   = (gap - DANGER_MAX) / (CAUTION_MAX - DANGER_MAX)
            reward = float(r.get("caution_penalty", -5.0)) * (1.0 - frac)
        elif gap <= TARGET_MAX:
            # Target zone: bonus, maximised at perfect_gap
            perfect_gap = float(self.target_speed / self.max_speed * TARGET_MAX
                                if hasattr(self, '_cfg_tg') else 10.0)
            gap_err     = abs(gap - perfect_gap) / (TARGET_MAX - TARGET_MIN)
            reward      = float(r.get("target_zone_bonus", 10.0)) * (1.0 - gap_err)
            # Speed-match bonus
            if abs(self.relative_speed) < 1.0:
                reward += float(r.get("speed_match_bonus", 5.0))
        elif gap <= WARNING_MAX:
            reward = float(r.get("warning_penalty", -3.0))
        else:
            reward = float(r.get("far_penalty", -50.0))

        self.total_reward += reward
        self.step_count   += 1
        return reward

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self, step: int, reward: float) -> StepData:
        """Build a StepData record for DataLogger.log_step()."""
        return StepData(
            step            = step,
            vehicle_id      = self.vehicle_id,
            x               = self.position[0],
            y               = self.position[1],
            speed           = round(self.speed,          4),
            gap_to_leader   = round(self.gap_to_leader,  4),
            relative_speed  = round(self.relative_speed, 4),
            ai_action       = self._last_action,
            target_speed    = round(self._last_target_speed, 4),
            safety_override = self._safety_override,
            reward          = round(reward, 4),
            fuel_consumption= round(self.fuel, 4),
        )

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset episode accumulators (called by SimulationManager.reset())."""
        self.total_reward   = 0.0
        self.step_count     = 0
        self.override_count = 0
        self._safety_override = False

    def __repr__(self) -> str:
        return (f"VehicleAgent({self.vehicle_id}, role={self.role}, "
                f"speed={self.speed:.1f}m/s, gap={self.gap_to_leader:.1f}m)")
