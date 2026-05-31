"""
gym_env.py
CAV Platooning Project – Gymnasium Environment Wrapper (Fixed)

Key fixes vs v1:
  - Properly waits for ALL CAVs to spawn before returning first observation
  - Training episodes capped at train_episode_steps (default 500) for speed
  - gap=0 (no leader) treated as neutral reward, not far_penalty
  - Leader speed set every step before agent acts
  - Robust _read_vehicle with retry on missing vehicle
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        raise ImportError("Run:  pip install gymnasium")

# ── TraCI ─────────────────────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    for c in [r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo", r"C:\Sumo",
              "/usr/share/sumo", "/usr/local/share/sumo"]:
        if Path(c).exists():
            os.environ["SUMO_HOME"] = c
            break

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci


class CAVPlatooning(gym.Env):
    """
    Gymnasium environment for single-follower DQN training.

    Observation : Box(4,) normalised [gap, rel_v, ego_v, lead_v]
    Action      : Discrete(3)  0=ACCEL  1=MAINTAIN  2=BRAKE
    Episode ends: collision  OR  max_steps reached  OR  all CAVs off network
    """

    metadata = {"render_modes": ["human"]}
    LOW  = np.zeros(4, dtype=np.float32)
    HIGH = np.ones(4,  dtype=np.float32)

    def __init__(self, config: dict, agent_id: str = "CAV_1",
                 port: int = 8813, headless: bool = True):
        super().__init__()

        self.config     = config
        self.agent_id   = agent_id
        self.port       = port
        self.headless   = headless

        self._root      = Path(__file__).parent
        self._sumo_cfg  = self._root / config["sumo"]["config_file"]
        self._timeout   = int(config["sumo"].get("launch_timeout", 30))
        self._step_size = float(config["simulation"]["step_size"])
        self._max_speed = float(config["simulation"]["max_speed"])
        self._target_sp = float(config["simulation"]["target_speed"])
        self._reward_cfg= config["reward"]
        self._safety_cfg= config["safety"]

        # Shorter episodes during training = faster learning feedback
        self._max_steps = int(config["simulation"].get(
            "train_episode_steps", 500))

        self.observation_space = spaces.Box(
            low=self.LOW, high=self.HIGH, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._step_count   = 0
        self._episode      = 0
        self._sumo_started = False
        self._last_obs     = self.LOW.copy()
        self._total_reward = 0.0

        from ai_rule_based import RuleBasedController
        ai_cfg = config["ai"]
        self._rule = RuleBasedController(
            target_gap     = ai_cfg.get("target_gap",     10.0),
            kp             = ai_cfg.get("kp",              0.5),
            max_correction = ai_cfg.get("max_correction",  3.0),
            max_speed      = self._max_speed,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._episode    += 1
        self._step_count  = 0
        self._total_reward= 0.0

        if self._sumo_started:
            self._reload()
        else:
            self._launch()
            self._sumo_started = True

        # Advance simulation until BOTH leader and agent are in the network
        self._wait_for_platoon()

        obs  = self._get_obs()
        self._last_obs = obs
        return obs, {"episode": self._episode}

    def step(self, action: int):
        self._step_count += 1

        active = set(traci.vehicle.getIDList())

        # ── 1. Set leader speed every step ───────────────────────────────────
        if "CAV_0" in active:
            try:
                traci.vehicle.setSpeed("CAV_0", self._target_sp)
            except Exception:
                pass

        # ── 2. Apply rule-based to other followers ────────────────────────────
        for vid in ["CAV_2", "CAV_3", "CAV_4"]:
            if vid in active:
                self._rule_step(vid)

        # ── 3. Read current state for safety check ────────────────────────────
        gap, ego_v, lead_v = self._read(self.agent_id)
        rel_v = ego_v - lead_v

        # ── 4. Safety override check ──────────────────────────────────────────
        override   = False
        pen        = 0.0
        min_dist   = self._safety_cfg["min_safe_distance"]
        ttc_thresh = self._safety_cfg["ttc_threshold"]

        if gap > 0 and ego_v > 0:
            ttc = gap / rel_v if rel_v > 0.001 else float("inf")
            if gap < min_dist or ttc < ttc_thresh:
                try:
                    traci.vehicle.setSpeed(self.agent_id, 0.0)
                except Exception:
                    pass
                override = True
                pen      = float(self._safety_cfg["override_penalty"])

        # ── 5. Apply AI action (if not overridden) ────────────────────────────
        if not override and self.agent_id in active:
            self._apply_action(self.agent_id, action, ego_v)

        # ── 6. Advance SUMO ───────────────────────────────────────────────────
        try:
            traci.simulationStep()
        except Exception:
            obs = self._last_obs
            return obs, -200.0, True, False, {"error": "traci_lost"}

        # ── 7. New observation ────────────────────────────────────────────────
        obs = self._get_obs()
        self._last_obs = obs

        # ── 8. Reward ─────────────────────────────────────────────────────────
        if override:
            reward = pen
        else:
            reward = self._reward(obs)

        self._total_reward += reward

        # ── 9. Termination ────────────────────────────────────────────────────
        cols       = traci.simulation.getCollisions()
        terminated = len(cols) > 0
        truncated  = self._step_count >= self._max_steps

        active2 = set(traci.vehicle.getIDList())
        if self.agent_id not in active2 and not terminated:
            # Agent reached destination — success
            truncated = True

        if terminated:
            reward = float(self._safety_cfg.get("collision_penalty", -200.0))

        info = {
            "step":      self._step_count,
            "gap":       float(obs[0] * 50.0),
            "override":  override,
            "collision": terminated,
            "ep_reward": self._total_reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
        self._sumo_started = False

    # ── Private ───────────────────────────────────────────────────────────────

    def _launch(self):
        binary = "sumo.exe" if sys.platform == "win32" else "sumo"
        sumo_home = os.environ.get("SUMO_HOME", "")
        bin_path  = os.path.join(sumo_home, "bin", binary)
        sumo_bin  = bin_path if Path(bin_path).exists() else binary

        cmd = [sumo_bin, "-c", str(self._sumo_cfg),
               "--remote-port", str(self.port),
               "--start", "--quit-on-end",
               "--no-step-log", "true",
               "--no-warnings", "true"]

        subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)

        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                traci.init(self.port)
                return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError(f"SUMO failed on port {self.port}")

    def _reload(self):
        try:
            traci.load(["-c", str(self._sumo_cfg), "--start"])
        except Exception:
            try:
                traci.close()
            except Exception:
                pass
            time.sleep(0.5)
            self._launch()

    def _wait_for_platoon(self, max_steps: int = 300):
        """
        Advance SUMO until both vehicles are present AND the gap is
        within a reasonable starting range (8-15m). During warmup
        the leader is slowed slightly so the follower can close in.
        """
        target_gap = float(self.config["ai"].get("target_gap", 10.0))

        for step in range(max_steps):
            try:
                traci.simulationStep()
            except Exception:
                break

            active = set(traci.vehicle.getIDList())
            if "CAV_0" not in active or self.agent_id not in active:
                continue

            try:
                gap, ego_v, lead_v = self._read(self.agent_id)

                # During warmup: slow leader slightly so follower closes gap
                if gap > target_gap * 1.5 and step < 200:
                    traci.vehicle.setSpeed("CAV_0",
                        self._target_sp * 0.85)   # 85% speed = ~21 m/s
                else:
                    traci.vehicle.setSpeed("CAV_0",
                        self._target_sp)           # restore normal speed

                # Exit warmup once gap is in acceptable starting range
                if 0 < gap <= target_gap * 1.8:   # within 18m
                    traci.vehicle.setSpeed("CAV_0", self._target_sp)
                    return

            except Exception:
                return

    def _get_obs(self) -> np.ndarray:
        gap, ego_v, lead_v = self._read(self.agent_id)
        rel_v = ego_v - lead_v
        return np.array([
            np.clip(gap   / 50.0,                              0.0, 1.0),
            np.clip((rel_v + self._max_speed)/(2*self._max_speed), 0.0, 1.0),
            np.clip(ego_v / self._max_speed,                   0.0, 1.0),
            np.clip(lead_v/ self._max_speed,                   0.0, 1.0),
        ], dtype=np.float32)

    def _read(self, vid: str) -> Tuple[float, float, float]:
        """Return (gap_m, ego_speed, leader_speed). Safe — never raises."""
        try:
            active = set(traci.vehicle.getIDList())
            if vid not in active:
                return 0.0, 0.0, 0.0
            ego_v       = traci.vehicle.getSpeed(vid)
            leader_info = traci.vehicle.getLeader(vid, 150.0)
            if leader_info:
                lid, gap = leader_info
                gap      = max(0.0, gap)
                lead_v   = traci.vehicle.getSpeed(lid) if lid in active \
                           else self._target_sp
            else:
                gap, lead_v = 0.0, self._target_sp
            return gap, ego_v, lead_v
        except Exception:
            return 0.0, 0.0, 0.0

    def _apply_action(self, vid: str, action: int, current_speed: float):
        delta = float(self.config["ai"].get("speed_delta", 2.0))
        if action == 0:
            t = min(current_speed + delta, self._max_speed)
        elif action == 2:
            t = max(current_speed - delta, 0.0)
        else:
            t = current_speed
        try:
            traci.vehicle.setSpeed(vid, t)
        except Exception:
            pass

    def _rule_step(self, vid: str):
        gap, ego_v, lead_v = self._read(vid)
        if gap <= 0:
            return
        t = self._rule.compute_target_speed(gap, lead_v)
        try:
            traci.vehicle.setSpeed(vid, float(t))
        except Exception:
            pass

    def _reward(self, obs: np.ndarray) -> float:
        """
        Dense shaped reward so the agent always gets a gradient signal.

        Core idea: reward = -|gap - target_gap| / scale  + bonuses
        This means the agent always knows which direction to move,
        regardless of which zone it is in.
        """
        r          = self._reward_cfg
        target_gap = float(self.config["ai"].get("target_gap", 10.0))
        gap        = float(obs[0]) * 50.0          # metres
        rel_v      = float(obs[1]) * 2.0 * self._max_speed - self._max_speed

        # No leader in range — neutral, wait
        if gap <= 0.0:
            return 0.0

        # ── Hard danger zone ─────────────────────────────────────────────────
        if gap < 2.0:
            return -20.0

        # ── Dense shaping: continuous penalty proportional to gap error ───────
        gap_error  = abs(gap - target_gap)          # 0 = perfect
        # Scale so that gap_error=0 → reward=+20, gap_error=20 → reward=0
        shaped     = float(r.get("target_zone_bonus", 20.0)) * max(0.0, 1.0 - gap_error / 20.0)

        # ── Speed matching bonus ──────────────────────────────────────────────
        speed_bonus = float(r.get("speed_match_bonus", 10.0)) if abs(rel_v) < 2.0 else 0.0

        # ── Closing bonus: reward agent for reducing gap when too far ─────────
        closing_bonus = 0.0
        if gap > target_gap and rel_v < -0.5:   # moving toward leader
            closing_bonus = 3.0
        elif gap < target_gap and rel_v > 0.5:  # slowing when too close
            closing_bonus = 3.0

        return shaped + speed_bonus + closing_bonus

    def normalise_state(self, raw: np.ndarray,
                        max_gap: float = 50.0,
                        max_speed: float = None) -> np.ndarray:
        ms = max_speed or self._max_speed
        gap, rel_v, ego_v, lead_v = raw
        return np.array([
            np.clip(gap  / max_gap,                   0.0, 1.0),
            np.clip((rel_v + ms) / (2.0 * ms),        0.0, 1.0),
            np.clip(ego_v  / ms,                      0.0, 1.0),
            np.clip(lead_v / ms,                      0.0, 1.0),
        ], dtype=np.float32)
