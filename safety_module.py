"""
safety_module.py
CAV Platooning Project – Safety Override Module

Hard-coded fail-safe layer.  Runs every simulation step for every follower
vehicle.  If either safety condition is true, it overrides the AI command
with maximum emergency braking.

Safety Conditions (SRS FR-10, SDD §8.2):
    1.  gap < min_safe_distance   (static proximity)
    2.  TTC  < ttc_threshold      (dynamic closing speed)

TTC = gap / relative_speed   (only when ego is faster than leader)

This module is intentionally simple and deterministic — correctness
matters more than cleverness here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular import; VehicleAgent type-hint used as string below


@dataclass
class ViolationEvent:
    """Record of a single safety override event."""
    step:       int
    vehicle_id: str
    gap:        float
    ttc:        float
    ego_speed:  float


class SafetyModule:
    """
    Gap and TTC watchdog.  Called by SimulationManager each step.

    Parameters
    ----------
    min_safe_distance : float  Gap threshold in metres (default 2.0 m).
    max_deceleration  : float  Emergency braking rate m/s² (default −4.5).
    ttc_threshold     : float  TTC threshold in seconds  (default 1.5 s).
    override_penalty  : float  Reward penalty on override (default −100).
    """

    def __init__(self,
                 min_safe_distance: float = 2.0,
                 max_deceleration:  float = -4.5,
                 ttc_threshold:     float = 1.5,
                 override_penalty:  float = -100.0):

        self.min_safe_distance = min_safe_distance
        self.max_deceleration  = max_deceleration
        self.ttc_threshold     = ttc_threshold
        self.override_penalty  = override_penalty

        # Counters and history
        self.violation_count: int                  = 0
        self.violations:      list[ViolationEvent] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def check_and_override(self, traci, agent_id: str,
                           gap: float, ego_speed: float,
                           leader_speed: float, step: int) -> tuple[bool, float]:
        """
        Evaluate safety for one vehicle.  If unsafe, apply emergency braking
        directly via TraCI and log the violation.

        Parameters
        ----------
        traci       : TraCI module (passed in to avoid import cycles).
        agent_id    : str   SUMO vehicle ID (e.g. 'CAV_1').
        gap         : float Current gap to leader in metres.
        ego_speed   : float Current speed of this vehicle in m/s.
        leader_speed: float Current speed of the leader vehicle in m/s.
        step        : int   Current simulation step number.

        Returns
        -------
        (override_triggered: bool, reward_penalty: float)
            override_triggered = True  → safety took control, AI skipped.
            reward_penalty     = override_penalty if triggered, else 0.0.
        """
        ttc = self._compute_ttc(gap, ego_speed, leader_speed)
        unsafe = self._is_unsafe(gap, ttc)

        if not unsafe:
            return False, 0.0

        # ── Apply emergency braking ───────────────────────────────────────────
        try:
            traci.vehicle.setSpeed(agent_id, 0.0)
        except Exception:
            pass  # vehicle may have already left network

        self._log_violation(step, agent_id, gap, ttc, ego_speed)
        return True, self.override_penalty

    def is_safe(self, gap: float, ego_speed: float, leader_speed: float) -> bool:
        """
        Pure check with no side effects.  Useful for unit tests.

        Returns True if the situation is SAFE (no override needed).
        """
        ttc = self._compute_ttc(gap, ego_speed, leader_speed)
        return not self._is_unsafe(gap, ttc)

    def reset(self) -> None:
        """Reset counters at the start of each episode."""
        self.violation_count = 0
        self.violations.clear()

    def summary(self) -> dict:
        """Return a summary dict for the DataLogger end-of-episode report."""
        return {
            "total_violations": self.violation_count,
            "violation_events": [
                {"step": v.step, "vehicle": v.vehicle_id,
                 "gap": round(v.gap, 2), "ttc": round(v.ttc, 3)}
                for v in self.violations
            ]
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_ttc(self, gap: float, ego_speed: float,
                     leader_speed: float) -> float:
        """
        Time-To-Collision in seconds.
        Returns float('inf') when ego is NOT closing on leader.
        """
        rel_speed = ego_speed - leader_speed   # positive = closing
        if rel_speed > 0.001:
            return gap / rel_speed
        return float("inf")

    def _is_unsafe(self, gap: float, ttc: float) -> bool:
        """True when EITHER safety condition is breached."""
        return (gap < self.min_safe_distance) or (ttc < self.ttc_threshold)

    def _log_violation(self, step: int, vehicle_id: str,
                       gap: float, ttc: float, ego_speed: float) -> None:
        """Record violation and print tagged console message."""
        self.violation_count += 1
        event = ViolationEvent(step=step, vehicle_id=vehicle_id,
                               gap=gap, ttc=ttc, ego_speed=ego_speed)
        self.violations.append(event)

        ttc_str = f"{ttc:.2f}s" if ttc != float("inf") else "∞"
        print(
            f"[SAFETY]  Step {step:>5} | {vehicle_id} | "
            f"gap={gap:.2f}m  TTC={ttc_str}  ego={ego_speed:.1f}m/s "
            f"→ OVERRIDE (total #{self.violation_count})"
        )
