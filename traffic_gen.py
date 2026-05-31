"""
traffic_gen.py
CAV Platooning Project – TrafficScenarioGenerator

Dynamically injects rogue (non-CAV) vehicles into the running SUMO
simulation beyond the static vehicles already defined in platoon.rou.xml.

Use this to test the platoon under varying traffic densities and profiles
without editing the route file each time.

SDD references: §4.2 (TrafficScenarioGenerator).
SRS references: FR-16, FR-17, FR-18.
"""

from __future__ import annotations

import random
from typing import Optional


class TrafficScenarioGenerator:
    """
    Injects rogue vehicles at runtime via TraCI.

    Parameters
    ----------
    profile         : str    'passive' or 'aggressive'.
    density         : int    Target vehicles per hour.
    max_speed       : float  Maximum speed for rogue vehicles (m/s).
    seed            : int    Random seed for reproducible injection.
    """

    # Vehicle type IDs must match platoon.rou.xml definitions
    VTYPE_PASSIVE    = "rogue_normal"
    VTYPE_AGGRESSIVE = "rogue_aggr"

    # Routes available for rogue injection (must exist in platoon.rou.xml)
    ROGUE_ROUTES = ["highway_route"]

    # Lanes available for injection (0 = right/slow, 1 = left/fast)
    LANES = [0, 1]

    def __init__(self,
                 profile:   str   = "passive",
                 density:   int   = 200,
                 max_speed: float = 33.33,
                 seed:      int   = 42):

        self.profile   = profile
        self.density   = density      # vehicles / hour
        self.max_speed = max_speed
        self.seed      = seed

        random.seed(seed)

        # Compute mean inter-arrival time in simulation seconds
        # density veh/h → step interval = 3600 / density  seconds
        self._interval  = 3600.0 / max(density, 1)
        self._countdown = self._interval   # seconds until next injection
        self._injected  = 0                # total vehicles injected this episode
        self._skipped   = 0                # insertion failures logged

        # ID counter to keep vehicle IDs unique across episodes
        self._id_counter = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, traci, sim_time: float, step_size: float) -> int:
        """
        Called once per simulation step.  Injects a vehicle if the
        inter-arrival countdown has elapsed.

        Parameters
        ----------
        traci     : TraCI module.
        sim_time  : float  Current simulation time in seconds.
        step_size : float  SUMO step size in seconds (e.g. 0.1).

        Returns
        -------
        int  Number of vehicles injected this step (0 or 1).
        """
        self._countdown -= step_size

        if self._countdown > 0:
            return 0

        # Reset countdown (add jitter ±25% for realistic headway variance)
        jitter = random.uniform(0.75, 1.25)
        self._countdown = self._interval * jitter

        return self._inject_one(traci, sim_time)

    def reset(self) -> None:
        """Reset counters for a new episode."""
        self._countdown  = self._interval
        self._injected   = 0
        self._skipped    = 0

    @property
    def stats(self) -> dict:
        return {"injected": self._injected, "skipped": self._skipped}

    # ── Private ───────────────────────────────────────────────────────────────

    def _inject_one(self, traci, sim_time: float) -> int:
        """
        Try to add one rogue vehicle.  Handles insertion failures gracefully
        (SRS FR-18): logs a warning and continues without terminating.
        """
        vtype = (self.VTYPE_AGGRESSIVE
                 if self.profile == "aggressive"
                 else self.VTYPE_PASSIVE)

        route  = random.choice(self.ROGUE_ROUTES)
        lane   = random.choice(self.LANES)
        speed  = random.uniform(self.max_speed * 0.7, self.max_speed * 0.95)
        vid    = f"dyn_{self._id_counter:05d}"
        self._id_counter += 1

        try:
            traci.vehicle.add(
                vehID       = vid,
                routeID     = route,
                typeID      = vtype,
                depart      = "now",
                departLane  = str(lane),
                departSpeed = str(round(speed, 2)),
                departPos   = "0",
            )
            self._injected += 1
            return 1

        except Exception as e:
            # Insertion can fail if the spawn position is occupied (congestion)
            self._skipped += 1
            if self._skipped <= 5 or self._skipped % 50 == 0:
                print(
                    f"[WARNING] Rogue insertion skipped (t={sim_time:.1f}s, "
                    f"route={route}, lane={lane}): {e}"
                )
            return 0
