"""
main.py
CAV Platooning Project – SimulationManager & Entry Point

Orchestrates the complete simulation lifecycle:
    1. Read config.yaml
    2. Launch SUMO via TraCI
    3. Spawn VehicleAgent objects
    4. Run the per-step control loop
    5. Safety checks, AI decisions, data logging
    6. Clean shutdown

Usage:
    python main.py                     # uses config.yaml in same directory
    python main.py --config my.yaml    # custom config file
    python main.py --headless          # force headless (no GUI)
    python main.py --episodes 3        # override episode count

SDD references: §4.2.1 (SimulationManager), §5.4.1 (pseudocode).
SRS references: FR-01 to FR-04, FR-19, FR-20, FR-21.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml

# ── TraCI import (requires SUMO_HOME to be set) ───────────────────────────────
try:
    if "SUMO_HOME" not in os.environ:
        # Try common install locations automatically
        candidates = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            r"C:\Sumo",
            "/usr/share/sumo",
            "/usr/local/share/sumo",
            os.path.expanduser("~/sumo"),
        ]
        for c in candidates:
            if Path(c).exists():
                os.environ["SUMO_HOME"] = c
                break

    if "SUMO_HOME" not in os.environ:
        print("[FATAL]  SUMO_HOME environment variable not set.")
        print("         Set it to your SUMO installation directory, e.g.:")
        print("         Windows: set SUMO_HOME=C:\\Program Files\\Eclipse\\Sumo")
        print("         Linux:   export SUMO_HOME=/usr/share/sumo")
        sys.exit(1)

    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import traci
    import traci.constants as tc

except ImportError as e:
    print(f"[FATAL]  Could not import TraCI: {e}")
    print("         Ensure SUMO is installed and SUMO_HOME is correct.")
    sys.exit(1)

from agent import VehicleAgent
from safety_module import SafetyModule
from data_logger import DataLogger
from ai_interface import AbstractAIModel
from ai_rule_based import RuleBasedController


# ─────────────────────────────────────────────────────────────────────────────
class SimulationManager:
    """
    Central orchestrator for the CAV platooning simulation.

    Parameters
    ----------
    config      : dict   Parsed contents of config.yaml.
    headless    : bool   If True, force sumo (no GUI) regardless of config.
    """

    def __init__(self, config: dict, headless: bool = False):
        self.config   = config
        self.headless = headless

        # Resolve paths relative to project root (directory of main.py)
        self._root      = Path(__file__).parent
        self._sumo_cfg  = self._root / config["sumo"]["config_file"]
        self._net_file  = self._root / config["sumo"]["net_file"]
        self._port      = int(config["sumo"].get("port", 8813))
        self._timeout   = int(config["sumo"].get("launch_timeout", 30))

        # Validate required files exist before launching SUMO
        self._validate_files()

        # Subsystems
        self.safety = SafetyModule(
            min_safe_distance = config["safety"]["min_safe_distance"],
            max_deceleration  = config["safety"]["max_deceleration"],
            ttc_threshold     = config["safety"]["ttc_threshold"],
            override_penalty  = config["safety"]["override_penalty"],
        )
        self.logger = DataLogger(
            output_dir     = str(self._root / config["logging"]["output_dir"]),
            flush_interval = config["logging"]["flush_interval"],
        )

        # Runtime state
        self.agents:      List[VehicleAgent] = []
        self.step_count:  int   = 0
        self.sim_time:    float = 0.0
        self.is_running:  bool  = False
        self._episode:    int   = 0

        # AI model factory (shared instance across all follower agents)
        self._ai_model: AbstractAIModel = self._build_ai_model()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Launch SUMO, open TraCI connection, spawn CAV vehicles.
        SRS FR-01, FR-02.
        """
        use_gui = self.config["simulation"].get("gui", True)

        # On Windows sumo.exe often refuses TraCI connections.
        # Always use sumo-gui.exe — just minimise the window if you don't need it.
        if sys.platform == "win32":
            binary = "sumo-gui.exe"
        else:
            binary = "sumo-gui" if use_gui else "sumo"

        sumo_bin = self._find_sumo_binary(binary)

        cmd = [
            sumo_bin,
            "-c", str(self._sumo_cfg),
            "--remote-port", str(self._port),
            "--start",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--collision.action", "remove",
        ]

        if not use_gui:
            cmd += ["--no-warnings", "true"]

        print(f"[INFO]   Launching SUMO: {binary}")
        print(f"[INFO]   Config: {self._sumo_cfg.name}")

        # Launch SUMO as a subprocess
        sumo_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for SUMO to accept the TraCI connection
        connected = False
        deadline  = time.time() + self._timeout
        last_err  = None

        while time.time() < deadline:
            try:
                traci.init(self._port)
                connected = True
                break
            except Exception as e:
                last_err = e
                time.sleep(0.5)

        if not connected:
            sumo_proc.terminate()
            print(f"[FATAL]  SUMO launch timeout after {self._timeout}s.")
            print(f"         Last error: {last_err}")
            sys.exit(1)

        print(f"[INFO]   TraCI connected on port {self._port}.")
        self.is_running = True

        # Spawn agents (vehicles are defined in platoon.rou.xml;
        # we just create Python-side wrappers here — SUMO inserts them
        # at their depart times)
        self._spawn_agents()
        print(f"[INFO]   {len(self.agents)} VehicleAgent objects created.")
        print(f"[INFO]   Simulation ready. Starting episode {self._episode + 1}...\n")

    def run_episode(self) -> dict:
        """
        Execute one complete simulation episode.

        Returns
        -------
        dict  Episode summary statistics.
        """
        self._episode    += 1
        self.step_count   = 0
        self.sim_time     = 0.0
        collision_occurred = False

        self.safety.reset()
        self.logger.reset_episode()
        for agent in self.agents:
            agent.reset()

        step_size    = self.config["simulation"]["step_size"]
        max_duration = self.config["simulation"]["duration"]
        console_ivl  = self.config["logging"].get("console_interval", 50)

        print(f"[INFO]   ── Episode {self._episode} ──────────────────────────")

        while self.sim_time < max_duration:
            # ── 1. Advance SUMO physics one step ─────────────────────────────
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                print("[ERROR]  SUMO connection lost mid-episode.")
                break

            self.step_count += 1
            self.sim_time    = round(self.step_count * step_size, 2)

            # ── 2. Update all agent states from TraCI ─────────────────────────
            active_ids = set(traci.vehicle.getIDList())
            for agent in self.agents:
                if agent.vehicle_id in active_ids:
                    agent.update_state(traci)

            # ── 3. Safety check + AI action for each follower ─────────────────
            for agent in self.agents:
                if agent.vehicle_id not in active_ids:
                    continue

                # Safety module: only check FOLLOWERS (leader has no vehicle ahead)
                if agent.role == "follower" and agent.gap_to_leader > 0:
                    override, penalty = self.safety.check_and_override(
                        traci        = traci,
                        agent_id     = agent.vehicle_id,
                        gap          = agent.gap_to_leader,
                        ego_speed    = agent.speed,
                        leader_speed = agent.leader_speed,
                        step         = self.step_count,
                    )
                else:
                    override, penalty = False, 0.0

                # AI / rule-based decision & actuation
                agent.decide_and_act(
                    traci                  = traci,
                    safety_override_active = override,
                    override_target_speed  = 0.0,
                )

                # Reward
                reward = agent.compute_reward(override)
                if override:
                    reward = penalty   # use safety module's penalty

                # Log this step
                snap = agent.snapshot(self.step_count, reward)
                self.logger.log_step(snap, self.sim_time)

            # ── 4. Collision detection ────────────────────────────────────────
            collisions = traci.simulation.getCollisions()
            if collisions:
                for col in collisions:
                    print(
                        f"[ERROR]  Collision: {col.collider} ↔ {col.victim} "
                        f"at step {self.step_count}"
                    )
                self.logger.log_collision()
                collision_occurred = True
                break   # terminate episode immediately (SRS FR-21)

            # ── 5. Console progress ───────────────────────────────────────────
            if console_ivl > 0 and self.step_count % console_ivl == 0:
                self._print_progress()

            # ── 6. Check if all CAVs have departed network ────────────────────
            cav_ids = {a.vehicle_id for a in self.agents}
            active_cavs = cav_ids.intersection(active_ids)
            if cav_ids and not active_cavs and self.sim_time > 10.0:
                print(f"[INFO]   All CAVs reached destination at t={self.sim_time}s")
                break
            elif cav_ids and not active_cavs and self.sim_time <= 10.0:
                # Vehicles disappeared too early - likely network issue, continue
                pass

        # ── Episode end ───────────────────────────────────────────────────────
        summary = self._finish_episode(collision_occurred)
        return summary

    def stop(self) -> None:
        """
        Flush all data, generate summary, close TraCI connection.
        SRS FR-20.
        """
        try:
            traci.close()
            print("[INFO]   TraCI connection closed.")
        except Exception:
            pass
        self.is_running = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_files(self) -> None:
        """Check all required input files exist before launching. SRS FR-03."""
        required = {
            "SUMO config":  self._sumo_cfg,
            "Network file": self._net_file,
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            for name in missing:
                print(f"[FATAL]  {name} not found: {required[name]}")
            print()
            print("         Run  python sumo_network/build_network.py  first")
            print("         to generate highway.net.xml from the source files.")
            sys.exit(1)

    def _find_sumo_binary(self, binary: str) -> str:
        """Locate the SUMO binary in SUMO_HOME/bin or system PATH."""
        sumo_home = os.environ.get("SUMO_HOME", "")
        candidates = [
            os.path.join(sumo_home, "bin", binary),
            binary,   # system PATH
        ]
        for c in candidates:
            if Path(c).exists() or _command_exists(c):
                return c
        print(f"[FATAL]  SUMO binary not found: {binary}")
        print(f"         SUMO_HOME = {sumo_home}")
        sys.exit(1)

    def _build_ai_model(self) -> AbstractAIModel:
        """Instantiate the AI model specified in config.yaml."""
        mode = self.config["ai"].get("mode", "rule_based")
        ai_cfg = self.config["ai"]

        if mode == "rule_based":
            model = RuleBasedController(
                target_gap     = ai_cfg.get("target_gap",     10.0),
                kp             = ai_cfg.get("kp",              0.5),
                max_correction = ai_cfg.get("max_correction",  3.0),
                max_speed      = self.config["simulation"]["max_speed"],
                speed_delta    = ai_cfg.get("speed_delta",     2.0),
            )
            print(f"[INFO]   AI mode: rule_based (proportional gap controller)")
            return model

        elif mode == "dqn":
            try:
                from ai_dqn import DQN_Solver
                model = DQN_Solver(
                    state_dim  = ai_cfg.get("state_dim",  4),
                    action_dim = ai_cfg.get("action_dim", 3),
                    config     = ai_cfg,
                )
                # Auto-load best or final model if it exists
                model_dir  = self._root / self.config["logging"]["model_dir"]
                best_path  = model_dir / "cav_model_best.pth"
                final_path = model_dir / "cav_model_final.pth"
                if best_path.exists():
                    model.load_model(str(best_path))
                    print(f"[INFO]   AI mode: dqn  (loaded {best_path.name})")
                elif final_path.exists():
                    model.load_model(str(final_path))
                    print(f"[INFO]   AI mode: dqn  (loaded {final_path.name})")
                else:
                    print("[INFO]   AI mode: dqn  (untrained — run train.py first)")
                return model
            except ImportError as e:
                print(f"[WARNING] DQN import failed: {e}")
                print("         Falling back to rule_based.")
                return RuleBasedController()

        else:
            print(f"[WARNING] Unknown AI mode '{mode}', falling back to rule_based.")
            return RuleBasedController()

    def _spawn_agents(self) -> None:
        """Create VehicleAgent wrappers for all configured CAV vehicles."""
        n = self.config["simulation"]["num_vehicles"]
        self.agents = []

        for i in range(n):
            vid  = f"CAV_{i}"
            role = "leader" if i == 0 else "follower"
            agent = VehicleAgent(
                vehicle_id = vid,
                ai_model   = self._ai_model,
                role       = role,
                config     = self.config,
            )
            self.agents.append(agent)

        print(f"[INFO]   Agents: {[a.vehicle_id for a in self.agents]}")
        print(f"[INFO]   Leader = CAV_0 (cruises at "
              f"{self.config['simulation']['target_speed']} m/s)")

    def _print_progress(self) -> None:
        """Print a one-line console status update."""
        active = set(traci.vehicle.getIDList())
        cavs   = [a for a in self.agents if a.vehicle_id in active]
        if not cavs:
            return

        leader = next((a for a in cavs if a.role == "leader"), cavs[0])
        follower_gaps = [
            a.gap_to_leader for a in cavs
            if a.role == "follower" and a.gap_to_leader > 0
        ]
        avg_gap = sum(follower_gaps) / len(follower_gaps) if follower_gaps else 0.0

        print(
            f"[INFO]   t={self.sim_time:>6.1f}s | "
            f"step={self.step_count:>5} | "
            f"leader_speed={leader.speed:>5.1f}m/s | "
            f"avg_gap={avg_gap:>5.1f}m | "
            f"violations={self.safety.violation_count}"
        )

    def _finish_episode(self, collision: bool) -> dict:
        """Generate summary, flush data, return stats dict."""
        summary = self.logger.generate_summary_report(
            episode         = self._episode,
            total_steps     = self.step_count,
            sim_duration    = self.sim_time,
            violation_count = self.safety.violation_count,
            config          = self.config,
        )

        result = {
            "episode":         self._episode,
            "steps":           self.step_count,
            "sim_time":        self.sim_time,
            "violations":      self.safety.violation_count,
            "collision":       collision,
            "avg_agent_reward": sum(a.total_reward for a in self.agents)
                                / max(len(self.agents), 1),
        }

        status = "COLLISION" if collision else "COMPLETED"
        print(f"[INFO]   Episode {self._episode} {status} | "
              f"steps={self.step_count} | "
              f"violations={self.safety.violation_count} | "
              f"collision={collision}")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _command_exists(cmd: str) -> bool:
    """Return True if cmd is findable on the system PATH."""
    import shutil
    return shutil.which(cmd) is not None


def load_config(path: str) -> dict:
    """Parse config.yaml and return as a dict."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"[FATAL]  Config file not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CAV Platooning Simulation – Rule-Based Controller"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without SUMO GUI (faster)"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override number of episodes from config"
    )
    args = parser.parse_args()

    # ── Load configuration ────────────────────────────────────────────────────
    config = load_config(args.config)

    if args.episodes is not None:
        config["simulation"]["num_episodes"] = args.episodes

    # Apply random seed
    import random, numpy as np
    seed = config["simulation"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    print(f"[INFO]   Seed: {seed}")

    # ── Create and start simulation ───────────────────────────────────────────
    manager = SimulationManager(config, headless=args.headless)

    try:
        manager.start()
        num_episodes = config["simulation"]["num_episodes"]

        all_results = []
        for ep in range(num_episodes):
            result = manager.run_episode()
            all_results.append(result)

            # Brief pause between episodes so SUMO reloads cleanly
            if ep < num_episodes - 1:
                time.sleep(0.5)
                try:
                    traci.load([
                        "-c", str(manager._sumo_cfg),
                        "--start",
                    ])
                except Exception:
                    # If reload fails, restart SUMO entirely
                    manager.stop()
                    manager.start()

        print(f"\n[INFO]   All {num_episodes} episode(s) complete.")

    except KeyboardInterrupt:
        print("\n[INFO]   Interrupted by user.")

    finally:
        manager.stop()


if __name__ == "__main__":
    main()
