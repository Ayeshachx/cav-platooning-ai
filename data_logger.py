"""
data_logger.py
CAV Platooning Project – DataLogger

Buffers per-step vehicle metrics in memory (Pandas DataFrame) and
flushes to a timestamped CSV file periodically and on episode end.

Also generates a plain-text summary report at the end of each run.

SDD references: §4.2.5, §3.2.1, §3.2 (file schema).
SRS references: FR-13, FR-14, FR-15, FR-22.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

from agent import StepData


class DataLogger:
    """
    Collects StepData records, buffers them, and flushes to CSV.

    Parameters
    ----------
    output_dir      : str   Directory for CSV and summary files.
    flush_interval  : int   Flush buffer to disk every N records.
    """

    COLUMNS = [
        "timestamp", "vehicle_id", "x", "y",
        "speed", "gap_to_leader", "relative_speed",
        "ai_action", "target_speed", "safety_override",
        "reward", "fuel_consumption",
    ]

    def __init__(self, output_dir: str = "output/", flush_interval: int = 1000):
        self.output_dir     = Path(output_dir)
        self.flush_interval = flush_interval

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique timestamped filenames for this run
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._csv_path     = self.output_dir / f"simulation_results_{ts}.csv"
        self._summary_path = self.output_dir / f"summary_{ts}.txt"

        # In-memory buffer
        self._buffer: List[dict] = []
        self._total_rows         = 0
        self._csv_initialized    = False

        # Episode-level accumulators (reset each episode)
        self._ep_rewards:    List[float] = []
        self._ep_gaps:       List[float] = []
        self._ep_speeds:     List[float] = []
        self._ep_overrides:  int         = 0
        self._ep_collisions: int         = 0
        self._ep_start_time: float       = time.time()

        print(f"[INFO]   DataLogger initialised → {self._csv_path.name}")

    # ── Per-step logging ──────────────────────────────────────────────────────

    def log_step(self, step_data: StepData, sim_time: float) -> None:
        """
        Append one StepData record to the buffer.

        Parameters
        ----------
        step_data : StepData  Snapshot from VehicleAgent.snapshot().
        sim_time  : float     SUMO simulation time in seconds.
        """
        row = {
            "timestamp":       round(sim_time, 2),
            "vehicle_id":      step_data.vehicle_id,
            "x":               step_data.x,
            "y":               step_data.y,
            "speed":           step_data.speed,
            "gap_to_leader":   step_data.gap_to_leader,
            "relative_speed":  step_data.relative_speed,
            "ai_action":       step_data.ai_action,
            "target_speed":    step_data.target_speed,
            "safety_override": int(step_data.safety_override),
            "reward":          step_data.reward,
            "fuel_consumption":step_data.fuel_consumption,
        }
        self._buffer.append(row)
        self._total_rows += 1

        # Episode accumulators
        if step_data.gap_to_leader > 0:
            self._ep_gaps.append(step_data.gap_to_leader)
        self._ep_speeds.append(step_data.speed)
        self._ep_rewards.append(step_data.reward)
        if step_data.safety_override:
            self._ep_overrides += 1

        # Auto-flush
        if len(self._buffer) >= self.flush_interval:
            self.flush_to_csv()

    def log_collision(self) -> None:
        """Record a collision event."""
        self._ep_collisions += 1

    # ── CSV flushing ──────────────────────────────────────────────────────────

    def flush_to_csv(self) -> None:
        """
        Write buffer to CSV.  Appends if file already exists (first write
        includes header).  Retries with _v2 suffix if PermissionError.
        (SRS FR-15)
        """
        if not self._buffer:
            return

        path = self._csv_path
        for suffix in ["", "_v2", "_v3", "_v4"]:
            if suffix:
                stem = self._csv_path.stem + suffix
                path = self._csv_path.with_stem(stem)
            try:
                self._write_rows(path)
                if suffix:
                    # Update path for future flushes
                    self._csv_path = path
                self._buffer.clear()
                return
            except PermissionError:
                print(f"[WARNING] CSV locked: {path.name}, trying {path.stem}_v2 ...")
                continue

        print(f"[ERROR]  Could not write CSV after 4 attempts — data lost for this flush.")

    def _write_rows(self, path: Path) -> None:
        """Low-level CSV write — appends rows, writes header on first call."""
        write_header = not path.exists() or not self._csv_initialized

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerows(self._buffer)

    # ── Summary report ────────────────────────────────────────────────────────

    def generate_summary_report(self,
                                 episode:        int,
                                 total_steps:    int,
                                 sim_duration:   float,
                                 violation_count: int,
                                 config:         dict) -> str:
        """
        Write a human-readable summary .txt file and return its path.
        (SRS FR-22)
        """
        # Flush any remaining data first
        self.flush_to_csv()

        elapsed = time.time() - self._ep_start_time
        avg_speed  = sum(self._ep_speeds)  / max(len(self._ep_speeds), 1)
        avg_gap    = sum(self._ep_gaps)    / max(len(self._ep_gaps),   1)
        avg_reward = sum(self._ep_rewards) / max(len(self._ep_rewards),1)
        total_rows = self._total_rows

        lines = [
            "=" * 62,
            "  CAV PLATOONING SIMULATION – SUMMARY REPORT",
            "=" * 62,
            f"  Episode        : {episode}",
            f"  Total Steps    : {total_steps}",
            f"  Sim Duration   : {sim_duration:.1f} s (simulated)",
            f"  Wall Clock     : {elapsed:.1f} s",
            "-" * 62,
            "  PERFORMANCE",
            f"    Avg Platoon Speed : {avg_speed:.2f} m/s  "
            f"({avg_speed * 3.6:.1f} km/h)",
            f"    Avg Inter-Gap     : {avg_gap:.2f} m",
            f"    Avg Step Reward   : {avg_reward:.3f}",
            f"    Total Data Rows   : {total_rows}",
            "-" * 62,
            "  SAFETY",
            f"    Override Events   : {violation_count}",
            f"    Collisions        : {self._ep_collisions}",
            "-" * 62,
            "  CONFIGURATION",
            f"    AI Mode           : {config['ai']['mode']}",
            f"    Num Vehicles      : {config['simulation']['num_vehicles']}",
            f"    Target Speed      : {config['simulation']['target_speed']} m/s",
            f"    Min Safe Distance : {config['safety']['min_safe_distance']} m",
            f"    TTC Threshold     : {config['safety']['ttc_threshold']} s",
            "-" * 62,
            f"  CSV Output : {self._csv_path.name}",
            "=" * 62,
        ]

        report = "\n".join(lines)

        with open(self._summary_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n{report}\n")
        print(f"[INFO]   Summary saved → {self._summary_path.name}")
        return str(self._summary_path)

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset per-episode accumulators for a new episode."""
        self.flush_to_csv()
        self._ep_rewards    = []
        self._ep_gaps       = []
        self._ep_speeds     = []
        self._ep_overrides  = 0
        self._ep_collisions = 0
        self._ep_start_time = time.time()
        self._csv_initialized = False   # new episode → new CSV block with header
