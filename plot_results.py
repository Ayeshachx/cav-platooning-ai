"""
plot_results.py
CAV Platooning Project – Results Visualiser

Reads the most recent (or specified) simulation_results CSV and
generates four performance plots:

    1. Speed over time for all CAVs
    2. Inter-vehicle gap over time (followers only)
    3. Reward signal over time (followers only)
    4. Safety override events timeline

Usage:
    python plot_results.py                         # auto-find latest CSV
    python plot_results.py --csv output/my.csv     # specific file
    python plot_results.py --save                  # save PNG instead of display
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Missing library: {e}")
    print("        Run:  pip install pandas matplotlib numpy")
    sys.exit(1)


def find_latest_csv(output_dir: str = "output/") -> Path:
    """Find the most recently modified CSV in the output directory."""
    output_path = Path(output_dir)
    csvs = sorted(output_path.glob("simulation_results_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        print(f"[ERROR] No simulation CSV files found in {output_dir}")
        sys.exit(1)
    return csvs[0]


def plot_results(csv_path: Path, save: bool = False) -> None:
    """Load CSV and render four-panel performance dashboard."""
    print(f"[INFO]  Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("[ERROR] CSV file is empty.")
        sys.exit(1)

    # Force numeric types - guard against string parsing
    for _col in ["timestamp", "speed", "gap_to_leader", "relative_speed",
                 "ai_action", "target_speed", "safety_override",
                 "reward", "fuel_consumption"]:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")

    print(f"[INFO]  Rows: {len(df):,}  |  "
          f"Vehicles: {df['vehicle_id'].nunique()}  |  "
          f"Duration: {float(df['timestamp'].max()):.1f}s")

    vehicle_ids = sorted(df["vehicle_id"].unique())
    followers   = [v for v in vehicle_ids if v != "CAV_0"]
    times       = df[df["vehicle_id"] == vehicle_ids[0]]["timestamp"].values

    # ── Colour palette ────────────────────────────────────────────────────────
    palette = {
        "CAV_0": "#C00000",   # leader = red
        "CAV_1": "#2E75B6",
        "CAV_2": "#1F7870",
        "CAV_3": "#7030A0",
        "CAV_4": "#E36C09",
    }
    def col(vid): return palette.get(vid, "#595959")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#1A1A2E")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.28,
                           left=0.07, right=0.97,
                           top=0.91, bottom=0.07)

    ax_speed  = fig.add_subplot(gs[0, 0])
    ax_gap    = fig.add_subplot(gs[0, 1])
    ax_reward = fig.add_subplot(gs[1, 0])
    ax_safety = fig.add_subplot(gs[1, 1])

    PANEL_BG   = "#16213E"
    GRID_COLOR = "#2A2A4A"
    TEXT_COLOR = "#E0E0E0"

    for ax in [ax_speed, ax_gap, ax_reward, ax_safety]:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3A3A5A")

    # ── Panel 1: Speed over time ──────────────────────────────────────────────
    for vid in vehicle_ids:
        sub = df[df["vehicle_id"] == vid]
        lw  = 2.2 if vid == "CAV_0" else 1.4
        ax_speed.plot(sub["timestamp"], sub["speed"],
                      label=vid, color=col(vid),
                      linewidth=lw, alpha=0.9)

    ax_speed.set_title("Vehicle Speed over Time", fontweight="bold", fontsize=11)
    ax_speed.set_xlabel("Simulation Time (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.legend(fontsize=8, facecolor=PANEL_BG,
                    edgecolor="#3A3A5A", labelcolor=TEXT_COLOR)
    # Target speed reference line
    if "target_speed" in df.columns:
        ts = df["target_speed"].median()
        ax_speed.axhline(ts, color="#FFC107", linewidth=1.0,
                         linestyle=":", alpha=0.6, label="target")

    # ── Panel 2: Gap over time (followers only) ───────────────────────────────
    for vid in followers:
        sub = df[(df["vehicle_id"] == vid) & (df["gap_to_leader"] > 0)]
        if sub.empty:
            continue
        ax_gap.plot(sub["timestamp"], sub["gap_to_leader"],
                    label=vid, color=col(vid), linewidth=1.4, alpha=0.9)

    ax_gap.axhspan(0,    2,  color="#C00000", alpha=0.15, label="Danger (<2m)")
    ax_gap.axhspan(2,    5,  color="#E36C09", alpha=0.12, label="Caution (2–5m)")
    ax_gap.axhspan(5,    15, color="#375623", alpha=0.10, label="Target (5–15m)")
    ax_gap.axhspan(15,   30, color="#2E75B6", alpha=0.08, label="Warning (15–30m)")

    ax_gap.set_title("Inter-Vehicle Gap over Time", fontweight="bold", fontsize=11)
    ax_gap.set_xlabel("Simulation Time (s)")
    ax_gap.set_ylabel("Gap to Leader (m)")
    ax_gap.set_ylim(bottom=0)
    ax_gap.legend(fontsize=7.5, facecolor=PANEL_BG,
                  edgecolor="#3A3A5A", labelcolor=TEXT_COLOR, loc="upper right")

    # ── Panel 3: Reward over time (rolling mean) ──────────────────────────────
    for vid in followers:
        sub = df[df["vehicle_id"] == vid].copy()
        if sub.empty:
            continue
        sub["reward_smooth"] = sub["reward"].rolling(window=20, min_periods=1).mean()
        ax_reward.plot(sub["timestamp"], sub["reward_smooth"],
                       label=vid, color=col(vid), linewidth=1.4, alpha=0.9)

    ax_reward.axhline(0, color="#E0E0E0", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_reward.set_title("Step Reward over Time (20-step rolling mean)",
                         fontweight="bold", fontsize=11)
    ax_reward.set_xlabel("Simulation Time (s)")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(fontsize=8, facecolor=PANEL_BG,
                     edgecolor="#3A3A5A", labelcolor=TEXT_COLOR)

    # ── Panel 4: Safety override events ──────────────────────────────────────
    for i, vid in enumerate(vehicle_ids):
        sub = df[(df["vehicle_id"] == vid) & (df["safety_override"] == 1)]
        if sub.empty:
            continue
        ax_safety.scatter(sub["timestamp"],
                          [i] * len(sub),
                          color=col(vid), marker="|",
                          s=80, linewidths=1.2, label=vid, alpha=0.8)

    total_overrides = int(df["safety_override"].sum())
    ax_safety.set_title(
        f"Safety Override Events  (total: {total_overrides})",
        fontweight="bold", fontsize=11)
    ax_safety.set_xlabel("Simulation Time (s)")
    ax_safety.set_yticks(range(len(vehicle_ids)))
    ax_safety.set_yticklabels(vehicle_ids, fontsize=9)
    ax_safety.set_ylabel("Vehicle")
    if total_overrides == 0:
        ax_safety.text(0.5, 0.5, "✓  No safety overrides",
                       transform=ax_safety.transAxes,
                       ha="center", va="center",
                       color="#4CAF50", fontsize=14, fontweight="bold")

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        f"CAV Platooning Simulation – Performance Dashboard\n"
        f"{csv_path.name}",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Save or show ──────────────────────────────────────────────────────────
    if save:
        out_path = csv_path.with_suffix(".png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[INFO]  Plot saved → {out_path.name}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot CAV platooning simulation results from CSV"
    )
    parser.add_argument("--csv",  default=None,
                        help="Path to CSV file (default: latest in output/)")
    parser.add_argument("--save", action="store_true",
                        help="Save plot as PNG instead of displaying")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv()
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    plot_results(csv_path, save=args.save)


if __name__ == "__main__":
    main()
