"""
train.py
CAV Platooning Project – DQN Training Script (Fixed)

Usage:
    python train.py                        # 200 episodes default
    python train.py --episodes 500
    python train.py --resume models/cav_model_ep200.pth
    python train.py --episodes 100 --eval
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except ImportError:
    print("[FATAL]  PyTorch not installed.  Run:  pip install torch")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False

from ai_dqn import DQN_Solver
from gym_env import CAVPlatooning


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config: dict, num_episodes: int,
          resume_path: str = None, eval_after: bool = False) -> None:

    import random
    seed = config["simulation"].get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model_dir = Path(__file__).parent / config["logging"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_interval = int(config["logging"].get("checkpoint_interval", 100))

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = DQN_Solver(
        state_dim  = config["ai"]["state_dim"],
        action_dim = config["ai"]["action_dim"],
        config     = config["ai"],
    )
    if resume_path:
        agent.load_model(resume_path)
        print(f"[INFO]   Resumed from {resume_path}")

    # ── Build environment ─────────────────────────────────────────────────────
    env = CAVPlatooning(
        config   = config,
        agent_id = "CAV_1",
        port     = config["sumo"].get("port", 8813),
        headless = True,
    )

    max_speed = config["simulation"]["max_speed"]

    print(f"\n[INFO]   DQN training — {num_episodes} episodes")
    print(f"[INFO]   Episode length: "
          f"{config['simulation'].get('train_episode_steps', 500)} steps")
    print(f"[INFO]   Buffer: {config['ai']['replay_buffer_size']}  "
          f"Batch: {config['ai']['batch_size']}  "
          f"γ={config['ai']['gamma']}")
    print("=" * 68)

    ep_rewards   = []
    ep_gaps      = []
    ep_violations= []
    ep_collisions= []
    best_avg     = -float("inf")
    t0           = time.time()

    for ep in range(1, num_episodes + 1):

        obs, _ = env.reset()
        ep_r   = 0.0
        ep_v   = 0
        col    = False
        gaps_this_ep = []

        target_gap = config["ai"].get("target_gap", 10.0)

        while True:
            # DQN_Solver.predict() expects RAW state, so we denormalise
            raw = _denorm(obs, max_speed)
            gap = raw[0]

            # Guided exploration: when epsilon is high AND gap is large,
            # bias 60% toward ACCELERATE so agent sees gap-closing rewards
            import random as _rnd
            if (agent.epsilon > 0.3
                    and gap > target_gap * 1.2
                    and _rnd.random() < 0.6):
                action = 0   # ACCELERATE — teach that closing gap = reward
            else:
                action = agent.predict(raw)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            raw_next = _denorm(next_obs, max_speed)
            agent.store(raw, action, reward, raw_next, done)
            agent.train_step()

            ep_r += reward
            g     = info.get("gap", 0.0)
            if g > 0:
                gaps_this_ep.append(g)
            if info.get("override"):
                ep_v += 1
            if info.get("collision"):
                col = True

            obs = next_obs
            if done:
                break

        agent.decay_epsilon()

        ep_rewards.append(ep_r)
        ep_violations.append(ep_v)
        ep_collisions.append(int(col))
        avg_gap = float(np.mean(gaps_this_ep)) if gaps_this_ep else 0.0
        ep_gaps.append(avg_gap)

        # Console every episode
        avg20   = float(np.mean(ep_rewards[-20:]))
        elapsed = time.time() - t0
        status  = "COLLISION" if col else "ok       "
        print(
            f"[INFO]   Ep {ep:>4}/{num_episodes} | {status} | "
            f"reward={ep_r:>8.1f} | avg20={avg20:>8.1f} | "
            f"gap={avg_gap:>5.1f}m | ε={agent.epsilon:.3f} | "
            f"viol={ep_v:>3} | loss={agent.mean_loss:.4f} | t={elapsed:.0f}s"
        )

        # Checkpoint
        if ep % ckpt_interval == 0:
            p = model_dir / f"cav_model_ep{ep}.pth"
            agent.save_model(str(p))
            if avg20 > best_avg:
                best_avg = avg20
                agent.save_model(str(model_dir / "cav_model_best.pth"))
                print(f"[INFO]   ★ New best avg20={avg20:.1f} → best model saved")

    # ── Done ──────────────────────────────────────────────────────────────────
    env.close()
    total_min = (time.time() - t0) / 60

    agent.save_model(str(model_dir / "cav_model_final.pth"))

    print("\n" + "=" * 68)
    print(f"  Training complete in {total_min:.1f} minutes")
    print(f"  Final ε          : {agent.epsilon:.4f}")
    print(f"  Best avg20 reward: {best_avg:.2f}")
    print(f"  Avg reward (last 50): {float(np.mean(ep_rewards[-50:])):.2f}")
    print(f"  Avg gap    (last 50): {float(np.mean(ep_gaps[-50:])):.2f} m")
    print(f"  Collision rate (last 50): "
          f"{float(np.mean(ep_collisions[-50:]))*100:.1f}%")
    print(f"  Model → {model_dir / 'cav_model_final.pth'}")
    print("=" * 68)

    if _PLOT:
        _save_curves(ep_rewards, ep_gaps, ep_violations,
                     ep_collisions, model_dir)

    if eval_after:
        evaluate(config, str(model_dir / "cav_model_final.pth"))


def evaluate(config: dict, model_path: str, num_episodes: int = 5) -> None:
    print(f"\n[INFO]   Evaluating: {model_path}")
    agent = DQN_Solver(
        state_dim=config["ai"]["state_dim"],
        action_dim=config["ai"]["action_dim"],
        config=config["ai"],
    )
    agent.load_model(model_path)
    agent.epsilon = 0.0   # pure greedy

    env = CAVPlatooning(config, agent_id="CAV_1",
                        port=config["sumo"].get("port", 8813),
                        headless=True)
    max_speed = config["simulation"]["max_speed"]

    print(f"[INFO]   {num_episodes} evaluation episodes (ε=0, no exploration)")
    results = []
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        r_tot, viol, col = 0.0, 0, False
        while True:
            action = agent.predict(_denorm(obs, max_speed))
            obs, r, term, trunc, info = env.step(action)
            r_tot += r
            if info.get("override"):  viol += 1
            if info.get("collision"): col   = True
            if term or trunc: break
        results.append((r_tot, viol, col))
        print(f"  Ep {ep}: reward={r_tot:.1f}  viol={viol}  "
              f"{'COLLISION' if col else 'OK'}")

    env.close()
    rewards    = [r[0] for r in results]
    violations = [r[1] for r in results]
    collisions = [r[2] for r in results]
    print(f"\n  Mean reward    : {float(np.mean(rewards)):.2f}")
    print(f"  Mean violations: {float(np.mean(violations)):.1f}")
    print(f"  Success rate   : {(1-float(np.mean(collisions)))*100:.0f}%")


def _denorm(obs: np.ndarray, max_speed: float = 33.33) -> np.ndarray:
    """Convert normalised [0,1] obs → raw state for DQN_Solver.predict()."""
    gap    = float(obs[0]) * 50.0
    rel_v  = float(obs[1]) * 2.0 * max_speed - max_speed
    ego_v  = float(obs[2]) * max_speed
    lead_v = float(obs[3]) * max_speed
    return np.array([gap, rel_v, ego_v, lead_v], dtype=np.float32)


def _save_curves(rewards, gaps, violations, collisions, model_dir) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.patch.set_facecolor("#1A1A2E")
    w = min(20, max(1, len(rewards) // 5))

    labels = ["Episode Reward", "Avg Gap (m)",
              "Safety Violations", "Collision Rate (%)"]
    colors = ["#4472C4", "#375623", "#E36C09", "#C00000"]
    data   = [rewards, gaps, violations,
              [c * 100 for c in collisions]]

    for ax, lbl, col, dat in zip(axes, labels, colors, data):
        ax.set_facecolor("#16213E")
        ax.tick_params(colors="#E0E0E0"); ax.title.set_color("#E0E0E0")
        ax.xaxis.label.set_color("#E0E0E0"); ax.yaxis.label.set_color("#E0E0E0")
        ax.grid(True, color="#2A2A4A", alpha=0.5, linestyle="--")
        eps = range(1, len(dat) + 1)
        ax.plot(eps, dat, color=col, alpha=0.3, linewidth=0.8)
        if len(dat) >= w:
            sm = np.convolve(dat, np.ones(w)/w, mode="valid")
            ax.plot(range(w, len(dat)+1), sm, color=col,
                    linewidth=2.0, label=f"{w}-ep mean")
        ax.set_title(lbl, fontweight="bold")
        ax.set_ylabel(lbl)
        if lbl == "Avg Gap (m)":
            ax.axhspan(5, 15, alpha=0.08, color="#375623", label="target zone")
        ax.legend(fontsize=8, facecolor="#16213E",
                  edgecolor="#3A3A5A", labelcolor="#E0E0E0")

    axes[-1].set_xlabel("Episode")
    fig.suptitle("DQN Training Curves", color="#E0E0E0",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = model_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO]   Training curves → {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--eval",     action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["ai"]["mode"] = "dqn"
    num_ep = args.episodes or config["simulation"].get("num_episodes", 200)

    train(config, num_ep, resume_path=args.resume, eval_after=args.eval)


if __name__ == "__main__":
    main()
