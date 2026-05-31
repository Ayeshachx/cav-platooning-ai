# CAV Platooning Simulation
**Agent-Based Simulation of CAV Platooning Using AI**
*Final Year Project – Ayesha Bibi – F22BSEEN1M01132*
*Supervisor: Mam Zara Mansoor – Department of Software Engineering*

---

## Project Overview

A Python simulation of a 5-vehicle Connected and Autonomous Vehicle (CAV)
platoon on a 5 km highway. Vehicles are controlled by a proportional
gap-keeping controller (Phase 1) with a Deep Q-Network (DQN) AI planned
for Phase 2. The simulation runs inside SUMO via the TraCI API.

---

## Project Structure

```
cav_platooning/
├── main.py               ← Entry point: run this to start simulation
├── agent.py              ← VehicleAgent class
├── ai_interface.py       ← AbstractAIModel (plug-in AI interface)
├── ai_rule_based.py      ← Rule-based proportional controller (Phase 1)
├── safety_module.py      ← SafetyModule: gap & TTC watchdog
├── data_logger.py        ← DataLogger: CSV buffering & summary reports
├── traffic_gen.py        ← TrafficScenarioGenerator: rogue injection
├── plot_results.py       ← Plot speed, gap, reward, safety events
├── config.yaml           ← ALL runtime parameters (edit this)
├── requirements.txt      ← Python dependencies
│
├── sumo_network/
│   ├── highway.nod.xml   ← Node definitions (junction points)
│   ├── highway.edg.xml   ← Edge definitions (road segments)
│   ├── highway.con.xml   ← Lane connections
│   ├── platoon.rou.xml   ← Vehicle types + routes + demand
│   ├── platoon.sumocfg   ← Master SUMO config file
│   ├── gui_settings.xml  ← SUMO-GUI visual settings
│   └── build_network.py  ← Run once to generate highway.net.xml
│
├── tests/
│   └── test_safety_and_agent.py  ← Unit tests (no SUMO needed)
│
├── output/               ← Auto-created: CSV logs + summary reports
└── models/               ← Auto-created: AI model checkpoints
```

---

## Quick Start (15 minutes)

### Step 1 – Install SUMO

Download and install SUMO 1.14+:
- **Windows**: https://sumo.dlr.de/docs/Downloads.php → MSI installer
- **Linux**:   `sudo apt-get install sumo sumo-tools sumo-doc`
- **macOS**:   `brew install sumo`

After install, set the environment variable:
```bash
# Windows (Command Prompt)
set SUMO_HOME=C:\Program Files\Eclipse\Sumo

# Windows (PowerShell)
$env:SUMO_HOME = "C:\Program Files\Eclipse\Sumo"

# Linux / macOS
export SUMO_HOME=/usr/share/sumo
```

> **Tip**: Add the export line to your `.bashrc` / `.zshrc` so it persists.

---

### Step 2 – Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 – Build the road network (run ONCE)

```bash
cd sumo_network/
python build_network.py
cd ..
```

This runs `netconvert` to generate `highway.net.xml` from the source files.
You should see:
```
[INFO]  Found netconvert: netconvert
[INFO]  Running netconvert to build highway.net.xml ...
[OK]    highway.net.xml generated successfully (XX KB)
```

---

### Step 4 – Run the simulation

```bash
python main.py
```

The SUMO-GUI window will open showing the 5 CAV vehicles (1 red leader +
4 blue followers) on the highway. Watch them maintain gap using the
proportional controller.

**Other launch options:**
```bash
python main.py --headless          # faster, no GUI window
python main.py --episodes 3        # run 3 episodes back-to-back
python main.py --config my.yaml    # use a different config file
```

---

### Step 5 – View results

After the simulation completes, results are saved to `output/`.

```bash
python plot_results.py             # auto-loads latest CSV + shows plots
python plot_results.py --save      # saves plots as PNG files
```

---

### Step 6 – Run unit tests (no SUMO needed)

```bash
python -m pytest tests/ -v
# or
python tests/test_safety_and_agent.py
```

---

## Configuration

All parameters are in `config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulation.num_vehicles` | 5 | Number of CAV vehicles |
| `simulation.target_speed` | 25.0 m/s | Leader cruise speed (90 km/h) |
| `simulation.gui` | true | Show SUMO-GUI window |
| `simulation.duration` | 600 s | Simulation length per episode |
| `safety.min_safe_distance` | 2.0 m | Safety override gap threshold |
| `safety.ttc_threshold` | 1.5 s | TTC override threshold |
| `ai.mode` | rule_based | `rule_based` or `dqn` |
| `ai.target_gap` | 10.0 m | Desired following gap |
| `traffic.profile` | passive | `passive` or `aggressive` rogue vehicles |

---

## Console Output Guide

```
[INFO]    Normal operational messages
[WARNING] Non-critical issues (config defaults used, CSV retry)
[SAFETY]  Safety override triggered — vehicle braking applied
[ERROR]   Collision detected — episode terminated
[FATAL]   Unrecoverable error — program exiting
```

---

## Phase 2 – DQN AI (Coming Next)

To switch from rule-based to DQN:

1. Uncomment PyTorch lines in `requirements.txt` and run `pip install -r requirements.txt`
2. In `config.yaml` set `ai.mode: dqn`
3. Run `python train.py --episodes 1000` to train the agent
4. Run `python main.py` — it will load the trained model automatically

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `SUMO_HOME not set` | Set the environment variable (Step 1) |
| `highway.net.xml not found` | Run `python sumo_network/build_network.py` |
| `netconvert not found` | Ensure SUMO bin/ is in your PATH |
| `TraCI connection timeout` | Check no other SUMO instance is using port 8813 |
| `ModuleNotFoundError: traci` | Ensure SUMO tools are in Python path (auto-handled) |
| SUMO-GUI doesn't open | Set `simulation.gui: false` and run headless |

---

## References

- SUMO Documentation: https://sumo.dlr.de/docs/
- TraCI Python API: https://sumo.dlr.de/docs/TraCI/
- SRS Document: `SRS_CAV_Platooning_v1.1_With_Diagrams.docx`
- SDD Document: `SDD_CAV_Platooning_v1.0.docx`
