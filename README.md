# Autonomous Navigation Agent

> 2D navigation simulator using Artificial Potential Fields, PID heading control, and simulated range sensors

A Python-based autonomous navigation system that simulates intelligent obstacle avoidance in a 2D world. The agent uses APF (Artificial Potential Fields) for reactive path planning, a PID controller for heading tracking, and ray-cast range sensors for obstacle detection. Designed as a DTU MSc Autonomous Systems portfolio project demonstrating real algorithms from the autonomous systems literature.

## Key Algorithms

- **Artificial Potential Fields** -- Linear repulsive forces with vortex circulation for smooth obstacle avoidance
- **PID Heading Controller** -- Anti-windup integral clamping with derivative filtering
- **Ray-Cast Range Sensor** -- Simulated LIDAR with configurable FOV, ray count, and Gaussian noise
- **Waypoint Navigation** -- Goal sequencing with automatic advancement

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run a scenario
python -m src.main --scenario corridor -o output/corridor.mp4

# Run all 8 scenarios
python run_all_scenarios.py
```

## Scenarios

| Scenario | Description |
|----------|-------------|
| `corridor` | Narrowing passage between wall obstacles |
| `gauntlet` | Slalom through center blockers and flanking obstacles |
| `dynamic` | Moving obstacles crossing the agent's path |
| `slalom` | S-curve through alternating left/right obstacles |
| `narrow_gap` | Precision navigation through a wall opening |
| `u_turn` | Detour around a large central obstacle |
| `crossing` | Dynamic obstacles crossing the agent's path |
| `dense` | Forest of scattered small obstacles |

```bash
# Named scenario
python -m src.main --scenario gauntlet -o output/gauntlet.mp4

# Random with reproducible seed
python -m src.main --scenario random --seed 42 -o output/random.mp4

# Custom duration
python -m src.main --scenario corridor --duration 30 -o output/long.mp4
```

## Architecture

```
Simulation Loop (each time step):
  env.step(dt)              -- advance dynamic obstacles
  navigator.check_and_advance()  -- auto-advance waypoints
  sensor.scan()             -- ray-cast range readings
  planner.compute()         -- APF forces + state classification
  planner.compute_speed()   -- proximity-based speed control
  controller.update()       -- PID heading control + kinematics
  collision response        -- push-out with velocity preservation
  visualization + render    -- draw frame to video
```

| Module | Purpose |
|--------|---------|
| `src/environment.py` | 2D world with circular obstacles and boundary walls |
| `src/sensors.py` | Ray-casting range sensor with noise |
| `src/apf_planner.py` | APF planner: linear repulsion, vortex field, adaptive EMA, symmetry breaking |
| `src/pid_controller.py` | PID heading controller with anti-windup |
| `src/control.py` | Agent controller: PID + kinematics |
| `src/visualization.py` | Frame renderer with world-to-screen transform |
| `src/waypoint_navigator.py` | Goal sequencing |
| `src/main.py` | CLI entry point and simulation orchestration |

## Testing

```bash
pytest                                    # run all tests (320 passing)
pytest --cov=src --cov-report=term-missing  # with coverage
ruff check src/ tests/                    # linting
mypy src/                                 # type checking
```

## Technical Details

**APF Design**: The planner uses linear repulsion (`F = k_rep * (1 - d/d0)`) instead of the classical Khatib 1/d² formulation, which concentrates force too close to obstacle surfaces. A vortex field (70% tangential / 30% radial) prevents head-on deadlocks. Adaptive EMA smoothing reduces heading oscillation near obstacles, and a gated symmetry breaker handles balanced opposing forces.

**Coordinate Convention**: All code uses radians with y-up (standard math convention). Only the visualization module converts to y-down for OpenCV rendering.

**Configuration**: All tunable parameters live in `config/default_config.yaml` with sections for environment, sensor, planner, controller, visualization, and simulation.

## License

MIT
