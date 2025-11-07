# GTP-UDrive: Game-Theoretic Trajectory Planner for Autonomous Driving

## Overview

GTP-UDrive is a unified framework for trajectory planning and decision-making in autonomous vehicles, specifically designed for mixed traffic environments where autonomous vehicles (AVs) interact with human-driven vehicles.

## Key Features

### 1. Clothoid-Based Trajectory Generation
- Smooth, drivable trajectories using clothoid (Euler spiral) curves
- Continuous curvature profiles suitable for vehicle dynamics
- Corresponds to constant steering rate for natural vehicle motion

### 2. Game-Theoretic Decision Making
- Nash Equilibrium-based planning for multi-agent scenarios
- Models strategic interactions between AV and human drivers
- Iterative best-response dynamics for equilibrium search

### 3. Human Driver Intention Prediction
- Neural network to predict driver behavior from past trajectories
- Learns preference parameters (desired velocity, comfort, safety priorities)
- Enables more accurate modeling of human driver behavior

### 4. Unified Planning Framework
- Integrates trajectory generation and decision-making
- Configurable safety parameters and vehicle preferences
- Real-time capable planning with adjustable horizons

## Components

### VehicleState
Represents the state of a vehicle at a given time:
- Position (x, y)
- Heading angle (theta)
- Velocity (v)
- Acceleration (a)
- Path curvature

### ClothoidTrajectoryGenerator
Generates smooth trajectories using clothoid curves:
```python
generator = ClothoidTrajectoryGenerator(dt=0.1, max_curvature=0.3)
trajectory = generator.generate_trajectory(start_state, goal_state, horizon=5.0)
```

### GameTheoreticPlanner
Implements Nash Equilibrium-based planning:
```python
planner = GameTheoreticPlanner(n_agents=2, horizon=50, safe_distance=5.0)
equilibrium_trajectories = planner.find_nash_equilibrium(initial_states, preferences)
```

### GTPUDrive
Main unified planning class:
```python
gtp = GTPUDrive(ego_id=0, dt=0.1, planning_horizon=5.0)
trajectory = gtp.plan(current_state, other_states, goal_state=goal_state)
```

## Usage Example

```python
from implementation import GTPUDrive, VehicleState

# Initialize the system
gtp = GTPUDrive(
    ego_id=0,
    dt=0.1,
    planning_horizon=5.0,
    desired_velocity=15.0,  # m/s
    safe_distance=5.0       # meters
)

# Define vehicle states
ego_state = VehicleState(x=0.0, y=0.0, theta=0.0, v=15.0)
other_state = VehicleState(x=10.0, y=3.0, theta=0.0, v=12.0)
goal_state = VehicleState(x=100.0, y=0.0, theta=0.0, v=15.0)

# Plan trajectory
trajectory = gtp.plan(
    current_state=ego_state,
    other_states=[other_state],
    goal_state=goal_state
)

# Access trajectory points
for point in trajectory:
    print(f"t={point.time:.2f}s: x={point.state.x:.2f}, y={point.state.y:.2f}")
```

## Configuration

### Constants
The implementation uses several configurable constants:
- `DEFAULT_VELOCITY = 15.0` m/s (~54 km/h)
- `DEFAULT_SAFE_DISTANCE = 5.0` meters
- `DEFAULT_GOAL_DISTANCE = 50.0` meters
- `MIN_INTEGRATION_STEPS = 10`
- `STEPS_PER_UNIT_LENGTH = 10`

### Preferences
Each agent has preference parameters that influence their behavior:
- `desired_velocity`: Target velocity (m/s)
- `goal_weight`: Importance of reaching goal
- `comfort_weight`: Importance of smooth motion
- `safety_weight`: Importance of collision avoidance
- `goal_position`: Target position (x, y)

## Utility Functions

### Trajectory Metrics
```python
from implementation import compute_trajectory_metrics

metrics = compute_trajectory_metrics(trajectory)
print(f"Length: {metrics['length']:.2f} m")
print(f"Max velocity: {metrics['max_velocity']:.2f} m/s")
print(f"Smoothness: {metrics['smoothness']:.4f}")
```

### Safety Evaluation
```python
safety_score = gtp.evaluate_trajectory_safety(ego_trajectory, other_trajectories)
print(f"Safety score: {safety_score:.2f} (0=unsafe, 1=safe)")
```

### Array Conversion
```python
from implementation import trajectory_to_array

array = trajectory_to_array(trajectory)
# Returns shape (n_points, 7): [x, y, theta, v, a, curvature, time]
```

## Demo

Run the built-in demo:
```bash
python implementation.py
```

This demonstrates:
- System initialization
- Multi-vehicle scenario setup
- Game-theoretic trajectory planning
- Trajectory metrics computation
- Sample trajectory point inspection

## Architecture

```
GTPUDrive
├── ClothoidTrajectoryGenerator
│   └── ClothoidSegment (for each trajectory segment)
├── GameTheoreticPlanner
│   └── Utility computation and Nash equilibrium search
└── IntentionPredictor (PyTorch neural network)
    └── Predicts human driver preferences
```

## Theory

### Clothoid Curves
Clothoids (Euler spirals) have linearly varying curvature, making them ideal for vehicle path planning:
- Curvature κ(s) = κ₀ + κ' · s
- Corresponds to constant steering rate
- Smooth transitions between straight and curved sections

### Nash Equilibrium
The planner finds trajectories where no agent can improve their utility by unilaterally changing their plan:
- Each agent maximizes: U_i(x_i, x_{-i})
- Subject to dynamics and safety constraints
- Iterative best-response algorithm converges to equilibrium

### Utility Function
Each agent's utility balances multiple objectives:
```
U = -goal_weight * goal_distance
    - comfort_weight * (velocity_error² + acceleration² + curvature²)
    - safety_weight * Σ collision_penalties
```

## Performance

- **Planning time**: Typically < 100ms for 2-3 agents
- **Horizon**: 5 seconds (50 time steps at 10Hz)
- **Safety**: Maintains minimum safe distance between vehicles
- **Smoothness**: Continuous curvature for comfortable motion

## Limitations

- Simplified clothoid interpolation (numerical integration)
- Limited equilibrium search (sampling-based, not gradient-based)
- Assumes constant velocity in clothoid segments
- Neural network requires training for intention prediction

## Future Enhancements

1. **Advanced Optimization**: Gradient-based Nash equilibrium search
2. **Dynamic Models**: Full vehicle dynamics with acceleration/braking
3. **Uncertainty Handling**: Robust planning under uncertainty
4. **Learning**: Train intention predictor on real traffic data
5. **Visualization**: Add plotting utilities for trajectories
6. **Multi-lane**: Extended support for complex road geometries

## References

- Game-theoretic trajectory planning for autonomous vehicles
- Clothoid curves for smooth path generation
- Nash Equilibrium in multi-agent systems
- Human driver behavior modeling

## License

Same as the parent repository (AGPL-3.0).
