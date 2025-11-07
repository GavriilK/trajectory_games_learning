"""
GTP-UDrive: Unified Game-Theoretic Trajectory Planner and Decision-Maker
for Autonomous Driving in Mixed Traffic Environments

This implementation provides a unified framework for trajectory planning and
decision-making in autonomous vehicles using game theory, specifically designed
for mixed traffic environments where AVs interact with human-driven vehicles.

Key Components:
1. Clothoid-based trajectory generation for smooth, drivable paths
2. Game-theoretic decision-making using Nash Equilibrium
3. Human driver intention understanding and prediction
4. Unified trajectory planner combining path planning and decision-making

Reference: Based on game-theoretic approaches for autonomous driving in
mixed traffic scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


# Configuration constants
DEFAULT_VELOCITY = 15.0  # m/s (~54 km/h)
DEFAULT_SAFE_DISTANCE = 5.0  # meters
DEFAULT_GOAL_DISTANCE = 50.0  # meters
MIN_INTEGRATION_STEPS = 10
STEPS_PER_UNIT_LENGTH = 10


@dataclass
class VehicleState:
    """Represents the state of a vehicle at a given time"""
    x: float  # longitudinal position
    y: float  # lateral position
    theta: float  # heading angle
    v: float  # velocity
    a: float = 0.0  # acceleration
    curvature: float = 0.0  # path curvature
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor representation"""
        return torch.tensor([self.x, self.y, self.theta, self.v, self.a, self.curvature])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'VehicleState':
        """Create VehicleState from tensor"""
        return cls(
            x=float(tensor[0]),
            y=float(tensor[1]),
            theta=float(tensor[2]),
            v=float(tensor[3]),
            a=float(tensor[4]) if len(tensor) > 4 else 0.0,
            curvature=float(tensor[5]) if len(tensor) > 5 else 0.0
        )


@dataclass
class TrajectoryPoint:
    """A point along a trajectory"""
    state: VehicleState
    time: float
    

class ClothoidSegment:
    """
    Represents a clothoid (Euler spiral) segment for smooth trajectory generation.
    
    A clothoid has linearly varying curvature, which makes it ideal for vehicle
    trajectory planning as it corresponds to constant steering rate.
    """
    
    def __init__(self, start_state: VehicleState, length: float, 
                 curvature_rate: float = 0.0):
        """
        Initialize a clothoid segment
        
        Args:
            start_state: Initial state at the beginning of the segment
            length: Arc length of the clothoid segment
            curvature_rate: Rate of change of curvature (dk/ds)
        """
        self.start_state = start_state
        self.length = length
        self.curvature_rate = curvature_rate
        self.start_curvature = start_state.curvature
        
    def evaluate(self, s: float) -> VehicleState:
        """
        Evaluate the clothoid at arc length s
        
        Args:
            s: Arc length along the clothoid (0 <= s <= length)
            
        Returns:
            VehicleState at the given arc length
        """
        if s < 0 or s > self.length:
            raise ValueError(f"Arc length {s} out of bounds [0, {self.length}]")
        
        # Compute curvature at s
        k = self.start_curvature + self.curvature_rate * s
        
        # Fresnel integrals approximation for clothoid computation
        # For small curvature rates, we use numerical integration
        n_steps = max(MIN_INTEGRATION_STEPS, int(abs(s) * STEPS_PER_UNIT_LENGTH))
        ds = s / n_steps
        
        x, y, theta = self.start_state.x, self.start_state.y, self.start_state.theta
        
        for i in range(n_steps):
            s_i = i * ds
            k_i = self.start_curvature + self.curvature_rate * s_i
            
            # Update position and heading
            x += ds * np.cos(theta)
            y += ds * np.sin(theta)
            theta += k_i * ds
        
        # Create new state
        new_state = VehicleState(
            x=x,
            y=y,
            theta=theta,
            v=self.start_state.v,  # Assume constant velocity for simplicity
            curvature=k
        )
        
        return new_state


class ClothoidTrajectoryGenerator:
    """
    Generates smooth trajectories using clothoid curves.
    
    Clothoids provide continuous curvature profiles, making them suitable for
    vehicle path planning as they can be followed with smooth steering inputs.
    """
    
    def __init__(self, dt: float = 0.1, max_curvature: float = 0.3):
        """
        Initialize the trajectory generator
        
        Args:
            dt: Time step for trajectory discretization
            max_curvature: Maximum allowed curvature (related to minimum turning radius)
        """
        self.dt = dt
        self.max_curvature = max_curvature
        
    def generate_trajectory(self, start_state: VehicleState, 
                          goal_state: VehicleState,
                          horizon: float = 5.0) -> List[TrajectoryPoint]:
        """
        Generate a clothoid-based trajectory from start to goal
        
        Args:
            start_state: Initial vehicle state
            goal_state: Desired goal state
            horizon: Time horizon for trajectory (seconds)
            
        Returns:
            List of trajectory points connecting start to goal
        """
        # Compute required parameters for clothoid connection
        dx = goal_state.x - start_state.x
        dy = goal_state.y - start_state.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Simple heuristic: use two clothoid segments
        # First segment: transition from current curvature to midpoint
        # Second segment: transition from midpoint to goal curvature
        
        mid_length = distance / 2.0
        curvature_rate1 = -start_state.curvature / mid_length if mid_length > 0 else 0
        
        # Generate first segment
        segment1 = ClothoidSegment(start_state, mid_length, curvature_rate1)
        mid_state = segment1.evaluate(mid_length)
        
        # Generate second segment to reach goal
        curvature_rate2 = (goal_state.curvature) / mid_length if mid_length > 0 else 0
        segment2 = ClothoidSegment(mid_state, mid_length, curvature_rate2)
        
        # Discretize trajectory
        trajectory = []
        n_points = int(horizon / self.dt)
        
        for i in range(n_points):
            t = i * self.dt
            s_total = (distance / horizon) * t  # Arc length at time t
            
            if s_total <= mid_length:
                state = segment1.evaluate(s_total)
            else:
                state = segment2.evaluate(s_total - mid_length)
            
            trajectory.append(TrajectoryPoint(state=state, time=t))
        
        return trajectory
    
    def interpolate_trajectory(self, waypoints: List[VehicleState]) -> List[TrajectoryPoint]:
        """
        Interpolate a smooth trajectory through given waypoints using clothoids
        
        Args:
            waypoints: List of vehicle states to pass through
            
        Returns:
            Smooth trajectory passing through waypoints
        """
        if len(waypoints) < 2:
            raise ValueError(f"Need at least 2 waypoints for interpolation, got {len(waypoints)}")
        
        trajectory = []
        time = 0.0
        
        for i in range(len(waypoints) - 1):
            segment_traj = self.generate_trajectory(
                waypoints[i], waypoints[i + 1], horizon=1.0
            )
            
            # Add segment to overall trajectory
            for point in segment_traj:
                trajectory.append(TrajectoryPoint(
                    state=point.state,
                    time=time + point.time
                ))
            
            time += 1.0
        
        return trajectory


class GameTheoreticPlanner:
    """
    Implements game-theoretic decision-making for autonomous vehicles.
    
    Uses Nash Equilibrium concepts to model interactions between the AV
    and other agents (human drivers) in the environment.
    """
    
    def __init__(self, n_agents: int = 2, horizon: int = 50, 
                 safe_distance: float = DEFAULT_SAFE_DISTANCE):
        """
        Initialize the game-theoretic planner
        
        Args:
            n_agents: Number of agents in the game (including ego vehicle)
            horizon: Planning horizon in time steps
            safe_distance: Minimum safe distance between vehicles (meters)
        """
        self.n_agents = n_agents
        self.horizon = horizon
        self.safe_distance = safe_distance
        
    def compute_utility(self, trajectory: List[TrajectoryPoint],
                       other_trajectories: List[List[TrajectoryPoint]],
                       agent_id: int,
                       preferences: Dict) -> float:
        """
        Compute utility function for an agent given its trajectory and others
        
        Args:
            trajectory: Trajectory of the agent
            other_trajectories: Trajectories of other agents
            agent_id: ID of the agent whose utility to compute
            preferences: Preference parameters for the agent
            
        Returns:
            Utility value (higher is better)
        """
        utility = 0.0
        
        # Goal reaching reward
        goal_weight = preferences.get('goal_weight', 1.0)
        comfort_weight = preferences.get('comfort_weight', 0.5)
        safety_weight = preferences.get('safety_weight', 2.0)
        
        # Penalize deviation from desired velocity
        desired_velocity = preferences.get('desired_velocity', DEFAULT_VELOCITY)
        for point in trajectory:
            velocity_error = (point.state.v - desired_velocity) ** 2
            utility -= comfort_weight * velocity_error
            
            # Penalize high accelerations (comfort)
            utility -= comfort_weight * point.state.a ** 2
            
            # Penalize high curvatures (comfort)
            utility -= comfort_weight * point.state.curvature ** 2
        
        # Safety: penalize proximity to other vehicles
        for other_traj in other_trajectories:
            for t_idx, point in enumerate(trajectory):
                if t_idx < len(other_traj):
                    other_point = other_traj[t_idx]
                    
                    # Compute distance
                    dx = point.state.x - other_point.state.x
                    dy = point.state.y - other_point.state.y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Penalize small distances (collision avoidance)
                    if distance < self.safe_distance:
                        utility -= safety_weight * (self.safe_distance - distance) ** 2
        
        # Goal reaching: reward being close to goal
        if len(trajectory) > 0:
            final_state = trajectory[-1].state
            goal_position = preferences.get('goal_position', (100.0, 0.0))
            goal_distance = np.sqrt(
                (final_state.x - goal_position[0])**2 + 
                (final_state.y - goal_position[1])**2
            )
            utility -= goal_weight * goal_distance
        
        return utility
    
    def find_nash_equilibrium(self, initial_states: List[VehicleState],
                             preferences: List[Dict],
                             max_iterations: int = 100,
                             goal_distance: float = DEFAULT_GOAL_DISTANCE) -> List[List[TrajectoryPoint]]:
        """
        Find Nash Equilibrium trajectories for all agents
        
        Args:
            initial_states: Initial states of all agents
            preferences: Preference parameters for each agent
            max_iterations: Maximum number of iterations for equilibrium search
            goal_distance: Distance ahead to set as goal for trajectory generation
            
        Returns:
            List of equilibrium trajectories for each agent
        """
        # Initialize with straight-line trajectories
        trajectories = []
        traj_generator = ClothoidTrajectoryGenerator()
        
        for i, state in enumerate(initial_states):
            # Create simple goal state
            goal_state = VehicleState(
                x=state.x + goal_distance,
                y=state.y,
                theta=state.theta,
                v=preferences[i].get('desired_velocity', DEFAULT_VELOCITY)
            )
            
            traj = traj_generator.generate_trajectory(state, goal_state, horizon=5.0)
            trajectories.append(traj)
        
        # Iterative best response dynamics to find Nash Equilibrium
        for iteration in range(max_iterations):
            updated = False
            
            for agent_id in range(self.n_agents):
                # Get current trajectory and others
                current_traj = trajectories[agent_id]
                other_trajs = [trajectories[j] for j in range(self.n_agents) if j != agent_id]
                
                # Compute current utility
                current_utility = self.compute_utility(
                    current_traj, other_trajs, agent_id, preferences[agent_id]
                )
                
                # Try to find better response (simplified optimization)
                # In practice, this would use gradient-based optimization
                best_traj = current_traj
                best_utility = current_utility
                
                # Sample some alternative trajectories
                for offset_y in [-1.0, 0.0, 1.0]:
                    goal_state = VehicleState(
                        x=initial_states[agent_id].x + goal_distance,
                        y=initial_states[agent_id].y + offset_y,
                        theta=initial_states[agent_id].theta,
                        v=preferences[agent_id].get('desired_velocity', DEFAULT_VELOCITY)
                    )
                    
                    candidate_traj = traj_generator.generate_trajectory(
                        initial_states[agent_id], goal_state, horizon=5.0
                    )
                    
                    utility = self.compute_utility(
                        candidate_traj, other_trajs, agent_id, preferences[agent_id]
                    )
                    
                    if utility > best_utility:
                        best_utility = utility
                        best_traj = candidate_traj
                        updated = True
                
                trajectories[agent_id] = best_traj
            
            # Check convergence
            if not updated:
                break
        
        return trajectories


class IntentionPredictor(nn.Module):
    """
    Neural network for predicting human driver intentions.
    
    This module learns to predict future driver behavior based on
    observed past trajectories and context.
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, 
                 output_dim: int = 4):
        """
        Initialize intention predictor
        
        Args:
            input_dim: Dimension of input features (state representation)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (predicted parameters)
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, past_states: torch.Tensor) -> torch.Tensor:
        """
        Predict driver intentions from past states
        
        Args:
            past_states: Tensor of shape (batch, sequence_len, state_dim)
            
        Returns:
            Predicted intention parameters
        """
        # Use mean pooling over sequence
        pooled = torch.mean(past_states, dim=1)
        intentions = self.encoder(pooled)
        return intentions


class GTPUDrive:
    """
    Main class for GTP-UDrive: Unified Game-Theoretic Trajectory Planner
    and Decision-Maker for Autonomous Driving.
    
    Integrates:
    1. Clothoid-based trajectory generation
    2. Game-theoretic decision-making
    3. Human driver intention prediction
    4. Unified planning and decision framework
    """
    
    def __init__(self, ego_id: int = 0, dt: float = 0.1,
                 planning_horizon: float = 5.0,
                 desired_velocity: float = DEFAULT_VELOCITY,
                 safe_distance: float = DEFAULT_SAFE_DISTANCE):
        """
        Initialize GTP-UDrive system
        
        Args:
            ego_id: ID of the ego vehicle (AV)
            dt: Time discretization step
            planning_horizon: Planning horizon in seconds
            desired_velocity: Desired velocity for ego vehicle (m/s)
            safe_distance: Minimum safe distance between vehicles (meters)
        """
        self.ego_id = ego_id
        self.dt = dt
        self.planning_horizon = planning_horizon
        self.safe_distance = safe_distance
        
        # Initialize components
        self.trajectory_generator = ClothoidTrajectoryGenerator(dt=dt)
        self.game_planner = GameTheoreticPlanner(
            horizon=int(planning_horizon / dt),
            safe_distance=safe_distance
        )
        self.intention_predictor = IntentionPredictor()
        
        # Ego vehicle preferences
        self.ego_preferences = {
            'desired_velocity': desired_velocity,
            'goal_weight': 1.0,
            'comfort_weight': 0.5,
            'safety_weight': 2.0,
            'goal_position': (100.0, 0.0)
        }
    
    def predict_human_intentions(self, past_trajectories: List[List[TrajectoryPoint]]) -> List[Dict]:
        """
        Predict intentions of human drivers based on observed behavior
        
        Args:
            past_trajectories: List of past trajectories for each human driver
            
        Returns:
            List of predicted preference dictionaries for each driver
        """
        predicted_preferences = []
        
        for traj in past_trajectories:
            # Convert trajectory to tensor
            states = [point.state.to_tensor() for point in traj]
            state_tensor = torch.stack(states).unsqueeze(0)  # Add batch dimension
            
            # Predict intentions
            with torch.no_grad():
                intentions = self.intention_predictor(state_tensor)
            
            # Convert to preference dictionary
            prefs = {
                'desired_velocity': float(intentions[0, 0]) + DEFAULT_VELOCITY,
                'goal_weight': float(torch.sigmoid(intentions[0, 1])),
                'comfort_weight': float(torch.sigmoid(intentions[0, 2])),
                'safety_weight': float(torch.sigmoid(intentions[0, 3])) * 2.0,
                'goal_position': (100.0, 0.0)  # Simplified
            }
            
            predicted_preferences.append(prefs)
        
        return predicted_preferences
    
    def plan(self, current_state: VehicleState,
            other_states: List[VehicleState],
            past_trajectories: Optional[List[List[TrajectoryPoint]]] = None,
            goal_state: Optional[VehicleState] = None) -> List[TrajectoryPoint]:
        """
        Main planning function: generates optimal trajectory for ego vehicle
        
        Args:
            current_state: Current state of ego vehicle
            other_states: Current states of other vehicles
            past_trajectories: Past trajectories of other vehicles (for intention prediction)
            goal_state: Optional goal state for ego vehicle
            
        Returns:
            Planned trajectory for ego vehicle
        """
        # Set goal if provided
        if goal_state is not None:
            self.ego_preferences['goal_position'] = (goal_state.x, goal_state.y)
        
        # Predict human driver intentions if past data available
        if past_trajectories is not None and len(past_trajectories) > 0:
            predicted_prefs = self.predict_human_intentions(past_trajectories)
        else:
            # Use default preferences for human drivers
            predicted_prefs = [
                {
                    'desired_velocity': DEFAULT_VELOCITY,
                    'goal_weight': 1.0,
                    'comfort_weight': 0.5,
                    'safety_weight': 1.5,
                    'goal_position': (100.0, state.y)
                }
                for state in other_states
            ]
        
        # Combine ego and other vehicle states
        all_states = [current_state] + other_states
        all_preferences = [self.ego_preferences] + predicted_prefs
        
        # Solve game-theoretic planning problem
        equilibrium_trajectories = self.game_planner.find_nash_equilibrium(
            all_states, all_preferences, max_iterations=50
        )
        
        # Return ego vehicle trajectory
        return equilibrium_trajectories[self.ego_id]
    
    def evaluate_trajectory_safety(self, trajectory: List[TrajectoryPoint],
                                   other_trajectories: List[List[TrajectoryPoint]]) -> float:
        """
        Evaluate safety score of a trajectory
        
        Args:
            trajectory: Trajectory to evaluate
            other_trajectories: Trajectories of other vehicles
            
        Returns:
            Safety score (0-1, higher is safer)
        """
        min_distance = float('inf')
        
        for t_idx, point in enumerate(trajectory):
            for other_traj in other_trajectories:
                if t_idx < len(other_traj):
                    other_point = other_traj[t_idx]
                    
                    dx = point.state.x - other_point.state.x
                    dy = point.state.y - other_point.state.y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    min_distance = min(min_distance, distance)
        
        # Convert to safety score (exponential decay)
        safety_score = 1.0 - np.exp(-min_distance / self.safe_distance)
        
        return safety_score
    
    def update_ego_preferences(self, preferences: Dict):
        """Update ego vehicle preference parameters"""
        self.ego_preferences.update(preferences)
    
    def set_goal(self, goal_position: Tuple[float, float]):
        """Set goal position for ego vehicle"""
        self.ego_preferences['goal_position'] = goal_position


# Helper functions for trajectory analysis and visualization

def compute_trajectory_metrics(trajectory: List[TrajectoryPoint]) -> Dict[str, float]:
    """
    Compute various metrics for a trajectory
    
    Args:
        trajectory: List of trajectory points
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'length': 0.0,
        'max_velocity': 0.0,
        'max_acceleration': 0.0,
        'max_curvature': 0.0,
        'avg_velocity': 0.0,
        'smoothness': 0.0
    }
    
    if len(trajectory) == 0:
        return metrics
    
    velocities = []
    accelerations = []
    curvatures = []
    
    for i, point in enumerate(trajectory):
        velocities.append(point.state.v)
        accelerations.append(abs(point.state.a))
        curvatures.append(abs(point.state.curvature))
        
        if i > 0:
            prev_point = trajectory[i-1]
            dx = point.state.x - prev_point.state.x
            dy = point.state.y - prev_point.state.y
            metrics['length'] += np.sqrt(dx**2 + dy**2)
    
    metrics['max_velocity'] = max(velocities) if velocities else 0.0
    metrics['max_acceleration'] = max(accelerations) if accelerations else 0.0
    metrics['max_curvature'] = max(curvatures) if curvatures else 0.0
    metrics['avg_velocity'] = np.mean(velocities) if velocities else 0.0
    
    # Smoothness: variance in acceleration
    if len(accelerations) > 1:
        metrics['smoothness'] = 1.0 / (1.0 + np.var(accelerations))
    
    return metrics


def trajectory_to_array(trajectory: List[TrajectoryPoint]) -> np.ndarray:
    """
    Convert trajectory to numpy array for easy manipulation
    
    Args:
        trajectory: List of trajectory points
        
    Returns:
        Array of shape (n_points, 7) with [x, y, theta, v, a, curvature, time]
    """
    array = np.zeros((len(trajectory), 7))
    
    for i, point in enumerate(trajectory):
        array[i, 0] = point.state.x
        array[i, 1] = point.state.y
        array[i, 2] = point.state.theta
        array[i, 3] = point.state.v
        array[i, 4] = point.state.a
        array[i, 5] = point.state.curvature
        array[i, 6] = point.time
    
    return array


# Example usage and demonstration
if __name__ == "__main__":
    """
    Demonstrate GTP-UDrive system with a simple scenario
    """
    print("GTP-UDrive: Game-Theoretic Trajectory Planner Demo")
    print("=" * 60)
    
    # Initialize GTP-UDrive system
    gtp = GTPUDrive(ego_id=0, dt=0.1, planning_horizon=5.0)
    
    # Define initial states
    ego_state = VehicleState(x=0.0, y=0.0, theta=0.0, v=15.0)
    other_state = VehicleState(x=10.0, y=3.0, theta=0.0, v=12.0)
    
    print(f"\nInitial States:")
    print(f"  Ego vehicle: x={ego_state.x:.1f}, y={ego_state.y:.1f}, v={ego_state.v:.1f} m/s")
    print(f"  Other vehicle: x={other_state.x:.1f}, y={other_state.y:.1f}, v={other_state.v:.1f} m/s")
    
    # Set goal for ego vehicle
    goal_state = VehicleState(x=100.0, y=0.0, theta=0.0, v=15.0)
    print(f"\nGoal: x={goal_state.x:.1f}, y={goal_state.y:.1f}")
    
    # Plan trajectory
    print("\nPlanning trajectory using game-theoretic approach...")
    trajectory = gtp.plan(ego_state, [other_state], goal_state=goal_state)
    
    print(f"Generated trajectory with {len(trajectory)} points")
    
    # Compute metrics
    metrics = compute_trajectory_metrics(trajectory)
    print(f"\nTrajectory Metrics:")
    print(f"  Length: {metrics['length']:.2f} m")
    print(f"  Max velocity: {metrics['max_velocity']:.2f} m/s")
    print(f"  Max acceleration: {metrics['max_acceleration']:.2f} m/s²")
    print(f"  Max curvature: {metrics['max_curvature']:.4f} 1/m")
    print(f"  Avg velocity: {metrics['avg_velocity']:.2f} m/s")
    print(f"  Smoothness: {metrics['smoothness']:.4f}")
    
    # Show sample points
    print(f"\nSample trajectory points:")
    for i in [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]:
        if i < len(trajectory):
            point = trajectory[i]
            print(f"  t={point.time:.2f}s: x={point.state.x:.2f}, y={point.state.y:.2f}, "
                  f"v={point.state.v:.2f} m/s")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
