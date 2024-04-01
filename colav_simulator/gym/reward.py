"""
    reward.py

    Summary:
        This file contains some rewarder classes for giving reward signals to the agent in the colav-simulator environment.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from colav_simulator.gym.action import Action
from colav_simulator.gym.observation import Observation

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


@dataclass
class ExistenceRewardParams:
    """Parameters for the Existence rewarder."""

    r_exists: float = -1.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = ExistenceRewardParams()
        cfg.r_exists = config_dict["r_exists"]
        return cfg

    def to_dict(self):
        return {"r_exists": self.r_exists}


@dataclass
class CollisionRewardParams:
    """Parameters for the Collision rewarder."""

    r_collision: float = -500.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = CollisionRewardParams()
        cfg.r_collision = config_dict["r_collision"]
        return cfg

    def to_dict(self):
        return {"r_collision": self.r_collision}


@dataclass
class GroundingRewardParams:
    """Parameters for the Grounding rewarder."""

    r_grounding: float = -500.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = GroundingRewardParams()
        cfg.r_grounding = config_dict["r_grounding"]
        return cfg

    def to_dict(self):
        return {"r_grounding": self.r_grounding}


@dataclass
class GoalReachedRewardParams:
    """Parameters for the Goal reached rewarder."""

    r_goal_reached: float = 10000.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = GoalReachedRewardParams()
        cfg.r_goal_reached = config_dict["r_goal_reached"]
        return cfg

    def to_dict(self):
        return {"r_goal_reached": self.r_goal_reached}


@dataclass
class DistanceToGoalRewardParams:
    """Paramters for the distance to goal rewarder."""

    r_d2g: float = 1.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = DistanceToGoalRewardParams()
        cfg.r_d2g = config_dict["r_d2g"]
        return cfg

    def to_dict(self):
        return {"r_d2g": self.r_d2g}


@dataclass
class TrajectoryTrackingRewardParams:
    """Parameters for the trajectory tracking rewarder."""

    r_tt: float = 1.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = TrajectoryTrackingRewardParams()
        cfg.r_tt = config_dict["r_tt"]
        return cfg

    def to_dict(self):
        return {"r_tt": self.r_tt}


class IReward(ABC):
    """Interface/base class for reward functions. The rewarder parameters should be public."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        self.env = env
        self._ownship = self.env.ownship

    @abstractmethod
    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        """Get the reward for the current state-action pair. Additional arguments can be passed if necessary."""


@dataclass
class Config:
    """Configuration class of parameters for the rewarder."""

    rewarders: list = field(default_factory=lambda: [ExistenceRewardParams()])

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = Config()
        rewarders = config_dict["rewarders"]
        for rewarder in rewarders:
            if "Existence_rewarder" in rewarder:
                cfg.rewarders.append(ExistenceRewardParams.from_dict(rewarder))
            elif "distance_to_goal_rewarder" in rewarder:
                cfg.rewarders.append(DistanceToGoalRewardParams.from_dict(rewarder))
            elif "collision_rewarder" in rewarder:
                cfg.rewarders.append(CollisionRewardParams.from_dict(rewarder))
            elif "grounding_rewarder" in rewarder:
                cfg.rewarders.append(GroundingRewardParams.from_dict(rewarder))
            elif "goal_reached_rewarder" in rewarder:
                cfg.rewarders.append(GoalReachedRewardParams.from_dict(rewarder))
        return cfg

    def to_dict(self) -> dict:
        rewarders = []
        for rewarder in self.rewarders:
            if isinstance(rewarder, ExistenceRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, DistanceToGoalRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, CollisionRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, GroundingRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, GoalReachedRewardParams):
                rewarders.append(rewarder.to_dict())
        return {"rewarders": rewarders}


class ExistenceRewarder(IReward):
    """Reward the agent negatively for existing."""

    def __init__(self, params: Optional[ExistenceRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            params (ExistenceRewardParams): The reward parameters.
        """
        self.params = params if params else ExistenceRewardParams()

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        return self.params.r_exists


class DistanceToGoalRewarder(IReward):
    """Reward the agent for getting closer to the goal."""

    def __init__(
        self, env: "COLAVEnvironment", goal: np.ndarray, params: Optional[DistanceToGoalRewardParams] = None
    ) -> None:
        """Initializes the reward function.

        Args:
            goal (np.ndarray): The goal position [x_g, y_g]^T
            params (DistanceToGoalRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.params = params if params else DistanceToGoalRewardParams()

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        if self.env.ownship._goal_state.size > 0:
            self.goal = self.env.ownship._goal_state[:2]
        elif self.env.ownship._waypoints.size > 1:
            self.goal = self.env.ownship._waypoints[:, -1]
        else:
            raise ValueError("No goal state or waypoints found")
        return -self.params.r_d2g * float(np.linalg.norm(self.env.ownship.csog_state[:2] - self.goal))


class TrajectoryTrackingRewarder(IReward):
    """Reward the agent for tracking a trajectory, on the form

    r(s, a) =  todo
    """

    def __init__(self, trajectory: np.ndarray, params: Optional[TrajectoryTrackingRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            trajectory (np.ndarray): The trajectory to track
        """
        self.trajectory = trajectory
        self.params = params if params else TrajectoryTrackingRewardParams()

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        return -float(np.linalg.norm(state[:2] - self.trajectory))


class CollisionRewarder(IReward):
    """Reward the agent negatively for colliding."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[CollisionRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            env (COLAVEnvironment): The environment
            params (CollisionRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.params = params if params else CollisionRewardParams()

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        """Reward for the current state-action pair. Calls function from simulator class to get collision or not."""
        collision = self.env.simulator.determine_ownship_collision()
        if collision:
            return self.params.r_collision
        else:
            return 0.0


class GroundingRewarder(IReward):
    """Reward the agent negatively for grounding."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[CollisionRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            env (COLAVEnvironment): The environment
            params (GroundingRewardParams): The reward parameters.
        """
        self.params = params if params else GroundingRewardParams()
        super().__init__(env)

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        """Reward for the current state-action pair. Calls function from simulator class to get grounding or not."""
        grounding = self.env.simulator.determine_ownship_grounding()
        if grounding:
            return self.params.r_grounding
        else:
            return 0.0


class GoalReachedRewarder(IReward):
    """Reward the agent for reaching the goal."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[GoalReachedRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            env (COLAVEnvironment): The environment
            params (DistanceToGoalRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.params = params if params else GoalReachedRewardParams()

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        """Reward for the current state-action pair. Calls function from simulator class to get goal reached or not."""
        goal_reached = self.env.simulator.determine_ownship_goal_reached()
        if goal_reached:
            return self.params.r_goal_reached
        return 0.0


class Rewarder(IReward):
    """The rewarder class."""

    def __init__(self, env: "COLAVEnvironment", config: Optional[Config] = None) -> None:
        """Initializes the rewarder.

        Args:
            env (COLAVEnvironment): The environment
            config (Config): The rewarder configuration
        """
        self.config = config if config else Config()
        self.rewarders = []
        for rewarder in self.config.rewarders:
            if isinstance(rewarder, ExistenceRewardParams):
                self.rewarders.append(ExistenceRewarder(rewarder))
            elif isinstance(rewarder, DistanceToGoalRewardParams):
                self.rewarders.append(DistanceToGoalRewarder(env, rewarder))
            elif isinstance(rewarder, CollisionRewardParams):
                self.rewarders.append(CollisionRewarder(env, rewarder))
            elif isinstance(rewarder, GroundingRewardParams):
                self.rewarders.append(GroundingRewarder(env, rewarder))
            elif isinstance(rewarder, GoalReachedRewardParams):
                self.rewarders.append(GoalReachedRewarder(env, rewarder))

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        reward = 0.0
        for rewarder in self.rewarders:
            reward += rewarder(state, action, **kwargs)
        return reward
