"""
    reward.py

    Summary:
        This file contains some rewarder classes for giving reward signals to the agent in the colav-simulator environment.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple, TypeVar

import numpy as np
import seacharts.enc as senc
from colav_simulator.core.ship import Ship
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

    def __init__(self, goal: np.ndarray, params: Optional[DistanceToGoalRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            goal (np.ndarray): The goal position [x_g, y_g]^T
            params (DistanceToGoalRewardParams): The reward parameters.
        """
        self.goal = goal
        self.params = params if params else DistanceToGoalRewardParams()

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:

        return -self.params.r_d2g * float(np.linalg.norm(state[:2] - self.goal))


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

    def __init__(self, params: Optional[CollisionRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            params (CollisionRewardParams): The reward parameters.
        """
        self.params = params if params else CollisionRewardParams()

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        """Reward for the current state-action pair. Passes additional argument in terms of bool describing if collision has occured.
                
        Args:
            state (Observation): The current state.
            action (Optional[Action], optional): The current action. Defaults to None.
            **kwargs: Additional arguments. In this case, collision (bool) is passed.
        """
        collision = kwargs.get('collision', False)
        if collision:
            collided = 1.0
        else: collided = 0.0
        return self.params.r_collision * collided


class Rewarder(IReward):
    """The rewarder class."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initializes the rewarder.

        Args:
            config (Config): The rewarder configuration
        """
        self.config = config if config else Config()
        self.rewarders = []
        for rewarder in self.config.rewarders:
            if isinstance(rewarder, ExistenceRewardParams):
                self.rewarders.append(ExistenceRewarder(rewarder))
            elif isinstance(rewarder, DistanceToGoalRewardParams):
                self.rewarders.append(DistanceToGoalRewarder(np.array([0.0, 0.0]), rewarder))
            elif isinstance(rewarder, CollisionRewardParams):
                self.rewarders.append(CollisionRewarder(rewarder))

    def __call__(self, state: Observation, action: Optional[Action] = None, **kwargs) -> float:
        reward = 0.0
        for rewarder in self.rewarders:
            reward += rewarder(state, action, **kwargs)
        return reward
