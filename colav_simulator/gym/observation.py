"""
    observation.py

    Summary:
        This file contains various observation space definitions for a ship agent in the colav-simulator.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, TypeVar, Union

import colav_simulator.core.sensing as sensing
import colav_simulator.core.tracking.trackers as trackers
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
from colav_simulator.core.ship import Ship

Observation = TypeVar("Observation")

if TYPE_CHECKING:
    from colav_simulator.gym.environment import BaseEnvironment


class ObservationType(ABC):
    name: str = "AbstractObservation"

    def __init__(self, env: "BaseEnvironment", ownship: Ship) -> None:
        self.env = env
        self.__observer_ship = ownship

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """Get the observation space."""

    @abstractmethod
    def observe(self) -> np.ndarray:
        """Get an observation of the environment state."""


class LidarLikeObservation(ObservationType):
    """A lidar-like observation space for the own-ship, i.e. a 360 degree scan of the environment as is used in Meyer et. al. 2020."""

    def __init__(
        self,
        env: "BaseEnvironment",
        ownship: Ship,
    ) -> None:
        super().__init__(env, ownship)
        self.n_do = len(self.env.dynamic_obstacles)
        self.enc = self.env.enc

        self.name = "LidarLikeObservation"

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_do,))

    def observe(self) -> np.ndarray:
        """Get an observation of the environment state."""

        obs = np.zeros(3)
        # sector_closenesses, sector_velocities = self.__observer_ship.perceive(self.obstacles)

        # obs = np.concatenate([reward_insight, navigation_states, sector_closenesses, sector_velocities])
        return obs


def observation_factory(env: "BaseEnvironment", ownship: Ship, observation_type: str = "ContinuousAutopilotReferenceAction") -> ObservationType:
    """Factory for creating observation spaces.

    Args:
        env (str, optional): Name of environment. Defaults to "AbstractEnv".
        observation_type (str, optional): Action type name. Defaults to "LidarLikeObservation".

    Returns:
        ActionType: Action type to use
    """
    if observation_type == "LidarLikeObservation":
        return LidarLikeObservation(env, ownship)
    else:
        raise ValueError("Unknown action type")
