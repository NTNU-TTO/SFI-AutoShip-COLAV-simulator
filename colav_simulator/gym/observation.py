"""
    observation.py

    Summary:
        This file contains various observation space definitions for a ship agent in the colav-simulator.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import colav_simulator.core.sensing as sensing
import colav_simulator.core.tracking.trackers as trackers
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
from colav_simulator.core.ship import Ship

Observation = Union[tuple, list, np.ndarray]

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


class ObservationType(ABC):
    """Interface class for observation types for the COLAVEnvironment gym."""

    name: str = "AbstractObservation"

    def __init__(self, env: "COLAVEnvironment", ownship: Ship) -> None:
        self.env = env
        self.__ownship = ownship

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """Get the observation space."""

    @abstractmethod
    def observe(self) -> Observation:
        """Get an observation of the environment state."""


class LidarLikeObservation(ObservationType):
    """A lidar-like observation space for the own-ship, i.e. a 360 degree scan of the environment as is used in Meyer et. al. 2020."""

    def __init__(
        self,
        env: "COLAVEnvironment",
        ownship: Ship,
    ) -> None:
        super().__init__(env, ownship)
        self.n_do = len(self.env.dynamic_obstacles)
        self.enc = self.env.enc

        self.name = "LidarLikeObservation"

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_do,))

    def observe(self) -> Observation:
        """Get an observation of the environment state."""

        obs = np.zeros(3)
        # sector_closenesses, sector_velocities = self.__ownship.get_track_information(self.obstacles)

        # obs = np.concatenate([reward_insight, navigation_states, sector_closenesses, sector_velocities])
        return obs


class NavigationStateObservation(ObservationType):
    """Observes the current own-ship state, possibly augmented by relevant info such as cross-track error, etc."""

    def __init__(
        self,
        env: "COLAVEnvironment",
        ownship: Ship,
    ) -> None:
        super().__init__(env, ownship)

        self.name = "NavigationStateObservation"
        self.size = len(ownship.state) + 0

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,))

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        state = self.__ownship.state
        extras = np.empty(0)
        obs = np.concatenate([state, extras])
        return obs


class TupleObservation(ObservationType):
    """Observation consisting of multiple observation types."""

    def __init__(self, env: "COLAVEnvironment", ownship: Ship, observation_configs: list, **kwargs) -> None:
        super().__init__(env, ownship)
        self.observation_types = [observation_factory(env, ownship, obs_config) for obs_config in observation_configs]

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> Observation:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


def observation_factory(env: "COLAVEnvironment", ownship: Ship, observation_type: str = "ContinuousAutopilotReferenceAction") -> ObservationType:
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
