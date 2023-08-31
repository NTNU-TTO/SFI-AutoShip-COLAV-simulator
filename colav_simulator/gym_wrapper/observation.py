"""
    observation.py

    Summary:
        This file contains various observation space definitions for a ship agent in the colav-simulator.

    Author: Trym Tengesdal
"""

import itertools
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import colav_simulator.core.sensing as sensing
import colav_simulator.core.tracking.trackers as trackers
import gymnasium as gym
import numpy as np
import pandas as pd
import seacharts.enc as senc
from colav_simulator.core.ship import Ship


class ObservationType(ABC):
    def __init__(self, ownship: Ship) -> None:
        self.__observer_ship = ownship

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """Get the observation space."""

    @abstractmethod
    def observe(self, enc: senc.ENC, obstacle_) -> np.ndarray:
        """Get an observation of the environment state."""


class LidarLikeObservation(ObservationType):
    """A lidar-like observation space for the own-ship, i.e. a 360 degree scan of the environment as is used in Meyer et. al. 2020."""

    def __init__(
        self,
    ) -> None:
        super().__init__(env, **kwargs)
        self.lol = "lol"
        self.n_obs = 360

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_obs,))

    def observe(self) -> np.ndarray:
        """Get an observation of the environment state."""
        reward_insight = self.rewarder.insight()
        sector_closenesses, sector_velocities = self.ship.perceive(self.obstacles)

        obs = np.concatenate([reward_insight, navigation_states, sector_closenesses, sector_velocities])
        return obs


def observation_factory(env: str = "AbstractEnv", observation_type: str = "ContinuousAutopilotReferenceAction") -> ObservationType:
    """Factory for creating observation spaces.

    Args:
        env (str, optional): Name of environment. Defaults to "AbstractEnv".
        observation_type (str, optional): Action type name. Defaults to "LidarLikeObservation".

    Returns:
        ActionType: Action type to use
    """
    if observation_type == "LidarLikeObservation":
        return LidarLikeObservation(env)
    else:
        raise ValueError("Unknown action type")
