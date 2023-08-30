"""
    observation.py

    Summary:
        This file contains various observation space definitions for a ship agent in the colav-simulator.

    Author: Trym Tengesdal
"""

import itertools
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd


class ObservationType(ABC):
    def __init__(self, env: str = "AbstractEnv", **kwargs) -> None:
        self.env = env
        self.__observer_ship = None

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """Get the observation space."""

    @abstractmethod
    def observe(self):
        """Get an observation of the environment state."""

    @property
    def observer_ship(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_ship or self.env.ownship

    @observer_ship.setter
    def observer_ship(self, ship):
        self.__observer_ship = ship


class LidarLikeObservation(ObservationType):
    """A lidar-like observation space for the own-ship, i.e. a 360 degree scan of the environment as is used in Meyer et. al. 2020."""

    def __init__(self, env: str = "AbstractEnv", **kwargs) -> None:
        super().__init__(env, **kwargs)

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(360,))

    def observe(self) -> np.ndarray:
        """Get an observation of the environment state."""
        reward_insight = self.rewarder.insight()

        if bool(self.config["sensing"]):
            sector_closenesses, sector_velocities = self.vessel.perceive(self.obstacles)
        else:
            sector_closenesses, sector_velocities = [], []

        obs = np.concatenate([reward_insight, navigation_states, sector_closenesses, sector_velocities])
        return obs
