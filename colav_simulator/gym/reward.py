"""
    reward.py

    Summary:
        This file contains classes used for giving reward signals to the agent in the colav-simulator environment.

    Author: Trym Tengesdal
"""

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Union

import colav_simulator.core.sensing as sensing
import colav_simulator.core.tracking.trackers as trackers
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
from colav_simulator.core.ship import Ship

if TYPE_CHECKING:
    from colav_simulator.gym.environment import BaseEnvironment


class BaseReward(ABC):
    def __init__(self, env: "BaseEnvironment", ownship: Ship) -> None:
        self.env = env
        self.ship = ownship

    @abstractmethod
    def reward(self) -> float:
        """Get the reward for the current state of the environment."""
