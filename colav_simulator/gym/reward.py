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

Action = TypeVar("Action")
Observation = TypeVar("Observation")


class IReward(ABC):
    def __init__(self, ownship: Ship) -> None:
        self.ownship = ownship

    @abstractmethod
    def __call__(self, state: Observation, action: Action) -> float:
        """Get the reward for the current state-action pair."""


class DistanceToGoalReward(IReward):
    """Reward the agent for getting closer to the goal."""

    def __init__(self, ownship: Ship, goal: np.ndarray) -> None:
        """Initializes the reward function.

        Args:
            ownship (Ship): The ownship
            goal (np.ndarray): The goal position [x_g, y_g]^T
        """
        super().__init__(ownship)
        self.goal = goal

    def __call__(self, state: Observation, action: Action) -> float:
        state = self.ownship.state
        return -np.linalg.norm(state[:2] - self.goal)
