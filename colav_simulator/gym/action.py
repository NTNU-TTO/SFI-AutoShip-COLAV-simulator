"""
    action.py

    Summary:
        This file contains various action space definitions for a ship agent in the colav-simulator.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from colav_simulator.core.ship import Ship

Action = Union[list, int, np.ndarray]

if TYPE_CHECKING:
    from colav_simulator.gym.environment import BaseEnvironment


class ActionType(ABC):
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    name: str = "AbstractAction"

    def __init__(self, env: "BaseEnvironment", ownship: Ship) -> None:
        self.env = env
        self.__ownship = ownship

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """The action space."""

    @abstractmethod
    def act(self, action: Action) -> None:
        """Execute the action on the own-ship

        Args:
            action (Action): The action to execute. Typically the output autopilot references from the COLAV-algorithm.

        """

    @property
    def ownship(self):
        """The ownship acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__ownship or self.env.ownship

    @ownship.setter
    def ownship(self, ship):
        self.__ownship = ship


class ContinuousAutopilotReferenceAction(ActionType):

    """
    A continuous action space for setting the own-ship autopilot references in speed and course, i.e.
    action a = [speed_ref, course_ref].
    """

    SPEED_RANGE = (-5.0, 10.0)
    """speed range: [-x, x], in m/s."""

    COURSE_RANGE = (-np.pi, np.pi)
    """Steering angle range: [-x, x], in rad."""

    def __init__(
        self,
        env: "BaseEnvironment",
        ownship: Ship,
        speed_range: Optional[Tuple[float, float]] = None,
        course_range: Optional[Tuple[float, float]] = None,
        clip: bool = True,
        **kwargs
    ) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course.

        Args:
            env (str, optional): Name of environment. Defaults to "AbstractEnv".
            ownship (Ship): The ownship to act upon.
            speed_range (Optional[Tuple[float, float]]): The range of speed references. Defaults to None.
            course_range (Optional[Tuple[float, float]]): The range of course references. Defaults to None.
            clip (bool, optional): Clip action to defined range. Defaults to True.
        """
        super().__init__(env, ownship)

        self.speed_range = speed_range if speed_range else self.SPEED_RANGE
        self.course_range = course_range if course_range else self.COURSE_RANGE
        self.size = 2
        self.clip = clip
        self.last_action = np.zeros(self.size)
        self.name = "ContinuousAutopilotReferenceAction"

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 1.0, shape=(self.size,), dtype=np.float32)

    def act(self, action: Action) -> None:
        # act
        self.last_action = action


def action_factory(env: "BaseEnvironment", ownship: Ship, action_type: str = "ContinuousAutopilotReferenceAction") -> ActionType:
    """Factory for creating action spaces.

    Args:
        env (str, optional): Name of environment. Defaults to BaseEnvironment.
        action_type (str, optional): Action type name. Defaults to "ContinuousAutopilotReferenceAction".

    Raises:
        ValueError: Unknown action type

    Returns:
        ActionType: Action type to use
    """
    if action_type == "ContinuousAutopilotReferenceAction":
        return ContinuousAutopilotReferenceAction(env, ownship)
    else:
        raise ValueError("Unknown action type")
