"""
    action.py

    Summary:
        This file contains various action space definitions for a ship agent in the colav-simulator.

    Author: Trym Tengesdal
"""
import itertools
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

Action = Union[list, int, np.ndarray]


class ActionType(ABC):

    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: str = "AbstractEnv", **kwargs) -> None:
        self.env = env
        self.__ownship = None

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
        self, env: str = "AbstractEnv", speed_range: Optional[Tuple[float, float]] = None, course_range: Optional[Tuple[float, float]] = None, clip: bool = True, **kwargs
    ) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course.

        Args:
            env (str, optional): Name of environment. Defaults to "AbstractEnv".
            speed_range (Optional[Tuple[float, float]]): The range of speed references. Defaults to None.
            course_range (Optional[Tuple[float, float]]): The range of course references. Defaults to None.
            clip (bool, optional): Clip action to defined range. Defaults to True.
        """
        super().__init__(env)

        self.speed_range = speed_range if speed_range else self.SPEED_RANGE
        self.course_range = course_range if course_range else self.COURSE_RANGE
        self.size = 2
        self.clip = clip
        self.last_action = np.zeros(self.size)

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            np.array([self.course_range[0], self.speed_range[0]]), np.array([self.course_range[1], self.speed_range[1]]), shape=(self.size,), dtype=np.float32
        )

    def act(self, action: np.ndarray) -> None:
        # act
        self.last_action = action


class DiscreteAutopilotReferenceAction(ContinuousAutopilotReferenceAction):
    """Discrete action space for setting the own-ship autopilot references in speed and course."""

    def __init__(
        self,
        env: str = "AbstractEnv",
        speed_range: Optional[Tuple[float, float]] = None,
        course_range: Optional[Tuple[float, float]] = None,
        num_speed_actions: int = 3,
        num_course_actions: int = 12,
        clip: bool = True,
        **kwargs
    ) -> None:
        super().__init__(env, speed_range=speed_range, course_range=course_range, clip=clip)
        self.num_speed_actions = num_speed_actions
        self.num_course_actions = num_course_actions

    def space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.num_speed_actions * self.num_course_actions)

    def act(self, action: int) -> None:
        cont_space = super().space()
        speed_axes = np.linspace(cont_space.low, cont_space.high, self.num_speed_actions).T
        course_axes = np.linspace(cont_space.low, cont_space.high, self.num_course_actions).T
        all_actions = list(itertools.product(*speed_axes, *course_axes))
        super().act(all_actions[action])


def action_factory(env: str = "AbstractEnv", action_type: str = "ContinuousAutopilotReferenceAction") -> ActionType:
    """Factory for creating action spaces.

    Args:
        env (str, optional): Name of environment. Defaults to "AbstractEnv".
        action_type (str, optional): Action type name. Defaults to "ContinuousAutopilotReferenceAction".

    Raises:
        ValueError: Unknown action type

    Returns:
        ActionType: Action type to use
    """
    if action_type == "ContinuousAutopilotReferenceAction":
        return ContinuousAutopilotReferenceAction(env)
    if action_type == "DiscreteAutopilotReferenceAction":
        return DiscreteAutopilotReferenceAction(env)
    else:
        raise ValueError("Unknown action type")
