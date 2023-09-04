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

Action = Union[list, np.ndarray]

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


class ActionType(ABC):
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    name: str = "AbstractAction"

    def __init__(self, env: "COLAVEnvironment") -> None:
        self.env = env
        self.__ownship = self.env.ownship

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

    def __init__(self, env: "COLAVEnvironment", **kwargs) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course.

        Args:
            env (str, optional): Name of environment. Defaults to "AbstractEnv".
            ownship (Ship): The ownship to act upon (reference to object).
        """
        super().__init__(env)
        self.size = 2
        self.last_action = np.zeros(self.size)
        self.name = "ContinuousAutopilotReferenceAction"

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    def act(self, action: Action) -> None:
        """Execute the action on the own-ship, which is to apply new autopilot references for course and speed.

        Args:
            action (Action): New course and speed references [course_ref, speed_ref]
        """
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        # ship references in general is a 9-entry array consisting of 3DOF pose, velocity and acceleartion
        refs = np.array([0.0, 0.0, action[0], action[1], 0.0, 0.0, 0.0, 0.0, 0.0])
        self.last_action = action
        self.__ownship.set_references(refs)


def action_factory(env: "COLAVEnvironment", action_type: Optional[str] = "continuous_autopilot_reference_action") -> ActionType:
    """Factory for creating action spaces.

    Args:
        env (str, optional): Name of environment. Defaults to COLAVEnvironment.
        action_type (str, optional): Action type name. Defaults to "continuous_autopilot_reference_action".

    Returns:
        ActionType: Action type to use
    """
    if action_type == "continuous_autopilot_reference_action":
        return ContinuousAutopilotReferenceAction(env)
    else:
        raise ValueError("Unknown action type")
