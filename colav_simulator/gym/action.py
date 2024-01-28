"""
    action.py

    Summary:
        This file contains various action type definitions for a ship agent in the colav-simulator.

        To add an action type:
        1: Create a new class inheriting from ActionType and implement the abstract methods.
        2: Add the new class to the `action_factory` method. Make sure lower case (snake case) string names are used for specifying the action type.
        3: Add the new action type to your scenario config file.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import gymnasium as gym
import numpy as np

Action = Union[list, np.ndarray]
import colav_simulator.common.math_functions as mf

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


class ActionType(ABC):
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    name: str = "AbstractAction"

    def __init__(self, env: "COLAVEnvironment") -> None:
        self.env = env
        self._ownship = self.env.ownship

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """The action space."""

    @abstractmethod
    def act(self, action: Action) -> None:
        """Execute the action on the own-ship

        Args:
            action (Action): The action to execute (normalized). Typically the output autopilot references from the COLAV-algorithm.
        """

    @property
    def ownship(self):
        """The ownship acted upon."""
        return self._ownship

    @ownship.setter
    def ownship(self, ship):
        self._ownship = ship


class ContinuousAutopilotReferenceAction(ActionType):

    """
    A continuous action space for setting the own-ship autopilot references in speed and course, i.e.
    action a = [speed_ref, course_ref].
    """

    def __init__(self, env: "COLAVEnvironment", **kwargs) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course.

        Args:
            env (str, optional): Name of environment. Defaults to "AbstractEnv".
        """
        super().__init__(env)
        assert self._ownship is not None, "Ownship must be set before using the action space"
        self.size = 2
        self.course_range = (-np.pi, np.pi)
        self.speed_range = (self._ownship.min_speed, self._ownship.max_speed)
        self.last_action = np.zeros(self.size)
        self.name = "ContinuousAutopilotReferenceAction"

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    def act(self, action: Action) -> None:
        """Execute the action on the own-ship, which is to apply new autopilot references for course and speed.

        Args:
            action (Action): New course and speed references [course_ref, speed_ref] within [-1, 1]
        """
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        # ship references in general is a 9-entry array consisting of 3DOF pose, velocity and acceleartion
        course_ref = mf.linear_map(action[0], (-1.0, 1.0), self.course_range)
        speed_ref = mf.linear_map(action[1], (-1.0, 1.0), self.speed_range)
        refs = np.array([0.0, 0.0, course_ref, speed_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.last_action = action
        self._ownship.set_references(refs)


def action_factory(
    env: "COLAVEnvironment", action_type: Optional[str] = "continuous_autopilot_reference_action"
) -> ActionType:
    """Factory for creating action spaces.

    Args:
        env (str, optional): Name of environment. Defaults to COLAVEnvironment.
        action_type (str, optional): Action type name. Defaults to "continuous_autopilot_reference_action".

    Returns:
        ActionType: Action type to use.
    """
    if action_type == "continuous_autopilot_reference_action":
        return ContinuousAutopilotReferenceAction(env)
    else:
        raise ValueError("Unknown action type")
