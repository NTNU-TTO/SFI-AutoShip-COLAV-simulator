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

import colav_simulator.core.guidances as guidances
import colav_simulator.core.stochasticity as stoch
import gymnasium as gym
import numpy as np

Action = Union[list, np.ndarray]
import colav_simulator.common.math_functions as mf

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


class ActionType(ABC):
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    name: str = "AbstractAction"

    def __init__(self, env: "COLAVEnvironment", sample_time: Optional[float] = None) -> None:
        self.env = env
        self._ownship = self.env.ownship
        self.sample_time = sample_time if sample_time is not None else env.simulator.dt

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """The action space."""

    def get_sampling_time(self) -> float:
        """The action sampling time for the action space, i.e the time between each action computation from the agent/policy.

        Returns:
            float: The sampling time.
        """
        return self.sample_time

    @abstractmethod
    def normalize(self, action: Action) -> Action:
        """Normalize the action to the action space.

        Args:
            action (Action): The action to normalize.

        Returns:
            Action: The normalized action.
        """

    @abstractmethod
    def unnormalize(self, action: Action) -> Action:
        """Unnormalize the action to the action space.

        Args:
            action (Action): The action to unnormalize.

        Returns:
            Action: The unnormalized action.
        """

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

    def __init__(self, env: "COLAVEnvironment", sample_time: Optional[float] = None, **kwargs) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course.

        Args:
            env (str, optional): Name of environment. Defaults to "AbstractEnv".
        """
        super().__init__(env, sample_time)
        assert self._ownship is not None, "Ownship must be set before using the action space"
        self.size = 2
        self.course_range = (-np.pi, np.pi)
        self.speed_range = (self._ownship.min_speed, self._ownship.max_speed)
        self.last_action = np.zeros(self.size)
        self.name = "ContinuousAutopilotReferenceAction"

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    def normalize(self, action: Action) -> Action:
        course_ref = mf.linear_map(action[0], self.course_range, (-1.0, 1.0))
        speed_ref = mf.linear_map(action[1], self.speed_range, (-1.0, 1.0))
        return np.array([course_ref, speed_ref])

    def unnormalize(self, action: Action) -> Action:
        course_ref = mf.linear_map(action[0], (-1.0, 1.0), self.course_range)
        speed_ref = mf.linear_map(action[1], (-1.0, 1.0), self.speed_range)
        return np.array([course_ref, speed_ref])

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


class ContinuousRelativeLOSReferenceAction(ActionType):
    """
    Action composed of

    a = [x_ld_rel, y_ld_rel, speed_ref_rel],

    where
        - x_ld_rel and y_ld_rel are the relative north and east distances to the look-ahead point which a simple LOS/WPP guidance
    algorithm uses to compute the low-level motion control course reference.
        - speed_ref_rel is the relative speed reference to the own-ship speed.

    When applying action a, the environmental disturbances are estimated and compensated for in the course reference.
    """

    def __init__(self, env: "COLAVEnvironment", sample_time: Optional[float] = None, **kwargs) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course.

        Args:
            env (str, optional): Name of environment. Defaults to "AbstractEnv".
        """
        super().__init__(env, sample_time)
        assert self._ownship is not None, "Ownship must be set before using the action space"
        self.size = 3
        self.position_range = (-1000, 1000)
        self.turn_rate_range = (-self._ownship.max_turn_rate, self._ownship.max_turn_rate)
        self.speed_range = (-self._ownship.max_speed, self._ownship.max_speed)
        self.last_action = np.zeros(self.size)
        self.name = "ContinuousRelativeLOSReferenceAction"
        self._ma_filter = stoch.MovingAverageFilter(input_dim=2, window_size=10)

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    def normalize(self, action: Action) -> Action:
        x_ld_rel = mf.linear_map(action[0], self.position_range, (-1.0, 1.0))
        y_ld_rel = mf.linear_map(action[1], self.position_range, (-1.0, 1.0))
        speed_ref_rel = mf.linear_map(action[2], self.speed_range, (-1.0, 1.0))
        # turn_rate_ref_rel = mf.linear_map(action[3], self.turn_rate_range, (-1.0, 1.0))
        return np.array([x_ld_rel, y_ld_rel, speed_ref_rel])

    def unnormalize(self, action: Action) -> Action:
        x_ld_rel = mf.linear_map(action[0], (-1.0, 1.0), self.position_range)
        y_ld_rel = mf.linear_map(action[1], (-1.0, 1.0), self.position_range)
        speed_ref_rel = mf.linear_map(action[2], (-1.0, 1.0), self.speed_range)
        # turn_rate_ref_rel = mf.linear_map(action[3], (-1.0, 1.0), self.turn_rate_range)
        return np.array([x_ld_rel, y_ld_rel, speed_ref_rel])

    def compute_disturbance_velocity_estimate(
        self,
    ) -> np.ndarray:
        """Computes the disturbance velocity estimate for the given time.

        Returns:
            np.ndarray: The disturbance velocity estimate.
        """
        w = self.env.disturbance.get() if self.env.disturbance is not None else None
        v_disturbance = np.array([0.0, 0.0])
        if w is None:
            return v_disturbance

        if "speed" in w.wind:
            V_w = w.wind["speed"]
            beta_w = w.wind["direction"]
            v_w = np.array([V_w * np.cos(beta_w), V_w * np.sin(beta_w)])
            v_disturbance += v_w
        if "speed" in w.currents:
            V_c = w.currents["speed"]
            beta_c = w.currents["direction"]
            v_c = np.array([V_c * np.cos(beta_c), V_c * np.sin(beta_c)])
            v_disturbance += v_c
        return self._ma_filter.update(v_disturbance)

    def act(self, action: Action) -> None:
        """Execute the action on the own-ship, which is to apply new (relative) autopilot references for course and speed.

        If disturbances are present, a feed forward term is used to compensate for their direction(s) in the course reference calculation.

        Args:
            action (Action): New action a = [x_ld_rel, y_ld_rel, speed_ref_rel] within [-1, 1].
        """
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        unnorm_action = self.unnormalize(action)
        x_ld = self._ownship.state[0] + unnorm_action[0]
        y_ld = self._ownship.state[1] + unnorm_action[1]

        speed_ref = self._ownship.csog_state[2] + unnorm_action[2]
        chi_ref = np.arctan2(y_ld - self._ownship.state[1], x_ld - self._ownship.state[0])

        # Disturbance feed forward in course reference computation
        v_disturbance = self.compute_disturbance_velocity_estimate()
        chi_disturbance = np.arctan2(v_disturbance[1], v_disturbance[0])
        chi_ref -= chi_disturbance
        refs = np.array([0.0, 0.0, mf.wrap_angle_to_pmpi(chi_ref), speed_ref, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
        self.last_action = action
        self._ownship.set_references(refs)


def action_factory(
    env: "COLAVEnvironment",
    action_type: Optional[str] = "continuous_relative_autopilot_reference_action",
    sample_time: Optional[float] = None,
    **kwargs
) -> ActionType:
    """Factory for creating action spaces.

    Args:
        env (str, optional): Name of environment. Defaults to COLAVEnvironment.
        action_type (str, optional): Action type name. Defaults to "continuous_relative_autopilot_reference_action".
        sample_time (float, optional): The time between each action computation from the agent/policy. Defaults to None.

    Returns:
        ActionType: Action type to use.
    """
    if action_type == "continuous_autopilot_reference_action":
        return ContinuousAutopilotReferenceAction(env, sample_time=sample_time, **kwargs)
    elif action_type == "continuous_relative_los_reference_action":
        return ContinuousRelativeLOSReferenceAction(env, sample_time=sample_time, **kwargs)
    else:
        raise ValueError("Unknown action type")
