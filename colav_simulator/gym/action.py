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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np

Action = Union[np.ndarray, Dict[str, float | np.ndarray]]
import colav_simulator.common.math_functions as mf

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


@dataclass
class ActionResult:
    """Result of an action execution"""

    success: bool
    info: Dict[str, Any]


class ActionType(ABC):
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    name: str = "AbstractAction"

    def __init__(self, env: "COLAVEnvironment", sample_time: Optional[float] = None) -> None:
        self.env = env
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
    def act(self, action: Action, **kwargs) -> ActionResult:
        """Execute the action on the own-ship. Extra arguments can be passed to the act method.

        Args:
            action (Action): The action to execute (normalized). Typically the output autopilot references from the COLAV-algorithm.

        Returns:
            - ActionResult: The result information from the action execution (if any).
        """


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
        assert self.env.ownship is not None, "Ownship must be set before using the action space"
        self.size = 2
        self.course_range = (-np.pi, np.pi)
        self.speed_range = (self.env.ownship.min_speed, self.env.ownship.max_speed)
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

    def act(self, action: Action, **kwargs) -> ActionResult:
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
        self.env.ownship.set_references(refs)
        return ActionResult(success=True, info={})


class RelativeCourseSpeedReferenceSequenceAction(ActionType):
    """(Continuous) Action consisting of a sequence of course (COG) and speed (SOG) references to be followed by the own-ship."""

    def __init__(
        self, env: "COLAVEnvironment", sample_time: Optional[float] = None, seq_length: int = 1, **kwargs
    ) -> None:
        """Create a continuous action space for setting the own-ship autopilot references in speed and course.

        Args:
            env (str, optional): Name of environment. Defaults to "AbstractEnv".
        """
        super().__init__(env, sample_time)
        assert self.env.ownship is not None, "Ownship must be set before using the action space"
        self.seq_length = seq_length  # Enhancement: make the sequence length a parameter
        self.course_range = (-np.pi / 4.0, np.pi / 4.0)
        self.speed_range = (-self.env.ownship.max_speed / 4.0, self.env.ownship.max_speed / 4.0)
        self.name = "RelativeCourseSpeedReferenceSequenceAction"
        self.course_refs = np.zeros(seq_length)
        self.speed_refs = np.zeros(seq_length)
        self.t_first_apply = 0.0
        self.reference_duration = 2.0  # equal to sampling time used in the MPC/planner
        self.ref_counter = 0

    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(self.seq_length * 2,), dtype=np.float32)

    def normalize(self, action: list | np.ndarray) -> list | np.ndarray:
        action_norm = np.zeros(self.seq_length * 2)
        for i in range(self.seq_length):
            action_norm[i * 2] = mf.linear_map(action[i * 2], self.course_range, (-1.0, 1.0))
            action_norm[i * 2 + 1] = mf.linear_map(action[i * 2 + 1], self.speed_range, (-1.0, 1.0))
        return action_norm

    def unnormalize(self, action: list | np.ndarray) -> list | np.ndarray:
        action_unnorm = np.zeros(self.seq_length * 2)
        for i in range(self.seq_length):
            action_unnorm[i * 2] = mf.linear_map(action[i * 2], (-1.0, 1.0), self.course_range)
            action_unnorm[i * 2 + 1] = mf.linear_map(action[i * 2 + 1], (-1.0, 1.0), self.speed_range)
        return action_unnorm

    def act(self, action: Action, **kwargs) -> ActionResult:
        """Execute the action on the own-ship, which is to apply new autopilot references for course and speed.

        Args:
            action (Action): New course and speed references [course_ref, speed_ref] within [-1, 1]
        """
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        assert (
            "applied" in kwargs
        ), "This action type needs to know if the current action is the first one applied in the current episode step."
        if not kwargs["applied"]:
            self.ref_counter = 0
            unnorm_action = self.unnormalize(action)
            # ship references in general is a 9-entry array consisting of 3DOF pose, velocity and acceleartion
            course = self.env.ownship.course
            speed = self.env.ownship.speed
            self.course_refs = np.zeros(self.seq_length)
            self.speed_refs = np.zeros(self.seq_length)
            for i in range(self.seq_length):
                self.course_refs[i] = mf.wrap_angle_to_pmpi(unnorm_action[i * 2] + course)
                self.speed_refs[i] = unnorm_action[i * 2 + 1] + speed
            self.t_first_apply = self.env.time
            self.speed_refs = np.clip(self.speed_refs, self.env.ownship.min_speed + 0.5, self.env.ownship.max_speed)

        t_now = self.env.time
        if t_now - self.t_first_apply > self.reference_duration:
            self.ref_counter += 1 if self.ref_counter < self.seq_length - 1 else 0

        course_ref = self.course_refs[self.ref_counter]
        speed_ref = self.speed_refs[self.ref_counter]
        refs = np.array([0.0, 0.0, course_ref, speed_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.env.ownship.set_references(refs)
        return ActionResult(success=True, info={})


def action_factory(
    env: "COLAVEnvironment",
    action_type: Optional[str] = "continuous_autopilot_reference_action",
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
    elif action_type == "relative_course_speed_reference_sequence_action":
        return RelativeCourseSpeedReferenceSequenceAction(env, sample_time=sample_time, **kwargs)
    else:
        raise ValueError("Unknown action type")
