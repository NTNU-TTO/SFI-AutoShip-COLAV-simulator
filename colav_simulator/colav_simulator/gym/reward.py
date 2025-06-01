"""
    reward.py

    Summary:
        This file contains some simple rewarder classes for giving reward signals to the agent in the colav-simulator environment.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from scipy.optimize import fsolve

import colav_simulator.gym.action as csgym_action
import colav_simulator.gym.observation as csgym_obs
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_evaluation_tool.common.miscellaneous_helper_methods as mhm_eval
import colav_simulator.common.math_functions as mf
import colav_simulator.core.colav.sbmpc.helper_functions as sbmpc_hf
from colav_evaluation_tool.evaluator import Evaluator
import numpy as np
import pandas as pd
import os

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


@dataclass
class ExistenceRewardParams:
    """Parameters for the Existence rewarder."""

    r_exists: float = 1.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = ExistenceRewardParams()
        cfg.r_exists = config_dict["r_exists"]
        return cfg

    def to_dict(self):
        return {"r_exists": self.r_exists}


@dataclass
class CollisionRewardParams:
    """Parameters for the Collision rewarder."""

    r_collision: float = 1000.0 #500.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = CollisionRewardParams()
        cfg.r_collision = config_dict["r_collision"]
        return cfg

    def to_dict(self):
        return {"r_collision": self.r_collision}


@dataclass
class GroundingRewardParams:
    """Parameters for the Grounding rewarder."""

    r_grounding: float = 1000.0 #500.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = GroundingRewardParams()
        cfg.r_grounding = config_dict["r_grounding"]
        return cfg

    def to_dict(self):
        return {"r_grounding": self.r_grounding}


@dataclass
class DistanceToGoalRewardParams:
    """Paramters for the distance to goal rewarder."""

    r_d2g: float = 1.0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = DistanceToGoalRewardParams()
        cfg.r_d2g = config_dict["r_d2g"]
        return cfg

    def to_dict(self):
        return {"r_d2g": self.r_d2g}


@dataclass
class TrajectoryTrackingRewardParams:
    """Parameters for the trajectory tracking rewarder."""

    gamma_r: float = 1.0  # Reward constant
    gamma_e: float = 0.5  # Cross-track error coefficient
    gamma_u: float = 0.8  # Speed error coefficient
    gamma_chi: float = 1.5  # Course angle error coefficient

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = TrajectoryTrackingRewardParams()
        config.gamma_r = config_dict["gamma_r"]
        config.gamma_e = config_dict["gamma_e"]
        config.gamma_u = config_dict["gamma_u"]
        config.gamma_chi = config_dict["gamma_chi"]
        return config


@dataclass
class StaticColavRewardParams:
    """Parameters for the static COLAV rewarder."""

    alpha_x = 75
    gamma_x = 0.1
    gamma_theta = 10

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = StaticColavRewardParams()

        config.alpha_x = config_dict["alpha_x"]
        config.gamma_x = config_dict["gamma_x"]
        config.gamma_theta = config_dict["gamma_theta"]

        return config


@dataclass
class DynamicColavRewardParams:
    """Parameters for the dynamic COLAV rewarder."""

    # Raw COLAV penalty scaling
    alpha_x: float = 75.0
    # Sensor angle coefficient
    gamma_theta: float = 1.0

    # Velocity multipliers
    gamma_v_stb: float = 0.09
    gamma_v_port: float = 0.01
    gamma_v_stern: float = 0.02

    # Distance multipliers
    gamma_x_stb: float = 7e-3
    gamma_x_port: float = 9e-3
    gamma_x_stern: float = 1e-2

    # Trade-off coefficient parameters.
    # The lists are given as [-, +] for negative and positive TS velocities.
    gamma_lambda: list = field(default_factory=lambda: [3e-5, 3e-3])
    alpha_lambda: list = field(default_factory=lambda: [2, 4])

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = DynamicColavRewardParams()

        config.alpha_x = config_dict["alpha_x"]
        config.gamma_theta = config_dict["gamma_theta"]
        config.gamma_v_stb = config_dict["gamma_v_stb"]
        config.gamma_v_port = config_dict["gamma_v_port"]
        config.gamma_v_stern = config_dict["gamma_v_stern"]
        config.gamma_x_stb = config_dict["gamma_x_stb"]
        config.gamma_x_port = config_dict["gamma_x_port"]
        config.gamma_x_stern = config_dict["gamma_x_stern"]
        config.gamma_lambda = config_dict["gamma_lambda"]
        config.alpha_lambda = config_dict["alpha_lambda"]

        return config


@dataclass
class AutopilotReferenceRewardParams:
    """Parameters for the autopilot reference rewarder."""

    course_coefficient: float = 1.0
    speed_coefficient: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = AutopilotReferenceRewardParams()

        cfg.course_coefficient = config_dict["course_coefficient"]
        cfg.speed_coefficient = config_dict["speed_coefficient"]

        return cfg


@dataclass
class EvaluatorRewardParams:
    """Parameters for the Evaluator rewarder."""

    r_max: float = 500
    zero_thres: float = 0.74

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = EvaluatorRewardParams()
        cfg.r_max = config_dict["r_max"]
        cfg.zero_thres = config_dict["zero_thres"]
        return cfg


@dataclass
class TrajectoryTrackingIndRewarderParams:
    """From https://github.com/NTNU-Autoship-Internal/rlmpc/blob/main/rlmpc/rewards.py"""
    rho_d2path: float = 0.11 #1.0  # path deviation reward weight
    rho_inverse_d2path: float = 0.0 #0.01 # inverse path distance reward weight (higher penalty for small deviations)
    rho_speed_dev: float = 0.0 #10.0  # speed deviation reward weight
    rho_d2goal: float = 0.0 #0.1  # final path deviation reward weight
    rho_course_dev: float = 0.0  # course deviation reward weight
    rho_turn_rate: float = 0.0  # turn rate reward weight
    rho_goal: float = 0.0 #100.0  # penalty for not reaching the goal
    goal_radius: float = 30.0  # radius around the goal point where the goal is considered reached

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = cls(**config_dict)
        return params


@dataclass
class ReadilyApparentManeuveringRewarderParams:
    """From https://github.com/NTNU-Autoship-Internal/rlmpc/blob/main/rlmpc/rewards.py"""
    K_app_course: float = 15.0  # rate cost weight for turn rate
    K_app_speed: float = 10.0  # rate cost weight for speed
    alpha_app_course: np.ndarray = field(
        default_factory=lambda: np.array([112.5, 0.00006])
    )  # Rate cost function parameters for turn rate
    alpha_app_speed: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 0.00025])
    )  # Rate cost function parameters for speed
    r_max: float = 6.0  # Maximum turn rate
    a_max: float = 2.0  # Maximum acceleration

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = cls()
        params.K_app_course = config_dict["K_app_course"]
        params.K_app_speed = config_dict["K_app_speed"]
        params.alpha_app_course = np.array(config_dict["alpha_app_course"])
        params.alpha_app_speed = np.array(config_dict["alpha_app_speed"])
        params.r_max = np.deg2rad(config_dict["r_max"])
        params.a_max = config_dict["a_max"]
        return params

    def to_dict(self) -> dict:
        return {
            "K_app_course": self.K_app_course,
            "K_app_speed": self.K_app_speed,
            "alpha_app_course": self.alpha_app_course.tolist(),
            "alpha_app_speed": self.alpha_app_speed.tolist(),
            "r_max": np.rad2deg(self.r_max),
            "a_max": self.a_max,
        }
    
@dataclass
class ActionChatterRewarderParams:
    rho_chatter: np.ndarray = 0.5 #0.5

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = cls(**config_dict)
        cfg.rho_chatter = cfg.rho_chatter
        return cfg

    def to_dict(self) -> dict:
        return { "rho_chatter": self.rho_chatter }
    
@dataclass
class GroundingAvoidanceRewarderParams:
    rho_ga: float = 50.0 #1.0    # Grounding avoidance reward weight
    safe_distance: float = 200.0  # [m] No penalty beyond this distance
    r_max: float = 50.0            # Maxmium reward (cost) per grounding hazard

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)


class IReward(ABC):
    """Interface/base class for reward functions. The rewarder parameters should be public."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        self.env = env

    @abstractmethod
    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        """Get the reward for the current state-action pair. Additional arguments can be passed if necessary."""

    @abstractmethod
    def get_last_rewards_as_dict(self) -> dict:
        """Get the last rewards as a dictionary with the individual reward components and their values."""


@dataclass
class Config:
    """Configuration class of parameters for the rewarder."""

    rewarders: list = field(default_factory=lambda: [ExistenceRewardParams()])

    @classmethod
    def from_dict(cls, config_dict: dict):
        cfg = Config()
        rewarders = config_dict["rewarders"]
        for rewarder in rewarders:
            if "existence_rewarder" in rewarder:
                cfg.rewarders.append(ExistenceRewardParams.from_dict(rewarder))
            elif "distance_to_goal_rewarder" in rewarder:
                cfg.rewarders.append(DistanceToGoalRewardParams.from_dict(rewarder))
            elif "collision_rewarder" in rewarder:
                cfg.rewarders.append(CollisionRewardParams.from_dict(rewarder))
            elif "grounding_rewarder" in rewarder:
                cfg.rewarders.append(GroundingRewardParams.from_dict(rewarder))
            elif "trajectory_tracking_rewarder" in rewarder:
                cfg.rewarders.append(TrajectoryTrackingRewardParams.from_dict(rewarder))
            elif "static_colav_rewarder" in rewarder:
                cfg.rewarders.append(StaticColavRewardParams.from_dict(rewarder))
            elif "dynamic_colav_rewarder" in rewarder:
                cfg.rewarders.append(DynamicColavRewardParams.from_dict(rewarder))
            elif "autopilot_reference_rewarder" in rewarder:
                cfg.rewarders.append(AutopilotReferenceRewardParams.from_dict(rewarder))
            elif "grounding_avoidance_rewarder" in rewarder:
                cfg.rewarders.append(GroundingAvoidanceRewarderParams.from_dict(rewarder))
        return cfg

    def to_dict(self) -> dict:
        rewarders = []
        for rewarder in self.rewarders:
            if isinstance(rewarder, ExistenceRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, DistanceToGoalRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, CollisionRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, GroundingRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, TrajectoryTrackingRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, StaticColavRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, DynamicColavRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, AutopilotReferenceRewardParams):
                rewarders.append(rewarder.to_dict())
            elif isinstance(rewarder, GroundingAvoidanceRewarderParams):
                rewarders.append(rewarder.to_dict())
        return {"rewarders": rewarders}


class ExistenceRewarder(IReward):
    """Reward the agent negatively for existing."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[ExistenceRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            params (ExistenceRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.last_reward = 0.0
        self.params = params if params else ExistenceRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        self.last_reward = -self.params.r_exists
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_exists": self.last_reward}


class DistanceToGoalRewarder(IReward):
    """Reward the agent for getting closer to the goal."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[DistanceToGoalRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            params (DistanceToGoalRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.last_reward = 0.0
        self.goal = np.array([0.0, 0.0])
        self.params = params if params else DistanceToGoalRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.ownship.goal_csog_state.size > 0:
            self.goal = self.env.ownship.goal_csog_state[:2]
        elif self.env.ownship.waypoints.size > 1:
            self.goal = self.env.ownship.waypoints[:, -1]
        else:
            raise ValueError("No goal state or waypoints found")
        self.last_reward = -self.params.r_d2g * float(np.linalg.norm(self.env.ownship.csog_state[:2] - self.goal))
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_d2g": self.last_reward}


class CollisionRewarder(IReward):
    """Reward the agent negatively for colliding."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[CollisionRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            env (COLAVEnvironment): The environment
            params (CollisionRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.last_reward = 0.0
        self.params = params if params else CollisionRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        """Reward for the current state-action pair. Calls function from simulator class to get collision or not."""
        collision = self.env.simulator.determine_ship_collision()
        if collision:
            self.last_reward = -self.params.r_collision
        else:
            self.last_reward = 0.0
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_collision": self.last_reward}


class GroundingRewarder(IReward):
    """Reward the agent negatively for grounding."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[CollisionRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            env (COLAVEnvironment): The environment
            params (GroundingRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.last_reward = 0.0
        self.params = params if params else GroundingRewardParams()

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        """Reward for the current state-action pair. Calls function from simulator class to get grounding or not."""
        grounding = self.env.simulator.determine_ship_grounding()
        if grounding:
            self.last_reward = -self.params.r_grounding
        else:
            self.last_reward = 0.0
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_grounding": self.last_reward}


class TrajectoryTrackingRewarder(IReward):
    """Trajectory tracking rewarder based on Meyer et.al.(2020). Extended to include speed error term.
    NOTE: Both the DynamicColavRewarder and NavigationPathObservation are required
    for this rewarder to function.
    """

    def __init__(self, env: "COLAVEnvironment", params: Optional[TrajectoryTrackingRewardParams] = None) -> None:
        super().__init__(env)

        # Find the NavigationPathObservation object
        if isinstance(env.observation_type, csgym_obs.DictObservation):
            for observation_type in env.observation_type.observation_types:
                if isinstance(observation_type, csgym_obs.NavigationPathObservation):
                    self.path_observation = observation_type
                    break
        elif isinstance(env.observation_type, csgym_obs.NavigationPathObservation):
            self.path_observation = env.observation_type

        self.params = params
        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        assert self.path_observation is not None, "NavigationPathObservation is not part of the observation!"

        # If no Dynamic COLAV rewarder, the tradeoff coefficient defaults to 1.0
        tradeoff_coefficient = 1.0
        # Find the DynamicColavRewarder object
        for rewarder in self.env.rewarder.rewarders:
            if isinstance(rewarder, DynamicColavRewarder):
                assert (
                    rewarder.last_reward_calculation_time == self.env.simulator.t
                ), "Dynamic rewarder needs to be called before path rewarder!"
                # Get tradeoff coefficient lambda from dynamic rewarder
                tradeoff_coefficient = rewarder.tradeoff_coefficient
                break

        # Unnormalize navigation observation
        navigation_observation = self.path_observation.unnormalize(state["NavigationPathObservation"])

        # Unpack observation
        cross_track_error = navigation_observation[3]
        course_angle_error = navigation_observation[4]
        speed_error = navigation_observation[6]

        # Calculate reward terms
        course_angle_term = self.params.gamma_chi * np.cos(course_angle_error) + self.params.gamma_r
        cte_term = np.exp(-self.params.gamma_e * abs(cross_track_error)) + self.params.gamma_r
        path_reward = course_angle_term * cte_term - self.params.gamma_r**2
        speed_reward = -self.params.gamma_u * (speed_error**2)

        self.last_reward = tradeoff_coefficient * (path_reward + speed_reward)
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_trajectory_tracking": self.last_reward}


class StaticColavRewarder(IReward):
    """Static obstacle COLAV rewarder as implemented in Meyer et.al.(2020)
    NOTE: The LidarLikeObservation is required as part of the environment observation"""

    def __init__(self, env: "COLAVEnvironment", params: Optional[StaticColavRewardParams] = None) -> None:
        super().__init__(env)
        # Extract the LidarLikeObservation object
        if isinstance(env.observation_type, csgym_obs.DictObservation):
            for observation_type in env.observation_type.observation_types:
                if isinstance(observation_type, csgym_obs.LidarLikeObservation):
                    self.sensor_suite = observation_type
                    self.sensor_angles = deepcopy(observation_type._sensor_angles)
                    break
        elif isinstance(env.observation_type, csgym_obs.LidarLikeObservation):
            self.sensor_suite = env.observation_type
            self.sensor_angles = deepcopy(env.observation_type._sensor_angles)

        self.params = params

        # Functions for reward calculation
        self.closeness_penalty = lambda x: self.params.alpha_x * np.exp(-self.params.gamma_x * x)
        self.weighting = lambda theta: 1 / (1 + self.params.gamma_theta * abs(theta))

        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        assert self.sensor_suite is not None, "LidarLikeObservation is not part of the observation!"
        # Extract current measured distances and sensor angles from LidarLikeObservation object
        dist_measurements = deepcopy(self.sensor_suite.current_dist_measurements)

        num = 0
        den = 0

        for distance, angle in zip(dist_measurements, self.sensor_angles):
            num += self.weighting(angle) * self.closeness_penalty(distance)
            den += self.weighting(angle)

        self.last_reward = -num / den
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_static_colav": self.last_reward}


class DynamicColavRewarder(IReward):
    """
    Rewards the agent for avoiding dynamic obstacles based on Meyer et al. (2020).
    Partial COLREGs compliance is also implemented.
    NOTE: The raw penalty calculation is altered, and does not take into account
    negative TS velocities. The gamma_v parameters are consequently simplified.
    """

    def __init__(self, env: "COLAVEnvironment", params: Optional[DynamicColavRewardParams] = None) -> None:
        super().__init__(env)
        if isinstance(env.observation_type, csgym_obs.DictObservation):
            for observation_type in env.observation_type.observation_types:
                if isinstance(observation_type, csgym_obs.LidarLikeObservation):
                    self.sensor_suite = observation_type
                    self.sensor_angles = deepcopy(observation_type._sensor_angles)
                    break
        elif isinstance(env.observation_type, csgym_obs.LidarLikeObservation):
            self.sensor_suite = env.observation_type
            self.sensor_angles = deepcopy(env.observation_type._sensor_angles)

        self.params = params
        self.tradeoff_coefficient = 1.0
        self.last_reward_calculation_time = self.env.simulator.t

        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        assert self.sensor_suite is not None, "LidarLikeObservation is not part of the observation!"
        # Extract current measured distances and velocities from LidarLikeObservation object
        dist_measurements = deepcopy(self.sensor_suite.current_dist_measurements)
        obstacle_velocities = deepcopy(self.sensor_suite.current_obstacle_velocities)

        self.last_reward_calculation_time = self.env.simulator.t
        num = 0
        den = 0
        lambdas = []

        for x, v, sensor_angle in zip(dist_measurements, obstacle_velocities, self.sensor_angles):
            v_y = v[1]
            if v_y != 0.0:
                zeta_v, zeta_x = self._determine_scaling(theta=sensor_angle)
                weight = 1 / (1 + np.exp(self.params.gamma_theta * abs(sensor_angle)))

                raw_penalty = self.params.alpha_x * np.exp(zeta_v * np.max((0, v_y)) - zeta_x * x)

                lambda_i = self._determine_tradeoff_coefficient(v_y=v_y, x_i=x)
                lambdas.append(lambda_i)

                num += weight * (1 - lambda_i) * raw_penalty
                den += weight

        if den == 0.0:
            # No dynamic obstacles detected
            self.tradeoff_coefficient = 1.0
            self.last_reward = 0.0
        else:
            self.tradeoff_coefficient = np.min(lambdas)
            self.last_reward = -num / den

        return self.last_reward

    def _determine_scaling(self, theta: float) -> tuple[float, float]:
        """Determines the velocity and distance scaling parameters for the
        reward calculation, based on the sensor angle theta.
        Args:
            theta (float): angle of the obstacle relative to the ownship in radians,
                           where 0 is in front of the ownship.

        Returns:
            Tuple(float, float): Velocity scaling parameter zeta_v and distance
            scaling parameter zeta_x
        """
        if theta >= 0.0 and theta < np.deg2rad(112.5):
            return self.params.gamma_v_stb, self.params.gamma_x_stb
        elif theta < 0.0 and theta > -np.deg2rad(112.5):
            return self.params.gamma_v_port, self.params.gamma_x_port
        else:
            return self.params.gamma_v_stern, self.params.gamma_x_stern

    def _determine_tradeoff_coefficient(self, v_y: float, x_i: float) -> float:
        """Calculates the tradeoff coefficient lambda based on the
        distance and velocity of the TS.

        Args:
            v_y (float): The body-relative velocity of the TS
            x_i (float): The distance to the TS

        Returns:
            float: The tradeoff coefficient lambda
        """
        if v_y >= 0.0:
            # Choose alpha_lambda+ and gamma_lambda+
            alpha_lambda = self.params.alpha_lambda[1]
            gamma_lambda = self.params.gamma_lambda[1]
        else:
            alpha_lambda = self.params.alpha_lambda[0]
            gamma_lambda = self.params.gamma_lambda[0]

        den = 1 + np.exp(-gamma_lambda * x_i + alpha_lambda)

        return 1 / den

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_dynamic_colav": self.last_reward}


class AutopilotReferenceRewarder(IReward):
    """Rewards the agent negatively for rapid changes in the output signals"""

    def __init__(self, env: "COLAVEnvironment", params: Optional[AutopilotReferenceRewardParams] = None) -> None:
        super().__init__(env)

        self.course_coefficient = params.course_coefficient
        self.speed_coefficient = params.speed_coefficient
        self.h = env.simulator.dt

        self.course_range = env.action_type.course_range

        self.prev_course_delta = 0.0
        self.prev_speed_delta = 0.0

        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        assert action is not None, "Action was not defined!"

        # Unpack the action
        speed_delta = action[1]
        course_delta = action[0]

        # Approximating the derivatives
        speed_delta_dot = (speed_delta - self.prev_speed_delta) / self.h
        course_delta_dot = (course_delta - self.prev_course_delta) / self.h

        # Calculate the rewards
        speed_dot_reward = -self.speed_coefficient * (speed_delta_dot**2)
        course_dot_reward = -self.course_coefficient * (course_delta_dot**2)

        self.prev_speed_delta = speed_delta
        self.prev_course_delta = course_delta

        self.last_reward = speed_dot_reward + course_dot_reward
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_autopilot_reference": self.last_reward}


class EvaluatorRewarder(IReward):
    """Reward the agent based on the vessel's recieved scores and penalites after evaluation of its trajectory by the Evaluation tool."""

    def __init__(self, env: "COLAVEnvironment", params: Optional[EvaluatorRewardParams] = None) -> None:
        """Initializes the reward function.

        Args:
            env (COLAVEnvironment): The environment
            params (EvaluatorRewardParams): The reward parameters.
        """
        super().__init__(env)
        self.last_reward = 0.0
        self.params = params if params else EvaluatorRewardParams()
        self.evaluator = Evaluator()
        self.simulator = env.simulator
        self.ship_info = {}
        self.verbose: bool = True
        self.save_results: bool = False
        self.save_params: bool = False
        self.result_weights = {
            "S_13": 1.0,
            "S_14": 1.0,
            "S_15": 1.0,
            "S_16": 1.0,
            "S_17": 1.0,
            #"S_17_stage_1": 1.0,
            #"S_17_stage_2": 1.0,
            #"S_17_stage_3": 1.0,
            "S_safety": 1.0,
            "S_safety_theta": 1.0,
            "S_safety_r": 1.0, #1.0,
            "P_13_ahead": 1.0,
            "P_14_sts": 1.0,
            "P_14_nsb": 1.0,
            "P_15_ahead": 1.0,
            "P_delay": 1.0,
            "P_pt": 1.0,
            #"P_delta_chi_stage_1": 1.0,
            #"P_delta_chi_stage_2": 1.0,
            #"P_16_na_delta_chi": 1.0,
            "P_16_na_man": 1.0
        }
        
    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:     
        goal_reached = self.env.simulator.determine_ship_goal_reached(ship_idx=0)
        truncated = kwargs.get("truncated", False)

        if goal_reached or truncated: 
            sim_data = pd.DataFrame(self.env.sim_data)
            
            self.ship_info = {}
            for i, ship_obj in enumerate(self.simulator.ship_list):
                self.ship_info[f"Ship{i}"] = ship_obj.get_ship_info()

            vessel_data = mhm.convert_simulation_data_to_vessel_data(sim_data, self.ship_info, self.env.scenario_config.utm_zone)

            self.evaluator.set_vessel_data(vessel_data)
            results = self.evaluator.evaluate()

            result_dict = mhm_eval.get_evaluator_scores_and_penalties(results, vessel_data[0], vessel_data)
            avg_result = self._get_avg_result(result_dict)

            # Mapping average evaluator result from (0, 1) to (-1, 1) and scaling it to get a reward
            Y0, V0, K = self._exp_fit(0, self.params.zero_thres, 1)
            evaluator_reward = (Y0 - V0/K*(1 - np.exp(-K*avg_result))) * self.params.r_max
            self.last_reward = evaluator_reward

            if self.verbose:
                situation = result_dict["situation"]
                print(f"\nSituation(s): {situation}")
                print(f"\tAverage result: {avg_result}")
                print(f"\tEvaluator reward: {self.last_reward}\n")
                self.evaluator.print_vessel_scores(vessel_id=0, save_results=False)

            if self.save_results:
                print("Saving results...")
                file_prefix = self.simulator.sconfig.filename.split(".")[0]
                self.evaluator.print_vessel_scores(vessel_id=0, file_prefix=file_prefix, save_results=True)

            if self.save_params:
                print("Saving parameters...")

                if self.simulator.sconfig.filename is not None:
                    file_prefix = self.simulator.sconfig.filename.split(".")[0]
                elif self.simulator.sconfig.name is not None:
                    file_prefix = self.simulator.sconfig.name.split(".")[0]
                else:
                    file_prefix = "default"

                for file_name in os.listdir('.'):
                    if file_name == "sbmpc_param_log.csv":
                        new_name = f"{file_prefix}_{file_name}"
                        os.rename(file_name, new_name)
                        break

        else:
            self.last_reward = 0.0
        
        return self.last_reward
    
    def _get_avg_result(self, result_dict: dict) -> float:
        total_weighted_sum = 0
        total_metrics_count = 0

        for category in ["scores", "penalties"]:
            for metric, obst in result_dict[category].items():
                metric_weight = self.result_weights.get(metric, 1.0)
                n_obst = len(obst)
                
                if category == "scores":
                    weighted_sum = sum(value * metric_weight for value in obst.values())
                else:
                    weighted_sum = sum((1 - value) * metric_weight for value in obst.values())
                
                total_weighted_sum += weighted_sum / n_obst
                total_metrics_count += 1

        avg_result = total_weighted_sum / total_metrics_count
        return avg_result
    
    def _exp_fit(self, x_min, x_zero, x_max) -> tuple:
        assert x_min < x_zero < x_max, "Input values are invalid"
        x = (x_min, x_zero, x_max)
        y = (-1, 0, 1)

        def equations(vars):
            Y0, V0, K = vars
            eq1 = Y0 - (V0 / K) * (1 - np.exp(-K * x[0])) - y[0]
            eq2 = Y0 - (V0 / K) * (1 - np.exp(-K * x[1])) - y[1]
            eq3 = Y0 - (V0 / K) * (1 - np.exp(-K * x[2])) - y[2]
            return [eq1, eq2, eq3]
        
        # Initial guess for Y0, V0, and K
        initial_guess = [1, 1, 1]
        
        # Solve the system of equations
        Y0, V0, K = fsolve(equations, initial_guess)
        
        return Y0, V0, K 

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_evaluator": self.last_reward}


class TrajectoryTrackingIndRewarder(IReward):
    """From https://github.com/NTNU-Autoship-Internal/rlmpc/blob/main/rlmpc/rewards.py"""

    def __init__(self, env: "COLAVEnvironment", config: TrajectoryTrackingIndRewarderParams) -> None:
        super().__init__(env)
        self.last_reward = 0.0
        self._config = config
        self._last_course_error = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self._last_course_error = 0.0
        goal_reached = self.env.simulator.determine_ship_goal_reached(ship_idx=0)
        if goal_reached:
            self.last_reward = 0.0 #self._config.rho_goal
            print(f"[{self.env.env_id.upper()}] Goal reached! Rewarding +{self.last_reward}.")
            return self.last_reward

        d2goal = np.linalg.norm(self.env.ownship.state[:2] - self.env.ownship.waypoints[:, -1])
        ownship_state = self.env.ownship.state
        do_list, _ = self.env.ownship.get_do_track_information()

        d2dos = np.array([(0, 1e12)])
        if len(do_list) > 0:
            d2dos = mhm.compute_distances_to_dynamic_obstacles(ownship_state, do_list)
        no_dos_in_the_way = d2dos[0][1] > 100.0
        truncated = kwargs.get("truncated", False)
        if truncated and not goal_reached and no_dos_in_the_way:
            self.last_reward = 0.0 #-0.002 * self._config.rho_goal * d2goal
            print(f"[{self.env.env_id.upper()}] Goal not reached! Rewarding {self.last_reward}.")
            return self.last_reward

        unnormalized_obs = self.env.observation_type.unnormalize(state)
        path_obs = unnormalized_obs["PathRelativeNavigationObservation"]
        huber_loss_d2path = sbmpc_hf.huber_loss(path_obs[0] ** 2, 1.0)
        huber_loss_d2goal = sbmpc_hf.huber_loss(path_obs[1] ** 2, 1.0)
        unwrapped_course_error = mf.unwrap_angle(self._last_course_error, path_obs[2])
        self._last_course_error = path_obs[2]

        rho_inverse_d2path = 0.0
        if huber_loss_d2path > 0.05:
            rho_inverse_d2path = self._config.rho_inverse_d2path

        tt_cost = (
            self._config.rho_d2path * huber_loss_d2path
            #rho_inverse_d2path * (1.0 / (huber_loss_d2path + 1e-6))
            + self._config.rho_d2goal * huber_loss_d2goal
            + self._config.rho_course_dev * unwrapped_course_error**2
            + self._config.rho_speed_dev * path_obs[3] ** 2
            + self._config.rho_turn_rate * path_obs[4] ** 2
        )
        self.last_reward = -tt_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_trajectory_tracking": self.last_reward}
    

class ReadilyApparentManeuveringRewarder(IReward):
    """From https://github.com/NTNU-Autoship-Internal/rlmpc/blob/main/rlmpc/rewards.py"""

    def __init__(self, env: "COLAVEnvironment", params: Optional[ReadilyApparentManeuveringRewarderParams] = None) -> None:
        super().__init__(env)
        self._config = params
        self.last_reward = 0.0
        self._config.r_max = self.env.ownship.max_turn_rate
        self.K_app = np.array([self._config.K_app_course, self._config.K_app_speed])
        self.alpha_app = np.concatenate([self._config.alpha_app_course, self._config.alpha_app_speed])
        self._prev_speed = self.env.ownship.speed
        self._distance_threshold = 500.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        if self.env.time < 0.0001:
            self._prev_speed = self.env.ownship.speed
        turn_rate = self.env.ownship.state[5]
        speed = self.env.ownship.speed

        true_ship_states = mhm.extract_do_states_from_ship_list(self.env.time, self.env.ship_list)
        do_list = mhm.get_relevant_do_states(true_ship_states, idx=0, add_empty_cov=True)
        distances_to_obstacles = mhm.compute_distances_to_dynamic_obstacles(
            ownship_state=self.env.ownship.state, do_list=do_list
        )
        if distances_to_obstacles[0][1] > self._distance_threshold:
            self.last_reward = 0.0
            return self.last_reward

        acceleration = (speed - self._prev_speed) / self.env.dt_action
        acceleration = np.clip(acceleration, -self._config.a_max, self._config.a_max)
        rate_cost, _, _ = sbmpc_hf.rate_cost(
            r=turn_rate,
            a=acceleration,
            K_app=self.K_app,
            alpha_app=self.alpha_app,
            r_max=self._config.r_max,
            a_max=self._config.a_max,
        )
        self._prev_speed = speed
        self.last_reward = -rate_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_ra_maneuvering": self.last_reward}
    
class ActionChatterRewarder(IReward):
    """
    From https://github.com/NTNU-Autoship-Internal/rlmpc/blob/main/rlmpc/rewards.py
    Used to penalize chattering actions for a relative course+speed action.
    """

    def __init__(
        self, env: "COLAVEnvironment", config: ActionChatterRewarderParams
    ) -> None:
        super().__init__(env)
        self._config = config
        self.last_reward = 0.0
        self.prev_action = np.zeros(self.env.action_space.shape[0])

    def __call__(
        self,
        state: csgym_obs.Observation,
        action: Optional[csgym_action.Action] = None,
        **kwargs,
    ) -> float:
        unnorm_action = self.env.action_type.unnormalize(action)
        if self.env.time < 0.0001:
            self.prev_action = unnorm_action
            return 0.0

        action_diff = unnorm_action - self.prev_action
        chatter_cost = self._config.rho_chatter * action_diff.T @ action_diff

        self.prev_action = unnorm_action
        self.last_reward = -chatter_cost
        return float(self.last_reward)

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_action_chatter": self.last_reward}

import math 
class GroundingAvoidanceRewarder(IReward):
    def __init__(self, env: "COLAVEnvironment", config: Optional[GroundingAvoidanceRewarderParams] = None):
        super().__init__(env)
        self.params = config if config else GroundingAvoidanceRewarderParams()
        self.last_reward = 0.0

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        unnormalized_obs = self.env.observation_type.unnormalize(state)
        hazard_obs = unnormalized_obs.get("DistanceBearingToHazardObservation")

        if hazard_obs is None or len(hazard_obs) == 0:
            self.last_reward = 0.0
            return self.last_reward
        
        ga_cost = 0.0
        for pt in range(len(hazard_obs)):
            dist    = hazard_obs[pt][0]
            bearing = hazard_obs[pt][1]
            weight  = max(0, np.cos(bearing))

            #print(f"dist: {dist}\n"
            #      f"bearing: {math.degrees(bearing)}\n"
            #      f"weight: {weight}\n"
            #      f"weighted_dist; {(self.params.safe_distance / dist)**2 * weight}\n"
            #      )
            
            if dist < self.params.safe_distance:
                weighted_dist = (self.params.rho_ga / dist)**2 * weight
                ga_cost += min(weighted_dist, self.params.r_max)

        self.last_reward = -ga_cost
        return self.last_reward

    def get_last_rewards_as_dict(self) -> dict:
        return {"r_grounding_avoidance": self.last_reward}

class SBMPCRewarder(IReward):
    """The SBMPC rewarder class. The sub-reward classes compute the RL stage cost, but
    return the negative of the cost to be consistent with the RL literature on reward maximization.
    """

    def __init__(self, env: "COLAVEnvironment", config: Config = Config()) -> None:
        super().__init__(env)
        self.reward_scale: float = 0.01
        self.last_reward: float = 0.0
        self._config = config
        
        self.evaluator_rewarder = EvaluatorRewarder(env, EvaluatorRewardParams())
        self.collision_rewarder = CollisionRewarder(env, CollisionRewardParams())
        self.trajectory_rewarder = TrajectoryTrackingIndRewarder(env, TrajectoryTrackingIndRewarderParams())
        self.readily_apparent_maneuvering = ReadilyApparentManeuveringRewarder(env, ReadilyApparentManeuveringRewarderParams())
        self.action_chatter_rewarder = ActionChatterRewarder(env, ActionChatterRewarderParams())
        self.grounding_avoidance_rewarder = GroundingAvoidanceRewarder(env, GroundingAvoidanceRewarderParams())
        self.grounding_rewarder = GroundingRewarder(env, GroundingRewardParams())

        self.r_evaluator: float = 0.0
        self.r_collision: float = 0.0
        self.r_trajectory: float = 0.0
        self.r_readily_apparent_maneuvering: float = 0.0
        self.r_action_chatter: float = 0.0
        self.r_grounding_avoidance: float = 0.0
        self.r_grounding: float = 0.0
        
        self.verbose: bool = False

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        self.r_evaluator = self.evaluator_rewarder(state, action, **kwargs)
        self.r_collision = self.collision_rewarder(state, action, **kwargs)
        self.r_trajectory = self.trajectory_rewarder(state, action, **kwargs)
        self.r_readily_apparent_maneuvering = self.readily_apparent_maneuvering(state, action, **kwargs)
        self.r_action_chatter = self.action_chatter_rewarder(state, action, **kwargs)
        self.r_grounding_avoidance = self.grounding_avoidance_rewarder(state, action, **kwargs)
        self.r_grounding = self.grounding_rewarder(state, action, **kwargs)

        reward = (
            self.r_evaluator
            + self.r_collision
            + self.r_trajectory
            + self.r_readily_apparent_maneuvering
            + self.r_action_chatter
            #+ self.r_grounding_avoidance
            + self.r_grounding
        )
        reward = reward * self.reward_scale
        #print(self.r_grounding_avoidance)
        #print("r_trajectory:", self.r_trajectory)
        #print("r_readily_apparent_maneuvering:", self.r_readily_apparent_maneuvering)
        #print("r_action_chatter:", self.r_action_chatter)
        #if self.r_grounding != 0.0:
        #    print(self.r_grounding)
        #if self.r_evaluator != 0.0:
        #    print(self.r_evaluator)

        """
        if self.r_evaluator != 0:
            print("\n--------------------------")
            print("Evaluator reward: ", self.r_evaluator)
            print("Collision reward: ", self.r_collision)
            print("Trajectory reward: ", self.r_trajectory)
            print("Readily apparent maneuvering reward: ", self.r_readily_apparent_maneuvering)
            print("Action chatter reward: ", self.r_action_chatter)
            print("--------------------------\n")
        """
        
        if self.verbose:
            print(
                f"[SBMPC-REWARDER | {self.env.env_id.upper()}]:\n\t- r_scaled: {reward:.4f} \n\t *-----------------* \n\t- r_evaluator: {self.r_evaluator:.4f} \n\t- r_collision: {self.r_collision:.4f} \n\t- r_trajectory: {self.r_trajectory:.4f} \n\t- r_readily_app_man: {self.r_readily_apparent_maneuvering:.4f} \n\t- r_action_chatter: {self.r_action_chatter:.4f}"
            )

        self.last_reward = reward
        return reward

    def get_last_rewards_as_dict(self) -> dict:
        return {
            "r_evaluator": self.r_evaluator * self.reward_scale,
            "r_collision": self.r_collision * self.reward_scale,
            "r_trajectory": self.r_trajectory * self.reward_scale,
            "r_readily_app_man": self.r_readily_apparent_maneuvering * self.reward_scale,
            "r_action_chatter": self.r_action_chatter * self.reward_scale
        }


class Rewarder(IReward):
    """The rewarder class."""

    def __init__(self, env: "COLAVEnvironment", config: Optional[Config] = None) -> None:
        """Initializes the rewarder.

        Args:
            env (COLAVEnvironment): The environment
            config (Config): The rewarder configuration
        """
        self.config = config if config else Config()
        self._last_reward = 0.0
        self.rewarders = []
        for rewarder in self.config.rewarders:
            if isinstance(rewarder, ExistenceRewardParams):
                self.rewarders.append(ExistenceRewarder(rewarder))
            elif isinstance(rewarder, DistanceToGoalRewardParams):
                self.rewarders.append(DistanceToGoalRewarder(env, rewarder))
            elif isinstance(rewarder, CollisionRewardParams):
                self.rewarders.append(CollisionRewarder(env, rewarder))
            elif isinstance(rewarder, GroundingRewardParams):
                self.rewarders.append(GroundingRewarder(env, rewarder))
            elif isinstance(rewarder, TrajectoryTrackingRewardParams):
                self.rewarders.append(TrajectoryTrackingRewarder(env, rewarder))
            elif isinstance(rewarder, StaticColavRewardParams):
                self.rewarders.append(StaticColavRewarder(env, rewarder))
            elif isinstance(rewarder, DynamicColavRewardParams):
                self.rewarders.append(DynamicColavRewarder(env, rewarder))
            elif isinstance(rewarder, AutopilotReferenceRewardParams):
                self.rewarders.append(AutopilotReferenceRewarder(env, rewarder))

    def __call__(self, state: csgym_obs.Observation, action: Optional[csgym_action.Action] = None, **kwargs) -> float:
        reward = 0.0
        for rewarder in self.rewarders:
            reward += rewarder(state, action, **kwargs)
        self._last_reward = reward
        return reward

    def get_last_rewards_as_dict(self) -> dict:
        rewards = {}
        for rewarder in self.rewarders:
            rewards.update(rewarder.get_last_rewards_as_dict())
        return rewards

    @property
    def last_reward(self):
        return self._last_reward
