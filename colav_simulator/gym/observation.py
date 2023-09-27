"""
    observation.py

    Summary:
        This file contains various observation type/space definitions for a ship agent in the colav-simulator.

        To add an observation type:
        1: Create a new class inheriting from ObservationType and implement the abstract methods.
        2: Add the new class to the `observation_factory` method. Make sure lower case (snake case) string names are used for specifying the action type.
        3: Add the new observation type to the `rl_observation_type` entry in the scenario schema file (read the Cerberus docs to learn config validation),
           such that it can be used and validated against in a scenario.
        4: Add the new observation type to your scenario config file.

    Author: Trym Tengesdal
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.core.sensing as sensing
import colav_simulator.core.tracking.trackers as trackers
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
import shapely.geometry as sgeo

Observation = Union[tuple, list, np.ndarray]

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


class ObservationType(ABC):
    """Interface class for observation types for the COLAVEnvironment gym."""

    name: str = "AbstractObservation"

    def __init__(self, env: "COLAVEnvironment") -> None:
        self.env = env
        self._ownship = self.env.ownship

    @abstractmethod
    def space(self) -> gym.spaces.Space:
        """Get the observation space."""

    @abstractmethod
    def observe(self) -> Observation:
        """Get an observation of the environment state (normalized)."""


class LidarLikeObservation(ObservationType):
    """A lidar-like observation space for the own-ship, i.e. a 360 degree scan of the environment as is used in Meyer et. al. 2020."""

    def __init__(
        self,
        env: "COLAVEnvironment",
    ) -> None:
        super().__init__(env)
        self.n_do = len(self.env.dynamic_obstacles)
        self.enc = self.env.enc
        self.define_observation_ranges()
        self.name = "LidarLikeObservation"

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_do,))

    def define_observation_ranges(self) -> None:
        """Define the ranges for the observation space."""
        max_position = np.array([self.enc.origin[0] + self.enc.size[0], self.enc.origin[1] + self.enc.size[1]])
        default_do_speed_range = np.array([0.0, 20.0])
        self.observation_range = {
            "position": (max_position[0], max_position[1]),
            "angles": (-np.pi, np.pi),
            "do_speed": default_do_speed_range,
        }

    def normalize(self, obs: Observation) -> Observation:
        """Normalize the input observation entries to be within the range [-1, 1], based on the
        ranges for each observation dimension.

        Args:
            obs (Observation): The observation to normalize.

        Returns:
            Observation: Normalized observation.
        """
        pass

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        dynamic_obstacles = self.env.dynamic_obstacles
        true_do_states = [(idx, mhm.convert_state_to_vxvy_state(do.csog_state)) for idx, do in enumerate(dynamic_obstacles)]
        do_tracks = self._ownship.track_obstacles(self.env.time, self.env.time_step, true_do_states)
        obs = np.zeros(1, dtype=np.float32)

        # TODO: Implement observation

        # sector_closenesses, sector_velocities = self.__ownship.get_track_information(self.obstacles)

        # obs = np.concatenate([reward_insight, navigation_states, sector_closenesses, sector_velocities])

        # Initializing variables
        # sensor_range = self.config["sensor_range"]

        # p0_point = sgeo.Point(*self._ownship.csog_state[:2])

        # # Loading nearby obstacles, i.e. obstacles within the vessel's detection range
        # if self._step_counter % self.config["sensor_interval_load_obstacles"] == 0:
        #     self._nearby_obstacles = list(filter(lambda obst: float(p0_point.distance(obst.boundary)) - self._width < sensor_range, obstacles))

        # if not self._nearby_obstacles:
        #     self._last_sensor_dist_measurements = np.ones((self._n_sensors,)) * sensor_range
        #     sector_feasible_distances = np.ones((self._n_sectors,)) * sensor_range
        #     sector_closenesses = np.zeros((self._n_sectors,))
        #     sector_velocities = np.zeros((2 * self._n_sectors,))
        #     collision = False

        # else:
        #     should_observe = (self._perceive_counter % self._observe_interval == 0) or self._virtual_environment is None
        #     if should_observe:
        #         geom_targets = self._nearby_obstacles
        #     else:
        #         geom_targets = self._virtual_environment

        #     # Simulating all sensors using _simulate_sensor subroutine
        #     sensor_angles_ned = self._sensor_angles + self.heading
        #     activate_sensor = lambda i: (i % self._sensor_interval) == (self._perceive_counter % self._sensor_interval)
        #     sensor_sim_args = (p0_point, sensor_range, geom_targets)
        #     sensor_output_arrs = list(
        #         map(
        #             lambda i: _simulate_sensor(sensor_angles_ned[i], *sensor_sim_args)
        #             if activate_sensor(i)
        #             else (self._last_sensor_dist_measurements[i], self._last_sensor_speed_measurements[i], True),
        #             range(self._n_sensors),
        #         )
        #     )
        #     sensor_dist_measurements, sensor_speed_measurements, sensor_blocked_arr = zip(*sensor_output_arrs)
        #     sensor_dist_measurements = np.array(sensor_dist_measurements)
        #     sensor_speed_measurements = np.array(sensor_speed_measurements)
        #     self._last_sensor_dist_measurements = sensor_dist_measurements
        #     self._last_sensor_speed_measurements = sensor_speed_measurements

        #     # Setting virtual obstacle
        #     if should_observe:
        #         line_segments = []
        #         tmp = []
        #         for i in range(self.n_sensors):
        #             if sensor_blocked_arr[i]:
        #                 point = (
        #                     self.position[0] + np.cos(sensor_angles_ned[i]) * sensor_dist_measurements[i],
        #                     self.position[1] + np.sin(sensor_angles_ned[i]) * sensor_dist_measurements[i],
        #                 )
        #                 tmp.append(point)
        #             elif len(tmp) > 1:
        #                 line_segments.append(tuple(tmp))
        #                 tmp = []

        #         self._virtual_environment = list(map(LineObstacle, line_segments))

        #     # Partitioning sensor readings into sectors
        #     sector_dist_measurements = np.split(sensor_dist_measurements, self._sector_start_indeces[1:])
        #     sector_speed_measurements = np.split(sensor_speed_measurements, self._sector_start_indeces[1:], axis=0)

        #     # Performing feasibility pooling
        #     sector_feasible_distances = np.array(list(map(lambda x: _feasibility_pooling(x, self._feasibility_width, self._d_sensor_angle), sector_dist_measurements)))

        #     # Calculating feasible closeness
        #     sector_closenesses = self._get_closeness(sector_feasible_distances)

        #     # Retrieving obstacle speed for closest obstacle within each sector
        #     closest_obst_sensor_indeces = list(map(np.argmin, sector_dist_measurements))
        #     sector_velocities = np.concatenate([sector_speed_measurements[i][closest_obst_sensor_indeces[i]] for i in range(self._n_sectors)])

        #     # Testing if vessel has collided
        #     collision = np.any(sensor_dist_measurements < self.width)

        # self._last_sector_dist_measurements = sector_closenesses
        # self._last_sector_feasible_dists = sector_feasible_distances
        # self._collision = collision
        # self._perceive_counter += 1

        # return self._get_closeness(self._last_sensor_dist_measurements.reshape(1, self.n_sensors))
        return obs


class Navigation3DOFStateObservation(ObservationType):
    """Observes the current own-ship 3DOF state, i.e. [x, y, psi, u, v, r]."""

    def __init__(
        self,
        env: "COLAVEnvironment",
    ) -> None:
        super().__init__(env)
        assert self._ownship is not None, "Ownship is not defined"
        self.name = "Navigation3DOFStateObservation"
        self.size = len(self._ownship.state)
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,))

    def define_observation_ranges(self) -> None:
        """Define the ranges for the observation space."""
        assert self._ownship is not None, "Ownship is not defined"
        (x_min, y_min, x_max, y_max) = self.env.enc.bbox
        self.observation_range = {
            "north": (y_min, y_max),
            "east": (x_min, x_max),
            "angles": (-np.pi, np.pi),
            "speed": (self._ownship.min_speed, self._ownship.max_speed),
            "turn_rate": (-self._ownship.max_turn_rate, self._ownship.max_turn_rate),
        }

    def normalize(self, obs: Observation) -> Observation:
        """Normalize the input observation entries to be within the range [-1, 1], based on the ranges for each observation dimension.

        Args:
            obs (Observation): The observation to normalize.

        Returns:
            Observation: Normalized observation.
        """
        normalized_obs = np.array(
            [
                mf.linear_map(obs[0], self.observation_range["north"], (-1.0, 1.0)),
                mf.linear_map(obs[1], self.observation_range["east"], (-1.0, 1.0)),
                mf.linear_map(obs[2], self.observation_range["angles"], (-1.0, 1.0)),
                mf.linear_map(obs[3], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[4], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[5], self.observation_range["turn_rate"], (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return normalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        state = self._ownship.state
        extras = np.empty(0)
        obs = np.concatenate([state, extras])
        return self.normalize(obs)


class NavigationCSOGStateObservation(ObservationType):
    """Observes the current own-ship CSOG state, i.e. [x, y, SOG, COG]."""

    def __init__(
        self,
        env: "COLAVEnvironment",
    ) -> None:
        super().__init__(env)
        assert self._ownship is not None, "Ownship is not defined"
        self.name = "NavigationCSOGStateObservation"
        self.size = len(self._ownship.csog_state)
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,))

    def define_observation_ranges(self) -> None:
        """Define the ranges for the observation space."""
        assert self._ownship is not None, "Ownship is not defined"
        (x_min, y_min, x_max, y_max) = self.env.enc.bbox
        self.observation_range = {
            "north": (y_min, y_max),
            "east": (x_min, x_max),
            "speed": (self._ownship.min_speed, self._ownship.max_speed),
            "angles": (-np.pi, np.pi),
        }

    def normalize(self, obs: Observation) -> Observation:
        """Normalize the input observation entries to be within the range [-1, 1], based on the ranges for each observation dimension.

        Args:
            obs (Observation): The observation to normalize.

        Returns:
            Observation: Normalized observation.
        """
        normalized_obs = np.array(
            [
                mf.linear_map(obs[0], self.observation_range["north"], (-1.0, 1.0)),
                mf.linear_map(obs[1], self.observation_range["east"], (-1.0, 1.0)),
                mf.linear_map(obs[2], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[3], self.observation_range["angles"], (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return normalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        state = self._ownship.csog_state
        extras = np.empty(0)
        obs = np.concatenate([state, extras])
        return self.normalize(obs)


class ImageObservation(ObservationType):
    """Observes a map image of the environment, possibly with multiple layers (static obstacles, dynamic obstacles, traffic separation schemes etc.)."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        super().__init__(env)

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        pass

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        pass


class TupleObservation(ObservationType):
    """Observation consisting of multiple observation types."""

    def __init__(self, env: "COLAVEnvironment", observation_configs: list, **kwargs) -> None:
        super().__init__(env)
        self.observation_types = [observation_factory(env, obs_config) for obs_config in observation_configs]

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> Observation:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


def observation_factory(env: "COLAVEnvironment", observation_type: str | dict = "lidar_like_observation", **kwargs) -> ObservationType:
    """Factory for creating observation spaces.

    Args:
        env: Used environment.
        observation_type (str): Observation type name.
        **kwargs: Additional arguments to pass to the observation type.

    Returns:
        ObservationType: Observation type to use
    """
    if observation_type == "lidar_like_observation":
        return LidarLikeObservation(env, **kwargs)
    elif observation_type == "navigation_3dof_state_observation":
        return Navigation3DOFStateObservation(env, **kwargs)
    elif observation_type == "navigation_csog_state_observation":
        return NavigationCSOGStateObservation(env, **kwargs)
    elif observation_type == "image_observation":
        return ImageObservation(env, **kwargs)
    elif "tuple_observation" in observation_type:
        return TupleObservation(env, observation_type["tuple_observation"], **kwargs)
    else:
        raise ValueError("Unknown observation type")
