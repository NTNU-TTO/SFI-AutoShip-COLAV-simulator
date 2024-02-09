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

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

import colav_simulator.common.image_helper_methods as imghf
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scimg
import shapely
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

    @abstractmethod
    def normalize(self, obs: Observation) -> Observation:
        """Normalize the observation entries to be within [-1, 1]."""

    @abstractmethod
    def unnormalize(self, obs: Observation) -> Observation:
        """Unnormalize the observation entries to be within the original range."""


class LidarLikeObservation(ObservationType):
    """A lidar-like observation space for the own-ship, i.e. a 360 degree scan of the environment as is used in Meyer et. al. 2020."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        super().__init__(env)
        self.n_do = len(self.env.dynamic_obstacles)
        self.enc = self.env.enc
        self.define_observation_ranges()
        self.name = "LidarLikeObservation"

        # Sensor parameters. TODO: Implement config to set/change these variables
        self.n_sensors = 180
        self.n_sectors = 9
        self.sensing_range = 1500

        self._partition_sensors()
        self._create_spatial_index()

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(3 * self.n_sectors,), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        """Define the ranges for the observation space."""
        closeness_range = np.array([0.0, 1.0])
        default_do_speed_range = np.array([-20.0, 20.0])
        self.observation_range = {
            "closeness": closeness_range,
            "do_speed": default_do_speed_range,
        }

    def unnormalize(self, obs: Observation) -> Observation:
        """Unnormalize the input normalized observation to be within the original range

        Args:
            obs (Observation): Normalized observation

        Returns:
            Observation: Unnormalized observation.
        """
        unnormalized_obs = np.array(
            np.concatenate(
                [
                    [
                        mf.linear_map(obs[i], (-1.0, 1.0), self.observation_range["closeness"])
                        for i in range(self.n_sectors)
                    ],
                    [
                        mf.linear_map(obs[j], (-1.0, 1.0), self.observation_range["do_speed"])
                        for j in range(self.n_sectors, self.n_sectors * 3)
                    ],
                ]
            ),
            dtype=np.float32,
        )
        return unnormalized_obs

    def normalize(self, obs: Observation) -> Observation:
        """Normalize the input observation entries to be within the range [-1, 1], based on the
        ranges for each observation dimension.

        Args:
            obs (Observation): The observation to normalize.

        Returns:
            Observation: Normalized observation.
        """
        normalized_obs = np.array(
            np.concatenate(
                [
                    [
                        mf.linear_map(obs[i], self.observation_range["closeness"], (-1.0, 1.0))
                        for i in range(self.n_sectors)
                    ],
                    [
                        mf.linear_map(obs[j], self.observation_range["do_speed"], (-1.0, 1.0))
                        for j in range(self.n_sectors, self.n_sectors * 3)
                    ],
                ]
            ),
            dtype=np.float32,
        )

        return normalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"

        ownship_pos = sgeo.Point(
            (self._ownship.state[1], self._ownship.state[0])
        )  # Ownship position as (easting, northing) coordinates

        sensor_range = self.sensing_range
        sensor_angles = self._sensor_angles + self._ownship.heading  # Rotation of sensor suite

        grounding_hazards = self.grounding_hazards
        dynamic_obstacles = self.env.dynamic_obstacles

        # Convert ship objects to polygons
        dynamic_obstacle_polygons = []
        for dynamic_obstacle in dynamic_obstacles:
            do_state = dynamic_obstacle.csog_state
            dynamic_obstacle_polygons.append(
                mapf.create_ship_polygon(
                    x=do_state[0],
                    y=do_state[1],
                    heading=dynamic_obstacle.heading,
                    length=dynamic_obstacle.length,
                    width=dynamic_obstacle.width,
                )
            )

        # Determine distance and velocities to obstacles in each sector
        obstacle_distances = list()
        obstacle_velocities = list()
        for isensor in range(self.n_sensors):
            closest_obstacle_dist, closest_obstacle_vel = self._cast_sensor_ray(
                ownship_pos=ownship_pos,
                sensor_angle=sensor_angles[isensor],
                sensor_range=sensor_range,
                grounding_hazards=grounding_hazards,
                dynamic_obstacles=dynamic_obstacle_polygons,
            )
            obstacle_distances.append(closest_obstacle_dist)
            obstacle_velocities.append(closest_obstacle_vel)

        obstacle_distances = np.array(obstacle_distances)
        obstacle_velocities = np.array(obstacle_velocities)

        # Split distance and velocity measurements into sectors
        obstacle_distances_by_sector = np.split(obstacle_distances, self._sector_start_indices[1:])
        obstacle_velocities_by_sector = np.split(obstacle_velocities, self._sector_start_indices[1:])

        # Feasibility pooling
        sector_closeness = []
        sector_velocities = []
        for i, _ in enumerate(obstacle_distances_by_sector):
            # Closeness of reachable distance
            sector_feasible_distance = self._feasibility_pooling(obstacle_distances_by_sector[i])
            sector_closeness.append(self._get_closeness(sector_feasible_distance))
            # Velocities of closest obstacle in sector
            closest_obstacle_index = np.argmin(obstacle_distances_by_sector[i])
            sector_velocities.append(obstacle_velocities_by_sector[i][closest_obstacle_index])

        obs = np.concatenate([np.array(sector_closeness), np.array(sector_velocities).flatten()])

        return self.normalize(obs)

    def _feasibility_pooling(self, x: list) -> float:
        """Implementation of algorithm 2 in Meyer et al (2020).
        Calculates the longest feasible distance within a given sensor sector based on
        the ownship's width

        Args:
            - x: Sensor measurements of current sector

        Returns:
            float: Longest feasible distance within the current sector

        """
        ship_width = self._ownship.get_ship_info()["width"]
        theta = self._delta_sensor_angle

        # Get sorted list of x with corresponding indices
        indices = np.argsort(x)
        for i in indices:
            arc_length = theta * x[i]
            opening_width = arc_length / 2
            opening_found = False
            for j, _ in enumerate(x):
                if x[j] > x[i]:
                    opening_width += arc_length
                    if opening_width > ship_width:
                        opening_found = True
                        continue
                else:
                    opening_width += arc_length / 2
                    if opening_width > ship_width:
                        opening_found = True
                        continue
                    opening_width = 0

            if not opening_found:
                # There does not exist a feasible path for the ship longer than x_i
                return max(0, x[i])

        return max(0, np.max(x))

    def _partition_sensors(self):
        """Partition the sensors into sectors based on the mapping used in Meyer et al(2020)
        A list of indices is assigned, where each index represent the first sensor of each sector.
        """
        self._delta_sensor_angle = 2 * np.pi / self.n_sensors  # Calculate angle between neighboring sensors
        self._sensor_angles = [-np.pi + i * self._delta_sensor_angle for i in range(self.n_sensors)]

        sector_start_indices = [0]
        current_sector = 0
        for isensor in range(self.n_sensors):
            sector = self._map_sensor_to_sector(isensor)
            if sector != current_sector:
                sector_start_indices.append(isensor)
                current_sector = sector

        self._sector_start_indices = sector_start_indices

    def _map_sensor_to_sector(self, isensor: int) -> int:
        """Maps a sensor index i in {1,...,N} to a sector index k in {1, ..., D}

        Args:
            - isensor (int): Index of current sensor

        Returns:
            int: Index of the corresponding sector
        """
        sigma = self._sigma(float(isensor))
        return int(np.floor(sigma(isensor) - sigma(0)))

    def _sigma(self, x: float) -> float:
        """Sigmoid function used for mapping sensor indices to sectors."""
        a = float(self.n_sensors)
        b = float(self.n_sectors)
        c = 0.1
        return b / (1.0 + np.exp((-x + a / 2.0) / (c * a)))

    def _get_closeness(self, distance: float):
        """Calculate closeness of obstacle as described in Meyer et al(2020).
        The function evaluates to 0 if obstacle is undetected, and 1 if the vessel has
        collided with the obstacle.

        Args:
            - distance(float): Distance from ownship to obstacle

        Returns:
            float: Closeness to obstacle ranging from 0 to 1

        """
        closeness = np.clip(a=(1 - np.log(distance + 1) / np.log(self.sensing_range + 1)), a_min=0, a_max=1)
        return closeness

    def _cast_sensor_ray(
        self,
        ownship_pos: sgeo.Point,
        sensor_angle: float,
        sensor_range: float,
        grounding_hazards: list,
        dynamic_obstacles: list,
    ):
        """Cast sensor ray and return coordinates of closest obstacle

        Args:
            - ownship_pos (Point): Ownship coordinates (east, north)
            - sensor_angle (float): Angle of sensor relative to ownships body frame [rad]
            - sensor_range (float): Range of sensor [m]
            - grounding_hazards (list): List of grounding hazards as a list of geometry objects
            - dynamic_obstacles (list): List of dynamic obstacles as polygons

        Returns:
            Tuple[float | float]: Distance and velocity of the closest detected obstacle as a tuple.
        """
        sensor_endpoint = (
            ownship_pos.x + np.sin(sensor_angle) * sensor_range,
            ownship_pos.y + np.cos(sensor_angle) * sensor_range,
        )

        sensor_ray = sgeo.LineString([ownship_pos, sensor_endpoint])
        closest_obstacle_dist = sensor_range
        closest_obstacle_vel = (0, 0)

        # Grounding hazards
        closest_hazard_idx = self.grounding_spatial_index.query_nearest(
            geometry=sensor_ray, max_distance=self.sensing_range, exclusive=True, all_matches=True
        )
        # Find closest obstacle among query results
        if np.any(closest_hazard_idx):
            for idx in closest_hazard_idx:
                if shapely.intersects(sensor_ray, grounding_hazards[idx]):
                    intersection = shapely.intersection(sensor_ray, grounding_hazards[idx])
                    intersection = mapf.standardize_polygon_intersections(
                        shapely.intersection(sensor_ray, grounding_hazards[idx])
                    )
                    if ownship_pos.distance(intersection) < closest_obstacle_dist:
                        closest_obstacle_dist = ownship_pos.distance(intersection)

        # Dynamic obstacles
        for idx, dynamic_obstacle in enumerate(dynamic_obstacles):
            if shapely.intersects(sensor_ray, dynamic_obstacle):
                intersection = sensor_ray.intersection(dynamic_obstacle)
                intersection = mapf.standardize_polygon_intersections(intersection)
                if ownship_pos.distance(intersection) < closest_obstacle_dist:
                    closest_obstacle_dist = ownship_pos.distance(intersection)

                    # Decompose velocity in sensor sector coordinates
                    csog_state = self.env.dynamic_obstacles[idx].csog_state
                    vxvy_state = mhm.convert_state_to_vxvy_state(csog_state)
                    closest_obstacle_vel = np.array([vxvy_state[2], vxvy_state[3]]).T
                    closest_obstacle_vel = mf.Rmtrx2D(-sensor_angle - np.pi / 2).dot(closest_obstacle_vel)

        return closest_obstacle_dist, closest_obstacle_vel

    def _create_spatial_index(self):
        """Creates a R-tree spatial index of the relevant grounding hazards."""
        grounding_hazards = np.array([])
        for poly in self.env.relevant_grounding_hazards:
            geoms = np.array([geom for geom in poly.geoms])
            np.concatenate([grounding_hazards, geoms])

        self.grounding_hazards = geoms
        self.grounding_spatial_index = shapely.STRtree(geoms)


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
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        """Define the ranges for the observation space."""
        assert self._ownship is not None, "Ownship is not defined"
        (x_min, y_min, x_max, y_max) = self.env.enc.bbox
        self.observation_range = {
            "north": (y_min, y_max),
            "east": (x_min, x_max),
            "angles": (-np.pi, np.pi),
            "surge": (-self._ownship.max_speed, self._ownship.max_speed),
            "sway": [-self._ownship.max_speed, self._ownship.max_speed],
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
                mf.linear_map(obs[3], self.observation_range["surge"], (-1.0, 1.0)),
                mf.linear_map(obs[4], self.observation_range["sway"], (-1.0, 1.0)),
                mf.linear_map(obs[5], self.observation_range["turn_rate"], (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return normalized_obs

    def unnormalize(self, obs: Observation) -> Observation:
        """Unnormalize the input normalized observation to be within the original range

        Args:
            obs (Observation): The observation to unnormalize.

        Returns:
            Observation: Unnormalized observation.
        """
        unnormalized_obs = np.array(
            [
                mf.linear_map(obs[0], (-1.0, 1.0), self.observation_range["north"]),
                mf.linear_map(obs[1], (-1.0, 1.0), self.observation_range["east"]),
                mf.linear_map(obs[2], (-1.0, 1.0), self.observation_range["angles"]),
                mf.linear_map(obs[3], (-1.0, 1.0), self.observation_range["surge"]),
                mf.linear_map(obs[4], (-1.0, 1.0), self.observation_range["sway"]),
                mf.linear_map(obs[5], (-1.0, 1.0), self.observation_range["turn_rate"]),
            ],
            dtype=np.float32,
        )
        return unnormalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        state = self._ownship.state
        obs = state
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
            "speed": (-self._ownship.max_speed, self._ownship.max_speed),
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

    def unnormalize(self, obs: Observation) -> Observation:
        """Unnormalize the input normalized observation to be within the original range

        Args:
            obs (Observation): The observation to unnormalize.

        Returns:
            Observation: Unnormalized observation.
        """
        unnormalized_obs = np.array(
            [
                mf.linear_map(obs[0], (-1.0, 1.0), self.observation_range["north"]),
                mf.linear_map(obs[1], (-1.0, 1.0), self.observation_range["east"]),
                mf.linear_map(obs[2], (-1.0, 1.0), self.observation_range["speed"]),
                mf.linear_map(obs[3], (-1.0, 1.0), self.observation_range["angles"]),
            ],
            dtype=np.float32,
        )
        return unnormalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        state = self._ownship.csog_state
        extras = np.empty(0)
        obs = np.concatenate([state, extras])
        return self.normalize(obs)


class TrackingObservation(ObservationType):
    """Observation containing a list of tracks/dynamic obstacles
    on the form (ID, state, cov, length, width), non-normalized.
    """

    def __init__(self, env: "COLAVEnvironment") -> None:
        super().__init__(env)
        self.n_do = len(self.env.dynamic_obstacles)
        self.name = "TrackingObservation"
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_do * 6,))

    def define_observation_ranges(self) -> None:
        """Define the ranges for the observation space."""
        assert self._ownship is not None, "Ownship is not defined"
        (x_min, y_min, x_max, y_max) = self.env.enc.bbox
        self.observation_range = {
            "north": (y_min, y_max),
            "east": (x_min, x_max),
            "speed": (-20.0, 20.0),
            "angles": (-np.pi, np.pi),
            "length": (0.0, 100.0),
            "width": (0.0, 100.0),
            "variance": (0.0, 500.0),
        }

    def normalize(self, obs: Observation) -> Observation:
        """Normalize the input observation entries to be within the range [-1, 1], based on the ranges for each observation dimension.

        Args:
            obs (Observation): The observation to normalize.

        Returns:
            Observation: Normalized observation.
        """
        # No normalization is provided for this observation type
        return obs

    def unnormalize(self, obs: Observation) -> Observation:
        """Unnormalize the input normalized observation to be within the original range

        Args:
            obs (Observation): The observation to unnormalize.

        Returns:
            Observation: Unnormalized observation.
        """
        # No unnormalization is provided for this observation type
        return obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        tracks, _ = self._ownship.get_do_track_information()
        obs = tracks
        return obs


class TimeObservation(ObservationType):
    """Observation containing the current time in the environment."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        super().__init__(env)
        self.name = "TimeObservation"
        self.t_end = self.env.time_truncated
        self.t_start = self.env.time

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def normalize(self, obs: Observation) -> Observation:
        """Normalize the input observation entries to be within the range [-1, 1], based on the ranges for each observation dimension.

        Args:
            obs (Observation): The observation to normalize.

        Returns:
            Observation: Normalized observation.
        """
        normalized_obs = np.array(
            [
                mf.linear_map(obs[0], (self.t_start, self.t_end), (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return normalized_obs

    def unnormalize(self, obs: Observation) -> Observation:
        """Unnormalize the input normalized observation to be within the original range

        Args:
            obs (Observation): The observation to unnormalize.

        Returns:
            Observation: Unnormalized observation.
        """
        unnormalized_obs = mf.linear_map(obs[0], (-1.0, 1.0), (self.t_start, self.t_end))
        return unnormalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        obs = np.array([self.env.time])
        return self.normalize(obs)


class PerceptionImageObservation(ObservationType):
    """Observation consisting of a perception image. INCOMPLETE"""

    def __init__(self, env: "COLAVEnvironment", image_dim: Tuple[int, int, int] = (400, 400, 3), **kwargs) -> None:
        super().__init__(env)
        self.name = "PerceptionImageObservation"
        self.image_dim = image_dim
        self.n_images = image_dim[2]  # Number of images (grayscale) to store in the observation
        self.observation_counter = 0
        self.previous_image_stack = np.zeros(image_dim, dtype=np.uint8)  # All black

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=0, high=255, shape=self.image_dim, dtype=np.uint8)

    def normalize(self, obs: Observation) -> Observation:
        """Normalize the input observation entries to be within the range [-1, 1], based on the ranges for each observation dimension.

        Args:
            obs (Observation): The observation to normalize.

        Returns:
            Observation: Normalized observation.
        """
        # No normalization is provided for this observation type
        return obs

    def unnormalize(self, obs: Observation) -> Observation:
        """Unnormalize the input normalized observation to be within the original range

        Args:
            obs (Observation): The observation to unnormalize.

        Returns:
            Observation: Unnormalized observation.
        """
        # No unnormalization is provided for this observation type
        return obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        t_now = time.time()
        img = self.env.liveplot_image
        pruned_img = imghf.remove_whitespace(img)
        # os_liveplot_zoom_width = self.env.liveplot_zoom_width
        os_heading = self._ownship.heading
        #
        # Rotate the image to align with the ownship heading
        rotated_img = scimg.rotate(pruned_img, os_heading * 180 / np.pi, reshape=False)
        npx, npy = rotated_img.shape[:2]

        # Crop the image to the vessel
        # image width and height corresponds to os_liveplot_zoom_width x os_liveplot_zoom_width
        # Image coordinate system is (0,0) in the upper left corner, height is the first index, width is the second index
        center_pixel_x = int(rotated_img.shape[0] // 2)
        center_pixel_y = int(rotated_img.shape[1] // 2)
        cutoff_index_below_vessel = int(0.1 * npx)  # corresponds to 200 m for a 2000 m zoom width
        cutoff_index_above_vessel = int(0.4 * npx)  # corresponds to 800 m for a 2000 m zoom width
        cutoff_width = int(0.25 * npy)  # corresponds to 500 m for a 2000 m zoom width

        cropped_img = rotated_img[
            center_pixel_x - cutoff_index_above_vessel : center_pixel_x + cutoff_index_below_vessel,
            center_pixel_y - cutoff_width : center_pixel_y + cutoff_width,
        ]

        # Resize image to a multiple of the image_dim
        diff_multiple = int(np.ceil(cropped_img.shape[0] / self.image_dim[0])), int(
            np.ceil(cropped_img.shape[1] / self.image_dim[1])
        )
        if self.image_dim[0] == self.image_dim[1]:
            img_resize_x = diff_multiple[0] * self.image_dim[0]
            img_resize_y = img_resize_x
        else:
            img_resize_x = diff_multiple[0] * self.image_dim[0]
            img_resize_y = diff_multiple[1] * self.image_dim[1]
        resized_img = cv2.resize(cropped_img, (img_resize_y, img_resize_x), interpolation=cv2.INTER_AREA)

        # downsample the image to configured image shape
        downsampled_img = cv2.resize(cropped_img, (self.image_dim[0], self.image_dim[1]), interpolation=cv2.INTER_AREA)
        grayscale_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2GRAY)

        if False:
            fig = plt.figure()
            axes = fig.subplot_mosaic(
                [
                    ["original", "rotated"],
                    ["cropped", "resized"],
                    ["downsampled", "grayscale"],
                ]
            )
            axes["original"].imshow(pruned_img, aspect="equal")
            axes["original"].axes.get_xaxis().set_visible(False)
            axes["original"].axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            axes["rotated"].imshow(rotated_img, aspect="equal")
            axes["rotated"].axes.get_xaxis().set_visible(False)
            axes["rotated"].axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            axes["cropped"].imshow(cropped_img, aspect="equal")
            axes["cropped"].axes.get_xaxis().set_visible(False)
            axes["cropped"].axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            axes["resized"].imshow(resized_img, aspect="equal")
            axes["resized"].axes.get_xaxis().set_visible(False)
            axes["resized"].axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            axes["downsampled"].imshow(downsampled_img, aspect="equal")
            axes["downsampled"].axes.get_xaxis().set_visible(False)
            axes["downsampled"].axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            axes["grayscale"].imshow(grayscale_img, aspect="equal")
            axes["grayscale"].axes.get_xaxis().set_visible(False)
            axes["grayscale"].axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.show(block=False)

        # shift the image stack and add the new one
        self.previous_image_stack = np.roll(self.previous_image_stack, shift=1, axis=2)
        self.previous_image_stack[0, :, :] = grayscale_img
        print("Time to process image: ", time.time() - t_now)
        # save_image = False
        # if save_image:
        #     cv2.imwrite("image.png", downsampled_img)

        return self.previous_image_stack


class TupleObservation(ObservationType):
    """Observation consisting of multiple observation types."""

    def __init__(self, env: "COLAVEnvironment", observation_configs: list, **kwargs) -> None:
        super().__init__(env)
        self.observation_types = [observation_factory(env, obs_config) for obs_config in observation_configs]

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def normalize(self, obs: Observation) -> Observation:
        return tuple(obs_type.normalize(obs) for obs_type in self.observation_types)

    def unnormalize(self, obs: Observation) -> Observation:
        return tuple(obs_type.unnormalize(obs) for obs_type in self.observation_types)

    def observe(self) -> Observation:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class DictObservation(ObservationType):
    """Observation consisting of multiple observation types.
    Observations are packed into a gymnasium type dictionary with a key for each
    observation type.
    """

    def __init__(self, env: "COLAVEnvironment", observation_configs: list, **kwargs) -> None:
        super().__init__(env)
        self.observation_types = [observation_factory(env, obs_config) for obs_config in observation_configs]

    def space(self) -> gym.spaces.Space:
        obs_space = dict()

        for obs_type in self.observation_types:
            obs_space[obs_type.name] = obs_type.space()

        return gym.spaces.Dict(obs_space)

    def unnormalize(self, obs: Observation) -> Observation:
        unnormalized_obs = dict()

        for obs_type in self.observation_types:
            unnormalized_obs[obs_type.name] = obs_type.unnormalize(obs[obs_type.name])

        return unnormalized_obs

    def normalize(self, obs: Observation) -> Observation:
        normalized_obs = dict()

        for obs_type in self.observation_types:
            normalized_obs[obs_type.name] = obs_type.normalize(obs[obs_type.name])

        return normalized_obs

    def observe(self) -> Observation:
        obs = dict()

        for obs_type in self.observation_types:
            obs[obs_type.name] = obs_type.observe()

        return obs


def observation_factory(
    env: "COLAVEnvironment", observation_type: str | dict = "time_observation", **kwargs
) -> ObservationType:
    """Factory for creating observation spaces.

    Args:
        env: Used environment.
        observation_type (str): Observation type name.
        **kwargs: Additional arguments to pass to the observation type.

    Returns:
        ObservationType: Observation type to use
    """
    if "lidar_like_observation" in observation_type:
        return LidarLikeObservation(env, **kwargs)
    elif "navigation_3dof_state_observation" in observation_type:
        return Navigation3DOFStateObservation(env, **kwargs)
    elif "navigation_csog_state_observation" in observation_type:
        return NavigationCSOGStateObservation(env, **kwargs)
    elif "perception_image_observation" in observation_type:
        return PerceptionImageObservation(env, **kwargs)
    elif "tracking_observation" in observation_type:
        return TrackingObservation(env, **kwargs)
    elif "time_observation" in observation_type:
        return TimeObservation(env, **kwargs)
    elif "tuple_observation" in observation_type:
        return TupleObservation(env, observation_type["tuple_observation"], **kwargs)
    elif "dict_observation" in observation_type:
        return DictObservation(env, observation_type["dict_observation"], **kwargs)
    else:
        raise ValueError("Unknown observation type")
