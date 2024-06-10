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
from typing import TYPE_CHECKING, Dict, Tuple, Union

import colav_simulator.common.image_helper_methods as imghf
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.plotters as plotters
import colav_simulator.core.guidances as guidances
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scimg
import shapely
import shapely.geometry as sgeo
from matplotlib import gridspec

Observation = Union[tuple, list, np.ndarray, Dict[str, np.ndarray]]

if TYPE_CHECKING:
    from colav_simulator.gym.environment import COLAVEnvironment


class ObservationType(ABC):
    """Interface class for observation types for the COLAVEnvironment gym."""

    name: str = "AbstractObservation"

    def __init__(self, env: "COLAVEnvironment") -> None:
        self.env = env

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
        assert self.env.ownship is not None, "Ownship is not defined"

        ownship_pos = sgeo.Point(
            (self.env.ownship.state[1], self.env.ownship.state[0])
        )  # Ownship position as (easting, northing) coordinates

        sensor_range = self.sensing_range
        sensor_angles = self._sensor_angles + self.env.ownship.heading  # Rotation of sensor suite

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
        ship_width = self.env.ownship.get_ship_info()["width"]
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


class PathRelativeNavigationObservation(ObservationType):
    """Observes the own-ship navigational info relative to a nominal geometric path on
    the form:

    [distance_to_path, speed_reference_error, u, v, r]
    """

    def __init__(
        self,
        env: "COLAVEnvironment",
    ) -> None:
        super().__init__(env)
        assert self.env.ownship is not None, "Ownship is not defined"
        assert (
            self.env.ownship.waypoints is not None and self.env.ownship.speed_plan is not None
        ), "Ownship waypoints and speed plan are not defined"
        self.name = "PathRelativeNavigationObservation"
        self.size = 5
        self.define_observation_ranges()
        self._ktp = guidances.KinematicTrajectoryPlanner()
        self._debug: bool = False
        self._map_origin: np.ndarray = np.array([0.0, 0.0])
        self._speed_spline = None
        self._final_arc_length = None
        self._path_coords = None
        self._path_linestring = None

    def plot_path(self) -> None:
        """Plot the nominal path."""
        if self._debug:
            self.env.enc.start_display()
            nominal_trajectory = self._ktp.compute_reference_trajectory(2.0)
            nominal_trajectory = nominal_trajectory + np.array(
                [self._map_origin[0], self._map_origin[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).reshape(9, 1)
            plotters.plot_waypoints(
                self.env.ownship.waypoints[:2, :],
                draft=1.0,
                enc=self.env.enc,
                color="orange",
                point_buffer=3.0,
                disk_buffer=6.0,
                hole_buffer=3.0,
                alpha=0.4,
            )
            plotters.plot_trajectory(
                nominal_trajectory[:2, :],
                self.env.enc,
                "yellow",
            )

    def create_path(self) -> None:
        """Creates a nominal path + speed spline based on the ownship waypoints and speed plan."""
        self._map_origin = self._map_origin = self.env.ownship.csog_state[:2]
        speed_plan = self.env.ownship.speed_plan.copy()
        speed_plan[speed_plan > 7.0] = 6.0
        speed_plan[speed_plan < 2.0] = 2.0
        speed_plan[-1] = 0.0
        os_nominal_path = self._ktp.compute_splines(
            waypoints=self.env.ownship.waypoints - np.array([self._map_origin[0], self._map_origin[1]]).reshape(2, 1),
            speed_plan=speed_plan,
            arc_length_parameterization=True,
        )
        x_spline, y_spline, _, self._speed_spline, self._final_arc_length = os_nominal_path
        self.plot_path()
        path_vals = np.linspace(0, self._final_arc_length, 1000)
        self._path_coords = x_spline(path_vals), y_spline(path_vals)
        self._path_linestring = sgeo.LineString(np.array([self._path_coords[0], self._path_coords[1]]).T)

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        self.observation_range = {
            "distance": (0.0, 2000.0),
            "speed": (-self.env.ownship.max_speed, self.env.ownship.max_speed),
            "turn_rate": (-self.env.ownship.max_turn_rate, self.env.ownship.max_turn_rate),
        }

    def normalize(self, obs: Observation) -> Observation:
        normalized_obs = np.array(
            [
                mf.linear_map(obs[0], self.observation_range["distance"], (-1.0, 1.0)),
                mf.linear_map(obs[1], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[2], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[3], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[4], self.observation_range["turn_rate"], (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return normalized_obs

    def unnormalize(self, obs: Observation) -> Observation:
        unnormalized_obs = np.array(
            [
                mf.linear_map(obs[0], (-1.0, 1.0), self.observation_range["distance"]),
                mf.linear_map(obs[1], (-1.0, 1.0), self.observation_range["speed"]),
                mf.linear_map(obs[2], (-1.0, 1.0), self.observation_range["speed"]),
                mf.linear_map(obs[3], (-1.0, 1.0), self.observation_range["speed"]),
                mf.linear_map(obs[4], (-1.0, 1.0), self.observation_range["turn_rate"]),
            ],
            dtype=np.float32,
        )
        return unnormalized_obs

    def get_closest_arclength(self, p: np.ndarray) -> float:
        """
        Returns the arc length value corresponding to the point
        on the path which is closest to the specified position.

        Args:
            p (np.ndarray): The position to find the closest arc length for.

        Returns:
            float: The arc length value corresponding to the closest point on the path.
        """
        return self._path_linestring.project(sgeo.Point(p))

    def distance_to_path(self, p: np.ndarray) -> float:
        """Calculate the distance from the position to the path.

        Args:
            p (np.ndarray): The current position of the ownship.

        Returns:
            float: The distance from the ownship to the path.
        """
        return self._path_linestring.distance(sgeo.Point(p[0], p[1]))

    def observe(self) -> Observation:
        if self.env.time < 0.0001:
            self.create_path()
        state = self.env.ownship.state - np.array([self._map_origin[0], self._map_origin[1], 0, 0, 0, 0])
        s = self.get_closest_arclength(state[:2])
        d2path = self.distance_to_path(state[:2])
        speed = np.linalg.norm(state[3:5])
        speed_diff = self._speed_spline(s) - speed
        obs = np.array([d2path, speed_diff, state[3], state[4], state[5]])
        return self.normalize(obs)


class Navigation3DOFStateObservation(ObservationType):
    """Observes the current own-ship 3DOF state, i.e. [x, y, psi, u, v, r]."""

    def __init__(
        self,
        env: "COLAVEnvironment",
    ) -> None:
        super().__init__(env)
        assert self.env.ownship is not None, "Ownship is not defined"
        self.name = "Navigation3DOFStateObservation"
        self.size = len(self.env.ownship.state)
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        (x_min, y_min, x_max, y_max) = self.env.enc.bbox
        self.observation_range = {
            "north": (y_min, y_max),
            "east": (x_min, x_max),
            "angles": (-np.pi, np.pi),
            "surge": (-self.env.ownship.max_speed, self.env.ownship.max_speed),
            "sway": [-self.env.ownship.max_speed, self.env.ownship.max_speed],
            "turn_rate": (-self.env.ownship.max_turn_rate, self.env.ownship.max_turn_rate),
        }

    def normalize(self, obs: Observation) -> Observation:
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
        assert self.env.ownship is not None, "Ownship is not defined"
        if self.env.time < 0.0001:
            self.define_observation_ranges()
        state = self.env.ownship.state
        obs = state
        return self.normalize(obs)


class DisturbanceObservation(ObservationType):
    """Observes the current disturbance state, i.e. [V_c, beta_c, V_w, beta_w]."""

    def __init__(
        self,
        env: "COLAVEnvironment",
    ) -> None:
        super().__init__(env)
        assert self.env.ownship is not None, "Ownship is not defined"
        self.name = "DisturbanceObservation"
        self.size = 4  # only current and wind is considered
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        self.observation_range = {
            "speed": (0.0, 20.0),
            "angles": (-np.pi, np.pi),
        }

    def normalize(self, obs: Observation) -> Observation:
        normalized_obs = np.array(
            [
                mf.linear_map(obs[0], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[1], self.observation_range["angles"], (-1.0, 1.0)),
                mf.linear_map(obs[2], self.observation_range["speed"], (-1.0, 1.0)),
                mf.linear_map(obs[3], self.observation_range["angles"], (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return normalized_obs

    def unnormalize(self, obs: Observation) -> Observation:
        unnormalized_obs = np.array(
            [
                mf.linear_map(obs[0], (-1.0, 1.0), self.observation_range["speed"]),
                mf.linear_map(obs[1], (-1.0, 1.0), self.observation_range["angles"]),
                mf.linear_map(obs[2], (-1.0, 1.0), self.observation_range["speed"]),
                mf.linear_map(obs[3], (-1.0, 1.0), self.observation_range["angles"]),
            ],
            dtype=np.float32,
        )
        return unnormalized_obs

    def observe(self) -> Observation:
        os_course = self.env.ownship.csog_state[3]
        disturbance = self.env.disturbance
        obs = np.zeros(self.size)
        if disturbance is None:
            return self.normalize(obs)

        ddata = disturbance.get()
        if "speed" in ddata.currents:
            obs[0] = ddata.currents["speed"]
            obs[1] = mf.wrap_angle_diff_to_pmpi(ddata.currents["direction"], os_course)
        if "speed" in ddata.wind:
            obs[2] = ddata.wind["speed"]
            obs[3] = mf.wrap_angle_diff_to_pmpi(ddata.wind["direction"], os_course)
        return self.normalize(obs)


class GroundTruthTrackingObservation(ObservationType):
    """Observation containing a dict of augmented states [x, y, vx, vy, length, width] and covariances for the dynamic obstacles, non-normalized and non-relative to the own-ship."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        super().__init__(env)
        assert self.env.ownship is not None, "Ownship is not defined"
        self.max_num_do = 15
        self.do_info_size = 6  # [x, y, Vx, Vy, length, width]
        self.name = "GroundTruthTrackingObservation"
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1e12, high=1e12, shape=(self.do_info_size, self.max_num_do), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        (x_min, y_min, x_max, y_max) = self.env.enc.bbox
        self.observation_range = {
            "north": (y_min, y_max),
            "east": (x_min, x_max),
            "speed": (-20.0, 20.0),
            "angles": (-np.pi, np.pi),
            "length": (0.0, 100.0),
            "width": (0.0, 100.0),
        }

    def normalize(self, obs: Observation) -> Observation:
        norm_obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx in range(self.max_num_do):
            norm_obs[:, idx] = np.array(
                [
                    mf.linear_map(obs[0, idx], self.observation_range["north"], (-1.0, 1.0)),
                    mf.linear_map(obs[1, idx], self.observation_range["east"], (-1.0, 1.0)),
                    mf.linear_map(obs[2, idx], self.observation_range["speed"], (-1.0, 1.0)),
                    mf.linear_map(obs[3, idx], self.observation_range["speed"], (-1.0, 1.0)),
                    mf.linear_map(obs[4, idx], self.observation_range["length"], (-1.0, 1.0)),
                    mf.linear_map(obs[5, idx], self.observation_range["width"], (-1.0, 1.0)),
                ]
            )
        return norm_obs

    def unnormalize(self, obs: Observation) -> Observation:
        unnorm_obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx in range(self.max_num_do):
            unnorm_obs[:, idx] = np.array(
                [
                    mf.linear_map(obs[0, idx], (-1.0, 1.0), self.observation_range["north"]),
                    mf.linear_map(obs[1, idx], (-1.0, 1.0), self.observation_range["east"]),
                    mf.linear_map(obs[2, idx], (-1.0, 1.0), self.observation_range["speed"]),
                    mf.linear_map(obs[3, idx], (-1.0, 1.0), self.observation_range["speed"]),
                    mf.linear_map(obs[4, idx], (-1.0, 1.0), self.observation_range["length"]),
                    mf.linear_map(obs[5, idx], (-1.0, 1.0), self.observation_range["width"]),
                ]
            )
        return unnorm_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self.env.ownship is not None, "Ownship is not defined"
        do_list = self.env.dynamic_obstacles
        obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx, do_ship in enumerate(do_list):
            do_csog_state = do_ship.csog_state
            do_state = mhm.convert_state_to_vxvy_state(do_csog_state)
            obs[:6, idx] = np.array([do_state[0], do_state[1], do_state[2], do_state[3], do_ship.length, do_ship.width])
        return obs


class TrackingObservation(ObservationType):
    """Observation containing a dict of augmented states [x, y, vx, vy, length, width] and covariances for the dynamic obstacles, non-normalized and non-relative to the own-ship."""

    def __init__(self, env: "COLAVEnvironment") -> None:
        super().__init__(env)
        assert self.env.ownship is not None, "Ownship is not defined"
        self.max_num_do = 15
        self.do_info_size = 6 + 16  # [x, y, Vx, Vy, length, width] + covariance matrix of 4x4
        self.name = "TrackingObservation"
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1e12, high=1e12, shape=(self.do_info_size, self.max_num_do), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        (x_min, y_min, x_max, y_max) = self.env.enc.bbox
        self.observation_range = {
            "north": (y_min, y_max),
            "east": (x_min, x_max),
            "speed": (-20.0, 20.0),
            "angles": (-np.pi, np.pi),
            "length": (0.0, 100.0),
            "width": (0.0, 100.0),
            "covariance": (-100.0, 100.0),
        }

    def normalize(self, obs: Observation) -> Observation:
        norm_obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx in range(self.max_num_do):
            norm_obs[:6, idx] = np.array(
                [
                    mf.linear_map(obs[0, idx], self.observation_range["north"], (-1.0, 1.0)),
                    mf.linear_map(obs[1, idx], self.observation_range["east"], (-1.0, 1.0)),
                    mf.linear_map(obs[2, idx], self.observation_range["speed"], (-1.0, 1.0)),
                    mf.linear_map(obs[3, idx], self.observation_range["speed"], (-1.0, 1.0)),
                    mf.linear_map(obs[4, idx], self.observation_range["length"], (-1.0, 1.0)),
                    mf.linear_map(obs[5, idx], self.observation_range["width"], (-1.0, 1.0)),
                ]
            )
            for el in range(6, self.do_info_size):
                norm_obs[el, idx] = mf.linear_map(obs[el, idx], self.observation_range["covariance"], (-1.0, 1.0))
        return norm_obs

    def unnormalize(self, obs: Observation) -> Observation:
        unnorm_obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx in range(self.max_num_do):
            unnorm_obs[:6, idx] = np.array(
                [
                    mf.linear_map(obs[0, idx], (-1.0, 1.0), self.observation_range["north"]),
                    mf.linear_map(obs[1, idx], (-1.0, 1.0), self.observation_range["east"]),
                    mf.linear_map(obs[2, idx], (-1.0, 1.0), self.observation_range["speed"]),
                    mf.linear_map(obs[3, idx], (-1.0, 1.0), self.observation_range["speed"]),
                    mf.linear_map(obs[4, idx], (-1.0, 1.0), self.observation_range["length"]),
                    mf.linear_map(obs[5, idx], (-1.0, 1.0), self.observation_range["width"]),
                ]
            )
            for el in range(6, self.do_info_size):
                unnorm_obs[el, idx] = mf.linear_map(obs[el, idx], (-1.0, 1.0), self.observation_range["covariance"])
        return unnorm_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self.env.ownship is not None, "Ownship is not defined"
        if self.env.time < 0.0001:
            true_ship_states = mhm.extract_do_states_from_ship_list(self.env.time, self.env.ship_list)
            relevant_do_states = mhm.get_relevant_do_states(true_ship_states, idx=0)
            self.env.ownship.track_obstacles(self.env.time, self.env.time_step, relevant_do_states)

        tracks, _ = self.env.ownship.get_do_track_information()
        obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx, (do_idx, do_state, do_cov, do_length, do_width) in enumerate(tracks):
            obs[:6, idx] = np.array([do_state[0], do_state[1], do_state[2], do_state[3], do_length, do_width])
            obs[6:, idx] = do_cov.flatten()
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
        return gym.spaces.Box(low=0.0, high=1500.0, shape=(1,), dtype=np.float32)

    def normalize(self, obs: Observation) -> Observation:
        normalized_obs = np.array(
            [
                mf.linear_map(obs[0], (self.t_start, self.t_end), (-1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return normalized_obs

    def unnormalize(self, obs: Observation) -> Observation:
        unnormalized_obs = np.array([mf.linear_map(obs[0], (-1.0, 1.0), (self.t_start, self.t_end))], dtype=np.float32)
        return unnormalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        if self.env.time < 0.0001:
            self.t_start = self.env.time
        obs = np.array([self.env.time], dtype=np.float32)
        return obs


class PerceptionImageObservation(ObservationType):
    """Observation consisting of a perception image. INCOMPLETE"""

    def __init__(self, env: "COLAVEnvironment", image_dim: Tuple[int, int, int] = (1, 256, 256), **kwargs) -> None:
        """
        Args:
            env (COLAVEnvironment): The environment to observe.
            image_dim (Tuple[int, int, int], optional): The dimensions of the image. Defaults to (1, 256, 256) (history window of 1)
        """
        super().__init__(env)
        self.name = "PerceptionImageObservation"
        self.image_dim = image_dim
        self.observation_counter: int = 0
        self.previous_image_stack: np.ndarray = np.zeros(image_dim, dtype=np.uint8)  # All black
        self.t_prev: float = 0.0
        self.render_rate: float = 0.5  # Hz

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=0, high=255, shape=self.image_dim, dtype=np.uint8)

    def normalize(self, obs: Observation) -> Observation:
        # No normalization is provided for this observation type
        return obs

    def unnormalize(self, obs: Observation) -> Observation:
        # No unnormalization is provided for this observation type
        return obs

    def toggle_unneccessary_liveplot_features(self, show: bool) -> None:
        self.env.viewer2d.toggle_uniform_seabed_color(show)
        self.env.viewer2d.toggle_liveplot_sensor_measurement_visibility(show)
        self.env.viewer2d.toggle_liveplot_trajectory_visibility(show)
        self.env.viewer2d.toggle_liveplot_waypoint_visibility(show)
        self.env.viewer2d.toggle_liveplot_disturbance_visibility(show)
        self.env.viewer2d.toggle_liveplot_dynamic_obstacle_visibility(show)
        self.env.viewer2d.toggle_misc_plot_visibility(show)

    def observe(self) -> Observation:
        assert self.env.ownship is not None, "Ownship is not defined"
        # t_now = time.time()
        if self.env.time < 0.001:
            self.t_prev = self.env.time

        self.toggle_unneccessary_liveplot_features(show=False)
        img = self.env.liveplot_image.copy()
        self.toggle_unneccessary_liveplot_features(show=True)
        pruned_img = img
        # pruned_img = imghf.remove_whitespace(img)
        # os_liveplot_zoom_width = self.env.liveplot_zoom_width

        os_heading = self.env.ownship.heading
        #
        # Rotate the image to align with the ownship heading
        rotated_img = scimg.rotate(pruned_img, np.rad2deg(os_heading), reshape=False)
        npx, npy = rotated_img.shape[:2]

        # Crop the image to the vessel
        # image width and height corresponds to os_liveplot_zoom_width x os_liveplot_zoom_width
        # Image coordinate system is (0,0) in the upper left corner, height is the first index, width is the second index
        center_pixel_x = int(rotated_img.shape[0] // 2)
        center_pixel_y = int(rotated_img.shape[1] // 2)
        cutoff_index_below_vessel = int(0.05 * npx)  # corresponds to 100 m for a 2000 m zoom width
        cutoff_index_below_vessel = (
            cutoff_index_below_vessel if cutoff_index_below_vessel <= center_pixel_y else center_pixel_y
        )

        cutoff_index_above_vessel = int(0.3 * npx)  # corresponds to 600 m for a 1000 m zoom width
        cutoff_index_above_vessel = (
            cutoff_index_above_vessel if cutoff_index_above_vessel <= center_pixel_x else center_pixel_x
        )
        cutoff_laterally = int(0.2 * npy)  # corresponds to 150 m for a 1000 m zoom width

        cropped_img = rotated_img[
            center_pixel_x - cutoff_index_above_vessel : center_pixel_x + cutoff_index_below_vessel,
            center_pixel_y - cutoff_laterally : center_pixel_y + cutoff_laterally,
        ]

        # downsample the image to configured image shape
        downsampled_img = cv2.resize(cropped_img, (self.image_dim[1], self.image_dim[2]), interpolation=cv2.INTER_AREA)
        grayscale_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2GRAY)

        if False:
            fig = plt.figure()
            gs = gridspec.GridSpec(
                3,
                2,
                fig,
                wspace=0.0,
                hspace=0.0,
                top=0.95,
                bottom=0.05,
                left=0.17,
                right=0.845,
            )

            plt.show(block=False)
            ax = plt.subplot(gs[0, 0])
            ax.imshow(img, aspect="equal")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax = plt.subplot(gs[0, 1])
            ax.imshow(rotated_img, aspect="equal")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax = plt.subplot(gs[1, 0])
            ax.imshow(cropped_img, aspect="equal")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax = plt.subplot(gs[1, 1])
            ax.imshow(downsampled_img, aspect="equal")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax = plt.subplot(gs[2, 0])
            ax.imshow(grayscale_img, aspect="equal")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax = plt.subplot(gs[2, 1])
            ax.imshow(grayscale_img, aspect="equal")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.subplots_adjust(wspace=0.01, hspace=0.01)

        # shift the image stack and add the new one
        if self.image_dim[0] > 1:
            self.previous_image_stack = np.roll(self.previous_image_stack, shift=1, axis=0)
        self.previous_image_stack[0, :, :] = grayscale_img
        # print("Time to process image: ", time.time() - t_now)
        # save_image = False
        # if save_image:
        #     cv2.imwrite("image.png", downsampled_img)
        self.t_prev = self.env.time
        return self.previous_image_stack


class DictObservation(ObservationType):
    """Observation consisting of multiple observation types.
    Observations are packed into a gymnasium type dictionary with a key for each
    observation type.
    """

    def __init__(self, env: "COLAVEnvironment", observation_configs: list, **kwargs) -> None:
        super().__init__(env)
        self.name = "DictObservation"
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


class RelativeTrackingObservation(ObservationType):
    """Compact observation containing an array of relative dynamic obstacle info
    on the form (relative distance, relative speed body-x, relative speed body-y, cov-var speed x, cov-var speed y, cross covar speed x-y), normalized.
    """

    def __init__(self, env: "COLAVEnvironment") -> None:
        super().__init__(env)
        self.max_num_do = 10
        self.do_info_size = 7  # [relative dist, relative speed body-x, relative speed body-y, cov-var speed x, cov-var speed y, cross covar speed x-y]
        self.name = "RelativeTrackingObservation"
        self.define_observation_ranges()

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.do_info_size, self.max_num_do), dtype=np.float32)

    def define_observation_ranges(self) -> None:
        assert self.env.ownship is not None, "Ownship is not defined"
        self.observation_range = {
            "distance": (0.0, 1300.0),
            "speed": (-20.0, 20.0),
            "angles": (-np.pi, np.pi),
            "variance": (0.0, 5.0),
            "cross_variance": (-5.0, 5.0),
        }

    def normalize(self, obs: Observation) -> Observation:
        norm_obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx in range(self.max_num_do):
            norm_obs[:, idx] = np.array(
                [
                    mf.linear_map(obs[0, idx], self.observation_range["distance"], (-1.0, 1.0)),
                    mf.linear_map(obs[1, idx], self.observation_range["angles"], (-1.0, 1.0)),
                    mf.linear_map(obs[2, idx], self.observation_range["speed"], (-1.0, 1.0)),
                    mf.linear_map(obs[3, idx], self.observation_range["speed"], (-1.0, 1.0)),
                    mf.linear_map(obs[4, idx], self.observation_range["variance"], (-1.0, 1.0)),
                    mf.linear_map(obs[5, idx], self.observation_range["variance"], (-1.0, 1.0)),
                    mf.linear_map(obs[6, idx], self.observation_range["cross_variance"], (-1.0, 1.0)),
                ]
            )
        return norm_obs

    def unnormalize(self, obs: Observation) -> Observation:
        unnorm_obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        for idx in range(self.max_num_do):
            unnorm_obs[:, idx] = np.array(
                [
                    mf.linear_map(obs[0, idx], (-1.0, 1.0), self.observation_range["distance"]),
                    mf.linear_map(obs[1, idx], (-1.0, 1.0), self.observation_range["angles"]),
                    mf.linear_map(obs[2, idx], (-1.0, 1.0), self.observation_range["speed"]),
                    mf.linear_map(obs[3, idx], (-1.0, 1.0), self.observation_range["speed"]),
                    mf.linear_map(obs[4, idx], (-1.0, 1.0), self.observation_range["variance"]),
                    mf.linear_map(obs[5, idx], (-1.0, 1.0), self.observation_range["variance"]),
                    mf.linear_map(obs[6, idx], (-1.0, 1.0), self.observation_range["cross_variance"]),
                ]
            )
        return unnorm_obs

    def observe(self) -> Observation:
        assert self.env.ownship is not None, "Ownship is not defined"
        if self.env.time < 0.0001:
            true_ship_states = mhm.extract_do_states_from_ship_list(self.env.time, self.env.ship_list)
            relevant_do_states = mhm.get_relevant_do_states(true_ship_states, idx=0)
            self.env.ownship.track_obstacles(self.env.time, self.env.time_step, relevant_do_states)

        os_state = self.env.ownship.state
        chi = self.env.ownship.course
        tracks, _ = self.env.ownship.get_do_track_information()
        obs = np.zeros((self.do_info_size, self.max_num_do), dtype=np.float32)
        obs[0, :] = self.observation_range["distance"][1]  # Set all distances to max value
        R_psi = mf.Rmtrx2D(os_state[2])
        for idx, (do_idx, do_state, do_cov, do_length, do_width) in enumerate(tracks):
            speed_cov = do_cov[2:4, 2:4]
            rel_speed = R_psi.T @ do_state[2:4] - os_state[3:5]
            rel_speed_cov = R_psi.T @ speed_cov @ R_psi
            rel_distance = np.linalg.norm(do_state[0:2] - os_state[0:2])
            los_angle = np.arctan2(do_state[1] - os_state[1], do_state[0] - os_state[0])
            rel_bearing = mf.wrap_angle_diff_to_pmpi(los_angle, chi)
            obs[:, idx] = np.array(
                [
                    rel_distance,
                    rel_bearing,
                    rel_speed[0],
                    rel_speed[1],
                    rel_speed_cov[0, 0],
                    rel_speed_cov[1, 1],
                    rel_speed_cov[0, 1],
                ]
            )
        obs = obs.T[obs.T[:, 0].argsort()][::-1].T
        # print(f"first tracking obs row = {obs[0, :]}")
        norm_obs = self.normalize(obs)
        norm_obs = np.clip(norm_obs, -1.0, 1.0)
        return norm_obs


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
    elif "path_relative_navigation_observation" in observation_type:
        return PathRelativeNavigationObservation(env, **kwargs)
    elif "navigation_3dof_state_observation" in observation_type:
        return Navigation3DOFStateObservation(env, **kwargs)
    elif "perception_image_observation" in observation_type:
        return PerceptionImageObservation(env, **kwargs)
    elif "relative_tracking_observation" in observation_type:
        return RelativeTrackingObservation(env, **kwargs)
    elif "ground_truth_tracking_observation" in observation_type:
        return GroundTruthTrackingObservation(env, **kwargs)
    elif "tracking_observation" in observation_type:
        return TrackingObservation(env, **kwargs)
    elif "disturbance_observation" in observation_type:
        return DisturbanceObservation(env, **kwargs)
    elif "time_observation" in observation_type:
        return TimeObservation(env, **kwargs)
    elif "dict_observation" in observation_type:
        return DictObservation(env, observation_type["dict_observation"], **kwargs)
    else:
        raise ValueError("Unknown observation type")
