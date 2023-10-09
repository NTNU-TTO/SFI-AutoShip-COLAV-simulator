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
import colav_simulator.common.map_functions as mapf
import colav_simulator.core.sensing as sensing
import colav_simulator.core.tracking.trackers as trackers
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
import shapely.geometry as sgeo
import shapely

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
        env: "COLAVEnvironment"
    ) -> None:
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

    def space(self) -> gym.spaces.Space:
        """Get the observation space."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(3*self.n_sectors,))

    def define_observation_ranges(self) -> None:
        """Define the ranges for the observation space."""
        closeness_range = np.array([0.0, 1.0])
        default_do_speed_range = np.array([-20.0, 20.0])
        self.observation_range = {
            "closeness": closeness_range,
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
        
        normalized_obs = np.concatenate(
            [
                [
                    mf.linear_map(obs[i], self.observation_range["closeness"], (-1.0, 1.0)) for i in range(self.n_sectors)
                ],
                [
                    mf.linear_map(obs[j], self.observation_range["do_speed"], (-1.0, 1.0)) for j in range(self.n_sectors, self.n_sectors*3)
                ]
            ]
        )
        
        return normalized_obs

    def observe(self) -> Observation:
        """Get an observation of the environment state."""
        assert self._ownship is not None, "Ownship is not defined"
        
        ownship_pos = sgeo.Point((self._ownship.state[1], self._ownship.state[0])) # Ownship position as (easting, northing) coordinates

        sensor_range = self.sensing_range
        sensor_angles = self._sensor_angles + self._ownship.heading # Rotation of sensor suite

        grounding_hazards = self.env.relevant_grounding_hazards
        dynamic_obstacles = self.env.dynamic_obstacles
        
        # Convert ship objects to polygons
        dynamic_obstacle_polygons = []
        for dynamic_obstacle in dynamic_obstacles:
            do_state = dynamic_obstacle.state
            dynamic_obstacle_polygons.append(mapf.create_ship_polygon(  x=do_state[0],
                                                                        y=do_state[1],
                                                                        heading=dynamic_obstacle.heading,
                                                                        length=dynamic_obstacle.length,
                                                                        width=dynamic_obstacle.width
                                                                        ))

        # Determine distance and velocities to obstacles in each sector
        obstacle_distances = list()
        obstacle_velocities = list()
        for isensor in range(self.n_sensors):
            closest_obstacle_dist, closest_obstacle_vel = self._cast_sensor_ray(ownship_pos=ownship_pos,
                                                                                sensor_angle=sensor_angles[isensor],
                                                                                sensor_range=sensor_range,
                                                                                grounding_hazards=grounding_hazards,
                                                                                dynamic_obstacles=dynamic_obstacle_polygons
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
        for i in range(len(obstacle_distances_by_sector)):
            # Closeness of reachable distance
            sector_feasible_distance = self._feasibility_pooling(obstacle_distances_by_sector[i])
            print(sector_feasible_distance)
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
        I = np.argsort(x)
        for i in I:
            arc_length = theta*x[i]
            opening_width = arc_length/2
            opening_found = False
            for j in range(len(x)):
                if x[j] > x[i]:
                    opening_width += arc_length
                    if opening_width > ship_width:
                        opening_found = True
                        continue
                else:
                    opening_width += arc_length/2
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
        self._delta_sensor_angle = 2*np.pi/self.n_sensors   # Calculate angle between neighboring sensors
        self._sensor_angles = [-np.pi + i*self._delta_sensor_angle for i in range(self.n_sensors)]

        sector_start_indices = [0]
        current_sector = 0
        for isensor in range(self.n_sensors):
            sector = self._map_sensor_to_sector(isensor)
            if sector != current_sector:
                sector_start_indices.append(isensor)
                current_sector = sector
        
        self._sector_start_indices = sector_start_indices

    def _map_sensor_to_sector(self, isensor):
        """ Maps a sensor index i in {1,...,N} to a sector index k in {1, ..., D}
        
        Args:
            - isensor (int): Index of current sensor
            
        Returns:
            int: Index of the corresponding sector
        """
        a = self.n_sensors
        b = self.n_sectors
        c = 0.1
        sigma = lambda x: b / (1 + np.exp((-x + a / 2) / (c * a)))
        return int(np.floor(sigma(isensor) - sigma(0)))

    def _get_closeness(self, distance: float):
        """ Calculate closeness of obstacle as described in Meyer et al(2020). 
        The function evaluates to 0 if obstacle is undetected, and 1 if the vessel has
        collided with the obstacle.
        
        Args:
            - distance(float): Distance from ownship to obstacle
        
        Returns:
            float: Closeness to obstacle ranging from 0 to 1
        
        """
        closeness = np.clip(a=(1 - np.log(distance + 1)/np.log(self.sensing_range + 1)), a_min=0, a_max=1)
        return closeness

    def _cast_sensor_ray(self, ownship_pos: sgeo.Point, sensor_angle: float, sensor_range: float, grounding_hazards: list, dynamic_obstacles: list):
        """Cast sensor ray and return coordinates of closest obstacle 
        
        Args:
            - ownship_pos (Point): Ownship coordinates (east, north)
            - sensor_angle (float): Angle of sensor relative to ownships body frame [rad]
            - sensor_range (float): Range of sensor [m]
            - grounding_hazards (list): List of grounding hazards as polygons
            - dynamic_obstacles (list): List of dynamic obstacles as polygons
            
        Returns:
            Tuple[float | float]: Distance and velocity of the closest detected obstacle as a tuple.
        """
        sensor_endpoint = (
                ownship_pos.x + np.sin(sensor_angle)*sensor_range,
                ownship_pos.y + np.cos(sensor_angle)*sensor_range
            )

        sensor_ray = sgeo.LineString([ownship_pos, sensor_endpoint])
        closest_obstacle_dist = sensor_range
        closest_obstacle_vel = (0,0)
        
        # Static obstacles
        for grounding_hazard in grounding_hazards:
            if shapely.intersects(sensor_ray, grounding_hazard):
                intersection = sensor_ray.intersection(grounding_hazard)
                intersection = mapf.standardize_polygon_intersections(intersection)
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
                    vxvy_state = mhm.convert_csog_state_to_vxvy_state(csog_state)
                    closest_obstacle_vel = np.array([vxvy_state[2], vxvy_state[3]]).T
                    closest_obstacle_vel = mf.Rmtrx2D(-sensor_angle - np.pi/2).dot(closest_obstacle_vel)
        
        return closest_obstacle_dist, closest_obstacle_vel
                

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
    elif "tuple_observation" in observation_type:
        return TupleObservation(env, observation_type["tuple_observation"], **kwargs)
    else:
        raise ValueError("Unknown observation type")
