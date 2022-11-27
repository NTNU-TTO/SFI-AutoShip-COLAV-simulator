"""
    ship.py

    Summary:
        Contains class definitions for ship classes.
        Every ship class must adhere to the interface
        IShip and must be built by a ShipBuilder.

    Author: Trym Tengesdal
"""
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import colav_simulator.common.math_functions as mf
import colav_simulator.ships.controllers as controllers
import colav_simulator.ships.guidances as guidances
import colav_simulator.ships.models as models
import numpy as np


@dataclass
class Config:
    """Configuration class for managing ship parameters."""

    # colav: colav.Config
    # tracker: tracker.config
    model: models.Config = models.Config()
    controller: controllers.Config = controllers.Config()
    guidance: guidances.Config = guidances.Config()
    pose: Optional[np.ndarray] = None
    waypoints: Optional[np.ndarray] = None
    speed_plan: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, config_dict: dict):

        config = Config()

        if "pose" in config_dict:
            config.pose = np.array(config_dict["pose"])
            config.pose[3] = np.deg2rad(config.pose[3])

        if "waypoints" in config_dict:
            config.waypoints = np.array(config_dict["waypoints"])

        if "speed_plan" in config_dict:
            config.speed_plan = np.array(config_dict["speed_plan"])

        if "default" in config_dict:
            return config

        config.model = models.Config.from_dict(config_dict["model"])

        config.controller = controllers.Config.from_dict(config_dict["controller"])

        config.guidance = guidances.Config.from_dict(config_dict["guidance"])

        return config

    def to_dict(self):
        config_dict = {}

        if self.pose is not None:
            config_dict["pose"] = self.pose.tolist()
            config_dict["pose"][3] = float(np.rad2deg(self.pose[3]))

        if self.waypoints is not None:
            config_dict["waypoints"] = self.waypoints.tolist()

        if self.speed_plan is not None:
            config_dict["speed_plan"] = self.speed_plan.tolist()

        config_dict["model"] = self.model.to_dict()

        config_dict["controller"] = self.controller.to_dict()

        config_dict["guidance"] = self.guidance.to_dict()

        return config_dict


class ShipBuilder:
    """Class for building all objects needed by a ship with a specific configuration of systems."""

    @classmethod
    def construct_ship(cls, config: Optional[Config] = None):
        """Builds a ship from the configuration

        Args:
            config (Optional[Config], optional): Ship configuration. Defaults to None.

        Returns:
            Tuple[Model, Controller, Guidance]: The subsystems comprising the ship: model, controller, guidance.
        """
        if config:
            model = cls.construct_model(config.model)
            controller = cls.construct_controller(config.controller)
            guidance_system = cls.construct_guidance(config.guidance)
        else:
            model = cls.construct_model()
            controller = cls.construct_controller()
            guidance_system = cls.construct_guidance()

        return model, controller, guidance_system

    @classmethod
    def construct_guidance(cls, config: Optional[guidances.Config] = None):
        """Builds a ship guidance method from the configuration

        Args:
            config (Optional[guidance.Config], optional): Guidance configuration. Defaults to None.

        Returns:
            Guidance: Guidance system as specified by the configuration, e.g. a LOSGuidance.
        """
        if config and config.los:
            return guidances.LOSGuidance(config)
        elif config and config.ktp:
            return guidances.KinematicTrajectoryPlanner(config)
        else:
            return guidances.LOSGuidance()

    @classmethod
    def construct_controller(cls, config: Optional[controllers.Config] = None):
        """Builds a ship model from the configuration

        Args:
            config (Optional[controllers.Config], optional): Model configuration. Defaults to None.

        Returns:
            Model: Model as specified by the configuration, e.g. a MIMOPID controller.
        """
        if config and config.pid:
            return controllers.MIMOPID(config)
        elif config and config.flsh:
            return controllers.FLSH(config)
        elif config and config.pass_through_cs:
            return controllers.PassThroughCS()
        else:
            return controllers.PassThroughCS()

    @classmethod
    def construct_model(cls, config: Optional[models.Config] = None):
        """Builds a ship model from the configuration

        Args:
            config (Optional[models.Config], optional): Model configuration. Defaults to None.

        Returns:
            Model: Model as specified by the configuration, e.g. a KinematicCSOG model.
        """
        if config and config.csog:
            return models.KinematicCSOG(config)
        elif config and config.telemetron:
            return models.Telemetron()
        else:
            return models.KinematicCSOG()


class IShip(ABC):
    @abstractmethod
    def forward(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Predict the ship dt seconds forward in time. Returns new state, inputs to get there and references used."


class Ship(IShip):
    """The Ship class implements a variety of models and guidance methods for use in simulating the ship behaviour.

    Internal variables:
        mmsi (float): Maritime Mobile Service Identity of the ship.
        ais_msg_nr (int): AIS message number (1, 2 or 3 for AIS Class A transponders, 18 for AIS Class B transponders).
        waypoints (np.ndarray): Nominal waypoints the ship is following.
        speed_plan (np.ndarray): Corresponding reference speeds the ship should follow between waypoint segments.
        state (np.ndarray): State of the ship, either `xs = [x, y, chi, U]` or `xs = [x, y, psi, u, v, r]^T`
                            where for the first case, `x` and `y` are planar coordinates (north-east), `chi` is course (rad),
                            `U` the ship forward speed (m/s). For the latter case, see the typical 3DOF surface vessel model
                            in `Fossen2011`.
    """

    _mmsi: int = 0
    _ais_msg_nr: int = 18
    _state: np.ndarray = np.zeros(4)
    _waypoints: np.ndarray = np.zeros((2, 0))
    _speed_plan: np.ndarray = np.zeros(0)

    def __init__(
        self,
        mmsi: int,
        pose: Optional[np.ndarray] = None,
        waypoints: Optional[np.ndarray] = None,
        speed_plan: Optional[np.ndarray] = None,
        config: Optional[Config] = None,
    ) -> None:

        self._mmsi = mmsi

        self._model, self._controller, self._guidance = ShipBuilder.construct_ship(config)

        if config and config.pose is not None:
            self.set_initial_state(config.pose)

        if config and config.waypoints is not None and config.speed_plan is not None:
            self.set_nominal_plan(config.waypoints, config.speed_plan)

        # Input pose, waypoints and speed plans take precedence over config specified ones
        if pose is not None:
            self.set_initial_state(pose)

        if waypoints is not None and speed_plan is not None:
            self.set_nominal_plan(waypoints, speed_plan)

        # Message number 1/2/3 for ships with AIS Class A, 18 for AIS Class B ships
        # AIS Class A is required for ships bigger than 300 GT. Approximately 45 m x 10 m ship would be 300 GT.
        if self.length <= 45:
            self._ais_msg_nr = 18
        else:
            self._ais_msg_nr = random.randint(1, 3)

    def forward(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts the ship state dt seconds forward in time.

        Args:
            dt (float): Time step (s) in the prediction.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The new state dt seconds ahead,
            inputs to get there and the references used.
        """
        if self._waypoints.size < 2 or self._speed_plan.size < 2:
            raise ValueError("Insufficient waypoints for the ship to follow!")

        if dt <= 0.0:
            raise ValueError("Time step must be strictly positive!")

        references = self._guidance.compute_references(self._waypoints, self._speed_plan, None, self._state, dt)

        u = self._controller.compute_inputs(references, self._state, dt, self._model)

        psi_prev = self._state[2]
        self._state = self._state + dt * self._model.dynamics(self._state, u)
        diff = mf.wrap_angle_diff_to_pmpi(self._state[2], psi_prev)
        if abs(diff) > np.deg2rad(5):
            print("psi_prev: {psi_prev} | psi_now: {self._state[2]}")

        return self._state, u, references

    def track_obstacles(self) -> np.ndarray:
        """Uses its target tracker to estimate the states of dynamic obstacles in the environment.

        Returns:
            np.ndarray: Updated states on obstacles in the environment.
        """
        # tracks = self._tracker.track()
        return np.zeros((2, 0))

    def set_initial_state(self, pose: np.ndarray) -> None:
        """Sets the initial state of the ship based on the input pose.

        Args:
            pose (np.ndarray): Initial pose = [x, y, U, chi] of the ship.
        """
        self._state = np.array([pose[0], pose[1], pose[3], pose[2], 0.0, 0.0])

    def set_nominal_plan(self, waypoints: np.ndarray, speed_plan: np.ndarray):
        """Reassigns waypoints and speed_plan to the ship, to change its objective.

        Args:
            waypoints (np.ndarray): New set of waypoints.
            speed_plan (np.ndarray): New corresponding set of speed references.
        """
        assert speed_plan.size == waypoints.shape[1]
        n_px, n_wps = waypoints.shape
        if n_px != 2:
            raise ValueError("Waypoints do not contain planar coordinates along each column!")

        if n_wps < 2:
            raise ValueError("Insufficient number of waypoints (< 2)!")

        self._waypoints = waypoints
        self._speed_plan = speed_plan

    def get_ship_nav_data(self, timestamp: int) -> dict:
        """Returns navigational++ related data for the ship

        Args:
            timestamp (int): UTC referenced timestamp.

        Returns:
            dict: Navigation data as dictionary
        """
        datetime_t = mf.utc_timestamp_to_datetime(timestamp)
        datetime_str = datetime_t.strftime("%d.%m.%Y %H:%M:%S")
        ship_nav_data = {
            "pose": self.pose,
            "waypoints": self._waypoints,
            "speed_plan": self._speed_plan,
            "timestamp": datetime_str,
            # predicted trajectory from COLAV/planner
            # "obstacles": self.track_obstacles(),
        }

        return ship_nav_data

    def get_ais_data(self, timestamp: int) -> dict:
        """Returns an AIS data message for the ship at the given timestamp.

        Args:
            timestamp (int): UTC referenced timestamp.

        Returns:
            dict: AIS message as a dictionary.
        """
        datetime_t = mf.utc_timestamp_to_datetime(timestamp)
        datetime_str = datetime_t.strftime("%d.%m.%Y %H:%M:%S")
        row = {
            "mmsi": self.mmsi,
            "lon": self.pose[1] * 10 ** (-5),
            "lat": self.pose[0] * 10 ** (-5),
            "date_time_utc": datetime_str,
            "sog": mf.mps2knots(self.pose[2]),
            "cog": int(np.rad2deg(self.pose[3])),
            "true_heading": int(np.rad2deg(self.heading)),
            "nav_status": 0,
            "message_nr": self.ais_msg_nr,
            "source": "",
        }
        return row

    @property
    def pose(self) -> np.ndarray:
        """Returns the ship pose as parameterized in an AIS message,
        i.e. `xs = [x, y, U, chi]` where `x` and `y` are planar coordinates (north-east),
        `U` the ship forward speed (m/s) and `chi` the course over ground (rad).

        Returns:
            np.ndarray: Ship pose.
        """
        if self._model.dims[0] == 4:
            return np.array([self._state[0], self._state[1], self._state[3], self._state[2]])
        else:  # self._model.dims[0] == 6
            heading = self._state[2]
            crab_angle = np.arctan2(self._state[4], self._state[3])
            cog = heading + crab_angle
            speed = np.sqrt(self._state[3] ** 2 + self._state[4] ** 2)
            return np.array([self._state[0], self._state[1], speed, cog])

    @property
    def max_speed(self) -> float:
        return self._model.pars.U_max

    @property
    def min_speed(self) -> float:
        return self._model.pars.U_min

    @property
    def max_turn_rate(self) -> float:
        return self._model.pars.r_max

    @property
    def draft(self) -> float:
        return self._model.pars.draft

    @property
    def length(self) -> float:
        return self._model.pars.length

    @property
    def width(self) -> float:
        return self._model.pars.width

    @property
    def mmsi(self) -> int:
        return self._mmsi

    @property
    def ais_msg_nr(self) -> int:
        return self._ais_msg_nr

    @property
    def ais_msg_freq(self) -> float:
        return 1.0 / 3.0

    @property
    def heading(self):
        if self._state.size == 4:
            return self._state[3]
        else:  # self._state.size == 6
            return self._state[2]

    @property
    def waypoints(self) -> np.ndarray:
        return self._waypoints
