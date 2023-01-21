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

import colav_evaluation_tool.vessel as colav_eval_vessel_data
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.sensors as ssensors
import colav_simulator.core.tracking.trackers as trackers
import numpy as np


@dataclass
class Config:
    """Configuration class for managing ship parameters."""

    # colav: colav.Config
    model: models.Config = models.Config()
    controller: controllers.Config = controllers.Config()
    guidance: guidances.Config = guidances.Config()
    sensors: ssensors.Config = ssensors.Config()
    tracker: trackers.Config = trackers.Config()
    mmsi: int = -1  # MMSI number of the ship, if configured equal to the MMSI of a ship in AIS data, the ship will be initialized with the data from the AIS data.
    id: int = -1  # Ship identifier
    t_start: Optional[float] = None  # Determines when the ship should start in the simulation
    t_end: Optional[float] = None  # Determines when the ship ends its part in the simulation
    random_generated: Optional[bool] = False  # True if the ship should have randomly generated pose, wps and speed plan. Takes priority over mmsi.
    pose: Optional[np.ndarray] = None  # In format [x[north], y[east], SOG [m/s], COG[deg]]
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

        if "t_start" in config_dict:
            config.t_start = config_dict["t_start"]

        if "t_end" in config_dict:
            config.t_end = config_dict["t_end"]

        if "random_generated" in config_dict:
            config.random_generated = config_dict["random_generated"]

        config.id = config_dict["id"]
        config.mmsi = config_dict["mmsi"]

        config.model = models.Config.from_dict(config_dict["model"])

        config.controller = controllers.Config.from_dict(config_dict["controller"])

        config.guidance = guidances.Config.from_dict(config_dict["guidance"])

        config.sensors = ssensors.Config.from_dict(config_dict["sensors"])

        config.tracker = trackers.Config.from_dict(config_dict["tracker"])

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

        if self.t_start is not None:
            config_dict["t_start"] = self.t_start

        if self.t_end is not None:
            config_dict["t_end"] = self.t_end

        if self.random_generated is not None:
            config_dict["random_generated"] = self.random_generated

        config_dict["id"] = self.id
        config_dict["mmsi"] = self.mmsi

        config_dict["model"] = self.model.to_dict()

        config_dict["controller"] = self.controller.to_dict()

        config_dict["guidance"] = self.guidance.to_dict()

        config_dict["sensors"] = self.sensors.to_dict_list()

        config_dict["tracker"] = self.tracker.to_dict()

        return config_dict


class ShipBuilder:
    """Class for building all objects needed by a ship with a specific configuration of systems."""

    @classmethod
    def construct_ship(cls, config: Optional[Config] = None):
        """Builds a ship from the configuration

        Args:
            config (Optional[Config], optional): Ship configuration. Defaults to None.

        Returns:
            Tuple[Model, Controller, Guidance, list[Sensor], Tracker]: The subsystems comprising the ship: model, controller, guidance.
        """
        if config:
            model = cls.construct_model(config.model)
            controller = cls.construct_controller(config.controller)
            guidance_system = cls.construct_guidance(config.guidance)
            sensors = cls.construct_sensors(config.sensors)
            tracker = cls.construct_tracker(sensors, config.tracker)
        else:
            model = cls.construct_model()
            controller = cls.construct_controller()
            guidance_system = cls.construct_guidance()
            sensors = cls.construct_sensors()
            tracker = cls.construct_tracker(sensors)

        return model, controller, guidance_system, sensors, tracker

    @classmethod
    def construct_tracker(cls, sensors: list, config: Optional[trackers.Config] = None):
        """Builds a tracker from the configuration

        Args:
            sensors (list): Sensors used by the tracker.
            config (Optional[Config], optional): Tracker configuration. Defaults to None.

        Returns:
            Tracker: The tracker.
        """
        if config and config.kf:
            return trackers.KF(sensors, config)
        else:
            return trackers.KF(sensors)

    @classmethod
    def construct_sensors(cls, config: Optional[ssensors.Config] = None) -> list:
        """Builds a list of sensors from the configuration

        Args:
            config (Optional[Config]): Configuration of ship sensors

        Returns:
            List[Sensor]: List of sensors.
        """
        if config:
            sensors = []
            for sensor_config in config.sensor_list:
                if isinstance(sensor_config, ssensors.RadarParams):
                    sensors.append(ssensors.Radar(sensor_config))
                elif isinstance(sensor_config, ssensors.AISParams):
                    sensors.append(ssensors.AIS(sensor_config))
        else:
            sensors = [ssensors.Radar()]

        return sensors

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
    """The Ship class is the main COLAV simulator object. It can be configured with various subsystems.

    Configurable subsystems include:
        - TODO: colav: The collision avoidance (colav) planner used. Can be used with or without the guidance system.
        - guidance: Guidance system, e.g. a LOS guidance.
        - controller: The low-level motion control strategy used, e.g. a PID controller.
        - tracker: The tracker used for estimating nearby dynamic obstacle states, e.g. a Kalman filter.
        - sensors: Sensor suite of the ship, e.g. a radar and AIS.
        - model: The ship model used for simulating its dynamics, e.g. a kinematic CSOG model.
    """

    def __init__(
        self,
        mmsi: int,
        identifier: int,
        pose: Optional[np.ndarray] = None,
        waypoints: Optional[np.ndarray] = None,
        speed_plan: Optional[np.ndarray] = None,
        config: Optional[Config] = None,
    ) -> None:

        self._mmsi = mmsi
        self._id = identifier
        self._ais_msg_nr: int = 18
        self._state: np.ndarray = np.zeros(4)
        self._waypoints: np.ndarray = np.empty(0)
        self._speed_plan: np.ndarray = np.zeros(0)
        self._trajectory: np.ndarray = np.empty(0)
        self._trajectory_sample: int = -1  # Index of current trajectory sample considered in the simulation (for AIS trajectories)
        self._first_valid_idx: int = -1  # Index of first valid AIS message in predefined trajectory
        self._last_valid_idx: int = -1  # Index of last valid AIS message in predefined trajectory
        self.t_start: float = 0.0  # The time when the ship appears in the simulation
        self.t_end: float = 1e12  # The time when the ship disappears from the simulation
        self._model, self._controller, self._guidance, self.sensors, self._tracker = ShipBuilder.construct_ship(config)

        if config:
            self._set_variables_from_config(config)

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

    def _set_variables_from_config(self, config: Config) -> None:
        self._id = config.id
        if config.pose is not None:
            self.set_initial_state(config.pose)

        if config.waypoints is not None and config.speed_plan is not None:
            self.set_nominal_plan(config.waypoints, config.speed_plan)

        if config.mmsi != -1:
            self._mmsi = config.mmsi

    def forward(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts the ship state dt seconds forward in time.

        If the ship is following a predefined trajectory,
        the state is updated to the (possibly interpolated) next state
        in the trajectory.

        Args:
            dt (float): Time step (s) in the prediction.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The new state dt seconds ahead,
            inputs to get there and the references used.
        """

        if self._trajectory.size > 0:
            if self._trajectory_sample < self._last_valid_idx - 1:
                self._trajectory_sample += 1

            self._state = np.array(
                [
                    self._trajectory[0, self._trajectory_sample],
                    self._trajectory[1, self._trajectory_sample],
                    self._trajectory[3, self._trajectory_sample],
                    self._trajectory[2, self._trajectory_sample],
                    0.0,
                    0.0,
                ]
            )
            return self._state, np.empty(3), np.empty(9)

        if self._waypoints.size < 2 or self._speed_plan.size < 2:
            raise ValueError("Insufficient waypoints for the ship to follow!")

        if dt <= 0.0:
            raise ValueError("Time step must be strictly positive!")

        references = self._guidance.compute_references(self._waypoints, self._speed_plan, None, self._state, dt)

        u = self._controller.compute_inputs(references, self._state, dt, self._model)

        self._state = self._state + dt * self._model.dynamics(self._state, u)

        return self._state, u, references

    def track_obstacles(self, t: float, dt: float, true_do_states: list) -> Tuple[list, list]:
        """Uses its target tracker to estimate the states of dynamic obstacles in the environment.

        Args:
            t (float): Current time (s).
            dt (float): Time step (s) in the simulation. I.e. difference between t and the previous time.
            true_do_states (list): List of true states of dynamic obstacles in the environment.

        Returns:
            Tuple[list, list]: Updated list of estimates and covariances on obstacles in the environment.
        """
        tracks = self._tracker.track(t, dt, true_do_states, mhm.convert_sog_cog_state_to_vxvy_state(self.pose))
        return tracks

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

    def get_ship_sim_data(self, t: float, timestamp_0: int) -> dict:
        """Returns simulation related data for the ship. Position are given in
        the local planar coordinate system.

        Args:
            t (float): Current time (s) relative to the start of the simulation.
            timestamp (int): UTC referenced timestamp at the start of the simulation.

        Returns:
            dict: Simulation data as dictionary
        """
        datetime_t = mhm.utc_timestamp_to_datetime(int(t) + timestamp_0)
        datetime_str = datetime_t.strftime("%d.%m.%Y %H:%M:%S")
        xs_i_upd, P_i_upd, NISes, labels = self.get_do_track_information()
        xs_i_upd = [xs.tolist() for xs in xs_i_upd]
        P_i_upd = [P.tolist() for P in P_i_upd]
        NISes = [float(NIS) for NIS in NISes]

        if self.t_start <= t < self.t_end:
            pose = self.pose
            active = True
        else:
            pose = np.ones(4) * np.nan
            active = False

        ship_sim_data = {
            "pose": pose,
            "waypoints": self._waypoints,
            "speed_plan": self._speed_plan,
            "date_time_utc": datetime_str,
            "timestamp": t,
            "do_estimates": xs_i_upd,
            "do_covariances": P_i_upd,
            "do_NISes": NISes,
            "do_labels": labels.copy(),
            "active": active,
            # predicted trajectory from COLAV/planner
        }

        return ship_sim_data

    def get_ais_data(self, t: float, timestamp_0: int, utm_zone: int) -> dict:
        """Returns a simulated AIS data message for the ship at the given timestamp.

        Args:
            t (float): Current time (s) relative to the start of the simulation.
            timestamp (int): UTC referenced timestamp at the start of the simulation.
            utm_zone (int): UTM zone used for the local planar coordinate system.

        Returns:
            dict: AIS message as a dictionary.
        """
        datetime_t = mhm.utc_timestamp_to_datetime(int(t) + timestamp_0)
        datetime_str = datetime_t.strftime("%d.%m.%Y %H:%M:%S")
        lat, lon = mapf.local2latlon(self.pose[1], self.pose[0], utm_zone)

        if self.t_start <= t < self.t_end:
            sog, cog, true_heading = (
                int(mf.mps2knots(self.pose[2])),
                int(np.rad2deg(self.pose[3])),
                np.rad2deg(self.heading),
            )
        else:
            lon, lat, sog, cog, true_heading = np.nan, np.nan, np.nan, np.nan, np.nan

        row = {
            "mmsi": self.mmsi,
            "lon": lon,
            "lat": lat,
            "date_time_utc": datetime_str,
            "sog": sog,
            "cog": cog,
            "true_heading": true_heading,
            "nav_status": 0,
            "message_nr": self.ais_msg_nr,
            "source": "",
        }
        return row

    def get_ship_info(self) -> dict:
        """Returns information about the ship contained in a dictionary."""
        output: dict = {}
        output["mmsi"] = self.mmsi
        output["length"] = self.length
        output["width"] = self.width
        output["draft"] = self.draft
        output["max_speed"] = self.max_speed
        output["min_speed"] = self.min_speed
        output["max_turn_rate"] = self.max_turn_rate
        return output

    def get_do_track_information(self) -> Tuple[list, list, list, list]:
        return self._tracker.get_track_information()

    def transfer_vessel_ais_data(
        self, vessel: colav_eval_vessel_data.VesselData, use_ais_trajectory: bool = True, t_start: Optional[float] = None, t_end: Optional[float] = None
    ) -> None:
        """Transfers vessel AIS data to a ship object. This includes the ship pose,
        length, width, draft etc..

        If the `use_ais_trajectory` flag is set, the Ship will follow its historical
        trajectory, and not the one provided by the onboard planner.

        If the t_start, t_end parameters are configured, the vessel data timestamps are
        not considered.

        Args:
            vessel (VesselData): AIS data of the ship.
            use_ais_trajectory (bool, optional): Use historical AIS trajectory or not.
        """

        self.set_initial_state(
            np.array(
                [
                    vessel.xy[1, vessel.first_valid_idx],
                    vessel.xy[0, vessel.first_valid_idx],
                    vessel.sog[vessel.first_valid_idx],
                    vessel.cog[vessel.first_valid_idx],
                ]
            )
        )

        self.t_start = vessel.timestamps[vessel.first_valid_idx]
        if t_start is not None:
            self.t_start = t_start

        if use_ais_trajectory:
            self._trajectory_sample = vessel.first_valid_idx
            self._trajectory = np.zeros((4, len(vessel.sog)))
            self._trajectory[0, :] = vessel.xy[1, :]
            self._trajectory[1, :] = vessel.xy[0, :]
            self._trajectory[2, :] = vessel.sog
            self._trajectory[3, :] = vessel.cog
            self.t_end = vessel.timestamps[vessel.last_valid_idx]

        if t_end is not None:
            self.t_end = t_end

        self._mmsi = vessel.mmsi
        self._first_valid_idx = vessel.first_valid_idx
        self._last_valid_idx = vessel.last_valid_idx
        self._model.params.length = vessel.length
        self._model.params.width = vessel.width
        self._model.params.draft = vessel.draft

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
        return self._model.params.U_max

    @property
    def min_speed(self) -> float:
        return self._model.params.U_min

    @property
    def max_turn_rate(self) -> float:
        return self._model.params.r_max

    @property
    def draft(self) -> float:
        return self._model.params.draft

    @property
    def length(self) -> float:
        return self._model.params.length

    @property
    def width(self) -> float:
        return self._model.params.width

    @property
    def mmsi(self) -> int:
        return self._mmsi

    @property
    def id(self) -> int:
        return self._id

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

    @property
    def trajectory(self) -> np.ndarray:
        if self._trajectory.size == 0:
            return np.array([])
        else:
            return self._trajectory[:, self._first_valid_idx : self._last_valid_idx + 1]
