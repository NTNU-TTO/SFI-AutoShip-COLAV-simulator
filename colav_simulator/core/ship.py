"""
    ship.py

    Summary:
        Contains class definitions for ship classes.
        Every ship class must adhere to the interface
        IShip and must be built by a ShipBuilder.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import colav_evaluation_tool.vessel as colav_eval_vessel_data
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.sensing as sensing
import colav_simulator.core.tracking.trackers as trackers
import numpy as np
import seacharts.enc as senc
from colav_simulator.core.integrators import erk4_integration_step


@dataclass
class Config:
    """Configuration class for managing ship parameters."""

    colav: Optional[ci.Config] = None
    guidance: Optional[guidances.Config] = guidances.Config()
    model: models.Config = models.Config()
    controller: controllers.Config = controllers.Config()
    sensors: sensing.Config = sensing.Config()
    tracker: trackers.Config = trackers.Config()
    mmsi: int = -1  # MMSI number of the ship, if configured equal to the MMSI of a ship in AIS data, the ship will be initialized with the data from the AIS data.
    id: int = -1  # Ship identifier
    t_start: Optional[float] = None  # Determines when the ship should start in the simulation
    t_end: Optional[float] = None  # Determines when the ship ends its part in the simulation
    random_generated: Optional[bool] = False  # True if the ship should have randomly generated COG-SOG-state, wps and speed plan. Takes priority over mmsi.
    csog_state: Optional[np.ndarray] = None  # In format [x[north], y[east], SOG [m/s], COG[deg]], similar to AIS data.
    goal_csog_state: Optional[np.ndarray] = None  # In format [x[north], y[east], SOG [m/s], COG[deg]], similar to AIS data.
    waypoints: Optional[np.ndarray] = None
    speed_plan: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "csog_state" in config_dict:
            config.csog_state = np.array(config_dict["csog_state"])
            config.csog_state[3] = np.deg2rad(config.csog_state[3])

        if "goal_csog_state" in config_dict:
            config.goal_csog_state = np.array(config_dict["goal_csog_state"])
            config.goal_csog_state[3] = np.deg2rad(config.goal_csog_state[3])

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

        config.sensors = sensing.Config.from_dict(config_dict["sensors"])

        config.tracker = trackers.Config.from_dict(config_dict["tracker"])

        if "guidance" in config_dict:
            config.guidance = guidances.Config.from_dict(config_dict["guidance"])

        if "colav" in config_dict:
            config.colav = ci.Config.from_dict(config_dict["colav"])

        # COLAV take priority over guidance, if both are specified.
        if config.colav and config.guidance:
            config.guidance = None

        assert config.colav is not None or config.guidance is not None, "Ship must have either a guidance or a colav system."

        return config

    def to_dict(self):
        config_dict = {}

        if self.csog_state is not None:
            config_dict["csog_state"] = self.csog_state.tolist()
            config_dict["csog_state"][3] = float(np.rad2deg(self.csog_state[3]))

        if self.goal_csog_state is not None:
            config_dict["goal_csog_state"] = self.goal_csog_state.tolist()
            config_dict["goal_csog_state"][3] = float(np.rad2deg(self.goal_csog_state[3]))

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

        config_dict["sensors"] = self.sensors.to_dict_list()

        config_dict["tracker"] = self.tracker.to_dict()

        if self.guidance is not None:
            config_dict["guidance"] = self.guidance.to_dict()

        if self.colav is not None:
            config_dict["colav"] = self.colav.to_dict()

        return config_dict


class ShipBuilder:
    """Class for building all objects needed by a ship with a specific configuration of systems."""

    @classmethod
    def construct_ship(cls, config: Optional[Config] = None):
        """Builds a ship from the configuration

        Args:
            config (Optional[Config]): Ship configuration. Defaults to None.

        Returns:
            Tuple[IModel, IController, IGuidance, list, ITracker, ICOLAV]: The subsystems comprising the ship: model, controller, guidance.
        """
        if config:
            model = cls.construct_model(config.model)
            controller = cls.construct_controller(config.controller)
            sensors = cls.construct_sensors(config.sensors)
            tracker = cls.construct_tracker(sensors, config.tracker)
            guidance_alg = None
            colav_alg = None
            if config.guidance is not None:
                guidance_alg = cls.construct_guidance(config.guidance)
            else:
                colav_alg = cls.construct_colav(config.colav)
        else:
            model = cls.construct_model()
            controller = cls.construct_controller()
            sensors = cls.construct_sensors()
            tracker = cls.construct_tracker(sensors)
            guidance_alg = cls.construct_guidance()
            colav_alg = None

        return model, controller, guidance_alg, sensors, tracker, colav_alg

    @classmethod
    def construct_colav(cls, config: Optional[ci.Config] = None) -> ci.ICOLAV:
        return ci.COLAVBuilder.construct_colav(config)

    @classmethod
    def construct_tracker(cls, sensors: list, config: Optional[trackers.Config] = None) -> trackers.ITracker:
        return trackers.TrackerBuilder.construct_tracker(sensors, config)

    @classmethod
    def construct_sensors(cls, config: Optional[sensing.Config] = None) -> list:
        return sensing.SensorSuiteBuilder.construct_sensors(config)

    @classmethod
    def construct_guidance(cls, config: Optional[guidances.Config] = None) -> guidances.IGuidance:
        return guidances.GuidanceBuilder.construct_guidance(config)

    @classmethod
    def construct_controller(cls, config: Optional[controllers.Config] = None) -> controllers.IController:
        return controllers.ControllerBuilder.construct_controller(config)

    @classmethod
    def construct_model(cls, config: Optional[models.Config] = None) -> models.IModel:
        return models.ModelBuilder.construct_model(config)


class IShip(ABC):
    @abstractmethod
    def forward(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Predict the ship dt seconds forward in time. Returns new state, inputs to get there and references used."

    @abstractmethod
    def track_obstacles(self, t: float, dt: float, true_do_states: list) -> Tuple[list, list]:
        "Track obstacles using the sensor suite, taking the obstacle states as inputs at the current time."

    @abstractmethod
    def plan(
        self,
        t: float,
        dt: float,
        do_list: list,
        enc: Optional[senc.ENC] = None,
    ) -> np.ndarray:
        "Plan a new trajectory for the ship, either using the onboard guidance system or COLAV system employed."


class Ship(IShip):
    """The Ship class is the main COLAV simulator object. It can be configured with various subsystems.

    Configurable subsystems include:
        - colav: The collision avoidance (colav) planner used.
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
        csog_state: Optional[np.ndarray] = None,
        goal_csog_state: Optional[np.ndarray] = None,
        waypoints: Optional[np.ndarray] = None,
        speed_plan: Optional[np.ndarray] = None,
        config: Optional[Config] = None,
        model: Optional[models.IModel] = None,
        controller: Optional[controllers.IController] = None,
        guidance: Optional[guidances.IGuidance] = None,
        sensors: Optional[list] = None,
        tracker: Optional[trackers.ITracker] = None,
        colav: Optional[ci.ICOLAV] = None,
    ) -> None:

        self._mmsi = mmsi
        self._id = identifier
        self._ais_msg_nr: int = 18
        self._state: np.ndarray = np.zeros(6)
        self._goal_state: np.ndarray = np.zeros(6)
        self._waypoints: np.ndarray = np.empty(0)
        self._speed_plan: np.ndarray = np.zeros(0)
        self._references: np.ndarray = np.zeros(0)
        self._trajectory: np.ndarray = np.empty(0)
        self._trajectory_sample: int = -1  # Index of current trajectory sample considered in the simulation (for AIS trajectories)
        self._first_valid_idx: int = -1  # Index of first valid AIS message in predefined trajectory
        self._last_valid_idx: int = -1  # Index of last valid AIS message in predefined trajectory
        self.t_start: float = 0.0  # The time when the ship appears in the simulation
        self.t_end: float = 1e12  # The time when the ship disappears from the simulation
        self._model, self._controller, self._guidance, self.sensors, self._tracker, self._colav = ShipBuilder.construct_ship(config)

        if model is not None:
            self._model = model

        if controller is not None:
            self._controller = controller

        if guidance is not None:
            self._guidance = guidance
            self._colav = None

        if sensors is not None:
            self._sensors = sensors

        if tracker is not None:
            self._tracker = tracker

        if colav is not None:
            self._colav = colav
            self._guidance = None

        if config:
            self._set_variables_from_config(config)

        # Input COG-SOG state, waypoints and speed plans take precedence over config specified ones
        if csog_state is not None:
            self.set_initial_state(csog_state)

        if goal_csog_state is not None:
            self._goal_csog_state = goal_csog_state

        if waypoints is not None and speed_plan is not None:
            self.set_nominal_plan(waypoints, speed_plan)

    def _set_variables_from_config(self, config: Config) -> None:
        self._id = config.id
        if config.csog_state is not None:
            self.set_initial_state(config.csog_state)

        if config.goal_csog_state is not None:
            self.set_goal_state(config.goal_csog_state)

        if config.waypoints is not None and config.speed_plan is not None:
            self.set_nominal_plan(config.waypoints, config.speed_plan)

        if config.t_start is not None:
            self.t_start = config.t_start

        if config.t_end is not None:
            self.t_end = config.t_end

        if config.mmsi != -1:
            self._mmsi = config.mmsi

    def plan(
        self,
        t: float,
        dt: float,
        do_list: list,
        enc: Optional[senc.ENC] = None,
    ) -> np.ndarray:

        if self._goal_state is None and (self._waypoints.size < 2 or self._speed_plan.size < 2):
            raise ValueError("Either the goal pose must be provided, or a sufficient number of waypoints for the ship to follow!")

        if self._colav is not None:
            self._references = self._colav.plan(
                t,
                self._waypoints,
                self._speed_plan,
                self._state,
                do_list,
                enc,
                self._goal_state,
                os_length=self._model.params.length,
                os_width=self._model.params.width,
                os_draft=self._model.params.draft,
            )
            return self._references

        self._references = self._guidance.compute_references(self._waypoints, self._speed_plan, None, self._state, dt)
        return self._references

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

        if dt <= 0.0:
            raise ValueError("Time step must be strictly positive!")

        u = self._controller.compute_inputs(self._references[:, 0], self._state, dt, self._model)

        self._state = erk4_integration_step(self._model.dynamics, self._state, u, dt)
        return self._state, u, self._references[:, 0]

    def track_obstacles(self, t: float, dt: float, true_do_states: list) -> Tuple[list, list]:
        """Tracks obstacles in the vicinity of the ship."""
        tracks = self._tracker.track(t, dt, true_do_states, mhm.convert_csog_state_to_vxvy_state(self.csog_state))
        return tracks

    def set_initial_state(self, csog_state: np.ndarray) -> None:
        """Sets the initial state of the ship based on the input kinematic state.

        Args:
            csog_state (np.ndarray): Initial COG-SOG state = [x, y, U, chi] of the ship.
        """
        self._state = np.array([csog_state[0], csog_state[1], csog_state[3], csog_state[2], 0.0, 0.0])

    def set_goal_state(self, csog_state: np.ndarray) -> None:
        """Sets the goal state of the ship based on the input kinematic state.

        Args:
            csog_state (np.ndarray): Initial COG-SOG state = [x, y, U, chi] of the ship.
        """
        self._goal_state = np.array([csog_state[0], csog_state[1], csog_state[3], csog_state[2], 0.0, 0.0])

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

    def set_colav_system(self, colav: Any | ci.ICOLAV) -> None:
        """Sets the COLAV system to be used by the ship.

        Args:
            colav (Any | ICOLAV): COLAV system, must implement the ICOLAV interface.
        """
        self._colav = colav

        if self._guidance is not None:
            self._guidance = None

    def get_sim_data(self, t: float, timestamp_0: int) -> dict:
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
        tracks, NISes = self.get_do_track_information()
        labels = [track[0] for track in tracks]
        xs_i_upd = [track[1].tolist() for track in tracks]
        P_i_upd = [track[2].tolist() for track in tracks]
        NISes = [float(NIS) for NIS in NISes]

        if self.t_start <= t < self.t_end:
            csog_state = self.csog_state
            active = True
        else:
            csog_state = np.ones(4) * np.nan
            active = False

        ship_sim_data = {
            "id": self.id,
            "mmsi": self.mmsi,
            "csog_state": csog_state,
            "waypoints": self._waypoints,
            "speed_plan": self._speed_plan,
            "date_time_utc": datetime_str,
            "timestamp": t,
            "do_estimates": xs_i_upd,
            "do_covariances": P_i_upd,
            "do_NISes": NISes,
            "do_labels": labels,
            "active": active,
            # predicted trajectory from COLAV/planner
        }

        return ship_sim_data

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

    def get_do_track_information(self) -> Tuple[list, list]:
        return self._tracker.get_track_information()

    def transfer_vessel_ais_data(
        self, vessel: colav_eval_vessel_data.VesselData, use_ais_trajectory: bool = True, t_start: Optional[float] = None, t_end: Optional[float] = None
    ) -> None:
        """Transfers vessel AIS data to a ship object. This includes the ship COG-SOG-state,
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
    def csog_state(self) -> np.ndarray:
        """Returns the ship COG/SOG state as parameterized in an AIS message,
        i.e. `xs = [x, y, U, chi]` where `x` and `y` are planar coordinates (north-east),
        `U` the ship speed over ground (m/s) and `chi` the course over ground (rad).

        Returns:
            np.ndarray: Ship csog-state.
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
