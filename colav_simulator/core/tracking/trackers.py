"""
    trackers.py

    Summary:
        Contains class definitions for dynamic obstacle
        target trackers. Every tracker must adhere to the
        ITracker interface.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import numpy as np
import scipy.linalg as la


class ITracker(ABC):
    @abstractmethod
    def track(self, t: float, dt: float, true_do_states: list, ownship_state: np.ndarray) -> Tuple[list, list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states."""

    def get_track_information(self) -> Tuple[list, list, list, list]:
        """Returns the estimates and covariances of the tracked dynamic obstacles.
        Also, it returns the associated Normalized Innovation error Squared (NIS) values for
        the most recent update step for each track, and the track labels.

        Returns:
            Tuple[list, list, list]: List of estimates, list of covariances and list of NISes.
        """


@dataclass
class KFParams:
    """Class for holding KF parameters."""

    P_0: np.ndarray = np.diag([49.0, 49.0, 0.1, 0.1])
    q: float = 0.15

    def to_dict(self):
        output_dict = {"P_0": self.P_0.diagonal().tolist(), "q": self.q}
        return output_dict

    @classmethod
    def from_dict(cls, config_dict):
        return KFParams(P_0=np.diag(config_dict["P_0"]), q=config_dict["q"])


@dataclass
class Config:
    """Class for holding tracker configuration parameters."""

    kf: Optional[KFParams] = KFParams()

    def to_dict(self) -> dict:
        output_dict = {}
        if self.kf is not None:
            output_dict["kf"] = self.kf.to_dict()

        return output_dict

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(None)

        if "kf" in config_dict:
            config.kf = cp.convert_settings_dict_to_dataclass(KFParams, config_dict["kf"])

        return config


class TrackerBuilder:
    @classmethod
    def construct_tracker(cls, sensors: list, config: Optional[Config] = None) -> ITracker:
        """Builds a tracker from the configuration

        Args:
            sensors (list): Sensors used by the tracker.
            config (Optional[Config]): Tracker configuration. Defaults to None.

        Returns:
            ITracker: The tracker.
        """
        if config and config.kf:
            return KF(sensors, config)
        else:
            return KF(sensors)


class KF(ITracker):
    """The KF class implements a linear Kalman filter based tracker."""

    def __init__(self, sensor_list: list, config: Optional[Config] = None) -> None:
        if sensor_list is None:
            raise ValueError("Sensor list must be provided.")

        if config and config.kf is not None:
            self._params: KFParams = config.kf
        else:
            self._params = KFParams()

        self._model = CVModel(self._params.q)

        self.sensors: list = sensor_list

        self._track_initialized: list = []
        self._track_terminated: list = []
        self._labels: list = []
        self._xs_p: list = []
        self._P_p: list = []
        self._xs_upd: list = []
        self._P_upd: list = []
        self._NIS: list = []

    def track(self, t: float, dt: float, true_do_states: list, ownship_state: np.ndarray) -> Tuple[list, list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states.

        NOTE: The KF assumes one generated measurement per dynamic obstacle per sensor.

        Args:
            dt (float): Time since last update
            t (float): Current time (assumed >= 0)
            true_do_states (list): List of tuples of true dynamic obstacle indices and states (do_idx, [x, y, Vx, Vy]) x n_do. Used for simulating sensor measurements.
            ownship_state (np.ndarray): Ownship state vector [x, y, Vx, Vy] used for simulating sensor measurements.

        Returns:
            Tuple[list, list, list]: List of updated dynamic obstacle estimates and covariances. Also, a list the sensor measurements used.
        """
        relevant_do_states = []
        max_sensor_range = max([sensor.max_range for sensor in self.sensors])
        for do_idx, do_state in true_do_states:
            dist_ownship_to_do = np.linalg.norm(do_state[:2] - ownship_state[:2])
            if do_idx not in self._labels and dist_ownship_to_do < max_sensor_range:
                # New track. TODO: Implement track initiation, e.g. n out of m based initiation.
                self._labels.append(do_idx)
                self._track_initialized.append(False)
                self._track_terminated.append(False)
                self._xs_upd.append(do_state)
                self._P_upd.append(self._params.P_0)
                self._xs_p.append(do_state)
                self._P_p.append(self._params.P_0)
                self._NIS.append(np.nan)
            elif do_idx in self._labels:
                self._track_initialized[self._labels.index(do_idx)] = True
                relevant_do_states.append((do_idx, do_state))

        n_tracked_do = len(self._xs_upd)
        # TODO: Implement track termination for when covariance is too large.
        for i in range(n_tracked_do):
            if np.sqrt(self._P_upd[i][0, 0]) > 50.0 or np.sqrt(self._P_upd[i][1, 1]) > 50.0:
                self._track_terminated[i] = True

        # Only generate measurements for initialized tracks
        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(t, relevant_do_states, ownship_state)
            sensor_measurements.append(z)

        for i in range(n_tracked_do):
            if self._track_initialized[i] and not self._track_terminated[i]:
                self._xs_p[i], self._P_p[i] = self.predict(self._xs_upd[i], self._P_upd[i], dt)
                self._xs_upd[i] = self._xs_p[i]
                self._P_upd[i] = self._P_p[i]

                for sensor_id in range(len(self.sensors)):
                    z = sensor_measurements[sensor_id][i]
                    self._xs_upd[i], self._P_upd[i], NIS_i = self.update(self._xs_upd[i], self._P_upd[i], z, sensor_id)

                    if not np.isnan(NIS_i):
                        self._NIS[i] = NIS_i

        # print(f"xs_p: {self._xs_p}, xs_upd: {self._xs_upd}")
        # print(f"P_p: {self._P_p}")
        # print(f"P_upd: {self._P_upd}")
        return self._xs_upd, self._P_upd, sensor_measurements

    def predict(self, xs_upd: np.ndarray, P_upd: np.ndarray, dt: float):
        F = self._model.F(dt)
        Q = self._model.Q(dt)

        x_pred = self._model.f(xs_upd, dt)
        P_pred = F @ P_upd @ F.T + Q

        return x_pred, P_pred

    def innovation(self, xs_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, sensor_id: int):
        zbar = self.sensors[sensor_id].h(xs_p)
        v = z - zbar

        H = self.sensors[sensor_id].H(xs_p)
        R = self.sensors[sensor_id].R(xs_p)
        S = H @ P_p @ H.T + R

        return v, S

    def update(self, xs_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, sensor_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
        if any(np.isnan(z)):
            return xs_p, P_p, np.nan

        v, S = self.innovation(xs_p, P_p, z, sensor_id)
        H = self.sensors[sensor_id].H(xs_p)

        K = P_p @ H.T @ la.inv(S)
        x_upd = xs_p + K @ v
        P_upd = P_p - K @ H @ P_p

        return x_upd, P_upd, NIS(v, S)

    def get_track_information(self) -> Tuple[list, list, list, list]:
        return self._xs_upd, self._P_upd, self._NIS, self._labels


def NIS(v: np.ndarray, S: np.ndarray) -> float:
    return v.T @ la.inv(S) @ v


class CVModel:
    """The CVModel class implements a constant velocity model."""

    def __init__(self, q: float) -> None:
        self._q: float = q

    def f(self, xs: np.ndarray, dt: float) -> np.ndarray:
        """Returns the r.h.s of the prediction model state transition function.

        Args:
            xs (np.ndarray): State vector [x, y, Vx, Vy]
            dt (float): Time step

        Returns:
            np.ndarray: New state vector dt seconds ahead

        """
        return xs + np.array([xs[2] * dt, xs[3] * dt, 0.0, 0.0])

    def F(self, dt: float) -> np.ndarray:
        """Returns the Jacobian of the prediction model state transition function.

        Args:
            xs (np.ndarray): xs (np.ndarray): State vector [x, y, Vx, Vy]
            dt (float): Time step

        Returns:
            np.ndarray: Jacobian of the prediction model state transition function
        """
        return np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def Q(self, dt: float) -> np.ndarray:
        """Returns the process noise covariance matrix.

        Args:
            dt (float): Time step

        Returns:
            np.ndarray: Process noise covariance matrix
        """
        return (
            np.array(
                [
                    [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
                    [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
                    [dt**2 / 2.0, 0.0, dt, 0.0],
                    [0.0, dt**2 / 2.0, 0.0, dt],
                ]
            )
            * self._q
        )
