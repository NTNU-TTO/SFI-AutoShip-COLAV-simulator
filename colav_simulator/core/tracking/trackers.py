"""
    trackers.py

    Summary:
        Contains class definitions for dynamic obstacle
        target trackers. Every tracker must adhere to the
        ITracker interface.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import numpy as np
import scipy.linalg as la


class ITracker(ABC):
    @abstractmethod
    def track(self, t: float, dt: float, true_do_states: list) -> Tuple[list, list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states."""

    def get_tracks(self) -> Tuple[list, list]:
        """Returns the current tracks (state estimates and covariances)."""


@dataclass
class KFParams:
    """Class for holding KF parameters."""

    P_0: np.ndarray = np.diag([20.0, 20.0, 0.1, 0.1])
    q: float = 0.05

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


class KF(ITracker):
    """The KF class implements a linear Kalman filter based tracker."""

    _params: KFParams
    sensors: list
    xs_p: list = []
    P_p: list = []
    xs_upd: list = []
    P_upd: list = []

    def __init__(self, sensor_list: list, config: Optional[Config] = None) -> None:
        if sensor_list is None:
            raise ValueError("Sensor list must be provided.")

        if config and config.kf is not None:
            self._params = config.kf
        else:
            self._params = KFParams()

        self._model = CVModel(self._params.q)

        self.sensors = sensor_list

    def track(self, t: float, dt: float, true_do_states: list) -> Tuple[list, list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states.

        NOTE: The KF assumes one generated measurement per dynamic obstacle per sensor.

        Args:
            dt (float): Time since last update
            t (float): Current time (assumed >= 0)
            true_do_states (list): List of true dynamic obstacle states [x, y, Vx, Vy] x n_do. Used for simulating sensor measurements.

        Returns:
            Tuple[list, list, list]: List of updated dynamic obstacle estimates and covariances. Also, a list of sensor measurements.
        """

        if t < 0.00001:
            self.xs_upd = true_do_states
            self.P_upd = [self._params.P_0 for _ in range(len(true_do_states))]
            self.xs_p = self.xs_upd.copy()
            self.P_p = self.P_upd.copy()
            return self.xs_upd, self.P_upd, []

        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(t, true_do_states)
            sensor_measurements.append(z)

        n_do = len(true_do_states)

        # TODO: Implement track initiation, e.g. n out of m based initiation.

        for i in range(n_do):

            self.xs_p[i], self.P_p[i] = self.predict(self.xs_upd[i], self.P_upd[i], dt)

            for sensor_id in range(len(self.sensors)):
                self.xs_upd[i], self.P_upd[i] = self.update(
                    self.xs_p[i], self.P_p[i], sensor_measurements[sensor_id][i], sensor_id
                )

        return self.xs_upd, self.P_upd, sensor_measurements

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

    def update(self, xs_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, sensor_id: int):
        if any(np.isnan(z)):
            return xs_p, P_p

        v, S = self.innovation(xs_p, P_p, z, sensor_id)
        H = self.sensors[sensor_id].H(xs_p)

        K = P_p @ H.T @ la.inv(S)
        x_upd = xs_p + K @ v
        P_upd = P_p - K @ H @ P_p

        return x_upd, P_upd

    def get_tracks(self) -> Tuple[list, list]:
        return self.xs_upd, self.P_upd


class CVModel:
    """The CVModel class implements a constant velocity model."""

    _q: float

    def __init__(self, q: float) -> None:
        self._q = q

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
