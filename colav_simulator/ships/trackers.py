"""
    trackers.py

    Summary:
        Contains class definitions for dynamic obstacle
        target trackers. Every tracker must adhere to the
        ITracker interface.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import scipy.linalg as la


class ITracker(ABC):
    @abstractmethod
    def track(self, dt: float, true_do_states: list) -> None:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states."""


class KF(ITracker):
    """The KF class implements a linear Kalman filter based tracker."""

    sensors: list
    xs_p: list = []
    P_p: list = []
    xs_upd: list = []
    P_upd: list = []

    def __init__(self, prediction_model, sensors):
        self._pred_model = prediction_model
        self.sensors = sensors

    def track(self, dt: float, true_do_states: list) -> Tuple[list, list, list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states.

        NOTE: The KF assumes one generated measurement per dynamic obstacle per sensor.

        Args:
            dt (float): Time since last update
            true_do_states (list): List of true dynamic obstacle states. Used for simulating sensor measurements.

        """

        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(dt, true_do_states)
            sensor_measurements.append(z)

        n_do = len(sensor_measurements)

        for i in range(n_do):

            self.xs_p[i], self.P_p[i] = self.predict(self.xs_upd[i], self.P_upd[i], dt)

            for sensor_id in range(len(self.sensors)):
                self.xs_upd[i], self.P_upd[i] = self.update(
                    self.xs_p[i], self.P_p[i], sensor_measurements[i], sensor_id
                )

        return self.xs_upd, self.P_upd, self.xs_p, self.P_p

    def predict(self, xs_upd: np.ndarray, P_upd: np.ndarray, dt: float):
        F = self._pred_model.F(dt)
        Q = self._pred_model.Q(dt)

        x_pred = self._pred_model.f(xs_upd, dt)
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
        v, S = self.innovation(xs_p, P_p, z, sensor_id)
        H = self.sensors[sensor_id].H(xs_p)

        K = P_p @ H.T @ la.inv(S)
        x_upd = xs_p + K @ v
        P_upd = P_p - K @ H @ P_p

        return x_upd, P_upd
