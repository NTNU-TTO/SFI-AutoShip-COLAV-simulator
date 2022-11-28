"""
    sensors.py

    Summary:
        Contains class definitions for various sensors.
        Every seonsor must adhere to the ISensor interface.

    Author: Trym Tengesdal
"""
import math
import random

import colav_simulator.common.math_functions as mf
import numpy as np
from abc import ABC, abstractmethod

class ISensor(ABC):

    @abstractmethod
    def generate_measurements(self, dt: float, true_do_states: list) -> list:
        """Generates sensor measurements from the input true dynamic obstacle states."""

    @abstractmethod
    def R(self, xs: np.ndarray)
        """Returns the measurement noise covariance matrix for the input state."""

    @abstractmethod
    def H(self, xs: np.ndarray):
        """Returns the measurement matrix for the input state."""

    @abstractmethod
    def h(self, xs: np.ndarray):
        """Returns the measurement function for the input state."""

class Radar:
    """
    output:
        [x_meas, y_meas, SOG_meas, COG_meas].T
    """

    def __init__(self, meas_rate: float, sigma_z: float):
        self.meas_rate = meas_rate
        self.sigma_z = sigma_z
        self._R = self.sigma_z**2 * np.diag([10, 10, 1, np.deg2rad(0.1)])
        self._H = np.eye(4)

    def R(self, x: np.ndarray):
        return self._R

    def H(self):
        return self._H

    def h(self, x: np.ndarray):
        return self._H @ x

    def simulate_measurement(self, x_true: np.ndarray, t: float):
        if t % self.meas_rate:
            return None
        else:
            return simulate_measurement(x_true, self._R)


class AIS:
    def __init__(self, meas_rate: float, sigma_z: float, loss_prob: float):
        self.meas_rate = meas_rate
        self.sigma_z = sigma_z
        self.loss_prob = loss_prob
        self._R = self.sigma_z**2 * np.diag([10, 10, 1, np.deg2rad(0.1)])
        self._H = np.eye(4)

    """
     R_realistic values from paper considering quantization effects
     https://link.springer.com/chapter/10.1007/978-3-319-55372-6_13#Sec14
    def R_realistic(self, x: np.ndarray):
        R_GNSS = np.diag([0.5, 0.5, 0.1, 0.1])**2
        R_v = np.diag([x[2]**2, x[3]**2, 0, 0])
        self._R = R_GNSS + (1/12)*R_v
    """

    def R(self, x: np.ndarray):
        return self._R

    def H(self):
        return self._H

    def h(self, x: np.ndarray):
        z = self._H @ x
        return z

    def simulate_measurement(self, x_true: np.ndarray, t: float):
        if t % self.meas_rate or random.uniform(0, 1) < self.loss_prob:
            return None
        else:
            return simulate_measurement(x_true, self._R)


class Estimator:
    """
    Estimates the states of the other ships
    No Kalman Filter used
    x: [x, y, SOG, COG]
    """

    def __init__(self, sensor_list):
        # Takes in list of sensor objects, ex: sensor_list = [ais: AIS, radar: Radar]
        self.sensors = sensor_list

    def pred_step(self, x: np.ndarray, dt: float):
        """
        Simple straight line, constant speed prediction
        x: [x, y, psi, u].T
        """
        x[0] += x[3] * math.cos(x[2]) * dt
        x[1] += x[3] * math.sin(x[2]) * dt
        return x

    def upd_step(self, x: np.ndarray, z: np.ndarray):
        """
        Simple mean of pred and measurement(s), given no Kalman Filter implementation
        """
        x_upd = x * 0
        x_upd[0:3] = (x[0:3] + z[0:3]) * 0.5
        x_upd[3] = math.atan2(np.sin(x[3]) + np.sin(z[3]), np.cos(x[3]) + np.cos(z[3]))  # angle mean
        x_upd[3] = wrap_to_pi(x_upd[3])
        return x_upd

    def step(self, x_true: np.ndarray, x_est: np.ndarray, t: int, dt: int):
        """
        Estimates the state of another ship for the next timestep
        x_true: true state
        x_est: last timestep estimate of state

        x_true could possibly be replaced with z, removing the simulation of measurements from the estimator
        """
        x_prd = self.pred_step(x_est, dt)
        x_upd = x_prd
        for sensor in self.sensors:
            z = sensor.simulate_measurement(x_true, t)
            if z is not None:
                x_upd = self.upd_step(x_upd, z)
        return x_upd


"""
    Radar:
    rate; accuracy; noise probabilities, predictor

    AIS/VDES:
    rate; accuracy; loss probabilities; predictor

        rate:
        Class A	Anchored / Moored	 Every 3 Minutes
        Class A	Sailing 0-14 knots	 Every 10 Seconds
        Class A	Sailing 14-23 knots	 Every 6 Seconds
        Class A	Sailing 0-14 knots and changing course	 Every 3.33 Seconds
        Class A	Sailing 14-23 knots and changing course	 Every 2 Seconds
        Class A	Sailing faster than 23 knots	 Every 2 Seconds
        Class A	Sailing faster than 23 knots and changing course	 Every 2 Seconds
        Class B	Stopped or sailing up to 2 knots	 Every 3 Minutes
        Class B	Sailing faster than 2 knots	 Every 30 Seconds

    Camera?
"""


def simulate_measurement(x_true: np.ndarray, P: np.ndarray):
    """ Simulates a measurement from the true state x_true with covariance P.

    Args:
        x_true: [x_true, y_true, U_true, chi_true].T
        P: MVN covariance matrix

    Returns:
        np.ndarray: Measurement z = [x_meas, y_meas, U_meas, chi_meas].T
    """
    z = np.random.multivariate_normal(mean=x_true, cov=P)
    z[3] = mf.wrap_to_pi(z[3])

    return z
