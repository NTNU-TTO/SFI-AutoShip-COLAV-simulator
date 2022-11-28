"""
    sensors.py

    Summary:
        Contains class definitions for various sensors.
        Every sensor must adhere to the ISensor interface.

    Author: Trym Tengesdal
"""
import math
import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import colav_simulator.common.math_functions as mf
import numpy as np


class ISensor(ABC):
    @abstractmethod
    def generate_measurements(self, t: float, true_do_states: list) -> Optional[list]:
        """Generates sensor measurements from the input true dynamic obstacle states."""

    @abstractmethod
    def R(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement noise covariance matrix for the input state."""

    @abstractmethod
    def H(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement matrix for the input state."""

    @abstractmethod
    def h(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement function for the input state."""


class RadarPars:
    """Configuration parameters for a radar sensor."""

    measurement_rate: float = 0.25
    R: np.ndarray = np.diag([5.0**2, 5.0**2])


class AISPars:
    """AIS parameter class."""

    measurement_rate: float = 0.25


class Config:
    """Class for holding sensor configuration parameters."""

    radar: Optional[RadarPars] = RadarPars()
    ais: Optional[AISPars] = AISPars()


class Radar(ISensor):
    """Implements functionality for a radar sensor."""

    _pars: RadarPars

    def __init__(self, config: Optional[Config] = None) -> None:
        if config and config.radar is not None:
            self._pars = config.radar
        else:
            self._pars = RadarPars()

        self._H = np.array([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])

    def R(self, xs: np.ndarray) -> np.ndarray:
        return self._pars.R

    def H(self, xs: np.ndarray) -> np.ndarray:
        return self._H

    def h(self, xs: np.ndarray) -> np.ndarray:
        return self._H @ xs

    def generate_measurements(self, t: float, true_do_states: list) -> Optional[list]:
        if t % self._pars.measurement_rate:
            return None
        else:
            measurements = []
            for state in true_do_states:
                z = self.h(state) + np.random.multivariate_normal(np.zeros(4), self.R(state))
                measurements.append(z)
        return measurements


class AIS:

    _pars: AISPars

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
