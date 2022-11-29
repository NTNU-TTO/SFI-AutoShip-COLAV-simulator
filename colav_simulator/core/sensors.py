"""
    sensors.py

    Summary:
        Contains class definitions for various sensors.
        Every sensor must adhere to the ISensor interface.

    Author: Trym Tengesdal
"""
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Tuple

import colav_simulator.common.math_functions as mf
import numpy as np


class ISensor(ABC):
    @abstractmethod
    def R(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement noise covariance matrix for the input state."""

    @abstractmethod
    def H(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement matrix for the input state."""

    @abstractmethod
    def h(self, xs: np.ndarray) -> np.ndarray:
        """Returns the measurement function for the input state."""

    @abstractmethod
    def generate_measurements(self, t: float, true_do_states: list) -> Optional[list]:
        """Generates sensor measurements from the input true dynamic obstacle states."""


@dataclass
class RadarPars:
    """Configuration parameters for a radar sensor."""

    measurement_rate: float = 0.4
    R: np.ndarray = np.diag([5.0**2, 5.0**2])

    @classmethod
    def from_dict(self, config_dict: dict):
        return RadarPars(measurement_rate=config_dict["measurement_rate"], R=np.diag(config_dict["R"]))

    def to_dict(self):
        output_dict = asdict(self)
        output_dict["R"] = self.R.tolist()


class AISClass(Enum):
    """AIS class A and B transponder types."""

    A = 0
    B = 1


@dataclass
class AISPars:
    """AIS parameter class."""

    ais_class: AISClass = AISClass.A
    R: np.ndarray = np.diag([5.0**2, 5.0**2, 0.1**2, 0.1**2])  # meas cov for a state vector of [x, y, Vx, Vy]

    @classmethod
    def from_dict(cls, config_dict: dict):
        return AISPars(ais_class=AISClass(config_dict["ais_class"]), R=np.diag(config_dict["R"]))

    def to_dict(self):
        output_dict = asdict(self)
        output_dict["R"] = self.R.tolist()


@dataclass
class Config:
    """Class for holding sensor(s) configuration parameters."""

    sensor_list: list = field(default_factory=[RadarPars()])

    def to_dict(self) -> dict:
        config_dict: dict = {}
        config_dict["sensor_list"] = []
        for sensor in self.sensor_list:
            sensor_dict = sensor.to_dict()
            config_dict["sensor_list"].append(sensor_dict)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(sensor_list=[])
        for sensor_name, sensor_pars in config_dict["sensor_list"]:
            if sensor_name == "radar":
                config.sensor_list.append(RadarPars(**sensor_pars))


class Radar(ISensor):
    """Implements functionality for a radar sensor."""

    _pars: RadarPars
    _H: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])

    def __init__(self, pars: RadarPars = RadarPars()) -> None:
        self._pars = pars

    def R(self, xs: np.ndarray) -> np.ndarray:
        return self._pars.R

    def H(self, xs: np.ndarray) -> np.ndarray:
        return self._H

    def h(self, xs: np.ndarray) -> np.ndarray:
        return self._H @ xs

    def generate_measurements(self, t: float, true_do_states: list) -> Optional[list]:
        measurements = []
        for xs in true_do_states:
            if t % self._pars.measurement_rate < 0.001:
                z = self.h(xs) + np.random.multivariate_normal(np.zeros(2), self.R(xs))
                measurements.append(z)
            else:
                z = -1e6 * np.ones(2)
        return measurements


class AIS:
    """Class for simulating AIS transponder measurements.

    AIS/VDES:
        Measurement rate depends on:
        Class A	Anchored / Moored	 Every 3 Minutes
        Class A	Sailing 0-14 knots	 Every 10 Seconds
        Class A	Sailing 14-23 knots	 Every 6 Seconds
        Class A	Sailing 0-14 knots and changing course	 Every 3.33 Seconds
        Class A	Sailing 14-23 knots and changing course	 Every 2 Seconds
        Class A	Sailing faster than 23 knots	 Every 2 Seconds
        Class A	Sailing faster than 23 knots and changing course	 Every 2 Seconds
        Class B	Stopped or sailing up to 2 knots	 Every 3 Minutes
        Class B	Sailing faster than 2 knots	 Every 30 Seconds

     R_realistic values from paper considering quantization effects
     https://link.springer.com/chapter/10.1007/978-3-319-55372-6_13#Sec14
    def R_realistic(self, x: np.ndarray):
        R_GNSS = np.diag([0.5, 0.5, 0.1, 0.1])**2
        R_v = np.diag([x[2]**2, x[3]**2, 0, 0])
        self._R = R_GNSS + (1/12)*R_v
    """

    _pars: AISPars
    _previous_meas_time: float = 0.0
    _H: np.ndarray

    def __init__(self, pars: AISPars = AISPars()) -> None:
        self._pars = pars
        self._H = np.eye(4)

    def R(self, xs: np.ndarray) -> np.ndarray:
        return self._pars.R

    def H(self, xs: np.ndarray) -> np.ndarray:
        return self._H

    def h(self, xs: np.ndarray) -> np.ndarray:
        z = self._H @ xs
        return z

    def generate_measurements(self, t: float, true_do_states: list) -> list:
        """Generates AIS measurements from the input true dynamic obstacle states.

        Args:
            t (float): Current time.
            true_do_states (list): List of true dynamic obstacle states.

        Returns:
            list: List of generated AIS measurements.
        """
        measurements = []
        for state in true_do_states:
            if t % self.measurement_rate(state) < 0.001:
                z = self.h(state) + np.random.multivariate_normal(np.zeros(4), self.R(state))
                measurements.append(z)
            else:
                z = -1e6 * np.ones(4)
        return measurements

    def measurement_rate(self, xs: np.ndarray) -> float:
        """Returns the measurement rate for the input state. This depends on
        the input state's speed and AIS class (and also if the course is changing,
        but this is not considered here (yet)).

        Args:
            xs (np.ndarray): The state vector of the dynamic obstacle = [x, y, Vx, Vy]

        Returns:
            float: The measurement rate in Hz.
        """
        sog = mf.ms2knots(float(np.linalg.norm(xs[2:4])))
        rate = 1.0
        if self._pars.ais_class == AISClass.A:
            if sog <= 0.001:
                rate = 1.0 / 180.0
            elif sog > 0.001 and sog <= 14.0:
                rate = 1.0 / 10.0
            elif sog > 14.0 and sog <= 23.0:
                rate = 1.0 / 6.0
            elif sog > 23.0:
                rate = 1.0 / 2.0
        elif self._pars.ais_class == AISClass.B:
            if sog <= 2.0:
                rate = 1.0 / 180.0
            elif sog > 2.0:
                rate = 1.0 / 30.0
        return rate
