"""
    sensors.py

    Summary:
        Contains class definitions for various sensors.
        Every sensor must adhere to the ISensor interface.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import colav_simulator.common.config_parsing as cp
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
        """Generates sensor measurements from the input tuple list of true dynamic obstacle indices and states, (do_idx, do_state) x n_do."""


@dataclass
class RadarParams:
    """Configuration parameters for a radar sensor."""

    measurement_rate: float = 0.8
    R: np.ndarray = np.diag([5.0**2, 5.0**2])

    @classmethod
    def from_dict(self, config_dict: dict):
        return RadarParams(measurement_rate=config_dict["measurement_rate"], R=np.diag(config_dict["R"]))

    def to_dict(self) -> dict:
        output_dict = asdict(self)
        output_dict["R"] = self.R.diagonal().tolist()
        return output_dict


class AISClass(Enum):
    """AIS class A and B transponder types."""

    A = 0
    B = 1


@dataclass
class AISParams:
    """AIS parameter class."""

    ais_class: AISClass = AISClass.A
    R: np.ndarray = np.diag([5.0**2, 5.0**2, 0.1**2, 0.08**2])  # meas cov for a state vector of [x, y, Vx, Vy]

    @classmethod
    def from_dict(cls, config_dict: dict):
        return AISParams(ais_class=AISClass(config_dict["ais_class"]), R=np.diag(config_dict["R"]))

    def to_dict(self) -> dict:
        output_dict = {"ais_class": self.ais_class.name, "R": self.R.diagonal().tolist()}
        return output_dict


@dataclass
class Config:
    """Class for holding sensor(s) configuration parameters."""

    sensor_list: list = field(default_factory=lambda: [RadarParams()])

    def to_dict_list(self) -> list:
        output_list: list = []
        for sensor in self.sensor_list:
            sensor_dict = {}
            if isinstance(sensor, RadarParams):
                sensor_dict["radar"] = sensor.to_dict()
            elif isinstance(sensor, AISParams):
                sensor_dict["ais"] = sensor.to_dict()
            output_list.append(sensor_dict)

        return output_list

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(sensor_list=[])
        for sensor_dict in config_dict:
            if "radar" in sensor_dict:
                config.sensor_list.append(cp.convert_settings_dict_to_dataclass(RadarParams, sensor_dict["radar"]))
            elif "ais" in sensor_dict:
                config.sensor_list.append(cp.convert_settings_dict_to_dataclass(AISParams, sensor_dict["ais"]))

        return config


class Radar(ISensor):
    """Implements functionality for a radar sensor."""

    # TODO: Implement clutter measurements and detection probability
    type: str = "radar"
    _H: np.ndarray = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def __init__(self, params: RadarParams = RadarParams()) -> None:
        self._params: RadarParams = params

    def R(self, xs: np.ndarray) -> np.ndarray:
        return self._params.R

    def H(self, xs: np.ndarray) -> np.ndarray:
        return self._H

    def h(self, xs: np.ndarray) -> np.ndarray:
        return self._H @ xs

    def generate_measurements(self, t: float, true_do_states: list) -> Optional[list]:
        measurements = []
        for _, xs in true_do_states:
            if t % 1.0 / self._params.measurement_rate < 0.001:
                z = self.h(xs) + np.random.multivariate_normal(np.zeros(2), self.R(xs))
            else:
                z = np.nan * np.ones(2)
            measurements.append(z)
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

    type: str = "ais"
    _H: np.ndarray = np.eye(4)

    def __init__(self, params: AISParams = AISParams()) -> None:
        self._params: AISParams = params
        self._previous_meas_time = 0.0

    def R(self, xs: np.ndarray) -> np.ndarray:
        return self._params.R

    def H(self, xs: np.ndarray) -> np.ndarray:
        return self._H

    def h(self, xs: np.ndarray) -> np.ndarray:
        z = self._H @ xs
        return z

    def generate_measurements(self, t: float, true_do_states: list) -> list:
        measurements = []
        for _, xs in true_do_states:
            if t % 1.0 / self.measurement_rate(xs) < 0.001:
                z = self.h(xs) + np.random.multivariate_normal(np.zeros(4), self.R(xs))
            else:
                z = np.nan * np.ones(4)
            measurements.append(z)

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
        sog = mf.ms2knots(float(np.linalg.norm(xs[2:3])))
        rate = 1.0
        if self._params.ais_class == AISClass.A:
            if sog <= 0.001:
                rate = 1.0 / 180.0
            elif sog > 0.001 and sog <= 14.0:
                rate = 1.0 / 10.0
            elif sog > 14.0 and sog <= 23.0:
                rate = 1.0 / 6.0
            elif sog > 23.0:
                rate = 1.0 / 2.0
        elif self._params.ais_class == AISClass.B:
            if sog <= 2.0:
                rate = 1.0 / 180.0
            elif sog > 2.0:
                rate = 1.0 / 30.0
        return rate
