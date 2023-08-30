"""
    stochasticity.py

    Summary:
        Contains class definitions for various stochastic disturbance models,
        from which wind, wave, current++ factors can be superposed on ship objects.
        Every disturbance class must adhere to the interface IDisturbance.

    Author: Trym Tengesdal
"""
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import numpy as np


@dataclass
class GaussMarkovDisturbanceParams:
    """Parameters for a gauss-markov disturbance model with random speed and direction (of e.g. wind or current).

    Initial speed and direction are optional, and if not configured, they are drawn from the specified ranges.
    """

    constant: bool = True
    initial_speed: Optional[float] = 0.5
    initial_direction: Optional[float] = 0.0
    speed_range: Tuple[float, float] = (0.0, 3.0)
    direction_range: Tuple[float, float] = (-np.pi, np.pi)
    mu_speed: float = 1e-5
    mu_direction: float = 1e-05
    sigma_speed: float = 0.005
    sigma_direction: float = 0.005

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = GaussMarkovDisturbanceParams()
        if "initial_speed" in config_dict:
            params.initial_speed = config_dict["initial_speed"]
        else:
            params.initial_speed = random.uniform(*config_dict["speed_range"])

        if "initial_direction" in config_dict:
            params.initial_direction = np.deg2rad(config_dict["initial_direction"])
        else:
            params.initial_direction = random.uniform(*config_dict["direction_range"])

        params.speed_range = tuple(config_dict["speed_range"])
        params.direction_range = tuple(np.deg2rad(config_dict["direction_range"]))
        params.mu_speed = config_dict["mu_speed"]
        params.mu_direction = config_dict["mu_direction"]
        params.sigma_speed = config_dict["sigma_speed"]
        params.sigma_direction = config_dict["sigma_direction"]
        params.constant = config_dict["constant"]
        return params

    def to_dict(self) -> dict:
        config_dict = {
            "constant": self.constant,
            "initial_speed": self.initial_speed,
            "initial_direction": np.rad2deg(self.initial_direction),
            "speed_range": list(self.speed_range),
            "direction_range": list(self.direction_range),
            "mu_speed": self.mu_speed,
            "mu_direction": self.mu_direction,
            "sigma_speed": self.sigma_speed,
            "sigma_direction": self.sigma_direction,
        }
        return config_dict


@dataclass
class Config:
    "Configuration class for managing environment disturbance/stochasticity parameters"

    wind: Optional[GaussMarkovDisturbanceParams] = None
    waves: Optional[dict] = None
    currents: Optional[GaussMarkovDisturbanceParams] = GaussMarkovDisturbanceParams()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "wind" in config_dict:
            config.wind = cp.convert_settings_dict_to_dataclass(GaussMarkovDisturbanceParams, config_dict["wind"])

        if "currents" in config_dict:
            config.currents = cp.convert_settings_dict_to_dataclass(GaussMarkovDisturbanceParams, config_dict["currents"])

        return config

    def to_dict(self) -> dict:
        config_dict: dict = {}
        if self.wind is not None:
            config_dict["wind"] = self.wind.to_dict()

        if self.currents is not None:
            config_dict["currents"] = self.currents.to_dict()

        return config_dict


class IDisturbance(ABC):
    @abstractmethod
    def update(self, t: float, dt: float) -> None:
        """Updates the disturbance process from time t to t + dt.

        Args:
            - t (float): Current time
            - dt (float): Time step
        """

    @abstractmethod
    def get(self) -> np.ndarray:
        """Fetches disturbance vector at time t.

        Returns:
            - np.ndarray: Disturbance data
        """


class GaussMarkovDisturbance(IDisturbance):
    """Gauss-Markov disturbance model with random speed and direction (of e.g. wind or current) in the North-East frame."""

    def __init__(self, params: GaussMarkovDisturbanceParams):
        self._params: GaussMarkovDisturbanceParams = params
        self._speed: float = params.initial_speed
        self._direction: float = params.initial_direction

    def update(self, t: float, dt: float) -> None:
        """Update speed and direction dynamics in the gauss-marko process

        Args:
            - t (float): Current time
            - dt (float): Time step

        """
        if self._params.constant:
            return
        w_V = random.normalvariate(0.0, self._params.sigma_speed)
        w_beta = random.normalvariate(0.0, self._params.sigma_direction)
        V_dot = -self._params.mu_speed * self._speed + w_V
        beta_dot = -self._params.mu_direction * self._direction + w_beta

        # Euler integration
        self._speed = mf.sat(self._speed + V_dot * dt, self._params.speed_range[0], self._params.speed_range[1])
        self._direction = mf.sat(self._direction + beta_dot * dt, self._params.direction_range[0], self._params.direction_range[1])
        self._direction = mf.wrap_angle_to_pmpi(self._direction)

    def get(self) -> np.ndarray:
        """Fetches gauss-markov process disturbance vector [speed, direction] at time t.

        Returns:
            - np.ndarray: Disturbance speed and direction in the North-East frame
        """
        return np.array([self._speed, self._direction])


@dataclass
class DisturbanceData:
    """Class used for containing disturbance data. Dictionaries are used preliminarily for containing
    info on the different disturbances, to be flexible wrt the disturbance models that are used."""

    wind: dict
    waves: dict
    currents: dict

    def __init__(self):
        self.wind = {}
        self.waves = {}
        self.currents = {}


class Disturbance:
    """Class for managing the different disturbances affecting the ship motion (wind, waves, currents)."""

    def __init__(self, config: Config):
        self._wind: Optional[GaussMarkovDisturbance] = None
        self._waves: Optional[bool] = None
        self._currents: Optional[GaussMarkovDisturbance] = None

        if config.wind is not None:
            self._wind = GaussMarkovDisturbance(config.wind)

        if config.currents is not None:
            self._currents = GaussMarkovDisturbance(config.currents)

    def update(self, t: float, dt: float) -> None:
        """Updates the disturbance processes from time t to t + dt

        Args:
            - t (float): Current time
            - dt (float): Time step
        """

        if self._wind is not None:
            self._wind.update(t, dt)

        if self._currents is not None:
            self._currents.update(t, dt)

    def get(self) -> DisturbanceData:
        """Fetches the disturbance data at time t"""

        disturbance_data = DisturbanceData()

        if self._wind is not None:
            disturbance_data.wind = {}
            wind = self._wind.get()
            disturbance_data.wind["speed"] = wind[0]
            disturbance_data.wind["direction"] = wind[1]

        if self._currents is not None:
            disturbance_data.currents = {}
            current = self._currents.get()
            disturbance_data.currents["speed"] = current[0]
            disturbance_data.currents["direction"] = current[1]

        return disturbance_data
