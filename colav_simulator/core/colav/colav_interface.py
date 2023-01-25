"""
    colav_interface.py

    Summary:
        Contains the interface used by all COLAV planning algorithms that
        wants to be run with the COLAV simulator.

        To add a new COLAV planning algorithm, create a new wrapper class for your
        COLAV algorithm which implements this interface. See an example for the Kuwata VO below.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.core.colav.kuwata_vo.kuwata_vo as kuwata_vo
import numpy as np
from seacharts.enc import ENC


@dataclass
class Config:
    """Configuration class for managing COLAV method parameters."""

    kuwata_vo: Optional[kuwata_vo.VOParams] = kuwata_vo.VOParams()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "kuwata_vo" in config_dict:
            config.kuwata_vo = cp.convert_settings_dict_to_dataclass(kuwata_vo.VOParams, config_dict["kuwata_vo"])

        return config

    def to_dict(self) -> dict:
        config_dict = {}

        if self.kuwata_vo is not None:
            config_dict["kuwata_vo"] = self.kuwata_vo.to_dict()

        return config_dict


class ICOLAV(ABC):
    @abstractmethod
    def plan(self, t: float, ownship_state: np.ndarray, do_list: list, enc: Optional[ENC] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plans a (hopefully) collision free trajectory for the ship to follow.

        Args:
            ownship_state (np.ndarray): The ownship state [x, y, psi, u, v, r].
            do_list (list): List of information on dynamic obstacles. This is a list of tuples of the form (id, state [x, y, Vx, Vy], covariance, length, width).
            enc (Optional[ENC]): The relevant Electronic Navigational Chart (ENC) for static obstacle info. Defaults to None.
            t (float): The current time.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The planned poses, velocities and accelerations over the COLAV planning horizon (if any).
        """

    @abstractmethod
    def get_current_plan(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the current planned trajectory.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The most recent planned poses, velocities and accelerations over the COLAV planning horizon (if any).
        """


class VOWrapper(ICOLAV):
    def __init__(self, config: kuwata_vo.VOParams, **kwargs) -> None:
        self._vo = kuwata_vo.VO(params=config, **kwargs)

    def plan(self, t: float, ownship_state: np.ndarray, do_list: list, enc: Optional[ENC] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._vo.plan(t, ownship_state, do_list, enc)

    def get_current_plan(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._vo.get_current_plan()


class COLAVBuilder:
    @classmethod
    def construct_colav(cls, config: Optional[Config] = None) -> ICOLAV:
        """Builds a colav system from the configuration

        Args:
            config (Optional[models.Config]): COLAV configuration. Defaults to None.

        Returns:
            ICOLAV: The COLAV system, e.g. Kuwata VO.
        """
        if config and config.kuwata_vo:
            return VOWrapper(config.kuwata_vo)
        else:
            return VOWrapper()
