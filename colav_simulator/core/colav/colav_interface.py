"""
    colav_interface.py

    Summary:
        Contains the interface used by all COLAV planning algorithms that
        wants to be run with the COLAV simulator.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from seacharts.enc import ENC


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
