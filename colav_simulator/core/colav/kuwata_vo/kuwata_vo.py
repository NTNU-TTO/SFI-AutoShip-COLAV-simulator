"""
    kuwata_vo.py

    Summary:
        A COLAV planning algorithm based on the paper "Safe Maritime Autonomous Navigation With COLREGS, Using Velocity Obstacles" by Kuwata et al.

    Author: Trym Tengesdal
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import numpy as np
from colav_simulator.core.colav.colav_interface import ICOLAV
from seacharts.enc import ENC


class VO(ICOLAV):
    def __init__(self) -> None:
        self._initialized = False

    def plan(self, t: float, ownship_state: np.ndarray, do_list: list, enc: Optional[ENC] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        poses = np.zeros(3)
        velocities = np.zeros(3)
        accelerations = np.zeros(3)

        p_os = ownship_state[0:2]
        psi_os = ownship_state[2]
        Rmtrx = mf.Rmtrx2D(psi_os)
        v_os = Rmtrx * ownship_state[3:5]
        n_do = len(do_list)
        for i, do_info in enumerate(do_list):
            id, state, cov, shape = do_info

            p_do = state[0:2]
            v_do = state[2:4]

        return poses, velocities, accelerations
