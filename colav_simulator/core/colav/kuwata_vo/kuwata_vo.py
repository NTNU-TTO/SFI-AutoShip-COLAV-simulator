"""
    kuwata_vo.py

    Summary:
        A COLAV planning algorithm based on the paper "Safe Maritime Autonomous Navigation With COLREGS, Using Velocity Obstacles" by Kuwata et al.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity as affinity
import shapely.geometry as geometry
from colav_simulator.core.colav.colav_interface import ICOLAV
from seacharts.enc import ENC


@dataclass
class VOParams:
    """Parameters for the velocity obstacle algorithm."""

    d_safe: float = 20.0  # The safe distance to keep from obstacles.
    horizon: float = 10.0  # The planning horizon.

    def to_dict(self) -> dict:
        return asdict(self)


class VO(ICOLAV):
    def __init__(self, length_os: float, width_os: float) -> None:
        self._initialized: bool = False
        self._t_prev: float = 0.0
        self._poly_os = geometry.Polygon([(-length_os / 2, -width_os / 2), (length_os / 2, -width_os / 2), (length_os / 2, width_os / 2), (-length_os / 2, width_os / 2)])

    def plan(self, t: float, ownship_state: np.ndarray, do_list: list, enc: Optional[ENC] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        poses = np.zeros(3)
        velocities = np.zeros(3)
        accelerations = np.zeros(3)

        p_os = ownship_state[0:2]
        psi_os = ownship_state[2]
        Rmtrx = mf.Rmtrx2D(psi_os)
        v_os = Rmtrx * ownship_state[3:5]
        poly_os: geometry.Polygon = affinity.rotate(self._poly_os, psi_os)
        poly_os = affinity.translate(self._poly_os, p_os[0], p_os[1])

        for i, do_info in enumerate(do_list):
            id, state, cov, length, width = do_info
            p_do = state[0:2]
            v_do = state[2:4]
            psi_do = np.arctan2(v_do[1], v_do[0])

            poly_do = geometry.Polygon([(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)])
            poly_do = affinity.rotate(poly_do, psi_do)
            poly_do = affinity.translate(poly_do, p_do[0], p_do[1])

            vo = self._compute_vo(poly_os, poly_do, v_os, v_do)

        return poses, velocities, accelerations

    def _compute_vo(self, poly_os: geometry.Polygon, poly_do: geometry.Polygon, v_os: np.ndarray, v_do: np.ndarray) -> geometry.Polygon:
        """Computes the velocity obstacle for the ownship and the dynamic obstacle.

        Args:
            poly_os (geometry.Polygon): Ownship polygon.
            poly_do (geometry.Polygon): Dynamic obstacle polygon.
            v_os (np.ndarray): Ownship velocity.
            v_do (np.ndarray): Dynamic obstacle velocity.

        Returns:
            geometry.Polygon: The velocity obstacle.
        """

        vo = geometry.Polygon()
        return vo


def compute_minowski_sum(poly1: geometry.Polygon, poly2: geometry.Polygon) -> geometry.Polygon:
    """Computes the Minkowski sum of two polygons.

    Args:
        poly1 (geometry.Polygon): First polygon.
        poly2 (geometry.Polygon): Second polygon.

    Returns:
        geometry.Polygon: The Minkowski sum of the two polygons.
    """
    vo_coords = []
    for p1 in poly1.exterior.coords:
        for p2 in poly2.exterior.coords:
            vo_coords.append((p1[0] + p2[0], p1[1] + p2[1]))

    return geometry.Polygon(vo_coords)


def compute_reflection(poly: geometry.Polygon) -> geometry.Polygon:
    """Computes the reflection of a polygon.

    Args:
        poly (geometry.Polygon): Polygon to reflect.

    Returns:
        geometry.Polygon: The reflected polygon.
    """
    new_coords = []
    for p in poly.exterior.coords:
        new_coords.append((-p[0], -p[1]))

    return geometry.Polygon(new_coords)


def plot_vo_situation(vo: geometry.Polygon, poly_os: geometry.Polygon, poly_do: geometry.Polygon, v_os: np.ndarray, v_do: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the vcurrent velocity obstacle situation.

    Args:
        vo (geometry.Polygon): The velocity obstacle.
        poly_os (geometry.Polygon): Ownship polygon.
        poly_do (geometry.Polygon): Dynamic obstacle polygon.
        v_os (np.ndarray): Ownship velocity.
        v_do (np.ndarray): Dynamic obstacle velocity.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes for the plot.
    """
    fig, ax = plt.subplots()
    vo_x, vo_y = vo.exterior.xy
    ax.plot(vo_x, vo_y, "r", label="VO")
    ax.plot(*poly_os.exterior.xy, "b", label="Ownship")
    ax.plot(*poly_do.exterior.xy, "g", label="Dynamic obstacle")

    plt.quiver(*poly_os.centroid.xy, v_os[1], v_os[0], color="b", scale=1)
    plt.quiver(*poly_do.centroid.xy, v_do[1], v_do[0], color="g", scale=1)
    plt.quiver(*poly_os.centroid.xy, v_do[1] - v_do[1], v_do[0] + v_os[0], color="k", scale=1)
    plt.show(block=False)

    return fig, ax
