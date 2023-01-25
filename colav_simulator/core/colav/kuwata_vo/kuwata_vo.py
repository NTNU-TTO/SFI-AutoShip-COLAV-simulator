"""
    kuwata_vo.py

    Summary:
        A COLAV planning algorithm based on the paper "Safe Maritime Autonomous Navigation With COLREGS, Using Velocity Obstacles" by Kuwata et al.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Tuple

import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity as affinity
import shapely.geometry as geometry
from seacharts.enc import ENC


class VOCOLREGSSituation(Enum):
    """Enum for the different COLREGS rules considered in VO."""

    HO = 0
    OT = 1
    CR_left = 2
    CR_right = 3


@dataclass
class COLREGSConstraints:
    """Parameters for the COLREGS constraints."""

    type: VOCOLREGSSituation
    heading_diff_limits: list
    bearing_limits: list
    cross_track_limits: list
    along_track_limits: list

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "COLREGSConstraints":
        output = COLREGSConstraints(
            type=VOCOLREGSSituation(data["type"]),
            heading_diff_limits=data["heading_diff_limits"],
            bearing_limits=data["bearing_limits"],
            cross_track_limits=data["cross_track_limits"],
            along_track_limits=data["along_track_limits"],
        )
        return output

    def check_if_satisfied(self, p_os: np.ndarray, psi_os: float, p_do: np.ndarray, psi_do: float) -> bool:
        """Checks if the COLREGS constraints are satisfied for the given ownship and dynamic obstacle states."""
        bearing_do_os = mf.compute_bearing(psi_os, p_do, p_os)
        heading_diff = mf.wrap_angle_diff_to_pmpi(psi_do, psi_os)

        Rmtrx2D_ned_body = mf.Rmtrx2D(psi_os)
        p_diff_body = Rmtrx2D_ned_body @ (p_do - p_os)

        heading_constraint_satisfied = self.heading_diff_limits[0] <= heading_diff <= self.heading_diff_limits[1]
        bearing_do_os_constraint_satisfied = self.bearing_limits[0] <= bearing_do_os <= self.bearing_limits[1]
        cross_track_constraint_satisfied = self.cross_track_limits[0] <= p_diff_body[1] <= self.cross_track_limits[1]
        along_track_constraint_satisfied = self.along_track_limits[0] <= p_diff_body[0] <= self.along_track_limits[1]

        satisfied = heading_constraint_satisfied and bearing_do_os_constraint_satisfied and cross_track_constraint_satisfied and along_track_constraint_satisfied
        return satisfied


@dataclass
class VOParams:
    """Parameters for the velocity obstacle algorithm."""

    planning_frequency: float = 1.0  # The planning frequency.
    t_max: float = 10.0  # Precollision check maximum time.
    d_min: float = 100.0  # Precollision check minimum distance.
    colregs_constraints: list = field(default_factory=lambda: [])  # List of COLREGS constraints for the different COLREGS situations.
    heading_set_limits: list = field(default_factory=lambda: [-179.0, 180.0])  # List of heading modifications to consider in the planning [deg].
    heading_set_spacing: float = 5.0  # The spacing between the headings to consider in the planning [deg]
    speed_set_limits: list = field(default_factory=lambda: [-10.0, 10.0])  # List of speed modifications to consider in the planning. [m/s]
    speed_set_spacing: float = 1.0  # The spacing between the speeds to consider in the planning [m/s]

    def to_dict(self):
        output = {
            "planning_frequency": self.planning_frequency,
            "t_max": self.t_max,
            "d_min": self.d_min,
            "heading_set_limits": self.heading_set_limits,
            "heading_set_spacing": self.heading_set_spacing,
            "speed_set_limits": self.speed_set_limits,
            "speed_set_spacing": self.speed_set_spacing,
        }
        output["colregs_constraints"] = []
        output["colregs_constraints"] = [constraint.to_dict() for constraint in self.colregs_constraints]
        return output

    @classmethod
    def from_dict(cls, data: dict):
        output = VOParams(
            planning_frequency=data["planning_frequency"],
            t_max=data["t_max"],
            d_min=data["d_min"],
            heading_set_limits=data["heading_set_limits"],
            heading_set_spacing=data["heading_set_spacing"],
            speed_set_limits=data["speed_set_limits"],
            speed_set_spacing=data["speed_set_spacing"],
        )
        output.colregs_constraints = [COLREGSConstraints.from_dict(constraint) for constraint in data["colregs_constraints"]]
        return output


class VO:
    def __init__(self, params: Optional[VOParams] = None, **kwargs) -> None:
        if params:
            self._params = params
        else:
            self._params = VOParams()

        self._initialized: bool = False
        self._t_prev: float = 0.0

        if kwargs and "length_os" in kwargs and "width_os" in kwargs:
            length_os = kwargs["length_os"]
            width_os = kwargs["width_os"]
        else:
            length_os = 10.0
            width_os = 5.0

        self._poly_os = geometry.Polygon([(-length_os / 2, -width_os / 2), (length_os / 2, -width_os / 2), (length_os / 2, width_os / 2), (-length_os / 2, width_os / 2)])
        self._relevant_do_list: list = []
        self._colregs_situations: list = []
        self._speed_set = np.arange(self._params.speed_set_limits[0], self._params.speed_set_limits[1], self._params.speed_set_spacing)
        self._heading_set = np.arange(self._params.heading_set_limits[0], self._params.heading_set_limits[1], self._params.heading_set_spacing)
        self._admissible_speed_headings: np.ndarray = np.ones((len(self._speed_set), len(self._heading_set)))
        self._speed_opt_prev: float = 0.0
        self._heading_opt_prev: float = 0.0

    def get_current_plan(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the current plan."""
        return np.array([0.0, 0.0, self._heading_opt_prev]), np.array([self._speed_opt_prev, 0.0, 0.0]), np.zeros(3)

    def plan(self, t: float, ownship_state: np.ndarray, do_list: list, enc: Optional[ENC] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self._initialized:
            self._t_prev = t
            self._initialized = True

        poses = np.zeros(3)
        velocities = np.zeros(3)
        accelerations = np.zeros(3)

        p_os = ownship_state[0:2]
        psi_os = ownship_state[2]
        Rmtrx = mf.Rmtrx2D(psi_os)
        v_os = Rmtrx * ownship_state[3:5]
        poly_os: geometry.Polygon = affinity.rotate(self._poly_os, psi_os)
        poly_os = affinity.translate(self._poly_os, p_os[0], p_os[1])

        expanded_poly_do_list = []
        for do_info in do_list:
            id, state, cov, length, width = do_info
            p_do = state[0:2]
            v_do = state[2:4]

            if mhm.check_if_vessel_is_passed_by(p_os, v_os, p_do, v_do):
                if id in self._relevant_do_list:
                    self._colregs_situations.pop(id)
                    self._relevant_do_list.pop(id)
                continue

            situation_started = False
            if id not in self._relevant_do_list:
                situation_started = self._precollision_check(p_os, v_os, p_do, v_do)

            if situation_started:
                self._relevant_do_list.append(id)
                self._colregs_situations.append((id, self._determine_colregs_situation(p_os, v_os, p_do, v_do)))

            psi_do = np.arctan2(v_do[1], v_do[0])

            poly_do = geometry.Polygon([(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)])
            poly_do = affinity.rotate(poly_do, psi_do)
            poly_do = affinity.translate(poly_do, p_do[0], p_do[1])

            expanded_poly_do = compute_expanded_do_polygon(poly_os, poly_do)
            expanded_poly_do_list.append((id, expanded_poly_do))

            self._check_if_ray_intersects_vo(expanded_poly_do, p_os, v_os, cov)

            plot_vo_situation(expanded_poly_do, poly_os, poly_do, v_os, v_do)

        return poses, velocities, accelerations

    def _determine_colregs_situation(self, p_os: np.ndarray, psi_os: float, p_do: np.ndarray, psi_do: float) -> VOCOLREGSSituation:
        """Determines the COLREGS situation of the ownship and the dynamic obstacle.

        Args:
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.
            p_do (np.ndarray): Dynamic obstacle position.
            v_do (np.ndarray): Dynamic obstacle velocity.


        Returns:
            VOCOLREGSSituation: The COLREGS situation of the ownship and the dynamic obstacle.
        """
        situation = VOCOLREGSSituation.HO
        for idx, constraint in self._params.colregs_constraints:
            satisfied = constraint.check_if_satisfied(p_os, psi_os, p_do, psi_do)
            if satisfied:
                situation = VOCOLREGSSituation(idx)
                break
        return situation

    def _precollision_check(self, p_os: np.ndarray, v_os: np.ndarray, p_do: np.ndarray, v_do: np.ndarray) -> bool:
        """Checks if the ownship and the dynamic obstacle are in a dangerous situation.

        Args:
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.
            p_do (np.ndarray): Dynamic obstacle position.
            v_do (np.ndarray): Dynamic obstacle velocity.

        Returns:
            bool: True if the ownship and the dynamic obstacle are in a dangerous situation.
        """
        t_cpa, d_cpa, __ = mhm.compute_vessel_pair_cpa(p_os, v_os, p_do, v_do)
        return t_cpa <= self._params.t_max and d_cpa <= self._params.d_min

    def _check_if_ray_intersects_vo(self, vo: geometry.Polygon, p_os: np.ndarray, v_os: np.ndarray) -> bool:
        """Checks if the ray from the ownship to the dynamic obstacle intersects the expanded DO shape.

        Args:
            vo (geometry.Polygon): The VO.
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.

        Returns:
            bool: True if the ray intersects the VO.
        """
        ray = geometry.LineString([p_os, p_os + v_os * self._params.t_max])
        return ray.intersects(vo)


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


def compute_expanded_do_polygon(poly_os: geometry.Polygon, poly_do: geometry.Polygon) -> geometry.Polygon:
    """Computes the minowski sum poly_do + (-poly_os) for the ownship (zero origin referenced) and the dynamic obstacle (referenced to its position).

    Args:
        poly_os (geometry.Polygon): Ownship polygon.
        poly_do (geometry.Polygon): Dynamic obstacle polygon.
        v_os (np.ndarray): Ownship velocity.
        v_do (np.ndarray): Dynamic obstacle velocity.

    Returns:
        geometry.Polygon: The velocity obstacle.
    """
    reflected_poly_os = compute_reflection(poly_os)
    expanded_poly_do = compute_minowski_sum(poly_do, reflected_poly_os)
    return expanded_poly_do


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
