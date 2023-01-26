"""
    kuwata_vo.py

    Summary:
        A reactive COLAV planning algorithm based on the paper
        "Safe Maritime Autonomous Navigation With COLREGS, Using Velocity Obstacles"
        by Kuwata et al.

    Author: Trym Tengesdal
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity as affinity
import shapely.geometry as geometry
from seacharts.enc import ENC


class VOCOLREGSSituation(Enum):
    """Enum for the different COLREGS rules considered in VO."""

    HO = 0  # Head-on
    OT_ing = 1  # Overtaking
    OT_en = 2  # Overtaken
    CR_PS = 3  # Crossing with obstacle on the left/port side
    CR_SS = 4  # Crossing with obstacle on the rigght/starboard side


# @dataclass
# class COLREGSConstraints:
#     """Parameters for the COLREGS constraints."""

#     type: VOCOLREGSSituation
#     heading_diff_limits: list
#     bearing_limits: list
#     cross_track_limits: list
#     along_track_limits: list

#     def to_dict(self) -> dict:
#         return asdict(self)

#     @classmethod
#     def from_dict(cls, data: dict) -> "COLREGSConstraints":
#         output = COLREGSConstraints(
#             type=VOCOLREGSSituation(data["type"]),
#             heading_diff_limits=data["heading_diff_limits"],
#             bearing_limits=data["bearing_limits"],
#             cross_track_limits=data["cross_track_limits"],
#             along_track_limits=data["along_track_limits"],
#         )
#         return output

#     def check_if_satisfied(self, p_os: np.ndarray, psi_os: float, p_do: np.ndarray, psi_do: float) -> bool:
#         """Checks if the COLREGS constraints are satisfied for the given ownship and dynamic obstacle states."""
#         bearing_do_os = mf.compute_bearing(psi_os, p_do, p_os)
#         heading_diff = mf.wrap_angle_diff_to_pmpi(psi_do, psi_os)

#         Rmtrx2D_ned_body = mf.Rmtrx2D(psi_os)
#         p_diff_body = Rmtrx2D_ned_body @ (p_do - p_os)

#         heading_constraint_satisfied = self.heading_diff_limits[0] <= heading_diff <= self.heading_diff_limits[1]
#         bearing_do_os_constraint_satisfied = self.bearing_limits[0] <= bearing_do_os <= self.bearing_limits[1]
#         cross_track_constraint_satisfied = self.cross_track_limits[0] <= p_diff_body[1] <= self.cross_track_limits[1]
#         along_track_constraint_satisfied = self.along_track_limits[0] <= p_diff_body[0] <= self.along_track_limits[1]

#         satisfied = heading_constraint_satisfied and bearing_do_os_constraint_satisfied and cross_track_constraint_satisfied and along_track_constraint_satisfied
#         return satisfied


@dataclass
class VOParams:
    """Parameters for the velocity obstacle algorithm."""

    length_os: float = 10.0  # The length of the ownship [m]
    width_os: float = 5.0  # The width of the ownship [m]
    draft_os: float = 2.0  # The draft of the ownship [m]
    planning_frequency: float = 1.0  # The planning frequency.
    t_max: float = 10.0  # Precollision check maximum time.
    d_min: float = 100.0  # Precollision check minimum distance.
    t_grounding_max: float = 10.0  # Maximum time to consider for grounding.
    t_colregs_update_interval: float = 1.0  # The interval at which the COLREGS situation for an obstacle is updated.
    head_on_width: float = 30.0  # The width of the head-on sector for determining COLREGS constraints/situations.
    overtaking_angle: float = 112.0  # The boundary angle for the overtaking sector for determining COLREGS constraints/situations.
    heading_set_limits: list = field(default_factory=lambda: [-179.0, 180.0])  # List of heading modifications to consider in the planning [deg].
    heading_set_spacing: float = 5.0  # The spacing between the headings to consider in the planning [deg]
    speed_set_limits: list = field(
        default_factory=lambda: [0.0, 10.0]
    )  # List of speed modifications to consider in the planning. [m/s] Should be set based on the ship min-max speed.
    speed_set_spacing: float = 0.5  # The spacing between the speeds to consider in the planning [m/s]
    max_speed_uncertainty: list = field(default_factory=lambda: [1.0, 1.0])  # The maximum speed uncertainty in x, y [m/s]
    Q: np.ndarray = field(default_factory=lambda: np.eye(2))  # The weighting matrix for the cost function.

    def to_dict(self):
        output = {
            "length_os": self.length_os,
            "width_os": self.width_os,
            "draft_os": self.draft_os,
            "planning_frequency": self.planning_frequency,
            "t_max": self.t_max,
            "d_min": self.d_min,
            "t_grounding_max": self.t_grounding_max,
            "t_colregs_update_interval": self.t_colregs_update_interval,
            "head_on_width": float(np.rad2deg(self.head_on_width)),
            "overtaking_angle": float(np.rad2deg(self.overtaking_angle)),
            "heading_set_limits": self.heading_set_limits,
            "heading_set_spacing": self.heading_set_spacing,
            "speed_set_limits": self.speed_set_limits,
            "speed_set_spacing": self.speed_set_spacing,
            "max_speed_uncertainty": self.max_speed_uncertainty,
            "Q": self.Q.diagonal().tolist(),
        }
        return output

    @classmethod
    def from_dict(cls, data: dict):
        output = VOParams(
            length_os=data["length_os"],
            width_os=data["width_os"],
            draft_os=data["draft_os"],
            planning_frequency=data["planning_frequency"],
            t_max=data["t_max"],
            d_min=data["d_min"],
            t_grounding_max=data["t_grounding_max"],
            t_colregs_update_interval=data["t_colregs_update_interval"],
            head_on_width=float(np.deg2rad(data["head_on_width"])),
            overtaking_angle=float(np.deg2rad(data["overtaking_angle"])),
            heading_set_limits=data["heading_set_limits"],
            heading_set_spacing=data["heading_set_spacing"],
            speed_set_limits=data["speed_set_limits"],
            speed_set_spacing=data["speed_set_spacing"],
            max_speed_uncertainty=data["max_speed_uncertainty"],
            Q=np.diag(data["Q"]),
        )
        return output


class VO:
    def __init__(self, config: Optional[VOParams] = None) -> None:
        if config:
            self._params = config
        else:
            self._params = VOParams()

        self._initialized: bool = False
        self._t_prev: float = 0.0

        self._poly_os = geometry.Polygon([(-self._params.length_os / 2, -self._params.width_os / 2), (self._params.length_os / 2, -self._params.width_os / 2), (self._params.length_os / 2, self._params.width_os / 2), (-self._params.length_os / 2, self._params.width_os / 2)])
        self._speed_uncertainty_poly = geometry.Polygon(
            [
                (-self._params.max_speed_uncertainty[0], -self._params.max_speed_uncertainty[1]),
                (self._params.max_speed_uncertainty[0], -self._params.max_speed_uncertainty[1]),
                (self._params.max_speed_uncertainty[0], self._params.max_speed_uncertainty[1]),
                (-self._params.max_speed_uncertainty[0], self._params.max_speed_uncertainty[1]),
            ]
        )
        self._relevant_do_list: list = []
        self._colregs_situations: list = []
        self._speed_set = np.arange(self._params.speed_set_limits[0], self._params.speed_set_limits[1], self._params.speed_set_spacing)
        self._heading_set = np.arange(self._params.heading_set_limits[0], self._params.heading_set_limits[1], self._params.heading_set_spacing)
        self._admissible_speed_headings: np.ndarray = np.ones((len(self._speed_set), len(self._heading_set)))
        self._speed_heading_costs: np.ndarray = np.zeros((len(self._speed_set), len(self._heading_set)))
        self._speed_opt_prev: float = 0.0
        self._heading_opt_prev: float = 0.0

    def get_current_plan(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the current plan."""
        return np.array([0.0, 0.0, self._heading_opt_prev]), np.array([self._speed_opt_prev, 0.0, 0.0]), np.zeros(3)

    def plan(self, t: float, v_ref: np.ndarray, ownship_state: np.ndarray, do_list: list, enc: Optional[ENC] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self._initialized:
            self._t_prev = t
            self._initialized = True

        if t - self._t_prev % (1.0 / self._params.planning_frequency) > 0.0001:
            return self.get_current_plan()

        p_os = ownship_state[0:2]
        psi_os = ownship_state[2]
        Rmtrx = mf.Rmtrx2D(psi_os)
        v_os = Rmtrx * ownship_state[3:5]
        poly_os: geometry.Polygon = affinity.rotate(self._poly_os, psi_os)
        # poly_os = affinity.translate(self._poly_os, p_os[0], p_os[1])

        self._admissible_speed_headings = np.ones((len(self._speed_set), len(self._heading_set)))
        expanded_poly_do_list = []
        for _, do_info in enumerate(do_list):
            id_do, state, _, length, width = do_info
            p_do = state[0:2]
            v_do = state[2:4]
            psi_do = np.arctan2(v_do[1], v_do[0])

            if mhm.check_if_vessel_is_passed_by(p_os, v_os, p_do, v_do):
                if id_do in self._relevant_do_list:
                    self._colregs_situations.pop(id_do)
                    self._relevant_do_list.pop(id_do)
                continue

            situation_started = False
            if id_do not in self._relevant_do_list:
                situation_started = self._precollision_check(p_os, v_os, p_do, v_do)

            if situation_started:
                self._relevant_do_list.append(id)
                self._colregs_situations.append((id_do, t, self._determine_colregs_situation(p_os, v_os, p_do, v_do)))
            else:
                self._update_colregs_situation(t, p_os, psi_os, p_do, psi_do, id_do)

            _, _, situation = [x for x in self._colregs_situations if x[0] == id_do][0]

            poly_do = geometry.Polygon([(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)])
            poly_do = affinity.rotate(poly_do, psi_do)
            poly_do = affinity.translate(poly_do, p_do[0], p_do[1])

            expanded_poly_do = self._compute_expanded_do_polygon(poly_os, poly_do)
            expanded_poly_do_list.append((id, expanded_poly_do))

            self._update_admissible_controls(situation, expanded_poly_do, p_do, v_do, p_os, v_os, enc)

            plot_vo_situation(expanded_poly_do, poly_os, poly_do, v_os, v_do)

        for i, speed in enumerate(self._speed_set):
            for j, heading in enumerate(self._heading_set):
                if self._admissible_speed_headings[i, j] == 0:
                    self._speed_heading_costs[i, j] = np.nan
                    continue

                v_os_new = np.array([speed * np.cos(heading), speed * np.sin(heading)])
                self._speed_heading_costs[i, j] = (v_ref - v_os_new) @ self._params.Q @ (v_ref - v_os_new)

        min_indices = np.nanargmin(self._speed_heading_costs)
        min_speed_idx: int = min_indices[0]
        min_heading_idx: int = min_indices[1]
        self._heading_opt_prev = self._heading_set[min_heading_idx]
        self._speed_opt_prev = self._speed_set[min_speed_idx]
        poses = np.array([0.0, 0.0, self._heading_opt_prev])
        velocities = np.array([self._speed_opt_prev, 0.0, 0.0])
        accelerations = np.zeros(3)
        return poses, velocities, accelerations

    def _update_admissible_controls(
        self,
        situation: VOCOLREGSSituation,
        expanded_poly_do: geometry.Polygon,
        p_do: np.ndarray,
        v_do: np.ndarray,
        p_os: np.ndarray,
        v_os: np.ndarray,
        enc: Optional[ENC] = None,
    ) -> None:
        """Updates the admissible controls based on the input dynamic obstacle polygon.

        Args:
            situation (VOCOLREGSSituation): The COLREGS situation.
            expanded_poly_do (geometry.Polygon): The expanded dynamic obstacle polygon.
            p_do (np.ndarray): Dynamic obstacle position.
            v_do (np.ndarray): Dynamic obstacle velocity.
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.
            enc (Optional[ENC], optional): The Electronic Navigational Charts. Defaults to None.
        """
        for i, speed in enumerate(self._speed_set):
            for j, heading in enumerate(self._heading_set):
                v_os_new = np.array([speed * np.cos(heading), speed * np.sin(heading)])

                # First check if VO is violated
                ray = geometry.LineString([p_os, p_os + v_os_new * self._params.t_max])
                if ray.intersects(expanded_poly_do):
                    self._admissible_speed_headings[i, j] = 0
                    continue

                # Check if own-ship follows a velocity in V3, i.e. it moves away from the DO
                in_v3 = False
                if np.dot(p_do - p_os, v_os_new - v_do) < 0:
                    in_v3 = True

                # Check if own-ship follows a velocity in V1, i.e. it moves such that the DO is on its starboard side
                in_v1 = False
                if not in_v3 and (p_do[0] - p_os[0]) * (v_os[1] - v_do[1]) - (p_do[1] - p_os[1]) * (v_os[0] - v_do[0]) < 0:
                    in_v1 = True

                if in_v1 and (situation == VOCOLREGSSituation.HO or situation == VOCOLREGSSituation.OT_ing or situation == VOCOLREGSSituation.CR_SS):
                    self._admissible_speed_headings[i, j] = 0

                # Check if velocity leads to collision with grounding hazard within t_max
                if self._check_grounding_risk(p_os, v_os_new, enc):
                    self._admissible_speed_headings[i, j] = 0

    def _check_grounding_risk(self, p_os: np.ndarray, v_os: np.ndarray, enc: ENC) -> bool:
        """Checks if the candidate velocity leads to a grounding hazard within t_max.

        Args:
            p_os (np.ndarray): Own-ship position.
            v_os (np.ndarray): Own-ship velocity.
            enc (ENC): Electronic Navigational Chart data.

        Returns:
            bool: True if the candidate velocity leads to a grounding hazard within t_max, False otherwise.
        """
        p2 = p_os + v_os * self._params.t_max
        return mapf.check_if_segment_crosses_grounding_hazards(enc, p2, p_os, self._params.draft_os)

    def _update_colregs_situation(self, t: float, p_os: np.ndarray, psi_os: float, p_do: np.ndarray, psi_do: float, id_do: int):
        """Updates the COLREGS situation of the ownship and the dynamic obstacle, if a time threshold is exceeded.

        Args:
            t (float): Current time.
            p_os (np.ndarray): Ownship position.
            v_os (np.ndarray): Ownship velocity.
            p_do (np.ndarray): Dynamic obstacle position.
            v_do (np.ndarray): Dynamic obstacle velocity.
            id_do (int): ID of the dynamic obstacle.
        """
        for idx, (id_, t_, _) in enumerate(self._colregs_situations):
            if id_ == id_do and t - t_ > self._params.t_colregs_update_interval:
                self._colregs_situations[idx] = (id_, t, self._determine_colregs_situation(p_os, psi_os, p_do, psi_do))
                return

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
        bearing_do_os = mf.compute_bearing(psi_os, p_do, p_os)
        bearing_os_do = mf.compute_bearing(psi_do, p_os, p_do)
        heading_diff = mf.wrap_angle_diff_to_pmpi(psi_do, psi_os)

        if (heading_diff < -np.pi + self._params.head_on_width / 2.0) or (heading_diff > np.pi - self._params.head_on_width / 2.0):
            return VOCOLREGSSituation.HO

        if bearing_os_do > self._params.overtaking_angle or bearing_os_do < -self._params.overtaking_angle:
            return VOCOLREGSSituation.OT_en

        if bearing_do_os > self._params.overtaking_angle or bearing_do_os < -self._params.overtaking_angle:
            return VOCOLREGSSituation.OT_ing

        if bearing_do_os < 0:
            return VOCOLREGSSituation.CR_PS

        return VOCOLREGSSituation.CR_SS

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

    def _compute_expanded_do_polygon(self, poly_os: geometry.Polygon, poly_do: geometry.Polygon) -> geometry.Polygon:
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
        expanded_poly_do_w_uncertainty = compute_minowski_sum(expanded_poly_do, self._speed_uncertainty_poly)
        return expanded_poly_do_w_uncertainty


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
