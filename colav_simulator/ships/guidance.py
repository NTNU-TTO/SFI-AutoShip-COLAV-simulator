"""
    guidance.py

    Summary:
        Contains class definitions for guidance methods.
        Every class must adhere to the model interface IGuidance.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import colav_simulator.utils.math_functions as mf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline


@dataclass
class LOSGuidancePars:
    """Parameter class for the LOS guidance method.

    Parameters:
        pass_angle_threshold (float): First threshold for switching between wp segments
        R_a (float): Radius of acceptance, second threshold for switching between wp segments
        K_p (float): Proportional gain in LOS law, K_p = 1 / lookahead distance
        K_i (float): Integral action gain in LOS law.
        e_int_max (float): Maximum integrated cross-track error when using integral action
    """

    pass_angle_threshold: float = 90.0
    R_a: float = 10.0
    K_p: float = 1.0 / 40.0
    K_i: float = 0.0
    e_int_max: float = 50.0


@dataclass
class Config:
    """Configuration class for managing guidance method parameters."""

    name: str
    los: Optional[LOSGuidancePars]


class IGuidance(ABC):
    """The InterfaceGuidance class is abstract and used to force
    the implementation of the below methods for all subclasses (guidance strategies),
    to comply with the guidance interface.
    """

    @abstractmethod
    def compute_references(
        self, waypoints: np.ndarray, speed_plan: np.ndarray, xs: np.ndarray, dt: float
    ) -> np.ndarray:
        "Computes guidance reference states for the ship controller to track."


class KinematicTrajectoryPlanner(IGuidance):
    """Class which implements a simple kinematic trajectory planner.

    The main functionality converts a path (described by waypoints) and speed plan into a continuous
    trajectory, using cubic splines, for which 3DOF references in position, speed and acceleration are generated.

    Important internal variables:
        s (float): Keeps track of the current path variable/state of
        reference vehicle along the trajectory.
        epsilon (float): Small value to avoid division by zero.
        init (bool): Flag to indicate if the trajectory has been initialized.
        x_spline (CubicSpline): Spline for x setpoints.
        y_spline (CubicSpline): Spline for y setpoints.
        heading_spline (CubicSpline): Spline for heading setpoint.
        speed_spline (CubicSpline): Spline for speed setpoint.
    """

    _s: float = 0.001
    _epsilon: float = 0.000000001
    _init: bool = False
    _x_spline: CubicSpline
    _y_spline: CubicSpline
    _heading_spline: CubicSpline
    _speed_spline: CubicSpline

    def compute_references(
        self, waypoints: np.ndarray, speed_plan: np.ndarray, xs: np.ndarray, dt: float
    ) -> np.ndarray:
        """Converts waypoints and speed plan into C² cubic spline,
         from which 3DOF reference states (not necessarily feasible) are computed.

        NOTE: The waypoints should be equidistant to ensure that the spline is well-behaved.

         Args:
            waypoints (np.array): 2 x n_wps waypoints array to follow.
            speed_plan (np.array): 1 x n_wps speed plan array to follow.
            xs (np.array): n x 1 dimensional state of the ship
            dt (float): Time step between the previous and current run of this function.

        Returns:
            np.ndarray: 2-element array containing desired course and desired speed.
        """

        n_px, n_wps = waypoints.shape
        if n_px != 2:
            raise ValueError("Waypoints do not contain planar coordinates along each column!")

        if n_wps < 2:
            raise ValueError("Insufficient number of waypoints (< 2)!")

        course_plan = np.zeros(n_wps)
        for i in range(n_wps - 1):
            course_plan[i] = np.arctan2(waypoints[1, i + 1] - waypoints[1, i], waypoints[0, i + 1] - waypoints[0, i])
        course_plan[-1] = course_plan[-2]

        linspace = np.linspace(0.0, 1.0, n_wps)
        if not self._init:
            self._init = True
            x_dot = xs[3] * np.cos(xs[2]) - xs[4] * np.sin(xs[2])
            y_dot = xs[3] * np.sin(xs[2]) - xs[4] * np.cos(xs[2])
            psi_dot = xs[5]
            self._x_spline = CubicSpline(linspace, waypoints[0, :], bc_type=((1, x_dot), (1, 0)))
            self._y_spline = CubicSpline(linspace, waypoints[1, :], bc_type=((1, y_dot), (1, 0)))
            self._heading_spline = CubicSpline(linspace, np.unwrap(course_plan), bc_type=((1, 0), (1, 0)))
            self._speed_spline = CubicSpline(linspace, speed_plan, bc_type=((1, 0), (1, 0)))
        else:
            self._x_spline = CubicSpline(linspace, waypoints[0, :], bc_type=((1, self._x_spline(self._s, 1)), (1, 0)))
            self._y_spline = CubicSpline(linspace, waypoints[1, :], bc_type=((1, self._y_spline(self._s, 1)), (1, 0)))
            self._heading_spline = CubicSpline(
                linspace, np.unwrap(course_plan), bc_type=((1, self._heading_spline(self._s, 1)), (1, 0))
            )
            self._speed_spline = CubicSpline(
                linspace, speed_plan, bc_type=((1, self._speed_spline(self._s, 1)), (1, 0))
            )

        s_dot, s_ddot = self._compute_path_variable_derivatives()

        plt.figure(1)
        plt.plot(waypoints[1, :], waypoints[0, :], "rx", label="Waypoints")
        plt.plot(
            self._y_spline(np.linspace(0.0, 1.0, 1000)),
            self._x_spline(np.linspace(0.0, 1.0, 1000)),
            "b",
            label="Spline",
        )
        plt.xlabel("South (m)")
        plt.ylabel("North (m)")
        plt.legend()
        plt.grid()

        plt.figure(2)
        plt.plot(linspace, course_plan, "rx", label="Waypoints")
        plt.plot(
            np.linspace(0.0, 1.0, 1000),
            self._heading_spline(np.linspace(0.0, 1.0, 1000)),
            "b",
            label="heading spline",
        )
        plt.legend()
        plt.grid()

        plt.figure(3)
        plt.plot(
            np.linspace(0.0, 1.0, 1000),
            self._heading_spline(np.linspace(0.0, 1.0, 1000), 1),
            "b",
            label="yaw rate spline",
        )
        plt.legend()
        plt.grid()
        plt.show()

        eta_ref = self._compute_eta_ref()
        eta_dot_ref = self._compute_eta_dot_ref(s_dot)
        eta_ddot_ref = self._compute_eta_ddot_ref(s_dot, s_ddot)

        # Increment path variable to propagate reference vehicle along trajectory.
        self._s = mf.sat(self._s + dt * s_dot, 0.0, 1.0)

        return np.concatenate((eta_ref, eta_dot_ref, eta_ddot_ref))

    def _compute_path_variable_derivatives(self) -> Tuple[float, float]:
        s_dot = self._speed_spline(self._s) / np.sqrt(
            self._epsilon + np.power(self._x_spline(self._s, 1), 2.0) + np.power(self._y_spline(self._s, 1), 2.0)
        )

        s_ddot = s_dot * (
            self._speed_spline(self._s, 1)
            / np.sqrt(
                self._epsilon + np.power(self._x_spline(self._s, 1), 2.0) + np.power(self._y_spline(self._s, 1), 2.0)
            )
            - self._speed_spline(self._s)
            * (
                self._x_spline(self._s, 1) * self._x_spline(self._s, 2)
                + self._y_spline(self._s, 1) * self._y_spline(self._s, 2)
            )
            / np.power(
                np.sqrt(
                    self._epsilon
                    + np.power(self._x_spline(self._s, 1), 2.0)
                    + np.power(self._y_spline(self._s, 1), 2.0)
                ),
                3.0,
            )
        )

        return s_dot, s_ddot

    def _compute_eta_ref(self) -> np.ndarray:
        return np.array([self._x_spline(self._s), self._y_spline(self._s), self._heading_spline(self._s)])

    def _compute_eta_dot_ref(self, s_dot: float) -> np.ndarray:
        return np.array(
            [
                self._x_spline(self._s, 1) * s_dot,
                self._y_spline(self._s, 1) * s_dot,
                self._heading_spline(self._s, 1) * s_dot,
            ]
        )

    def _compute_eta_ddot_ref(self, s_dot: float, s_ddot: float) -> np.ndarray:
        return np.array(
            [
                self._x_spline(self._s, 2) * np.power(s_dot, 2.0) + self._x_spline(self._s, 1) * s_ddot,
                self._y_spline(self._s, 2) * np.power(s_dot, 2.0) + self._y_spline(self._s, 1) * s_ddot,
                self._heading_spline(self._s, 2) * np.power(s_dot, 2.0) + self._heading_spline(self._s, 1) * s_ddot,
            ]
        )


class LOSGuidance(IGuidance):
    """Class which implements the Line-of-Sight guidance strategy.

    Internal variables:
        wp_counter (float): Keeps track of the current waypoint segment
    """

    _wp_counter: int = 0
    _e_int: float = 0.0
    _pars: LOSGuidancePars

    def __init__(self, config: Optional[Config] = None) -> None:
        if config and config.los is not None:
            self._pars = config.los
        else:
            self._pars = LOSGuidancePars()

    def find_active_wp_segment(self, waypoints: np.ndarray, xs: np.ndarray) -> None:
        """Finds the active line segment between waypoints to follow.

        Args:
            waypoints (np.array): 2 x n_wps waypoints array to follow.
            xs (np.array): n x 1 dimensional state of the ship
        """
        n_px, n_wps = waypoints.shape
        if n_px != 2:
            raise ValueError("Waypoints do not contain planar coordinates along each column!")

        if n_wps < 2:
            raise ValueError("Insufficient number of waypoints (< 2)!")

        if xs.size < 2:
            raise ValueError("Wrong state dimension!")

        for i in range(self._wp_counter, n_wps - 1):
            d_0wp_vec = waypoints[:, i + 1] - xs[0:2]
            L_wp_segment = waypoints[:, i + 1] - waypoints[:, i]

            segment_passed = self._check_for_wp_segment_switch(L_wp_segment, d_0wp_vec)
            if segment_passed:
                self._wp_counter += 1
                print(f"Segment {i} passed!")
            else:
                break

    def compute_references(
        self, waypoints: np.ndarray, speed_plan: np.ndarray, xs: np.ndarray, dt: float
    ) -> np.ndarray:
        """Computes references in course and speed using the LOS guidance law.

        Args:
            waypoints (np.array): 2 x n_wps waypoints array to follow.
            speed_plan (np.array): 1 x n_wps speed plan array to follow.
            xs (np.array): n x 1 dimensional state of the ship
            dt (float): Time step between the previous and current run of this function.

        Returns:
            np.ndarray: 2-element array containing desired course and desired speed.
        """
        self.find_active_wp_segment(waypoints, xs)

        n_sp_dim = speed_plan.ndim
        if n_sp_dim != 1:
            raise ValueError("Speed plan does not consist of scalar reference values!")
        n_wps = speed_plan.size
        L_wp_segment = np.zeros(2)
        if self._wp_counter + 1 >= n_wps:
            L_wp_segment = waypoints[:, self._wp_counter] - waypoints[:, self._wp_counter - 1]
        else:
            L_wp_segment = waypoints[:, self._wp_counter + 1] - waypoints[:, self._wp_counter]

        alpha = np.arctan2(L_wp_segment[1], L_wp_segment[0])
        e = -(xs[0] - waypoints[0, self._wp_counter]) * np.sin(alpha) + (
            xs[1] - waypoints[1, self._wp_counter]
        ) * np.cos(alpha)
        self._e_int += e * dt
        if self._e_int >= self._pars.e_int_max:
            self._e_int -= e * dt

        chi_r = np.arctan2(-(self._pars.K_p * e + self._pars.K_i * self._e_int), 1)
        chi_d = mf.wrap_angle_to_pmpi(alpha + chi_r)

        U_d = speed_plan[self._wp_counter]

        return np.array([U_d, chi_d])

    def _check_for_wp_segment_switch(self, wp_segment: np.ndarray, d_0wp: np.ndarray) -> bool:
        """Checks if a switch should be made from the current to the next
        waypoint segment.

        Args:
            wp_segment (np.array): 2D vector describing the distance from waypoint i to i + 1 in the current segment.
            d_0wp (np.array): 2D distance vector from state to waypoint i + 1.

        Returns:
            bool: If the switch should be made or not.
        """
        wp_segment = mf.normalize_vec(wp_segment)
        d_0wp_norm = np.linalg.norm(d_0wp)
        d_0wp = mf.normalize_vec(d_0wp)

        segment_passed = wp_segment.dot(d_0wp) < np.cos(np.deg2rad(self._pars.pass_angle_threshold))

        segment_passed = segment_passed or d_0wp_norm <= self._pars.R_a

        return segment_passed
