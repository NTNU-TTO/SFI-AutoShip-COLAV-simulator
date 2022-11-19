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

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator


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
class KTPGuidancePars:
    """Parameter class for the Kinematic Trajectory Planner.

    Parameters:
        epsilon (float): Small value to avoid division by zero in derivative calculation.
        dt_seg (float): Time step for segment calculation when calculating reference heading values from waypoints.
    """

    epsilon: float = 0.00001
    dt_seg: float = 0.1


@dataclass
class Config:
    """Configuration class for managing guidance method parameters."""

    los: Optional[LOSGuidancePars] = None
    ktp: Optional[KTPGuidancePars] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "los" in config_dict:
            config.los = cp.convert_settings_dict_to_dataclass(LOSGuidancePars, config_dict["los"])

        if "ktp" in config_dict:
            config.ktp = cp.convert_settings_dict_to_dataclass(KTPGuidancePars, config_dict["ktp"])

        return config


class IGuidance(ABC):
    """The InterfaceGuidance class is abstract and used to force
    the implementation of the below methods for all subclasses (guidance strategies),
    to comply with the guidance interface.
    """

    @abstractmethod
    def compute_references(
        self, waypoints: np.ndarray, speed_plan: np.ndarray, times: Optional[np.ndarray], xs: np.ndarray, dt: float
    ) -> np.ndarray:
        "Computes guidance reference states for the ship controller to track."


class KinematicTrajectoryPlanner(IGuidance):
    """Class which implements a simple kinematic trajectory planner.

    The main functionality converts a path (described by waypoints) and speed plan into a continuous
    trajectory, using cubic splines, for which 3DOF references in position, speed and acceleration are generated.

    Important internal variables:
        s (float): Keeps track of the current path variable/state of
        reference vehicle along the trajectory.
        init (bool): Flag to indicate if the trajectory has been initialized.
        x_spline (CubicSpline): Spline for x setpoints.
        y_spline (CubicSpline): Spline for y setpoints.
        heading_spline (PchipInterpolator): Spline for heading setpoint. Usage of piecewise cubic Hermite interpolator to reduce overshoot.
        speed_spline (PchipInterpolator): Spline for speed setpoint. Usage of piecewise cubic Hermite interpolator to reduce overshoot.
    """

    _pars: KTPGuidancePars
    _s: float = 0.001
    _init: bool = False
    _x_spline: CubicSpline
    _y_spline: CubicSpline
    _heading_spline: PchipInterpolator
    _speed_spline: PchipInterpolator

    def __init__(self, config: Optional[Config] = None) -> None:
        if config and config.ktp is not None:
            self._pars = config.ktp
        else:
            self._pars = KTPGuidancePars()

    def compute_references(
        self, waypoints: np.ndarray, speed_plan: np.ndarray, times: Optional[np.ndarray], xs: np.ndarray, dt: float
    ) -> np.ndarray:
        """Converts waypoints and speed plan into C² cubic spline,
         from which 3DOF reference states (not necessarily feasible) are computed.

        NOTE:
            The waypoints should be equidistant to ensure that the spline is well-behaved.
            Assumes validated inputs.

         Args:
            waypoints (np.array): 2 x n_wps waypoints array to follow.
            speed_plan (np.array): 1 x n_wps speed plan array to follow.
            times (np.array): 1 x n_wps of time instances corresponding to the waypoints.
            xs (np.array): n x 1 dimensional state of the ship
            dt (float): Time step between the previous and current run of this function.

        Returns:
            np.ndarray: 2-element array containing desired course and desired speed.
        """
        _, n_wps = waypoints.shape

        if times:
            linspace = times
        else:
            linspace = np.linspace(0.0, 1.0, n_wps)

        if not self._init:
            self._init = True
            self._x_spline = CubicSpline(linspace, waypoints[0, :])
            self._y_spline = CubicSpline(linspace, waypoints[1, :])
        else:
            self._x_spline = CubicSpline(linspace, waypoints[0, :], bc_type=((1, self._x_spline(self._s, 1)), (1, 0)))
            self._y_spline = CubicSpline(linspace, waypoints[1, :], bc_type=((1, self._y_spline(self._s, 1)), (1, 0)))

        # x_new = self._x_spline(np.linspace(0.0, 1.0, 100))
        # y_new = self._y_spline(np.linspace(0.0, 1.0, 100))

        # delta_s = np.sqrt(np.diff(x_new) ** 2 + np.diff(y_new) ** 2)
        # new_s_vals = np.cumsum([0, *np.sqrt(np.diff(x_new) ** 2 + np.diff(y_new) ** 2)])
        # new_s_vals = new_s_vals / np.max(new_s_vals)

        # self._x_spline = CubicSpline(new_s_vals, x_new)
        # self._y_spline = CubicSpline(new_s_vals, y_new)

        # Create reference heading values based on angle of extracted linear path segments in the x, y splines.
        n_heading_samples = round(n_wps / self._pars.dt_seg)
        path_values = np.linspace(0.0, 1.0, n_heading_samples)
        heading_references = np.zeros(n_heading_samples)
        for i in range(n_heading_samples - 1):
            x_d_diff = self._x_spline(path_values[i + 1]) - self._x_spline(path_values[i])
            y_d_diff = self._y_spline(path_values[i + 1]) - self._y_spline(path_values[i])
            heading_references[i] = np.arctan2(y_d_diff, x_d_diff)
        heading_references[-1] = heading_references[-2]

        self._heading_spline = PchipInterpolator(path_values, np.unwrap(heading_references))
        self._speed_spline = PchipInterpolator(linspace, speed_plan)

        # der_heading = [0, *np.diff(self._heading_spline(path_values))] / path_values

        s_dot, s_ddot = self._compute_path_variable_derivatives()

        eta_ref = self._compute_eta_ref()
        eta_dot_ref = self._compute_eta_dot_ref(s_dot)
        eta_ddot_ref = self._compute_eta_ddot_ref(s_dot, s_ddot)

        # Increment path variable to propagate reference vehicle along trajectory.
        self._s = mf.sat(self._s + dt * s_dot, 0.0, 1.0)

        plot = False
        if plot:

            figgca = plt.gcf()
            gca = figgca.axes[0]
            gca.plot(waypoints[1, :], waypoints[0, :], "rx", label="Waypoints", linewidth=2.0)
            gca.plot(
                self._y_spline(np.linspace(0.0, 1.0, 100)),
                self._x_spline(np.linspace(0.0, 1.0, 100)),
                "b.",
                label="Spline",
            )
            gca.set_xlabel("South (m)")
            gca.set_ylabel("North (m)")
            gca.legend()
            gca.grid()

            fig = plt.figure(figsize=(5, 10))
            axs = fig.subplot_mosaic(
                [
                    ["xy", "psi", "r"],
                    ["U", "Udot", "x"],
                ]
            )

            axs["psi"].plot(path_values, np.unwrap(heading_references), "rx", label="Waypoints")
            axs["psi"].plot(
                np.linspace(0.0, 1.0, 100),
                self._heading_spline(np.linspace(0.0, 1.0, 100)),
                "b.",
                label="heading spline",
            )
            axs["psi"].legend()
            axs["psi"].grid()

            axs["r"].plot(
                np.linspace(0.0, 1.0, 100),
                self._heading_spline(np.linspace(0.0, 1.0, 100), 1),
                "b.",
                label="yaw rate spline",
            )
            axs["r"].legend()
            axs["r"].grid()

            axs["U"].plot(linspace, self._speed_spline(linspace), "rx", label="Waypoints")
            axs["U"].plot(
                np.linspace(0.0, 1.0, 100),
                self._speed_spline(np.linspace(0.0, 1.0, 100)),
                "b.",
                label="speed spline",
            )
            axs["U"].legend()
            axs["U"].grid()

            axs["Udot"].plot(
                np.linspace(0.0, 1.0, 100),
                self._speed_spline(np.linspace(0.0, 1.0, 100), 1),
                "b.",
                label="speed der spline",
            )
            axs["Udot"].legend()
            axs["Udot"].grid()

            axs["x"].plot(linspace, waypoints[0, :], "rx")
            axs["x"].plot(
                np.linspace(0.0, 1.0, 100),
                self._x_spline(np.linspace(0.0, 1.0, 100)),
                "b.",
                label="x spline",
            )
            axs["x"].legend()
            axs["x"].grid()
            plt.show()
            plt.close()

        return np.concatenate((eta_ref, eta_dot_ref, eta_ddot_ref))

    def _compute_path_variable_derivatives(self) -> Tuple[float, float]:
        s_dot = self._speed_spline(self._s) / np.sqrt(
            self._pars.epsilon + np.power(self._x_spline(self._s, 1), 2.0) + np.power(self._y_spline(self._s, 1), 2.0)
        )

        s_ddot = s_dot * (
            self._speed_spline(self._s, 1)
            / np.sqrt(
                self._pars.epsilon
                + np.power(self._x_spline(self._s, 1), 2.0)
                + np.power(self._y_spline(self._s, 1), 2.0)
            )
            - self._speed_spline(self._s)
            * (
                self._x_spline(self._s, 1) * self._x_spline(self._s, 2)
                + self._y_spline(self._s, 1) * self._y_spline(self._s, 2)
            )
            / np.power(
                np.sqrt(
                    self._pars.epsilon
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

    def compute_references(
        self, waypoints: np.ndarray, speed_plan: np.ndarray, times: Optional[np.ndarray], xs: np.ndarray, dt: float
    ) -> np.ndarray:
        """Computes references in course and speed using the LOS guidance law.

        Args:
            waypoints (np.array): 2 x n_wps waypoints array to follow.
            speed_plan (np.array): 1 x n_wps speed plan array to follow.
            times (np.array): 1 x n_wps of time instances corresponding to the waypoints.
            xs (np.array): n x 1 dimensional state of the ship
            dt (float): Time step between the previous and current run of this function.

        Returns:
            np.ndarray: 2-element array containing desired course and desired speed.
        """
        self._find_active_wp_segment(waypoints, xs)

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

        return np.array([0.0, 0.0, chi_d, U_d, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _find_active_wp_segment(self, waypoints: np.ndarray, xs: np.ndarray) -> None:
        """Finds the active line segment between waypoints to follow.

            Assumes validated inputs.
        Args:
            waypoints (np.array): 2 x n_wps waypoints array to follow.
            xs (np.array): n x 1 dimensional state of the ship
        """
        _, n_wps = waypoints.shape

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
