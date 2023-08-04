"""
    models.py

    Summary:
        Contains class definitions for various models.
        Every model class must adhere to the interface IModel.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import numpy as np
import colav_simulator.core.stochasticity as stochasticity


@dataclass
class KinematicCSOGParams:
    """Parameter class for the Course and SPeed over Ground model.

    Parameters:
        T_chi (float): First order time constant for the course dynamics.
        T_U (float): First order time constant for the speed dynamics.

    """

    draft: float = 2.0
    length: float = 10.0
    ship_vertices: np.ndarray = np.empty(2)
    width: float = 3.0
    T_chi: float = 3.0
    T_U: float = 5.0
    r_max: float = float(np.deg2rad(4))
    U_min: float = 0.0
    U_max: float = 15.0

    @classmethod
    def from_dict(self, params_dict: dict):
        params = KinematicCSOGParams(
            draft=params_dict["draft"],
            length=params_dict["length"],
            width=params_dict["width"],
            ship_vertices=np.empty(2),
            T_chi=params_dict["T_chi"],
            T_U=params_dict["T_U"],
            r_max=np.deg2rad(params_dict["r_max"]),
            U_min=params_dict["U_min"],
            U_max=params_dict["U_max"],
        )
        params.ship_vertices = np.array(
            [
                [params.length / 2.0, -params.width / 2.0],
                [params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, params.width / 2.0],
                [-params.length / 2.0, -params.width / 2.0],
            ]
        ).T
        return params

    def to_dict(self):
        output_dict = asdict(self)
        output_dict["ship_vertices"] = self.ship_vertices.tolist()
        output_dict["r_max"] = np.rad2deg(self.r_max)
        return output_dict


@dataclass
class TelemetronParams:
    """Parameters for the Telemetron vessel (read only / fixed)."""

    draft: float = 0.5
    length: float = 8.0
    width: float = 3.0
    ship_vertices: np.ndarray = np.array([[3.75, 1.5], [4.25, 0.0], [3.75, -1.5], [-3.75, -1.5], [-3.75, 1.5]]).T
    l_r: float = 4.0  # Distance from CG to rudder
    M_rb: np.ndarray = np.diag([3980.0, 3980.0, 19703.0])  # Rigid body mass matrix
    M_a: np.ndarray = np.zeros((3, 3))  # Added mass matrix
    D_c: np.ndarray = np.diag([0.0, 0.0, 3224.0])  # Third order/cubic damping
    D_q: np.ndarray = np.diag([135.0, 2000.0, 0.0])  # Second order/quadratic damping
    D_l: np.ndarray = np.diag([50.0, 200.0, 1281.0])  # First order/linear damping
    Fx_limits: np.ndarray = np.array([-6550.0, 13100.0])  # Force limits in x
    Fy_limits: np.ndarray = np.array([-645.0, 645.0])  # Force limits in y
    r_max: float = float(np.deg2rad(15))
    U_min: float = 0.0
    U_max: float = 15.0


@dataclass
class CyberShip2Params:
    """Parameters for the CyberShip2 vessel (read only / fixed)."""

    rho: float = 1000.0  # Density of water
    draft: float = 5.0
    length: float = 1.255 * 70.0
    width: float = 0.29 * 70.0
    ship_vertices: np.ndarray = np.array(
        [[0.9 * length / 2.0, width / 2.0], [length / 2.0, 0.0], [0.9 * length / 2.0, -width / 2.0], [-length / 2.0, -width / 2.0], [-length / 2.0, width / 2.0]]
    ).T

    # The parameters below are scaled up 70 times to match the size of the real ship
    M_rb: np.ndarray = np.array(
        [[23.800, 0.0, 0.0], [0.0, 23.800, 23.800 * 0.046], [0.0, 23.800 * 0.046, 1.760]]
    )  # Rigid body mass, m = 23.800 kg, I_z = 1.760 kgm², x_g = 0.046 matrix
    M_a: np.ndarray = np.array([[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])  # Added mass matrix
    D_l: np.ndarray = np.array([[0.72253, 0.0, 0.0], [0.0, 0.88965, 7.250], [0.0, -0.03130, 1.9]])  # First order/linear damping
    # Nonlinear damping related parameters:
    X_uu: float = -1.32742
    X_uuu: float = -5.86643

    Y_vv: float = -36.47287
    Y_vr: float = -0.845
    Y_rv: float = -0.805
    Y_rr: float = -3.450

    N_vv: float = 3.95645
    N_rv: float = 0.130
    N_vr: float = 0.080
    N_rr: float = -0.750

    B: np.ndarray = np.array([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0], [0.078, -0.078, 0.466, 0.549, 0.549]])  # Actuator configuration matrix
    # Main propeller thruster 1 and 2 coefficients:
    T_nn_plus: float = 3.65034e-03
    T_nu_plus: float = 1.52468e-04
    T_nn_minus: float = 5.10256e-03
    T_nu_minus: float = 4.55822e-02
    # Bow thruster force coefficient
    T_n3n3: float = 1.56822e-04

    d_rud: float = 0.08  # Rudder diameter
    k_u: float = 0.5  # Induced velocity factor on fluid at rudder surface
    # Rudder lift force coefficient
    L_delta_plus: float = 6.43306
    L_ddelta_plus: float = 5.83594
    L_delta_minus: float = 3.19573
    L_ddelta_minus: float = 2.34356

    Fx_limits: np.ndarray = np.array([-2.0, 2.0])  # Force limits in x (unscaled)
    Fy_limits: np.ndarray = np.array([-2.0, 2.0])  # Force limits in y (unscaled)
    N_limits: np.ndarray = np.array([-1.5, 1.5])  # Torque limits in z (unscaled)
    max_propeller_speed: float = 25.0 # (unscaled)
    max_rudder_angle: float = 35.0 * np.pi / 180.0  # (unscaled)

    U_min: float = 0.0  # Min speed
    U_max: float = 15.0  # Max speed

    scaling_factor: float = 70.0  # Scaling factor for the ship size


@dataclass
class Config:
    """Configuration class for managing model parameters."""

    csog: Optional[KinematicCSOGParams] = KinematicCSOGParams()
    telemetron: Optional[TelemetronParams] = None
    cybership2: Optional[CyberShip2Params] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "csog" in config_dict:
            config.csog = cp.convert_settings_dict_to_dataclass(KinematicCSOGParams, config_dict["csog"])
            config.telemetron = None

        if "telemetron" in config_dict:
            config.telemetron = TelemetronParams()
            config.csog = None
            config.cybership2 = None

        if "cybership2" in config_dict:
            config.cybership2 = CyberShip2Params()
            config.csog = None
            config.telemetron = None

        return config

    def to_dict(self) -> dict:
        config_dict = {}

        if self.csog is not None:
            config_dict["csog"] = self.csog.to_dict()

        if self.telemetron is not None:
            config_dict["telemetron"] = ""

        return config_dict


class IModel(ABC):
    @abstractmethod
    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The r.h.s of the ODE x_k+1 = f(x_k, u_k) for the considered model in discrete time.

        Args:
            xs (np.ndarray): The state vector x_k
            u (np.ndarray): The input vector u_k

        NOTE: The state and input dimension may change depending on the model. Make sure to check compatibility between the controller you are using and the model.
        """

    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the lower and upper bounds for the inputs and states in the model.
        The output is on the form (lbu, ubu, lbx, ubx)."""


class ModelBuilder:
    @classmethod
    def construct_model(cls, config: Optional[Config] = None) -> IModel:
        """Builds a ship model from the configuration

        Args:
            config (Optional[models.Config]): Model configuration. Defaults to None.

        Returns:
            Model: Model as specified by the configuration, e.g. a KinematicCSOG model.
        """
        if config and config.csog:
            return KinematicCSOG(config.csog)
        elif config and config.telemetron:
            return Telemetron()
        else:
            return KinematicCSOG()


class KinematicCSOG(IModel):
    """Implements a planar kinematic model using Course over ground (COG) and Speed over ground (SOG):

    x_k+1 = x_k + U_k cos(chi_k)
    y_k+1 = y_k + U_k sin(chi_k)
    chi_k+1 = chi_k + (1 / T_chi)(chi_d - chi_k)
    U_k+1 = U_k + (1 / T_U)(U_d - U_k)


    where x,y are the planar coordinates, U the vessel SOG and chi the vessel COG => xs = [x, y, chi, U, 0, 0]
    """

    _n_x: int = 4
    _n_u: int = 2

    def __init__(self, params: Optional[KinematicCSOGParams] = None) -> None:
        if params is not None:
            self._params: KinematicCSOGParams = params
        else:
            self._params = KinematicCSOGParams()

    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State x_k = [x_k, y_k, chi_k, U_k, 0.0, 0.0]
            u (np.ndarray): Input equal to [chi_d, U_d, 0]

        Returns:
            np.ndarray: New state x_k+1.
        """
        if len(u) != 3:
            raise ValueError("Dimension of input array should be 3!")

        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        chi_d = u[0]
        U_d = mf.sat(u[1], 0.0, self._params.U_max)

        chi_diff = mf.wrap_angle_diff_to_pmpi(chi_d, xs[2])
        xs[3] = mf.sat(xs[3], 0.0, self._params.U_max)

        ode_fun = np.zeros(6)
        ode_fun[0] = xs[3] * np.cos(xs[2])
        ode_fun[1] = xs[3] * np.sin(xs[2])
        ode_fun[2] = mf.sat(chi_diff / self._params.T_chi, -self._params.r_max, self._params.r_max)
        ode_fun[3] = (U_d - xs[3]) / self._params.T_U

        return ode_fun

    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lbu = np.array([-np.inf, 0.0, -np.inf])
        ubu = np.array([np.inf, self._params.U_max, np.inf])
        lbx = np.array([-np.inf, -np.inf, -np.inf, -self._params.U_max, -self._params.U_max, -self._params.r_max])
        ubx = np.array([np.inf, np.inf, np.inf, self._params.U_max, self._params.U_max, self._params.r_max])
        return lbu, ubu, lbx, ubx

    @property
    def params(self):
        return self._params

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u


class Telemetron(IModel):
    """Implements a 3DOF underactuated vessel maneuvering model

    eta_dot = Rpsi(eta) * nu
    (M_rb + M_a) * nu_dot + C(nu) * nu + (D_l(nu) + D_nl) * nu = tau

    with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.

    Parameters:
        M_rb: Rigid body mass matrix
        M_a: Added mass matrix
        C: Coriolis matrix, computed from M = M_rb + M_a
        D_l: Linear damping matrix
        D_nl: Nonlinear damping matrix

    NOTE: When using Euler`s method, keep the time step small enough (e.g. around 0.1 or less) to ensure numerical stability.
    """

    _n_x: int = 6
    _n_u: int = 3

    def __init__(self) -> None:
        self._params: TelemetronParams = TelemetronParams()

    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State xs = [eta, nu]^T
            u (np.ndarray): Input vector u = tau (generalized forces in X, Y and N)

        Returns:
            np.ndarray: New state xs.
        """
        if u.size != self._n_u:
            raise ValueError("Dimension of input array should be 3!")
        if xs.size != self._n_x:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        eta[2] = mf.wrap_angle_to_pmpi(eta[2])

        nu = xs[3:6]
        nu[0] = mf.sat(nu[0], -1e10, self._params.U_max)
        nu[2] = mf.sat(nu[2], -self._params.r_max, self._params.r_max)

        u[0] = mf.sat(u[0], self._params.Fx_limits[0], self._params.Fx_limits[1])
        u[1] = mf.sat(u[1], self._params.Fy_limits[0], self._params.Fy_limits[1])
        u[2] = mf.sat(u[2], self._params.Fy_limits[0] * self._params.l_r, self._params.Fy_limits[1] * self._params.l_r)

        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)
        Cvv = mf.Cmtrx(self._params.M_rb + self._params.M_a, nu) @ nu
        Dvv = mf.Dmtrx(self._params.D_l, self._params.D_q, self._params.D_c, nu) @ nu

        ode_fun = np.zeros(6)
        ode_fun[0:3] = mf.Rmtrx(eta[2]) @ nu
        ode_fun[3:6] = Minv @ (-Cvv - Dvv + u)

        if np.abs(nu[0]) < 0.1:
            ode_fun[2] = 0.0

        return ode_fun

    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lbu = np.array([self._params.Fx_limits[0], self._params.Fy_limits[0], self._params.Fy_limits[0] * self._params.l_r])
        ubu = np.array([self._params.Fx_limits[1], self._params.Fy_limits[1], self._params.Fy_limits[1] * self._params.l_r])
        lbx = np.array([-np.inf, -np.inf, -np.inf, -self._params.U_max, -self._params.U_max, -self._params.r_max])
        ubx = np.array([np.inf, np.inf, np.inf, self._params.U_max, self._params.U_max, self._params.r_max])
        return lbu, ubu, lbx, ubx

    @property
    def params(self):
        "Returns the parameters of the considered model."
        return self._params

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u


class CyberShip2(IModel):
    """Implements a modified version of the 3DOF nonlinear maneuvering model for the Cybership 2 vessel with actuator model and consideration of wind, waves and currents (added although the model is not identified considering these disturbance effects). The disturbance effects can be toggled on/off. The model is on the form

    eta_dot = Rpsi(eta) * nu
    M_rb * nu_dot + C_rb(nu) * nu + M_A * nu_v_r_dot + C_A(nu_r) * nu_r + D(nu_r) * nu_r = tau + tau_wind + tau_wave

    with tau = B f_c(u, nu_r)

    where the input vector is u = [n_1, n_2, n_3, delta_1, delta_2]^T. State vector is xs = [eta, nu]^T, where eta = [x, y, psi]^T and nu = [u, v, r]^T.

    See "A Nonlinear Ship Manoeuvering Model: Identification and adaptive control with experiments for a model ship" https://www.mic-journal.no/ABS/MIC-2004-1-1/ for more details.

    NOTE: When using Euler`s method, keep the time step small enough (e.g. around 0.1 or less) to ensure numerical stability.
    """

    _n_x: int = 6
    _n_u: int = 5

    def __init__(self) -> None:
        self._params: CyberShip2Params = CyberShip2Params()

    def nonlinear_damping_matrix(self, nu_r: np.ndarray) -> np.ndarray:
        """Computes the nonlinear damping matrix D_nl(nu_r)

        Args:
            nu (np.ndarray): Velocity vector nu = [u, v, r]^T or relative velocity vector nu_r = [u_r, v_r, r]^T

        Returns:
            np.ndarray: Nonlinear damping matrix D_nl(nu)
        """

        d_11 = self._params.X_uu * np.abs(nu_r[0]) + self._params.X_uuu * nu_r[0] ** 2
        d_22 = self._params.Y_vv * np.abs(nu_r[1]) + self._params.Y_rv * np.abs(nu_r[2])
        d_23 = self._params.Y_vr * np.abs(nu_r[1]) + self._params.Y_rr * np.abs(nu_r[2])
        d_32 = self._params.N_vv * np.abs(nu_r[1]) + self._params.N_rv * np.abs(nu_r[2])
        d_33 = self._params.N_vr * np.abs(nu_r[1]) + self._params.N_rr * np.abs(nu_r[2])
        return np.array([[-d_11, 0.0, 0.0], [0.0, -d_22, -d_23], [0.0, -d_32, -d_33]])

    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State xs = [eta, nu]^T (for the real ship, must be scaled to be used with the CS2-model)
            u (np.ndarray):  Input equal to [n_1, n_2, n_3, delta_1, delta_2]^T consisting of the two main propeller speeds, bow propeller speed and main propeller rudder angles.

        Returns:
            np.ndarray: New state xs.
        """
        if u.size != self._n_u:
            raise ValueError("Dimension of input array should be 5!")
        if xs.size != self._n_x:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        eta[2] = mf.wrap_angle_to_pmpi(eta[2])

        nu = xs[3:6]
        nu_c = self._currents

        tau = self.input_to_generalized_forces(u, nu)

        Minv = np.linalg.inv(self._params.M_rb + self._params.M_a)

        ode_fun = np.zeros(6)
        ode_fun[0:3] = mf.Rmtrx(eta[2]) @ nu
        ode_fun[3:6] = Minv * @ (-Cvv - Dvv + tau)

        if np.abs(nu[0]) < 0.1:
            ode_fun[2] = 0.0

        return ode_fun

    def input_to_generalized_forces(self, u: np.ndarray, nu_r: np.ndarray) -> np.ndarray:
        """Computes generalized forces tau from the input u and relative velocity nu_r.

        Args:
            u (np.ndarray): Input equal to [n_1, n_2, n_3, delta_1, delta_2]^T consisting of the two main propeller speeds, bow propeller speed and main propeller rudder angles.
            nu_r (np.ndarray): Relative velocity vector nu_r = [u_r, v_r, r]^T

        Returns:
            np.ndarray: Generalized forces tau = [X, Y, N]^T
        """
        T_1 = self.main_propeller_speed_to_thrust_force(u[0], nu_r[0])
        T_2 = self.main_propeller_speed_to_thrust_force(u[1], nu_r[0])
        T_3 = self.bow_propeller_speed_to_thrust_force(u[2], nu_r[0])
        L_1 = self.main_propeller_rudder_angle_to_lift_force(u[0], nu_r[0])
        L_2 = self.main_propeller_rudder_angle_to_lift_force(u[1], nu_r[0])
        tau = self._params.B @ np.array([T_1, T_2, T_3, L_1, L_2])
        return tau

    def main_propeller_speed_to_thrust_force(self, n: float, u_r: float) -> float:
        """ Computes the thrust force T from the main propeller speed n and relative surge speed using the actuator model.

        Args:
            n (float): Main propeller speed n
            u_r (float): Relative surge speed u_r

        Returns:
            float: Thrust force T
        """
        n_top = np.max(0.0, u_r * self._params.T_nu_plus / self._params.T_nn_plus)
        n_bot = np.min(0.0, u_r * self._params.T_nu_minus / self._params.T_nn_minus)
        T = 0.0
        if n >= n_top:
            T = self._params.T_nn_plus * np.abs(n) * n - self._params.T_nu_plus * np.abs(n) * u_r
        elif n <= n_bot:
            T = self._params.T_nn_minus * np.abs(n) * n - self._params.T_nu_minus * np.abs(n) * u_r
        return T

    def bow_propeller_speed_to_thrust_force(self, n: float) -> float:
        """ Computes the thrust force T from the bow propeller speed n using the actuator model.

        Args:
            n (float): Propeller speed n

        Returns:
            float: Thrust force T
        """
        return self._params.T_n3n3 * np.abs(n) * n

    def main_propeller_rudder_angle_to_lift_force(self, delta: float, u_r: float) -> float:
        """ Computes the lift force L from the main propeller rudder angle delta and relative surge speed using the actuator model.

        Args:
            delta (float): Main propeller rudder angle delta
            u_r (float): Relative surge speed u_r

        Returns:
            float: Lift force L
        """
        u_rud = u_r
        if u_r >= 0.0:
            u_rud = u_r + self._params.k_u * (np.sqrt(np.max(0.0, 8.0 * T / (np.pi * self._params.rho * self._params.d_rud) + u_r**2)) - u_r)

        if u_rud >= 0.0:
            L = (self._params.L_delta_plus * delta - self._params.L_ddelta_plus * np.abs(delta) * delta) * np.abs(u_rud) * u_rud
        else:
            L = (self._params.L_delta_minus * delta - self._params.L_ddelta_minus * np.abs(delta) * delta) * np.abs(u_rud) * u_rud
        return L




    def bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the bounds on the forces the actuators can produce on the ship. """
        lamb = self._params.scaling_factor
        lbu = np.array([self._params.Fx_limits[0] * lamb * lamb * lamb, self._params.Fy_limits[0] * lamb * lamb * lamb, self._params.N_limits[0] * lamb * lamb * lamb * lamb * lamb])
        ubu = np.array([self._params.Fx_limits[1], self._params.Fy_limits[1], self._params.N_limits[1]])
        lbx = np.array([-np.inf, -np.inf, -np.inf, -self._params.U_max, -self._params.U_max, -np.inf])
        ubx = np.array([np.inf, np.inf, np.inf, self._params.U_max, self._params.U_max, -np.inf])
        return lbu, ubu, lbx, ubx

    @property
    def params(self):
        "Returns the parameters of the considered model."
        return self._params

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u
