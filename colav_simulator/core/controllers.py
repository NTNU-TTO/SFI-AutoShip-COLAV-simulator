"""
    controllers.py

    Summary:
        Contains class definitions for various control strategies.
        Every controller must adhere to the interface IController.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import numpy as np


@dataclass
class MIMOPIDParams:
    "Parameters for a Proportional-Integral-Derivative controller."
    wn: np.ndarray = np.diag([0.3, 0.2, 0.35])
    zeta: np.ndarray = np.diag([1.0, 1.0, 1.0])
    eta_diff_max: np.ndarray = np.zeros(3)

    def to_dict(self):
        output_dict = {}
        output_dict["wn"] = self.wn.diagonal().tolist()
        output_dict["zeta"] = self.zeta.diagonal().tolist()
        output_dict["eta_diff_max"] = self.eta_diff_max.tolist()


@dataclass
class FLSHParams:
    "Parameters for the feedback linearizing surge-heading controller."
    K_p_u: float = 3.0
    K_i_u: float = 0.1
    K_p_psi: float = 3.0
    K_d_psi: float = 1.75
    K_i_psi: float = 0.003
    max_speed_error_int: float = 2.0
    speed_error_int_threshold: float = 1.0
    max_psi_error_int: float = 50.0 * np.pi / 180.0
    psi_error_int_threshold: float = 15.0 * np.pi / 180.0

    def to_dict(self):
        output = asdict(self)
        output["max_psi_error_int"] = np.rad2deg(output["max_psi_error_int"])
        output["psi_error_int_threshold"] = np.rad2deg(output["psi_error_int_threshold"])
        return output


@dataclass
class Config:
    """Configuration class for managing controller parameters."""

    pid: Optional[MIMOPIDParams] = None
    flsh: Optional[FLSHParams] = None
    pass_through_cs: Optional[bool] = True

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "pid" in config_dict:
            config.pid = MIMOPIDParams(
                wn=np.diag(config_dict["pid"]["wn"]),
                zeta=np.diag(config_dict["pid"]["zeta"]),
                eta_diff_max=np.array(config_dict["pid"]["eta_diff_max"]),
            )
            config.flsh = None
            config.pass_through_cs = None

        if "flsh" in config_dict:
            config.flsh = cp.convert_settings_dict_to_dataclass(FLSHParams, config_dict["flsh"])
            config.flsh.max_psi_error_int = np.deg2rad(config.flsh.max_psi_error_int)
            config.flsh.psi_error_int_threshold = np.deg2rad(config.flsh.psi_error_int_threshold)
            config.pid = None
            config.pass_through_cs = None

        if "pass_through_cs" in config_dict:
            config.pass_through_cs = True
            config.pid = None
            config.flsh = None

        return config

    def to_dict(self) -> dict:
        config_dict = {}
        if self.pid is not None:
            config_dict["pid"] = self.pid.to_dict()

        if self.flsh is not None:
            config_dict["flsh"] = self.flsh.to_dict()

        if self.pass_through_cs is not None:
            config_dict["pass_through_cs"] = ""

        return config_dict


class IController(ABC):
    @abstractmethod
    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        """Computes inputs using the specific controller strategy.

        References should be of dimension 9 to include pose (typically in NED),
        pose derivative (NED or BODY), and pose double derivative (NED or BODY).
        """


class ControllerBuilder:
    @classmethod
    def construct_controller(cls, config: Optional[Config] = None) -> IController:
        """Builds a controller from the configuration

        Args:
            config (Optional[controllers.Config]): Model configuration. Defaults to None.

        Returns:
            Model: Model as specified by the configuration, e.g. a MIMOPID controller.
        """
        if config and config.pid:
            return MIMOPID(config.pid)
        elif config and config.flsh:
            return FLSH(config.flsh)
        elif config and config.pass_through_cs:
            return PassThroughCS()
        else:
            return PassThroughCS()


@dataclass
class PassThroughCS(IController):
    """This controller just feeds through the course (heading) and forward speed references."""

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        """Takes out relevant parts of references as inputs directly

        Args:
            refs (np.ndarray): Desired/references = [x, y, psi, u, v, r, ax, ay, rdot]
            xs (np.ndarray): State xs
            dt (float): Time step
            model (IModel): Model object to fetch parameters from, optional in many cases.

        Returns:
            np.ndarray: 3 x 1 inputs u = [chi_d, U_d, 0.0]^T, where chi_d and U_d are the desired course and speed over ground
        """

        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        # Course taken directly as heading reference.
        return np.array([refs[2], np.sqrt(refs[3] ** 2 + refs[4] ** 2), 0.0])


@dataclass
class MIMOPID(IController):
    """Implements a multiple input multiple output (MIMO) Proportional-Integral-Derivative (PID) controller

    tau = J_Theta(eta)^T *
          (-K_p (eta_d - eta)
           -K_d (eta_dot - eta_dot_d)
           -K_i integral_{0}^{t} (eta(l) - eta_d(l)) dl)

    for a system

    eta_dot = J_Theta(eta) * nu
    M * nu_dot + C(nu) * nu + D(nu) * nu = tau

    where J_Theta(eta) = R(eta) for the 3DOF case with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.

    See Fossen 2011, Ch. 12.
    """

    def __init__(self, params: Optional[MIMOPIDParams] = None) -> None:
        if params is not None:
            self._params: MIMOPIDParams = params
        else:
            self._params = MIMOPIDParams()

        self._eta_diff_int: np.ndarray = np.zeros(3)

    def _reset_integrator(self):
        self._eta_diff_int = np.zeros(3)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d, eta_dot_d, eta_ddot_d]^T
            xs (np.ndarray): State xs = [eta, nu]^T
            dt (float): Time step
            model (IModel): Model object to fetch parameters from, not used here.

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        nu = xs[3:]

        R_n_b = mf.Rmtrx(eta[2])
        eta_dot = R_n_b @ nu

        Mmtrx = model.params.M_rb + model.params.M_a
        Dmtrx = mf.Dmtrx(model.params.D_l, model.params.D_q, model.params.D_c, nu)

        K_p, K_d, K_i = pole_placement(Mmtrx, Dmtrx, self._params.wn, self._params.zeta)

        eta_d = refs[0:3]
        eta_diff = eta - eta_d
        eta_diff[2] = mf.wrap_angle_diff_to_pmpi(eta[2], eta_d[2])

        eta_dot_d = refs[3:6]
        eta_dot_diff = eta_dot - eta_dot_d

        self._eta_diff_int = mf.sat(self._eta_diff_int + eta_diff * dt, np.zeros(3), self._params.eta_diff_max)

        tau = -K_p @ eta_diff - K_d @ eta_dot_diff - K_i @ self._eta_diff_int
        tau = R_n_b.T @ tau

        return tau


def pole_placement(Mmtrx: np.ndarray, Dmtrx: np.ndarray, wn: np.ndarray, zeta) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dynamic positioning controller pole placement based on Ex. 12.7 in Fossen 2011.

    Args:
        Mmtrx (np.ndarray): Mass matrix
        Dmtrx (np.ndarray): Damping matrix
        wn (np.ndarray): Natural frequency vector [wn_x, wn_y, wn_psi] > 0
        zeta (float): Relative damping ratio > 0
    """
    K_p = wn @ wn @ Mmtrx
    K_d = 2.0 * zeta @ wn @ Mmtrx - Dmtrx
    K_i = (1.0 / 10.0) * wn @ Mmtrx
    return K_p, K_d, K_i


@dataclass
class FLSH(IController):
    """Implements a feedback-linearizing surge-heading (FLSH) controller for a thruster/rudder vessel using

    Fx = (C(nu) * nu)[0] + (D(nu) * nu)[0] + M[0, 0] * (K_p,u * (u_d - u) + int_0^t K_i,u * (u_d - u))
    Fy = (M[2, 2] / l_r) * (K_p,psi * (psi_d - psi) + K_d,psi * (r_d - r) + int_0^t K_i,psi * (psi_d - psi))

    for the system

    eta_dot = J_Theta(eta) * nu
    M * nu_dot + C(nu) * nu + D(nu) * nu = tau

    with a rudder placed l_r units away from CG, such that tau = [Fx, Fy, Fy * l_r]^T.

    Here, J_Theta(eta) = R(eta) for the 3DOF case with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.
    """

    def __init__(self, params: Optional[FLSHParams] = None) -> None:
        if params is not None:
            self._params: FLSHParams = params
        else:
            self._params = FLSHParams()
        self._speed_error_int = 0.0
        self._psi_error_int = 0.0
        self._psi_d_prev = 0.0
        self._psi_prev = 0.0

    def update_integrators(self, speed_error: float, psi_error: float, dt: float) -> None:
        if abs(speed_error) <= self._params.speed_error_int_threshold:
            self._speed_error_int += speed_error * dt

        if abs(self._speed_error_int) > self._params.max_speed_error_int:
            self._speed_error_int -= speed_error * dt

        if abs(speed_error) <= 0.001:
            self._speed_error_int = 0.0

        if abs(psi_error) <= self._params.psi_error_int_threshold:
            self._psi_error_int = mf.unwrap_angle(self._psi_error_int, psi_error * dt)

        if abs(self._psi_error_int) > self._params.max_speed_error_int:
            self._psi_error_int = mf.unwrap_angle(self._psi_error_int, -psi_error * dt)

        if abs(psi_error) <= 0.01 * np.pi / 180.0:
            self._psi_error_int = 0.0
        self._psi_error_int = mf.wrap_angle_to_pmpi(self._psi_error_int)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d^T, eta_dot_d^T, eta_ddot_d^T]^T
            (equal to [0, 0, psi_d, U_d, 0, 0, 0, 0, 0]^T in case of LOSGuidance)
            xs (np.ndarray): State xs = [eta^T, nu^T]^T
            dt (float): Time step
            model (IModel): Model object to fetch parameters from.

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")

        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        nu = xs[3:]

        u_d = refs[3]
        psi_d: float = mf.wrap_angle_to_pmpi(refs[2])
        psi_d_unwrapped = mf.unwrap_angle(self._psi_d_prev, psi_d)
        r_d = mf.sat(refs[5], -model.params.r_max, model.params.r_max)

        psi: float = mf.wrap_angle_to_pmpi(eta[2])
        psi_unwrapped = mf.unwrap_angle(self._psi_prev, psi)

        self._psi_d_prev = psi_d
        self._psi_prev = eta[2]

        Mmtrx = model.params.M_rb + model.params.M_a
        Cvv = mf.Cmtrx(Mmtrx, nu) @ nu
        Dvv = mf.Dmtrx(model.params.D_l, model.params.D_q, model.params.D_c, nu) @ nu

        speed_error = u_d - nu[0]
        psi_error: float = mf.wrap_angle_diff_to_pmpi(psi_d_unwrapped, psi_unwrapped)
        self.update_integrators(speed_error, psi_error, dt)

        # if abs(speed_error) < 0.01:
        #     print(f"speed error int: {self._speed_error_int} | speed error: {speed_error}")

        Fx = Cvv[0] + Dvv[0] + Mmtrx[0, 0] * (self._params.K_p_u * speed_error + self._params.K_i_u * self._speed_error_int)
        Fy = (Mmtrx[2, 2] / -model.params.l_r) * (self._params.K_p_psi * psi_error + self._params.K_d_psi * (r_d - nu[2]) + self._params.K_i_psi * self._psi_error_int)

        tau = np.array([float(Fx), float(Fy), float(-Fy * model.params.l_r)])

        return tau
