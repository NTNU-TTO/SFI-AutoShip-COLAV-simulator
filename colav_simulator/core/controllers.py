"""
    controllers.py

    Summary:
        Contains class definitions for various control strategies.
        Every controller must adhere to the interface IController.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import numpy as np


@dataclass
class MIMOPIDParams:
    "Parameters for a Proportional-Integral-Derivative controller."
    wn: np.ndarray = field(default_factory=lambda: np.diag([0.3, 0.2, 0.35]))
    zeta: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 1.0]))
    eta_diff_max: np.ndarray = np.zeros(3)

    def to_dict(self):
        output_dict = {}
        output_dict["wn"] = self.wn.diagonal().tolist()
        output_dict["zeta"] = self.zeta.diagonal().tolist()
        output_dict["eta_diff_max"] = self.eta_diff_max.tolist()


@dataclass
class SHPIDParams:
    "Parameters for a PID controller for surge, sway + heading control with feedback linearization of the mass+coriolis+damping."
    K_p: np.ndarray
    K_d: np.ndarray
    K_i: np.ndarray
    z_diff_max: np.ndarray
    V: np.ndarray

    def __init__(
        self,
        K_p: np.ndarray = field(default_factory=lambda: np.diag([5.0, 1.3, 1.4])),
        K_d: np.ndarray = field(default_factory=lambda: np.diag([0.0, 5.0, 15.0])),
        K_i: np.ndarray = field(default_factory=lambda: np.diag([0.25, 0.1, 0.1])),
        z_diff_max: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 15.0 * np.pi / 180.0])),
    ) -> None:
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i
        self.z_diff_max = z_diff_max
        self.V = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

    @classmethod
    def from_dict(cls, config_dict: dict):
        params = SHPIDParams(
            K_p=np.diag(config_dict["K_p"]), K_d=np.diag(config_dict["K_d"]), K_i=np.diag(config_dict["K_i"]), z_diff_max=np.array(config_dict["z_diff_max"])
        )
        params.z_diff_max[2] = np.deg2rad(params.z_diff_max[2])
        return params

    def to_dict(self):
        output_dict = {}
        output_dict["K_p"] = self.K_p.diagonal().tolist()
        output_dict["K_d"] = self.K_d.diagonal().tolist()
        output_dict["K_i"] = self.K_i.diagonal().tolist()
        self.z_diff_max[2] = np.rad2deg(self.z_diff_max[2])
        output_dict["z_diff_max"] = self.z_diff_max.tolist()


@dataclass
class FLSHParams:
    "Parameters for the feedback linearizing surge-heading controller."
    K_p_u: float = 2.0
    K_i_u: float = 0.1
    K_p_psi: float = 2.5
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
class PassThroughInputsParams:
    "Parameters for the pass-through inputs controller."

    rudder_propeller_mapping: bool = True

    def to_dict(self):
        return asdict(self)

    def from_dict(self, config_dict: dict):
        return PassThroughInputsParams(**config_dict)


@dataclass
class Config:
    """Configuration class for managing controller parameters."""

    pid: Optional[MIMOPIDParams] = None
    flsh: Optional[FLSHParams] = None
    shpid: Optional[SHPIDParams] = None
    pass_through_cs: Optional[bool] = True
    pass_through_inputs: Optional[PassThroughInputsParams] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(pid=None, flsh=None, pass_through_cs=None, pass_through_inputs=None)
        if "pid" in config_dict:
            config.pid = MIMOPIDParams(
                wn=np.diag(config_dict["pid"]["wn"]),
                zeta=np.diag(config_dict["pid"]["zeta"]),
                eta_diff_max=np.array(config_dict["pid"]["eta_diff_max"]),
            )

        if "flsh" in config_dict:
            config.flsh = cp.convert_settings_dict_to_dataclass(FLSHParams, config_dict["flsh"])
            config.flsh.max_psi_error_int = np.deg2rad(config.flsh.max_psi_error_int)
            config.flsh.psi_error_int_threshold = np.deg2rad(config.flsh.psi_error_int_threshold)

        if "shpid" in config_dict:
            config.shpid = cp.convert_settings_dict_to_dataclass(SHPIDParams, config_dict["shpid"])

        if "pass_through_cs" in config_dict:
            config.pass_through_cs = True

        if "pass_through_inputs" in config_dict:
            config.pass_through_inputs = PassThroughInputsParams(**config_dict["pass_through_inputs"])

        return config

    def to_dict(self) -> dict:
        config_dict = {}
        if self.pid is not None:
            config_dict["pid"] = self.pid.to_dict()

        if self.flsh is not None:
            config_dict["flsh"] = self.flsh.to_dict()

        if self.shpid is not None:
            config_dict["shpid"] = self.shpid.to_dict()

        if self.pass_through_cs is not None:
            config_dict["pass_through_cs"] = ""

        if self.pass_through_inputs is not None:
            config_dict["pass_through_inputs"] = self.pass_through_inputs.to_dict()

        return config_dict


class IController(ABC):
    @abstractmethod
    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Computes inputs using the specific controller strategy.

        References should be of dimension 9 to be able to include pose (typically in NED),
        pose derivative (NED or BODY), and pose double derivative (NED or BODY).
        """


class ControllerBuilder:
    @classmethod
    def construct_controller(cls, model_params: Any, config: Optional[Config] = None) -> IController:
        """Builds a controller from the configuration

        Args:
            model_params (Any): Model parameters used by the controller.
            config (Optional[controllers.Config]): Model configuration. Defaults to None.

        Returns:
            Model: Model as specified by the configuration, e.g. a MIMOPID controller.
        """
        if config and config.pid:
            return MIMOPID(model_params, config.pid)
        elif config and config.flsh:
            return FLSH(model_params, config.flsh)
        elif config and config.shpid:
            return SHPID(model_params, config.shpid)
        elif config and config.pass_through_cs:
            return PassThroughCS()
        elif config and config.pass_through_inputs:
            return PassThroughInputs(config.pass_through_inputs)
        else:
            return PassThroughCS()


@dataclass
class PassThroughCS(IController):
    """This controller just feeds through the course (heading) and forward speed references."""

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Takes out relevant parts of references as inputs directly

        Args:
            refs (np.ndarray): Desired/references = [x, y, psi, u, v, r, ax, ay, rdot]
            xs (np.ndarray): State xs
            dt (float): Time step

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
class PassThroughInputs(IController):
    """This controller feeds through the force inputs u = [X, Y, N] or u = [X, Y], depending on the model and mapping chosen.

    If the model is a rudder-propeller model, the references are [X, Y], and the mapping is [X, Y] -> [X, Y, -Y * l_r],
    where l_r is the distance from the center of rotation to the rudder.

    Otherwise, the mapping is [X, Y, N] -> [X, Y, N]."""

    def __init__(self, model_params, params: Optional[PassThroughInputsParams] = None) -> None:
        self._model_params = model_params
        if params is None:
            params = PassThroughInputsParams()
        else:
            self.params = params

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Takes out relevant parts of references as inputs directly

        Args:
            refs (np.ndarray): Force inputs = [X, Y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] or [X, Y, -Y * l_r, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] depending on the model.
            xs (np.ndarray): State xs
            dt (float): Time step

        Returns:
            np.ndarray: 3 x 1 inputs u = [X, Y, N]^T
        """

        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        if self.params.rudder_propeller_mapping:
            return np.array([refs[0], refs[1], -refs[1] * self._model_params.l_r])
        else:
            return np.array([refs[0], refs[1], refs[2]])


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

    def __init__(self, model_params, params: Optional[MIMOPIDParams] = None) -> None:
        self._model_params = model_params
        if params is not None:
            self._params: MIMOPIDParams = params
        else:
            self._params = MIMOPIDParams()

        self._eta_diff_int: np.ndarray = np.zeros(3)

    def _reset_integrator(self):
        self._eta_diff_int = np.zeros(3)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d, eta_dot_d, eta_ddot_d]^T
            xs (np.ndarray): State xs = [eta, nu]^T
            dt (float): Time step

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

        Mmtrx = self._model_params.M_rb + self._model_params.M_a
        Dmtrx = mf.Dmtrx(self._model_params.D_l, self._model_params.D_q, self._model_params.D_c, nu)

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
    K_i = (1.0 / 10.0) * wn @ K_p
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

    def __init__(self, model_params, params: Optional[FLSHParams] = None) -> None:
        self._model_params = model_params
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

        if abs(psi_error) <= self._params.psi_error_int_threshold:
            self._psi_error_int = mf.unwrap_angle(self._psi_error_int, psi_error * dt)

        if abs(self._psi_error_int) > self._params.max_speed_error_int:
            self._psi_error_int = mf.unwrap_angle(self._psi_error_int, -psi_error * dt)

        self._psi_error_int = mf.wrap_angle_to_pmpi(self._psi_error_int)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Computes inputs based on the proposed control law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d^T, eta_dot_d^T, eta_ddot_d^T]^T
            (equal to [0, 0, psi_d, U_d, 0, 0, 0, 0, 0]^T in case of LOSGuidance)
            xs (np.ndarray): State xs = [eta^T, nu^T]^T
            dt (float): Time step

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")

        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        nu = xs[3:]

        # assume eta_dot_d is in BODY frame
        u_d = refs[3]
        psi_d: float = mf.wrap_angle_to_pmpi(refs[2])
        psi_d_unwrapped = mf.unwrap_angle(self._psi_d_prev, psi_d)
        r_d = mf.sat(refs[5], -self._model_params.r_max, self._model_params.r_max)

        psi: float = mf.wrap_angle_to_pmpi(eta[2])
        psi_unwrapped = mf.unwrap_angle(self._psi_prev, psi)

        self._psi_d_prev = psi_d
        self._psi_prev = eta[2]

        Mmtrx = self._model_params.M_rb + self._model_params.M_a
        Cvv = mf.Cmtrx(Mmtrx, nu) @ nu
        Dvv = mf.Dmtrx(self._model_params.D_l, self._model_params.D_q, self._model_params.D_c, nu) @ nu

        speed_error = u_d - nu[0]
        psi_error: float = mf.wrap_angle_diff_to_pmpi(psi_d_unwrapped, psi_unwrapped)
        self.update_integrators(speed_error, psi_error, dt)

        # print(f"speed error int: {self._speed_error_int} | speed error: {speed_error} | psi error int: {self._psi_error_int} | psi error: {psi_error}")
        # if abs(speed_error) < 0.01:
        #     print(f"speed error int: {self._speed_error_int} | speed error: {speed_error}")

        Fx = Cvv[0] + Dvv[0] + Mmtrx[0, 0] * (self._params.K_p_u * speed_error + self._params.K_i_u * self._speed_error_int)
        Fy = -(Mmtrx[2, 2] / self._model_params.l_r) * (
            self._params.K_p_psi * psi_error + self._params.K_d_psi * (r_d - nu[2]) + self._params.K_i_psi * self._psi_error_int
        )

        tau = np.array([float(Fx), float(Fy), float(-Fy * self._model_params.l_r)])

        return tau


class SHPID(IController):
    """Implements a surge-heading PID-controller with feedback linearization for the mass+coriolis+damping terms in the (RVGunnerus) model."""

    def __init__(self, model_params, params: Optional[SHPIDParams] = None) -> None:
        self._model_params = model_params
        if params is not None:
            self._params: SHPIDParams = params
        else:
            self._params = SHPIDParams()
        self._z_diff_int: np.ndarray = np.zeros(3)
        self._psi_d_prev: float = 0.0
        self._psi_prev: float = 0.0
        self._n_u: int = 5

    def _reset_integrator(self):
        self._z_diff_int = np.zeros(3)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d, eta_dot_d, eta_ddot_d]^T. Could in practice be = [-, -, psi_d, u_d, -, -, -, -, -]^T if LOS-guidance is used.
            xs (np.ndarray): State xs = [eta, nu]^T
            dt (float): Time step

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) != 9:
            raise ValueError("Dimension of reference array should be equal to 9!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        nu = xs[3:]

        psi_d: float = mf.wrap_angle_to_pmpi(refs[2])
        psi_d_unwrapped = mf.unwrap_angle(self._psi_d_prev, psi_d)

        psi: float = mf.wrap_angle_to_pmpi(xs[2])
        psi_unwrapped = mf.unwrap_angle(self._psi_prev, psi)

        self._psi_d_prev = psi_d
        self._psi_prev = psi

        z = self._params.V @ xs
        z[2] = psi_unwrapped

        # Reference = [u, 0.0, psi]
        z_d = np.array([refs[3], 0.0, psi_d_unwrapped])

        psi_error: float = mf.wrap_angle_diff_to_pmpi(psi_d_unwrapped, psi_unwrapped)
        z_diff = z_d - z
        z_diff[2] = psi_error

        self.update_integrator(z_diff, dt)

        M = self._model_params.M_rb + self._model_params.M_a
        C_RB = mf.coriolis_matrix_rigid_body(self._model_params.M_rb, nu)
        C_A = mf.coriolis_matrix_added_mass(self._model_params.M_a, nu)
        Cvv = C_RB @ nu + C_A @ nu
        Dvv = (self._model_params.D_l + self._model_params.D_u * abs(nu[0]) + self._model_params.D_v * abs(nu[1]) + self._model_params.D_r * abs(nu[2])) @ nu

        tau = Cvv + Dvv + M @ (self._params.K_p @ z_diff - self._params.K_d @ nu + self._params.K_i @ self._z_diff_int)

        tau[0] = mf.sat(tau[0], self._model_params.Fx_limits[0], self._model_params.Fx_limits[1])
        tau[1] = mf.sat(tau[1], self._model_params.Fy_limits[0], self._model_params.Fy_limits[1])
        tau[2] = mf.sat(tau[2], self._model_params.N_limits[0], self._model_params.N_limits[1])

        u = tau

        return u

    def update_integrator(self, z_diff: np.ndarray, dt: float) -> None:
        """Updates the integrator state for the PID controller.

        Args:
            z_diff (np.ndarray): Difference between desired and actual state
            dt (float): Time step
        """
        self._z_diff_int = mf.sat(self._z_diff_int + z_diff * dt, -self._params.z_diff_max, self._params.z_diff_max)

    # def thrust_allocation_two_main_propellers_rudders_and_bow_thruster(self, tau: np.ndarray, nu_r: np.ndarray) -> np.ndarray:
    #     """Computes the propeller speeds n and rudder angles delta from the generalized forces tau = [Fx, Fy, N]^T.

    #     Args:
    #         tau (np.ndarray): Generalized forces tau = [Fx, Fy, N]^T
    #         nu_r (np.ndarray): Relative velocity vector nu_r = [u_r, v_r, r]^T

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: Propeller speeds n and rudder angles delta
    #     """
    #     W = np.eye(self._n_u)
    #     B_pseudo_inv = np.linalg.inv(W) @ self._model_params.B.T @ np.linalg.inv(self._model_params.B @ np.linalg.inv(W) @ self._model_params.B.T)
    #     f = B_pseudo_inv @ tau
    #     n_1 = self.main_propeller_thrust_force_to_speed(f[0], nu_r[0])
    #     n_2 = self.main_propeller_thrust_force_to_speed(f[1], nu_r[0])
    #     n_3 = self.bow_thrust_force_to_propeller_speed(f[2])
    #     delta_1 = self.main_propeller_lift_force_to_rudder_angle(f[3], f[0], nu_r[0])
    #     delta_2 = self.main_propeller_lift_force_to_rudder_angle(f[4], f[1], nu_r[0])

    #     n_1 = mf.sat(n_1, 0.0, self._model_params.max_propeller_speed)
    #     n_2 = mf.sat(n_2, 0.0, self._model_params.max_propeller_speed)
    #     n_3 = mf.sat(n_3, 0.0, self._model_params.max_propeller_speed)
    #     delta_1 = mf.sat(delta_1, -self._model_params.max_rudder_angle, self._model_params.max_rudder_angle)
    #     delta_2 = mf.sat(delta_2, -self._model_params.max_rudder_angle, self._model_params.max_rudder_angle)

    #     return np.array([n_1, n_2, n_3, delta_1, delta_2])

    # def main_propeller_thrust_force_to_speed(self, T: float, u_r: float) -> float:
    #     """Computes the main propeller speed n from the thrust force T and relative velocity u_r.

    #     Args:
    #         T (float): Thrust force T
    #         u_r (float): Relative surge speed u_r

    #     Returns:
    #         float: Main propeller speed n
    #     """
    #     if T > 0.0:
    #         n = u_r * self._model_params.T_nu_plus / (2 * self._model_params.T_nn_plus) + np.sqrt(
    #             (self._model_params.T_nu_plus * u_r) ** 2 + 4 * self._model_params.T_nn_plus * T
    #         ) / (2 * self._model_params.T_nn_plus)
    #     elif T == 0.0:
    #         n = 0.0
    #     else:
    #         n = u_r * self._model_params.T_nu_minus / (2 * self._model_params.T_nn_minus) - np.sqrt(
    #             (self._model_params.T_nu_minus * u_r) ** 2 - 4 * self._model_params.T_nn_minus * T
    #         ) / (2 * self._model_params.T_nu_minus)
    #     return n

    # def bow_thrust_force_to_propeller_speed(self, T: float) -> float:
    #     """Computes the bow thruster speed n from the thrust force T and relative surge speed u_r.

    #     Args:
    #         T (float): Thrust force T

    #     Returns:
    #         float: Bow thruster speed n
    #     """
    #     n = np.sign(T) * np.sqrt(self._model_params.T_n3n3 * abs(T)) / self._model_params.T_n3n3
    #     return n

    # def main_propeller_lift_force_to_rudder_angle(self, L: float, T: float, u_r: float) -> float:
    #     """Computes the rudder angle delta from the lift force L and relative surge speed u_r.

    #     Args:
    #         L (float): Lift force L
    #         T (float): Thrust force T for propeller at the same location
    #         u_r (float): Relative surge speed u_r

    #     Returns:
    #         float: Rudder angle delta
    #     """
    #     # Fluid velocity at rudder surface
    #     u_rud = u_r
    #     if u_r >= 0.0:
    #         u_rud = u_r + self._model_params.k_u * (np.sqrt(max(0.0, 8.0 * T / (np.pi * self._model_params.rho * self._model_params.d_rud**2) + u_r**2)) - u_r)

    #     epsilon = 0.0001
    #     delta = 0.0
    #     if u_rud >= epsilon:
    #         delta = (np.sign(L) / (2.0 * self._model_params.L_ddelta_plus)) * (
    #             self._model_params.L_delta_plus
    #             + np.sqrt((self._model_params.L_delta_plus * u_rud**2) ** 2 + 4 * self._model_params.L_ddelta_plus * u_rud**2 * abs(L)) / (u_rud**2)
    #         )
    #     elif abs(u_rud) < epsilon:
    #         delta = 0.0
    #     elif u_rud <= -epsilon:
    #         delta = -(np.sign(L) / (2.0 * self._model_params.L_ddelta_minus)) * (
    #             self._model_params.L_delta_minus
    #             - np.sqrt((self._model_params.L_delta_minus * u_rud**2) ** 2 - 4 * self._model_params.L_ddelta_minus * u_rud**2 * abs(L)) / (u_rud**2)
    #         )
    #     return delta
