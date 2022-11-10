"""
    controllers.py

    Summary:
        Contains class definitions for various control strategies.
        Every controller must adhere to the interface IController.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import numpy as np


@dataclass
class MIMOPIDPars:
    "Parameters for a Proportional-Integral-Derivative controller."
    wn: np.ndarray = np.diag([0.3, 0.3, 0.3])
    zeta: np.ndarray = np.diag([1.0, 1.0, 1.0])
    eta_diff_max: np.ndarray = np.zeros(3)


@dataclass
class FLSHPars:
    "Parameters for the feedback linearizing surge-heading controller."
    K_p_u: float = 5.0
    K_p_psi: float = 8.0
    K_d_psi: float = 10.0


@dataclass
class Config:
    """Configuration class for managing controller parameters."""

    pid: Optional[MIMOPIDPars] = None
    flsh: Optional[FLSHPars] = None
    pass_through: Optional[Any] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        if "pid" in config_dict:
            cls.pid = MIMOPIDPars(
                wn=np.diag(config_dict["pid"]["wn"]),
                zeta=np.diag(config_dict["pid"]["zeta"]),
                eta_diff_max=np.array(config_dict["pid"]["eta_diff_max"]),
            )

        if "flsh" in config_dict:
            cls.flsh = cp.convert_settings_dict_to_dataclass(FLSHPars, config_dict["flsh"])

        if "pass_through" in config_dict:
            cls.pass_through = True

        return cls


class IController(ABC):
    """The InterfaceController class is abstract and used to force
    the implementation of the below methods for all subclasses (controllers),
    to comply with the model interface.
    """

    @abstractmethod
    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        "Computes inputs using the specific controller strategy"


@dataclass
class PassThrough(IController):
    """This controller just feeds through the references"""

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        """Takes inputs directly as references.

        Args:
            refs (np.ndarray): Desired/references [U_d, chi_d]^T
            xs (np.ndarray): State xs = [x, y, chi, U]^T
            dt (float): Time step
            model (IModel): Model object to fetch parameters from, optional in many cases.

        Returns:
            np.ndarray: Inputs u = refs to pass through.
        """
        return refs


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

    _eta_diff_int: np.ndarray
    _pars: MIMOPIDPars

    def __init__(self, config: Optional[Config] = None) -> None:
        self._eta_diff_int = np.zeros(3)
        if config and config.pid is not None:
            self._pars = config.pid
        else:
            self._pars = MIMOPIDPars()

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
        if len(refs) < 6:
            raise ValueError("Dimension of reference array should be 6 or more!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        nu = xs[3:]

        R_n_b = mf.Rpsi(eta[2])
        eta_dot = R_n_b @ nu

        Mmtrx = model.pars.M_rb + model.pars.M_a
        Dmtrx = mf.Dmtrx(model.pars.D_l, model.pars.D_q, model.pars.D_c, nu)

        K_p, K_d, K_i = pole_placement(Mmtrx, Dmtrx, self._pars.wn, self._pars.zeta)

        eta_d = refs[0:3]
        eta_diff = eta - eta_d
        eta_diff[2] = mf.wrap_angle_diff_to_pmpi(eta[2], eta_d[2])

        eta_dot_d = refs[3:6]
        eta_dot_diff = eta_dot - eta_dot_d

        self._eta_diff_int = mf.sat(self._eta_diff_int + eta_diff * dt, np.zeros(3), self._pars.eta_diff_max)

        tau = -K_p @ eta_diff - K_d @ eta_dot_diff - K_i @ self._eta_diff_int
        tau = R_n_b.T @ tau

        if eta[0] > 40.0:
            print(
                f"x_diff = {-eta_diff[0]} | y_diff = {-eta_diff[1]} psi_diff: {-eta_diff[2]} | r_d: {refs[5]} | r: {nu[2]}"
            )

        return tau


def pole_placement(
    Mmtrx: np.ndarray, Dmtrx: np.ndarray, wn: np.ndarray, zeta
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Fx = (C(nu) * nu)[0] + (D(nu) * nu)[0] + M[0, 0] * K_p,u * (u_d - u)
    Fy = (M[2, 2] / l_r) * ( K_p,psi * (psi_d - psi) - K_d,psi * r )

    for the system

    eta_dot = J_Theta(eta) * nu
    M * nu_dot + C(nu) * nu + D(nu) * nu = tau

    with a rudder placed l_r units away from CG, such that tau = [Fx, Fy, Fy * l_r]^T.

    Here, J_Theta(eta) = R(eta) for the 3DOF case with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.
    """

    _eta_diff_int: np.ndarray
    _pars: FLSHPars

    def __init__(self, config: Optional[Config] = None) -> None:
        self._eta_diff_int = np.zeros(3)
        if config and config.flsh is not None:
            self._pars = config.flsh
        else:
            self._pars = FLSHPars()

    def _reset_integrator(self):
        self._eta_diff_int = np.zeros(3)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [U_d, psi_d]^T or [eta_d, eta_dot_d, eta_ddot_d]^T
            xs (np.ndarray): State xs = [eta, nu]^T
            dt (float): Time step
            model (IModel): Model object to fetch parameters from.

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) == 2:
            u_d, psi_d, r_d = refs[0], refs[1], 0.0
        elif len(refs) >= 6:
            u_d, psi_d, r_d = (
                mf.sat(np.sqrt(refs[3] ** 2 + refs[4] ** 2), 0.0, model.pars.U_max),
                refs[2],
                mf.sat(refs[5], -model.pars.r_max, model.pars.r_max),
            )
        else:
            raise ValueError("Dimension of reference array should be equal to 2 or >=6!")

        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")

        eta = xs[0:3]
        nu = xs[3:]

        Mmtrx = model.pars.M_rb + model.pars.M_a
        Cvv = mf.Cmtrx(Mmtrx, nu) @ nu
        Dvv = mf.Dmtrx(model.pars.D_l, model.pars.D_q, model.pars.D_c, nu) @ nu

        psi_diff = mf.wrap_angle_diff_to_pmpi(psi_d, eta[2])

        Fx = Cvv[0] + Dvv[0] + Mmtrx[0, 0] * self._pars.K_p_u * (u_d - nu[0])
        Fy = (Mmtrx[2, 2] / model.pars.l_r) * (self._pars.K_p_psi * psi_diff + self._pars.K_d_psi * (r_d - nu[2]))

        tau = np.array([Fx, Fy, Fy * model.pars.l_r])

        return tau
