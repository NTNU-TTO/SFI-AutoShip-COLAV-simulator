"""
    controllers.py

    Summary:
        Contains class definitions for various control strategies.
        Every controller must adhere to the interface IController.

    Author: Trym Tengesdal
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import colav_simulator.utils.math_functions as mf
import numpy as np


@dataclass
class MIMOPIDPars:
    "Parameters for a Proportional-Integral-Derivative controller."
    wn: np.ndarray = np.diag([0.3, 0.3, 0.15])
    zeta: np.ndarray = np.diag([1.0, 1.0, 1.0])
    K_p: np.ndarray = np.diag([100.0, 100.0, 100.0 * math.pi])
    K_d: np.ndarray = np.diag([100.0, 100.0, 100 * math.pi])
    K_i: np.ndarray = np.diag([0.0, 0.0, 0.0])
    eta_diff_max: np.ndarray = np.zeros(3)


@dataclass
class Config:
    """Configuration class for managing controller parameters."""

    name: str
    pid: Optional[MIMOPIDPars]


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
    """Implements a Proportional-Integral-Derivative controller

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

        np.diag([0.3, 0.1, 0.3])  # PID pole placement
        self.zeta = np.diag([1.0, 1.0, 1.0])

    def _reset_integrator(self):
        self._eta_diff_int = np.zeros(3)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, dt: float, model) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d, eta_dot_d]^T
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
        eta_dot = mf.Rpsi(eta[2]) @ nu

        Mmtrx = model.pars.M_rb + model.pars.M_a
        Dmtrx = mf.Dmtrx(model.pars.D_l, model.pars.D_q, model.pars.D_c, nu)

        K_p, K_d, K_i = pole_placement(Mmtrx, Dmtrx, self._pars.wn, self._pars.zeta)

        eta_d = refs[0:3]
        eta_diff = eta - eta_d
        eta_diff[2] = mf.wrap_angle_diff_to_pmpi(eta[2], eta_d[2])

        eta_dot_d = refs[3:6]
        eta_dot_diff = eta_dot - eta_dot_d

        self._eta_diff_int = mf.sat_vec(self._eta_diff_int + eta_diff * dt, np.zeros(3), self._pars.eta_diff_max)

        # Compute control input in NED frame.
        tau = -K_p @ eta_diff - K_d @ eta_dot_diff - K_i @ self._eta_diff_int
        # Rotate to BODY frame.
        tau = mf.Rpsi(eta[2]) @ tau
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
