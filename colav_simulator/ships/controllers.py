"""
    controllers.py

    Summary:
        Contains class definitions for various control strategies.
        Every controller must adhere to the interface IController.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import colav_simulator.utils.math_functions as mf
import numpy as np


@dataclass
class MIMOPIDPars:
    "Parameters for a Proportional-Integral-Derivative controller."
    K_p: np.ndarray = np.diag([200.0, 200.0, 800.0])
    K_d: np.ndarray = np.diag([700.0, 700.0, 1600.0])
    K_i: np.ndarray = np.diag([0.0, 0.0, 0.0])
    eta_diff: np.ndarray = np.zeros(3)


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
    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, model) -> np.ndarray:
        "Computes inputs using the specific controller strategy"


@dataclass
class PassThrough(IController):
    """This controller just feeds through the references"""

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, model) -> np.ndarray:
        """Takes inputs directly as references.

        Args:
            refs (np.ndarray): Desired/references [U_d, chi_d]^T
            xs (np.ndarray): State xs = [x, y, chi, U]^T

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

    def _reset_integrator(self):
        self._eta_diff_int = np.zeros(3)

    def compute_inputs(self, refs: np.ndarray, xs: np.ndarray, model) -> np.ndarray:
        """Computes inputs based on the PID law.

        Args:
            refs (np.ndarray): Desired/reference state xs_d = [eta_d, eta_dot_d]^T
            xs (np.ndarray): State xs = [eta, nu]^T

        Returns:
            np.ndarray: Inputs u = tau to apply to the system.
        """
        if len(refs) < 6:
            raise ValueError("Dimension of reference array should be 6 or more!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")
        eta_d = refs[0:3]
        eta_dot_d = refs[3:]

        eta = xs[0:3]
        eta_dot = xs[3:]

        eta_diff = np.zeros(3)
        eta_diff[0] = eta[0] - eta_d[0]
        eta_diff[1] = eta[1] - eta_d[1]
        eta_diff[2] = mf.wrap_angle_diff_to_pmpi(eta[2], eta_d[2])

        eta_dot_diff = np.zeros(3)
        eta_dot_diff[0] = eta_dot[0] - eta_dot_d[0]
        eta_dot_diff[1] = eta_dot[1] - eta_dot_d[1]
        eta_dot_diff[2] = eta_dot[2] - eta_dot_d[2]

        tau = mf.Rpsi(eta[2]) @ (
            -self._pars.K_p @ eta_diff - self._pars.K_d @ eta_dot_diff - self._pars.K_i @ self._eta_diff_int
        )
        return tau
