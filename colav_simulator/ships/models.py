"""
    models.py

    Summary:
        Contains class definitions for various models.
        Every class must adhere to the model interface IModel.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import colav_simulator.utils.math_functions as mf
import numpy as np


@dataclass
class KinematicCSOGPars:
    """Parameter class for the Course and SPeed over Ground model.

    Parameters:
        T_chi (float): First order time constant for the course dynamics.
        T_U (float): First order time constant for the speed dynamics.

    """

    T_chi: float = 3.0
    T_U: float = 5.0


@dataclass
class TelemetronPars:
    """Parameters for the Telemetron vessel (read only)."""

    length: float = 10.0
    width: float = 3.0
    rudder_dist: float = 4.0  # Distance from CG to rudder
    M_rb: np.ndarray = np.diag([3980.0, 3980.0, 19703.0])  # Rigid body mass matrix
    M_a: np.ndarray = np.zeros((3, 3))
    D_c: np.ndarray = np.diag([0.0, 0.0, -3224.0])  # Third order damping
    D_q: np.ndarray = np.diag([-135.0, -2000.0, 0.0])  # Second order/quadratic damping
    D_l: np.ndarray = np.diag([-50.0, -200.0, -1281.0])  # First order/linear damping
    Fx_limits: np.ndarray = np.array([-6550.0, 13100.0])  # Force limits in x
    Fy_limits: np.ndarray = np.array([-645.0, 645.0])  # Force limits in y
    r_max: float = np.deg2rad(0.34)


@dataclass
class Config:
    """Configuration class for managing model parameters."""

    name: str
    csog: Optional[KinematicCSOGPars]
    telemetron: Optional[TelemetronPars]


class IModel(ABC):
    """The InterfaceModel class is abstract and used to force
    the implementation of the below methods for all subclasses (models),
    to comply with the model interface.
    """

    @abstractmethod
    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        "The r.h.s of the ODE x_k+1 = f(x_k, u_k) for the considered model in discrete time."


@dataclass
class KinematicCSOG(IModel):
    """Implements a planar kinematic model using Course over ground (COG) and Speed over ground (SOG):

    x_k+1 = x_k + U_k cos(chi_k)
    y_k+1 = y_k + U_k sin(chi_k)
    chi_k+1 = chi_k + (1 / T_chi)(chi_d - chi_k)
    U_k+1 = U_k + (1 / T_U)(U_d - U_k)

    where x,y are the planar coordinates, chi the vessel COG
    and U the vessel SOG. => xs = [x, y, psi, U]
    """

    _pars: KinematicCSOGPars

    def __init__(self, config: Optional[Config] = None) -> None:
        if config and config.csog is not None:
            self._pars = config.csog
        else:
            self._pars = KinematicCSOGPars()

    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes r.h.s of ODE f(x, u) in x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State x_k = [x_k, y_k, chi_k, U_k]
            u (np.ndarray): Input equal to u_k = [U_d, chi_d]

        Returns:
            np.ndarray: New state x_k+1.
        """
        if len(u) != 2:
            raise ValueError("Dimension of input array should be 2!")
        if len(xs) != 4:
            raise ValueError("Dimension of state should be 4!")
        U_d = u[0]
        chi_d = u[1]
        chi_diff = mf.wrap_angle_diff_to_pmpi(chi_d, xs[2])

        ode_fun = np.zeros(4)
        ode_fun[0] = xs[3] * np.cos(xs[2])
        ode_fun[1] = xs[3] * np.sin(xs[2])
        ode_fun[2] = chi_diff / self._pars.T_chi
        ode_fun[3] = (U_d - xs[3]) / self._pars.T_U
        return ode_fun

    @property
    def pars(self):
        return self._pars


@dataclass
class Telemetron(IModel):
    """Implements a 3DOF vessel maneuvering model

    eta_dot = Rpsi(eta) * nu
    (M_rb + M_a) * nu_dot + C(nu) * nu + (D_l(nu) + D_nl) * nu = tau

    with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.

    Parameters:
        M_rb: Rigid body mass matrix
        M_a: Added mass matrix
        C: Coriolis matrix, computed from M = M_rb + M_a
        D_l: Linear damping matrix
        D_nl: Nonlinear damping matrix
    """

    _pars: TelemetronPars

    def __init__(self) -> None:
        self._pars = TelemetronPars()

    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes r.h.s of ODE f(x, u) in x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State xs = [eta, nu]^T
            u (np.ndarray): Input equal to u = tau = [Fx, Fy, rudder_dist * Fy]

        Returns:
            np.ndarray: New state xs.
        """
        if len(u) != 3:
            raise ValueError("Dimension of input array should be 3!")
        if len(xs) != 6:
            raise ValueError("Dimension of state should be 6!")
        eta = xs[0:3]
        nu = xs[3:6]

        Mmtrx = self._pars.M_rb + self._pars.M_a

        ode_fun = np.zeros(6)
        ode_fun[0:3] = mf.Rpsi(eta[2]) @ nu
        ode_fun[3:6] = np.linalg.inv(Mmtrx) * (-self._Cvv(nu) - self._Dvv(nu) + u)
        return ode_fun

    def _Cvv(self, nu: np.ndarray) -> np.ndarray:

        Mmtrx = self._pars.M_rb + self._pars.M_a
        c13 = -(Mmtrx[1, 0] * nu[0] + Mmtrx[1, 1] * nu[1] + Mmtrx[1, 2] * nu[2])
        c23 = Mmtrx[0, 0] * nu[0] + Mmtrx[0, 1] * nu[1] + Mmtrx[0, 2] * nu[2]

        Cvv_vec = np.array([c13 * nu[2], c23 * nu[2], -c13 * nu[0] - c23 * nu[1]])
        return Cvv_vec

    def _Dvv(self, nu: np.ndarray) -> np.ndarray:
        """Calculates damping vector D(nu) * nu

        Assumes decoupled surge and sway-yaw dynamics.
        See eq. (7.24) in Fossen 2011 +

        Args:
            nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T

        Returns:
            np.ndarray: Damping vector
        """
        Dvv_vec = np.zeros(3)
        Dvv_vec[0] = (
            -self._pars.D_l[0, 0] * nu[0]
            - self._pars.D_q[0, 0] * abs(nu[0]) * nu[0]
            - self._pars.D_c[0, 0] * nu[0] * nu[0] * nu[0]
        )

        Dvv_vec[1] = (
            -self._pars.D_l[1, 1] * nu[1]
            - self._pars.D_l[1, 2] * nu[2]
            - self._pars.D_q[1, 1] * abs(nu[1]) * nu[1]
            - self._pars.D_q[1, 2] * abs(nu[1]) * nu[2]
            - self._pars.D_c[1, 1] * nu[1] * nu[1] * nu[1]
        )

        Dvv_vec[2] = (
            -self._pars.D_l[2, 1] * nu[1]
            - self._pars.D_l[2, 2] * nu[2]
            - self._pars.D_q[2, 1] * abs(nu[1]) * nu[1]
            - self._pars.D_q[2, 2] * abs(nu[1]) * nu[2]
            - self._pars.D_c[2, 2] * nu[2] * nu[2] * nu[2]
        )
        return Dvv_vec
