"""
    models.py

    Summary:
        Contains class definitions for various models.
        Every model class must adhere to the interface IModel.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.math_functions as mf
import numpy as np


@dataclass
class KinematicCSOGPars:
    """Parameter class for the Course and SPeed over Ground model.

    Parameters:
        T_chi (float): First order time constant for the course dynamics.
        T_U (float): First order time constant for the speed dynamics.

    """

    draft: float = 2.0
    length: float = 10.0
    width: float = 3.0
    T_chi: float = 3.0
    T_U: float = 5.0
    r_max: float = np.deg2rad(4)
    U_min: float = 0.0
    U_max: float = 15.0


@dataclass
class TelemetronPars:
    """Parameters for the Telemetron vessel (read only / Fixed)."""

    draft: float = 1.0
    length: float = 10.0
    width: float = 3.0
    l_r: float = 4.0  # Distance from CG to rudder
    M_rb: np.ndarray = np.diag([3980.0, 3980.0, 19703.0])  # Rigid body mass matrix
    M_a: np.ndarray = np.zeros((3, 3))
    D_c: np.ndarray = np.diag([0.0, 0.0, 3224.0])  # Third order/cubic damping
    D_q: np.ndarray = np.diag([135.0, 2000.0, 0.0])  # Second order/quadratic damping
    D_l: np.ndarray = np.diag([50.0, 200.0, 1281.0])  # First order/linear damping
    Fx_limits: np.ndarray = np.array([-6550.0, 13100.0])  # Force limits in x
    Fy_limits: np.ndarray = np.array([-645.0, 645.0])  # Force limits in y
    r_max: float = np.deg2rad(15)
    U_min: float = 2.0
    U_max: float = 18.0


@dataclass
class Config:
    """Configuration class for managing model parameters."""

    csog: Optional[KinematicCSOGPars] = None
    telemetron: Optional[TelemetronPars] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        if "csog" in config_dict:
            config.csog = cp.convert_settings_dict_to_dataclass(KinematicCSOGPars, config_dict["csog"])

        if "telemetron" in config_dict:
            config.telemetron = TelemetronPars()

        return config


class IModel(ABC):
    """The InterfaceModel class is abstract and used to force
    the implementation of the below methods for all subclasses (models),
    to comply with the model interface.
    """

    @abstractmethod
    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The r.h.s of the ODE x_k+1 = f(x_k, u_k) for the considered model in discrete time.

        The state should be 6 x 1
        The input should be 3 x 1.
        """


@dataclass
class KinematicCSOG(IModel):
    """Implements a planar kinematic model using Course over ground (COG) and Speed over ground (SOG):

    x_k+1 = x_k + U_k cos(chi_k)
    y_k+1 = y_k + U_k sin(chi_k)
    chi_k+1 = chi_k + (1 / T_chi)(chi_d - chi_k)
    U_k+1 = U_k + (1 / T_U)(U_d - U_k)


    where x,y are the planar coordinates, U the vessel SOG and chi the vessel COG => xs = [x, y, chi, U, 0, 0]
    """

    _pars: KinematicCSOGPars
    _n_x: int = 4
    _n_u: int = 2

    def __init__(self, config: Optional[Config] = None) -> None:
        if config and config.csog is not None:
            self._pars = config.csog
        else:
            self._pars = KinematicCSOGPars()

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
        U_d = mf.sat(u[1], 0, self._pars.U_max)

        chi_diff = mf.wrap_angle_diff_to_pmpi(chi_d, xs[2])
        xs[3] = mf.sat(xs[3], 0.0, self._pars.U_max)

        ode_fun = np.zeros(6)
        ode_fun[0] = xs[3] * np.cos(xs[2])
        ode_fun[1] = xs[3] * np.sin(xs[2])
        ode_fun[2] = mf.sat(chi_diff / self._pars.T_chi, -self._pars.r_max, self._pars.r_max)
        ode_fun[3] = (U_d - xs[3]) / self._pars.T_U

        return ode_fun

    @property
    def pars(self):
        return self._pars

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u


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
    _n_x: int = 6
    _n_u: int = 3

    def __init__(self) -> None:
        self._pars = TelemetronPars()

    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes r.h.s of ODE x_k+1 = f(x_k, u_k), where

        Args:
            xs (np.ndarray): State xs = [eta, nu]^T
            u (np.ndarray): Input vector u = tau (generalized forces in X, Y and Z)

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
        nu[0] = mf.sat(nu[0], -1e10, self._pars.U_max)
        nu[2] = mf.sat(nu[2], -self._pars.r_max, self._pars.r_max)

        u[0] = mf.sat(u[0], self._pars.Fx_limits[0], self._pars.Fx_limits[1])
        u[1] = mf.sat(u[1], self._pars.Fy_limits[0], self._pars.Fy_limits[1])
        u[2] = mf.sat(u[2], self._pars.Fy_limits[0] * self._pars.l_r, self._pars.Fy_limits[1] * self._pars.l_r)

        Minv = np.linalg.inv(self._pars.M_rb + self._pars.M_a)
        Cvv = mf.Cmtrx(self._pars.M_rb + self._pars.M_a, nu) @ nu
        Dvv = mf.Dmtrx(self._pars.D_l, self._pars.D_q, self._pars.D_c, nu) @ nu

        ode_fun = np.zeros(6)
        ode_fun[0:3] = mf.Rpsi(eta[2]) @ nu
        ode_fun[3:6] = Minv @ (-Cvv - Dvv + u)

        if np.abs(nu[0]) < 0.1:
            ode_fun[2] = 0.0

        if nu[0] < 0.05:
            print(f"psi = {eta[2]}, ddt_psi = {ode_fun[2]}")
        return ode_fun

    @property
    def pars(self):
        "Returns the parameters of the considered model."
        return self._pars

    @property
    def dims(self):
        """Returns the ACTUAL state and input dimensions considered in the model.

        NOTE: Not to be mistaken with the model interface state (6) and input (3) dimension requirements."""
        return self._n_x, self._n_u
