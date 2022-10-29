"""
    models.py

    Summary:
        Contains class definitions for various models.
        Every class must adhere to the model interface IModel.

    Author: Trym Tengesdal
"""
from abc import ABCMeta
from dataclasses import dataclass

import numpy as np
from zope import interface


class Model(metaclass=ABCMeta):
    pass


class IModel(interface.Interface):
    def dynamics(xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        "The ODE of the implemented model in discrete time."


@interface.implementer(IModel)
@dataclass
class COGSOGModel:
    """Implements a planar kinematic model using Course over ground (COG) and Speed over ground (SOG):

    x_k+1 = x_k + U_k cos(chi_k)
    y_k+1 = y_k + U_k sin(chi_k)
    chi_k+1 = chi_k + (1 / T_chi)(chi_d - chi_k)
    U_k+1 = U_k + (1 / T_U)(U_d - U_k)

    where x,y are the planar coordinates, chi the vessel COG
    and U the vessel SOG. => xs = [x, y, psi, U]
    """

    _T_chi: float = 10.0
    _T_U: float = 10.0

    def dynamics(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        if len(u) != 2:
            raise ValueError("Dimension of input array should be 2!")
        if len(xs) != 4:
            raise ValueError("Dimension of state should be 4!")
        U_d = u[0]
        chi_d = u[1]

        ode_fun = np.zeros([4, 1])
        ode_fun[0] = xs[3] * np.cos(xs[2])
        ode_fun[1] = xs[3] * np.sin(xs[2])
        ode_fun[2] = (chi_d - xs[2]) / self._T_chi
        ode_fun[3] = (U_d - xs[3]) / self._T_U
        return ode_fun
