"""
    integrators.py

    Summary:
        Contains ODE integrator functionality used in the simulator.

    Author: Trym Tengesdal
"""

from typing import Any

import numpy as np


def erk4_integration_step(f: Any, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Summary:
        Performs a single step of a 4th order Runge-Kutta integration scheme.

    Args:
        f (function): Function to be integrated.
        x (np.ndarray): State vector.
        u (np.ndarray): Input vector.
        t (float): Current time.
        dt (float): Time step.

    Returns:
        np.ndarray: State vector at time t + dt.
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_integration_step(f: Any, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Summary:
        Performs a single step of a Euler integration scheme.

    Args:
        f (Any): Function to be integrated.
        x (np.ndarray): State vector.
        u (np.ndarray): Input vector.
        t (float): Current time.
        dt (float): Time step.

    Returns:
        np.ndarray: State vector at time t + dt.
    """
    return x + dt * f(x, u)
