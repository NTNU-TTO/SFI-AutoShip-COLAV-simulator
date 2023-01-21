"""
    math_functions.py

    Summary:
        Contains common math functions.

    Author: Trym Tengesdal
"""
import math

import numpy as np


def wrap_min_max(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """Wraps input x to [x_min, x_max)

    Args:
        x (float or np.ndarray): Unwrapped value
        x_min (float or np.ndarray): Minimum value
        x_max (float or np.ndarray): Maximum value

    Returns:
        float or np.ndarray: Wrapped value
    """
    if isinstance(x, np.ndarray):
        return x_min + np.mod(x - x_min, x_max - x_min)
    else:
        return x_min + (x - x_min) % (x_max - x_min)


def wrap_angle_to_pmpi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [-pi, pi)

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, -np.pi * np.ones(angle.size), np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, -np.pi, np.pi)


def wrap_angle_to_02pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [0, 2pi)

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, np.zeros(angle.size), 2 * np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, 0, 2 * np.pi)


def wrap_angle_diff_to_pmpi(a_1: float | np.ndarray, a_2: float | np.ndarray) -> float | np.ndarray:
    """Wraps angle difference a_1 - a_2 to within [-pi, pi)

    Args:
        a_1 (float or np.ndarray): Angle in radians
        a_2 (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle difference
    """
    diff = wrap_angle_to_pmpi(a_1) - wrap_angle_to_pmpi(a_2)
    if isinstance(diff, np.ndarray):
        return wrap_min_max(diff, -np.pi * np.ones(diff.size), np.pi * np.ones(diff.size))
    else:
        return wrap_min_max(diff, -np.pi, np.pi)


def knots2mps(knots: float) -> float:
    """Converts from knots to miles per second.

    Args:
        knots (float): Knots to convert.

    Returns:
        float: Knots converted to mps.
    """
    mps = knots * 1.852 / 3.6
    return mps


def cm2inch(cm: float) -> float:  # inch to cm
    """Converts from cm to inches.

    Args:
        cm (float): centimetres to convert.

    Returns:
        float: Resulting inches.
    """
    return cm / 2.54


def mps2knots(mps):
    """Converts from miles per second to knots.

    Args:
        mps (float): Mps to convert.

    Returns:
        float: mps converted to knots.
    """
    knots = mps * 3.6 / 1.852
    return knots


def ms2knots(ms: float) -> float:
    """Converts from m/s to knots.

    Args:
        ms (float): m/s to convert.

    Returns:
        float: Resulting knots.
    """
    return ms * 1.94384


def knots2ms(knots: float) -> float:
    """Converts from knots to m/s.

    Args:
        knots (float): Knots to convert.

    Returns:
        float: Resulting m/s.
    """
    return knots * 0.514444


def normalize_vec(v: np.ndarray):
    """Normalize vector v to length 1.

    Args:
        v (np.ndarray): Vector to normalize

    Returns:
        np.ndarray: Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm < 0.000001:
        return v
    else:
        return v / norm


def sat(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """x = sat(x, x_min, x_max) saturates a signal x such that x_min <= x <= x_max

    Args:
        x (float or np.ndarray): Signal to saturate
        x_min (float or np.ndarray): Minimum value
        x_max (float or np.ndarray): Maximum value

    Returns:
        float or np.ndarray: Saturated signal
    """
    return np.clip(x, x_min, x_max)


def Cmtrx(Mmtrx: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates coriolis matrix C(v)

    Assumes decoupled surge and sway-yaw dynamics.
    See eq. (7.12) - (7.15) in Fossen2011

    Args:
        Mmtrx (np.ndarray): Mass matrix.
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T

    Returns:
        np.ndarray: Coriolis matrix C(v)
    """
    c13 = -(Mmtrx[1, 1] * nu[1] + Mmtrx[1, 2] * nu[2])
    c23 = Mmtrx[0, 0] * nu[0]

    return np.array([[0, 0, c13], [0, 0, c23], [-c13, -c23, 0]])


def Dmtrx(D_l: np.ndarray, D_q: np.ndarray, D_c: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Calculates damping matrix D

    Assumes decoupled surge and sway-yaw dynamics.
    See eq. (7.24) in Fossen2011+

    Args:
        D_l (np.ndarray): Linear damping matrix.
        D_q (np.ndarray): Quadratic damping matrix.
        D_c (np.ndarray): Cubic damping matrix.
        nu (np.ndarray): Body-frame velocity nu = [u, v, r]^T

    Returns:
        np.ndarray: Damping matrix D = D_l + D_q(nu) + D_c(nu)
    """
    return D_l + D_q * np.abs(nu) + D_c * (nu * nu)


def Smtrx(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b.
    """

    S = np.ndarray([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    return S


def Hmtrx(r):
    """
    H = Hmtrx(r) computes the 6x6 system transformation matrix
    H = [eye(3)     S'
         zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)
    If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
    CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
    force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG
    """

    H = np.identity(6, float)
    H[0:3, 3:6] = Smtrx(r).T

    return H


def Rzyx(phi, theta, psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = np.ndarray(
        [
            [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
            [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
            [-sth, cth * sphi, cth * cphi],
        ]
    )

    return R


def Rmtrx(psi):
    """
    R = Rmtrx(psi) computes the 3x3 rotation matrix of an angle psi about the z-axis
    """
    return np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])


def Rmtrx2D(psi):
    """
    R = Rmtrx2D(psi) computes the 2D rotation matrix.
    Rmtrx = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)])
    """
    return np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])


def Jtheta(Theta: np.ndarray) -> np.ndarray:
    """
    J = Jtheta(Theta) computes the transformation matrix in

    eta_dot = J_Theta(eta) * nu using the zyx convention. eta = [x, y, z, phi, theta, psi]
    """
    phi = Theta[0]
    theta = Theta[1]
    psi = Theta[2]
    Jmtrx = np.zeros((6, 6))
    Jmtrx[0:3, 0:3] = Rzyx(phi, theta, psi)
    Jmtrx[3:6, 3:6] = Tzyx(phi, theta)
    return Jmtrx


def Tzyx(phi, theta) -> np.ndarray:
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)

    try:
        Tmtrx = np.array([[1, sphi * sth / cth, cphi * sth / cth], [0, cphi, -sphi], [0, sphi / cth, cphi / cth]])

    except ZeroDivisionError:
        print("Tzyx is singular for theta = +-90 degrees.")

    return Tmtrx


def attitudeEuler(eta: np.ndarray, nu: np.ndarray, sampleTime: float) -> np.ndarray:
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized
    position/Euler angles eta[k+1]
    """

    p_dot = np.matmul(Rzyx(eta[3], eta[4], eta[5]), nu[0:3])
    v_dot = np.matmul(Tzyx(eta[3], eta[4]), nu[3:6])

    # Forward Euler integration
    eta[0:3] = eta[0:3] + sampleTime * p_dot
    eta[3:6] = eta[3:6] + sampleTime * v_dot

    return eta


def m2c(M: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    Cmtrx = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """
    M = 0.5 * (M + M.T)  # systematization of the inertia matrix

    #  6-DOF model
    if len(nu) == 6:

        M11mtrx = M[0:3, 0:3]
        M12mtrx = M[0:3, 3:6]
        M21mtrx = M12mtrx.T
        M22mtrx = M[3:6, 3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11mtrx, nu1) + np.matmul(M12mtrx, nu2)
        dt_dnu2 = np.matmul(M21mtrx, nu1) + np.matmul(M22mtrx, nu2)

        # Cmtrx  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        Cmtrx = np.zeros((6, 6))
        Cmtrx[0:3, 3:6] = -Smtrx(dt_dnu1)
        Cmtrx[3:6, 0:3] = -Smtrx(dt_dnu1)
        Cmtrx[3:6, 3:6] = -Smtrx(dt_dnu2)

    else:  # 3-DOF model (surge, sway and yaw)
        # Cmtrx = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]
        Cmtrx = np.zeros((3, 3))
        Cmtrx[0, 2] = -M[1, 1] * nu[1] - M[1, 2] * nu[2]
        Cmtrx[1, 2] = M[0, 0] * nu[0]
        Cmtrx[2, 0] = -Cmtrx[0, 2]
        Cmtrx[2, 1] = -Cmtrx[1, 2]

    return Cmtrx
