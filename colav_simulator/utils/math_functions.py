"""Contains commonly used math functions."""
import math

import numpy as np


def wrap_angle_to_pmpi(angle: float) -> float:
    """Wraps input angle to [-pi, pi)

    Args:
        angle (float): Angle in radians

    Returns:
        float: Wrapped angle
    """
    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle

def wrap_angle_to_02pi(angle: float) -> float:
    """Wraps input angle to [0, 2pi)

    Args:
        angle (float): Angle in radians

    Returns:
        float: Wrapped angle
    """
    while angle <= 0:
        angle += 2 * math.pi
    while angle > 2 * math.pi:
        angle -= 2 * math.pi
    return angle

def wrap_angle_diff_to_pmpi(a_1: float, a_2: float) -> float:
    """Wraps angle difference a_1 - a_2 to within [-pi, pi)

    Args:
        a_1 (float): Angle in radians
        a_2 (float): Angle in radians

    Returns:
        float: Wrapped angle difference
    """
    diff = wrap_angle_to_pmpi(a_1) - wrap_angle_to_pmpi(a_2)
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff

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

def normalize_vec(v: np.array):
    """Normalize vector v to length 1.

    Args:
        v (np.ndarray): Vector to normalize

    Returns:
        np.ndarray: Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm


#------------------------------------------------------------------------------

def ssa(angle: float) -> float:
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + math.pi) % (2 * math.pi) - math.pi

    return angle

#------------------------------------------------------------------------------

def sat(x: float, x_min: float, x_max: float) -> float:
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    if x > x_max:
        x = x_max
    elif x < x_min:
        x = x_min
    return x

def Smtrx(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b.
    """

    S = np.array([
        [ 0, -a[2], a[1] ],
        [ a[2],   0,     -a[0] ],
        [-a[1],   a[0],   0 ]  ])

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

    H = np.identity(6,float)
    H[0:3, 3:6] = Smtrx(r).T

    return H

def Rzyx(phi,theta,psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = np.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R

def Tzyx(phi,theta) -> np.array:
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)

    try:
        T = np.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])

    except ZeroDivisionError:
        print ("Tzyx is singular for theta = +-90 degrees." )

    return T

def attitudeEuler(eta: np.array, nu: np.array, sampleTime: float) -> np.array:
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized
    position/Euler angles eta[k+1]
    """

    p_dot   = np.matmul( Rzyx(eta[3], eta[4], eta[5]), nu[0:3] )
    v_dot   = np.matmul( Tzyx(eta[3], eta[4]), nu[3:6] )

    # Forward Euler integration
    eta[0:3] = eta[0:3] + sampleTime * p_dot
    eta[3:6] = eta[3:6] + sampleTime * v_dot

    return eta


#------------------------------------------------------------------------------

def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    if (len(nu) == 6):      #  6-DOF model

        M11 = M[0:3,0:3]
        M12 = M[0:3,3:6]
        M21 = M12.T
        M22 = M[3:6,3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
        dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

        #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        C = np.zeros( (6,6) )
        C[0:3,3:6] = -Smtrx(dt_dnu1)
        C[3:6,0:3] = -Smtrx(dt_dnu1)
        C[3:6,3:6] = -Smtrx(dt_dnu2)

    else:   # 3-DOF model (surge, sway and yaw)
        #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]
        C = np.zeros( (3,3) )
        C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
        C[1,2] =  M[0,0] * nu[0]
        C[2,0] = -C[0,2]
        C[2,1] = -C[1,2]

    return C
