"""
    Ship parameters based on Inger Hagen's Telemetron class
    https://github.com/ingerbha/sb_mpc/blob/master/ship_model.cpp
"""


import math
import random
import numpy as np


class Telemetron:
    def __init__(self):
        self.use_kinematic_model = True

        # ship parameters:
        self.rudder_dist = 4.0  # m
        self.A = 5  # m
        self.B = 5  # m
        self.C = 1.5  # m
        self.D = 1.5  # m
        self.length = self.A + self.B  # length m
        self.width = self.C + self.D  # width m
        self.draft = 1  # m
        self.os_x = self.A - self.B  # position offsets m
        self.os_y = self.D - self.C  # position offsets m
        self.M = 3989.0  # mass kg
        self.I_z = 19703.0  # moment of inertia on z axis kg/m2

        # Motion limits
        self.r_max = np.deg2rad(4)  # 0.34 [rad/s] default max yaw rate
        self.U_max = 18 / 0.51  # in knots, based on paper where telematron was used
        self.a_max = 1  # random value

        # coriolis and centripetal matrix
        self.Cvv = np.zeros((3, 1))
        # Damping matrix
        self.Dvv = np.zeros((3, 1))
        # Force matrix
        self.tau = np.zeros((3, 1))

        # added mass terms
        self.X_udot = 0.0
        self.Y_vdot = 0.0
        self.Y_rdot = 0.0
        self.N_vdot = 0.0
        self.N_rdot = 0.0

        # linear damping terms
        self.X_u = -50.0
        self.Y_v = -200.0
        self.Y_r = 0.0
        self.N_v = 0.0
        self.N_r = -1281

        # nonlinear damping terms
        self.X_uu = -135.0
        self.Y_vv = -2000.0
        self.N_rr = 0.0
        self.X_uuu = 0.0
        self.Y_vvv = 0.0
        self.N_rrr = -3224.0

        # inertia matrix = rigid body mass matrix + added mass matrix
        self.Mtot = np.array([[self.M - self.X_udot, 0, 0],
                              [0, self.M - self.Y_vdot, -self.Y_rdot],
                              [0, -self.Y_rdot, self.I_z - self.N_rdot]])
        self.Minv = np.linalg.inv(self.Mtot)

        # force limits to define saturation
        self.Fx_min = -6550.0
        self.Fx_max = 13100.0
        self.Fy_min = -645.0
        self.Fy_max = 645.0

        # controller parameters
        self.Kp_u = 1.0
        self.Kp_psi = 5.0
        self.Kd_psi = 1.0
        self.Kp_r = 8.0





