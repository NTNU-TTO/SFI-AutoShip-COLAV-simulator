"""
    Ship parameters based on Inger Hagen's Telematron class
    https://github.com/ingerbha/sb_mpc/blob/master/ship_model.cpp
"""


import math
import random
import numpy as np

class Telematron:
    
    def __init__(self):
        #self.l_r = 4.0
        self.A = 14.5 # 5 m
        self.B = 5.0
        self.C = 1.5
        self.D = 1.5
        self.length = self.A + self.B
        self.width = self.C + self.D
        self.os_x = self.A - self.B
        self.os_y = self.D - self.C
        self.draft = self.length/10 # random draft definition

        # Motion limits
        self.r_max = np.deg2rad(0.34) # [rad/s] default max yaw rate
        self.U_max = 20 # random value in knots
        self.a_max = 1 # random value

        self.m = 3980.0
        #self.I_z = 19703.0

        """# Added M terms
        X_udot = 0.0
        Y_vdot = 0.0
        Y_rdot = 0.0
        N_vdot = 0.0
        N_rdot = 0.0

        # Linear damping terms [X_u, Y_v, Y_r, N_v, N_r]
        X_u	= -50.0
        Y_v = -200.0
        Y_r = 0.0
        N_v = 0.0
        N_r = -1281.0

        # Nonlinear damping terms [X_|u|u, Y_|v|v, N_|r|r, X_uuu, Y_vvv, N_rrr]
        self.X_uu = -135
        self.Yvv = self.Y_vr = self.Y_rv = self.Y_rr = 0.0
        self.N_vv = self.N_vr = self.N_rv = self.N_rr = 0.0
        self.X_uuu = 0.0
        self.Y_vvv = 0.0
        self.N_rrr = -3224.0

        M_RB = np.diag([m, m, I_z])
        M_A = -np.array([[X_udot, 0, 0], [0, Y_vdot, Y_rdot], [0, N_vdot, N_rdot]])
        self.M = M_RB + M_A
        self.M_inv = np.linalg.inv(self.M)

        self.D_l = -np.array([[X_u, 0, 0], [0, Y_v, Y_r], [0, N_v, N_r]])

        # Force limits
        self.Fx_min = -6550.0
        self.Fx_max = 13100.0
        self.Fy_min = -645.0
        self.Fy_max = 645.0

        # Controller parameters
        self.Kp_u = 1.0
        self.Kp_psi = 5.0
        self.Kd_psi = 1.0
        self.Kp_r = 8.0"""
        




