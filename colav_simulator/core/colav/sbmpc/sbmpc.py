import math
from dataclasses import dataclass, field

import numpy as np
from colav_simulator.common.math_functions import wrap_angle_to_pmpi
# from utils import normalize_angle

@dataclass
class SBMPCParams:
    """Parameters for the SB-MPC algorithm."""

    def to_dict(self):
        output = {
        }
        return output
    
    @classmethod
    def from_dict(cls, data: dict):
        output = SBMPCParams(
        )
        return output


class SBMPC:
    def __init__(self):
        # NB os_ship: copy of own ship initialized class
        self.T_ = 300.0  # 400                         # prediction horizon [s]
        self.DT_ = 5.0  # 0.1                          # time step [s]
        self.n_samp = int(self.T_ / self.DT_)  # number of samplings

        self.P_ = 1.0  # weights the importance of time until the event of collision occurs
        self.Q_ = 4.0  # exponent to satisfy colregs rule 16
        self.D_INIT_ = 1000  # should be >= D_CLOSE   # distance to an obstacle to activate sbmpc [m]
        self.D_CLOSE_ = 200.0  # distance for an nearby obstacle [m]
        self.D_SAFE_ = 40.0  # distance of safety zone [m]
        self.K_COLL_ = 0.5  # cost scaling factor
        self.PHI_AH_ = np.deg2rad(68.5)  # colregs angle - ahead [deg]
        self.PHI_OT_ = np.deg2rad(68.5)  # colregs angle - overtaken [deg]
        self.PHI_HO_ = np.deg2rad(22.5)  # colregs angle -  head on [deg]
        self.PHI_CR_ = np.deg2rad(68.5)  # colregs angle -  crossing [deg]
        self.KAPPA_ = 3.0  # cost function parameter
        self.K_P_ = 2.5  # cost function parameter
        self.K_CHI_ = 1.3  # cost function parameter
        self.K_DP_ = 2.0  # cost function parameter
        self.K_DCHI_SB_ = 0.9  # cost function parameter
        self.K_DCHI_P_ = 1.2  # cost function parameter

        self.P_ca_last_ = 1  # last control change
        self.Chi_ca_last_ = 0  # last course change

        self.cost_ = np.inf

        self.Chi_ca_ = np.deg2rad(
            np.array([-90.0, -75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        )  # control behaviors - course offset [deg]
        self.Chi_ca_
        # self.P_ca_ = np.array([-1.0, 0.0, 0.5, 1.0])
        self.P_ca_ = np.array([0.0, 0.5, 1.0])  # control behaviors - speed factor

        self.own_ship = Ship_model(self.T_, self.DT_)

    def get_optimal_ctrl_offset(self, u_d: float, chi_d: float, os_state: np.ndarray, obs_states: np.ndarray):
        """
        os_state: np.array(6)
        obs_state: list of np.array(4)
        """
        cost = np.inf
        cost_i = 0
        colav_active = False
        d = np.zeros(2)

        if obs_states is None:
            u_os_best = 1
            chi_os_best = 0
            self.P_ca_last_ = 1
            self.Chi_ca_last_ = 0
            return u_os_best, chi_os_best
        else:
            obstacles = []
            n_obst = len(obs_states)
            for obs_state in obs_states:
                obstacle = Obstacle(obs_state, self.T_, self.DT_)
                obstacles.append(obstacle)

        # check if obstacles are within init range
        for obs in obstacles:
            d[0] = obs.x_[0] - os_state[0]
            d[1] = obs.y_[0] - os_state[1]
            if np.linalg.norm(d) < self.D_INIT_:
                colav_active = True
        if not colav_active:
            u_os_best = 1
            chi_os_best = 0
            self.P_ca_last_ = 1
            self.Chi_ca_last_ = 0
            return u_os_best, chi_os_best

        for i in range(len(self.Chi_ca_)):
            for j in range(len(self.P_ca_)):
                self.own_ship.linear_pred(os_state, u_d * self.P_ca_[j], chi_d + self.Chi_ca_[i])

                cost_i = -1
                for k in range(n_obst):
                    cost_k = self.cost_func(self.P_ca_[j], self.Chi_ca_[i], obstacles[k])
                    if cost_k > cost_i:
                        cost_i = cost_k
                if cost_i < cost:
                    cost = cost_i
                    u_os_best = self.P_ca_[j]
                    chi_os_best = self.Chi_ca_[i]

        self.P_ca_last_ = u_os_best
        self.Chi_ca_last_ = chi_os_best

        return u_os_best, chi_os_best

    def cost_func(self, P_ca: float, Chi_ca: float, obstacle):
        obs_l = obstacle.l
        obs_w = obstacle.w
        os_l = self.own_ship.l
        os_w = self.own_ship.w

        d, los, los_inv, v_o, v_s = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
        self.combined_radius = os_l + obs_l
        d_safe = self.D_SAFE_
        d_close = self.D_CLOSE_
        H0, H1, H2 = 0, 0, 0
        cost = 0
        t = 0
        t0 = 0

        for i in range(self.n_samp):

            t += self.DT_

            d[0] = obstacle.x_[i] - self.own_ship.x_[i]
            d[1] = obstacle.y_[i] - self.own_ship.y_[i]
            dist = np.linalg.norm(d)

            R = 0
            C = 0
            mu = 0

            if dist < d_close:
                v_o[0] = obstacle.u_[i]
                v_o[1] = obstacle.v_[i]
                v_o = self.rot2d(obstacle.psi_, v_o)

                v_s[0] = self.own_ship.u_[i]
                v_s[1] = self.own_ship.v_[i]
                v_s = self.rot2d(self.own_ship.psi_[i], v_s)

                psi_o = wrap_angle_to_pmpi(obstacle.psi_)
                phi = wrap_angle_to_pmpi(math.atan2(d[1], d[0]) - self.own_ship.psi_[i])
                psi_rel = wrap_angle_to_pmpi(psi_o - self.own_ship.psi_[i])

                los = d / dist
                los_inv = -d / dist

                if phi < self.PHI_AH_:
                    d_safe_i = d_safe + os_l / 2
                elif phi > self.PHI_OT_:
                    d_safe_i = 0.5 * d_safe + os_l / 2
                else:
                    d_safe_i = d_safe + os_w / 2

                phi_o = wrap_angle_to_pmpi(math.atan2(-d[1], -d[0]) - obstacle.psi_)

                if phi_o < self.PHI_AH_:
                    d_safe_i += d_safe + obs_l / 2
                elif phi_o > self.PHI_OT_:
                    d_safe_i += 0.5 * d_safe + obs_l / 2
                else:
                    d_safe_i += d_safe + +obs_w / 2

                if (np.dot(v_s, v_o)) > np.cos(np.deg2rad(self.PHI_OT_)) * np.linalg.norm(v_s) * np.linalg.norm(
                    v_o
                ) and np.linalg.norm(v_s) > np.linalg.norm(v_o):
                    d_safe_i = d_safe + os_l / 2 + obs_l / 2

                if dist < d_safe_i:
                    R = (1 / (abs(t - t0) ** self.P_)) * (d_safe / dist) ** self.Q_
                    k_koll = self.K_COLL_ * os_l * obs_l
                    C = k_koll * np.linalg.norm(v_s - v_o) ** 2

                # Overtaken by obstacle
                OT = (np.dot(v_s, v_o)) > np.cos(np.deg2rad(self.PHI_OT_)) * np.linalg.norm(v_s) * np.linalg.norm(
                    v_o
                ) and np.linalg.norm(v_s) < np.linalg.norm(v_o)

                # Obstacle on starboard side
                SB = phi >= 0

                # Obstacle Head-on
                HO = (
                    np.linalg.norm(v_o) > 0.05
                    and (np.dot(v_s, v_o))
                    < -np.cos(np.deg2rad(self.PHI_HO_)) * np.linalg.norm(v_s) * np.linalg.norm(v_o)
                    and (np.dot(v_s, v_o)) > np.cos(np.deg2rad(self.PHI_AH_)) * np.linalg.norm(v_s)
                )

                # Crossing situation
                CR = (np.dot(v_s, v_o)) < np.cos(np.deg2rad(self.PHI_CR_)) * np.linalg.norm(v_s) * np.linalg.norm(
                    v_o
                ) and (SB and psi_rel < 0)

                mu = (SB and HO) or (CR and not OT)

            H0 = C * R + self.KAPPA_ * mu

            if H0 > H1:
                H1 = H0

        H2 = self.K_P_ * (1 - P_ca) + self.K_CHI_ * Chi_ca**2 + self.delta_P(P_ca) + self.delta_Chi(Chi_ca)
        cost = H1 + H2

        return cost

    def delta_P(self, P_ca):
        return self.K_DP_ * abs(self.P_ca_last_ - P_ca)

    def delta_Chi(self, Chi_ca):
        d_chi = Chi_ca - self.Chi_ca_last_
        if d_chi > 0:
            return self.K_DCHI_SB_ * d_chi**2
        elif d_chi < 0:
            return self.K_DCHI_P_ * d_chi**2
        else:
            return 0

    def rot2d(self, yaw: float, vec: np.ndarray):
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return R @ vec


class Obstacle:
    def __init__(self, state: np.ndarray, T: np.double, dt: np.double):
        self.n_samp_ = int(T / dt)

        self.T_ = T
        self.dt_ = dt

        self.x_ = np.zeros(self.n_samp_)
        self.y_ = np.zeros(self.n_samp_)
        self.u_ = np.zeros(self.n_samp_)
        self.v_ = np.zeros(self.n_samp_)

        self.A_ = state[5]
        self.B_ = state[6]
        self.C_ = state[7]
        self.D_ = state[8]

        self.l = self.A_ + self.B_
        self.w = self.C_ + self.D_

        self.calculate_pos_offsets()

        self.psi_ = state[2]
        self.x_[0] = state[0] + self.os_x * np.cos(self.psi_) - self.os_y * np.sin(self.psi_)
        self.y_[0] = state[1] + self.os_x * np.sin(self.psi_) + self.os_y * np.cos(self.psi_)
        self.u_[0] = state[3]
        self.v_[0] = state[4]

        self.r11_ = np.cos(self.psi_)
        self.r12_ = -np.sin(self.psi_)
        self.r21_ = np.sin(self.psi_)
        self.r22_ = np.cos(self.psi_)

        self.calculate_trajectory()

    def calculate_pos_offsets(self):
        self.os_x = self.A_ - self.B_
        self.os_y = self.D_ - self.C_

    def calculate_trajectory(self):
        for i in range(1, self.n_samp_):
            self.x_[i] = self.x_[i - 1] + (self.r11_ * self.u_[i - 1] + self.r12_ * self.v_[i - 1]) * self.dt_
            self.y_[i] = self.y_[i - 1] + (self.r21_ * self.u_[i - 1] + self.r22_ * self.v_[i - 1]) * self.dt_
            self.u_[i] = self.u_[i - 1]
            self.v_[i] = self.v_[i - 1]


class Ship_model:
    def __init__(self, T: np.double, dt: np.double):
        self.n_samp_ = int(T / dt)

        self.T_ = T
        self.DT_ = dt

        self.x_ = np.zeros(self.n_samp_)
        self.y_ = np.zeros(self.n_samp_)
        self.psi_ = np.zeros(self.n_samp_)
        self.u_ = np.zeros(self.n_samp_)
        self.v_ = np.zeros(self.n_samp_)
        self.r_ = np.zeros(self.n_samp_)

        self.A_ = 5
        self.B_ = 5
        self.C_ = 1.5
        self.D_ = 1.5

        self.l = self.A_ + self.B_
        self.w = self.C_ + self.D_

        self.calculate_pos_offsets()

    def calculate_pos_offsets(self):
        self.os_x = self.A_ - self.B_
        self.os_y = self.D_ - self.C_

    def linear_pred(self, state, u_d, psi_d):
        self.psi_[0] = wrap_angle_to_pmpi(psi_d)
        self.x_[0] = state[0] + self.os_x * np.cos(state[2]) - self.os_y * np.sin(state[2])
        self.y_[0] = state[1] + self.os_x * np.sin(state[2]) + self.os_y * np.cos(state[2])
        self.u_[0] = state[3]
        self.v_[0] = state[4]
        self.r_[0] = state[5]

        r11 = np.cos(psi_d)
        r12 = -np.sin(psi_d)
        r21 = np.sin(psi_d)
        r22 = np.cos(psi_d)

        for i in range(1, self.n_samp_):

            self.x_[i] = self.x_[i - 1] + self.DT_ * (r11 * self.u_[i - 1] + r12 * self.v_[i - 1])
            self.y_[i] = self.y_[i - 1] + self.DT_ * (r21 * self.u_[i - 1] + r22 * self.v_[i - 1])
            self.psi_[i] = psi_d  # self.psi_[i-1] + self.DT_*self.r_[i-1]
            self.u_[i] = u_d  # self.u_[i-1] + self.DT_*(u_d-self.u_[i-1])
            self.v_[i] = 0
            self.r_[i] = 0  # math.atan2(np.sin(psi_d - self.psi_[i-1]), np.cos(psi_d - self.psi_[i-1]))


def create_sbmpc_input(ships, os_idx):
    """
    Using direct values for now
    """
    u_d = ships[os_idx].u_d
    chi_d = ships[os_idx].chi_d
    os_state = ships[os_idx].get_full_state()
    obs_states = []
    for ix, ship in enumerate(ships):
        if not ix == os_idx:
            obs_state = np.array(
                [
                    ship.x,
                    ship.y,
                    ship.psi,
                    ship.u,
                    ship.v,
                    ship.length / 2,
                    ship.length / 2,
                    ship.length / 4,
                    ship.length / 4,
                ]
            )
            obs_states.append(obs_state)
    return u_d, chi_d, os_state, obs_states


def create_colav_input(ships, time):
    """
    Creates input data to use with PSB-MPC colav algorithm
    """
    colav_input = {}

    # time information
    colav_input["time"] = time

    # own ship states [x, y, psi, u, v, r]
    colav_input["os_states"] = np.array(
        [
            round(ships[0].x, 2),
            round(ships[0].y, 2),
            int(ships[0].psi),
            round(ships[0].u, 2),
            round(ships[0].v, 2),
            round(ships[0].r, 0),
        ]
    )

    # own ship's reference surge and course to the next waypoint
    colav_input["ref_surge"] = round(ships[0].u_d, 2)
    colav_input["ref_course"] = int(ships[0].chi_d)  # in radians

    # remaining waypoint coordinates
    colav_input["remaining_wp"] = ships[0].wp[ships[0].idx_next_wp :]

    # polygons coordinates
    # colav_input['polygons'] = enc.shore.mapping['coordinates']

    # target ships states [x, y, psi, u, v, A, B, C, D, ship_id]. [x, y, V_x, V_y, A, B, C, D, ID]
    other_ship_state_estimates = ships[0].get_converted_target_x_est()
    for ix, ship in enumerate(ships[1:]):
        colav_input[f"ts{ix}"] = np.append(
            other_ship_state_estimates[ix],
            [ship.length / 2, ship.length / 2, ship.length / 4, ship.length / 4, ship.mmsi],
        )
    return colav_input
