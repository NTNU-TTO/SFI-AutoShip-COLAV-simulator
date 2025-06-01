"""
    sbmpc.py

    Summary: This module contains an implementation of the SB-MPC algorithm for COLAV.

    Author: Peder H. Lycke
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import colav_simulator.common.math_functions as mf
import numpy as np
import seacharts.enc as senc

import time
import csv
import os

@dataclass
class SBMPCParams:
    """Parameters for the SB-MPC algorithm."""

    P_: float = 1.0  # weights the importance of time until the event of collision occurs
    Q_: float = 4.0  # exponent to satisfy colregs rule 16
    D_INIT_: float = 1000.0  # should be >= D_CLOSE   # distance to an obstacle to activate sbmpc [m]
    D_CLOSE_: float = 1000.0  # distance for an nearby obstacle [m]
    D_SAFE_: float = 40.0  # distance of safety zone [m]
    K_COLL_: float = 0.5 #0.5 # cost scaling factor
    PHI_AH_: float = np.deg2rad(68.5)  # colregs angle - ahead [deg]
    PHI_OT_: float = np.deg2rad(22.5)  # colregs angle - overtaken [deg]
    PHI_HO_: float = np.deg2rad(22.5)  # colregs angle -  head on [deg]
    PHI_CR_: float = np.deg2rad(22.5)  # colregs angle -  crossing [deg]
    KAPPA_: float = 10.0 #10.0  # cost function parameter
    K_P_: float = 2.5  # cost function parameter
    K_CHI_: float = 1.5 #1.5  # cost function parameter
    K_DP_: float = 2.0  # cost function parameter
    K_DCHI_SB_: float = 1.0  # cost function parameter
    K_DCHI_P_: float = 1.4  # cost function parameter
    LAMBDA_: float = 50.0 # cost function parameter

    P_ca_last_: float = 1.0  # last control change
    Chi_ca_last_: float = 0.0  # last course change

    Chi_ca_: np.array = field(
        default_factory=lambda: np.deg2rad(
            np.array([-90.0, -75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        )
    )  # control behaviors - course offset [deg]
    P_ca_: np.array = field(default_factory=lambda: np.array([0.25, 0.5, 1.0]))  # control behaviors - speed factor

    def to_dict(self):
        output = {
            "P_": self.P_,
            "Q_": self.Q_,
            "D_INIT_": self.D_INIT_,
            "D_CLOSE_": self.D_CLOSE_,
            "D_SAFE_": self.D_SAFE_,
            "K_COLL_": self.K_COLL_,
            "PHI_AH_": self.PHI_AH_,
            "PHI_OT_": self.PHI_OT_,
            "PHI_HO_": self.PHI_HO_,
            "PHI_CR_": self.PHI_CR_,
            "KAPPA_": self.KAPPA_,
            "K_P_": self.K_P_,
            "K_CHI_": self.K_CHI_,
            "K_DP_": self.K_DP_,
            "K_DCHI_SB_": self.K_DCHI_SB_,
            "K_DCHI_P_": self.K_DCHI_P_,
            "P_ca_last": self.P_ca_last_,
            "Chi_ca_last": self.Chi_ca_last_,
            "Chi_ca_": self.Chi_ca_,
            "P_ca_": self.P_ca_,
        }
        return output

    @classmethod
    def from_dict(cls, data: dict):
        output = SBMPCParams(
            P_=data["P_"],
            Q_=data["Q_"],
            D_INIT_=data["D_INIT_"],
            D_CLOSE_=data["D_CLOSE_"],
            D_SAFE_=data["D_SAFE_"],
            K_COLL_=data["K_COLL_"],
            PHI_AH_=data["PHI_AH_"],
            PHI_OT_=data["PHI_OT_"],
            PHI_HO_=data["PHI_HO_"],
            PHI_CR_=data["PHI_CR_"],
            KAPPA_=data["KAPPA_"],
            K_P_=data["K_P_"],
            K_CHI_=data["K_CHI_"],
            K_DP_=data["K_DP_"],
            K_DCHI_SB_=data["K_DCHI_SB_"],
            K_DCHI_P_=data["K_DCHI_P_"],
            P_ca_last_=data["P_ca_last_"],
            Chi_ca_last_=data["Chi_ca_last_"],
            Chi_ca_=data["Chi_ca_"],
            P_ca_=data["P_ca_"],
        )
        return output
    
    @classmethod
    def get_tuneable_param_info(cls) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Returns the adjustable parameter ranges and increments."""

        param_ranges = {
            "Q_": [1.0, 10.0],
            "KAPPA_": [3.0, 15.0],
            "P_": [0.5, 1.5],
            "K_CHI_": [0.8, 10.0],
            "K_DCHI_SB_": [1.0, 10.0],
            "K_DCHI_P_": [1.0, 10.0],

            "K_COLL_": [0.001, 0.02],
        }

        param_incr_ranges = {
            "Q_": [-0.2, 0.2],
            "KAPPA_": [-0.2, 0.2], #"KAPPA_": [-0.2, 0.2],
            "P_": [-0.05, 0.05],
            "K_CHI_": [-0.2, 0.2],
            "K_DCHI_SB_": [-0.2, 0.2],
            "K_DCHI_P_": [-0.2, 0.2],

            "K_COLL_": [-0.01, 0.01],
        }

        return param_ranges, param_incr_ranges
    
    def get_tuneable_param_values(self, param_list: List[str]) -> np.ndarray:
        """Returns an array of the adjustable parameter values by the RL scheme."""

        param_values = []
        param_dict = self.to_dict()
        for param in param_list:
            param_values.append(param_dict[param])

        return np.array(param_values)
    
    def set_tuneable_params(self, param_subset: Dict[str, float | np.ndarray]):
        """Sets the adjustable parameters in the subset."""

        for key, value in param_subset.items():
            if key == "Q_":
                self.Q_ = float(value)
            elif key == "KAPPA_":
                self.KAPPA_ = float(value)
            elif key == "P_":
                self.P_ = float(value)
            elif key == "K_CHI_":
                self.K_CHI_ = float(value)
            elif key == "K_DCHI_SB_":
                self.K_DCHI_SB_ = float(value)
            elif key == "K_DCHI_P_":
                self.K_DCHI_P_ = float(value)
            elif key == "K_P_":
                self.K_P_ = float(value)
            elif key == "K_DP_":
                self.K_DP_ = float(value)
            elif key == "K_COLL_":
                self.K_COLL_ = float(value)
            else:
                raise ValueError(f"Parameter {key} not a tuneable parameter.")
            
    def print_parameter_values(
        self, 
        tuneable_param_list: List[str], 
        param_values: Dict[str, Union[float, np.ndarray]], 
        t: float
    ) -> None:
        
        if t % 5 == 0:
            print(f"Parameter values at t = {t} s:")
            for i, _ in enumerate(tuneable_param_list):
                print(f"{tuneable_param_list[i]}: {param_values[tuneable_param_list[i]]}")
            print("*****************************")


class SBMPC:
    def __init__(self, config: Optional[SBMPCParams] = None) -> None:
        # NB os_ship: copy of own ship initialized class
        self.T_ = 150.0  # 400                         # prediction horizon [s]
        self.DT_ = 2.5  # 0.1                          # time step [s]
        self.step_index = 0
        self.n_samp = int(self.T_ / self.DT_)  # number of samplings

        self.cost_ = np.inf

        self.ownship = ShipModel(self.T_, self.DT_)

        if config:
            self._params = config
        else:
            self._params = SBMPCParams()

    def get_optimal_ctrl_offset(
        self,
        u_d: float,
        chi_d: float,
        os_state: np.ndarray,
        do_list: List[Tuple[int, np.ndarray, np.ndarray, float, float]],
        enc: senc.ENC,
        hazards: Optional[list]=None,
        t: float = None
    ) -> Tuple[float, float]:
        """Calculates the optimal control offset for the own ship using the SB-MPC algorithm.

        Args:
            u_d (float): Nominal surge speed reference for the own ship.
            chi_d (float): Nominal course reference for the own ship.
            os_state (np.ndarray): Current state of the own ship.
            do_list (List[Tuple[int, np.ndarray, np.ndarray, float, float]]): List of tuples containing the dynamic obstale info
            enc (senc.ENC): Electronic navigational chart.

        Returns:
            Tuple[float, float]: Optimal control offset to the own ship nominal LOS references, (speed factor, course offset).
        """
        cost = np.inf
        cost_i = 0
        colav_active = False
        d, los, v_o, v_s = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
        CR_0, OTN_0, OTG_0 = np.zeros(len(do_list)), np.zeros(len(do_list)), np.zeros(len(do_list))
        SB_0, AH_0, HO_0, CRGW, DO_PASSED_0 = np.zeros(len(do_list)), np.zeros(len(do_list)), np.zeros(len(do_list)), np.zeros(len(do_list)), np.zeros(len(do_list))

        if do_list is None:
            u_os_best = 1
            chi_os_best = 0
            self._params.P_ca_last_ = 1
            self._params.Chi_ca_last_ = 0
            return u_os_best, chi_os_best
        else:
            obstacles = []
            n_obst = len(do_list)
            for obs_state in do_list:
                obstacle = Obstacle(obs_state, self.T_, self.DT_)
                obstacles.append(obstacle)

        # check if obstacles are within init range
        for i, obs in enumerate(obstacles):
            d[0] = obs.x_[0] - os_state[0]
            d[1] = obs.y_[0] - os_state[1]
            if np.linalg.norm(d) < self._params.D_INIT_:
                colav_active = True
            
            d_0 = np.linalg.norm(d)
            los = d/d_0
            v_o[0] = obs.u_[0]
            v_o[1] = obs.v_[0]
            v_o = self.rot2d(obs.psi_, v_o)

            v_s[0] = os_state[3]
            v_s[1] = os_state[4]
            v_s = self.rot2d(mf.wrap_angle_to_pmpi(os_state[2]), v_s)

            phi = mf.wrap_angle_to_pmpi(math.atan2(d[1], d[0]) - os_state[2])

            # obstacle on starboard side
            if (obs.SB_0 > -1 and ((np.deg2rad(-15) < phi < np.deg2rad(15)) or (np.deg2rad(-165) <= phi < np.deg2rad(-180)) 
                or (np.deg2rad(165) <= phi <= np.deg2rad(180)))):
                SB_0[i] = obs.SB_0
            else:
                SB_0[i] = phi > np.deg2rad(10)

            obs.SB_0 = int(SB_0[i])
        
            # obstacle ahead
            AH_0[i] = (np.dot(v_s, los)) > np.cos(self._params.PHI_AH_) * np.linalg.norm(v_s)

            # overtaking obstacle
            OTG_0[i] = ((np.dot(v_s, v_o)) > np.cos(self._params.PHI_OT_) * np.linalg.norm(v_s) * np.linalg.norm(
                v_o) and np.linalg.norm(v_o) > 0.05) and (np.linalg.norm(v_s) > np.linalg.norm(v_o)) and AH_0[i]

            # overtaken by obstacle
            OTN_0[i] = (np.dot(v_s, v_o)) > np.cos(self._params.PHI_OT_) * np.linalg.norm(v_s) * np.linalg.norm(
                v_o) and (np.linalg.norm(v_s) < np.linalg.norm(v_o)) and (not AH_0[i]) and (d_0 < self._params.D_CLOSE_)
            
            # obstacle head-on
            HO_0[i] = (np.dot(v_s, v_o)) < -np.cos(self._params.PHI_HO_) * np.linalg.norm(v_s) * np.linalg.norm(v_o) and AH_0[i]

            # crossing
            #CR_0[i] = (((np.dot(v_s, v_o)) < np.cos(self._params.PHI_CR_) * np.linalg.norm(v_s) * np.linalg.norm(
            #    v_o) and (not OTG_0[i]) and (not OTN_0[i]) and (not HO_0[i])) and AH_0[i]) and np.linalg.norm(v_o) > 0.05
            CR_0[i] = (((np.dot(v_s, v_o)) < np.cos(self._params.PHI_CR_) * np.linalg.norm(v_s) * np.linalg.norm(
                v_o) and (not OTG_0[i]) and (not OTN_0[i])) and AH_0[i]) and np.linalg.norm(v_o) > 0.05
            
            """
            if AH_0[i]:
                print("\n----------------------------------")
                print("Obstacle: ", i, "at distance", d_0, "\n")
                if SB_0[i]:
                    print("Obstacle on starboard side")
                if HO_0[i]:
                    print("Obstacle head-on")
                if OTG_0[i]:
                    print("Overtaking obstacle")
                if CR_0[i]:
                    print("Crossing")
                print("----------------------------------")
            """
                
        if not colav_active:
            u_os_best = 1
            chi_os_best = 0
            self._params.P_ca_last_ = 1
            self._params.Chi_ca_last_ = 0
            return u_os_best, chi_os_best
        
        H1, H2, P_cost, CHI_cost, delta_P, delta_Chi_ca = 0, 0, 0, 0, 0, 0

        for i in range(len(self._params.Chi_ca_)):
            #print()
            for j in range(len(self._params.P_ca_)):
                self.ownship.linear_pred(os_state, u_d * self._params.P_ca_[j], chi_d + self._params.Chi_ca_[i])

                cost_i = -1
                for k in range(n_obst):
                    cost_k, H1_k, H2_k, P_cost_k, CHI_cost_k, delta_P_k, delta_Chi_ca_k = self.cost_func(self._params.P_ca_[j], self._params.Chi_ca_[i], self._params.Chi_ca_last_, obstacles[k], 
                                            bool(SB_0[k]), bool(CR_0[k]), bool(OTN_0[k]), bool(OTG_0[k]))
                    if cost_k > cost_i:
                        cost_i = cost_k

                        # Debug
                        H1 = H1_k
                        H2 = H2_k
                        P_cost = P_cost_k
                        CHI_cost = CHI_cost_k
                        delta_P = delta_P_k
                        delta_Chi_ca = delta_Chi_ca_k
                
                """
                print("cost: ", f"{cost_k:.2f}")
                print("Chi_ca: ", int(math.degrees(self._params.Chi_ca_[i])))
                print("H1: ", f"{H1:.2f}")
                print("H2: ", f"{H2:.2f}")
                print("P_cost: ", f"{P_cost:.2f}")
                print("CHI_cost: ", f"{CHI_cost:.2f}")
                print("delta_P: ", f"{delta_P:.2f}")
                print("delta_Chi: ", f"{delta_Chi_ca:.2f}")
                print()
                """             
                
                if cost_i < cost:
                    cost = cost_i
                    u_os_best = self._params.P_ca_[j]
                    chi_os_best = self._params.Chi_ca_[i]

        #print("\nOptimal:")
        #print("u_os_best: ", f"{u_os_best:.2f}")
        #print("chi_os_best: ", f"{int(math.degrees(chi_os_best))}")
        #print("cost: ", f"{cost:.2f}", "\n")

        self._params.P_ca_last_ = u_os_best
        self._params.Chi_ca_last_ = chi_os_best

        param_list = ["K_CHI_", "KAPPA_", "K_DCHI_SB_", "K_DCHI_P_", "Q_", "P_"]
        param_values = self._params.get_tuneable_param_values(param_list)
        print_params = False
        log_params   = True
        if len(param_list) > 0:
            if log_params:
                self.log_params_to_csv(t, param_values, param_list)
            if print_params:
                print("-------------------------")
                for i in range(len(param_list)):
                    print(f"{param_list[i]}: {param_values[i]}")
                print("-------------------------\n")
        
        return u_os_best, chi_os_best

    def log_params_to_csv(self, t: float, param_values: np.ndarray, param_list: list, filename: str = "sbmpc_param_log.csv"):
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header only once
            if not file_exists:
                writer.writerow(["time"] + param_list)

            writer.writerow([t] + list(param_values))


    def cost_func(self, P_ca: float, Chi_ca: float, Chi_ca_last: float, obstacle, SB_0, CR_0, OTN_0, OTG_0):
        obs_l = obstacle.l
        obs_w = obstacle.w
        os_l = self.ownship.l
        os_w = self.ownship.w

        d, los, los_inv, v_o, v_s = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
        self.combined_radius = os_l + obs_l
        d_safe = self._params.D_SAFE_
        d_close = self._params.D_CLOSE_
        H0, H1, H2 = 0, 0, 0
        cost = 0
        t = 0
        t0 = 0

        for i in range(self.n_samp):

            t += self.DT_

            d[0] = obstacle.x_[i] - self.ownship.x_[i]
            d[1] = obstacle.y_[i] - self.ownship.y_[i]
            dist = np.linalg.norm(d)

            R = 0
            C = 0
            TC = 0
            mu = 0

            if dist < d_close:
                v_o[0] = obstacle.u_[i]
                v_o[1] = obstacle.v_[i]
                v_o = self.rot2d(obstacle.psi_, v_o)

                v_s[0] = self.ownship.u_[i]
                v_s[1] = self.ownship.v_[i]
                v_s = self.rot2d(self.ownship.psi_[i], v_s)

                psi_o = mf.wrap_angle_to_pmpi(obstacle.psi_)
                phi = mf.wrap_angle_to_pmpi(math.atan2(d[1], d[0]) - self.ownship.psi_[i])
                psi_rel = mf.wrap_angle_to_pmpi(psi_o - self.ownship.psi_[i])

                los = d / dist
                los_inv = -d / dist

                if phi < self._params.PHI_AH_:
                    d_safe_i = d_safe + os_l / 2
                elif phi > self._params.PHI_OT_:
                    d_safe_i = 0.5 * d_safe + os_l / 2
                else:
                    d_safe_i = d_safe + os_w / 2

                phi_o = mf.wrap_angle_to_pmpi(math.atan2(-d[1], -d[0]) - obstacle.psi_)

                if phi_o < self._params.PHI_AH_:
                    d_safe_i += d_safe + obs_l / 2
                elif phi_o > self._params.PHI_OT_:
                    d_safe_i += 0.5 * d_safe + obs_l / 2
                else:
                    d_safe_i += d_safe + +obs_w / 2

                if (np.dot(v_s, v_o)) > np.cos(self._params.PHI_OT_) * np.linalg.norm(v_s) * np.linalg.norm(
                    v_o
                ) and np.linalg.norm(v_s) > np.linalg.norm(v_o):
                    d_safe_i = d_safe + os_l / 2 + obs_l / 2

                if dist < d_safe_i:
                    R = (1 / (abs(t - t0) ** self._params.P_)) * (d_safe / dist) ** self._params.Q_
                    k_koll = self._params.K_COLL_ * os_l * obs_l
                    C = k_koll * np.linalg.norm(v_s - v_o) ** 2

                # Overtaken by obstacle
                OT = (np.dot(v_s, v_o)) > np.cos(self._params.PHI_OT_) * np.linalg.norm(
                    v_s
                ) * np.linalg.norm(v_o) and np.linalg.norm(v_s) < np.linalg.norm(v_o)

                # Obstacle on starboard side
                SB = phi > np.deg2rad(10)

                # Obstacle Head-on
                HO = (
                    np.linalg.norm(v_o) > 0.05
                    and (np.dot(v_s, v_o))
                    < -np.cos(self._params.PHI_HO_) * np.linalg.norm(v_s) * np.linalg.norm(v_o)
                    and (np.dot(v_s, v_o)) > np.cos(self._params.PHI_AH_) * np.linalg.norm(v_s)
                )

                # Crossing situation (Give-Way)
                CR = (np.dot(v_s, v_o)) < np.cos(self._params.PHI_CR_) * np.linalg.norm(
                    v_s
                ) * np.linalg.norm(v_o) and (SB and psi_rel < 0)

                mu = (SB and HO) or (CR and not OT)

                if SB_0: OTG = OTG_0 and (not SB)
                if (not SB_0): OTG = OTG_0 and SB

                #TODO: Legg til condition som sjekker om obstacle ship viker eller ikke (antar foreløpig ingen viking fra obstacle ship)

                #if SB_0: OTN = OTN_0 and (not SB) and (Chi_ca - Chi_ca_last < 0) 
                #if (not SB_0): OTN = OTN_0 and SB and (Chi_ca - Chi_ca_last < 0)
                OTN = OTN_0 and (Chi_ca - Chi_ca_last < 0)

                X = CR_0 and SB_0 and SB and (Chi_ca - Chi_ca_last < 0) and (psi_o < -0.2)
                #X_GW = CR_0 and (psi_o < -np.deg2rad(18)) and (Chi_ca - Chi_ca_last < 0) 
                #X_SO = CR_0 and (psi_o > np.deg2rad(18)) and (Chi_ca - Chi_ca_last > 0)

                TC = OTG or OTN or X #X_GW or X_SO # Transitional cost

            H0 = C * R + self._params.KAPPA_ * mu + self._params.LAMBDA_ * TC

            if H0 > H1:
                H1 = H0

        H2 = (
            self._params.K_P_ * (1 - P_ca)
            + self._params.K_CHI_ * Chi_ca**2
            + self.delta_P(P_ca)
            + self.delta_Chi(Chi_ca)
        )
        cost = H1 + H2

        """
        if int(math.degrees(Chi_ca)) >= 0:
            print("\n--------------------------------")
            print("Chi_ca", int(math.degrees(Chi_ca)))
            print("H1: ", H1)
            print("H2: ", H2)
            print(self._params.K_P_ * (1 - P_ca))
            print(self._params.K_CHI_ * Chi_ca**2)
            print(self.delta_P(P_ca))
            print(self.delta_Chi(Chi_ca))
            print("--------------------------------\n")
        """
        
        return cost, H1, H2, self._params.K_P_ * (1 - P_ca), self._params.K_CHI_ * Chi_ca**2, self.delta_P(P_ca), self.delta_Chi(Chi_ca)

    def delta_P(self, P_ca):
        return self._params.K_DP_ * abs(self._params.P_ca_last_ - P_ca)

    def delta_Chi(self, Chi_ca):
        d_chi = Chi_ca - self._params.Chi_ca_last_
        if d_chi > 0:
            return self._params.K_DCHI_SB_ * d_chi**2
        elif d_chi < 0:
            return self._params.K_DCHI_P_ * d_chi**2
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

        """
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
        """

        self.x_[0] = state[1][0]
        self.y_[0] = state[1][1]
        V_x = state[1][2]
        V_y = state[1][3]
        self.psi_ = np.arctan2(V_y, V_x)  # chi

        self.l = state[3]
        self.w = state[4]

        self.r11_ = np.cos(self.psi_)
        self.r12_ = -np.sin(self.psi_)
        self.r21_ = np.sin(self.psi_)
        self.r22_ = np.cos(self.psi_)

        self.u_[0] = self.r22_ * V_x + self.r21_ * V_y
        self.v_[0] = self.r12_ * V_x + self.r11_ * V_y

        self.calculate_trajectory()

        self.SB_0 = -1

    def calculate_trajectory(self):
        for i in range(1, self.n_samp_):
            self.x_[i] = self.x_[i - 1] + (self.r11_ * self.u_[i - 1] + self.r12_ * self.v_[i - 1]) * self.dt_
            self.y_[i] = self.y_[i - 1] + (self.r21_ * self.u_[i - 1] + self.r22_ * self.v_[i - 1]) * self.dt_
            self.u_[i] = self.u_[i - 1]
            self.v_[i] = self.v_[i - 1]


class ShipModel:
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
        self.psi_[0] = mf.wrap_angle_to_pmpi(psi_d)
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