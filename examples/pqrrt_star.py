"""
    Demonstrates how to use the PQ-RRT* algorithm with the colav-simulator.

    Remember to install the dependencies (COLAV-simulator) and rrt-rs https://github.com/NTNU-Autoship-Internal/rrt-rs before running this script.

    This script is also found in the rrt-rs repository examples folder.

    Author: Trym Tengesdal
"""
from dataclasses import asdict, dataclass
from typing import Optional

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.stochasticity as stochasticity
import matplotlib.pyplot as plt
import numpy as np
import rrt_star_lib
import seacharts.enc as senc
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator
from shapely import strtree


@dataclass
class PQRRTStarParams:
    max_nodes: int = 3000
    max_iter: int = 25000
    iter_between_direct_goal_growth: int = 250
    min_node_dist: float = 10.0
    goal_radius: float = 700.0
    step_size: float = 0.5
    min_steering_time: float = 5.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 10.0
    max_nn_node_dist: float = 120.0
    gamma: float = 1200.0
    max_sample_adjustments: int = 100
    lambda_sample_adjustment: float = 4.0
    safe_distance: float = 0.0
    max_ancestry_level: int = 2

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


@dataclass
class RRTConfig:
    params: PQRRTStarParams = PQRRTStarParams()
    model: models.KinematicCSOGParams = models.KinematicCSOGParams(
        name="KinematicCSOG", draft=0.5, length=10.0, width=3.0, T_chi=10.0, T_U=7.0, r_max=np.deg2rad(4), U_min=0.0, U_max=15.0
    )
    los: guidances.LOSGuidanceParams = guidances.LOSGuidanceParams(K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0)

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTConfig(
            params=PQRRTStarParams.from_dict(config_dict["params"]),
            model=models.KinematicCSOGParams.from_dict(config_dict["model"]),
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
        )

        return config


@dataclass
class RRTPlannerParams:
    los: guidances.LOSGuidanceParams = guidances.LOSGuidanceParams(K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0)
    rrt: RRTConfig = RRTConfig()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTPlannerParams(los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]), rrt=RRTConfig.from_dict(config_dict["pq-rrt"]))

        return config


class PQRRTStar(ci.ICOLAV):
    def __init__(self, config: RRTPlannerParams) -> None:
        self._rrt_config = config.rrt
        self._rrt = rrt_star_lib.PQRRTStar(config.rrt.los, config.rrt.model, config.rrt.params)
        self._los = guidances.LOSGuidance(config.los)

        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._rrt_waypoints: np.ndarray = np.empty(3)
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

        self._min_depth: int = 0

        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized = False
        self._t_prev: float = 0.0

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        w: Optional[stochasticity.DisturbanceData] = None,
        **kwargs,
    ) -> np.ndarray:
        assert goal_state is not None, "Goal state must be provided to the RRT"
        assert enc is not None, "ENC must be provided to the RRT"
        if not self._initialized:
            self._min_depth = mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._map_origin = ownship_state[:2]
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(self._min_depth, enc, buffer=None)
            self._geometry_tree, _ = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)
            safe_sea_triangulation = mapf.create_safe_sea_triangulation(enc, self._min_depth, show_plots=False)
            self._rrt.transfer_enc_hazards(relevant_grounding_hazards[0])
            self._rrt.transfer_safe_sea_triangulation(safe_sea_triangulation)
            self._rrt.set_init_state(ownship_state.tolist())
            self._rrt.set_goal_state(goal_state.tolist())

            U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed
            rrt_solution: dict = self._rrt.grow_towards_goal(ownship_state.tolist(), U_d, [])
            # rrt_solution = hf.load_rrt_solution()
            times = np.array(rrt_solution["times"])
            n_samples = len(times)
            if n_samples == 0:
                raise RuntimeError("RRT did not find a solution")
            self._rrt_trajectory = np.zeros((6, n_samples))
            self._rrt_inputs = np.zeros((3, n_samples - 1))
            n_wps = len(rrt_solution["waypoints"])
            self._rrt_waypoints = np.zeros((3, n_wps))
            for k in range(n_wps):
                self._rrt_waypoints[:, k] = np.array(rrt_solution["waypoints"][k])
            for k in range(n_samples):
                self._rrt_trajectory[:, k] = np.array(rrt_solution["states"][k])
                if k < n_samples - 1:
                    self._rrt_inputs[:, k] = np.array(rrt_solution["inputs"][k])

            # Plots for debugging. Set breakpoint here to inspect
            if enc is not None:
                mapf.plot_rrt_tree(self._rrt.get_tree_as_list_of_dicts(), enc)
                mapf.plot_trajectory(self._rrt_waypoints, enc, "orange", marker_type="o")
                mapf.plot_trajectory(self._rrt_trajectory, enc, "magenta")
                mapf.plot_dynamic_obstacles(do_list, enc, 100.0, self._rrt_config.params.step_size)
                ship_poly = mapf.create_ship_polygon(ownship_state[0], ownship_state[1], ownship_state[2], kwargs["os_length"], kwargs["os_width"], 1.0, 1.0)
                enc.draw_polygon(ship_poly, color="pink")
        else:
            self._rrt_trajectory = self._rrt_trajectory[:, 1:]
            self._rrt_inputs = self._rrt_inputs[:, 1:]

        self._t_prev = t
        # Alternative 1: Use LOS-guidance to track the trajectory
        self._references = self._los.compute_references(
            self._rrt_waypoints[:2, :], speed_plan=self._rrt_waypoints[2, :], times=None, xs=ownship_state, dt=t - self._t_prev
        )

        # Alternative 2: Apply inputs directly to the ownship (bad todo in practice)
        # self._references = np.zeros((9, len(self._rrt_inputs[0, :])))
        # self._references[:2, :] = self._rrt_inputs[:2, :]
        return self._references

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        if not self._initialized:
            return {
                "nominal_trajectory": np.zeros((6, 1)),
                "nominal_inputs": np.zeros((3, 1)),
                "params": self._rrt_config.params,
                "t": self._t_prev,
            }
        else:
            return {
                "nominal_trajectory": self._rrt_trajectory,
                "nominal_inputs": self._rrt_inputs,
                "params": self._rrt_config.params,
                "t": self._t_prev,
            }

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:

        if self._rrt_trajectory.size > 6:
            plt_handles["colav_nominal_trajectory"].set_xdata(self._rrt_trajectory[1, 0:])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._rrt_trajectory[0, 0:])
        return plt_handles


if __name__ == "__main__":
    rrt = PQRRTStar(RRTPlannerParams())

    scenario_file = dp.scenarios / "rl_scenario.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file)
    simulator = Simulator()
    output = simulator.run([scenario_data], ownship_colav_system=rrt)
    print("done")
