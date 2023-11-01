"""
    behavior_generator.py

    Summary:
        Contains a class for generating random ship behaviors/waypoints+speed plans/trajectories.

    Author: Trym Tengesdal
"""

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.ship as ship
import numpy as np
import rrt_star_lib
import seacharts.enc as senc

np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class PQRRTStarParams:
    max_nodes: int = 10000
    max_iter: int = 25000
    iter_between_direct_goal_growth: int = 100000
    min_node_dist: float = 10.0
    goal_radius: float = 50.0
    step_size: float = 1.0
    min_steering_time: float = 2.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 10.0
    max_nn_node_dist: float = 100.0
    gamma: float = 1500.0
    max_sample_adjustments: int = 100
    lambda_sample_adjustment: float = 1.0
    safe_distance: float = 0.0
    max_ancestry_level: int = 2

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


@dataclass
class RRTParams:
    max_nodes: int = 10000
    max_iter: int = 20000
    iter_between_direct_goal_growth: int = 100000
    min_node_dist: float = 10.0
    goal_radius: float = 50.0
    step_size: float = 1.0
    min_steering_time: float = 2.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 10.0
    max_nn_node_dist: float = 100.0
    gamma: float = 1500.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


@dataclass
class RRTConfig:
    params: RRTParams | PQRRTStarParams = RRTParams()
    model: models.KinematicCSOGParams = models.KinematicCSOGParams(
        name="KinematicCSOG",
        draft=0.5,
        length=10.0,
        width=3.0,
        T_chi=10.0,
        T_U=7.0,
        r_max=np.deg2rad(4),
        U_min=0.0,
        U_max=15.0,
    )
    los: guidances.LOSGuidanceParams = guidances.LOSGuidanceParams(
        K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0
    )

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTConfig()
        if "max_sample_adjustments" in config_dict["params"]:
            config.params = PQRRTStarParams.from_dict(config_dict["params"])
        else:
            config.params = RRTParams.from_dict(config_dict["params"])
        config.model = models.KinematicCSOGParams.from_dict(config_dict["model"]["csog"])
        config.los = guidances.LOSGuidanceParams.from_dict(config_dict["los"])
        return config


class BehaviorGenerationMethod(Enum):
    """Enum for the supported methods for generating ship behaviors/waypoints."""

    ConstantSpeedAndCourse = 0  # Constant ship speed and course
    ConstantSpeedRandomWaypoints = 1  # Constant ship speed, uniform distribution to generate ship waypoints
    VaryingSpeedRandomWaypoints = 2  # Uniformly varying ship speed, uniform distribution to generate ship waypoints
    RapidlyExploringRandomTree = 3  # Use PQRRT*/RRT to generate ship trajectories/waypoints, with constant speed
    Any = 4  # Any of the above methods


@dataclass
class Config:
    """Configuration class for the behavior generator."""

    method: BehaviorGenerationMethod = BehaviorGenerationMethod.ConstantSpeedAndCourse
    n_wps_range: list = field(default_factory=lambda: [2, 4])  # Range of number of waypoints to be generated
    speed_plan_variation_range: list = field(
        default_factory=lambda: [-1.0, 1.0]
    )  # Determines maximal +- change in speed plan from one segment to the next
    waypoint_dist_range: list = field(
        default_factory=lambda: [200.0, 1000.0]
    )  # Range of [min, max] change in distance between randomly created waypoints
    waypoint_ang_range: list = field(
        default_factory=lambda: [-45.0, 45.0]
    )  # Range of [min, max] change in angle between randomly created waypoints
    rrt: Optional[RRTConfig] = None
    pqrrt: Optional[RRTConfig] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()
        config.method = BehaviorGenerationMethod[config_dict["method"]]

        if "pqrrt" in config_dict:
            config.pqrrt = RRTConfig.from_dict(config_dict["pqrrt"])

        if "rrt" in config_dict:
            config.rrt = RRTConfig.from_dict(config_dict["rrt"])
        return config

    def to_dict(self):
        output = asdict(self)
        output.rrt = self.rrt.to_dict()
        output.pqrrt = self.pqrrt.to_dict()
        output["method"] = self.method.name
        return output


class BehaviorGenerator:
    """Class for generating random ship behaviors, using one or multiple methods."""

    def __init__(self, config: Config) -> None:
        self._config: Config = config
        self._enc: senc.ENC = None
        self._safe_sea_cdt: list = None
        self._rrt_list: list = None
        self._pqrrt_list: list = None
        self._grounding_hazards: list = []

    def setup(
        self,
        rng: np.random.Generator,
        ship_list: list,
        enc: senc.ENC,
        simulation_timespan: float,
        show_plots: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Setup the environment for the behavior generator by e.g. transferring ENC data to
        the RRTs if configured, and creating a safe sea triangulation for more efficient sampling.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_list (list): List of ships to be considered in simulation.
            enc (senc.ENC): Electronic navigational chart.
            simulation_timespan (float): Simulation timespan.
            show_plots (bool, optional): Whether to show plots. Defaults to False.
        """
        self._enc = copy.deepcopy(enc)

        # Assume equal min depth = 5 for all ships, for now
        self._safe_sea_cdt = mapf.create_safe_sea_triangulation(enc=self._enc, vessel_min_depth=5, show_plots=False)
        self._grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
            vessel_min_depth=5, enc=self._enc, buffer=None, show_plots=show_plots
        )

        if (
            self._config.method == BehaviorGenerationMethod.RapidlyExploringRandomTree
            or self._config.method == BehaviorGenerationMethod.Any
        ):
            assert (
                self._config.rrt is not None or self._config.pqrrt is not None
            ), "RRT/PQRRT config must be provided if method is set to RRT/PQRRT"
            self._rrt_list = []
            self._pqrrt_list = []
            for ship_obj in ship_list[1:]:  # Only generate RRTs for target ships
                goal_position = mapf.generate_random_goal_position(
                    rng=rng,
                    enc=self._enc,
                    xs_start=ship_obj.csog_state,
                    safe_sea_cdt=self._safe_sea_cdt,
                    min_distance_from_start=300.0,
                    max_distance_from_start=ship_obj.max_speed * simulation_timespan,
                )
                goal_state = np.array([goal_position[0], goal_position[1], 0.0, 0.0, 0.0, 0.0])
                if ship_obj.goal_state.size > 0:
                    goal_state = ship_obj.goal_state

                if ship_obj.id > 0:  # Only generate RRTs for target ships
                    bbox = mapf.create_bbox_from_points(
                        self._enc, ship_obj.csog_state[:2], goal_state[:2], buffer=500.0
                    )
                    relevant_hazards = mapf.extract_hazards_within_bounding_box(
                        self._grounding_hazards, bbox, self._enc, show_plots=True
                    )
                    planning_cdt = mapf.create_safe_sea_triangulation(
                        self._enc,
                        bbox=bbox,
                        show_plots=False,
                    )

                    rrt = rrt_star_lib.RRT(self._config.rrt.los, self._config.rrt.model, self._config.rrt.params)
                    rrt.transfer_enc_hazards(relevant_hazards[0])
                    rrt.transfer_safe_sea_triangulation(planning_cdt)
                    rrt.set_init_state(ship_obj.state.tolist())
                    rrt.set_goal_state(goal_state.tolist())
                    U_d = ship_obj.csog_state[2]  # Constant desired speed given by the initial own-ship speed
                    rrt.reset(seed)
                    rrt.grow_towards_goal(
                        ownship_state=ship_obj.state.tolist(),
                        U_d=U_d,
                        do_list=[],
                        initialized=False,
                        return_on_first_solution=False,
                    )
                    mapf.plot_rrt_tree(rrt.get_tree_as_list_of_dicts(), self._enc)
                    self._enc.draw_circle(
                        (goal_state[1], goal_state[0]), self._config.rrt.params.goal_radius, color="orange", alpha=0.4
                    )

                    pqrrt = rrt_star_lib.PQRRTStar(
                        self._config.pqrrt.los, self._config.pqrrt.model, self._config.pqrrt.params
                    )
                    pqrrt.transfer_enc_hazards(relevant_hazards[0])
                    pqrrt.transfer_safe_sea_triangulation(planning_cdt)
                    pqrrt.set_init_state(ship_obj.state.tolist())
                    pqrrt.set_goal_state(goal_state.tolist())
                    # pqrrt.grow_towards_goal(
                    #     ownship_state=ship_obj.state.tolist(),
                    #     U_d=U_d,
                    #     do_list=[],
                    #     initialized=False,
                    #     return_on_first_solution=True,
                    # )
                    self._rrt_list.append(rrt)
                    self._pqrrt_list.append(pqrrt)

    def generate(
        self,
        rng: np.random.Generator,
        ship_list: list,
        ship_config_list: list,
        simulation_timespan: float,
        show_plots: bool = True,
    ) -> Tuple[list, list]:
        """Generates ship behaviors in the form of waypoints + speed plan that the ships will follow, for ships that have not been configured fully yet.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_list (list): List of ships to be considered in simulation.
            ship_config_list (list): List of ship configurations.
            simulation_timespan (float): Simulation timespan.
            show_plots (bool, optional): Whether to show plots. Defaults to False.

        Returns:
            - Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: Tuple containing the waypoints, speed plan and possibly trajectory for the ship.
        """
        waypoints = np.zeros((0, 2))
        speed_plan = np.zeros((0, 1))
        method = self._config.method
        ownship = ship_list[0]

        for ship_cfg_idx, ship_config in enumerate(ship_config_list):
            ship_obj = ship_list[ship_cfg_idx]
            if ship_config.waypoints is not None or ship_obj.trajectory.size > 1:
                continue

            if method == BehaviorGenerationMethod.Any:
                method = BehaviorGenerationMethod(rng.integers(0, 4))

            if method == BehaviorGenerationMethod.ConstantSpeedAndCourse:
                waypoints, speed_plan = self.generate_constant_speed_and_course_waypoints(
                    ship_obj.csog_state, simulation_timespan
                )
            elif method == BehaviorGenerationMethod.ConstantSpeedRandomWaypoints:
                waypoints = self.generate_random_waypoints(
                    rng, ship_obj.csog_state[0], ship_obj.csog_state[1], ship_obj.csog_state[3], ship_obj.draft
                )
                speed_plan = ship_obj.csog_state[2] * np.ones(waypoints.shape[1])
            elif method == BehaviorGenerationMethod.VaryingSpeedRandomWaypoints:
                waypoints = self.generate_random_waypoints(
                    rng, ship_obj.csog_state[0], ship_obj.csog_state[1], ship_obj.csog_state[3], ship_obj.draft
                )
                speed_plan = self.generate_random_speed_plan(
                    rng,
                    ship_obj.csog_state[2],
                    U_min=ship_obj.min_speed,
                    U_max=ship_obj.max_speed,
                    n_wps=waypoints.shape[1],
                )
            elif method == BehaviorGenerationMethod.RapidlyExploringRandomTree and ship_cfg_idx > 0:
                waypoints, speed_plan = self.generate_rrt_behavior(
                    rng, ship_obj, ship_cfg_idx, ownship, show_plots=True
                )

            if self._enc is not None and show_plots:
                color = "yellow" if ship_obj.id > 0 else "magenta"
                # self._enc.draw_line(
                #     [(p[1], p[0]) for p in waypoints.T], color=color, width=0.0, thickness=5.0, edge_style="dashdot"
                # )
                ship_poly = mapf.create_ship_polygon(
                    ship_obj.csog_state[0],
                    ship_obj.csog_state[1],
                    mf.wrap_angle_to_pmpi(ship_obj.csog_state[3]),
                    ship_obj.length,
                    ship_obj.width,
                    5.0,
                    5.0,
                )
                self._enc.draw_polygon(ship_poly, color=color)
                color = "orange" if ship_obj.id > 0 else "pink"
                mapf.plot_waypoints(
                    waypoints,
                    ship_obj.draft,
                    self._enc,
                    color=color,
                    point_buffer=5.0,
                    disk_buffer=15.0,
                    hole_buffer=5.0,
                )

            ship_config.waypoints = waypoints
            ship_config.speed_plan = speed_plan
            ship_obj.set_nominal_plan(ship_config.waypoints, ship_config.speed_plan)

            ship_list.append(ship_obj)
            ship_config_list.append(ship_config)

        return ship_list, ship_config_list

    def generate_rrt_behavior(
        self, rng: np.random.Generator, ship_obj: ship.Ship, ship_idx: int, ownship: ship.Ship, show_plots: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a ship behavior using RRT/PQRRT.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_obj (ship.Ship): The ship to generate a behavior for.
            ship_idx (int): The index of the ship in the ship list.
            ownship (ship.Ship): The ownship.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the resulting waypoints and speed plan.
        """
        p_os = ownship.csog_state[0:2]
        v_os = np.array(
            [
                ownship.csog_state[2] * np.cos(ownship.csog_state[3]),
                ownship.csog_state[2] * np.sin(ownship.csog_state[3]),
            ]
        )
        os_waypoints = (
            ownship.waypoints
            if ownship.waypoints is not None
            else np.array([ownship.csog_state[0:2], ownship.goal_csog_state[0:2]])
        )

        rrt = self._rrt_list[ship_idx]
        pqrrt = self._pqrrt_list[ship_idx]

        choice = 2
        if choice == 0:
            # Draw random trajectory/waypoint sample from RRT/PQRRT with leaf node near the own-ship
            # 1) Ownship goal state
            p_rand = rng.multivariate_normal(mean=os_waypoints[:, -1], cov=np.diag([25.0**2, 25.0**2]))
        elif choice == 1:
            # 2) Ownship trajectory/waypoints
            p_rand = mhm.sample_from_waypoint_corridor(rng, os_waypoints, 100.0)
        elif choice == 2:
            # 3) Minimum dCPA between ownship and target ship
            p_target = ship_obj.csog_state[0:2]
            v_target = np.array(
                [
                    ship_obj.csog_state[2] * np.cos(ship_obj.csog_state[3]),
                    ship_obj.csog_state[2] * np.sin(ship_obj.csog_state[3]),
                ]
            )

            t_cpa, d_cpa = mf.cpa(p_target, v_target, p_os, v_os)
            p_target_cpa = p_target + v_target * t_cpa
            p_rand = rng.multivariate_normal(mean=p_target_cpa, cov=np.diag([25.0**2, 25.0**2]))
        random_solution = rrt.nearest_solution(p_rand.tolist())
        waypoints, _, _, _ = mhm.parse_rrt_solution(random_solution)
        speed_plan = ship_obj.csog_state[2] * np.ones(waypoints.shape[1])
        if self._enc is not None and show_plots:
            self._enc.draw_circle(
                center=(os_waypoints[1, -1], os_waypoints[0, -1]), radius=20.0, color="green", alpha=0.3
            )
            self._enc.draw_circle(center=(p_rand[1], p_rand[0]), radius=20.0, color="orange", alpha=0.6)
        return waypoints, speed_plan

    def generate_constant_speed_and_course_waypoints(
        self, csog_state: np.ndarray, simulation_timespan: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates waypoints and speed plan for a ship with constant speed and course.

        Args:
            csog_state (np.ndarray): The ship's CSOG state.
            simulation_timespan (float): Simulation timespan.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the waypoints and speed plan.
        """
        waypoints = np.zeros((2, 2))
        waypoints[:, 0] = csog_state[0:2]
        U = csog_state[2]
        chi = csog_state[3]
        waypoints[:, 1] = waypoints[:, 0] + U * np.array([np.cos(chi), np.sin(chi)]) * simulation_timespan
        speed_plan = U * np.ones(2)
        return waypoints, speed_plan

    def generate_random_waypoints(
        self, rng: np.random.Generator, x: float, y: float, psi: float, draft: float = 5.0, n_wps: Optional[int] = None
    ) -> np.ndarray:
        """Creates random waypoints starting from a ship position and heading.

        Args:
            - rng (np.random.Generator): Random number generator.
            - x (float): x position (north) of the ship.
            - y (float): y position (east) of the ship.
            - psi (float): heading of the ship in radians.
            - draft (float, optional): How deep the ship keel is into the water. Defaults to 5.
            - n_wps (Optional[int]): Number of waypoints to create.

        Returns:
            - np.ndarray: 2 x n_wps array of waypoints.
        """
        if n_wps is None:
            n_wps = rng.integers(self._config.n_wps_range[0], self._config.n_wps_range[1])

        east_min, north_min, east_max, north_max = self._enc.bbox
        waypoints = np.zeros((2, n_wps))
        waypoints[:, 0] = np.array([x, y])
        for i in range(1, n_wps):
            crosses_grounding_hazards = True
            iter_count = -1
            while crosses_grounding_hazards:
                iter_count += 1

                distance_wp_to_wp = rng.uniform(
                    self._config.waypoint_dist_range[0], self._config.waypoint_dist_range[1]
                )

                alpha = np.deg2rad(rng.uniform(self._config.waypoint_ang_range[0], self._config.waypoint_ang_range[1]))

                new_wp = np.array(
                    [
                        waypoints[0, i - 1] + distance_wp_to_wp * np.cos(psi + alpha),
                        waypoints[1, i - 1] + distance_wp_to_wp * np.sin(psi + alpha),
                    ],
                )

                crosses_grounding_hazards = mapf.check_if_segment_crosses_grounding_hazards(
                    self._enc, new_wp, waypoints[:, i - 1], draft
                )

                if iter_count >= 200 or not crosses_grounding_hazards:
                    break

            if iter_count >= 50:
                waypoints = waypoints[:, 0:i]
                if i == 1:  # stand-still, no waypoints under given parameters that avoids grounding hazards
                    waypoints = np.append(waypoints, waypoints, axis=1)
                break

            waypoints[:, i] = new_wp
            waypoints[:, i - 1 : i + 1], clipped = mhm.clip_waypoint_segment_to_bbox(
                waypoints[:, i - 1 : i + 1], (float(north_min), float(east_min), float(north_max), float(east_max))
            )

            if clipped:
                waypoints = waypoints[:, : i + 1]
                break

        return waypoints

    def generate_random_speed_plan(
        self, rng: np.random.Generator, U: float, U_min: float = 1.0, U_max: float = 15.0, n_wps: Optional[int] = None
    ) -> np.ndarray:
        """Creates a random speed plan using the input speed and min/max speed of the ship.

        Args:
            - rng (np.random.Generator): Random number generator.
            - U (float): The ship's speed.
            - U_min (float, optional): The ship's minimum speed. Defaults to 1.0.
            - U_max (float, optional): The ship's maximum speed. Defaults to 15.0.
            - n_wps (Optional[int]): Number of waypoints to create.

        Returns:
            - np.ndarray: 1 x n_wps array containing the speed plan.
        """
        if n_wps is None:
            n_wps = rng.integers(self._config.n_wps_range[0], self._config.n_wps_range[1])

        speed_plan = np.zeros(n_wps)
        speed_plan[0] = U
        for i in range(1, n_wps):
            U_mod = rng.uniform(self._config.speed_plan_variation_range[0], self._config.speed_plan_variation_range[1])
            speed_plan[i] = mf.sat(speed_plan[i - 1] + U_mod, U_min, U_max)

            if i == n_wps - 1:
                speed_plan[i] = 0.0

        return speed_plan
