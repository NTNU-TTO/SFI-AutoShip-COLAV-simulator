"""
    behavior_generator.py

    Summary:
        Contains a class for generating random ship behaviors/waypoints+speed plans/trajectories.

    Author: Trym Tengesdal
"""

import copy
import time
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
import seacharts.enc as senc

RRT_LIB_FOUND = True
try:
    import rrt_star_lib
except ModuleNotFoundError as err:
    # Error handling
    print(err)
    RRT_LIB_FOUND = False


np.set_printoptions(suppress=True, formatter={"float_kind": "{:.2f}".format})


@dataclass
class PQRRTStarParams:
    max_nodes: int = 10000
    max_iter: int = 25000
    max_time: float = 5000.0
    iter_between_direct_goal_growth: int = 100000
    min_node_dist: float = 10.0
    goal_radius: float = 50.0
    step_size: float = 1.0
    min_steering_time: float = 2.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 10.0
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
class RRTStarParams:
    max_nodes: int = 10000
    max_iter: int = 25000
    max_time: float = 5000.0
    iter_between_direct_goal_growth: int = 100000
    min_node_dist: float = 10.0
    goal_radius: float = 50.0
    step_size: float = 1.0
    min_steering_time: float = 2.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 10.0
    gamma: float = 1500.0

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
    max_time: float = 5000.0
    iter_between_direct_goal_growth: int = 100000
    goal_radius: float = 50.0
    step_size: float = 1.0
    min_steering_time: float = 2.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 10.0
    gamma: float = 1500.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


@dataclass
class RRTConfig:
    params: RRTParams | RRTStarParams | PQRRTStarParams = RRTParams()
    model: models.KinematicCSOGParams = models.KinematicCSOGParams(
        name="KinematicCSOG",
        draft=0.5,
        length=15.0,
        width=4.0,
        T_chi=6.0,
        T_U=6.0,
        r_max=np.deg2rad(10),
        U_min=0.0,
        U_max=10.0,
    )
    los: guidances.LOSGuidanceParams = guidances.LOSGuidanceParams(
        K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0
    )

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTConfig()
        if "max_sample_adjustments" in config_dict["params"]:
            config.params = PQRRTStarParams.from_dict(config_dict["params"])
        elif "gamma" in config_dict["params"]:
            config.params = RRTStarParams.from_dict(config_dict["params"])
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
    RRT = 3  # Use baseline RRT to generate ship trajectories/waypoints, with constant speed
    RRTStar = 4  # Use RRT* to generate ship trajectories/waypoints, with constant speed
    PQRRTStar = 5  # Use PQ-RRT* to generate ship trajectories/waypoints, with constant speed
    Any = 6  # Any of the above methods


@dataclass
class Config:
    """Configuration class for the behavior generator."""

    ownship_method: BehaviorGenerationMethod = BehaviorGenerationMethod.ConstantSpeedAndCourse
    target_ship_method: BehaviorGenerationMethod = BehaviorGenerationMethod.ConstantSpeedAndCourse
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
    rrtstar: Optional[RRTConfig] = None
    pqrrtstar: Optional[RRTConfig] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(
            ownship_method=BehaviorGenerationMethod[config_dict["ownship_method"]],
            target_ship_method=BehaviorGenerationMethod[config_dict["target_ship_method"]],
            n_wps_range=config_dict["n_wps_range"],
            speed_plan_variation_range=config_dict["speed_plan_variation_range"],
            waypoint_dist_range=config_dict["waypoint_dist_range"],
            waypoint_ang_range=config_dict["waypoint_ang_range"],
        )

        if "pqrrtstar" in config_dict:
            config.pqrrtstar = RRTConfig.from_dict(config_dict["pqrrtstar"])

        if "rrtstar" in config_dict:
            config.rrtstar = RRTConfig.from_dict(config_dict["rrtstar"])

        if "rrt" in config_dict:
            config.rrt = RRTConfig.from_dict(config_dict["rrt"])
        return config

    def to_dict(self):
        output = asdict(self)
        output["rrt"] = self.rrt.to_dict()
        output["rrtstar"] = self.rrtstar.to_dict()
        output["pqrrtstar"] = self.pqrrtstar.to_dict()
        output["ownship_method"] = self.ownship_method.name
        output["target_ship_method"] = self.target_ship_method.name
        return output


class BehaviorGenerator:
    """Class for generating random ship behaviors, using one or multiple methods."""

    def __init__(self, config: Config) -> None:
        self._config: Config = config
        self._enc: senc.ENC = None
        self._safe_sea_cdt: list = None

        self._simulation_timespan: float = 0.0
        self._planning_bbox_list: list = []
        self._planning_hazard_list: list = []
        self._planning_cdt_list: list = []
        self._rrt_list: list = []
        self._rrtstar_list: list = []
        self._pqrrtstar_list: list = []
        self._grounding_hazards: list = []
        self._seed: Optional[int] = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Seeds the behavior generator, i.e. all RRTs/pqrrtstars.

        Args:
            seed (Optional[int]): Integer seed. Defaults to None.
        """
        self._seed = seed
        if len(self._rrt_list) > 0:
            for rrt in self._rrt_list:
                rrt.reset(seed)
            for rrtstar in self._rrtstar_list:
                rrtstar.reset(seed)
            for pqrrtstar in self._pqrrtstar_list:
                pqrrtstar.reset(seed)

    def setup(
        self,
        rng: np.random.Generator,
        ship_list: list,
        enc: senc.ENC,
        simulation_timespan: float,
        show_plots: bool = True,
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
        # self._seed = 1
        ownship = ship_list[0]
        self._enc = copy.deepcopy(enc)

        # Assume equal min depth = 5 for all ships, for now
        self._safe_sea_cdt = mapf.create_safe_sea_triangulation(enc=self._enc, vessel_min_depth=5, show_plots=False)
        self._grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
            vessel_min_depth=5, enc=self._enc, buffer=None, show_plots=show_plots
        )
        self._simulation_timespan = simulation_timespan

        self._planning_bbox_list = []
        self._planning_hazard_list = []
        self._planning_cdt_list = []
        self._rrt_list = []
        self._rrtstar_list = []
        self._pqrrtstar_list = []
        for ship_obj in ship_list:  # Only generate RRTs for target ships
            method = self._config.target_ship_method if ship_obj.id > 0 else self._config.ownship_method
            if method.value < BehaviorGenerationMethod.RRT.value:
                continue

            if not RRT_LIB_FOUND:
                print("RRT* library not found, exiting...")
                exit(0)

            ownship_bbox = None
            # if ship_obj.id > 0:
            #     ownship_waypoints = (
            #         ownship.waypoints
            #         if ownship.waypoints.size > 0
            #         else np.concatenate(
            #             [ownship.csog_state[0:2].reshape(-1, 1), ownship.goal_csog_state[0:2].reshape(-1, 1)]
            #         )
            #     )

            #     ownship_bbox = mapf.create_bbox_from_points(
            #         self._enc, ownship_waypoints[:, 0], ownship_waypoints[:, 1], buffer=500.0
            #     )

            goal_position = mapf.generate_random_goal_position(
                rng=rng,
                enc=self._enc,
                xs_start=ship_obj.csog_state,
                safe_sea_cdt=self._safe_sea_cdt,
                bbox=ownship_bbox,
                min_distance_from_start=400.0,
                max_distance_from_start=0.5 * ship_obj.speed * simulation_timespan,
            )
            goal_state = np.array([goal_position[0], goal_position[1], 0.0, 0.0, 0.0, 0.0])
            if ship_obj.goal_state.size > 0:
                goal_state = ship_obj.goal_state
            ship_obj.set_goal_state(goal_state)

            bbox = mapf.create_bbox_from_points(self._enc, ship_obj.csog_state[:2], goal_state[:2], buffer=400.0)
            relevant_hazards = mapf.extract_hazards_within_bounding_box(
                self._grounding_hazards, bbox, self._enc, show_plots=True
            )
            planning_cdt = mapf.create_safe_sea_triangulation(
                self._enc,
                bbox=bbox,
                show_plots=False,
            )

            self._planning_bbox_list.append(bbox)
            self._planning_hazard_list.append(relevant_hazards[0])
            self._planning_cdt_list.append(planning_cdt)

            rrt = rrt_star_lib.RRT(self._config.rrt.los, self._config.rrt.model, self._config.rrt.params)
            rrt.transfer_bbox(bbox)
            rrt.transfer_enc_hazards(relevant_hazards[0])
            rrt.transfer_safe_sea_triangulation(planning_cdt)
            rrt.set_init_state(ship_obj.state.tolist())
            rrt.set_goal_state(goal_state.tolist())
            U_d = ship_obj.csog_state[2]  # Constant desired speed given by the initial own-ship speed
            rrt.reset(self._seed)
            rrt.grow_towards_goal(
                ownship_state=ship_obj.state.tolist(),
                U_d=U_d,
                initialized=False,
                return_on_first_solution=False,
            )
            mapf.plot_rrt_tree(rrt.get_tree_as_list_of_dicts(), self._enc)
            self._rrt_list.append(rrt)

            rrtstar = rrt_star_lib.RRTStar(
                self._config.rrtstar.los, self._config.rrtstar.model, self._config.rrtstar.params
            )
            rrtstar.transfer_bbox(bbox)
            rrtstar.transfer_enc_hazards(relevant_hazards[0])
            rrtstar.transfer_safe_sea_triangulation(planning_cdt)
            rrtstar.set_init_state(ship_obj.state.tolist())
            rrtstar.set_goal_state(goal_state.tolist())
            U_d = ship_obj.csog_state[2]  # Constant desired speed given by the initial own-ship speed
            rrtstar.reset(self._seed)
            rrtstar.grow_towards_goal(
                ownship_state=ship_obj.state.tolist(),
                U_d=U_d,
                initialized=False,
                return_on_first_solution=False,
            )
            # mapf.plot_rrt_tree(rrtstar.get_tree_as_list_of_dicts(), self._enc)
            self._rrtstar_list.append(rrtstar)

            pqrrtstar = rrt_star_lib.PQRRTStar(
                self._config.pqrrtstar.los, self._config.pqrrtstar.model, self._config.pqrrtstar.params
            )
            pqrrtstar.transfer_bbox(bbox)
            pqrrtstar.transfer_enc_hazards(relevant_hazards[0])
            pqrrtstar.transfer_safe_sea_triangulation(planning_cdt)
            pqrrtstar.set_init_state(ship_obj.state.tolist())
            pqrrtstar.set_goal_state(goal_state.tolist())
            pqrrtstar.reset(self._seed)
            pqrrtstar.grow_towards_goal(
                ownship_state=ship_obj.state.tolist(),
                U_d=U_d,
                initialized=False,
                return_on_first_solution=False,
            )
            # mapf.plot_rrt_tree(pqrrtstar.get_tree_as_list_of_dicts(), self._enc)
            # self._enc.draw_circle(
            #     (goal_state[1], goal_state[0]), self._config.rrt.params.goal_radius, color="orange", alpha=0.4
            # )

            self._pqrrtstar_list.append(pqrrtstar)

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
        ownship_method = self._config.ownship_method
        target_ship_method = self._config.target_ship_method
        ownship = ship_list[0]

        for ship_cfg_idx, ship_config in enumerate(ship_config_list):
            ship_obj = ship_list[ship_cfg_idx]
            if (
                ship_config.waypoints is not None
                or ship_obj.trajectory.size > 1
                or ship_config.goal_csog_state is not None
            ):
                continue

            method = target_ship_method if ship_obj.id > 0 else ownship_method
            if method == BehaviorGenerationMethod.Any:
                method = BehaviorGenerationMethod(rng.integers(0, BehaviorGenerationMethod.Any.value))

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
            elif (
                RRT_LIB_FOUND
                and BehaviorGenerationMethod.RRT.value <= method.value <= BehaviorGenerationMethod.PQRRTStar.value
            ):
                waypoints, speed_plan, trajectory = self.generate_rrt_behavior(
                    rng, ship_obj, ship_cfg_idx, ownship, method, ownship_method, show_plots=True
                )

            if self._enc is not None and show_plots:
                color = "orange" if ship_obj.id > 0 else "pink"
                mapf.plot_waypoints(
                    waypoints,
                    ship_obj.draft,
                    self._enc,
                    color=color,
                    point_buffer=2.0,
                    disk_buffer=6.0,
                    hole_buffer=2.0,
                )
                # if RRT_LIB_FOUND and method == BehaviorGenerationMethod.RapidlyExploringRandomTree:
                #     mapf.plot_trajectory(trajectory, self._enc, color="grey")

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

            ship_config.waypoints = waypoints
            ship_config.speed_plan = speed_plan
            ship_obj.set_nominal_plan(ship_config.waypoints, ship_config.speed_plan)

            ship_list[ship_cfg_idx] = ship_obj
            ship_config_list[ship_cfg_idx] = ship_config

        if self._enc is not None and show_plots:
            color = "pink"
            mapf.plot_waypoints(
                ownship.waypoints,
                ownship.draft,
                self._enc,
                color=color,
                point_buffer=2.0,
                disk_buffer=6.0,
                hole_buffer=2.0,
            )
            # if RRT_LIB_FOUND and method == BehaviorGenerationMethod.RapidlyExploringRandomTree:
            #     mapf.plot_trajectory(trajectory, self._enc, color="grey")

            color = "magenta"
            # self._enc.draw_line(
            #     [(p[1], p[0]) for p in waypoints.T], color=color, width=0.0, thickness=5.0, edge_style="dashdot"
            # )
            ship_poly = mapf.create_ship_polygon(
                ownship.csog_state[0],
                ownship.csog_state[1],
                mf.wrap_angle_to_pmpi(ownship.csog_state[3]),
                ownship.length,
                ownship.width,
                5.0,
                5.0,
            )
            self._enc.draw_polygon(ship_poly, color=color)
        return ship_list, ship_config_list

    def generate_rrt_behavior(
        self,
        rng: np.random.Generator,
        ship_obj: ship.Ship,
        ship_idx: int,
        ownship: ship.Ship,
        rrt_method: BehaviorGenerationMethod,
        ownship_bg_method: BehaviorGenerationMethod,
        show_plots: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a ship behavior using RRT/pqrrtstar.

        Args:
            rng (np.random.Generator): Random number generator.
            ship_obj (ship.Ship): The ship to generate a behavior for.
            ship_idx (int): The index of the ship in the ship list.
            ownship (ship.Ship): The ownship.
            rrt_method (BehaviorGenerationMethod): The RRT method to use.
            ownship_bg_method (BehaviorGenerationMethod): The method used to generate the ownship behavior.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the resulting waypoints, speed plan and trajectory.
        """
        assert (
            rrt_method == BehaviorGenerationMethod.RRT
            or rrt_method == BehaviorGenerationMethod.RRTStar
            or rrt_method == BehaviorGenerationMethod.PQRRTStar
        )
        p_os = ownship.csog_state[0:2]
        v_os = np.array(
            [
                ownship.csog_state[2] * np.cos(ownship.csog_state[3]),
                ownship.csog_state[2] * np.sin(ownship.csog_state[3]),
            ]
        )
        p_target = ship_obj.csog_state[0:2]
        v_target = np.array(
            [
                ship_obj.csog_state[2] * np.cos(ship_obj.csog_state[3]),
                ship_obj.csog_state[2] * np.sin(ship_obj.csog_state[3]),
            ]
        )

        ownship_waypoints = (
            ownship.waypoints
            if ownship.waypoints is not None
            else np.array([ownship.csog_state[0:2], ownship.goal_csog_state[0:2]])
        )

        idx = ship_idx - 1
        if BehaviorGenerationMethod.RRT.value <= ownship_bg_method.value <= BehaviorGenerationMethod.PQRRTStar.value:
            idx += 1

        planning_bbox = self._planning_bbox_list[idx]
        rrt_alg = self._rrt_list[idx]
        if rrt_method == BehaviorGenerationMethod.RRT:
            rrt_alg = self._rrt_list[idx]
            print("Using RRT for behavior generation...")
        elif rrt_method == BehaviorGenerationMethod.RRTStar:
            rrt_alg = self._rrtstar_list[idx]
            print("Using RRT* for behavior generation...")
        elif rrt_method == BehaviorGenerationMethod.PQRRTStar:
            rrt_alg = self._pqrrtstar_list[idx]
            print("Using PQ-RRT* for behavior generation...")

        choice = 2
        if ship_obj.id == ownship.id == 0:
            choice = 0

        n_samples = 100
        sample_runtimes = np.zeros(n_samples)
        for s in range(n_samples):
            time_now = time.time()
            for iter in range(1):
                # Test different sampling methods
                if choice == 0:
                    # Draw random trajectory/waypoint sample from RRT/pqrrtstar with leaf node near the own-ship end goal position
                    # 1) Ownship goal state
                    p_rand = rng.multivariate_normal(mean=ownship_waypoints[:, -1], cov=np.diag([50.0**2, 50.0**2]))
                elif choice == 1:
                    # 2) Ownship trajectory/waypoints
                    p_rand = mhm.sample_from_waypoint_corridor(rng, ownship_waypoints, 200.0)
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
                    p_rand = rng.multivariate_normal(mean=p_target_cpa, cov=np.diag([50.0**2, 50.0**2]))
                elif choice == 3:
                    # 4) Sample from waypoint corridor along initial heading
                    p_end = p_target + v_target * 0.6 * self._simulation_timespan
                    corridor_waypoints, _ = mhm.clip_waypoint_segment_to_bbox(
                        np.array([p_target, p_end]).T,
                        (planning_bbox[1], planning_bbox[0], planning_bbox[3], planning_bbox[2]),
                    )
                    corridor_poly = mapf.generate_enveloping_polygon(corridor_waypoints, 200.0)
                    bbox_poly = mapf.bbox_to_polygon(planning_bbox)
                    corridor_poly_inside_bbox = corridor_poly.intersection(bbox_poly)
                    if s == 0:
                        self._enc.draw_polygon(corridor_poly_inside_bbox, color="black", alpha=0.3)
                    p_rand = mhm.sample_from_waypoint_corridor(rng, corridor_waypoints, 100.0)

                # Distance from start to sample must be at least 50m
                if np.linalg.norm(p_rand - p_target) > 50.0:
                    break

            random_solution = rrt_alg.nearest_solution(p_rand.tolist())
            waypoints, trajectory, _, _ = mhm.parse_rrt_solution(random_solution)

            speed_plan = waypoints[2, :]
            waypoints = waypoints[0:2, :]
            t_elapsed = time.time() - time_now
            sample_runtimes[s] = t_elapsed

            if self._enc is not None and show_plots:
                color = "orange" if ship_obj.id > 0 else "pink"
                mapf.plot_waypoints(
                    waypoints,
                    ship_obj.draft,
                    self._enc,
                    color=color,
                    point_buffer=3.0,
                    disk_buffer=7.0,
                    hole_buffer=3.0,
                )
                # mapf.plot_trajectory(trajectory, self._enc, color="grey")
                # self._enc.draw_circle(center=(p_rand[1], p_rand[0]), radius=10.0, color="green", alpha=0.6)
        print(
            f"t_solve: {sample_runtimes.mean():.5f} +/- {sample_runtimes.std():.5f} s | t_solve (min, max): {sample_runtimes.min():.5f}, {sample_runtimes.max():.5f} s"
        )
        return waypoints, speed_plan, trajectory

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
            n_wps = rng.integers(self._config.n_wps_range[0], self._config.n_wps_range[1] + 1)

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

                alpha = 0
                if i > 1:
                    alpha = np.deg2rad(
                        rng.uniform(self._config.waypoint_ang_range[0], self._config.waypoint_ang_range[1])
                    )

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
        waypoints = np.unique(waypoints, axis=1)
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
            n_wps = rng.integers(self._config.n_wps_range[0], self._config.n_wps_range[1] + 1)

        speed_plan = np.zeros(n_wps)
        speed_plan[0] = U
        for i in range(1, n_wps):
            U_mod = rng.uniform(self._config.speed_plan_variation_range[0], self._config.speed_plan_variation_range[1])
            speed_plan[i] = mf.sat(speed_plan[i - 1] + U_mod, U_min, U_max)

            if i == n_wps - 1:
                speed_plan[i] = 0.0

        return speed_plan
