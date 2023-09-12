"""
    environment.py

    Summary:
        This module wraps the colav-simulator for use with Gymnasium.

    Author: Trym Tengesdal
"""
import pathlib
from typing import Optional, Tuple

import colav_simulator.gym.reward as rw
import colav_simulator.scenario_management as sm
import colav_simulator.simulator as cssim
import gymnasium as gym
import numpy as np
import seacharts.enc as senc
from colav_simulator.core.ship import Ship
from colav_simulator.gym.action import Action, ActionType, action_factory
from colav_simulator.gym.observation import Observation, ObservationType, observation_factory


class COLAVEnvironment(gym.Env):
    """
    An environment for performing ship collision-free planning tasks.

    The environment is centered on the own-ship (single-agent), and consists of a maritime scenario with possibly multiple other vessels and grounding hazards from ENC data.
    """

    metadata = {
        "render_modes": ["human"],
    }
    observation_type: ObservationType
    action_type: ActionType
    scenario_config: sm.ScenarioConfig
    scenario_data_tup: tuple

    def __init__(
        self,
        simulator_config: Optional[cssim.Config] = None,
        scenario_generator_config: Optional[sm.Config] = None,
        scenario_config: Optional[sm.ScenarioConfig] = None,
        scenario_config_file: Optional[pathlib.Path] = None,
        scenario_files: Optional[list] = None,
        rewarder_config: Optional[rw.Config] = None,
        test_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
        **kwargs
    ) -> None:
        """Initializes the environment.

        Note that the scenario config object takes precedence over the scenario config file, which again takes precedence over the scenario file list.

        Args:
            simulator_config (Optional[cssim.Config]): Simulator configuration. Defaults to None.
            scenario_generator_config (Optional[sm.Config]): Scenario generator configuration. Defaults to None.
            scenario_config (Optional[sm.ScenarioConfig]): Scenario configuration. Defaults to None.
            scenario_config_file (Optional[pathlib.Path]): Scenario configuration file. Defaults to None.
            scenario_files (Optional[list]): List of scenario files. Defaults to None.
            rewarder_config (Optional[rw.Config]): Rewarder configuration. Defaults to None.
            test_mode (Optional[bool]): If test mode is true, the environment will not be automatically reset due to too low cumulative reward or too large distance from the path. Defaults to False.
            verbose (Optional[bool]): Wheter to print debugging info or not. Defaults to False.
        """
        super().__init__()

        # Dummy spaces, must be overwritten by _define_spaces after call to reset the environment
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)

        self.simulator: cssim.Simulator = cssim.Simulator(config=simulator_config)
        self.scenario_generator: sm.ScenarioGenerator = sm.ScenarioGenerator(config=scenario_generator_config)
        self.scenario_config: Optional[sm.ScenarioConfig] = scenario_config
        self.scenario_config_file: Optional[pathlib.Path] = scenario_config_file
        self.scenario_files: Optional[list] = scenario_files
        self._generate(scenario_config=scenario_config, scenario_config_file=scenario_config_file)

        self.rewarder = rw.Rewarder(config=rewarder_config)

        self.done = False
        self.steps: int = 0
        self.episodes: int = 0
        self.ownship: Optional[Ship] = None
        self.render_mode = "human"
        self._viewer2d = self.simulator.visualizer
        self.test_mode = test_mode
        self.verbose: bool = verbose

    def close(self):
        """Closes the environment. To be called after usage."""
        self.done = True
        if self._viewer2d is not None:
            self._viewer2d.close_live_plot()

    def _define_spaces(self) -> None:
        """Defines the action and observation spaces."""
        assert self.scenario_config is not None, "Scenario config not initialized!"
        self.action_type = action_factory(self, self.scenario_config.rl_action_type)
        self.action_space = self.action_type.space()

        self.observation_type = observation_factory(self, self.scenario_config.rl_observation_type)
        self.observation_space = self.observation_type.space()

    def _is_terminated(self) -> bool:
        """Check whether the current state is a terminal state.

        Returns:
            bool: Whether the current state is a terminal state
        """
        collided = self.simulator.determine_ownship_collision()
        grounded = self.simulator.determine_ownship_grounding()
        return collided or grounded

    def _is_truncated(self) -> bool:
        """Check whether the current state is a truncated state (time limit reached).

        Returns:
            bool: Whether the current state is a truncated state
        """
        truncated = self.simulator.t > self.simulator.t_end
        # if self.verbose and truncated:
        #     print("Time limit reached!")
        return truncated

    def _generate(
        self, scenario_config: Optional[sm.ScenarioConfig] = None, scenario_config_file: Optional[pathlib.Path] = None, reload_map: Optional[bool] = None
    ) -> None:
        """Generate new scenario from the input configuration.

        Args:
            scenario_config (Optional[sm.ScenarioConfig]): Scenario configuration. Defaults to None.
            scenario_config_file (Optional[pathlib.Path]): Scenario configuration file. Defaults to None.
            reload_map (bool): Whether to reload the scenario map. Defaults to False.
        """
        # if self.verbose:
        #     print("Generating new scenario...")
        self.scenario_data_tup = self.scenario_generator.generate(config=scenario_config, config_file=scenario_config_file, new_load_of_map_data=reload_map)
        self.scenario_config = self.scenario_data_tup[0][0]["config"]

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """Returns a dictionary of additional information as defined by the environment.

        Args:
            obs (Observation): Observation vector from the environment
            action (Optional[Action]): Action vector applied by the agent. Defaults to None.

        Returns:
            dict: Dictionary of additional information
        """
        assert self.ownship is not None, "Environment not initialized!"
        info = {
            "speed": self.ownship.csog_state[2],
            "course": self.ownship.csog_state[3],
            "position": self.ownship.csog_state[:2],
            "collision": self.simulator.determine_ownship_collision(),
            "grounding": self.simulator.determine_ownship_grounding(),
            "action": action,
            "obs": obs,
            "reward": self.rewarder(obs, action),
        }
        return info

    def seed(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """Re-seed the environment. This is useful for reproducibility.

        Args:
            seed (Optional[int]): Seed for the random number generator. Defaults to None.
            options (Optional[dict]): Options for the environment. Defaults to None.
        """
        super().reset(seed=seed, options=options)
        self.scenario_generator.seed(seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        """Reset the environment to a new scenario episode. If a scenario config or config file is provided, a new scenario is generated. Otherwise, the next episode of the current scenario is used, if any.

        Args:
            seed (Optional[int]): Seed for the random number generator. Defaults to None.
            options (Optional[dict]): Options for the environment. Defaults to None.
            scenario_config (Optional[sm.ScenarioConfig]): Scenario configuration. Defaults to None.
            scenario_config_file (Optional[pathlib.Path]): Scenario configuration file. Defaults to None.

        Returns:
            Tuple[Observation, dict]: Initial observation and additional information
        """
        self.seed(seed=seed, options=options)
        self.steps = 0  # Actions performed
        self.episodes = 0  # Episodes performed
        self.done = False

        assert self.scenario_config is not None, "Scenario config not initialized!"
        (scenario_episode_list, scenario_enc) = self.scenario_data_tup
        if not scenario_episode_list:
            self._generate(scenario_config=self.scenario_config, scenario_config_file=self.scenario_config_file, reload_map=False)
        (scenario_episode_list, scenario_enc) = self.scenario_data_tup
        episode_data = scenario_episode_list.pop(0)

        self.simulator.initialize_scenario_episode(
            ship_list=episode_data["ship_list"], sconfig=episode_data["config"], enc=scenario_enc, disturbance=episode_data["disturbance"], ownship_colav_system=None
        )
        self.ownship = self.simulator.ownship

        self._define_spaces()

        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        self._init_render()

        return obs, info

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Perform an action in the environment and return the new observation, the reward, whether the task is terminated, and additional information.

        Args:
            action (Action): Action vector applied by the agent

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: New observation, reward, whether the task is terminated, whether the state is truncated, and additional information.
        """
        # map action from [-1, 1] to colav system/autopilot references ranges

        self.action_type.act(action)
        sim_data_dict = self.simulator.step(remote_actor=True)

        obs = self.observation_type.observe()  # normalized observation
        reward = self.rewarder(obs, action)  # normalized reward
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        self.steps += 1

        return obs, reward, terminated, truncated, info

    def _init_render(self) -> None:
        """Initializes the renderer."""
        if self.render_mode == "human":
            self._viewer2d.toggle_liveplot_visibility(show=True)
            self._viewer2d.init_live_plot(self.enc, self.simulator.ship_list)

    def render(self, step_interval: int = 10) -> None:
        """Renders the environment in 2D at the given step interval.

        Args:
            step_interval (int): The step interval at which to render the environment. Defaults to 10.
        """
        if self.steps % step_interval == 1:
            self._viewer2d.update_live_plot(self.simulator.t, self.enc, self.simulator.ship_list, self.simulator.recent_sensor_measurements)

    @property
    def enc(self) -> senc.ENC:
        """The ENC used for the environment."""
        return self.simulator.enc

    @property
    def time(self) -> float:
        """The current simulation time."""
        return self.simulator.t

    @property
    def time_step(self) -> float:
        """The current simulation time step."""
        return self.simulator.dt

    @property
    def dynamic_obstacles(self) -> list:
        """The dynamic obstacles in the environment."""
        return self.simulator.ship_list[1:]

    @property
    def relevant_grounding_hazards(self) -> list:
        """The nearby ownship grounding hazards in the environment."""
        return self.simulator.relevant_grounding_hazards
