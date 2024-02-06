"""
    environment.py

    Summary:
        This module wraps the colav-simulator for use with Gymnasium.
        To add your own custom observation or action type, see their respective files observation.py and action.py.

    Author: Trym Tengesdal
"""

import pathlib
from typing import Optional, Tuple

import colav_simulator.gym.reward as rw
import colav_simulator.scenario_config as sc
import colav_simulator.scenario_generator as sg
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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "video.frames_per_second": 30}
    observation_type: ObservationType
    action_type: ActionType
    scenario_config: sc.ScenarioConfig
    scenario_data_tup: tuple

    def __init__(
        self,
        simulator_config: Optional[cssim.Config] = None,
        scenario_generator_config: Optional[sg.Config] = None,
        scenario_config: Optional[sc.ScenarioConfig | pathlib.Path] = None,
        scenario_file_folder: Optional[pathlib.Path] = None,
        reload_map: Optional[bool] = True,
        rewarder_config: Optional[rw.Config] = None,
        action_type: Optional[str] = None,
        observation_type: Optional[dict | str] = None,
        render_mode: Optional[str] = "rgb_array",
        render_update_interval: Optional[float] = None,
        test_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
        show_loaded_scenario_data: Optional[bool] = False,
        max_number_of_episodes: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initializes the environment.

        Note that the scenario config object takes precedence over the scenario config file, which again takes precedence over the scenario file list.

        Args:
            simulator_config (Optional[cssim.Config]): Simulator configuration. Defaults to None.
            scenario_generator_config (Optional[sg.Config]): Scenario generator configuration. Defaults to None.
            scenario_config (Optional[sc.ScenarioConfig | Path]): Scenario configuration, either config object or path. Defaults to None.
            scenario_file_folder (Optional[list | Path): Folder path to scenario episode files. Defaults to None.
            reload_map (Optional[bool]): Whether to reload the scenario ENC map. Defaults to False. NOTE: Might cause issues with vectorized environments due to race conditions.
            rewarder_config (Optional[rw.Config]): Rewarder configuration. Defaults to None.
            action_type (Optional[str]): Action type. Defaults to None.
            observation_type (Optional[dict | str]): Observation type. Defaults to None.
            render_mode (Optional[str]): Render mode. Defaults to "human".
            render_update_interval (Optional[float]): Render update interval. Defaults to 0.2.
            test_mode (Optional[bool]): If test mode is true, the environment will not be automatically reset due to too low cumulative reward or too large distance from the path. Defaults to False.
            verbose (Optional[bool]): Wheter to print debugging info or not. Defaults to False.
            show_loaded_scenario_data (Optional[bool]): Whether to show the loaded scenario data or not. Defaults to False.
            max_number_of_episodes (Optional[int]): Maximum number of episodes to generate/load. Defaults to none (i.e. no limit).
        """
        super().__init__()
        assert (
            scenario_config is not None or scenario_file_folder is not None
        ), "Either scenario config or scenario file folder must be provided!"
        assert (
            scenario_config is None or scenario_file_folder is None
        ), "Either scenario config or scenario file folder must be provided, not both!"

        # Dummy spaces, must be overwritten by _define_spaces after call to reset the environment
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)

        self.simulator: cssim.Simulator = cssim.Simulator(config=simulator_config)
        self.scenario_generator: sg.ScenarioGenerator = sg.ScenarioGenerator(config=scenario_generator_config)
        self.scenario_config: Optional[sc.ScenarioConfig | pathlib.Path] = scenario_config
        self.scenario_file_folder: Optional[pathlib.Path] = scenario_file_folder
        self.scenario_data_tup: Optional[tuple] = None
        self.reload_map: bool = reload_map
        self._has_init_generated: bool = False
        self.max_number_of_episodes: Optional[int] = max_number_of_episodes

        self.done = False
        self.steps: int = 0
        self.episodes: int = 0
        self.n_episodes: int = 0
        self.ownship: Optional[Ship] = None
        self.render_mode = render_mode
        self.render_update_interval = render_update_interval
        self._viewer2d = self.simulator.visualizer
        self.test_mode = test_mode
        self.verbose: bool = verbose
        self.current_frame: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)

        self.rewarder: rw.Rewarder = rw.Rewarder(env=self, config=rewarder_config)

        if self.scenario_file_folder is not None:
            self._load(
                scenario_file_folder=self.scenario_file_folder,
                reload_map=self.reload_map,
                show=show_loaded_scenario_data,
            )
        else:
            self._generate(scenario_config=self.scenario_config, reload_map=self.reload_map)

        assert isinstance(self.scenario_config, sc.ScenarioConfig), "Scenario config not initialized properly!"
        self._has_init_generated = True
        (scenario_episode_list, scenario_enc) = self.scenario_data_tup
        episode_data = scenario_episode_list[0]
        self.n_episodes = len(scenario_episode_list)

        self.simulator.initialize_scenario_episode(
            ship_list=episode_data["ship_list"],
            sconfig=episode_data["config"],
            enc=scenario_enc,
            disturbance=episode_data["disturbance"],
        )
        self.ownship = self.simulator.ownship

        self.action_type_cfg = action_type if action_type is not None else self.scenario_config.rl_action_type
        self.observation_type_cfg = (
            observation_type if observation_type is not None else self.scenario_config.rl_observation_type
        )
        self._define_spaces()

    def close(self):
        """Closes the environment. To be called after usage."""
        self.done = True
        if self._viewer2d is not None:
            self._viewer2d.close_live_plot()

    def _define_spaces(self) -> None:
        """Defines the action and observation spaces."""
        assert self.scenario_config is not None, "Scenario config not initialized!"

        self.action_type = action_factory(self, self.action_type_cfg)
        self.observation_type = observation_factory(self, self.observation_type_cfg)

        self.action_space = self.action_type.space()
        self.observation_space = self.observation_type.space()

    def _is_terminated(self) -> bool:
        """Check whether the current state is a terminal state.

        Returns:
            bool: Whether the current state is a terminal state
        """
        return bool(self.simulator.is_terminated(self.verbose))

    def _is_truncated(self) -> bool:
        """Check whether the current state is a truncated state (time limit reached).

        Returns:
            bool: Whether the current state is a truncated state
        """
        return bool(self.simulator.is_truncated(self.verbose))

    def _load(self, scenario_file_folder: pathlib.Path, reload_map: bool = True, show: bool = False) -> None:
        """Load scenario episodes from files or a folder.

        Args:
            scenario_file_folder (pathlib.Path): Folder path where all episodes for a scenario is found.
            reload_map (bool): Whether to reload the scenario map. Defaults to False.
            show (bool): Whether to show the scenario data or not. Defaults to False.
        """
        name = scenario_file_folder.name
        self.scenario_data_tup = self.scenario_generator.load_scenario_from_folder(
            scenario_file_folder,
            scenario_name=name,
            reload_map=reload_map,
            show=show,
            max_number_of_episodes=self.max_number_of_episodes,
        )
        self.scenario_config = self.scenario_data_tup[0][0]["config"]

    def _generate(
        self,
        scenario_config: Optional[sc.ScenarioConfig | pathlib.Path] = None,
        reload_map: Optional[bool] = None,
    ) -> None:
        """Generate new scenario from the input configuration.

        Args:
            scenario_config (Optional[sm.ScenarioConfig]): Scenario configuration. Defaults to None.
            scenario_config_file (Optional[pathlib.Path]): Scenario configuration file. Defaults to None.
            reload_map (bool): Whether to reload the scenario map. Defaults to False.
        """
        if isinstance(scenario_config, pathlib.Path):
            self.scenario_data_tup = self.scenario_generator.generate(
                config_file=scenario_config,
                new_load_of_map_data=reload_map,
                n_episodes=self.max_number_of_episodes,
            )
        else:
            self.scenario_data_tup = self.scenario_generator.generate(
                config=scenario_config,
                new_load_of_map_data=reload_map,
                n_episodes=self.max_number_of_episodes,
            )
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
        unnormalized_obs = self.observation_type.unnormalize(obs)
        info = {
            "speed": self.ownship.csog_state[2],
            "course": self.ownship.csog_state[3],
            "position": self.ownship.csog_state[:2],
            "collision": self.simulator.determine_ownship_collision(),
            "grounding": self.simulator.determine_ownship_grounding(),
            "action": action,
            "obs": obs,
            "unnormalized_obs": unnormalized_obs,
            "reward": self.rewarder(obs, action),
        }
        return info

    def seed(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """Re-seed the environment. This is useful for reproducibility.

        Args:
            seed (Optional[int]): Seed for the random number generator. Defaults to None.
            options (Optional[dict]): Options for the environment. Defaults to None.
        """
        super().reset(seed=None, options=options)
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

        Returns:
            Tuple[Observation, dict]: Initial observation and additional information
        """
        self.seed(seed=seed, options=options)
        self.steps = 0  # Actions performed
        self.done = False

        if self.episodes == self.n_episodes:
            self._generate(scenario_config=self.scenario_config, reload_map=False)
            self.episodes = 0

        assert self.scenario_config is not None, "Scenario config not initialized!"
        (scenario_episode_list, scenario_enc) = self.scenario_data_tup
        episode_data = scenario_episode_list.pop(0)

        self.simulator.initialize_scenario_episode(
            ship_list=episode_data["ship_list"],
            sconfig=episode_data["config"],
            enc=scenario_enc,
            disturbance=episode_data["disturbance"],
        )
        self.ownship = self.simulator.ownship

        self._define_spaces()
        self._init_render()

        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())

        self.episodes += 1  # Episodes performed
        if self.verbose:
            print(f"Episode {self.episodes} started!")
        return obs, info

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Perform an action in the environment and return the new observation, the reward, whether the task is terminated, and additional information.

        Args:
            action (Action): Action vector applied by the agent

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: New observation, reward, whether the task is terminated, whether the state is truncated, and additional information.
        """
        self.action_type.act(action)
        sim_data_dict = self.simulator.step(remote_actor=True)

        obs = self.observation_type.observe()
        reward = self.rewarder(obs, action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        self.steps += 1

        return obs, reward, terminated, truncated, info

    def _init_render(self) -> None:
        """Initializes the renderer."""
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._viewer2d.toggle_liveplot_visibility(show=True)
            if self.render_update_interval is not None:
                self._viewer2d.set_update_rate(self.render_update_interval)
            self._viewer2d.init_live_plot(self.enc, self.simulator.ship_list)
            self._viewer2d.update_live_plot(
                self.simulator.t, self.enc, self.simulator.ship_list, self.simulator.recent_sensor_measurements
            )

    def render(self):
        """Renders the environment in 2D."""
        img = None
        self._viewer2d.update_live_plot(
            self.simulator.t, self.enc, self.simulator.ship_list, self.simulator.recent_sensor_measurements
        )

        if self.render_mode == "rgb_array":
            self.current_frame = self._viewer2d.get_live_plot_image()
            img = self.current_frame
        return img

    @property
    def liveplot_image(self) -> np.ndarray:
        """The current live plot image."""
        if self._viewer2d is not None and self.render_mode == "rgb_array":
            return self._viewer2d.get_live_plot_image()

    @property
    def liveplot_zoom_width(self) -> float:
        """The width of the live plot."""
        return self._viewer2d.zoom_window_width

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
    def time_truncated(self) -> float:
        """The truncation time (max time)."""
        return self.simulator.t_end

    @property
    def dynamic_obstacles(self) -> list:
        """The dynamic obstacles in the environment, seen from the own-ship perspective (which has ID 0)."""
        return self.simulator.ship_list[1:]

    @property
    def relevant_grounding_hazards(self) -> list:
        """The nearby ownship grounding hazards in the environment."""
        return self.simulator.relevant_grounding_hazards
