"""
    vizualizer.py

    Summary:
        Contains functionality for visualizing/animating ship scenarios.

    Author: Trym Tengesdal, Magne Aune, Melih Akdag, Joachim Miller
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.math_functions as mf
import colav_simulator.common.paths as dp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from seacharts.enc import ENC


@dataclass
class Config:
    """Configuration class for specifying the look of the visualization."""

    show_measurements: bool = False
    show_tracks: bool = True
    show_waypoints: bool = False
    show_animation: bool = True
    show_results: bool = True
    save_animation: bool = False
    n_snapshots: int = 3 # number of scenario snapshots to show in trajectory result plotting
    frame_delay: float = 200.0
    figsize: list = field(default_factory=lambda: [12, 10])
    margins: list = field(default_factory=lambda: [0.01, 0.01])
    ship_linewidth: float = 0.9
    ship_colors: list = field(
        default_factory=lambda: [
            "xkcd:black",
            "xkcd:red",
            "xkcd:green",
            "xkcd:cyan",
            "xkcd:orange",
            "xkcd:fuchsia",
            "xkcd:yellow",
            "xkcd:grey",
            "xkcd:reddish brown",
            "xkcd:bubblegum",
            "xkcd:baby shit brown",
            "xkcd:eggshell",
            "xkcd:cloudy blue",
            "xkcd:pale aqua",
            "xkcd:light lilac",
            "xkcd:dark forest green",
            "xkcd:powder blue",
        ]
    )
    do_colors: list = field(
        default_factory=lambda: [
            "xkcd:light red",
            "xkcd:blue green",
            "xkcd:aqua",
            "xkcd:peach",
            "xkcd:pale purple",
            "xkcd:goldenrod",
            "xkcd:light grey",
            "xkcd:burnt sienna",
            "xkcd:barbie pink",
            "xkcd:ugly brown",
            "xkcd:light tan",
            "xkcd:stormy blue",
            "xkcd:light aquamarine",
            "xkcd:pale lilac",
            "xkcd:very dark green",
            "xkcd:pastel blue",
        ]
    )
    do_linewidth: float = 1.3
    radar_color: str = "xkcd:grey"
    ais_color: str = "xkcd:dark lavender"


class Visualizer:
    """Class with functionality for visualizing/animating ship scenarios, and plotting/saving the simulation results.

    Internal variables:
        fig (Figure): Matplotlib figure object for the visualization.
        axes (list): List of matplotlib axes objects for the visualization.

    """

    _config: Config
    fig: plt.figure  # handle to figure for live plotting
    axes: list  # handle to axes for live plotting
    background: Any  # map background in live plotting fig
    ship_plt_handles: list  # handles used for live plotting

    def __init__(self, enc: ENC, config_file: Path = dp.visualizer_config) -> None:
        self._config = cp.extract(Config, config_file, dp.visualizer_schema)

        self._init_figure(enc)
        self.artists = []

    def _init_figure(self, enc: ENC) -> None:
        plt.close()

        self.fig, ax_map = plt.subplots(figsize=self._config.figsize)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        ax_map.set_autoscale_on(True)
        ax_map.set_xlabel("East [m]")
        ax_map.set_ylabel("North [m]")

        mapf.plot_background(ax_map, enc)

        self.ship_plt_handles = []
        self.background = self.fig.canvas.copy_from_bbox(ax_map.bbox)

        self.axes = [ax_map]

        # TODO: Add more axes for plotting vessel speed, heading, vessel-vessel distances etc..

    def clear_ship_handles(self) -> None:
        for ship_i_handle in self.ship_plt_handles:
            if "patch" in ship_i_handle:
                ship_i_handle["patch"].remove()

            if "trajectory" in ship_i_handle:
                ship_i_handle["trajectory"].remove()

            if "predicted_trajectory" in ship_i_handle:
                ship_i_handle["predicted_trajectory"].remove()

            if "waypoints" in ship_i_handle:
                ship_i_handle["waypoints"].remove()

            if "do_tracks" in ship_i_handle:
                for do_track in ship_i_handle["do_tracks"]:
                    do_track.remove()

                for do_covariance in ship_i_handle["do_covariances"]:
                    do_covariance.remove()

            if "radar" in ship_i_handle:
                ship_i_handle["radar"].remove()

            if "ais" in ship_i_handle:
                ship_i_handle["ais"].remove()

        self.ship_plt_handles = []

    def init_live_plot(self, ship_list: list) -> None:
        """Initializes the plot handles of the live plot for a simulation
        given by the ship list.

        Args:
            ship_list (list): List of configured ships in the simulation.
        """
        ax_map = self.axes[0]

        self.clear_ship_handles()

        self.fig.canvas.restore_region(self.background)

        n_ships = len(ship_list)
        for i, ship in enumerate(ship_list):

            c = self._config.ship_colors[i]
            lw = self._config.ship_linewidth

            ship_i_handles: dict = {}

            if i == 0:
                ship_name = "OS"

                c_do = self._config.do_colors
                do_lw = self._config.do_linewidth

                if self._config.show_tracks:
                    ship_i_handles["do_tracks"] = []
                    ship_i_handles["do_covariances"] = []
                    for j in range(1, n_ships):
                        ship_i_handles["do_tracks"].append(
                            ax_map.plot([], [], linewidth=do_lw, color=c_do[j - 1], label=f"DO{j-1} track")[0]
                        )

                        ship_i_handles["do_covariances"].append(
                            ax_map.fill([], [], linewidth=lw, color=c_do[j - 1], alpha=0.5, label=f"DO{j-1} cov.")[0]
                        )

                if self._config.show_measurements:
                    for sensor in ship.sensors:
                        if sensor.type == "radar":
                            ship_i_handles["radar"] = ax_map.plot(
                                [],
                                [],
                                color=self._config.radar_color,
                                linewidth=lw,
                                linestyle="None",
                                marker="o",
                                markersize=8,
                                label=ship_name + "Radar meas.",
                            )[0]
                        elif sensor.type == "ais":
                            ship_i_handles["ais"] = ax_map.plot(
                                [],
                                [],
                                color=self._config.ais_color,
                                linewidth=lw,
                                linestyle="None",
                                marker="*",
                                markersize=8,
                                label="AIS meas.",
                            )[0]

            else:
                ship_name = "DO " + str(i + 1)

            ship_i_handles["patch"] = ax_map.fill([], [], color=c, linewidth=lw, label=ship_name)[0]

            ship_i_handles["trajectory"] = ax_map.plot([], [], color=c, linewidth=lw, label=ship_name + " traj.")[0]

            ship_i_handles["predicted_trajectory"] = ax_map.plot(
                [], [], color=c, linewidth=lw, marker="*", linestyle="-", label=ship_name + " pred. traj."
            )[0]

            if self._config.show_waypoints:
                ship_i_handles["waypoints"] = ax_map.plot(
                    ship.waypoints[1, :],
                    ship.waypoints[0, :],
                    color=c,
                    marker="o",
                    markersize=11,
                    linestyle="--",
                    linewidth=lw,
                    alpha=0.3,
                    label=ship_name + " waypoints",
                )[0]

            self.ship_plt_handles.append(ship_i_handles)

    def update_live_plot(self, ship_list: list, sensor_measurements: list) -> None:
        """Updates the live plot with the current data of the ships in the simulation.

        Args:
            ship_list (list): List of configured ships in the simulation.
            sensor_measurements (list): List of sensor measurements generated from each ship at the current time.
        """

        n_ships = len(ship_list)
        for i, ship in enumerate(ship_list):
            pose_i = ship.pose

            if i == 0:
                tracks = ship.get_do_tracks()
                if self._config.show_tracks and tracks is not None:
                    for j in range(1, n_ships):
                        pose_j = ship_list[j].pose

                        self.ship_plt_handles[i]["do_tracks"][j - 1].set_xdata(
                            [*self.ship_plt_handles[i]["do_tracks"][j - 1].get_xdata(), tracks[0][j - 1][1]]
                        )
                        self.ship_plt_handles[i]["do_tracks"][j - 1].set_ydata(
                            [*self.ship_plt_handles[i]["do_tracks"][j - 1].get_ydata(), tracks[0][j - 1][0]]
                        )
                        self.axes[0].draw_artist(self.ship_plt_handles[i]["do_tracks"][j - 1])

                        ellipse_x, ellipse_y = mf.create_probability_ellipse(tracks[1][j - 1], 0.99)
                        self.ship_plt_handles[i]["do_covariances"][j - 1].set_xy(
                            np.vstack((ellipse_y + pose_j[1], ellipse_x + pose_j[0])).T
                        )
                        self.axes[0].draw_artist(self.ship_plt_handles[i]["do_covariances"][j - 1])

                        if self._config.show_measurements and len(sensor_measurements[i]) > 0:
                            for sensor_id, sensor in enumerate(ship.sensors):
                                sensor_data = sensor_measurements[i][j - 1][sensor_id]

                                if any(np.isnan(sensor_data)):
                                    continue

                                if sensor.type == "radar":
                                    self.ship_plt_handles[i]["radar"].set_xdata(
                                        [*self.ship_plt_handles[i]["radar"].get_xdata(), sensor_data[1]]
                                    )
                                    self.ship_plt_handles[i]["radar"].set_ydata(
                                        [*self.ship_plt_handles[i]["radar"].get_ydata(), sensor_data[0]]
                                    )
                                    self.axes[0].draw_artist(self.ship_plt_handles[i]["radar"])

                                elif sensor.type == "ais":
                                    self.ship_plt_handles[i]["ais"].set_xdata(
                                        [*self.ship_plt_handles[i]["ais"].get_xdata(), sensor_data[1]]
                                    )
                                    self.ship_plt_handles[i]["ais"].set_ydata(
                                        [*self.ship_plt_handles[i]["ais"].get_ydata(), sensor_data[0]]
                                    )
                                    self.axes[0].draw_artist(self.ship_plt_handles[i]["ais"])

            # Update ship patch
            ship_poly = mapf.create_ship_polygon(pose_i[0], pose_i[1], pose_i[3], ship.length, ship.width, 3.0)
            y_ship, x_ship = ship_poly.exterior.xy
            self.ship_plt_handles[i]["patch"].set_xy(np.array([y_ship, x_ship]).T)

            self.axes[0].draw_artist(self.ship_plt_handles[i]["patch"])

            # Update ship trajectory
            self.ship_plt_handles[i]["trajectory"].set_xdata(
                [*self.ship_plt_handles[i]["trajectory"].get_xdata(), pose_i[1]]
            )
            self.ship_plt_handles[i]["trajectory"].set_ydata(
                [*self.ship_plt_handles[i]["trajectory"].get_ydata(), pose_i[0]]
            )
            self.axes[0].draw_artist(self.ship_plt_handles[i]["trajectory"])

            if self._config.show_waypoints:
                self.ship_plt_handles[i]["waypoints"].set_xdata(ship.waypoints[1, :])
                self.ship_plt_handles[i]["waypoints"].set_ydata(ship.waypoints[0, :])
                self.axes[0].draw_artist(self.ship_plt_handles[i]["waypoints"])

            # TODO: Update predicted ship trajectory

        self.fig.canvas.blit(self.axes[0].bbox)

        plt.legend()

        self.fig.canvas.flush_events()
        plt.pause(0.00001)  # plt.pause(self._config.frame_delay / 1000)

    def visualize(
        self,
        ship_list: list,
        data: DataFrame,
        times: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """Visualize the ship trajectories.

        Args:
            enc (ENC): Electronic Navigational Chart object.
            ship_list: List of ships in the simulation.
            data (DataFrame): Pandas DataFrame containing the ship simulation data.
            times (List/np.ndarray): List/array of times to consider.
            save_path (Optional[pathlib.Path]): Path to file where the animation is saved. Defaults to None.
        """

        vessels = []
        trajectories = []

        # data: n_ships x 1 dataframe, where each ship entry contains n_samples x 1 dictionaries of pose, waypoints, speed plans etc..
        n_ships = len(data.columns)
        for i in range(n_ships):
            c = self._config.ship_colors[i]
            lw = self._config.ship_linewidth
            if i == 0:
                vessels.append(plt.fill([], [], color=c, linewidth=lw, label="OS")[0])
                trajectories.append(plt.plot([], [], c, label="OS traj.")[0])
            else:
                vessels.append(plt.fill([], [], color=c, linewidth=lw, label=f"DO {i - 1}")[0])
                trajectories.append(plt.plot([], [], color=c, linewidth=lw, label=f"DO {i - 1} traj.")[0])
            if self._config.show_waypoints:  # Nominal waypoints
                waypoints = data[f"Ship{i}"][0]["waypoints"]
                self.axes[0].plot(
                    waypoints[1, :],
                    waypoints[0, :],
                    color=c,
                    marker="o",
                    markersize=15,
                    linestyle="--",
                    linewidth=lw,
                    alpha=0.3,
                )

        self.artists = vessels + trajectories

        def init():
            # ax_map.set_xlim(x_lim[0], x_lim[1])
            # ax_map.set_ylim(y_lim[0], y_lim[1])
            return self.artists

        def update(i):
            for j in range(n_ships):
                x_i = data[f"Ship{j}"][i]["pose"][0]
                y_i = data[f"Ship{j}"][i]["pose"][1]
                chi_i = data[f"Ship{j}"][i]["pose"][3]
                length = ship_list[j].length
                width = ship_list[j].width

                # Update vessel visualization
                ship_poly = mapf.create_ship_polygon(x_i, y_i, chi_i, length, width, 2.0)
                y_ship, x_ship = ship_poly.exterior.xy
                vessels[j].set_xy(np.array([y_ship, x_ship]).T)

                # Update trajectory visualization
                trajectories[j].set_xdata([*trajectories[j].get_xdata(), y_i])
                trajectories[j].set_ydata([*trajectories[j].get_ydata(), x_i])

                # TODO: Update predicted trajectory visualization

                # TODO: Update own-ship safety zone visualization

                # TODO: Update obstacle tracks visualization

            self.artists = vessels + trajectories
            return self.artists

        plt.legend(loc="upper right")

        anim = animation.FuncAnimation(
            self.fig,
            func=update,
            init_func=init,
            frames=len(times) - 1,
            repeat=False,
            interval=self._config.frame_delay,
            blit=True,
        )

        if self._config.show_animation:
            self.wait_fig()

        if self._config.save_animation:
            anim.save(save_path, writer="ffmpeg", progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"))

    def wait_fig(self) -> None:
        # Block the execution of the code until the figure is closed.
        # This works even with multiprocessing.
        if plt.isinteractive():
            plt.ioff()  # this is necessary in mutliprocessing
            plt.show(block=True)
            plt.ion()  # restitute the interractive state
        else:
            plt.show(block=True)
        return

    def visualize_results(
        self, enc: ENC, ship_list: list, sim_data: DataFrame, sim_times: np.ndarray, t_snapshots: Optional[list] = None, save_figs: bool = True
    ) -> Tuple[list, list]:
        """Visualize the results of a scenario simulation.

        Args:
            ship_list (list): List of ships in the simulation.
            sim_data (list): List of simulation data.

        Returns:
            Tuple[list, list]: List of figure and axes handles
        """
        n_samples = len(sim_times)
        if t_snapshots is None:
            t_snapshots = [sim_times[n_samples / 5], sim_times[3 * n_samples / 5], sim_times[4 * n_samples / 5]]
        figs = []
        axes = []
        fig_map, ax_map = plt.subplots(1, 1, figsize=(10, 10))
        ax_map.set_xlabel("East [m]")
        ax_map.set_ylabel("North [m]")

        mapf.plot_background(ax_map, enc)

        ship_lw = self._config.ship_linewidth

        xlimits = [1e10, -1e10]
        ylimits = [1e10, -1e10]

        n_ships = len(ship_list)

        t_snapshots = []
        for k in range(self._config.n_snapshots):
            t_snapshots.append(sim_times[])


        for i, ship in enumerate(ship_list):
            ship_sim_data = sim_data[f"Ship{i}"]

            ship_color = self._config.ship_colors[i]
            ax_map.plot(
                ship.waypoints[1, :],
                ship.waypoints[0, :],
                "o",
                color=ship_color,
                marker="o",
                markersize=15,
                linestyle="None",
                linewidth=ship_lw,
                alpha=0.3,
            )

            X = extract_trajectory_data_from_dataframe(ship_sim_data)

            for k in range(self._config.n_snapshots):

            ax_map.plot(X[1, :], X[0, :], color=ship_color, linewidth=ship_lw, label=f"Ship {i}")

            do_estimates, do_covariances = extract_track_data_from_dataframe(ship_sim_data)
            for j, do_estimates_j in enumerate(do_estimates):
                do_color = self._config.do_colors[j]

                ax_map.plot(
                    do_estimates_j[1, :],
                    do_estimates_j[0, :],
                    color=do_color,
                    linewidth=ship_lw,
                    linestyle="--",
                    alpha=0.3,
                )

            xlimits, ylimits = update_xy_limits_from_trajectory_data(X, xlimits, ylimits)

        ax_map.set_xlim(ylimits)
        ax_map.set_ylim(xlimits)
        figs.append(fig_map)
        axes.append(ax_map)

        # plot NE with vessel trajectories at multiple time steps

        # plot own-ship DO tracking results (NIS, RMSE, etc.)

        # plot own-ship trajectory tracking results ()

        # save results in figs as .eps and .fig

        return figs, axes


def extract_trajectory_data_from_dataframe(ship_df: DataFrame) -> np.ndarray:
    """Extract the trajectory data from a ship dataframe.

    Args:
        ship_df (Dataframe): Dataframe containing the ship simulation data.

    Returns:
        np.ndarray: Array containing the trajectory data.
    """
    X = np.zeros((4, len(ship_df)))
    for k, ship_df_k in enumerate(ship_df):
        X[:, k] = ship_df_k["pose"]
    return X


def extract_track_data_from_dataframe(ship_df: DataFrame) -> Tuple[list, list]:
    """Extract the dynamic obstacle track data from a ship dataframe.

    Args:
        ship_df (Dataframe): Dataframe containing the ship simulation data.

    Returns:
        Tuple[list, list]: List of dynamic obstacle estimates and covariances
    """
    do_estimates = []
    do_covariances = []

    n_samples = len(ship_df)
    n_do = len(ship_df[n_samples - 1]["do_states"])

    for i in range(n_do):
        do_estimates.append(np.zeros((4, n_samples)))
        do_covariances.append(np.zeros((4, 4, n_samples)))

        for k, ship_df in enumerate(ship_df):
            do_estimates[i][:, k] = ship_df["do_states"][i]
            do_covariances[i][:, :, k] = ship_df["do_covariances"][i]

    return do_estimates, do_covariances


def update_xy_limits_from_trajectory_data(X: np.ndarray, xlimits: list, ylimits: list) -> Tuple[np.ndarray, np.ndarray]:
    """Update the x and y limits from the trajectory data.

    Args:
        X (np.ndarray): Trajectory data.
        xlimits (list): List containing the x limits.
        ylimits (list): List containing the y limits.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y limits.
    """

    min_x = np.min(X[0, :])
    max_x = np.max(X[0, :])
    min_y = np.min(X[1, :])
    max_y = np.max(X[1, :])

    if min_x < xlimits[0]:
        xlimits[0] = min_x

    if max_x > xlimits[1]:
        xlimits[1] = max_x

    if min_y < ylimits[0]:
        ylimits[0] = min_y

    if max_y > ylimits[1]:
        ylimits[1] = max_y

    return xlimits, ylimits
