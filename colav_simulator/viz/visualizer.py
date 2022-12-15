"""
    vizualizer.py

    Summary:
        Contains functionality for visualizing/animating ship scenarios.

    Author: Trym Tengesdal, Magne Aune, Melih Akdag, Joachim Miller
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import ShapelyFeature
from pandas import DataFrame
from scipy.stats import chi2, norm
from seacharts.enc import ENC
from shapely.geometry import Polygon


@dataclass
class Config:
    """Configuration class for specifying the look of the visualization."""

    show_liveplot: bool = True
    show_measurements: bool = False
    show_tracks: bool = True
    show_waypoints: bool = False
    show_animation: bool = True
    show_results: bool = True
    save_animation: bool = False
    n_snapshots: int = 3  # number of scenario snapshots to show in trajectory result plotting
    frame_delay: float = 200.0
    figsize: list = field(default_factory=lambda: [12, 10])
    margins: list = field(default_factory=lambda: [0.01, 0.01])
    ship_linewidth: float = 0.9
    ship_scaling: float = 5.0
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
            "xkcd:pale lilac",
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
            "xkcd:very dark green",
            "xkcd:pastel blue",
            "xkcd:blue green",
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
    t_start: float = 0.0  # start time of the visualization of one scenario

    def __init__(self, enc: Optional[ENC] = None, config_file: Path = dp.visualizer_config) -> None:
        self._config = cp.extract(Config, config_file, dp.visualizer_schema)

        if enc:
            self.init_figure(enc)

    def init_figure(self, enc: ENC) -> None:
        plt.close()

        self.fig = plt.figure("Simulation Live Plot", figsize=self._config.figsize)

        ax_map = self.fig.add_subplot(projection=enc.crs)
        mapf.plot_background(ax_map, enc)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        ax_map.set_xlabel("East [m]")
        ax_map.set_ylabel("North [m]")
        ax_map.set_xticks(ax_map.get_xticks())
        ax_map.set_yticks(ax_map.get_yticks())
        plt.ion()

        # dark_mode_color = "#142c38"
        # self.fig.set_facecolor(dark_mode_color)

        self.ship_plt_handles = []
        self.axes = [ax_map]
        plt.show(block=False)

    def clear_ship_handles(self) -> None:
        """Clears the handles for the ships in the live visualization."""
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

    def init_live_plot(self, enc: ENC, ship_list: list) -> None:
        """Initializes the plot handles of the live plot for a simulation
        given by the ship list.

        Args:
            enc (ENC): ENC object containing the map data.
            ship_list (list): List of configured ships in the simulation.
        """
        if not self._config.show_liveplot:
            return

        self.init_figure(enc)
        ax_map = self.axes[0]
        self.background = self.fig.canvas.copy_from_bbox(ax_map.bbox)

        xlimits = [1e10, -1e10]
        ylimits = [1e10, -1e10]

        n_ships = len(ship_list)
        for i, ship in enumerate(ship_list):

            # If number of ships is greater than 16, use the same color for all target ships
            if i > 0 and n_ships > 16:
                c = self._config.ship_colors[1]
            else:
                c = self._config.ship_colors[i]

            lw = self._config.ship_linewidth

            ship_i_handles: dict = {}

            if ship.trajectory.size > 0:
                xlimits, ylimits = update_xy_limits_from_trajectory_data(ship.trajectory, xlimits, ylimits)
            elif ship.waypoints.size > 0:
                xlimits, ylimits = update_xy_limits_from_trajectory_data(ship.waypoints, xlimits, ylimits)

            if i == 0:
                ship_name = "OS"

                do_lw = self._config.do_linewidth

                if self._config.show_tracks:
                    ship_i_handles["do_tracks"] = []
                    ship_i_handles["do_covariances"] = []
                    for j in range(1, n_ships):

                        # If number of ships is greater than 16, use the same color for all target ships
                        if n_ships > 16:
                            do_c = self._config.do_colors[1]
                        else:
                            do_c = self._config.do_colors[j]

                        # Add 0.0 to data to avoid matplotlib error when plotting empty trajectory
                        ship_i_handles["do_tracks"].append(
                            ax_map.plot(
                                [0.0],
                                [0.0],
                                linewidth=do_lw,
                                color=do_c,
                                label=f"DO{j} est. traj.",
                                transform=enc.crs,
                            )[0]
                        )

                        ship_i_handles["do_covariances"].append(
                            ax_map.add_feature(
                                ShapelyFeature(
                                    [],
                                    linewidth=lw,
                                    color=do_c,
                                    alpha=0.6,
                                    label=f"DO{j} est. cov.",
                                    crs=enc.crs,
                                )
                            )
                        )

                if self._config.show_measurements:
                    for sensor in ship.sensors:
                        if sensor.type == "radar":
                            ship_i_handles["radar"] = ax_map.plot(
                                [0.0],
                                [0.0],
                                color=self._config.radar_color,
                                linewidth=lw,
                                linestyle="None",
                                marker="o",
                                markersize=8,
                                label=ship_name + "Radar meas.",
                                transform=enc.crs,
                            )[0]
                        elif sensor.type == "ais":
                            ship_i_handles["ais"] = ax_map.plot(
                                [0.0],
                                [0.0],
                                color=self._config.ais_color,
                                linewidth=lw,
                                linestyle="None",
                                marker="*",
                                markersize=8,
                                label="AIS meas.",
                                transform=enc.crs,
                            )[0]

            else:
                ship_name = "DO " + str(i - 1)

            ship_i_handles["patch"] = ax_map.add_feature(
                ShapelyFeature([], color=c, linewidth=lw, label="", crs=enc.crs)
            )

            # Add 0.0 to data to avoid matplotlib error when plotting empty trajectory
            ship_i_handles["trajectory"] = ax_map.plot(
                [0.0], [0.0], color=c, linewidth=lw, label=ship_name + " true traj.", transform=enc.crs
            )[0]

            ship_i_handles["predicted_trajectory"] = ax_map.plot(
                [0.0],
                [0.0],
                color=c,
                linewidth=lw,
                marker="*",
                linestyle="-",
                label=ship_name + " pred. traj.",
                transform=enc.crs,
            )[0]

            if self._config.show_waypoints and ship.waypoints.size > 0:
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
                    transform=enc.crs,
                )[0]

            ship_i_handles["started"] = False
            self.ship_plt_handles.append(ship_i_handles)

        xlimits = [xlimits[0] - 500, xlimits[1] + 500]
        ylimits = [ylimits[0] - 500, ylimits[1] + 500]
        # ax_map.set_extent([ylimits[0], ylimits[1], xlimits[0], xlimits[1]], crs=enc.crs)
        self.background = self.fig.canvas.copy_from_bbox(ax_map.bbox)

    def update_live_plot(self, t: float, enc: ENC, ship_list: list, sensor_measurements: list) -> None:
        """Updates the live plot with the current data of the ships in the simulation.

        Args:
            t (float): Current time in the simulation.
            enc (ENC): ENC object containing the map data.
            ship_list (list): List of configured ships in the simulation.
            sensor_measurements (list): List of sensor measurements generated from each ship at the current time.
        """
        if not self._config.show_liveplot:
            return

        self.fig.canvas.restore_region(self.background)

        ax_map = self.axes[0]
        n_ships = len(ship_list)
        for i, ship_obj in enumerate(ship_list):
            if ship_obj.t_start > t:
                continue

            # Hack to avoid ValueError from matplotlib, see previous function for more info
            if self.ship_plt_handles[i]["started"]:
                start_idx_line_data = 0
            else:
                start_idx_line_data = 1

            pose_i = ship_obj.pose

            # If number of ships is greater than 16, use the same color for all target ships
            if i > 0 and n_ships > 16:
                c = self._config.ship_colors[1]
            else:
                c = self._config.ship_colors[i]

            lw = self._config.ship_linewidth

            if i == 0:
                do_estimates, do_covariances, _, do_labels = ship_obj.get_do_track_information()
                if self._config.show_tracks and len(do_estimates) > 0:
                    lw = self._config.do_linewidth
                    for j in range(len(do_estimates)):  # pylint: disable=consider-using-enumerate
                        # If number of ships is greater than 16, use the same color for all target ships
                        if n_ships > 16:
                            do_c = self._config.do_colors[1]
                        else:
                            do_c = self._config.do_colors[do_labels[j] - 1]

                        self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].set_xdata(
                            [
                                *self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].get_xdata()[
                                    start_idx_line_data:
                                ],
                                do_estimates[j][1],
                            ]
                        )
                        self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].set_ydata(
                            [
                                *self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].get_ydata()[
                                    start_idx_line_data:
                                ],
                                do_estimates[j][0],
                            ]
                        )

                        ellipse_x, ellipse_y = mhm.create_probability_ellipse(do_covariances[j], 0.99)
                        ell_geometry = Polygon(zip(ellipse_y, ellipse_x))
                        if self.ship_plt_handles[i]["do_covariances"][do_labels[j] - 1] is not None:
                            self.ship_plt_handles[i]["do_covariances"][do_labels[j] - 1].remove()
                        self.ship_plt_handles[i]["do_covariances"][do_labels[j] - 1] = ax_map.add_feature(
                            ShapelyFeature(
                                [ell_geometry],
                                linewidth=lw,
                                color=do_c,
                                alpha=0.6,
                                label=f"DO{do_labels[j]} est. cov.",
                                crs=enc.crs,
                            )
                        )

                        if self._config.show_measurements and len(sensor_measurements[i]) > 0:
                            for sensor_id, sensor in enumerate(ship_obj.sensors):
                                sensor_data = sensor_measurements[i][j - 1][sensor_id]

                                if any(np.isnan(sensor_data)):
                                    continue

                                if sensor.type == "radar":
                                    self.ship_plt_handles[i]["radar"].set_xdata(
                                        [
                                            *self.ship_plt_handles[i]["radar"].get_xdata()[start_idx_line_data:],
                                            sensor_data[1],
                                        ]
                                    )
                                    self.ship_plt_handles[i]["radar"].set_ydata(
                                        [
                                            *self.ship_plt_handles[i]["radar"].get_ydata()[start_idx_line_data:],
                                            sensor_data[0],
                                        ]
                                    )

                                elif sensor.type == "ais":
                                    self.ship_plt_handles[i]["ais"].set_xdata(
                                        [
                                            *self.ship_plt_handles[i]["ais"].get_xdata()[start_idx_line_data:],
                                            sensor_data[1],
                                        ]
                                    )
                                    self.ship_plt_handles[i]["ais"].set_ydata(
                                        [
                                            *self.ship_plt_handles[i]["ais"].get_ydata()[start_idx_line_data:],
                                            sensor_data[0],
                                        ]
                                    )

            # Update ship patch
            ship_poly = mapf.create_ship_polygon(
                pose_i[0], pose_i[1], pose_i[3], ship_obj.length, ship_obj.width, self._config.ship_scaling
            )
            if self.ship_plt_handles[i]["patch"] is not None:
                self.ship_plt_handles[i]["patch"].remove()
            self.ship_plt_handles[i]["patch"] = ax_map.add_feature(
                ShapelyFeature([ship_poly], color=c, linewidth=lw, crs=enc.crs)
            )

            self.ship_plt_handles[i]["trajectory"].set_xdata(
                [*self.ship_plt_handles[i]["trajectory"].get_xdata()[start_idx_line_data:], pose_i[1]]
            )
            self.ship_plt_handles[i]["trajectory"].set_ydata(
                [*self.ship_plt_handles[i]["trajectory"].get_ydata()[start_idx_line_data:], pose_i[0]]
            )

            if self._config.show_waypoints and ship_obj.waypoints.size > 0:
                self.ship_plt_handles[i]["waypoints"].set_xdata(ship_obj.waypoints[1, :])
                self.ship_plt_handles[i]["waypoints"].set_ydata(ship_obj.waypoints[0, :])

            self.ship_plt_handles[i]["started"] = True
            # TODO: Update predicted ship trajectory

        if n_ships < 3:  # to avoid cluttering the legend
            plt.legend(loc="upper right")
        self.fig.canvas.blit(ax_map.bbox)

        self.fig.canvas.flush_events()

        # tx = "Mean Frame Rate:\n {fps:.3f}FPS".format(fps=((i + 1) / (time.time() - self.t_start)))
        # plt.pause(0.00001)  # plt.pause(self._config.frame_delay / 1000)

    def visualize_results(
        self,
        enc: ENC,
        ship_list: list,
        sim_data: DataFrame,
        sim_times: np.ndarray,
        k_snapshots: Optional[list] = None,
        save_figs: bool = True,
        save_file_path: Optional[Path] = None,
    ) -> Tuple[list, list]:
        """Visualize the results of a scenario simulation.

        Args:
            enc (ENC): Electronic Navigational Chart object.
            ship_list (list): List of ships in the simulation.
            sim_data (list): List of simulation data.
            sim_times (list): List of simulation times.
            k_snapshots (Optional[list], optional): List of snapshots to visualize.
            save_figs (bool, optional): Whether to save the figures.
            save_file_path (Optional[Path], optional): Path to the file where the figures are saved.

        Returns:
            Tuple[list, list]: List of figure and axes handles
        """
        n_samples = len(sim_times)
        if k_snapshots is None:
            k_snapshots = [round(n_samples / 5), round(3 * n_samples / 5), round(4 * n_samples / 5)]

        figs = []
        axes = []
        fig_map = plt.figure("Scenario", figsize=self._config.figsize)
        ax_map = fig_map.add_subplot(projection=enc.crs)
        mapf.plot_background(ax_map, enc)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        ax_map.set_xlabel("East [m]")
        ax_map.set_ylabel("North [m]")
        ax_map.set_xticks(ax_map.get_xticks())
        ax_map.set_yticks(ax_map.get_yticks())
        xlimits = [1e10, -1e10]
        ylimits = [1e10, -1e10]

        plt.ion()

        figs_tracking: list = []
        axes_tracking: list = []
        ship_lw = self._config.ship_linewidth
        for i, ship in enumerate(ship_list):
            ship_sim_data = sim_data[f"Ship{i}"]
            ship_color = self._config.ship_colors[i]
            X = extract_trajectory_data_from_dataframe(ship_sim_data)
            xlimits, ylimits = update_xy_limits_from_trajectory_data(X, xlimits, ylimits)

            if i == 0:
                ship_name = "OS"
            else:
                ship_name = "DO " + str(i - 1)

            # Plot ship nominal waypoints
            if ship.waypoints.size > 0:
                ax_map.plot(
                    ship.waypoints[1, :],
                    ship.waypoints[0, :],
                    color=ship_color,
                    marker="o",
                    markersize=12,
                    linestyle="None",
                    linewidth=ship_lw,
                    alpha=0.3,
                )

            # Plot ship trajectory and shape at all considered snapshots
            ax_map.plot(
                X[1, : k_snapshots[-1]],
                X[0, : k_snapshots[-1]],
                color=ship_color,
                linewidth=ship_lw,
                label=ship_name + " traj.",
            )

            count = 1
            for k in k_snapshots:
                ship_poly = mapf.create_ship_polygon(
                    X[0, k], X[1, k], X[3, k], ship.length, ship.width, self._config.ship_scaling
                )
                y_ship, x_ship = ship_poly.exterior.xy
                ax_map.fill(y_ship, x_ship, color=ship_color, linewidth=ship_lw, label="")
                ax_map.text(X[1, k] - 30, X[0, k] + 40, f"$t_{count}$", fontsize=12)
                count += 1

            # If the ship is the own-ship: Also plot dynamic obstacle tracks, and tracking results
            if i == 0:
                track_data = extract_track_data_from_dataframe(ship_sim_data)
                do_estimates = track_data["do_estimates"]
                do_covariances = track_data["do_covariances"]
                do_NISes = track_data["do_NISes"]
                do_labels = track_data["do_labels"]

                for j, do_estimates_j in enumerate(do_estimates):
                    first_valid_idx, last_valid_idx = mhm.index_of_first_and_last_non_nan(do_estimates_j[0, :])

                    do_color = self._config.do_colors[j]
                    do_lw = self._config.do_linewidth
                    do_true_states_j = extract_trajectory_data_from_dataframe(sim_data[f"Ship{do_labels[j]}"])
                    do_true_states_j = mhm.convert_sog_cog_state_to_vxvy_state(do_true_states_j)

                    ax_map.plot(
                        do_estimates_j[1, first_valid_idx : k_snapshots[-1]],
                        do_estimates_j[0, first_valid_idx : k_snapshots[-1]],
                        color=do_color,
                        linewidth=ship_lw,
                        linestyle="--",
                        alpha=0.3,
                    )
                    for k in k_snapshots:
                        if k < first_valid_idx or k > last_valid_idx:
                            continue

                        ellipse_x, ellipse_y = mhm.create_probability_ellipse(do_covariances[j][:2, :2, k], 0.99)
                        ax_map.fill(
                            ellipse_y + do_estimates_j[1, k],
                            ellipse_x + do_estimates_j[0, k],
                            color=do_color,
                            alpha=0.5,
                            linewidth=do_lw,
                        )

                    fig_do_j, axes_do_j = self._plot_do_tracking_results(
                        sim_times,
                        do_true_states_j,
                        do_estimates_j,
                        do_covariances[j],
                        do_NISes[j],
                        j,
                        do_lw,
                    )
                    figs_tracking.append(fig_do_j)
                    axes_tracking.append(axes_do_j)

        xlimits = [xlimits[0] - 100, xlimits[1] + 100]
        ylimits = [ylimits[0] - 100, ylimits[1] + 100]
        ax_map.set_extent([xlimits[0], xlimits[1], ylimits[0], ylimits[1]])

        figs.append(fig_map)
        axes.append(ax_map)

        plt.show()

        if save_figs:
            if save_file_path is None:
                save_file_path = dp.animation_output / ("scenario_ne" + "eps")
            fig_map.savefig(save_file_path, format="eps", dpi=1000)

        return figs, axes

    def _plot_do_tracking_results(
        self,
        sim_times: np.ndarray,
        do_true_states: np.ndarray,
        do_estimates: np.ndarray,
        do_covariances: np.ndarray,
        do_NIS: np.ndarray,
        do_idx: int,
        do_lw: float = 1.0,
        confidence_level: float = 0.95,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the tracking results of a specific dynamic obstacle (DO).

        Args:
            axes (plt.Axes): Axes handle.
            sim_times (np.ndarray): Simulation times.
            do_true_states (np.ndarray): True states of the DO
            do_estimates (np.ndarray): Estimated states of the DO
            do_covariances (np.ndarray): Covariances of the DO.
            do_NIS (np.ndarray): Normalized Innovation error Squared (NIS) values of the DO.
            do_idx (int): Index of the DO.
            do_lw (float, optional): Line width of the DO. Defaults to 1.0.
            confidence_level (float, optional): Confidence level considered for the uncertainty plotting. Defaults to 0.95.

        Returns:
            plt.Figure, plt.Axes: Figure and axes handles for the DO <do_idx> tracking results.
        """
        fig = plt.figure(num="Tracking results DO" + str(do_idx), figsize=(10, 10))
        axes = fig.subplot_mosaic(
            [
                ["x", "y"],
                ["Vx", "Vy"],
                ["NIS", "errs"],
            ]
        )

        z_val = norm.ppf(confidence_level)
        axes["x"].plot(sim_times, do_estimates[0, :], color="xkcd:blue", linewidth=do_lw, label="estimate")
        axes["x"].plot(sim_times, do_true_states[0, :], color="xkcd:red", linewidth=do_lw, label="true")
        std_x = np.sqrt(do_covariances[0, 0, :])
        axes["x"].fill_between(
            sim_times,
            do_estimates[0, :] - z_val * std_x,
            do_estimates[0, :] + z_val * std_x,
            color="xkcd:blue",
            alpha=0.3,
        )
        axes["x"].set_xlabel("Time [s]")
        axes["x"].set_ylabel("North [m]")
        axes["x"].legend()

        axes["y"].plot(sim_times, do_estimates[1, :], color="xkcd:blue", linewidth=do_lw, label="estimate")
        axes["y"].plot(sim_times, do_true_states[1, :], color="xkcd:red", linewidth=do_lw, label="true")
        std_y = np.sqrt(do_covariances[1, 1, :])
        axes["y"].fill_between(
            sim_times,
            do_estimates[1, :] - z_val * std_y,
            do_estimates[1, :] + z_val * std_y,
            color="xkcd:blue",
            alpha=0.3,
        )
        axes["y"].set_xlabel("Time [s]")
        axes["y"].set_ylabel("East [m]")
        axes["y"].legend()

        axes["Vx"].plot(sim_times, do_estimates[2, :], color="xkcd:blue", linewidth=do_lw, label="estimate")
        axes["Vx"].plot(sim_times, do_true_states[2, :], color="xkcd:red", linewidth=do_lw, label="true")
        std_Vx = np.sqrt(do_covariances[2, 2, :])
        axes["Vx"].fill_between(
            sim_times,
            do_estimates[2, :] - z_val * std_Vx,
            do_estimates[2, :] + z_val * std_Vx,
            color="xkcd:blue",
            alpha=0.3,
        )
        axes["Vx"].set_xlabel("Time [s]")
        axes["Vx"].set_ylabel("North speed [m/s]")
        axes["Vx"].legend()

        axes["Vy"].plot(sim_times, do_estimates[3, :], color="xkcd:blue", linewidth=do_lw, label="estimate")
        axes["Vy"].plot(sim_times, do_true_states[3, :], color="xkcd:red", linewidth=do_lw, label="true")
        std_Vy = np.sqrt(do_covariances[3, 3, :])
        axes["Vy"].fill_between(
            sim_times,
            do_estimates[3, :] - z_val * std_Vy,
            do_estimates[3, :] + z_val * std_Vy,
            color="xkcd:blue",
            alpha=0.3,
        )
        axes["Vy"].set_xlabel("Time [s]")
        axes["Vy"].set_ylabel("East speed [m/s]")
        axes["Vy"].legend()

        alpha = 0.05
        CI2 = np.array(chi2.ppf(q=[alpha / 2, 1 - alpha / 2], df=2))

        inCIpos = np.mean(np.multiply(np.less_equal(do_NIS, CI2[1]), np.greater_equal(do_NIS, CI2[0])) * 100)
        print(f"DO{do_idx}: {inCIpos}% of estimates inside {(1 - alpha) * 100} CI")
        axes["NIS"].plot(
            CI2[0] * np.ones(len(do_NIS)),
            color="xkcd:red",
            linewidth=do_lw,
            linestyle="--",
            label="Confidence bounds",
        )
        axes["NIS"].plot(
            CI2[1] * np.ones(len(do_NIS)),
            color="xkcd:red",
            linewidth=do_lw,
            linestyle="--",
            label="",
        )
        axes["NIS"].plot(do_NIS, color="xkcd:blue", linewidth=do_lw, label="NIS")
        axes["NIS"].set_ylabel("NIS")
        axes["NIS"].legend()

        error = do_true_states - do_estimates
        pos_error = np.sqrt(error[0, :] ** 2 + error[1, :] ** 2)
        vel_error = np.sqrt(error[2, :] ** 2 + error[3, :] ** 2)
        axes["errs"].plot(sim_times, pos_error, color="xkcd:blue", linewidth=do_lw, label="pos. error")
        axes["errs"].plot(sim_times, vel_error, color="xkcd:red", linewidth=do_lw, label="vel. error")
        axes["errs"].set_xlabel("Time [s]")
        axes["errs"].legend()

        return fig, axes

    def toggle_ship_legend_visibility(self, idx: int, state: bool) -> None:
        """Toggle the visibility of the plot items for the ship with index idx.

        Args:
            state (bool): Visibility state of the legend.
        """
        if "patch" in self.ship_plt_handles[idx]:
            self.ship_plt_handles[idx]["patch"].set_visible(state)

        if "trajectory" in self.ship_plt_handles[idx]:
            self.ship_plt_handles[idx]["trajectory"].set_visible(state)

        if "predicted_trajectory" in self.ship_plt_handles[idx]:
            self.ship_plt_handles[idx]["predicted_trajectory"].set_visible(state)

        if "waypoints" in self.ship_plt_handles[idx]:
            self.ship_plt_handles[idx]["waypoints"].set_visible(state)

        if "do_tracks" in self.ship_plt_handles[idx]:
            for do_track in self.ship_plt_handles[idx]["do_tracks"]:
                do_track.set_visible(state)

            for do_covariance in self.ship_plt_handles[idx]["do_covariances"]:
                do_covariance.set_visible(state)

            if "radar" in self.ship_plt_handles[idx]:
                self.ship_plt_handles[idx]["radar"].set_visible(state)

            if "ais" in self.ship_plt_handles[idx]:
                self.ship_plt_handles[idx]["ais"].set_visible(state)


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


def extract_track_data_from_dataframe(ship_df: DataFrame) -> dict:
    """Extract the dynamic obstacle track data from a ship dataframe.

    Args:
        ship_df (Dataframe): Dataframe containing the ship simulation data.

    Returns:
        Tuple[list, list]: List of dynamic obstacle estimates and covariances
    """
    output = {}
    do_estimates = []
    do_covariances = []
    do_NISes = []

    n_samples = len(ship_df)
    n_do = len(ship_df[n_samples - 1]["do_estimates"])
    do_labels = ship_df[n_samples - 1]["do_labels"]

    for i in range(n_do):
        do_estimates.append(np.nan * np.ones((4, n_samples)))
        do_covariances.append(np.nan * np.ones((4, 4, n_samples)))
        do_NISes.append(np.nan * np.ones(n_samples))

    for i in range(n_do):
        for k, ship_df_k in enumerate(ship_df):
            for idx, _ in enumerate(ship_df_k["do_labels"]):
                do_estimates[idx][:, k] = ship_df_k["do_estimates"][idx]
                do_covariances[idx][:, :, k] = ship_df_k["do_covariances"][idx]
                do_NISes[idx][k] = ship_df_k["do_NISes"][idx]

    output["do_estimates"] = do_estimates
    output["do_covariances"] = do_covariances
    output["do_NISes"] = do_NISes
    output["do_labels"] = do_labels
    return output


def update_xy_limits_from_trajectory_data(trajectory: np.ndarray, xlimits: list, ylimits: list) -> Tuple[list, list]:
    """Update the x and y limits from the trajectory data (either predefined trajectory or nominal trajectory/waypoints for the ship).

    Args:
        X (np.ndarray): waypoint data.
        xlimits (list): List containing the x limits.
        ylimits (list): List containing the y limits.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y limits.
    """

    min_x = np.min(trajectory[0, :])
    max_x = np.max(trajectory[0, :])
    min_y = np.min(trajectory[1, :])
    max_y = np.max(trajectory[1, :])

    if min_x < xlimits[0]:
        xlimits[0] = min_x

    if max_x > xlimits[1]:
        xlimits[1] = max_x

    if min_y < ylimits[0]:
        ylimits[0] = min_y

    if max_y > ylimits[1]:
        ylimits[1] = max_y

    return xlimits, ylimits


#     def visualize(
#         self,
#         ship_list: list,
#         data: DataFrame,
#         times: np.ndarray,
#         save_path: Optional[Path] = None,
#     ) -> None:
#         """Visualize the ship trajectories.

#         Args:
#             enc (ENC): Electronic Navigational Chart object.
#             ship_list: List of ships in the simulation.
#             data (DataFrame): Pandas DataFrame containing the ship simulation data.
#             times (List/np.ndarray): List/array of times to consider.
#             save_path (Optional[pathlib.Path]): Path to file where the animation is saved. Defaults to None.
#         """

#         vessels = []
#         trajectories = []

#         # data: n_ships x 1 dataframe, where each ship entry contains n_samples x 1 dictionaries of pose, waypoints, speed plans etc..
#         n_ships = len(data.columns)
#         for i in range(n_ships):
#             c = self._config.ship_colors[i]
#             lw = self._config.ship_linewidth
#             if i == 0:
#                 vessels.append(plt.fill([], [], color=c, linewidth=lw, label="")[0])
#                 trajectories.append(plt.plot([], [], c, label="OS traj.")[0])
#             else:
#                 vessels.append(plt.fill([], [], color=c, linewidth=lw, label="")[0])
#                 trajectories.append(plt.plot([], [], color=c, linewidth=lw, label=f"DO {i - 1} true traj.")[0])
#             if self._config.show_waypoints:  # Nominal waypoints
#                 waypoints = data[f"Ship{i}"][0]["waypoints"]
#                 self.axes[0].plot(
#                     waypoints[1, :],
#                     waypoints[0, :],
#                     color=c,
#                     marker="o",
#                     markersize=15,
#                     linestyle="--",
#                     linewidth=lw,
#                     alpha=0.3,
#                 )

#         self.artists = vessels + trajectories

#         def init():
#             # ax_map.set_xlim(x_lim[0], x_lim[1])
#             # ax_map.set_ylim(y_lim[0], y_lim[1])
#             return self.artists

#         def update(i):
#             for j in range(n_ships):
#                 x_i = data[f"Ship{j}"][i]["pose"][0]
#                 y_i = data[f"Ship{j}"][i]["pose"][1]
#                 chi_i = data[f"Ship{j}"][i]["pose"][3]
#                 length = ship_list[j].length
#                 width = ship_list[j].width

#                 # Update vessel visualization
#                 ship_poly = mapf.create_ship_polygon(x_i, y_i, chi_i, length, width, 2.0)
#                 y_ship, x_ship = ship_poly.exterior.xy
#                 vessels[j].set_xy(np.array([y_ship, x_ship]).T)

#                 # Update trajectory visualization
#                 trajectories[j].set_xdata([*trajectories[j].get_xdata(), y_i])
#                 trajectories[j].set_ydata([*trajectories[j].get_ydata(), x_i])

#                 # TODO: Update predicted trajectory visualization

#                 # TODO: Update own-ship safety zone visualization

#                 # TODO: Update obstacle tracks visualization

#             self.artists = vessels + trajectories
#             return self.artists

#         plt.legend(loc="upper right")

#         anim = animation.FuncAnimation(
#             self.fig,
#             func=update,
#             init_func=init,
#             frames=len(times) - 1,
#             repeat=False,
#             interval=self._config.frame_delay,
#             blit=True,
#         )

#         if self._config.show_animation:
#             self.wait_fig()

#         if self._config.save_animation:
#             anim.save(save_path, writer="ffmpeg", progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"))


# #
