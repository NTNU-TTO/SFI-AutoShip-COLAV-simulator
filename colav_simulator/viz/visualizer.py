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
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import ShapelyFeature
from matplotlib_scalebar.scalebar import ScaleBar
from pandas import DataFrame
from scipy.stats import chi2, norm
from seacharts.enc import ENC
from shapely.geometry import Polygon


@dataclass
class Config:
    """Configuration class for specifying the look of the visualization."""

    show_liveplot: bool = True
    show_results: bool = True
    show_measurements: bool = False
    show_tracks: bool = True
    show_waypoints: bool = False
    show_animation: bool = True
    save_animation: bool = False
    n_snapshots: int = 3  # number of scenario snapshots to show in trajectory result plotting
    frame_delay: float = 200.0
    figsize: list = field(default_factory=lambda: [12, 10])
    margins: list = field(default_factory=lambda: [0.0, 0.0])
    ship_linewidth: float = 0.9
    ship_scaling: list = field(default_factory=lambda: [5.0, 2.0])
    ship_info_fontsize: int = 13
    ship_colors: list = field(
        default_factory=lambda: [
            "xkcd:black",
            "xkcd:red",
            "xkcd:eggshell",
            "xkcd:purple",
            "xkcd:cyan",
            "xkcd:orange",
            "xkcd:fuchsia",
            "xkcd:yellow",
            "xkcd:grey",
            "xkcd:reddish brown",
            "xkcd:bubblegum",
            "xkcd:baby shit brown",
            "xkcd:khaki",
            "xkcd:cloudy blue",
            "xkcd:pale aqua",
            "xkcd:light lilac",
            "xkcd:lemon",
            "xkcd:powder blue",
            "xkcd:wine",
            "xkcd:amber",
            "xkcd:wheat",
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
            "xkcd:wine",
            "xkcd:amber",
            "xkcd:wheat",
            "xkcd:burnt sienna",
            "xkcd:barbie pink",
            "xkcd:ugly brown",
            "xkcd:light tan",
            "xkcd:stormy blue",
            "xkcd:light aquamarine",
            "xkcd:lemon",
            "xkcd:pastel blue",
            "xkcd:blue green",
            "xkcd:eggshell",
            "xkcd:purple",
            "xkcd:cyan",
            "xkcd:orange",
            "xkcd:fuchsia",
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
    misc_plt_handles: dict  # Extra handles used for live plotting
    t_start: float = 0.0  # start time of the visualization of one scenario

    def __init__(self, enc: Optional[ENC] = None, config_file: Path = dp.visualizer_config) -> None:
        self._config = cp.extract(Config, config_file, dp.visualizer_schema)

        if enc:
            self.init_figure(enc)

    def init_figure(self, enc: ENC, extent: Optional[list] = None) -> None:
        plt.close()

        self.fig = plt.figure("Simulation Live Plot", figsize=self._config.figsize)

        ax_map = self.fig.add_subplot(projection=enc.crs)
        mapf.plot_background(ax_map, enc)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])

        if extent is not None:
            ax_map.set_extent(extent, crs=enc.crs)
        ax_map.gridlines(draw_labels=True, dms=True, color="gray", linewidth=1.5, linestyle="--", alpha=0.3, x_inline=False, y_inline=False)
        plt.ion()

        self.misc_plt_handles = {}
        self.ship_plt_handles = []

        ax_map.add_artist(ScaleBar(1, units="m", location="lower left", frameon=False, color="white", box_alpha=0.0, pad=0.5, font_properties={"size": 12}))
        self.axes = [ax_map]
        plt.show(block=False)

    def clear_ship_handles(self) -> None:
        """Clears the handles for the ships in the live visualization."""

        if "time" in self.misc_plt_handles:
            self.misc_plt_handles["time"].remove()

        for ship_i_handle in self.ship_plt_handles:
            if "patch" in ship_i_handle:
                ship_i_handle["patch"].remove()

            if "info" in ship_i_handle:
                ship_i_handle["info"].remove()

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

        # Find the limits of the map, based on the own-ship trajectory
        xlimits = [1e10, -1e10]
        ylimits = [1e10, -1e10]
        if ship_list[0].trajectory.size > 0:
            buffer = 1000
            xlimits, ylimits = mhm.update_xy_limits_from_trajectory_data(ship_list[0].trajectory, xlimits, ylimits)
        elif ship_list[0].waypoints.size > 0:
            buffer = 2000
            xlimits, ylimits = mhm.update_xy_limits_from_trajectory_data(ship_list[0].waypoints, xlimits, ylimits)
        buffer = 2000
        xlimits = [xlimits[0] - buffer, xlimits[1] + buffer]
        ylimits = [ylimits[0] - buffer, ylimits[1] + buffer]

        self.init_figure(enc, [ylimits[0], ylimits[1], xlimits[0], xlimits[1]])
        ax_map = self.axes[0]
        self.background = self.fig.canvas.copy_from_bbox(ax_map.bbox)

        n_ships = len(ship_list)
        for i, ship in enumerate(ship_list):

            # If number of ships is greater than 16, use the same color for all target ships
            if i > 0 and n_ships > len(self._config.ship_colors):
                c = self._config.ship_colors[1]
            else:
                c = self._config.ship_colors[i]

            if i == 0:
                zorder_patch = 4
            else:
                zorder_patch = 3

            lw = self._config.ship_linewidth

            ship_i_handles: dict = {}

            if i == 0:
                ship_name = "OS"

                do_lw = self._config.do_linewidth

                if self._config.show_tracks:
                    ship_i_handles["do_tracks"] = []
                    ship_i_handles["do_covariances"] = []
                    ship_i_handles["track_started"] = []
                    for j in range(1, n_ships):
                        ship_i_handles["track_started"].append(False)

                        if n_ships > len(self._config.ship_colors):
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
                                label=f"DO {j - 1} est. traj.",
                                transform=enc.crs,
                                zorder=zorder_patch - 2,
                            )[0]
                        )

                        ship_i_handles["do_covariances"].append(
                            ax_map.add_feature(
                                ShapelyFeature(
                                    [],
                                    linewidth=lw,
                                    color=do_c,
                                    alpha=0.5,
                                    label=f"DO {j - 1} est. 3sigma cov.",
                                    crs=enc.crs,
                                    zorder=zorder_patch - 2,
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
                                marker=".",
                                markersize=8,
                                # label="Radar meas.",
                                transform=enc.crs,
                                zorder=zorder_patch - 3,
                            )[0]
                        elif sensor.type == "ais":
                            ship_i_handles["ais"] = ax_map.plot(
                                [0.0],
                                [0.0],
                                color=self._config.ais_color,
                                linewidth=lw,
                                linestyle="None",
                                marker="*",
                                markersize=10,
                                label="AIS meas.",
                                transform=enc.crs,
                                zorder=zorder_patch - 3,
                            )[0]

            else:
                print("i = {}".format(i))
                ship_name = "DO " + str(i - 1)

            ship_i_handles["info"] = ax_map.text(
                0.0,
                0.0,
                ship_name,  # + " | mmsi:" + str(ship.mmsi),
                color=c,
                transform=enc.crs,
                fontsize=self._config.ship_info_fontsize,
                verticalalignment="center",
                horizontalalignment="center",
                zorder=5,
            )

            ship_i_handles["patch"] = ax_map.add_feature(ShapelyFeature([], edgecolor="k", facecolor=c, linewidth=lw, label="", crs=enc.crs, zorder=zorder_patch))

            # Add 0.0 to data to avoid matplotlib error when plotting empty trajectory
            ship_i_handles["trajectory"] = ax_map.plot([0.0], [0.0], color=c, linewidth=lw, label=ship_name + " true traj.", transform=enc.crs, zorder=zorder_patch - 2)[
                0
            ]

            ship_i_handles["predicted_trajectory"] = ax_map.plot(
                [0.0],
                [0.0],
                color=c,
                linewidth=lw,
                marker="*",
                linestyle="-",
                label="",  # ship_name + " pred. traj.",
                transform=enc.crs,
                zorder=zorder_patch - 2,
            )[0]

            if self._config.show_waypoints and ship.waypoints.size > 0:
                ship_i_handles["waypoints"] = ax_map.plot(
                    ship.waypoints[1, :],
                    ship.waypoints[0, :],
                    color=c,
                    marker=".",
                    markersize=8,
                    linestyle="dashed",
                    linewidth=lw,
                    alpha=0.3,
                    label=ship_name + " waypoints",
                    transform=enc.crs,
                    zorder=-6,
                )[0]

            ship_i_handles["ship_started"] = False
            self.ship_plt_handles.append(ship_i_handles)

        extent = ax_map.get_extent()
        self.misc_plt_handles["time"] = ax_map.text(
            extent[0] + 1000,
            extent[3] - 500,
            "t = 0.0 s",
            transform=enc.crs,
            fontsize=13,
            verticalalignment="top",
            horizontalalignment="left",
            zorder=10,
        )

    def update_live_plot(self, t: float, enc: ENC, ship_list: list, sensor_measurements: list) -> None:
        """Updates the live plot with the current data of the ships in the simulation.

        Args:
            t (float): Current time in the simulation.
            enc (ENC): ENC object containing the map data.
            ship_list (list): List of configured ships in the simulation.
            sensor_measurements (list): Most recent sensor measurements generated from the own-ship sensors.
        """
        if not self._config.show_liveplot:
            return

        self.fig.canvas.restore_region(self.background)

        self.misc_plt_handles["time"].set_text(f"t = {t:.2f} s")

        ax_map = self.axes[0]
        n_ships = len(ship_list)
        for i, ship_obj in enumerate(ship_list):
            if t < ship_obj.t_start or t > ship_obj.t_end:
                continue

            # Hack to avoid ValueError from matplotlib, see previous function for more info
            if self.ship_plt_handles[i]["ship_started"]:
                start_idx_ship_line_data = 0
            else:
                start_idx_ship_line_data = 1
                self.ship_plt_handles[i]["ship_started"] = True

            if i == 0:
                zorder_patch = 4
            else:
                zorder_patch = 3

            pose_i = ship_obj.pose

            # If number of ships is greater than 16, use the same color for all target ships
            if i > 0 and n_ships > len(self._config.ship_colors):
                c = self._config.ship_colors[1]
            else:
                c = self._config.ship_colors[i]

            lw = self._config.ship_linewidth

            if i == 0:
                do_estimates, do_covariances, _, do_labels = ship_obj.get_do_track_information()
                if self._config.show_tracks and len(do_estimates) > 0:
                    lw = self._config.do_linewidth
                    for j in range(len(do_estimates)):  # pylint: disable=consider-using-enumerate
                        if n_ships > len(self._config.ship_colors):
                            do_c = self._config.do_colors[1]
                        else:
                            do_c = self._config.do_colors[do_labels[j] - 1]

                        if self.ship_plt_handles[i]["track_started"][do_labels[j] - 1]:
                            start_idx_track_line_data = 0
                        else:
                            start_idx_track_line_data = 1
                            self.ship_plt_handles[i]["track_started"][do_labels[j] - 1] = True

                        self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].set_xdata(
                            [
                                *self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].get_xdata()[start_idx_track_line_data:],
                                do_estimates[j][1],
                            ]
                        )
                        self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].set_ydata(
                            [
                                *self.ship_plt_handles[i]["do_tracks"][do_labels[j] - 1].get_ydata()[start_idx_track_line_data:],
                                do_estimates[j][0],
                            ]
                        )

                        ellipse_x, ellipse_y = mhm.create_probability_ellipse(do_covariances[j], 0.99)
                        ell_geometry = Polygon(zip(ellipse_y + do_estimates[j][1], ellipse_x + do_estimates[j][0]))
                        if self.ship_plt_handles[i]["do_covariances"][do_labels[j] - 1] is not None:
                            self.ship_plt_handles[i]["do_covariances"][do_labels[j] - 1].remove()
                        self.ship_plt_handles[i]["do_covariances"][do_labels[j] - 1] = ax_map.add_feature(
                            ShapelyFeature(
                                [ell_geometry],
                                linewidth=lw,
                                color=do_c,
                                alpha=0.3,
                                label=f"DO {j - 1} est. 3sigma cov.",
                                crs=enc.crs,
                                zorder=zorder_patch - 2,
                            )
                        )

                        if self._config.show_measurements:
                            for sensor_id, sensor in enumerate(ship_obj.sensors):
                                sensor_data = sensor_measurements[sensor_id]
                                if not sensor_data:
                                    continue

                                if np.isnan(sensor_data).any():
                                    continue
                                xdata = []
                                ydata = []
                                for measurements in sensor_data:
                                    for meas in measurements:
                                        xdata.append(meas[1])
                                        ydata.append(meas[0])
                                if sensor.type == "radar":
                                    self.ship_plt_handles[i]["radar"].set_xdata(xdata)
                                    self.ship_plt_handles[i]["radar"].set_ydata(ydata)

                                elif sensor.type == "ais":
                                    self.ship_plt_handles[i]["ais"].set_xdata(xdata)
                                    self.ship_plt_handles[i]["ais"].set_ydata(ydata)

            # Update ship patch
            ship_poly = mapf.create_ship_polygon(
                pose_i[0], pose_i[1], pose_i[3], ship_obj.length, ship_obj.width, self._config.ship_scaling[0], self._config.ship_scaling[1]
            )
            if self.ship_plt_handles[i]["patch"] is not None:
                self.ship_plt_handles[i]["patch"].remove()
            self.ship_plt_handles[i]["patch"] = ax_map.add_feature(ShapelyFeature([ship_poly], color=c, linewidth=lw, crs=enc.crs, zorder=zorder_patch))

            self.ship_plt_handles[i]["info"].set_x(pose_i[1] - 200)
            self.ship_plt_handles[i]["info"].set_y(pose_i[0] + 250)

            self.ship_plt_handles[i]["trajectory"].set_xdata([*self.ship_plt_handles[i]["trajectory"].get_xdata()[start_idx_ship_line_data:], pose_i[1]])
            self.ship_plt_handles[i]["trajectory"].set_ydata([*self.ship_plt_handles[i]["trajectory"].get_ydata()[start_idx_ship_line_data:], pose_i[0]])

            if self._config.show_waypoints and ship_obj.waypoints.size > 0:
                self.ship_plt_handles[i]["waypoints"].set_xdata(ship_obj.waypoints[1, :])
                self.ship_plt_handles[i]["waypoints"].set_ydata(ship_obj.waypoints[0, :])

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
        show_tracking_results: bool = False,
    ) -> Tuple[list, list]:
        """Visualize the results of a scenario simulation.

        Args:
            enc (ENC): Electronic Navigational Chart object.
            ship_list (list): List of ships in the simulation.
            sim_data (DataFrame): Dataframe of simulation.
            sim_times (list): List of simulation times.
            k_snapshots (Optional[list], optional): List of snapshots to visualize.
            save_figs (bool, optional): Whether to save the figures.
            save_file_path (Optional[Path], optional): Path to the file where the figures are saved.
            show_tracking_results (bool, optional): Whether to show tracking results or not.

        Returns:
            Tuple[list, list]: List of figure and axes handles
        """
        if not self._config.show_results:
            return [], []

        if save_file_path is None:
            save_file_path = dp.figure_output / "scenario_ne.pdf"
        else:
            save_file_path = Path(str(save_file_path) + ".pdf")

        ship_data = mhm.extract_ship_data_from_sim_dataframe(ship_list, sim_data)
        trajectory_list = ship_data["trajectory_list"]
        cpa_indices = ship_data["cpa_indices"]

        n_samples = len(sim_times)
        if k_snapshots is None:
            k_snapshots = [round(0.1 * n_samples), round(0.3 * n_samples), round(0.6 * n_samples)]

        figs = []
        axes = []
        fig_map = plt.figure("Scenario: " + str(save_file_path.stem), figsize=self._config.figsize)
        ax_map = fig_map.add_subplot(projection=enc.crs)
        mapf.plot_background(ax_map, enc)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        xlimits = [1e10, -1e10]
        ylimits = [1e10, -1e10]
        plt.show(block=False)

        figs_tracking: list = []
        axes_tracking: list = []
        ship_lw = self._config.ship_linewidth
        n_ships = len(ship_list)
        for i, ship in enumerate(ship_list):
            ship_sim_data = sim_data[f"Ship{i}"]

            # If number of ships is greater than 16, use the same color for all target ships
            if i > 0 and n_ships > len(self._config.ship_colors):
                ship_color = self._config.ship_colors[1]
            else:
                ship_color = self._config.ship_colors[i]

            X = trajectory_list[i]
            first_valid_idx, last_valid_idx = mhm.index_of_first_and_last_non_nan(X[0, :])
            if first_valid_idx == -1 and last_valid_idx == -1:
                continue

            # Plot ship trajectory and shape at all considered snapshots
            end_idx = k_snapshots[-1]
            if last_valid_idx < end_idx:
                end_idx = last_valid_idx + 1

            if end_idx < first_valid_idx:
                continue

            is_inside_map = True
            if i == 0:
                ship_name = "OS"
                xlimits, ylimits = mhm.update_xy_limits_from_trajectory_data(X[:, first_valid_idx:end_idx], xlimits, ylimits)
                xlimits = [xlimits[0] - 2000, xlimits[1] + 2000]
                ylimits = [ylimits[0] - 2000, ylimits[1] + 2000]
                zorder_patch = 4
            else:
                ship_name = "DO " + str(i - 1)
                zorder_patch = 3
                is_inside_map = mhm.check_if_trajectory_is_within_xy_limits(X[:, first_valid_idx:end_idx], xlimits, ylimits)
                if not is_inside_map:
                    continue

            # Plot ship nominal waypoints
            if ship.waypoints.size > 0:
                ax_map.plot(
                    ship.waypoints[1, :],
                    ship.waypoints[0, :],
                    color=ship_color,
                    marker="o",
                    markersize=4,
                    linestyle="--",
                    linewidth=ship_lw,
                    transform=enc.crs,
                    label="",
                    zorder=zorder_patch - 5,
                )

            ax_map.plot(
                X[1, first_valid_idx:end_idx],
                X[0, first_valid_idx:end_idx],
                color=ship_color,
                linewidth=ship_lw,
                label=ship_name + " traj.",
                transform=enc.crs,
                zorder=zorder_patch - 2,
            )

            # If the ship is the own-ship: Also plot dynamic obstacle tracks, and tracking results
            if i == 0:
                track_data = mhm.extract_track_data_from_dataframe(ship_sim_data)
                do_estimates = track_data["do_estimates"]
                do_covariances = track_data["do_covariances"]
                do_NISes = track_data["do_NISes"]
                do_labels = track_data["do_labels"]

                for j, do_estimates_j in enumerate(do_estimates):
                    first_valid_idx_track, last_valid_idx_track = mhm.index_of_first_and_last_non_nan(do_estimates_j[0, :])

                    end_idx_j = k_snapshots[-1]
                    if last_valid_idx_track < end_idx_j:
                        end_idx_j = last_valid_idx + 1

                    if first_valid_idx_track >= end_idx_j:
                        continue

                    do_color = self._config.do_colors[j]
                    do_lw = self._config.do_linewidth
                    do_true_states_j, _, _ = mhm.extract_trajectory_data_from_ship_dataframe(sim_data[f"Ship{do_labels[j]}"])
                    do_true_states_j = mhm.convert_sog_cog_state_to_vxvy_state(do_true_states_j)

                    # ax_map.plot(
                    #     do_estimates_j[1, first_valid_idx_track:end_idx_j],
                    #     do_estimates_j[0, first_valid_idx_track:end_idx_j],
                    #     color=do_color,
                    #     linewidth=ship_lw,
                    #     transform=enc.crs,
                    #     label=f"DO {do_labels[j] -1} est. traj.",
                    #     zorder=zorder_patch - 2,
                    # )
                    for k in k_snapshots:
                        if k < first_valid_idx_track or k > end_idx_j:
                            continue

                        # ellipse_x, ellipse_y = mhm.create_probability_ellipse(do_covariances[j][:2, :2, k], 0.99)
                        # ell_geometry = Polygon(zip(ellipse_y + do_estimates_j[1, k], ellipse_x + do_estimates_j[0, k]))
                        # ax_map.add_feature(
                        #     ShapelyFeature(
                        #         [ell_geometry],
                        #         linewidth=do_lw,
                        #         color=do_color,
                        #         alpha=0.3,
                        #         label=f"DO {do_labels[j] - 1} est. cov.",
                        #         crs=enc.crs,
                        #         zorder=zorder_patch - 2,
                        #     )
                        # )

                    if show_tracking_results:
                        fig_do_j, axes_do_j = self._plot_do_tracking_results(
                            sim_times[first_valid_idx_track:end_idx_j],
                            do_true_states_j[:, first_valid_idx_track:end_idx_j],
                            do_estimates_j[:, first_valid_idx_track:end_idx_j],
                            do_covariances[j][:, :, first_valid_idx_track:end_idx_j],
                            do_NISes[j],
                            j,
                            do_lw,
                        )
                        figs_tracking.append(fig_do_j)
                        axes_tracking.append(axes_do_j)

            count = 1
            for k in k_snapshots:
                if k < first_valid_idx or k > end_idx:
                    count += 1
                    continue

                ship_poly = mapf.create_ship_polygon(
                    x=X[0, k],
                    y=X[1, k],
                    heading=X[3, k],
                    length=ship.length,
                    width=ship.width,
                    length_scaling=2 * self._config.ship_scaling[0],
                    width_scaling=2 * self._config.ship_scaling[1],
                )
                ax_map.add_feature(
                    ShapelyFeature(
                        [ship_poly],
                        linewidth=ship_lw,
                        color=ship_color,
                        # label=ship_name",
                        crs=enc.crs,
                        zorder=zorder_patch,
                    )
                )
                ax_map.text(
                    X[1, k] - 150,
                    X[0, k] + 200,
                    f"$t_{count}$",
                    fontsize=12,
                    zorder=zorder_patch + 1,
                )
                count += 1

        ax_map.set_extent([ylimits[0], ylimits[1], xlimits[0], xlimits[1]], crs=enc.crs)
        ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax_map.add_artist(ScaleBar(1, units="m", location="lower left", frameon=False, color="white", box_alpha=0.0, pad=0.5, font_properties={"size": 12}))

        plt.legend()

        if save_figs:
            if not save_file_path.parents[0].exists():
                save_file_path.parents[0].mkdir(parents=True)
            fig_map.savefig(save_file_path, format="pdf", dpi=500, bbox_inches="tight")

        figs.append(fig_map)
        axes.append(ax_map)
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


# #
