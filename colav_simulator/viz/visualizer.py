"""
    vizualizer.py

    Summary:
        Contains functionality for visualizing/animating ship scenarios.

    Author: Trym Tengesdal, Magne Aune, Melih Akdag, Joachim Miller
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.core.ship as ship
import matplotlib
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
    zoom_in_liveplot_on_ownship: bool = True
    show_colav_results_live: bool = True
    show_waypoints: bool = False
    show_measurements: bool = False
    show_liveplot_tracks: bool = True
    show_results: bool = True
    show_target_tracking_results: bool = True
    show_trajectory_tracking_results: bool = True
    n_snapshots: int = 3  # number of scenario shape snapshots to show in result plotting
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

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    def to_dict(self):
        output = asdict(self)
        return output


class Visualizer:
    """Class with functionality for visualizing/animating ship scenarios, and plotting/saving the simulation results.

    Internal variables:
        - fig (Figure): Matplotlib figure object for the visualization.
        - axes (list): List of matplotlib axes objects for the visualization.

    """

    _config: Config
    fig: plt.figure  # handle to figure for live plotting
    axes: list  # handle to axes for live plotting
    background: Any  # map background in live plotting fig
    ship_plt_handles: list  # handles used for live plotting
    misc_plt_handles: dict  # Extra handles used for live plotting
    t_start: float = 0.0  # start time of the visualization of one scenario

    def __init__(self, config: Optional[Config] = None, enc: Optional[ENC] = None) -> None:
        if config:
            self._config = config
        else:
            self._config = Config()

        if enc:
            self.init_figure(enc)

    def init_figure(self, enc: ENC, extent: Optional[list] = None) -> None:
        """Initialize the figure for live plotting.

        Args:
            - enc (ENC): ENC object for the map background.
            - extent (list): List specifying the extent of the map.
        """
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

    def find_plot_limits(self, ownship: ship.Ship, buffer: float = 500.0) -> Tuple[list, list]:
        """Finds the limits of the map, based on the own-ship trajectory

        Args:
            - ownship (ship.Ship): The own-ship object
            - buffer (float): Buffer to add to the limits

        Returns:
            Tuple[list, list]: The x and y limits of the map
        """
        xlimits = [1e10, -1e10]
        ylimits = [1e10, -1e10]
        if ownship.trajectory.size > 0:
            xlimits, ylimits = mhm.update_xy_limits_from_trajectory_data(ownship.trajectory, xlimits, ylimits)
        elif ownship.waypoints.size > 0:
            xlimits, ylimits = mhm.update_xy_limits_from_trajectory_data(ownship.waypoints, xlimits, ylimits)
        xlimits = [xlimits[0] - buffer, xlimits[1] + buffer]
        ylimits = [ylimits[0] - buffer, ylimits[1] + buffer]
        return xlimits, ylimits

    def close_live_plot(self) -> None:
        """Closes the live plot."""
        plt.close(fig=self.fig)

    def init_live_plot(self, enc: ENC, ship_list: list) -> None:
        """Initializes the plot handles of the live plot for a simulation
        given by the ship list.

        Args:
            - enc (ENC): ENC object containing the map data.
            - ship_list (list): List of configured ships in the simulation.
        """

        if not self._config.show_liveplot:
            return

        plt.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42

        xlimits, ylimits = self.find_plot_limits(ship_list[0])

        if self._config.zoom_in_liveplot_on_ownship:
            self.init_figure(enc, [ylimits[0], ylimits[1], xlimits[0], xlimits[1]])
        else:
            self.init_figure(enc)
        ax_map = self.axes[0]
        self.background = self.fig.canvas.copy_from_bbox(ax_map.bbox)

        n_ships = len(ship_list)
        for i, ship_obj in enumerate(ship_list):
            lw = self._config.ship_linewidth
            # If number of ships is greater than 16, use the same color for all target ships
            if i > 0 and n_ships > len(self._config.ship_colors):
                c = self._config.ship_colors[1]
            else:
                c = self._config.ship_colors[i]

            # Plot the own-ship (i = 0) above the target ships
            if i == 0:
                zorder_patch = 4
            else:
                zorder_patch = 3

            ship_i_handles: dict = {}
            if i == 0:
                ship_name = "OS"

                do_lw = self._config.do_linewidth

                if self._config.show_liveplot_tracks:
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
                    for sensor in ship_obj.sensors:
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
                # print("i = {}".format(i))
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

            if self._config.show_colav_results_live:
                ship_i_handles["colav_nominal_trajectory"] = ax_map.plot(
                    [0.0],
                    [0.0],
                    color=c,
                    linewidth=lw,
                    marker="",
                    markersize=4,
                    linestyle="--",
                    label=ship_name + " nom. traj.",
                    transform=enc.crs,
                    zorder=zorder_patch - 2,
                )[0]

                ship_i_handles["colav_predicted_trajectory"] = ax_map.plot(
                    [0.0],
                    [0.0],
                    color=c,
                    linewidth=lw,
                    marker="8",
                    markersize=4,
                    linestyle="dotted",
                    label=ship_name + " pred. traj.",
                    transform=enc.crs,
                    zorder=zorder_patch - 2,
                )[0]

                ship_i_handles["colav_relevant_static_obstacles"] = ax_map.add_feature(
                    ShapelyFeature([], edgecolor="k", facecolor="r", linewidth=lw, label="", crs=enc.crs, zorder=zorder_patch - 1)
                )
                ship_i_handles["colav_relevant_dynamic_obstacles"] = ax_map.add_feature(
                    ShapelyFeature([], edgecolor="k", facecolor="r", linewidth=lw, label="", crs=enc.crs, zorder=zorder_patch - 1)
                )

            if self._config.show_waypoints and ship_obj.waypoints.size > 0:
                ship_i_handles["waypoints"] = ax_map.plot(
                    ship_obj.waypoints[1, :],
                    ship_obj.waypoints[0, :],
                    color=c,
                    marker="o",
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

    def update_ownship_live_tracking_data(self, ownship: ship.Ship, sensor_measurements: list, n_ships: int, enc: ENC) -> None:
        """Updates tracking-related plots for the own-ship

        Args:
            - ownship (ship.Ship): The own-ship object
            - sensor_measurements (list): List of sensor measurements
            - n_ships (int): Number of ships in the simulation
            - enc (ENC): The ENC object
        """
        tracks: list = []
        tracks, _ = ownship.get_do_track_information()
        do_labels = [track[0] for track in tracks]
        do_estimates = [track[1] for track in tracks]
        do_covariances = [track[2] for track in tracks]
        ax_map = self.axes[0]
        zorder_patch = 4
        if self._config.show_liveplot_tracks and len(do_estimates) > 0:
            lw = self._config.do_linewidth
            for j in range(len(do_estimates)):  # pylint: disable=consider-using-enumerate
                if n_ships > len(self._config.ship_colors):
                    do_c = self._config.do_colors[1]
                else:
                    do_c = self._config.do_colors[do_labels[j] - 1]

                if self.ship_plt_handles[0]["track_started"][do_labels[j] - 1]:
                    start_idx_track_line_data = 0
                else:
                    start_idx_track_line_data = 1
                    self.ship_plt_handles[0]["track_started"][do_labels[j] - 1] = True

                self.ship_plt_handles[0]["do_tracks"][do_labels[j] - 1].set_xdata(
                    [
                        *self.ship_plt_handles[0]["do_tracks"][do_labels[j] - 1].get_xdata()[start_idx_track_line_data:],
                        do_estimates[j][1],
                    ]
                )
                self.ship_plt_handles[0]["do_tracks"][do_labels[j] - 1].set_ydata(
                    [
                        *self.ship_plt_handles[0]["do_tracks"][do_labels[j] - 1].get_ydata()[start_idx_track_line_data:],
                        do_estimates[j][0],
                    ]
                )

                ellipse_x, ellipse_y = mhm.create_probability_ellipse(do_covariances[j], 0.99)
                ell_geometry = Polygon(zip(ellipse_y + do_estimates[j][1], ellipse_x + do_estimates[j][0]))
                if self.ship_plt_handles[0]["do_covariances"][do_labels[j] - 1] is not None:
                    self.ship_plt_handles[0]["do_covariances"][do_labels[j] - 1].remove()
                self.ship_plt_handles[0]["do_covariances"][do_labels[j] - 1] = ax_map.add_feature(
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
                    for sensor_id, sensor in enumerate(ownship.sensors):
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
                            self.ship_plt_handles[0]["radar"].set_xdata(xdata)
                            self.ship_plt_handles[0]["radar"].set_ydata(ydata)

                        elif sensor.type == "ais":
                            self.ship_plt_handles[0]["ais"].set_xdata(xdata)
                            self.ship_plt_handles[0]["ais"].set_ydata(ydata)

    def update_ship_live_data(self, ship_obj: ship.Ship, idx: int, enc: ENC, **kwargs) -> None:
        """Updates the live plot with the current data of the input ship object.

        Args:
            - ship_obj (ship.Ship): The ship object to update the live plot with.
            - idx (int): The index of the ship object in the simulation.
            - enc (ENC): The ENC object.
        """
        lw = kwargs["lw"] if "lw" in kwargs else self._config.ship_linewidth
        c = kwargs["c"] if "c" in kwargs else self._config.ship_colors[idx]
        start_idx_ship_line_data = kwargs["start_idx_ship_line_data"] if "start_idx_ship_line_data" in kwargs else 0
        ax_map = self.axes[0]
        zorder_patch = 3
        csog_state = ship_obj.csog_state
        ship_poly = mapf.create_ship_polygon(
            csog_state[0], csog_state[1], csog_state[3], ship_obj.length, ship_obj.width, self._config.ship_scaling[0], self._config.ship_scaling[1]
        )
        if self.ship_plt_handles[idx]["patch"] is not None:
            self.ship_plt_handles[idx]["patch"].remove()
        self.ship_plt_handles[idx]["patch"] = ax_map.add_feature(ShapelyFeature([ship_poly], color=c, linewidth=lw, crs=enc.crs, zorder=zorder_patch))

        self.ship_plt_handles[idx]["info"].set_x(csog_state[1] - 300)
        self.ship_plt_handles[idx]["info"].set_y(csog_state[0] + 350)

        self.ship_plt_handles[idx]["trajectory"].set_xdata([*self.ship_plt_handles[idx]["trajectory"].get_xdata()[start_idx_ship_line_data:], csog_state[1]])
        self.ship_plt_handles[idx]["trajectory"].set_ydata([*self.ship_plt_handles[idx]["trajectory"].get_ydata()[start_idx_ship_line_data:], csog_state[0]])

        if self._config.show_waypoints and ship_obj.waypoints.size > 0:
            self.ship_plt_handles[idx]["waypoints"].set_xdata(ship_obj.waypoints[1, :])
            self.ship_plt_handles[idx]["waypoints"].set_ydata(ship_obj.waypoints[0, :])

        if self._config.show_colav_results_live:
            self.ship_plt_handles[idx] = ship_obj.plot_colav_results(ax_map, enc, self.ship_plt_handles[idx], **kwargs)

    def update_live_plot(self, t: float, enc: ENC, ship_list: list, sensor_measurements: list) -> None:
        """Updates the live plot with the current data of the ships in the simulation.

        Args:
            - t (float): Current time in the simulation.
            - enc (ENC): ENC object containing the map data.
            - ship_list (list): List of configured ships in the simulation.
            - sensor_measurements (list): Most recent sensor measurements generated from the own-ship sensors.
        """
        if not self._config.show_liveplot:
            return

        self.fig.canvas.restore_region(self.background)
        self.misc_plt_handles["time"].set_text(f"t = {t:.2f} s")

        ax_map = self.axes[0]
        n_ships = len(ship_list)
        self.update_ownship_live_tracking_data(ship_list[0], sensor_measurements, n_ships, enc)
        for i, ship_obj in enumerate(ship_list):
            if t < ship_obj.t_start or t > ship_obj.t_end:
                continue

            # Hack to avoid ValueError from matplotlib, see previous function for more info
            if self.ship_plt_handles[i]["ship_started"]:
                start_idx_ship_line_data = 0
            else:
                start_idx_ship_line_data = 1
                self.ship_plt_handles[i]["ship_started"] = True

            # If number of ships is greater than 16, use the same color for all target ships
            if i > 0 and n_ships > len(self._config.ship_colors):
                c = self._config.ship_colors[1]
            else:
                c = self._config.ship_colors[i]
            lw = self._config.ship_linewidth

            self.update_ship_live_data(ship_obj, i, enc, lw=lw, c=c, start_idx_ship_line_data=start_idx_ship_line_data)

        if n_ships < 3:  # to avoid cluttering the legend
            plt.legend(loc="upper right")
        self.fig.canvas.blit(ax_map.bbox)
        self.fig.canvas.flush_events()

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
            - enc (ENC): Electronic Navigational Chart object.
            - ship_list (list): List of ships in the simulation.
            - sim_data (DataFrame): Dataframe of simulation.
            - sim_times (list): List of simulation times.
            - k_snapshots (Optional[list], optional): List of snapshots to visualize.
            - save_figs (bool, optional): Whether to save the figures.
            - save_file_path (Optional[Path], optional): Path to the file where the figures are saved.

        Returns:
            Tuple[list, list]: List of figure and axes handles
        """
        if not self._config.show_results:
            return [], []

        plt.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42

        if save_file_path is None:
            save_file_path = dp.figure_output / "scenario_ne.pdf"
        else:
            save_file_path = Path(str(save_file_path) + ".pdf")

        ship_data = mhm.extract_ship_data_from_sim_dataframe(ship_list, sim_data)
        trajectory_list = ship_data["trajectory_list"]
        colav_data_list = ship_data["colav_data_list"]
        nominal_trajectory_list = []
        for colav_data in colav_data_list:
            if "nominal_trajectory" in colav_data[0]:
                nominal_trajectory_list.append(colav_data[0]["nominal_trajectory"])

        os_colav_stats = {}
        if colav_data_list[0] and "mpc_soln" in colav_data_list[0]:
            t_solve = []
            cost_vals = []
            n_iters = []
            final_residuals = []
            for os_colav_data in colav_data_list[0]:
                mpc_soln = os_colav_data["mpc_soln"]
                t_solve.append(mpc_soln["t_solve"])
                cost_vals.append(mpc_soln["cost_val"])
                n_iters.append(mpc_soln["n_iter"])
                final_residuals.append(mpc_soln["final_residuals"])
            os_colav_stats = {"t_solve": t_solve, "cost_vals": cost_vals, "n_iters": n_iters, "final_residuals": final_residuals}

        cpa_indices = ship_data["cpa_indices"]
        min_os_depth = mapf.find_minimum_depth(ship_list[0].draft, enc)

        n_samples = len(sim_times)
        if k_snapshots is None:
            k_snapshots = [round(0.09 * n_samples), round(0.25 * n_samples), round(0.6 * n_samples)]

        figs = []
        axes = []
        fig_map = plt.figure("Scenario: " + str(save_file_path.stem), figsize=self._config.figsize)
        ax_map = fig_map.add_subplot(projection=enc.crs)
        mapf.plot_background(ax_map, enc)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        xlimits, ylimits = self.find_plot_limits(ship_list[0], buffer=0.0)
        plt.show(block=False)

        figs_tracking: list = []
        axes_tracking: list = []
        ship_lw = self._config.ship_linewidth
        n_ships = len(ship_list)
        for i, ship_obj in enumerate(ship_list):
            ship_sim_data = sim_data[f"Ship{i}"]
            end_idx = k_snapshots[-1]

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
            if last_valid_idx < end_idx:
                end_idx = last_valid_idx + 1

            if end_idx < first_valid_idx:
                continue

            is_inside_map = True
            if i == 0:
                ship_name = "OS"
                zorder_patch = 4
            else:
                ship_name = "DO " + str(i - 1)
                zorder_patch = 3
                is_inside_map = mhm.check_if_trajectory_is_within_xy_limits(X[:, first_valid_idx:end_idx], xlimits, ylimits)
                if not is_inside_map:
                    continue

            # Plot ship nominal waypoints
            if ship_obj.waypoints.size > 0:
                ax_map.plot(
                    ship_obj.waypoints[1, :],
                    ship_obj.waypoints[0, :],
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

            # If the ship is the own-ship: Also plot dynamic obstacle tracks, and trajectory tracking results
            if i == 0:

                if self._config.show_trajectory_tracking_results and len(nominal_trajectory_list) > 0:
                    self.plot_trajectory_tracking_results(i, sim_times, X, nominal_trajectory_list[i], linewidth=1.0)

                track_data = mhm.extract_track_data_from_dataframe(ship_sim_data)
                do_estimates = track_data["do_estimates"]
                do_covariances = track_data["do_covariances"]
                do_NISes = track_data["do_NISes"]
                do_labels = track_data["do_labels"]

                # Plot distance to own-ship
                self.plot_obstacle_distances_to_ownship(sim_times, trajectory_list, do_estimates, do_covariances, do_labels, min_os_depth, enc)

                for j, do_estimates_j in enumerate(do_estimates):
                    first_valid_idx_track, last_valid_idx_track = mhm.index_of_first_and_last_non_nan(do_estimates_j[0, :])

                    end_idx_j = k_snapshots[-1]
                    if last_valid_idx_track < end_idx_j:
                        end_idx_j = last_valid_idx + 1

                    if first_valid_idx_track >= end_idx_j:
                        continue

                    do_color = self._config.do_colors[j]
                    do_lw = self._config.do_linewidth
                    do_true_states_j = trajectory_list[do_labels[j]]
                    do_true_states_j = mhm.convert_csog_state_to_vxvy_state(do_true_states_j)

                    ax_map.plot(
                        do_estimates_j[1, first_valid_idx_track:end_idx_j],
                        do_estimates_j[0, first_valid_idx_track:end_idx_j],
                        color=do_color,
                        linewidth=ship_lw,
                        transform=enc.crs,
                        label=f"DO {do_labels[j] -1} est. traj.",
                        zorder=zorder_patch - 2,
                    )

                    # for k in k_snapshots:
                    #     if k < first_valid_idx_track or k > end_idx_j:
                    #         continue

                    #     ellipse_x, ellipse_y = mhm.create_probability_ellipse(do_covariances[j][:2, :2, k], 0.99)
                    #     ell_geometry = Polygon(zip(ellipse_y + do_estimates_j[1, k], ellipse_x + do_estimates_j[0, k]))
                    #     ax_map.add_feature(
                    #         ShapelyFeature(
                    #             [ell_geometry],
                    #             linewidth=do_lw,
                    #             color=do_color,
                    #             alpha=0.3,
                    #             label=f"DO {do_labels[j] - 1} est. cov.",
                    #             crs=enc.crs,
                    #             zorder=zorder_patch - 2,
                    #         )
                    #     )

                    if self._config.show_target_tracking_results:
                        fig_do_j, axes_do_j = self.plot_do_tracking_results(
                            i,
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

            # Plot ship shape at all considered snapshots
            count = 1
            for k in k_snapshots:
                if k < first_valid_idx or k > end_idx:
                    count += 1
                    continue

                ship_poly = mapf.create_ship_polygon(
                    x=X[0, k],
                    y=X[1, k],
                    heading=X[2, k],
                    length=ship_obj.length,
                    width=ship_obj.width,
                    length_scaling=self._config.ship_scaling[0],
                    width_scaling=self._config.ship_scaling[1],
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
                # ax_map.text(
                #     X[1, k] - 100,
                #     X[0, k] + 200,
                #     f"$t_{count}$",
                #     fontsize=12,
                #     zorder=zorder_patch + 1,
                # )
                count += 1

        ax_map.set_extent([ylimits[0], ylimits[1], xlimits[0], xlimits[1]], crs=enc.crs)
        ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax_map.add_artist(ScaleBar(1, units="m", location="lower left", frameon=False, color="white", box_alpha=0.0, pad=0.5, font_properties={"size": 12}))
        plt.legend()

        if save_figs:
            if not save_file_path.parents[0].exists():
                save_file_path.parents[0].mkdir(parents=True)
            fig_map.savefig(save_file_path, format="pdf", dpi=300, bbox_inches="tight")

        figs.append(fig_map)
        axes.append(ax_map)
        return figs, axes

    def plot_obstacle_distances_to_ownship(
        self,
        sim_times: np.ndarray,
        trajectory_list: list,
        do_estimates: list,
        do_covariances: list,
        do_labels: list,
        min_os_depth: int,
        enc: ENC,
        d_safe_so: float = 5.0,
        d_safe_do: float = 5.0,
        confidence_level: float = 0.95,
    ) -> Tuple[plt.Figure, list]:
        """Plots the obstacle (both dynamic and static) distances to the ownship.

        Args:
            - sim_times (np.ndarray): Simulation times.
            - trajectory_list (list): List of trajectories for all vessels involved in the scenario episode.
            - do_estimates (list): List of DO estimates.
            - do_covariances (list): List of DO covariances.
            - do_labels (list): List of DO labels.
            - min_os_depth (int): Minimum allowable depth for the own-ship.
            - enc (ENC): Electronic Navigational Chart object.
            - d_safe_so (float, optional): Safe distance to static obstacles to be kept by the COLAV system. Defaults to 5.0.
            - d_safe_do (float, optional): Safe distance to dynamic obstacles to be kept by the COLAV system. Defaults to 5.0.
            - confidence_level (float, optional): Confidence level for the uncertainty ellipses. Defaults to 0.95.

        Returns:
            Tuple[plt.Figure, list]: Figure and axes of the output plots.
        """

        fig = plt.figure(num="Own-ship distance to obstacles", figsize=(10, 10))
        n_do = len(do_labels)
        axes = fig.subplots(n_do + 1, 1, sharex=True)
        plt.show(block=False)

        os_traj = trajectory_list[0]
        os_en_traj = os_traj[:2, :].copy()
        os_en_traj[0, :] = os_traj[1, :]
        os_en_traj[1, :] = os_traj[0, :]
        distance_vectors = mapf.compute_distance_vectors_to_grounding(os_en_traj, min_os_depth, enc)
        dist2closest_grounding_hazard = np.linalg.norm(distance_vectors, axis=0)
        if n_do == 0:
            axes = [axes]
        # axes[0].plot(sim_times, dist2closest_grounding_hazard, "b", label="Distance to closest grounding hazard")
        # axes[0].plot(sim_times, d_safe_so * np.ones_like(sim_times), "r--", label="Minimum safety margin")
        axes[0].semilogy(sim_times, dist2closest_grounding_hazard, "b", label="Distance to closest grounding hazard")
        axes[0].semilogy(sim_times, d_safe_so * np.ones_like(sim_times), "r--", label="Minimum safety margin")
        axes[0].set_ylabel("Distance [m]")
        axes[0].set_xlabel("Time [s]")
        axes[0].legend()

        for j, (do_estimates_j, do_covariances_j) in enumerate(zip(do_estimates, do_covariances)):
            first_valid_idx_track, last_valid_idx_track = mhm.index_of_first_and_last_non_nan(do_estimates_j[0, :])

            if first_valid_idx_track >= last_valid_idx_track:
                continue

            do_true_states_j = trajectory_list[do_labels[j]]
            do_true_states_j = mhm.convert_csog_state_to_vxvy_state(do_true_states_j)

            # z_val = norm.ppf(confidence_level)
            # std_x = np.sqrt(do_covariances_j[0, 0, :])
            # axes[j + 1].fill_between(
            #     sim_times,
            #     do_estimates_j[0, :] - z_val * std_x,
            #     do_estimates_j[0, :] + z_val * std_x,
            #     color="xkcd:blue",
            #     alpha=0.3,
            # )

            est_dist2do_j = np.linalg.norm(do_estimates_j[:2, :] - os_traj[:2, :], axis=0)
            dist2do_j = np.linalg.norm(do_true_states_j[:2, :] - os_traj[:2, :], axis=0)
            # axes[j + 1].plot(sim_times, dist2do_j, "b", label=f"Distance to DO{do_labels[j]}")
            # axes[j + 1].plot(sim_times, d_safe_do * np.ones_like(sim_times), "r--", label="Minimum safety margin")
            axes[j + 1].semilogy(sim_times, dist2do_j, "b", label=f"Distance to DO{do_labels[j]}")
            axes[j + 1].semilogy(sim_times, d_safe_do * np.ones_like(sim_times), "r--", label="Minimum safety margin")
            axes[j + 1].set_ylabel("Distance [m]")
            axes[j + 1].set_xlabel("Time [s]")
            axes[j + 1].legend()

        return fig, axes

    def plot_trajectory_tracking_results(
        self, ship_idx: int, sim_times: np.ndarray, trajectory: np.ndarray, reference_trajectory: np.ndarray, linewidth: float = 1.0
    ) -> Tuple[plt.Figure, list]:
        """Plots the trajectory tracking results of a ship.

        Args:
            - sim_times (np.ndarray): Simulation times.
            - trajectory (np.ndarray): Trajectory of the ship, same length as sim_times.
            - reference_trajectory (np.ndarray): Reference trajectory of the ship.
            - ship_idx (int): Index of the ship.

        Returns:
            - Tuple[plt.Figure, list]: Figure and axes of the output plots.
        """
        n_samples = min(sim_times.shape[0], reference_trajectory.shape[1])
        fig = plt.figure(num=f"Ship{ship_idx}: Trajectory tracking results", figsize=(10, 10))
        axes = fig.subplot_mosaic(
            [
                ["x"],
                ["y"],
                ["psi"],
                ["u"],
                ["v"],
                ["r"],
            ]
        )

        axes["x"].plot(sim_times[:n_samples], trajectory[0, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual")
        axes["x"].plot(sim_times[:n_samples], reference_trajectory[0, :n_samples], color="xkcd:red", linestyle="--", linewidth=linewidth, label="nominal")
        # axes["x"].set_xlabel("Time [s]")
        axes["x"].set_ylabel("North [m]")
        axes["x"].legend()

        axes["y"].plot(sim_times[:n_samples], trajectory[1, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual")
        axes["y"].plot(sim_times[:n_samples], reference_trajectory[1, :n_samples], color="xkcd:red", linestyle="--", linewidth=linewidth, label="nominal")
        # axes["y"].set_xlabel("Time [s]")
        axes["y"].set_ylabel("East [m]")
        axes["y"].legend()

        axes["psi"].plot(sim_times[:n_samples], trajectory[2, :n_samples] * 180.0 / np.pi, color="xkcd:blue", linewidth=linewidth, label="actual")
        axes["psi"].plot(
            sim_times[:n_samples], reference_trajectory[2, :n_samples] * 180.0 / np.pi, color="xkcd:red", linestyle="--", linewidth=linewidth, label="nominal"
        )
        # axes["psi"].set_xlabel("Time [s]")
        axes["psi"].set_ylabel("Heading [deg]")
        axes["psi"].legend()

        axes["u"].plot(sim_times[:n_samples], trajectory[3, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual")
        axes["u"].plot(sim_times[:n_samples], reference_trajectory[3, :n_samples], color="xkcd:red", linestyle="--", linewidth=linewidth, label="nominal")
        # axes["u"].set_xlabel("Time [s]")
        axes["u"].set_ylabel("Surge [m/s]")
        axes["u"].legend()

        axes["v"].plot(sim_times[:n_samples], trajectory[4, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual")
        axes["v"].plot(sim_times[:n_samples], reference_trajectory[4, :n_samples], color="xkcd:red", linestyle="--", linewidth=linewidth, label="nominal")
        # axes["v"].set_xlabel("Time [s]")
        axes["v"].set_ylabel("Sway [m/s]")
        axes["v"].legend()

        axes["r"].plot(sim_times[:n_samples], trajectory[5, :n_samples] * 180.0 / np.pi, color="xkcd:blue", linewidth=linewidth, label="actual")
        axes["r"].plot(sim_times[:n_samples], reference_trajectory[5, :n_samples] * 180.0 / np.pi, linestyle="--", color="xkcd:red", linewidth=linewidth, label="nominal")
        axes["r"].set_xlabel("Time [s]")
        axes["r"].set_ylabel("Yaw [deg/s]")
        axes["r"].legend()
        plt.show(block=False)
        return fig, axes

    def plot_do_tracking_results(
        self,
        ship_idx: int,
        sim_times: np.ndarray,
        do_true_states: np.ndarray,
        do_estimates: np.ndarray,
        do_covariances: np.ndarray,
        do_NIS: np.ndarray,
        do_idx: int,
        do_lw: float = 1.0,
        confidence_level: float = 0.95,
    ) -> Tuple[plt.Figure, list]:
        """Plot the tracking (for ship <ship_idx>) results of a specific dynamic obstacle (DO).

        Args:
            ship_idx (int): Index of the ship with the tracker.
            sim_times (np.ndarray): Simulation times.
            do_true_states (np.ndarray): True states of the DO
            do_estimates (np.ndarray): Estimated states of the DO
            do_covariances (np.ndarray): Covariances of the DO.
            do_NIS (np.ndarray): Normalized Innovation error Squared (NIS) values of the DO.
            do_idx (int): Index of the DO.
            do_lw (float, optional): Line width of the DO. Defaults to 1.0.
            confidence_level (float, optional): Confidence level considered for the uncertainty plotting. Defaults to 0.95.

        Returns:
            Tuple[plt.Figure, list]: Figure and axes handles for the DO <do_idx> tracking results.
        """
        fig = plt.figure(num=f"Ship{ship_idx}: Tracking results DO" + str(do_idx), figsize=(10, 10))
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
        plt.show(block=False)
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
