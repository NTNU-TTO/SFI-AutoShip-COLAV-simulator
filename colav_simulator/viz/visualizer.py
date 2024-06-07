"""
    vizualizer.py

    Summary:
        Contains functionality for visualizing/animating ship scenarios.

    Author: Trym Tengesdal, Magne Aune, Melih Akdag, Joachim Miller
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.paths as dp
import colav_simulator.common.plotters as plotters
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stoch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import animation
from matplotlib_scalebar.scalebar import ScaleBar
from pandas import DataFrame
from scipy.stats import chi2, norm
from seacharts.enc import ENC
from shapely.geometry import Polygon

plt.rcParams["animation.convert_path"] = "/usr/bin/convert"
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


@dataclass
class Config:
    """Configuration class for specifying the look of the visualization."""

    show_liveplot: bool = True
    zoom_in_liveplot_on_ownship: bool = True  # If true, the live plot zooms in on the ownship
    zoom_window_width: float = 1000.0  # Width of the zoom window in meters
    show_liveplot_colav_results: bool = True  # If true, the COLAV results are shown in the live plot
    show_liveplot_target_waypoints: bool = False  # If true, the target waypoints are shown in the live plot
    show_liveplot_ownship_waypoints: bool = False  # If true, the ownship waypoints are shown in the live plot
    show_liveplot_ownship_trajectory: bool = (
        True  # If true, the ownship (ground truth) trajectory is shown in the live plot
    )
    show_liveplot_ground_truth_target_pose: bool = (
        False  # If true, the ground truth target pose is shown in the live plot, otherwise the estimated pose is shown
    )
    show_liveplot_disturbances: bool = (
        False  # If true, the disturbances are shown in the live plot (if any currents and/or wind)
    )
    show_liveplot_measurements: bool = False
    show_liveplot_target_ships: bool = True  # If true, the target ships are shown in the live plot
    show_liveplot_target_tracks: bool = True  # If true, the target tracks are shown in the live plot
    show_liveplot_target_trajectories: bool = (
        True  # If true, the target (ground truth) trajectories are shown in the live plot
    )
    show_liveplot_map_axes: bool = False
    show_liveplot_scalebar: bool = True
    show_liveplot_time: bool = True
    show_results: bool = True
    show_target_tracking_results: bool = True
    show_trajectory_tracking_results: bool = True
    dark_mode_liveplot: bool = True
    update_rate_liveplot: float = 1.0  # Update rate of the live plot in Hz
    save_result_figures: bool = False
    save_liveplot_animation: bool = False
    n_snapshots: int = 3  # number of scenario shape snapshots to show in result plotting
    figsize: list = field(default_factory=lambda: [12, 10])
    margins: list = field(default_factory=lambda: [0.0, 0.0])
    uniform_seabed_color: bool = True
    black_land: bool = True
    black_shore: bool = True
    disable_ship_labels: bool = True
    ownship_trajectory_color: str = "xkcd:pink"
    ownship_waypoint_color: str = "xkcd:yellow"
    target_trajectory_color: str = "xkcd:peach"
    target_waypoint_color: str = "xkcd:orange"
    ship_linewidth: float = 0.9
    ship_scaling: list = field(default_factory=lambda: [5.0, 2.0])
    ship_info_fontsize: int = 13
    ship_colors: list = field(
        default_factory=lambda: [
            "xkcd:green",
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
    """Class with functionality for visualizing/animating ship scenarios, and plotting/saving the simulation results."""

    def __init__(self, config: Optional[Config] = None, enc: Optional[ENC] = None) -> None:
        if config:
            self._config: Config = config
        else:
            self._config = Config()

        self.fig: plt.figure = None  # handle to figure for live plotting
        self.axes: list = []  # handle to axes for live plotting
        self.ship_plt_handles: list = []  # handles used for live plotting
        self.misc_plt_handles: dict = {}  # Extra handles used for live plotting
        self.background: Any = None  # background for live plotting

        self.xlimits = [-1e10, 1e10]
        self.ylimits = [-1e10, 1e10]

        if enc:
            self.xlimits = [enc.bbox[1], enc.bbox[3]]
            self.ylimits = [enc.bbox[0], enc.bbox[2]]
            self.init_figure(enc, [self.ylimits[0], self.ylimits[1], self.xlimits[0], self.xlimits[1]])

        self._t_prev_update = 0.0
        self.t_start = 0.0
        self.frames = []

    def toggle_liveplot_visibility(self, show: bool) -> None:
        """Toggles the visibility of the live plot."""
        self._config.show_liveplot = show

    def toggle_liveplot_axis_labels(self, show: bool) -> None:
        """Toggles the visibility of the axis labels in the live plot."""
        for ax in self.axes:
            if show:
                ax.set_xlabel("Easting [m]")
                ax.set_ylabel("Northing [m]")
            else:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    def toggle_results_visibility(self, show: bool) -> None:
        """Toggles the visibility of the result plots."""
        self._config.show_results = show

    def set_update_rate(self, update_rate: float) -> None:
        """Sets the update rate of the live plot."""
        assert self._config.show_liveplot, "Live plot must be enabled to set this parameter"
        self._config.update_rate_liveplot = update_rate

    def init_figure(self, enc: ENC, extent: list, fignum: Optional[int] = None) -> None:
        """Initialize the figure for live plotting.

        Args:
            - enc (ENC): ENC object for the map background.
            - extent (list): List specifying the extent of the map.
        """
        self.frames = []
        self.fig = plt.figure(num=fignum, figsize=self._config.figsize)

        ax_map = self.fig.add_subplot(1, 1, 1)
        plotters.plot_background(
            ax_map,
            enc,
            dark_mode=self._config.dark_mode_liveplot,
            uniform_seabed_color=self._config.uniform_seabed_color,
            land_color="black" if self._config.black_land else None,
            shore_color="black" if self._config.black_shore else None,
        )
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        # plt.ion()

        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])

        self.misc_plt_handles = {}
        self.ship_plt_handles = []
        self.scalebar = None
        if self._config.show_liveplot_scalebar:
            self.scalebar = ax_map.add_artist(
                ScaleBar(
                    1,
                    units="m",
                    location="lower left",
                    frameon=False,
                    color="white",
                    box_alpha=0.0,
                    pad=0.5,
                    font_properties={"size": 12},
                )
            )
        self.axes = [ax_map]
        ax_map.set_aspect("equal")
        self.toggle_liveplot_axis_labels(self._config.show_liveplot_map_axes)
        plt.show(block=False)
        self.fig.tight_layout(pad=0.0)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    def find_plot_limits(self, enc: ENC, ownship: ship.Ship, buffer: float = 500.0) -> Tuple[list, list]:
        """Finds the limits of the map, based on the own-ship trajectory

        Args:
            - enc (ENC): ENC object containing the map data
            - ownship (ship.Ship): The own-ship object
            - buffer (float): Buffer to add to the limits

        Returns:
            Tuple[list, list]: The x and y limits of the map
        """
        xlimits = [-1e10, 1e10]
        ylimits = [-1e10, 1e10]
        if ownship.trajectory.size > 0:
            xlimits, ylimits = mhm.update_xy_limits_from_trajectory_data(ownship.trajectory, xlimits, ylimits)
        elif ownship.waypoints.size > 0:
            xlimits, ylimits = mhm.update_xy_limits_from_trajectory_data(ownship.waypoints, xlimits, ylimits)
        xlimits = [xlimits[0] - buffer, xlimits[1] + buffer]
        ylimits = [ylimits[0] - buffer, ylimits[1] + buffer]

        enc_ymin, enc_xmin, enc_ymax, enc_xmax = enc.bbox
        xlimits = [max(xlimits[0], enc_xmin), min(xlimits[1], enc_xmax)]
        ylimits = [max(ylimits[0], enc_ymin), min(ylimits[1], enc_ymax)]
        return xlimits, ylimits

    def close_live_plot(self) -> None:
        """Closes the live plot."""
        if self._config.show_liveplot:
            plt.close(fig=self.fig)

    def get_live_plot_image(self) -> np.ndarray:
        """Returns the live plot image as a numpy array."""
        if not self._config.show_liveplot:
            raise ValueError("Live plot is not enabled")

        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def init_live_plot(self, enc: ENC, ship_list: List[ship.Ship], fignum: Optional[int] = None) -> None:
        """Initializes the plot handles of the live plot for a simulation
        given by the ship list.

        Args:
            - enc (ENC): ENC object containing the map data.
            - ship_list (list): List of configured ships in the simulation.
            - fignum (int, optional): Figure number for the live plot.
        """
        if not self._config.show_liveplot:
            return

        # if self.fig is not None:
        #     self.fig.clf(fignum)
        #     self.background = None

        self._t_prev_update = 0.0
        plt.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42

        self.xlimits, self.ylimits = self.find_plot_limits(enc, ship_list[0])
        self.init_figure(enc, [self.ylimits[0], self.ylimits[1], self.xlimits[0], self.xlimits[1]], fignum=fignum)
        if self._config.zoom_in_liveplot_on_ownship:
            self.zoom_in_live_plot_on_ownship(enc, ship_list[0].csog_state)

        ax_map = self.axes[0]
        self.background = self.fig.canvas.copy_from_bbox(ax_map.bbox)

        n_ships = len(ship_list)
        for i, ship_obj in enumerate(ship_list):
            lw = self._config.ship_linewidth
            # If number of ships is greater than 16, use the same color for all target ships
            ship_color = self._config.ship_colors[0] if i == 0 else self._config.do_colors[0]

            # Plot the own-ship (i = 0) above the target ships
            if i == 0:
                zorder_patch = 4
            else:
                zorder_patch = 3

            ship_i_handles: dict = {}
            if i == 0:
                ship_name = "OS"
                do_c = self._config.do_colors[0]

                do_lw = self._config.do_linewidth
                ship_i_handles["do_track_poses"] = []
                ship_i_handles["do_tracks"] = []
                ship_i_handles["do_covariances"] = []
                ship_i_handles["track_started"] = []
                for j in range(1, n_ships):
                    ship_i_handles["track_started"].append(False)

                    if not self._config.show_liveplot_ground_truth_target_pose:
                        ship_i_handles["do_track_poses"].append(
                            ax_map.fill([], [], color=do_c, linewidth=lw, label="", zorder=zorder_patch - 1)[0]
                        )

                    if self._config.show_liveplot_target_tracks:
                        # Add 0.0 to data to avoid matplotlib error when plotting empty trajectory
                        ship_i_handles["do_tracks"].append(
                            ax_map.plot(
                                [0.0],
                                [0.0],
                                linewidth=do_lw,
                                color=do_c,
                                label=f"DO {j - 1} est. traj.",
                                zorder=zorder_patch - 2,
                            )[0]
                        )

                        ship_i_handles["do_covariances"].append(
                            ax_map.fill(
                                [],
                                [],
                                linewidth=lw,
                                color=do_c,
                                alpha=0.5,
                                label=f"DO {j - 1} est. 1sigma cov.",
                                zorder=zorder_patch - 2,
                            )[0]
                        )

                if self._config.show_liveplot_measurements:
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
                                label="Radar meas.",
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
                                zorder=zorder_patch - 3,
                            )[0]

            else:
                # print("i = {}".format(i))
                ship_name = "DO " + str(i - 1)

            if not self._config.disable_ship_labels:
                ship_i_handles["info"] = ax_map.text(
                    0.0,
                    0.0,
                    ship_name,  # + " | mmsi:" + str(ship.mmsi),
                    color=ship_color,
                    fontsize=self._config.ship_info_fontsize,
                    verticalalignment="center",
                    horizontalalignment="center",
                    zorder=5,
                    label="",
                )

            ship_i_handles["ground_truth_patch"] = None
            if ship_obj.id == 0 or (ship_obj.id > 0 and self._config.show_liveplot_ground_truth_target_pose):
                ship_i_handles["ground_truth_patch"] = ax_map.fill(
                    [], [], edgecolor="k", facecolor=ship_color, linewidth=lw, label="", zorder=zorder_patch
                )[0]

            # Add 0.0 to data to avoid matplotlib error when plotting empty trajectory
            ship_i_handles["trajectory"] = None
            if (ship_obj.id == 0 and self._config.show_liveplot_ownship_trajectory) or (
                ship_obj.id > 0 and self._config.show_liveplot_target_trajectories
            ):
                traj_color = (
                    self._config.ownship_trajectory_color if ship_obj.id == 0 else self._config.target_trajectory_color
                )
                ship_i_handles["trajectory"] = ax_map.plot(
                    [0.0],
                    [0.0],
                    color=traj_color,
                    linewidth=lw,
                    label=ship_name + " true traj.",
                    zorder=zorder_patch - 2,
                )[0]

            if self._config.show_liveplot_colav_results:
                ship_i_handles["colav_nominal_trajectory"] = ax_map.plot(
                    [0.0],
                    [0.0],
                    color=ship_color,
                    linewidth=lw,
                    marker="8",
                    markersize=4,
                    linestyle="dotted",
                    label=ship_name + " nom. traj.",
                    zorder=zorder_patch - 2,
                )[0]

                ship_i_handles["colav_predicted_trajectory"] = ax_map.plot(
                    [0.0],
                    [0.0],
                    color=ship_color,
                    linewidth=lw,
                    marker="",
                    markersize=4,
                    linestyle="--",
                    label=ship_name + " pred. traj.",
                    zorder=zorder_patch - 2,
                )[0]

                # ship_i_handles["colav_relevant_static_obstacles"] = ax_map.add_feature(
                #     ShapelyFeature(
                #         [], edgecolor="k", facecolor="r", linewidth=lw, label="", zorder=zorder_patch - 1
                #     )
                # )
                # ship_i_handles["colav_relevant_dynamic_obstacles"] = ax_map.add_feature(
                #     ShapelyFeature(
                #         [], edgecolor="k", facecolor="r", linewidth=lw, label="", zorder=zorder_patch - 1
                #     )
                # )

            if (
                (ship_obj.id == 0 and self._config.show_liveplot_ownship_waypoints)
                or (ship_obj.id > 0 and self._config.show_liveplot_target_waypoints)
            ) and ship_obj.waypoints.size > 0:
                waypoint_color = (
                    self._config.ownship_waypoint_color if ship_obj.id == 0 else self._config.target_waypoint_color
                )
                path_poly = mapf.create_path_polygon(
                    ship_obj.waypoints, point_buffer=2, disk_buffer=4, hole_buffer=2, show_annuluses=True
                )
                ship_i_handles["waypoints"] = ax_map.plot(
                    *path_poly.exterior.xy,
                    # ship_obj.waypoints[1, :],
                    # ship_obj.waypoints[0, :],
                    color=waypoint_color,
                    linewidth=1,
                    alpha=0.6,
                    label=ship_name + " waypoints",
                    zorder=-6,
                )[0]

            ship_i_handles["ship_started"] = False
            self.ship_plt_handles.append(ship_i_handles)

        if self._config.show_liveplot_time:
            ylim = ax_map.get_xlim()  # easting
            xlim = ax_map.get_ylim()  # northing
            self.misc_plt_handles["time"] = ax_map.text(
                ylim[0] + 30,
                xlim[0] + 150,
                "t = 0.0 s",
                fontsize=15,
                color="white",
                verticalalignment="top",
                horizontalalignment="left",
                zorder=10,
                label="",
            )

        if self._config.show_liveplot_disturbances:
            corner_offset = (-70, -70)
            self.misc_plt_handles["disturbance"] = {}
            dhandles = {
                "currents": {
                    "arrow": ax_map.quiver([], [], [], [], color="blue", scale=1000, zorder=10),
                    "text": ax_map.text(
                        ylim[1] + corner_offset[0] - 50,
                        xlim[1] + corner_offset[1] - 85,
                        "Currents: 0.0 m/s",
                        fontsize=10,
                        color="white",
                        verticalalignment="top",
                        horizontalalignment="left",
                        zorder=10,
                        label="",
                    ),
                },
                "wind": {
                    "arrow": ax_map.quiver([], [], [], [], color="yellow", scale=1000, zorder=10),
                    "text": ax_map.text(
                        ylim[1] + corner_offset[0] - 50,
                        xlim[1] + corner_offset[1] - 105,
                        "Wind: 0.0 m/s",
                        fontsize=10,
                        color="yellow",
                        verticalalignment="top",
                        horizontalalignment="left",
                        zorder=10,
                        label="",
                    ),
                },
                "circle": ax_map.fill(
                    [],
                    [],
                    linewidth=1.0,
                    color="white",
                    alpha=0.2,
                    label="",
                    zorder=zorder_patch - 2,
                )[0],
            }
            self.misc_plt_handles["disturbance"] = dhandles

        plt.tight_layout()
        # self.frames.append(self.get_live_plot_image())
        # if n_ships < 3:  # to avoid cluttering the legend
        #     plt.legend(loc="upper right")

    def update_disturbance_live_data(self, ax_map: plt.Axes, w: Optional[stoch.DisturbanceData]) -> None:
        """Updates the disturbance-related plots in the live plot.

        Args:
            ax_map (plt.Axes): The axes object of the live plot.
        """
        if not self._config.show_liveplot_disturbances:
            return

        assert "disturbance" in self.misc_plt_handles, "Disturbance handles not initialized"
        dhandles = self.misc_plt_handles["disturbance"]
        ylim = ax_map.get_xlim()  # easting
        xlim = ax_map.get_ylim()  # northing
        arrow_scale = 60
        circ_x, circ_y = mhm.create_circle(75, 100)
        corner_offset = (-100, -100)
        circ_poly = Polygon(zip(circ_y + ylim[1] + corner_offset[0], circ_x + xlim[1] + corner_offset[1]))
        dhandles["circle"].remove()
        dhandles["circle"] = ax_map.fill(*circ_poly.exterior.xy, color="white", alpha=0.2, zorder=10, label="")[0]
        if w is not None and w.currents is not None:
            speed = w.currents["speed"]
            direction = w.currents["direction"]
            dhandles["currents"]["text"].remove()
            dhandles["currents"]["text"] = ax_map.text(
                ylim[1] + corner_offset[0] - 50,
                xlim[1] + corner_offset[1] - 85,
                f"Currents: {speed:.2f} m/s",
                fontsize=10,
                color="white",
                verticalalignment="top",
                horizontalalignment="left",
                zorder=10,
                label="",
            )
            dhandles["currents"]["arrow"].remove()
            dhandles["currents"]["arrow"] = ax_map.arrow(
                ylim[1] + corner_offset[0],
                xlim[1] + corner_offset[1],
                arrow_scale * np.sin(direction),
                arrow_scale * np.cos(direction),
                color="white",
                head_width=7.0,
                shape="full",
                linewidth=3.0,
                zorder=10,
            )
        else:
            dhandles["currents"]["text"].remove()
            dhandles["currents"]["text"] = ax_map.text(
                ylim[1] + corner_offset[0] - 50,
                xlim[1] + corner_offset[1] - 85,
                "Currents: 0.0 m/s",
                fontsize=10,
                color="white",
                verticalalignment="top",
                horizontalalignment="left",
                zorder=10,
                label="",
            )
        if w is not None and w.wind is not None:
            speed = w.wind["speed"]
            direction = w.wind["direction"]
            dhandles["wind"]["text"].remove()
            dhandles["wind"]["text"] = ax_map.text(
                ylim[1] + corner_offset[0] - 50,
                xlim[1] + corner_offset[1] - 105,
                f"Wind: {speed:.2f} m/s",
                fontsize=10,
                color="yellow",
                verticalalignment="top",
                horizontalalignment="left",
                zorder=10,
                label="",
            )
            dhandles["wind"]["arrow"].remove()
            dhandles["wind"]["arrow"] = ax_map.arrow(
                ylim[1] + corner_offset[0],
                xlim[1] + corner_offset[1],
                arrow_scale * np.sin(direction),
                arrow_scale * np.cos(direction),
                head_width=7.0,
                shape="full",
                linewidth=3.0,
                color="yellow",
                zorder=10,
            )
        else:
            dhandles["wind"]["text"].remove()
            dhandles["wind"]["text"] = ax_map.text(
                ylim[1] + corner_offset[0] - 50,
                xlim[1] + corner_offset[1] - 105,
                "Wind: 0.0 m/s",
                fontsize=10,
                color="yellow",
                verticalalignment="top",
                horizontalalignment="left",
                zorder=10,
                label="",
            )

    def update_ownship_live_tracking_data(
        self, ownship: ship.Ship, sensor_measurements: list, n_ships: int, enc: ENC
    ) -> None:
        """Updates tracking-related plots for the own-ship

        Args:
            - ownship (ship.Ship): The own-ship object
            - sensor_measurements (list): List of sensor measurements
            - n_ships (int): Number of ships in the simulation
            - enc (ENC): The ENC object
        """
        if not self._config.show_liveplot_target_ships:
            return

        tracks: list = []
        tracks, _ = ownship.get_do_track_information()
        do_labels = [track[0] for track in tracks]
        do_estimates = [track[1] for track in tracks]
        do_covariances = [track[2] for track in tracks]
        do_lengths = [track[3] for track in tracks]
        do_widths = [track[4] for track in tracks]
        ax_map = self.axes[0]
        zorder_patch = 4
        if len(do_estimates) > 0:
            lw = self._config.do_linewidth
            do_c = self._config.do_colors[0]
            for j, do_estimate in enumerate(do_estimates):  # pylint: disable=consider-using-enumerate
                plt_idx = do_labels[j] - 1  # -1 to account for own-ship being idx 0

                if self.ship_plt_handles[0]["track_started"][plt_idx]:
                    start_idx_track_line_data = 0
                else:
                    start_idx_track_line_data = 1
                    self.ship_plt_handles[0]["track_started"][plt_idx] = True

                if not self._config.show_liveplot_ground_truth_target_pose:
                    if self.ship_plt_handles[0]["do_track_poses"][plt_idx] is not None:
                        self.ship_plt_handles[0]["do_track_poses"][plt_idx].remove()
                    chi_j = np.arctan2(do_estimate[3], do_estimate[2])
                    target_ship_polygon = mapf.create_ship_polygon(
                        do_estimate[0],
                        do_estimate[1],
                        chi_j,
                        do_lengths[j],
                        do_widths[j],
                        self._config.ship_scaling[0],
                        self._config.ship_scaling[1],
                    )
                    self.ship_plt_handles[0]["do_track_poses"][plt_idx] = ax_map.fill(
                        *target_ship_polygon.exterior.xy,
                        color=do_c,
                        linewidth=lw,
                        label="",
                        zorder=zorder_patch - 1,
                    )[0]

                if self._config.show_liveplot_target_tracks:
                    self.ship_plt_handles[0]["do_tracks"][plt_idx].set_xdata(
                        [
                            *self.ship_plt_handles[0]["do_tracks"][plt_idx].get_xdata()[start_idx_track_line_data:],
                            do_estimate[1],
                        ]
                    )
                    self.ship_plt_handles[0]["do_tracks"][plt_idx].set_ydata(
                        [
                            *self.ship_plt_handles[0]["do_tracks"][plt_idx].get_ydata()[start_idx_track_line_data:],
                            do_estimate[0],
                        ]
                    )

                    ellipse_x, ellipse_y = mhm.create_probability_ellipse(do_covariances[j], 0.67)
                    ell_geometry = Polygon(zip(ellipse_y + do_estimates[j][1], ellipse_x + do_estimates[j][0]))
                    if self.ship_plt_handles[0]["do_covariances"][plt_idx] is not None:
                        self.ship_plt_handles[0]["do_covariances"][plt_idx].remove()
                    self.ship_plt_handles[0]["do_covariances"][plt_idx] = ax_map.fill(
                        *ell_geometry.exterior.xy,
                        linewidth=lw,
                        color="orange",
                        alpha=0.2,
                        label=f"DO {j - 1} est. 1sigma cov.",
                        zorder=zorder_patch - 2,
                    )[0]

        if self._config.show_liveplot_measurements:
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

                if not xdata or not ydata:
                    continue

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
        if idx > 0 and not self._config.show_liveplot_target_ships:
            return

        lw = kwargs["lw"] if "lw" in kwargs else self._config.ship_linewidth
        c = kwargs["c"] if "c" in kwargs else self._config.ship_colors[idx]
        start_idx_ship_line_data = kwargs["start_idx_ship_line_data"] if "start_idx_ship_line_data" in kwargs else 0
        ax_map = self.axes[0]
        zorder_patch = 3
        csog_state = ship_obj.csog_state
        ship_poly = mapf.create_ship_polygon(
            csog_state[0],
            csog_state[1],
            csog_state[3],
            ship_obj.length,
            ship_obj.width,
            self._config.ship_scaling[0],
            self._config.ship_scaling[1],
        )
        if self.ship_plt_handles[idx]["ground_truth_patch"] is not None:
            self.ship_plt_handles[idx]["ground_truth_patch"].remove()

        if ship_obj.id == 0 or (ship_obj.id > 0 and self._config.show_liveplot_ground_truth_target_pose):
            self.ship_plt_handles[idx]["ground_truth_patch"] = ax_map.fill(
                *ship_poly.exterior.xy, color=c, linewidth=lw, zorder=zorder_patch, label=""
            )[0]

        if not self._config.disable_ship_labels:
            self.ship_plt_handles[idx]["info"].set_x(csog_state[1] - 50)
            self.ship_plt_handles[idx]["info"].set_y(csog_state[0] + 50)

        if (ship_obj.id == 0 and self._config.show_liveplot_ownship_trajectory) or (
            ship_obj.id > 0 and self._config.show_liveplot_target_trajectories
        ):
            self.ship_plt_handles[idx]["trajectory"].set_xdata(
                [*self.ship_plt_handles[idx]["trajectory"].get_xdata()[start_idx_ship_line_data:], csog_state[1]]
            )
            self.ship_plt_handles[idx]["trajectory"].set_ydata(
                [*self.ship_plt_handles[idx]["trajectory"].get_ydata()[start_idx_ship_line_data:], csog_state[0]]
            )

        if self._config.show_liveplot_colav_results:
            self.ship_plt_handles[idx] = ship_obj.plot_colav_results(ax_map, enc, self.ship_plt_handles[idx], **kwargs)

    def update_live_plot(
        self,
        t: float,
        enc: ENC,
        ship_list: List[ship.Ship],
        sensor_measurements: list,
        w: Optional[stoch.DisturbanceData] = None,
        **kwargs,
    ) -> None:
        """Updates the live plot with the current data of the ships in the simulation.

        Args:
            - t (float): Current time in the simulation.
            - enc (ENC): ENC object containing the map data.
            - ship_list (list): List of configured ships in the simulation.
            - sensor_measurements (list): Most recent sensor measurements generated from the own-ship sensors.
        """
        if not self._config.show_liveplot:
            return

        if t > 0.0 and (t - self._t_prev_update < (1.0 / self._config.update_rate_liveplot)):
            return

        self._t_prev_update = t
        self.fig.canvas.restore_region(self.background)
        ax_map = self.axes[0]
        if self._config.show_liveplot_time:
            self.misc_plt_handles["time"].remove()
            ylim = ax_map.get_xlim()  # easting
            xlim = ax_map.get_ylim()  # northing
            self.misc_plt_handles["time"] = ax_map.text(
                ylim[0] + 30,
                xlim[0] + 150,
                f"t = {t:.2f} s",
                fontsize=15,
                color="white",
                verticalalignment="top",
                horizontalalignment="left",
                zorder=10,
                label="",
            )

        # if self._config.show_liveplot_scalebar:
        #     self.scalebar.remove()
        #     self.scalebar = ax_map.add_artist(
        #         ScaleBar(
        #             1,
        #             units="m",
        #             location="lower left",
        #             frameon=False,
        #             color="white",
        #             box_alpha=0.0,
        #             pad=0.5,
        #             font_properties={"size": 12},
        #         )
        #     )

        if self._config.zoom_in_liveplot_on_ownship:
            self.zoom_in_live_plot_on_ownship(enc, ship_list[0].csog_state)

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

            self.update_ship_live_data(
                ship_obj, i, enc, lw=lw, c=c, start_idx_ship_line_data=start_idx_ship_line_data, **kwargs
            )

        self.update_disturbance_live_data(ax_map, w)

        self.fig.canvas.blit(ax_map.bbox)
        self.fig.canvas.flush_events()
        plt.show(block=False)
        self.frames.append(self.get_live_plot_image())

    def zoom_in_live_plot_on_ownship(self, enc: ENC, os_state: np.ndarray) -> None:
        """Narrows the live plot extent to the own-ship position.

        Args:
            - enc (ENC): ENC object containing the map data.
            os_state (np.ndarray): Own-ship CSOG state.
        """
        buffer = self._config.zoom_window_width / 2.0
        xlimits_os = [os_state[0] - buffer, os_state[0] + buffer]
        ylimits_os = [os_state[1] - buffer, os_state[1] + buffer]
        upd_xlimits = xlimits_os  # [max(xlimits_os[0], self.xlimits[0]), min(xlimits_os[1], self.xlimits[1])]
        upd_ylimits = ylimits_os  # [max(ylimits_os[0], self.ylimits[0]), min(ylimits_os[1], self.ylimits[1])]
        self.axes[0].set_xlim(upd_ylimits[0], upd_ylimits[1])
        self.axes[0].set_ylim(upd_xlimits[0], upd_xlimits[1])
        # plt.axis("equal")

    def save_live_plot_animation(self, filename: Path = dp.animation_output / "liveplot.gif") -> None:
        """Saves the live plot animation to a file if enabled.

        Args:
            filename (Path): Path to the file where the animation is saved.
        """
        if not (self._config.save_liveplot_animation and self._config.show_liveplot):
            return

        fig = plt.figure(figsize=(self.frames[0].shape[1] / 72.0, self.frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(self.frames[0], aspect="auto")
        plt.axis("off")

        def init():
            patch.set_data(self.frames[0])
            return (patch,)

        def animate(i):
            patch.set_data(self.frames[i])
            return (patch,)

        anim = animation.FuncAnimation(
            fig=fig, func=animate, init_func=init, blit=True, frames=len(self.frames), interval=50, repeat=True
        )
        anim.save(
            filename=filename.as_posix(),
            writer=animation.PillowWriter(fps=20),
            progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
        )

    def visualize_results(
        self,
        enc: ENC,
        ship_list: List[ship.Ship],
        sim_data: DataFrame,
        sim_times: np.ndarray,
        k_snapshots: Optional[list] = None,
        save_file_path: Optional[Path] = dp.figure_output / "scenario_ne",
    ) -> Tuple[list, list]:
        """Visualize the results of a scenario simulation, save figures (only map figure as of now) to file if enabled.

        Args:
            - enc (ENC): Electronic Navigational Chart object.
            - ship_list (list): List of ships in the simulation.
            - sim_data (DataFrame): Dataframe of simulation.
            - sim_times (list): List of simulation times.
            - k_snapshots (Optional[list], optional): List of snapshots to visualize.
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
            os_colav_stats = {
                "t_solve": t_solve,
                "cost_vals": cost_vals,
                "n_iters": n_iters,
                "final_residuals": final_residuals,
            }

        cpa_indices = ship_data["cpa_indices"]
        min_os_depth = mapf.find_minimum_depth(ship_list[0].draft, enc)

        n_samples = len(sim_times)
        if k_snapshots is None:
            k_snapshots = [round(0.09 * n_samples), round(0.25 * n_samples), round(0.6 * n_samples)]

        figs = []
        axes = []
        fig_map = plt.figure("Scenario: " + str(save_file_path.stem), figsize=self._config.figsize)
        ax_map = fig_map.add_subplot(1, 1, 1)
        plotters.plot_background(ax_map, enc)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        xlimits, ylimits = self.find_plot_limits(enc, ship_list[0], buffer=0.0)
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
                is_inside_map = mhm.check_if_trajectory_is_within_xy_limits(
                    X[:, first_valid_idx:end_idx], xlimits, ylimits
                )
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
                    label="",
                    zorder=zorder_patch - 5,
                )

            ax_map.plot(
                X[1, first_valid_idx:end_idx],
                X[0, first_valid_idx:end_idx],
                color=ship_color,
                linewidth=ship_lw,
                label=ship_name + " traj.",
                zorder=zorder_patch - 2,
            )

            # If the ship is the own-ship: Also plot dynamic obstacle tracks, and trajectory tracking results
            if i == 0:
                if self._config.show_trajectory_tracking_results and len(nominal_trajectory_list) > 0:
                    fig_tt, axes_tt = self.plot_trajectory_tracking_results(
                        i, sim_times, X, nominal_trajectory_list[i], linewidth=1.0
                    )

                track_data = mhm.extract_track_data_from_dataframe(ship_sim_data)
                do_estimates = track_data["do_estimates"]
                do_covariances = track_data["do_covariances"]
                do_NISes = track_data["do_NISes"]
                do_labels = track_data["do_labels"]

                # Plot distance to own-ship
                fig_obst_dist, axes_obst_dist = self.plot_obstacle_distances_to_ownship(
                    sim_times, trajectory_list, do_estimates, do_covariances, do_labels, min_os_depth, enc
                )

                for j, do_estimates_j in enumerate(do_estimates):
                    first_valid_idx_track, last_valid_idx_track = mhm.index_of_first_and_last_non_nan(
                        do_estimates_j[0, :]
                    )

                    end_idx_j = k_snapshots[-1]
                    if last_valid_idx_track < end_idx_j:
                        end_idx_j = last_valid_idx + 1

                    if first_valid_idx_track >= end_idx_j:
                        continue

                    do_color = self._config.do_colors[j]
                    do_lw = self._config.do_linewidth
                    do_true_states_j = trajectory_list[do_labels[j]]
                    do_true_states_j = mhm.convert_state_to_vxvy_state(do_true_states_j)

                    ax_map.plot(
                        do_estimates_j[1, first_valid_idx_track:end_idx_j],
                        do_estimates_j[0, first_valid_idx_track:end_idx_j],
                        color=do_color,
                        linewidth=ship_lw,
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
                    #             label=f"DO {plt_idx} est. cov.",
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
                ax_map.fill(
                    *ship_poly.exterior.xy,
                    linewidth=ship_lw,
                    color=ship_color,
                    # label=ship_name",
                    zorder=zorder_patch,
                )

                # ax_map.text(
                #     X[1, k] - 100,
                #     X[0, k] + 200,
                #     f"$t_{count}$",
                #     fontsize=12,
                #     zorder=zorder_patch + 1,
                # )
                count += 1

        ax_map.set_xlim([ylimits[0], ylimits[1]])
        ax_map.set_ylim([xlimits[0], xlimits[1]])
        ax_map.add_artist(
            ScaleBar(
                1,
                units="m",
                location="lower left",
                frameon=False,
                color="white",
                box_alpha=0.0,
                pad=0.5,
                font_properties={"size": 12},
            )
        )
        plt.legend()

        if self._config.save_result_figures:
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
        dist2closest_grounding_hazard = np.linalg.norm(distance_vectors, axis=0)[: sim_times.shape[0]]
        if n_do == 0:
            axes = [axes]
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
            do_true_states_j = mhm.convert_state_to_vxvy_state(do_true_states_j)

            # z_val = norm.ppf(confidence_level)
            # std_x = np.sqrt(do_covariances_j[0, 0, :])
            # axes[j + 1].fill_between(
            #     sim_times,
            #     do_estimates_j[0, :] - z_val * std_x,
            #     do_estimates_j[0, :] + z_val * std_x,
            #     color="xkcd:blue",
            #     alpha=0.3,
            # )

            est_dist2do_j = np.linalg.norm(do_estimates_j[:2, :] - os_traj[:2, :], axis=0)[: sim_times.shape[0]]
            dist2do_j = np.linalg.norm(do_true_states_j[:2, :] - os_traj[:2, :], axis=0)[: sim_times.shape[0]]
            # axes[j + 1].plot(sim_times, dist2do_j, "b", label=f"Distance to DO{do_labels[j]}")
            # axes[j + 1].plot(sim_times, d_safe_do * np.ones_like(sim_times), "r--", label="Minimum safety margin")
            axes[j + 1].semilogy(sim_times, dist2do_j, "b", label=f"Distance to DO{do_labels[j]}")
            axes[j + 1].semilogy(sim_times, d_safe_do * np.ones_like(sim_times), "r--", label="Minimum safety margin")
            axes[j + 1].set_ylabel("Distance [m]")
            axes[j + 1].set_xlabel("Time [s]")
            axes[j + 1].legend()

        return fig, axes

    def plot_trajectory_tracking_results(
        self,
        ship_idx: int,
        sim_times: np.ndarray,
        trajectory: np.ndarray,
        reference_trajectory: np.ndarray,
        linewidth: float = 1.0,
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

        axes["x"].plot(
            sim_times[:n_samples], trajectory[0, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual"
        )
        axes["x"].plot(
            sim_times[:n_samples],
            reference_trajectory[0, :n_samples],
            color="xkcd:red",
            linestyle="--",
            linewidth=linewidth,
            label="nominal",
        )
        # axes["x"].set_xlabel("Time [s]")
        axes["x"].set_ylabel("North [m]")
        axes["x"].legend()
        current_values = axes["x"].get_yticks().tolist()
        axes["x"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["x"].set_yticklabels(["{:.1f}".format(y) for y in current_values])

        axes["y"].plot(
            sim_times[:n_samples], trajectory[1, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual"
        )
        axes["y"].plot(
            sim_times[:n_samples],
            reference_trajectory[1, :n_samples],
            color="xkcd:red",
            linestyle="--",
            linewidth=linewidth,
            label="nominal",
        )
        # axes["y"].set_xlabel("Time [s]")
        axes["y"].set_ylabel("East [m]")
        axes["y"].legend()
        current_values = axes["y"].get_yticks().tolist()
        axes["y"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["y"].set_yticklabels(["{:.1f}".format(y) for y in current_values])

        axes["psi"].plot(
            sim_times[:n_samples],
            trajectory[2, :n_samples] * 180.0 / np.pi,
            color="xkcd:blue",
            linewidth=linewidth,
            label="actual",
        )
        axes["psi"].plot(
            sim_times[:n_samples],
            reference_trajectory[2, :n_samples] * 180.0 / np.pi,
            color="xkcd:red",
            linestyle="--",
            linewidth=linewidth,
            label="nominal",
        )
        # axes["psi"].set_xlabel("Time [s]")
        axes["psi"].set_ylabel("Heading [deg]")
        axes["psi"].legend()

        axes["u"].plot(
            sim_times[:n_samples], trajectory[3, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual"
        )
        axes["u"].plot(
            sim_times[:n_samples],
            reference_trajectory[3, :n_samples],
            color="xkcd:red",
            linestyle="--",
            linewidth=linewidth,
            label="nominal",
        )
        # axes["u"].set_xlabel("Time [s]")
        axes["u"].set_ylabel("Surge [m/s]")
        axes["u"].legend()

        axes["v"].plot(
            sim_times[:n_samples], trajectory[4, :n_samples], color="xkcd:blue", linewidth=linewidth, label="actual"
        )
        axes["v"].plot(
            sim_times[:n_samples],
            reference_trajectory[4, :n_samples],
            color="xkcd:red",
            linestyle="--",
            linewidth=linewidth,
            label="nominal",
        )
        # axes["v"].set_xlabel("Time [s]")
        axes["v"].set_ylabel("Sway [m/s]")
        axes["v"].legend()

        axes["r"].plot(
            sim_times[:n_samples],
            trajectory[5, :n_samples] * 180.0 / np.pi,
            color="xkcd:blue",
            linewidth=linewidth,
            label="actual",
        )
        axes["r"].plot(
            sim_times[:n_samples],
            reference_trajectory[5, :n_samples] * 180.0 / np.pi,
            linestyle="--",
            color="xkcd:red",
            linewidth=linewidth,
            label="nominal",
        )
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
        axes["x"].plot(
            sim_times, do_estimates[0, :].round(decimals=1), color="xkcd:blue", linewidth=do_lw, label="estimate"
        )
        axes["x"].plot(
            sim_times, do_true_states[0, :].round(decimals=1), color="xkcd:red", linewidth=do_lw, label="true"
        )
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
        current_values = axes["x"].get_yticks().tolist()
        axes["x"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["x"].set_yticklabels(["{:.1f}".format(x) for x in current_values])

        axes["y"].plot(
            sim_times, do_estimates[1, :].round(decimals=1), color="xkcd:blue", linewidth=do_lw, label="estimate"
        )
        axes["y"].plot(
            sim_times, do_true_states[1, :].round(decimals=1), color="xkcd:red", linewidth=do_lw, label="true"
        )
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
        current_values = axes["y"].get_yticks().tolist()
        axes["y"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["y"].set_yticklabels(["{:.1f}".format(y) for y in current_values])

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
        current_values = axes["Vx"].get_yticks().tolist()
        axes["Vx"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["Vx"].set_yticklabels(["{:.2f}".format(x) for x in current_values])

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
        current_values = axes["Vy"].get_yticks().tolist()
        axes["Vy"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["Vy"].set_yticklabels(["{:.2f}".format(x) for x in current_values])

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
        current_values = axes["NIS"].get_yticks().tolist()
        axes["NIS"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["NIS"].set_yticklabels(["{:.2f}".format(x) for x in current_values])

        error = do_true_states - do_estimates
        pos_error = np.sqrt(error[0, :] ** 2 + error[1, :] ** 2)
        vel_error = np.sqrt(error[2, :] ** 2 + error[3, :] ** 2)
        axes["errs"].plot(sim_times, pos_error, color="xkcd:blue", linewidth=do_lw, label="pos. error")
        axes["errs"].plot(sim_times, vel_error, color="xkcd:red", linewidth=do_lw, label="vel. error")
        axes["errs"].set_xlabel("Time [s]")
        axes["errs"].legend()
        current_values = axes["errs"].get_yticks().tolist()
        axes["errs"].yaxis.set_major_locator(mticker.FixedLocator(current_values))
        axes["errs"].set_yticklabels(["{:.4f}".format(x) for x in current_values])

        plt.show(block=False)
        return fig, axes

    @property
    def zoom_window_width(self) -> float:
        return self._config.zoom_window_width
