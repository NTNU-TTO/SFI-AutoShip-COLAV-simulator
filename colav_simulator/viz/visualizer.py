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
import colav_simulator.common.paths as dp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from seacharts.enc import ENC


@dataclass
class Config:
    """Configuration class for specifying the look of the visualization."""

    show_waypoints: bool = False
    show_animation: bool = True
    save_animation: bool = False
    frame_delay: float = 200.0
    linewidth: float = 1.0
    figsize: list = field(default_factory=lambda: [12, 10])
    margins: list = field(default_factory=lambda: [0.01, 0.01])
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


class Visualizer:
    """Class with functionality for visualizing/animating ship scenarios, and plotting/saving the simulation results.

    Internal variables:
        fig (Figure): Matplotlib figure object for the visualization.
        axes (list): List of matplotlib axes objects for the visualization.

    """

    _config: Config
    fig: plt.figure
    axes: list
    background: Any
    ship_plt_handles: list

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

        for i, ship in enumerate(ship_list):

            c = self._config.ship_colors[i]
            lw = self._config.linewidth

            ship_i_handles = {}

            if i == 0:
                ship_name = "OS"
            else:
                ship_name = "DO " + str(i + 1)

            ship_i_handles["patch"] = ax_map.fill([], [], color=c, linewidth=lw, label=ship_name)[0]

            ship_i_handles["trajectory"] = ax_map.plot([], [], color=c, linewidth=lw, label=ship_name + " traj.")[0]

            ship_i_handles["predicted_trajectory"] = ax_map.plot(
                [], [], color=c, linewidth=lw, marker="*", linestyle="-", label=ship_name + " pred. traj."
            )[0]

            ship_i_handles["waypoints"] = ax_map.plot(
                ship.waypoints[1, :],
                ship.waypoints[0, :],
                color=c,
                marker="o",
                markersize=13,
                linestyle="--",
                linewidth=lw,
                alpha=0.3,
                label=ship_name + " waypoints",
            )[0]

            self.ship_plt_handles.append(ship_i_handles)

    def update_live_plot(self, ship_list: list) -> None:
        """Updates the live plot with the current data of the ships in the simulation.

        Args:
            ship_list (list): List of configured ships in the simulation.
        """

        for i, ship in enumerate(ship_list):
            pose_i = ship.pose

            # Update ship patch
            ship_poly = mapf.create_ship_polygon(pose_i[0], pose_i[1], pose_i[3], ship.length, ship.width, 2.0)
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

            self.ship_plt_handles[i]["waypoints"].set_xdata(ship.waypoints[1, :])
            self.ship_plt_handles[i]["waypoints"].set_ydata(ship.waypoints[0, :])
            self.axes[0].draw_artist(self.ship_plt_handles[i]["waypoints"])

            # TODO: Update predicted ship trajectory

        self.fig.canvas.blit(self.axes[0].bbox)

        plt.pause(self._config.frame_delay / 1000)

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
            ship_list
            data (DataFrame): Pandas DataFrame containing the ship simulation data.
            times (List/np.ndarray): List/array of times to consider.
            show_waypoints (Optional[bool]): _description_. Defaults to True.
            show_animation (Optional[bool]): _description_. Defaults to True.
            save_animation (Optional[bool]): Boolean flag for saving the animation. Defaults to True.
            save_path (Optional[pathlib.Path]): Path to file where the animation is saved. Defaults to None.
        """
        self._clear_artists()

        vessels = []
        trajectories = []

        # data: n_ships x 1 dataframe, where each ship entry contains n_samples x 1 dictionaries of pose, waypoints, speed plans etc..
        n_ships = len(data.columns)
        for i in range(n_ships):
            c = self._config.ship_colors[i]
            lw = self._config.linewidth
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
                U_i = data[f"Ship{j}"][i]["pose"][2]
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
