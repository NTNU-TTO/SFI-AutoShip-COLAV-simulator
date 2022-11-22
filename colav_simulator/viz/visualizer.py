"""
    vizualizer.py

    Summary:
        Contains functionality for visualizing/animating ship scenarios.

    Author: Trym Tengesdal, Magne Aune, Melih Akdag, Joachim Miller
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

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
    artists: list

    def __init__(self, enc: ENC, config_file: Path = dp.visualizer_config) -> None:
        self._config = cp.extract(Config, config_file, dp.visualizer_schema)

        self.fig, self.axes = self._init_figure(enc)
        self.artists = []

    def _init_figure(self, enc: ENC) -> Tuple[plt.Figure, list]:
        plt.close()

        fig, ax_map = plt.subplots(figsize=self._config.figsize)
        ax_map.margins(x=self._config.margins[0], y=self._config.margins[0])
        ax_map.set_autoscale_on(True)

        mapf.plot_background(ax_map, enc)

        axes = [ax_map]
        # TODO: Add more axes for plotting vessel speed, heading, vessel-vessel distances etc..

        return fig, axes

    def _clear_artists(self) -> None:
        for artist in self.artists:
            artist.remove()
            del artist

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
            plt.show()

        if self._config.save_animation:
            anim.save(save_path, writer="ffmpeg", progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"))
