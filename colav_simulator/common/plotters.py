"""
    plotters.py

    Summary:
        Contains methods for plotting images and data.

    Author: Trym Tengesdal
"""

import copy
from typing import Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import matplotlib.pyplot as plt
import numpy as np
import seacharts.display.colors as colors
from seacharts.enc import ENC
from shapely.geometry import MultiPolygon, Polygon


def plot_trajectory(
    trajectory: np.ndarray,
    enc: ENC,
    color: str,
    edge_style: Optional[str] = None,
    buffer: Optional[float] = 0.5,
    linewidth: Optional[float] = 1.0,
    alpha: Optional[float] = 1.0,
) -> None:
    """Plots the trajectory on the ENC.

    Args:
        trajectory (np.ndarray): Input trajectory, minimum 2 x n_samples.
        enc (ENC): Electronic Navigational Chart object
        color (str): Color of the trajectory
        marker_type (Optional[str], optional): Marker type for the trajectory. Defaults to None.@
        marker_size (Optional[float], optional): Marker size for the trajectory. Defaults to None.
        edge_style (Optional[str], optional): Edge style for the trajectory. Defaults to None.
        buffer (Optional[float], optional): Buffer of the trajectory. Defaults to 0.5.
        linewidth (Optional[float], optional): linewidth of the trajectory. Defaults to 0.5.
    """
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    enc.draw_line(
        trajectory_line,
        color=color,
        buffer=buffer,
        linewidth=linewidth,
        edge_style=edge_style,
        alpha=alpha,
    )


def plot_disturbance(
    magnitude: float,
    direction: float,
    name: str,
    enc: ENC,
    color: str,
    linewidth: Optional[float] = 2.5,
    location: Optional[str] = "topright",
    text_location_offset: Optional[Tuple[float, float]] = (0.0, 0.0),
) -> plt.axes:
    """Plots a disturbance vector on the ENC as a vector arrow inside a circle.
    The name of the disturbance is plotted below the circle, with an offset given by text_location_offset.

    Args:
        magnitude (float): Magnitude of the disturbance / length of the disturbance vector
        direction (float): Direction of the disturbance (defined in a north-east coordinate system)
        name (str): Name of the disturbance
        enc (ENC): Electronic Navigational Chart object
        color (str): Color of the disturbance vector
        linewidth (Optional[float]): Arrow thickness. Defaults to 1.0.
        location (Optional[str]): Location of the disturbance vector in ["topleft", "topright", "bottomleft", "bottomright"]. Defaults to "topright".
        text_location_offset (Optional[Tuple[float, float]]): Offset of the text location. Defaults to (0.0, 0.0).
    """
    enc.start_display()
    xmin, ymin, xmax, ymax = enc.bbox  # x is east, y is north
    if location == "topright":
        origin = (xmax - 0.1 * (xmax - xmin), ymax - 0.1 * (ymax - ymin))
    elif location == "topleft":
        origin = (xmin + 0.1 * (xmax - xmin), ymax - 0.1 * (ymax - ymin))
    elif location == "bottomright":
        origin = (xmax - 0.1 * (xmax - xmin), ymin + 0.1 * (ymax - ymin))
    elif location == "bottomleft":
        origin = (xmin + 0.1 * (xmax - xmin), ymin + 0.1 * (ymax - ymin))

    arrow_start = origin
    arrow_end = (origin[0] + magnitude * np.sin(direction), origin[1] + magnitude * np.cos(direction))
    text_location = (
        origin[0] + text_location_offset[0] - 0.8 * magnitude,
        origin[1] - 1.2 * magnitude + text_location_offset[1],
    )

    circle_handle = enc.draw_circle(origin, radius=magnitude, color="white", fill=True, alpha=0.2)
    arrow_handle = enc.draw_arrow(arrow_start, arrow_end, color=color, width=linewidth, fill=True)
    text_handle = enc.draw_text(name, text_location, color=color, size=10)
    return [circle_handle, arrow_handle, text_handle]


def plot_shapely_multipolygon(
    ax: plt.Axes, mp: MultiPolygon, color: str, fill: bool = True, alpha: float = 1.0, zorder: int = 1
) -> plt.Axes:
    """Plots a shapely MultiPolygon object on a matplotlib axes.

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        mp (MultiPolygon): MultiPolygon object to plot.
        color (str, optional): Color of the MultiPolygon.
        fill (bool, optional): Option for filling the MultiPolygon. Defaults to False.
        alpha (float, optional): Transparency of the MultiPolygon. Defaults to 1.0.
        zorder (int, optional): Z-order of the MultiPolygon. Defaults to 1.
    """
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])

    for poly in mp.geoms:
        if fill:
            ax.fill(*poly.exterior.xy, color=color, alpha=alpha, zorder=zorder)
        else:
            ax.plot(*poly.exterior.xy, color=color, zorder=zorder)
    return ax


def plot_background(
    ax: plt.Axes,
    enc: ENC,
    show_shore: bool = True,
    show_seabed: bool = True,
    dark_mode: bool = True,
    uniform_seabed_color: bool = False,
    land_color: Optional[str] = None,
    shore_color: Optional[str] = None,
) -> None:
    """Creates a static background based on the input seacharts

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        enc (ENC): Electronic Navigational Chart object
        show_shore (bool, optional): Option for showing the shore. Defaults to True.
        show_seabed (bool, optional): Option for showing the seabed. Defaults to True.
        dark_mode (bool, optional): Option for dark mode. Defaults to True.
        uniform_seabed_color (bool, optional): Option for using a uniform color for the seabed. Defaults to False.
        land_color (Optional[str], optional): Color of the land to override the default. Defaults to None.
        shore_color (Optional[str], optional): Color of the shore to override the default. Defaults to None.

    Returns:
        Tuple[]: Tuple of limits in x and y for the background extent
    """
    # For every layer put in list and assign a color
    if enc.land:
        color = "#142c38" if dark_mode else colors.color_picker(enc.land.color)
        if land_color is not None:
            color = land_color
        plot_shapely_multipolygon(ax, enc.land.geometry, color=color, zorder=enc.land.z_order)

    if show_shore and enc.shore:
        color = "#142c38" if dark_mode else colors.color_picker(enc.shore.color)
        if shore_color is not None:
            color = shore_color
        plot_shapely_multipolygon(ax, enc.shore.geometry, color=color, zorder=enc.shore.z_order)

    if show_seabed and enc.seabed:
        bins = len(enc.seabed.keys())
        count = 0
        for _, layer in enc.seabed.items():
            if uniform_seabed_color:
                rank = enc.seabed[0].z_order
                color = colors.color_picker(0, bins)
            else:
                rank = layer.z_order + count
                color = colors.color_picker(count, bins)
            plot_shapely_multipolygon(ax, layer.geometry, color=color, zorder=rank)
            count += 1

    x_min, y_min, x_max, y_max = enc.bbox
    ax.set_xlim((x_min, x_max))  # Easting
    ax.set_ylim((y_min, y_max))  # Northing


def plot_waypoints(
    waypoints: np.ndarray,
    enc: ENC,
    color: str,
    point_buffer: Optional[float] = 10,
    disk_buffer: Optional[float] = 80,
    hole_buffer: Optional[float] = 10,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = 0.6,
    show_annuluses: Optional[bool] = True,
    draft: Optional[float] = 5.0,
):
    path = mapf.create_path_polygon(waypoints, point_buffer, disk_buffer, hole_buffer, show_annuluses)

    # hazards = extract_relevant_grounding_hazards_as_union(find_minimum_depth(draft, enc), enc)[0]
    # if path.intersects(hazards):
    #     overlap = path.intersection(hazards)
    #     enc.draw_polygon(overlap, "red", thickness=linewidth, alpha=alpha)
    #     path = path.difference(hazards)
    enc.draw_polygon(path, color, thickness=linewidth, alpha=alpha)


def plot_dynamic_obstacles(
    dynamic_obstacles: list, color: str, enc: ENC, T: float, dt: float, map_origin: Optional[np.ndarray] = None
) -> None:
    """Plots the dynamic obstacles as ellipses and ship polygons.

    Args:
        dynamic_obstacles (list): List of tuples containing (ID, state, cov, length, width)
        color (string): Color of the ellipses
        enc (ENC): Electronic Navigational Chart object
        T (float): Horizon to predict straight line trajectories for the dynamic obstacles
        dt (float): Time step for the straight line trajectories
        map_origin (np.ndarray, optional): Origin of the map in the form [x, y]^T
    """
    N = int(T / dt)
    enc.start_display()
    dynamic_obstacles_copy = copy.deepcopy(dynamic_obstacles)
    for ID, state, cov, length, width in dynamic_obstacles_copy:
        if map_origin is not None:
            state[:2] += map_origin
        ellipse_x, ellipse_y = mhm.create_probability_ellipse(cov, 0.67)
        ell_geometry = Polygon(zip(ellipse_y + state[1], ellipse_x + state[0]))
        # enc.draw_polygon(ell_geometry, color=color, alpha=0.4)

        for k in range(0, N, 5):
            do_poly = mapf.create_ship_polygon(
                state[0] + k * dt * state[2],
                state[1] + k * dt * state[3],
                np.arctan2(state[3], state[2]),
                length,
                width,
                length_scaling=1.0,
                width_scaling=1.0,
            )
            enc.draw_polygon(do_poly, color=color)
        do_poly = mapf.create_ship_polygon(
            state[0], state[1], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0
        )
        enc.draw_polygon(do_poly, color=color)


def plot_rrt_tree(node_list: list, enc: ENC) -> None:
    """Plots an RRT tree given by the list of nodes containing (state, parent_id, id, trajectory, inputs, cost)

    Args:
        node_list (list): List of nodes containing (state, parent_id, id, trajectory, inputs, cost)
        enc (ENC): Electronic Navigational Chart object
    """
    enc.start_display()
    for node in node_list:
        # enc.draw_circle(
        #     (node["state"][1], node["state"][0]), 2.5, color="green", fill=False, thickness=0.8, edge_style=None
        # )
        for sub_node in node_list:
            if node["id"] == sub_node["id"] or sub_node["parent_id"] != node["id"]:
                continue
            points = [(tt[1], tt[0]) for tt in sub_node["trajectory"]]
            if len(points) > 1:
                enc.draw_line(points, color="white", buffer=0.5, linewidth=0.5)
