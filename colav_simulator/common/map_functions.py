"""
    map_functions.py

    Summary:
        Contains functionality for plotting the Electronic Navigational Chart (ENC),
        computing distance to land polygons, generating random ship starting positions etc.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
import random
from typing import Tuple

import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import seacharts.display.colors as colors
from cartopy.feature import ShapelyFeature
from osgeo import osr
from seacharts.enc import ENC
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon


def local2latlon(
    x: float | list | np.ndarray, y: float | list | np.ndarray, utm_zone: int
) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from x (east), y (north) to latitude, longitude.

    Args:
        x (float | list): East coordinate(s) in a local UTM coordinate system.
        y (float | list): North coordinate(s) in a local UTM coordinate system.
        utm_zone (int): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of latitude and longitude coordinates.
    """
    to_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        from_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        from_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(x, y)))
        lat = [coord[0] for coord in coordinates]
        lon = [coord[1] for coord in coordinates]
    else:
        lat, lon, _ = transform.TransformPoint(x, y)

    return lat, lon


def latlon2local(
    lat: float | list | np.ndarray, lon: float | list | np.ndarray, utm_zone: int
) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
    """Transform coordinates from latitude, longitude to UTM32 or UTM33

    Args:
        lat (float | list): Latitude coordinate(s)
        lon (float | list): Longitude coordinate(s)
        utm_zone (str): UTM zone.

    Raises:
        ValueError: If the input string is not correct.

    Returns:
        Tuple[float | list | np.ndarray, float | list | np.ndarray]: Tuple of east and north coordinates.
    """
    from_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        to_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        to_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "utm_zone" is not correct. Supported zones sofar are 32 and 33.')

    src = osr.SpatialReference()
    src.ImportFromEPSG(from_zone)
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(to_zone)
    transform = osr.CoordinateTransformation(src, tgt)

    if isinstance(lat, (list, np.ndarray)) and isinstance(lon, (list, np.ndarray)):
        coordinates = transform.TransformPoints(list(zip(lat, lon)))
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
    else:
        x, y, _ = transform.TransformPoint(lat, lon)

    return x, y


def dist_between_latlon_coords(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the distance between two latitude, longitude coordinates.

    Args:
        lat1 (float): Latitude of first coordinate.
        lon1 (float): Longitude of first coordinate.
        lat2 (float): Latitude of second coordinate.
        lon2 (float): Longitude of second coordinate.

    Returns:
        float: Distance between the two coordinates in meters
    """
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).m


def create_ship_polygon(x: float, y: float, heading: float, length: float, width: float, scale: float = 1.0) -> Polygon:
    """Creates a ship polygon from the ship`s position, heading, length and width.

    Args:
        x (float): The ship`s north position
        y (float): The ship`s east position
        heading (float): The ship`s heading
        length (float): Length of the ship
        width (float): Width of the ship
        scale (float, optional): Scale factor. Defaults to 1.0.

    Returns:
        np.ndarray: Ship polygon
    """
    eff_length = length * scale
    eff_width = width * scale

    x_min, x_max = x - eff_length / 2.0, x + eff_length / 2.0 - eff_width
    y_min, y_max = y - eff_width / 2.0, y + eff_width / 2.0
    left_aft, right_aft = (y_min, x_min), (y_max, x_min)
    left_bow, right_bow = (y_min, x_max), (y_max, x_max)
    coords = [left_aft, left_bow, (y, x + eff_length / 2.0), right_bow, right_aft]
    poly = Polygon(coords)
    return affinity.rotate(poly, -heading, origin=(y, x), use_radians=True)


def plot_background(ax: plt.Axes, enc: ENC, show_shore: bool = True, show_seabed: bool = True) -> None:
    """Creates a static background based on the input seacharts

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        enc (ENC): Electronic Navigational Chart object
        show = Option for visualization

    Returns:
        Tuple[]: Tuple of limits in x and y for the background extent
    """
    # For every layer put in list and assign a color
    if enc.land:
        color = colors.color_picker(enc.land.color)
        ax.add_feature(ShapelyFeature([enc.land.geometry], color=color, zorder=enc.land.z_order, crs=enc.crs))

    if show_shore and enc.shore:
        color = colors.color_picker(enc.shore.color)
        ax.add_feature(ShapelyFeature([enc.shore.geometry], color=color, zorder=enc.shore.z_order, crs=enc.crs))

    if show_seabed and enc.seabed:
        bins = len(enc.seabed.keys())
        count = 0
        for _, layer in enc.seabed.items():
            rank = layer.z_order + count
            color = colors.color_picker(count, bins)
            ax.add_feature(ShapelyFeature([layer.geometry], color=color, zorder=rank, crs=enc.crs))
            count += 1

    x_min, y_min, x_max, y_max = enc.bbox
    ax.set_extent((x_min, x_max, y_min, y_max), crs=enc.crs)


def generate_random_seabed_depth_from_draft(enc: ENC, draft: float) -> float:
    """Randomly chooses a valid sea depth (seabed) depending on a ship's draft.

    Args:
        draft (float): Ship's draft in meters.

    Returns:
        float: Seabed depth in meters.
    """
    feasible_depth_vals = list(enc.seabed.keys())
    for d in enc.seabed.keys():
        if len(enc.seabed[d].mapping["coordinates"]) in [0, 1] or float(d) < draft:
            feasible_depth_vals.remove(d)

    index = random.randint(0, len(feasible_depth_vals) - 1)
    return feasible_depth_vals[index]


def generate_random_start_position_from_draft(
    enc: ENC, draft: float, land_clearance: float = 0.0
) -> Tuple[float, float]:
    """
    Randomly defining starting easting and northing coordinates of a ship
    inside the safe sea region by considering a ship draft, with an optional land clearance distance.

    Args:
        enc (ENC): Electronic Navigational Chart object
        draft (float): Ship's draft in meters.
        land_clearance (float): Minimum distance to land in meters.

    Returns:
        Tuple[float, float]: Tuple of starting x and y coordinates for the ship.
    """
    depth = generate_random_seabed_depth_from_draft(enc, draft)
    safe_sea = enc.seabed[depth]
    ss_bounds = safe_sea.geometry.bounds

    is_safe = False
    while not is_safe:
        easting = random.uniform(ss_bounds[0], ss_bounds[2])
        northing = random.uniform(ss_bounds[1], ss_bounds[3])

        is_ok_clearance = min_distance_to_land(enc, easting, northing) >= land_clearance

        is_safe = safe_sea.geometry.contains(Point(easting, northing)) and is_ok_clearance

    return northing, easting


def min_distance_to_land(enc: ENC, y: float, x: float) -> float:
    """Compute the minimum distance to land from a given point.

    Args:
        enc (ENC): Electronic Navigational Chart object
        y (float): Ship's easting coordinate
        x (float): Ship's northing coordinate

    Returns:
        float: Minimum distance to land in meters.
    """
    position = Point(y, x)
    distance = enc.land.geometry.distance(position)
    return distance


def check_if_segment_crosses_grounding_hazards(
    enc: ENC, next_wp: np.ndarray, prev_wp: np.ndarray, draft: float = 5.0
) -> bool:
    """Checks if a line segment between two waypoints crosses nearby grounding hazards (land, shore).

    Args:
        enc (ENC): Electronic Navigational Chart object
        next_wp (np.ndarray): Next waypoint position.
        prev_wp (np.ndarray): Previous waypoint position.
        draft (float): Ship's draft in meters.

    Returns:
        bool: True if path segment crosses land, False otherwise.

    """
    # Create linestring with east as the x value and north as the y value
    p1 = (next_wp[1], next_wp[0])
    p2 = (prev_wp[1], prev_wp[0])
    wp_line = LineString([p1, p2])

    entire_seabed = enc.seabed[0].geometry
    depths = list(enc.seabed.keys())
    for depth in depths:
        if float(depth) > draft:
            seabed_below_draft = enc.seabed[depth].geometry
            break

    seabed_down_to_draft = entire_seabed.difference(seabed_below_draft)

    intersects_relevant_seabed = wp_line.intersects(seabed_down_to_draft)

    intersects_land_or_shore = wp_line.intersects(enc.shore.geometry)

    crosses_grounding_hazards = intersects_land_or_shore or intersects_relevant_seabed

    return crosses_grounding_hazards
