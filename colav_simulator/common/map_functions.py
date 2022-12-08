"""
    map_functions.py

    Summary:
        Contains functionality for plotting the Electronic Navigational Chart (ENC),
        computing distance to land polygons, generating random ship starting positions etc.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
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
        list: List of transformed coordinates.
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
        list: List of transformed coordinates.
    """
    from_zone = 4326  # Latitude Longitude
    if utm_zone == 32:
        to_zone = 6172  # ETRS89 / UTM zone 32 + NN54 height Møre og Romsdal
    elif utm_zone == 33:
        to_zone = 6173  # ETRS89 / UTM zone 33 + NN54 height
    else:
        raise ValueError('Input "coord_tf_str" is not correct. Correct strings are e.g.: LL_UTM32, UTM32_LL')

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
    """Calculate the distance between two lat, lon points in metres

    Args:
        lon1 (float): Latitude of point 1
        lat1 (float): Longitude of point 1
        lon2 (float): Latitude of point 2
        lat2 (float): Longitude of point 2

    Returns:
        float: Distance in metres
    """
    if lon1 < 0 or lat1 < 0 or lon2 < 0 or lat2 < 0:
        return 999999
    r = 6362.132
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d = r * c * 1000
    return round(d, 1)


def get_blue_colors(bins: int):
    """Returns a blue color palette for the input number of bins.

    Args:
        bins (int): Number of colors to get.

    Returns:
        cmap: Color map/palette of length bins.
    """
    # blue color palette
    return plt.get_cmap("Blues")(np.linspace(0.6, 0.9, bins))


def get_green_colors(bins: int):
    """Returns a green palette for the input number of bins.

    Args:
        bins (int): Number of colors to get.

    Returns:
        cmap: Color map/palette of length bins.
    """
    return plt.get_cmap("Greens")(np.linspace(0.6, 0.5, bins))


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


def plot_background(
    ax: plt.Axes, enc: ENC, show_shore: bool = True, show_seabed: bool = True
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Creates a static background based on the input seacharts

    Args:
        ax (plt.Axes): Matplotlib axes handle.
        enc (ENC): Electronic Navigational Chart object
        show = Option for visualization

    Returns:
        Tuple[]: Tuple of limits in x and y for the background extent
    """
    layers = []
    colors = []
    # For every layer put in list and assign a color
    if enc.land:
        layers.append(enc.land)
        colors.append("#142c38")  # get_green_colors(1))

    if show_shore and enc.shore:
        layers.append(enc.shore)
        if enc.land:
            del colors[0]
            colors.append(get_green_colors(2))
        else:
            colors.append(get_green_colors(1))

    if show_seabed and enc.seabed:
        sea = enc.seabed
        sea_layer = list(sea.keys())
        for key in sea_layer:
            if len(sea[key].mapping["coordinates"]) == 0:
                sea_layer.remove(key)
            layers.append(enc.seabed[key])
        colors.append(get_blue_colors(len(sea_layer)))

    # Flatten color list
    colors_new = []
    for sublist in colors:
        for color in sublist:
            colors_new.append(color)

    # Plot the contour for every layer and assign color
    for c, layer in enumerate(layers):
        # ax.add_feature(layer, facecolor=colors_new[c], zorder=layer.z_order)
        for i in range(len(layer.mapping["coordinates"])):
            pol = list(layer.mapping["coordinates"][i][0])
            if len(pol) >= 3:
                poly = Polygon(pol)
                x, y = poly.exterior.xy
                ax.fill(x, y, c=colors_new[c], zorder=layer.z_order)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    return x_lim, y_lim


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
    min_distance = 1e10
    land = enc.land.mapping["coordinates"]
    for i, _ in enumerate(land):
        poly = list(land[i][0])
        polygon = Polygon(poly)
        distance = position.distance(polygon)
        if distance < min_distance:
            min_distance = distance
    return min_distance


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
