"""
    map_functions.py

    Summary:
        Contains functionality for plotting the Electronic Navigational Chart (ENC),
        computing distance to land polygons, generating random ship starting positions etc.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
import random
from typing import Tuple

from math import cos, sin, pi
import colav_simulator.common.miscellaneous_helper_methods as mhm
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import seacharts.display.colors as colors
import shapely.ops as ops
from cartopy.feature import ShapelyFeature
from osgeo import osr
from seacharts.enc import ENC
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


def local2latlon(x: float | list | np.ndarray, y: float | list | np.ndarray, utm_zone: int) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
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


def latlon2local(lat: float | list | np.ndarray, lon: float | list | np.ndarray, utm_zone: int) -> Tuple[float | list | np.ndarray, float | list | np.ndarray]:
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


def create_ship_polygon(x: float, y: float, heading: float, length: float, width: float, length_scaling: float = 1.0, width_scaling: float = 1.0) -> Polygon:
    """Creates a ship polygon from the ship`s position, heading, length and width.

    Args:
        x (float): The ship`s north position
        y (float): The ship`s east position
        heading (float): The ship`s heading
        length (float): Length of the ship
        width (float): Width of the ship
        length_scaling (float, optional): Length scale factor. Defaults to 1.0.
        width_scaling (float, optional): Length scale factor. Defaults to 1.0.

    Returns:
        np.ndarray: Ship polygon
    """
    eff_length = length * length_scaling
    eff_width = width * width_scaling

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


def find_minimum_depth(vessel_draft: float, enc: ENC):
    """Find the minimum seabed depth for the given vessel draft (for it to avoid grounding)

    Args:
        vessel_draft (float): The vessel`s draft.

    Returns:
        float: The minimum seabed depth required for a safe journey for the vessel.
    """
    lowest_possible_depth = 0
    for depth in enc.seabed:
        if vessel_draft <= float(depth):
            lowest_possible_depth = depth
            break
    return lowest_possible_depth


def extract_relevant_grounding_hazards(vessel_min_depth: int, enc: ENC) -> list:
    """Extracts the relevant grounding hazards from the ENC as a list of polygons.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]


def extract_relevant_grounding_hazards_as_union(vessel_min_depth: int, enc: ENC, show_plots: bool = False) -> list:
    """Extracts the union of the relevant grounding hazards from the ENC as a list of polygons.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        - vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (senc.ENC): The ENC to check for grounding.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        list: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    # return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        filtered_relevant_hazards.append(MultiPolygon(Polygon(p.exterior) for p in hazard.geoms if isinstance(p, Polygon)))

    if show_plots:
        enc.start_display()
        for hazard in filtered_relevant_hazards:
            enc.draw_polygon(hazard, color="red", alpha=0.5)
    return filtered_relevant_hazards


def generate_random_start_position_from_draft(enc: ENC, draft: float, min_land_clearance: float = 100.0) -> Tuple[float, float]:
    """
    Randomly defining starting easting and northing coordinates of a ship
    inside the safe sea region by considering a ship draft, with an optional land clearance distance.

    Args:
        - enc (ENC): Electronic Navigational Chart object
        - draft (float): Ship's draft in meters.
        - min_land_clearance (float): Minimum distance to land in meters.

    Returns:
        - Tuple[float, float]: Tuple of starting x and y coordinates for the ship.
    """
    depth = find_minimum_depth(draft, enc)
    safe_sea = enc.seabed[depth]
    bbox = enc.bbox

    is_safe = False
    iter_count = 0
    while not is_safe:
        easting = random.uniform(bbox[0], bbox[2])
        northing = random.uniform(bbox[1], bbox[3])
        is_ok_clearance = min_distance_to_land(enc, easting, northing) >= min_land_clearance
        if safe_sea.geometry.contains(Point(easting, northing)) and is_ok_clearance:
            break

        iter_count += 1
        if iter_count > 1000:
            raise Exception("Could not find a valid start position. Check the map data of your ENC object.")

    return northing, easting


def compute_distance_vectors_to_grounding(vessel_trajectory: np.ndarray, minimum_vessel_depth: int, enc: ENC, show_plots: bool = False) -> np.ndarray:
    """Computes the distance vectors to grounding at each step of the given vessel trajectory.

    Args:
        - vessel_trajectory (np.ndarray): The vessel`s trajectory, 2 x n_samples.
        - minimum_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (ENC): The ENC to check for grounding.

    Returns:
        - np.ndarray: The distance to grounding at each step of the vessel trajectory.
    """
    if show_plots:
        enc.start_display()
    relevant_hazards = extract_relevant_grounding_hazards_as_union(minimum_vessel_depth, enc)
    vessel_traj_linestring = mhm.ndarray_to_linestring(vessel_trajectory)

    distance_vectors = np.ndarray((2, vessel_trajectory.shape[1]))
    for idx, point in enumerate(vessel_traj_linestring.coords):
        for hazard in relevant_hazards:
            dist = hazard.distance(Point(point))

            point = Point(vessel_traj_linestring.coords[idx])
            nearest_poly_points = []
            for hazard in relevant_hazards:
                nearest_point = ops.nearest_points(point, hazard)[1]
                nearest_poly_points.append(nearest_point)

            min_dist = 1e12
            min_dist_vec = np.array([1e6, 1e6])
            for _, near_point in enumerate(nearest_poly_points):
                points = [
                    (np.asarray(point.coords.xy[0])[0], np.asarray(point.coords.xy[1])[0]),
                    (np.asarray(near_point.coords.xy[0])[0], np.asarray(near_point.coords.xy[1])[0]),
                ]

                if show_plots:
                    enc.draw_line(points, color="black", width=0.5, marker_type="o")

                dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
                if np.linalg.norm(dist_vec) <= min_dist:
                    min_dist_vec = dist_vec
                    min_dist = np.linalg.norm(min_dist_vec)
            distance_vectors[:, idx] = min_dist_vec
    return distance_vectors


def compute_closest_grounding_dist(vessel_trajectory: np.ndarray, minimum_vessel_depth: int, enc: ENC, show_enc: bool = False) -> Tuple[float, np.ndarray, int]:
    """Computes the closest distance to grounding for the given vessel trajectory.

    Args:
        - vessel_trajectory (np.ndarray): The vessel`s trajectory, 2 x n_samples.
        - minimum_vessel_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enc (senc.ENC): The ENC to check for grounding.

    Returns:
        - Tuple[float, int]: The closest distance to grounding, corresponding distance vector and the index of the trajectory point.
    """
    relevant_hazards = extract_relevant_grounding_hazards(minimum_vessel_depth, enc)
    vessel_traj_linestring = mhm.ndarray_to_linestring(vessel_trajectory)
    if enc and show_enc:
        enc.start_display()
        for hazard in relevant_hazards:
            enc.draw_polygon(hazard, color="red")
    # intersection_points = find_intersections_line_polygon(vessel_traj_linestring, relevant_hazards, enc)

    # Will find the closest grounding point.
    min_dist = 1e12
    for idx, point in enumerate(vessel_traj_linestring.coords):
        for hazard in relevant_hazards:
            dist = hazard.distance(Point(point))
            if dist < min_dist:
                min_dist = dist
                min_idx = idx

    closest_point = Point(vessel_traj_linestring.coords[min_idx])
    nearest_poly_points = []
    for hazard in relevant_hazards:
        nearest_point = ops.nearest_points(closest_point, hazard)[1]
        nearest_poly_points.append(nearest_point)

    epsilon = 0.01
    for i, point in enumerate(nearest_poly_points):
        points = [
            (np.asarray(closest_point.coords.xy[0])[0], np.asarray(closest_point.coords.xy[1])[0]),
            (np.asarray(point.coords.xy[0])[0], np.asarray(point.coords.xy[1])[0]),
        ]

        if enc and show_enc:
            enc.draw_line(points, color="cyan", marker_type="o")

        min_dist_vec = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
        if np.linalg.norm(min_dist_vec) <= min_dist + epsilon and np.linalg.norm(min_dist_vec) >= min_dist - epsilon:
            break

    if enc and show_enc:
        enc.close_display()
    return min_dist, min_dist_vec, min_idx


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


def min_distance_to_hazards(hazards: list, x: float, y: float) -> float:
    """Compute the minimum distance to hazards from a given point.

    Args:
        hazards (list): List of Multipolygon/Polygon objects that are relevant
        x (float): Ship's easting coordinate
        y (float): Ship's northing coordinate

    Returns:
        float: Minimum distance to hazards in meters.
    """
    min_dist = 1e12
    for hazard in hazards:
        dist = hazard.distance(Point(x, y))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def check_if_segment_crosses_grounding_hazards(enc: ENC, p2: np.ndarray, p1: np.ndarray, draft: float = 5.0) -> bool:
    """Checks if a line segment between two positions/points crosses nearby grounding hazards (land, shore).

    Args:
        enc (ENC): Electronic Navigational Chart object
        p2 (np.ndarray): Second position.
        p1 (np.ndarray): First position.
        draft (float): Ship's draft in meters.

    Returns:
        bool: True if path segment crosses land, False otherwise.

    """
    # Create linestring with east as the x value and north as the y value
    p2_reverse = (p2[1], p2[0])
    p1_reverse = (p1[1], p1[0])
    wp_line = LineString([p1_reverse, p2_reverse])

    entire_seabed = enc.seabed[0].geometry
    min_depth = find_minimum_depth(draft, enc)

    seabed_down_to_draft = entire_seabed.difference(enc.seabed[min_depth].geometry)

    intersects_relevant_seabed = wp_line.intersects(seabed_down_to_draft)

    intersects_land_or_shore = wp_line.intersects(enc.shore.geometry)

    crosses_grounding_hazards = intersects_land_or_shore or intersects_relevant_seabed

    return crosses_grounding_hazards


def generate_grounding_hazard_polys(pos_x, pos_y, chi, safety_radius):
    """Generates hazard polygons of nearby land in portside, front and starboardside.

    Args:
        pos_x (float): X position of ship
        pos_y (float): Y position of ship
        chi (float): Course over ground of ship
        safety_radius (int): radius for collision polygons on port, front and starboard of ship

    Returns:
        zone_port: Polygon of collision zone on portside of ship
        zone_front: Polygon of collision zone in front of ship
        zone_starboard: Polygon of collision zone on starboardside of ship

    """
    
    ship_center = Point([pos_x, pos_y])
    
    # zone parameters
    angle = pi/4 # zone intersection angle
    offset = pi/2 # zone offset angle
    num_points = 100
    
    # Close coast zone port
    angle_range_port = np.linspace(-chi + 2*offset - angle,
                                   -chi + 2*offset + angle, num_points)
    arc_port = [(ship_center.x + safety_radius * np.cos(angle),
                 ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_port]
    arc_line_port = LineString(arc_port)
    zone_port = Polygon(list(arc_line_port.coords) + [ship_center])
    
    # Close coast zone front
    angle_range_front = np.linspace(-chi + offset - angle,
                                    -chi + offset + angle, num_points)
    arc_front = [(ship_center.x + safety_radius * np.cos(angle),
                  ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_front]
    arc_line_front = LineString(arc_front)
    zone_front = Polygon(list(arc_line_front.coords) + [ship_center])
    
    # Close coast zone starboard
    angle_range_starboard = np.linspace(-chi - angle,
                                        -chi + angle, num_points)
    arc_starboard = [(ship_center.x + safety_radius * np.cos(angle),
                      ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_starboard]
    arc_line_starboard = LineString(arc_starboard)
    zone_starboard = Polygon(list(arc_line_starboard.coords) + [ship_center])
    
    return zone_port, zone_front, zone_starboard

def distances_to_coast(poly_port, poly_front, poly_starboard, poly_ship, poly_land):
    """Calculates distance to coast based on intersection between collision polygons and land polygon

    Args:
        poly_port: Polygon of collision zone on portside of ship
        poly_front: Polygon of collision zone in front of ship
        poly_starboard: Polygon of collision zone on starboardside of ship
        poly_ship: Polygon of ship
        poly_land: Polygon of land

    Returns:
        dist_port: Distance to land on portside
        dist_front: Distance to land in front
        dist_starboard: Distance to land on starboardside

    """
    
    port_intersec_land = poly_port.intersection(poly_land)
    front_intersec_land = poly_front.intersection(poly_land)
    starboard_intersec_land = poly_starboard.intersection(poly_land)
    
    dist_port = poly_ship.distance(port_intersec_land)
    dist_front = poly_ship.distance(front_intersec_land)
    dist_starboard = poly_ship.distance(starboard_intersec_land)
    
    return dist_port, dist_front, dist_starboard