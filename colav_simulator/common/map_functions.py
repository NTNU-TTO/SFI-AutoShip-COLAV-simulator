"""
    map_functions.py

    Summary:
        Contains functionality for plotting the Electronic Navigational Chart (ENC),
        computing distance to land polygons, generating random ship starting positions etc.

    Author: Trym Tengesdal, Magne Aune, Joachim Miller
"""
from typing import Optional, Tuple

import colav_simulator.common.miscellaneous_helper_methods as mhm
import geopandas as gpd
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as scipy_spatial
import seacharts.display.colors as colors
import shapely.ops as ops
from cartopy.feature import ShapelyFeature
from osgeo import osr
from seacharts.enc import ENC
from shapely import affinity, strtree
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon


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


def create_point_list_from_polygons(polygons: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east) coordinates of the polygons.
    """
    px, py, ls = [], [], []
    for i, poly in enumerate(polygons):
        y, x = poly.exterior.coords.xy
        a = np.array(x.tolist())
        b = np.array(y.tolist())
        la, lx = len(a), len(px)
        c = [(i + lx, (i + 1) % la + lx) for i in range(la - 1)]
        px += a.tolist()
        py += b.tolist()
        ls += c

    points = np.array([px, py]).T
    P1, P2 = points[ls][:, 0], points[ls][:, 1]
    return P1, P2


def extract_vertices_from_polygon_list(polygons: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east) coordinates of the polygons.
    """
    px, py = [], []
    for i, poly in enumerate(polygons):
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly:
                y, x = sub_poly.exterior.coords.xy
                px.extend(x[:-1].tolist())
                py.extend(y[:-1].tolist())
        elif isinstance(poly, Polygon):
            y, x = poly.exterior.coords.xy
            px.extend(x[:-1].tolist())
            py.extend(y[:-1].tolist())
        else:
            continue
    return np.array(px), np.array(py)


def extract_safe_sea_area(
    min_depth: int, enveloping_polygon: Polygon, enc: Optional[ENC] = None, as_polygon_list: bool = False, show_plots: bool = False
) -> MultiPolygon | list:
    """Extracts the safe sea area from the ENC as a list of polygons.

    This includes sea polygons that are above the vessel`s minimum depth.

    Args:
        - min_depth (int): The minimum depth required for the vessel to avoid grounding.
        - enveloping_polygon (geometry.Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - as_polygon_list (bool, optional): Option for returning the safe sea area as a list of polygons. Defaults to False.
        - show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        MultiPolygon | list: The safe sea area.
    """
    safe_sea = enc.seabed[min_depth].geometry.intersection(enveloping_polygon)
    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(safe_sea, color="green", alpha=0.25, fill=False)

    if as_polygon_list:
        if isinstance(safe_sea, MultiPolygon):
            return [poly for poly in safe_sea.geoms]
        elif isinstance(safe_sea, Polygon):
            return [safe_sea]
        else:
            return []
    return safe_sea


def create_free_boundary_points_from_enc(enc: ENC, hazards: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates an array of points on the ENC boundary which is free from grounding hazards.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        hazards (list): List of relevant grounding hazards.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and y coordinates of the free boundary points.
    """
    (xmin, ymin, xmax, ymax) = enc.bbox
    n_pts_per_side = 50
    x = np.linspace(xmin, xmax, n_pts_per_side)
    y = np.linspace(ymin, ymax, n_pts_per_side)
    points = []
    for i in range(n_pts_per_side):
        p = Point(x[i], ymin)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(x[i], ymax)
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(xmin, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)

    for i in range(n_pts_per_side):
        p = Point(xmax, y[i])
        if any(p.touches(hazard) for hazard in hazards):
            continue
        points.append(p)
    # [enc.draw_circle((p.x, p.y), radius=0.5, color="yellow") for p in points]
    Y = np.array([p.x for p in points])
    X = np.array([p.y for p in points])
    return X, Y


def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> Polygon:
    """Converts a bounding box to a polygon.

    Args:
        bbox (Tuple[float, float, float, float]): The bounding box.

    Returns:
        Polygon: The polygon.
    """
    (xmin, ymin, xmax, ymax) = bbox
    return Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])


def point_in_polygon_list(point: Point, polygons: list) -> bool:
    """Checks if a point is in a list of polygons.

    Args:
        point (Point): The point to check.
        polygons (list): List of polygons.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    for poly in polygons:
        if point.within(poly) or point.touches(poly):
            return True
    return False


def point_in_point_list(point: Point, points: list) -> bool:
    """Checks if a point is in a list of points.

    Args:
        point (Point): The point to check.
        points (list): List of points.

    Returns:
        bool: True if the point is in a hazard, False otherwise.
    """
    for p in points:
        if point.within(p) or point.touches(p):
            return True
    return False


def create_safe_sea_voronoi_diagram(enc: ENC, vessel_min_depth: int = 5) -> Tuple[scipy_spatial.Voronoi, list]:
    """Creates a Voronoi diagram of the safe sea region (i.e. its vertices).

    Args:
        enc (ENC): The Electronic Navigational Chart object.
        vessel_min_depth (float): The safe minimum depth for the vessel to voyage in.

    Returns:
        scipy_spatial.Voronoi: The Voronoi diagram of the safe sea region.
    """
    bbox = enc.bbox
    enc_bbox_poly = bbox_to_polygon(bbox)
    safe_sea = extract_safe_sea_area(vessel_min_depth, enc_bbox_poly, enc, as_polygon_list=True, show_plots=True)
    polygons = []
    for sea_poly in safe_sea:
        if isinstance(sea_poly, MultiPolygon):
            for poly in sea_poly:
                polygons.append(poly)
        elif isinstance(sea_poly, Polygon):
            polygons.append(sea_poly)
        else:
            continue
    px, py = extract_vertices_from_polygon_list(polygons)
    points = np.vstack((py, px)).T
    vor = scipy_spatial.Voronoi(points)
    region_polygons = create_region_polygons_from_voronoi(vor, enc=enc)
    for point in points:
        enc.draw_circle((point[0], point[1]), radius=0.4, color="red")

    # # Keep all voronoi region boundary points that are in the safe sea area
    # safe_points = []
    # for region in vor.regions:
    #     region_vertices = vor.vertices[region]
    #     for vertex in region_vertices:
    #         point = Point(vertex)
    #         if point_in_polygon_list(point, polygons):
    #             safe_points.append((point.x, point.y))
    # settt = set(safe_points)
    # safe_points = list(settt)
    # for point in safe_points:
    #     enc.draw_circle((point[0], point[1]), radius=0.4, color="magenta")
    return vor, region_polygons


def create_safe_sea_triangulation(enc: ENC, vessel_min_depth: int = 5, show_plots: bool = True) -> list:
    """Creates a constrained delaunay triangulation of the safe sea region.

    Args:
        enc (ENC): Electronic Navigational Chart object.
        vessel_min_depth (int, optional): The safe minimum depth for the vessel to voyage in. Defaults to 5.

    Returns:
        list: List of triangles.
    """
    safe_sea_poly_list = extract_safe_sea_area(vessel_min_depth, bbox_to_polygon(enc.bbox), enc, as_polygon_list=True, show_plots=True)
    cdt_list = []
    largest_poly_area = 0.0
    for poly in safe_sea_poly_list:
        cdt = constrained_delaunay_triangulation_custom(poly)
        if poly.area > largest_poly_area:
            largest_poly_area = poly.area
            cdt_largest = cdt
        if show_plots:
            enc.draw_polygon(poly, color="orange", alpha=0.5)
            enc.start_display()
            for triangle in cdt:
                enc.draw_polygon(triangle, color="black", fill=False)
        cdt_list.append(cdt)

    return cdt_largest


def create_region_polygons_from_voronoi(vor: scipy_spatial.Voronoi, enc: Optional[ENC] = None) -> list:
    """Creates a list of polygons from the Voronoi diagram.

    Args:
        vor (scipy_spatial.Voronoi): The Voronoi diagram.
        enc (Optional[ENC], optional): The Electronic Navigational Chart object. Defaults to None.

    Returns:
        list: List of polygons.
    """
    polygons = []
    for region in vor.regions:
        if not region:
            continue
        region_vertices = vor.vertices[region]
        if region_vertices.shape[0] < 3:
            continue
        region_poly = Polygon(region_vertices)
        if region_poly.area < 1.0:
            continue
        polygons.append(region_poly)
        if enc:
            enc.start_display()
            enc.draw_polygon(region_poly, color="yellow", alpha=0.5)
    return polygons


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


def plot_background(ax: plt.Axes, enc: ENC, show_shore: bool = True, show_seabed: bool = True, dark_mode: bool = True) -> None:
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
        color = "#142c38" if dark_mode else colors.color_picker(enc.land.color)
        ax.add_feature(ShapelyFeature([enc.land.geometry], color=color, zorder=enc.land.z_order, crs=enc.crs))

    if show_shore and enc.shore:
        color = "#142c38" if dark_mode else colors.color_picker(enc.shore.color)
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


def extract_relevant_grounding_hazards_as_union(vessel_min_depth: int, enc: ENC, buffer: Optional[float] = None, show_plots: bool = False) -> list:
    """Extracts the relevant grounding hazards from the ENC as a multipolygon.

    This includes land, shore and seabed polygons that are below the vessel`s minimum depth.

    Args:
        vessel_min_depth (int): The minimum depth required for the vessel to avoid grounding.
        enc (senc.ENC): The ENC to check for grounding.
        buffer (Optional[float], optional): Buffer for polygons. Defaults to None.
        show_plots (bool, optional): Option for visualization. Defaults to False.

    Returns:
        geometry.MultiPolygon: The relevant grounding hazards.
    """
    dangerous_seabed = enc.seabed[0].geometry.difference(enc.seabed[vessel_min_depth].geometry)
    # return [enc.land.geometry, enc.shore.geometry, dangerous_seabed]
    relevant_hazards = [enc.land.geometry.union(enc.shore.geometry).union(dangerous_seabed)]
    filtered_relevant_hazards = []
    for hazard in relevant_hazards:
        poly = MultiPolygon(Polygon(p.exterior) for p in hazard.geoms if isinstance(p, Polygon))
        if buffer is not None:
            poly = poly.buffer(buffer)
        filtered_relevant_hazards.append(poly)

    if show_plots:
        enc.start_display()
        for hazard in filtered_relevant_hazards:
            enc.draw_polygon(hazard, color="red", alpha=0.5)
    return filtered_relevant_hazards


def fill_rtree_with_geometries(geometries: list) -> Tuple[strtree.STRtree, list]:
    """Fills an rtree with the given multipolygon geometries. Used for fast spatial queries.

    Args:
        - geometries (list): The geometries to fill the rtree with.

    Returns:
        Tuple[strtree.STRtree, list]: The rtree containing the geometries, and the Polygon objects used to build it.
    """
    poly_list = []
    for poly in geometries:
        assert isinstance(poly, MultiPolygon), "Only MultiPolygon members are supported"
        for sub_poly in poly.geoms:
            poly_list.append(sub_poly)
    return strtree.STRtree(poly_list), poly_list


def generate_random_start_position_from_draft(
    rng: np.random.Generator, enc: ENC, draft: float, min_land_clearance: float = 100.0, safe_sea_cdt: Optional[list] = None
) -> Tuple[float, float]:
    """
    Randomly defining starting easting and northing coordinates of a ship
    inside the safe sea region by considering a ship draft, with an optional land clearance distance.

    Args:
        - rng (np.random.Generator): Numpy random generator.
        - enc (ENC): Electronic Navigational Chart object
        - draft (float): Ship's draft in meters.
        - min_land_clearance (float): Minimum distance to land in meters.
        - safe_sea_cdt (Optional[list]): List of triangles defining the safe sea region, used to sample more efficiently. Defaults to None.

    Returns:
        - Tuple[float, float]: Tuple of starting x and y coordinates for the ship.
    """
    depth = find_minimum_depth(draft, enc)
    safe_sea = enc.seabed[depth]
    bbox = enc.bbox

    is_safe = False
    iter_count = 0
    while not is_safe:
        if safe_sea_cdt is not None:
            random_triangle = rng.choice(safe_sea_cdt)
            assert isinstance(random_triangle, Polygon) and len(random_triangle.exterior.coords) >= 4, "The safe sea region must be a polygon and triangle."
            x, y = random_triangle.exterior.coords.xy
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[1], y[1]])
            p3 = np.array([x[2], y[2]])
            random_point = mhm.sample_from_triangle_region(p1, p2, p3, rng)
            easting, northing = random_point[0], random_point[1]
        else:
            easting, northing = rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3])

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
        - show_plots (bool, optional): Option for visualization. Defaults to False.

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
        if hazard.is_empty:
            continue

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


def generate_ship_sector_polygons(pos_x: float, pos_y: float, chi: float, safety_radius: float) -> Tuple[Polygon, Polygon, Polygon]:
    """Generates sector polygons for the ship portside, front and starboardside.

    Args:
        pos_x (float): X position of ship
        pos_y (float): Y position of ship
        chi (float): Course over ground of ship
        safety_radius (int): radius for collision polygons on port, front and starboard of ship

    Returns:
        Tuple[Polygon, Polygon, Polygon]: Tuple of polygons for port, front and starboard collision zones
    """

    ship_center = Point([pos_x, pos_y])

    # zone parameters
    angle = np.pi / 4  # zone intersection angle
    offset = np.pi / 2  # zone offset angle
    num_points = 100

    # Close coast zone port
    angle_range_port = np.linspace(-chi + 2 * offset - angle, -chi + 2 * offset + angle, num_points)
    arc_port = [(ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_port]
    arc_line_port = LineString(arc_port)
    zone_port = Polygon(list(arc_line_port.coords) + [ship_center])

    # Close coast zone front
    angle_range_front = np.linspace(-chi + offset - angle, -chi + offset + angle, num_points)
    arc_front = [(ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_front]
    arc_line_front = LineString(arc_front)
    zone_front = Polygon(list(arc_line_front.coords) + [ship_center])

    # Close coast zone starboard
    angle_range_starboard = np.linspace(-chi - angle, -chi + angle, num_points)
    arc_starboard = [(ship_center.x + safety_radius * np.cos(angle), ship_center.y + safety_radius * np.sin(angle)) for angle in angle_range_starboard]
    arc_line_starboard = LineString(arc_starboard)
    zone_starboard = Polygon(list(arc_line_starboard.coords) + [ship_center])

    return zone_port, zone_front, zone_starboard


def distances_to_coast(poly_port: Polygon, poly_front: Polygon, poly_starboard: Polygon, poly_ship: Polygon, poly_land: MultiPolygon) -> Tuple[float, float, float]:
    """Calculates distance to coast based on intersection between collision polygons and land polygon.

    Args:
        poly_port (Polygon): Polygon of collision zone on portside of ship
        poly_front (Polygon): Polygon of collision zone in front of ship
        poly_starboard (Polygon): Polygon of collision zone on starboardside of ship
        poly_ship (Polygon): Polygon of ship
        poly_land (MultiPolygon): Polygon of land

    Returns:
        Tuple[float, float, float]: Tuple of distances to land on portside, front and starboardside
    """
    port_intersec_land = poly_port.intersection(poly_land)
    front_intersec_land = poly_front.intersection(poly_land)
    starboard_intersec_land = poly_starboard.intersection(poly_land)

    dist_port = poly_ship.distance(port_intersec_land)
    dist_front = poly_ship.distance(front_intersec_land)
    dist_starboard = poly_ship.distance(starboard_intersec_land)

    return dist_port, dist_front, dist_starboard


def generate_enveloping_polygon(trajectory: np.ndarray, buffer: float) -> Polygon:
    """Creates an enveloping polygon around the trajectory of the vessel, buffered by the given amount.

    Args:
        - trajectory (np.ndarray): Trajectory with columns [x, y, psi, u, v, r]
        - buffer (float): Buffer size

    Returns:
        Polygon: The query polygon
    """
    point_list = []
    for k in range(trajectory.shape[1]):
        point_list.append((trajectory[1, k], trajectory[0, k]))
    trajectory_linestring = LineString(point_list).buffer(buffer)
    return trajectory_linestring


def extract_polygons_near_trajectory(
    trajectory: np.ndarray, geometry_tree: strtree.STRtree, buffer: float, enc: Optional[ENC] = None, show_plots: bool = False
) -> Tuple[list, Polygon]:
    """Extracts the polygons that are relevant for the trajectory of the vessel, inside a corridor of the given buffer size.

    Args:
        - trajectory (np.ndarray): Trajectory with columns [x, y, psi, u, v, r]
        - geometry_tree (strtree.STRtree): The rtree containing the relevant grounding hazard polygons.
        - buffer (float): Buffer size
        - enc (Optional[ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        Tuple[list, Polygon]: List of tuples of relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to. Also returns the query polygon.
    """
    enveloping_polygon = generate_enveloping_polygon(trajectory, buffer)
    polygons_near_trajectory = geometry_tree.query(enveloping_polygon)
    poly_list = []
    for poly in polygons_near_trajectory:
        relevant_poly_list = []
        intersection_poly = enveloping_polygon.intersection(poly)
        if intersection_poly.area == 0.0 and intersection_poly.length == 0.0:
            continue

        if isinstance(intersection_poly, MultiPolygon):
            for sub_poly in intersection_poly.geoms:
                relevant_poly_list.append(sub_poly)
        else:
            relevant_poly_list.append(intersection_poly)
        poly_list.append((relevant_poly_list, poly))

    if enc is not None and show_plots:
        enc.start_display()
        enc.draw_polygon(enveloping_polygon, color="yellow", alpha=0.2)
        # for poly_sublist, _ in poly_list:
        #     for poly in poly_sublist:
        #         enc.draw_polygon(poly, color="red", fill=False)

    return poly_list, enveloping_polygon


def extract_vertices_from_polygon_list(polygons: list) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a list of x and y coordinates from a list of polygons.

    Args:
        polygons (list): List of shapely polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the x (north) and y (east) coordinates of the polygons.
    """
    px, py = [], []
    for i, poly in enumerate(polygons):
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly:
                y, x = sub_poly.exterior.coords.xy
                px.extend(x[:-1].tolist())
                py.extend(y[:-1].tolist())
        elif isinstance(poly, Polygon):
            y, x = poly.exterior.coords.xy
            px.extend(x[:-1].tolist())
            py.extend(y[:-1].tolist())
        else:
            continue
    return np.array(px), np.array(py)


def extract_boundary_polygons_inside_envelope(poly_tuple_list: list, enveloping_polygon: Polygon, enc: Optional[ENC] = None, show_plots: bool = True) -> list:
    """Extracts the boundary trianguled polygons that are relevant for the trajectory of the vessel, inside the given envelope polygon.

    Args:
        - poly_tuple_list (list): List of tuples with relevant polygons inside query/envelope polygon and the corresponding original polygon they belong to.
        - enveloping_polygon (Polygon): The query polygon.
        - enc (Optional[senc.ENC]): Electronic Navigational Chart object used for plotting. Defaults to None.
        - show_plots (bool, optional): Whether to show plots or not. Defaults to False.

    Returns:
        list: List of boundary polygons.
    """
    boundary_polygons = []
    for relevant_poly_list, original_polygon in poly_tuple_list:
        for relevant_polygon in relevant_poly_list:
            triangle_boundaries = extract_triangle_boundaries_from_polygon(relevant_polygon, enveloping_polygon, original_polygon)
            if not triangle_boundaries:
                continue

            if enc is not None and show_plots:
                # enc.draw_polygon(poly, color="pink", alpha=0.3)
                for tri in triangle_boundaries:
                    enc.draw_polygon(tri, color="red", fill=False)

            boundary_polygons.extend(triangle_boundaries)
    return boundary_polygons


def extract_triangle_boundaries_from_polygon(polygon: Polygon, planning_area_envelope: Polygon, original_polygon: Polygon) -> list:
    """Extracts the triangles that comprise the boundary of the polygon.

    Triangles are filtered out if they have two vertices on the envelope boundary and is inside of the original polygon.

    Args:
        - polygon (Polygon): The polygon in consideration inside the envelope polygon.
        - planning_area_envelope (Polygon): A polygon representing the relevant area the vessel is planning to navigate in.
        - original_polygon (Polygon): The original polygon that the relevant polygon belongs to.

    Returns:
        list: List of shapely polygons representing the boundary triangles for the polygon.
    """
    cdt = constrained_delaunay_triangulation_custom(polygon)
    # return cdt
    original_polygon_boundary = LineString(original_polygon.exterior.coords).buffer(0.0001)
    boundary_triangles = []
    if len(cdt) == 1:
        return cdt

    # Check if triangle has two vertices on the envelope boundary and is inside of the original polygon
    for tri in cdt:
        v_count = 0
        idx_prev = 0
        for idx, v in enumerate(tri.exterior.coords):
            if v_count == 2 and idx_prev == idx - 1 and tri not in boundary_triangles:
                boundary_triangles.append(tri)
                break
            v_point = Point(v)
            if original_polygon_boundary.contains(v_point):
                v_count += 1
                idx_prev = idx

    return boundary_triangles


# def constrained_delaunay_triangulation(polygon: Polygon) -> list:
#     """Uses the triangle library to compute a constrained delaunay triangulation.

#     Args:
#         polygon (Polygon): The polygon to triangulate.

#     Returns:
#         list: List of triangles as shapely polygons.
#     """
#     x, y = polygon.exterior.coords.xy
#     vertices = np.array([list(a) for a in zip(x, y)])
#     cdt = tr.triangulate({"vertices": vertices})
#     triangle_indices = cdt["triangles"]
#     triangles = [Polygon([cdt["vertices"][i] for i in tri]) for tri in triangle_indices]

#     cdt_triangles = []
#     for tri in triangles:
#         intersection_poly = tri.intersection(polygon)

#         if isinstance(intersection_poly, Point) or isinstance(intersection_poly, LineString):
#             continue

#         if intersection_poly.area == 0.0:
#             continue

#         # cdt_triangles.append(tri)
#         if isinstance(intersection_poly, MultiPolygon) or isinstance(intersection_poly, GeometryCollection):
#             for sub_poly in intersection_poly.geoms:
#                 if sub_poly.area == 0.0 or isinstance(sub_poly, Point) or isinstance(sub_poly, LineString):
#                     continue
#                 cdt_triangles.append(sub_poly)
#         else:
#             cdt_triangles.append(intersection_poly)
#     return cdt_triangles


def constrained_delaunay_triangulation_custom(polygon: Polygon) -> list:
    """Converts a polygon to a list of triangles. Basically constrained delaunay triangulation.

    Args:
        - polygon (Polygon): The polygon to triangulate.

    Returns:
        list: List of triangles as shapely polygons.
    """
    res_intersection_gdf = gpd.GeoDataFrame(geometry=[polygon])
    # Create ID to identify overlapping polygons
    res_intersection_gdf["TRI_ID"] = res_intersection_gdf.index
    # List to keep triangulated geometries
    triangles = []
    # List to keep the original IDs
    triangle_ids = []
    # Triangulate single or multi-polygons
    for i, _ in res_intersection_gdf.iterrows():
        tri_ = ops.triangulate(res_intersection_gdf.geometry.values[i])
        triangles.append(tri_)
        for _ in range(0, len(tri_)):
            triangle_ids.append(res_intersection_gdf.TRI_ID.values[i])
    # Check if it is a single or multi-polygon
    len_list = len(triangles)
    triangles = np.array(triangles).flatten().tolist()
    # unlist geometries for multi-polygons
    if len_list > 1:
        triangles = [item for sublist in triangles for item in sublist]
    # Create triangulated polygons
    filtered_triangles = gpd.GeoDataFrame(triangles)
    filtered_triangles = filtered_triangles.set_geometry(triangles)
    del filtered_triangles[0]
    # Assign original IDs to each triangle
    filtered_triangles["TRI_ID"] = triangle_ids
    # Create new ID for each triangle
    filtered_triangles["LINK_ID"] = filtered_triangles.index
    # Create centroids from all triangles
    filtered_triangles["centroid"] = filtered_triangles.centroid
    filtered_triangles_centroid = filtered_triangles.set_geometry("centroid")
    del filtered_triangles_centroid["geometry"]
    del filtered_triangles["centroid"]
    # Find triangle centroids inside original polygon
    filtered_triangles_join = gpd.sjoin(
        filtered_triangles_centroid[["centroid", "TRI_ID", "LINK_ID"]], res_intersection_gdf[["geometry", "TRI_ID"]], how="inner", predicate="within"
    )
    # Remove overlapping from other triangles (Necessary for multi-polygons overlapping or close to each other)
    filtered_triangles_join = filtered_triangles_join[filtered_triangles_join["TRI_ID_left"] == filtered_triangles_join["TRI_ID_right"]]
    # Remove overload triangles from same filtered_triangless
    filtered_triangles = filtered_triangles[filtered_triangles["LINK_ID"].isin(filtered_triangles_join["LINK_ID"])]
    filtered_triangles = filtered_triangles.geometry.values
    # double check
    cdt_triangles = []
    area_eps = 1e-4
    for tri in triangles:
        intersection_poly = tri.intersection(polygon)
        if isinstance(intersection_poly, Point) or isinstance(intersection_poly, LineString):
            continue
        if intersection_poly.area < area_eps:
            continue

        if isinstance(intersection_poly, MultiPolygon) or isinstance(intersection_poly, GeometryCollection):
            for sub_poly in intersection_poly.geoms:
                if sub_poly.area < area_eps or isinstance(sub_poly, Point) or isinstance(sub_poly, LineString):
                    continue
                cdt_triangles.append(sub_poly)
        else:
            cdt_triangles.append(intersection_poly)
    return cdt_triangles


def plot_trajectory(trajectory: np.ndarray, enc: ENC, color: str, marker_type: Optional[str] = None, edge_style: Optional[str] = None) -> None:
    """Plots the trajectory on the ENC.

    Args:
        trajectory (np.ndarray): Input trajectory, minimum 2 x n_samples.
        enc (ENC): Electronic Navigational Chart object
        color (str): Color of the trajectory
    """
    enc.start_display()
    trajectory_line = []
    for k in range(trajectory.shape[1]):
        trajectory_line.append((trajectory[1, k], trajectory[0, k]))
    enc.draw_line(trajectory_line, color=color, width=0.5, thickness=0.5, marker_type=marker_type, edge_style=edge_style)


def plot_dynamic_obstacles(dynamic_obstacles: list, enc: ENC, T: float, dt: float) -> None:
    """Plots the dynamic obstacles as ellipses and ship polygons.

    Args:
        dynamic_obstacles (list): List of tuples containing (ID, state, cov, length, width)
        enc (ENC): Electronic Navigational Chart object
        T (float): Horizon to predict straight line trajectories for the dynamic obstacles
        dt (float): Time step for the straight line trajectories
    """
    N = int(T / dt)
    enc.start_display()
    for (ID, state, cov, length, width) in dynamic_obstacles:
        ellipse_x, ellipse_y = mhm.create_probability_ellipse(cov, 0.99)
        ell_geometry = Polygon(zip(ellipse_y + state[1], ellipse_x + state[0]))
        enc.draw_polygon(ell_geometry, color="orange", alpha=0.3)

        for k in range(0, N, 10):
            do_poly = create_ship_polygon(
                state[0] + k * dt * state[2], state[1] + k * dt * state[3], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0
            )
            enc.draw_polygon(do_poly, color="red")
        do_poly = create_ship_polygon(state[0], state[1], np.arctan2(state[3], state[2]), length, width, length_scaling=1.0, width_scaling=1.0)
        enc.draw_polygon(do_poly, color="red")


def plot_rrt_tree(node_list: list, enc: ENC) -> None:
    """Plots an RRT tree given by the list of nodes containing (state, parent_id, id, trajectory, inputs, cost)

    Args:
        node_list (list): List of nodes containing (state, parent_id, id, trajectory, inputs, cost)
        enc (ENC): Electronic Navigational Chart object
    """
    enc.start_display()
    for node in node_list:
        enc.draw_circle((node["state"][1], node["state"][0]), 2.5, color="green", fill=False, thickness=0.8, edge_style=None)
        for sub_node in node_list:
            if node["id"] == sub_node["id"] or sub_node["parent_id"] != node["id"]:
                continue
            points = [(tt[1], tt[0]) for tt in sub_node["trajectory"]]
            if len(points) > 1:
                enc.draw_line(points, color="white", width=0.5, thickness=0.5, marker_type=None)
