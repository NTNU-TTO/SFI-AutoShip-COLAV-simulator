from shapely.geometry import Point
import random
import numpy as np
import matplotlib.pyplot as plt
import seacharts
from geopandas import GeoSeries


enc = seacharts.ENC()
enc.close_display()


def blues(bins):
    # blue color palette
    return plt.get_cmap('Blues')(np.linspace(0.6, 0.9, bins))


def greens(bins):
    # green color palette
    return plt.get_cmap('Greens')(np.linspace(0.6, 0.5, bins))


def background(*show):
    """
    Creates a static background based on seacharts module
    arg: show = Option for visualization
    return: tuple of x-dimensions amd y-dimensions  of background
    """
    layers = []
    colors = []
    # For every layer put in list and assign a color
    if enc.land:
        land = enc.land
        layers.append(land)
        colors.append(greens(1))
    if enc.shore:
        shore = enc.shore
        layers.append(shore)
        if enc.land:
            del colors[0]
            colors.append(greens(2))
        else:
            colors.append(greens(1))
    if enc.seabed:
        sea = enc.seabed
        sea_k = sea.keys()
        for key in sea_k:
            layers.append(enc.seabed[key])
        colors.append(blues(len(sea_k)))

    # Flatten color list
    colors_new = []
    for sublist in colors:
        for color in sublist:
            colors_new.append(color)
    # Plot the contour for every layer and assign color
    for c, layer in enumerate(layers):
        for i in range(len(layer.geometry)):
            poly = layer.geometry[i]
            x, y = poly.exterior.xy
            plt.fill(x, y, c=colors_new[c], zorder=layer.z_order)
    x_lim = plt.xlim()
    y_lim = plt.ylim()

    # need only dimension of map for initialization
    if not show:
        plt.close()
    return x_lim, y_lim


def draft_to_seabed(draft):
    """
    Randomly chooses a valid sea depth (seabed) depending on ship's draft
    """
    seabeds = [5, 10, 20, 50, 100, 200, 350, 500]
    for seabed in seabeds:
        if not enc.seabed[seabed].geometry:
            seabeds.remove(seabed)
    if 1 <= draft < 2:
        index = random.randint(0, len(seabeds)-1)
    elif 2 <= draft < 5:
        index = random.randint(2, len(seabeds)-1)
    else:
        index = random.randint(3, len(seabeds)-1)
    return seabeds[index]


def start_position(draft):
    """
    Randomly chooses a starting position for the ship depending on ship's draft
    """
    # choose a random valid polygon
    seabed = draft_to_seabed(draft)
    n_polygons = len(enc.seabed[seabed].mapping['coordinates'])-1
    random_index = random.randint(0, n_polygons)
    polygon = enc.seabed[seabed].geometry[random_index]
    # make a rectangular box of points over polygon
    exterior = polygon.exterior.xy
    x_min, x_max = int(min(exterior[0])), int(max(exterior[0]))
    y_min, y_max = int(min(exterior[1])), int(max(exterior[1]))
    step_x, step_y = (x_max - x_min)/10, (y_max - y_min)/10
    x, y = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))
    x, y = x.flatten(), y.flatten()
    points = GeoSeries(map(Point, zip(x, y)))
    # find the points that lay within polygon
    points_inside = []
    for point in points:
        if point.within(polygon):
            points_inside.append(point)
    # choose a randomly over the valid points
    index = random.randint(0, len(points_inside)-1)
    start_pos = points_inside[index]
    return start_pos.x, start_pos.y

