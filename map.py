from shapely.geometry import Point, Polygon
import random
import numpy as np
import matplotlib.pyplot as plt
import seacharts
from geopandas import GeoSeries

# Defining a local region of the map.
size = 7000, 7000              # w, h (east, north) distance in meters
center = 44300, 6956450         # easting/northing (UTM zone 33N)
files = ['More_og_Romsdal.gdb']  # Norwegian county database name

# Creating shapefiles from the defined region
enc = seacharts.ENC(size=size, center=center, files=files, new_data=False)
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
        sea_k = list(sea.keys())
        for key in sea_k:
            if len(sea[key].mapping['coordinates']) == 0:
                sea_k.remove(key)
            layers.append(enc.seabed[key])
        colors.append(blues(len(sea_k)))

    # Flatten color list
    colors_new = []
    for sublist in colors:
        for color in sublist:
            colors_new.append(color)
    # Plot the contour for every layer and assign color
    for c, layer in enumerate(layers):
        for i in range(len(layer.mapping['coordinates'])):
            pol = list(layer.mapping['coordinates'][i][0])
            poly = Polygon(pol)
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
    sea = enc.seabed
    seabed = [5, 10, 20, 50, 100, 200, 350, 500]
    seabed_val = [5, 10, 20, 50, 100, 200, 350, 500]
    for i in seabed:
        if len(sea[i].mapping['coordinates']) == 0:
            seabed_val.remove(i)
        if len(sea[i].mapping['coordinates']) == 1:
            seabed_val.remove(i)
    if 1 <= draft < 2:
        index = random.randint(0, len(seabed_val)-1)
    elif 2 <= draft < 5:
        index = random.randint(2, len(seabed_val)-1)
    else:
        index = random.randint(3, len(seabed_val)-1)
    return seabed_val[index]

def start_position(draft):
    """
    Randomly defining x and y coordinates of a ship inside the safe sea region by considering ship draft.
    """
    # Filtering seabed regions bigger than the ship's draft
    seabed_id = draft_to_seabed(draft)
    safe_sea = enc.seabed[seabed_id]
    # Creating random x and y positions of the ship inside the safe sea area.
    rand_x = random.randint(center[0] - int(size[0]/2) + 900, center[0] + int(size[0]/2) -900)
    rand_y = random.randint(center[1] - int(size[1]/2) + 600, center[1] + int(size[1]/2) -600)
    random_point = Point(rand_x, rand_y)
    while not safe_sea.geometry.contains(random_point):
        rand_x = random.randint(center[0] - int(size[0]/2) + 900, center[0] + int(size[0]/2) -900)
        rand_y = random.randint(center[1] - int(size[1]/2) + 600, center[1] + int(size[1]/2) -600)
        random_point = Point(rand_x, rand_y)
    return rand_x, rand_y



