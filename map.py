from shapely.geometry import Point, Polygon, LineString, GeometryCollection
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import seacharts
import shapely.ops as so
import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib.pyplot import figure


def polygon(x0, y0, size):
    polygon1 = Polygon([(x0 - size, y0 - size), (x0 + size, y0 - size), (x0 + size, y0 + size), (x0 - size, y0 + size)])
    return polygon1


def random_point_polygon(poly, map_width, map_length):
    x_r = random.randint(-map_width / 2, map_width / 2)
    y_r = random.randint(-map_length / 2, map_length / 2)
    p = Point(x_r, y_r)
    print('before')
    while poly.within(p):
        print('inside')
        x_r = random.randint(-map_width / 2, map_width / 2)
        y_r = random.randint(-map_length / 2, map_length / 2)
        p = Point(x_r, y_r)
    return x_r, y_r


enc = seacharts.ENC()
enc.close_display()



def blues(bins):
    return plt.get_cmap('Blues')(np.linspace(0.6, 0.9, bins))

def greens(bins):
    return plt.get_cmap('Greens')(np.linspace(0.6, 0.5, bins))



#figure(figsize=(9, 6), dpi=80)

def background():
    layers = []
    colors = []
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

    colors_new = []
    for sublist in colors:
        for color in sublist:
            colors_new.append(color)

    for c, layer in enumerate(layers):
        for i in range(len(layer.geometry)):
            poly = layer.geometry[i]
            x, y = poly.exterior.xy
            plt.fill(x, y, c =colors_new[c], zorder=layer.z_order)
    #print(plt.xlim())
    #print(plt.ylim())
    #plt.show()



#background()


