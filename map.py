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



def blues(bins=11):
    return plt.get_cmap('Blues')(np.linspace(0.6, 0.9, bins))

def greens(bins=9):
    return plt.get_cmap('Greens')(np.linspace(0.3, 0.9, bins))

blue = blues(11)
#print(blue[0])

#print(blue[0])
#np.append( blue,'mediumblue')
#np.append( blue,'darkblue')
#np.append( blue,'midnightblue')


#figure(figsize=(9, 6), dpi=80)

def background():
    '''
    layers = []
    colors = []
    if enc.land:
        land = enc.land
        layers.append(land)
        colors.append(greens(1))
    elif enc.shore:
        shore = enc.shore
        layers.append(shore)
        if enc.land:
            del colors[0]
        else:
            colors.append(greens(2))
        colors
    '''

    land = enc.land
    shore = enc.shore
    sea = enc.seabed.keys()
    layers = [land, shore]
    for key in sea:
        layers.append(enc.seabed[key])
    colors = ['g', 'wheat', 'lightblue', 'skyblue', 'deepskyblue', 'cornflowerblue', 'royalblue', 'blue', 'mediumblue', 'darkblue', 'midnightblue']
    for c, layer in enumerate(layers):
        for i in range(len(layer.geometry)):
            poly = layer.geometry[i]
            x, y = poly.exterior.xy
            plt.fill(x, y, c =blue[c], zorder=layer.z_order)
    #print(plt.xlim())
    #print(plt.ylim())
    #plt.show()



#background()


