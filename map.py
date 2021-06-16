from shapely.geometry import Point, Polygon, LineString, GeometryCollection
import random


def polygon(x0, y0, size):
    polygon1 = Polygon([(x0 - size, y0 - size), (x0 + size, y0 - size), (x0 + size, y0 + size), (x0 - size, y0 + size)])
    return polygon1


def random_point_polygon(poly):
    x, y = poly.exterior.coords.xy
    x_min, x_max = x[0], x[1]
    y_min, y_max = y[0], y[2]
    x_r = random.randint(x_min, x_max)
    y_r = random.randint(y_min, y_max)
    return x_r, y_r

