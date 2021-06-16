from shapely.geometry import Point, Polygon, LineString, GeometryCollection
import random


def polygon(x0, y0, size):
    polygon1 = Polygon([(x0 - size, y0 - size), (x0 + size, y0 - size), (x0 + size, y0 + size), (x0 - size, y0 + size)])
    return polygon1


def random_point_polygon(poly, map_width, map_length):
    x_r = random.randint(-map_width / 2, map_width / 2)
    y_r = random.randint(-map_length / 2, map_length / 2)
    p = Point(x_r, y_r)
    while poly.within(p):
        x_r = random.randint(-map_width / 2, map_width / 2)
        y_r = random.randint(-map_length / 2, map_length / 2)
        p = Point(x_r, y_r)
    return x_r, y_r



