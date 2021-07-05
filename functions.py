import math
import random
import numpy as np
from matplotlib.patches import Ellipse, Circle
import shapely
from shapely.geometry import Point, Polygon, LineString, GeometryCollection
from map import *


class Ship:
    '''
    x, y : Initial x and y values of the ship.
    speed: Initial speed value.
    heading: Initial true heading or course of the ship. It is a value between 0-359 degrees.
    name: Name of the ship.
    x_t, y_t: x and y position in t future. They are used for course and speed visualization vector.
    noise: Used to create a random value for ship to be off the route.
    '''
    def __init__(self, x, y, speed, heading, length, draft, mmsi):
        self.x = x
        self.y = y
        self.v = speed * 0.51           # converting knots to meters per second
        self.c = math.radians(heading)  # In radians
        self.length = length
        self.draft = draft
        self.mmsi = mmsi
        self.x_t = 0
        self.y_t = 0

        noise = random.uniform(0, 10)
        self.wp = [(self.x + noise, self.y + noise)]

        # Message number 1/2/3 for ships with AIS Class A, 18 for AIS Class B ships
        # AIS Class A is required for ships bigger than 300 GT. Approximately 45 m x 10 m ship would be 300 GT.
        if self.length <= 45:
            self.message_nr = 18
        else:
            self.message_nr = random.randint(1, 3)

    def waypoints(self, wp_number):
        '''
        Creates random waypoints starting from the ship's position.
        wp_number: Number of waypoints to create.
        n: Random distance between waypoints.
        alpha: Angle in radians to make waypoints in zigzag shape.
               A bigger value makes the every odd waypoint away from the initial direction.
        '''
        for each in range(wp_number):
            n = random.randint(200, 1000)
            alpha = random.uniform(0, 0.2)
            wp_x = self.wp[each][0] - n * math.sin(-self.c + alpha)
            wp_y = self.wp[each][1] + n * math.cos(-self.c + alpha)
            # check if the waypoint path is intersecting with the shore polygon
            wp_line = LineString([(self.wp[each][0], self.wp[each][1]), (wp_x, wp_y)])
            if wp_line.intersects(enc.shore.geometry) == True:
                wp_x = self.wp[each][0]
                wp_y = self.wp[each][1]
            self.wp.append((wp_x, wp_y))

        return self.wp

    def move(self, dt):
        self.x -= self.v * math.sin(-self.c) * dt
        self.y += self.v * math.cos(-self.c) * dt
        # check if the ship in the future is inside the shore polygons. If so, stop the ship.
        self.future_pos(10)
        ship_pos = Point(self.x_t, self.y_t)
        if enc.shore.geometry.contains(ship_pos) == True:
            self.v = 0
        # check if the ship's position is 50 meters range of the last waypoint. If so, stop the ship.
        elif self.wp[-1][0] - 50 < self.x < self.wp[-1][0] + 50 and self.wp[-1][1] - 50 < self.y < self.wp[-1][1] + 50:
            self.v = 0

    def follow_waypoints(self, dt, waypoints, each):
        #if waypoints[each][0] - 100 < self.x < waypoints[each][0] + 100 and waypoints[each][1] - 100 < self.y < waypoints[each][1] + 100:
        los_angle = math.degrees(math.atan2((waypoints[each + 1][0] - self.x),
                                                (waypoints[each + 1][1] - self.y)))
        if los_angle < 0:
            los_angle = 360 + los_angle
        self.c = math.radians(los_angle)

    def future_pos(self, time):
        # Future position in defined time will be used to visualize ship's heading
        self.x_t = self.x - self.v * math.sin(-self.c) * time
        self.y_t = self.y + self.v * math.cos(-self.c) * time

'''
def distance(ship1, ship2):
    # calculates euclidean distance between the ships (pixels)
    euc_dist = int(math.sqrt((ship1.x - ship2.x) ** 2 + (ship1.y - ship2.y) ** 2))
    return euc_dist


def true_bearing(obj1, obj2):
    # True bearing (Clockwise angle from the North)
    true_bearing = math.degrees(math.atan2((obj2.x - obj1.x), (obj2.y - obj1.y)))
    if true_bearing < 0:
        true_bearing = 360 + true_bearing
    return true_bearing


def relative_bearing(true_bearing, course):
    # Relative bearing of the other ship (Angle on the port or starboard side from the ships heading)
    relative_bearing = true_bearing - math.degrees(course)
    if relative_bearing > 180:
        relative_bearing = relative_bearing - 360
    elif relative_bearing < -180:
        relative_bearing = relative_bearing + 360
    return int(relative_bearing)


def dcpa_tcpa(ship1, ship2, distance, true_bearing):
    # NEEDS SOME WORK IT DOESN'T SHOW THE CORRECT VALUE!
    # Reference -> https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9314130

    # relative velocity
    vr = ship1.v * math.sqrt(1 + (ship2.v / ship1.v) ** 2 - 2 * (ship2.v / ship1.v) * math.cos(ship1.c - ship2.c))

    # relative angle calculated from relative velocity vector
    rel_angle = math.acos((ship1.v - ship2.v * math.cos(ship1.c - ship2.c)) / vr)

    dcpa = int(distance * math.sin(rel_angle - math.radians(true_bearing) - math.pi))
    tcpa = int(distance * math.cos(rel_angle - math.radians(true_bearing) - math.pi) / vr)
    return dcpa, tcpa


def iter_colreg_rules(own_ship, ship_list, colreg_rules):
    # Demonstrates the COLREG rule to be applied.
    # rel_bearing_os_ts: Relative bearing of TS from OS
    # rel_bearing_ts_os: Relative bearing of OS from TS

    for each in ship_list[1:]:
        tr_bearing_os_ts = true_bearing(own_ship, each)
        rel_bearing_os_ts = relative_bearing(tr_bearing_os_ts, own_ship.c)

        tr_bearing_ts_os = true_bearing(each, own_ship)
        rel_bearing_ts_os = relative_bearing(tr_bearing_ts_os, each.c)

        if abs(rel_bearing_os_ts) <= 14 and abs(rel_bearing_ts_os) <= 14:
            colreg_rules.append({each.name: 'Head-on'})
        elif abs(rel_bearing_os_ts) > 112.5 and abs(rel_bearing_ts_os) < 13:
            colreg_rules.append({each.name: 'Overtaken'})
        elif abs(rel_bearing_ts_os) > 112.5 and abs(rel_bearing_os_ts) < 13:
            colreg_rules.append({each.name: 'Overtaking'})
        elif rel_bearing_os_ts > 0 and rel_bearing_os_ts < 112.5 and rel_bearing_ts_os < 0 and rel_bearing_ts_os > -112.5:
            colreg_rules.append({each.name: 'Crossing, Give way!'})
        elif rel_bearing_os_ts < 0 and rel_bearing_os_ts > -112.5 and rel_bearing_ts_os > 0 and rel_bearing_ts_os < 112.5:
            colreg_rules.append({each.name: 'Crossing, Stand on!'})
        else:
            colreg_rules.append({each.name: '-'})
'''