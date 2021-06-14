import math
import random


class Ship:
    def __init__(self, x, y, speed, heading, name):
        self.x = x
        self.y = y
        self.v = speed
        self.c = math.radians(heading)  # In radians
        self.pivot = 100  # Length of pivot point from the rudder
        self.name = name
        self.x_t = 0  # x position in the future for course visualisation
        self.y_t = 0  # y position in the future for course visualisation

        # noise for waypoints
        noise = random.uniform(0, 20)
        self.wp = [(self.x + noise, self.y + noise)]

    def waypoints(self, wp_number):
        for each in range(wp_number):
            n = random.randint(200, 1000)
            alpha = random.uniform(0, 0.3)
            if each % 2 != 0:
                self.wp.append((self.wp[each][0] - n * math.sin(-self.c + alpha),
                                self.wp[each][1] + n * math.cos(-self.c + alpha)))
            else:
                self.wp.append((self.wp[each][0] - n * math.sin(-self.c),
                                self.wp[each][1] + n * math.cos(-self.c)))
        return self.wp

    def move(self, dt):
        self.x -= self.v * math.sin(-self.c) * dt
        self.y += self.v * math.cos(-self.c) * dt

    def follow_waypoints(self, dt, waypoints, each):
        if waypoints[each][0] - 50 < self.x < waypoints[each][0] + 50 and \
                waypoints[each][1] - 50 < self.y < waypoints[each][1] + 50:
            los_angle = math.degrees(math.atan2((waypoints[each + 1][0] - self.x),
                                                (waypoints[each + 1][1] - self.y)))
            if los_angle < 0:
                los_angle = 360 + los_angle
            self.c = math.radians(los_angle)

    def future_pos(self, time):
        # Future position in defined time will be used to visualize ship's heading
        self.x_t = self.x - self.v * math.sin(-self.c) * time
        self.y_t = self.y + self.v * math.cos(-self.c) * time


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
