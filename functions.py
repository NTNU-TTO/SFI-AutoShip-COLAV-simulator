import math
import numpy as np
import random



class Ship:
    def __init__(self, x, y, speed, heading, name):
        self.x = x
        self.y = y
        self.v = speed
        self.c = math.radians(-heading)    # In radians
        self.pivot = 100    # Length of pivot point from the rudder
        self.name = name
        self.x_t = 0        # x position in the future for course visualisation
        self.y_t = 0        # y position in the future for course visualisation

    def move(self, dt):
        self.x -= self.v * math.sin(-self.c) * dt
        self.y += self.v * math.cos(-self.c) * dt

    def future_pos(self, time):
        self.x_t = self.x - self.v * math.sin(-self.c) * time
        self.y_t = self.y + self.v * math.cos(-self.c) * time

    def update_heading(self, w1, w2):
        print('before update', math.degrees(self.c))
        self.c += math.radians(270 + angle(w1,w2))
        print('after update', math.degrees(self.c))
        print('---')







class waypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def angle( obj1, obj2):
    angle = math.degrees(math.atan2((obj2.x - obj1.x), (obj2.y - obj1.y)))
    if angle < 0:
        angle = 360 + angle
    return angle

def angle1(obj1,obj2):
    distance = distance1(obj1, obj2)
    angle = math.acos((obj2.x - obj1.x) / distance)
    if obj1.y > obj2.y:
        angle = 2 * math.pi - angle
    return math.degrees(angle)


def distance1( obj1, obj2):
    euc_dist = int(math.sqrt((obj1.x - obj2.x) ** 2 + (obj1.y - obj2.y) ** 2))
    return euc_dist





def distance(ship1_x, ship1_y, ship2_x, ship2_y):
    euc_dist = int(math.sqrt((ship1_x - ship2_x)**2 + (ship1_y - ship2_y)**2))
    return euc_dist


def true_bearing(obj1, obj2):
    true_bearing = math.degrees(math.atan2((obj2.x - obj1.x), (obj2.y - obj1.y)))
    if true_bearing < 0:
        true_bearing = 360 + true_bearing
    return true_bearing


def relative_bearing(true_bearing, course):
    relative_bearing = true_bearing - course
    if relative_bearing > 180:
        relative_bearing = relative_bearing - 360
    elif relative_bearing < -180:
        relative_bearing = relative_bearing + 360
    return int(relative_bearing)


def dcpa_tcpa(ship1, ship2, distance, TB_os_ts):
    # Reference -> https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9314130
    vr = ship1.speed * math.sqrt(1 + (ship2.speed / ship1.speed)**2 - 2 * (ship2.speed / ship1.speed) *
                                 math.cos(math.radians(ship1.course - ship2.course)))
    psi_rel = math.acos((ship1.speed - ship2.speed * math.cos(math.radians(ship1.course - ship2.course))) / vr)

    dcpa = int(distance * math.sin(psi_rel - math.radians(TB_os_ts) - math.pi))
    tcpa = int(distance * math.cos(psi_rel - math.radians(TB_os_ts) - math.pi) / vr)
    if tcpa < 0:
        dcpa = 0
        tcpa = 0
    return dcpa, tcpa


def colreg_rule(RB_os_ts, RB_ts_os):
    # RB_os_ts: Relative bearing of TS from OS
    # RB_ts_os: Relative bearing of OS from TS
    if abs(RB_os_ts) <= 14 and abs(RB_ts_os) <= 14:
        rule = 'Head-on'
    elif abs(RB_os_ts) > 112.5 and abs(RB_ts_os) < 13:
        rule = 'Overtaken'
    elif abs(RB_ts_os) > 112.5 and abs(RB_os_ts) < 13:
        rule = 'Overtaking'
    elif RB_os_ts > 0 and RB_os_ts < 112.5 and RB_ts_os < 0 and RB_ts_os > -112.5:
        rule = 'Crossing, Giwe way!'
    elif RB_os_ts < 0 and RB_os_ts > -112.5 and RB_ts_os > 0 and RB_ts_os < 112.5:
        rule = 'Crossing, Stand on!'
    else:
        rule = '-'
    return rule



