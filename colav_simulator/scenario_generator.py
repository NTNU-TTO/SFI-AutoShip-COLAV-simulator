import math
import random

import numpy as np

np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})

from map import min_distance_to_land, path_crosses_land, start_position
from utils import create_ship_model


def create_scenario(num_ships, scenario_num, ship_model_name_list, os_max_speed, ts_max_speed, wp_number):
    """
        Creates a COLREG scenario (based on scenario_num), with a random plans for all ships
    """
    # Create the initial poses
    x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(scenario_num, os_max_speed, ts_max_speed, ship_model_name_list[0])
    pose_list = [[x1, y1, speed1, heading1], [x2, y2, speed2, heading2]]
    for i in range(2, num_ships):
        x, y, speed, heading = random_pose(ts_max_speed, ship_model_name=ship_model_name_list[i])
        pose_list.append([x, y, speed, heading])

    # Create plan (waypoints, speed_plan) for all ships
    waypoint_list = []
    speed_plan_list = []
    for i in range(num_ships):
        wp = create_random_waypoints(pose_list[i][0], pose_list[i][1], pose_list[i][3], wp_number)
        speed_plan = create_random_speed_plan(pose_list[i][2], wp_number)
        waypoint_list.append(wp)
        speed_plan_list.append(speed_plan)
    return pose_list, waypoint_list, speed_plan_list

def random_pose(os_max_speed, ship_model_name):
    own_ship_model = create_ship_model(ship_model_name)
    draft = own_ship_model.draft

    # start position for ship
    x, y = start_position(draft)
    # random speed
    speed = round(random.uniform(1, os_max_speed), 1)
    # random heading angle in degrees
    heading = random.randint(0, 359)
    return x, y, speed, heading

def create_random_waypoints(x, y, psi, wp_number):
    '''
    Creates random waypoints starting from the ship's position.
    wp_number: Number of waypoints to create.
    n: Random distance between waypoints.
    alpha: Angle in radians to make waypoints in zigzag shape.
            A bigger value makes the every odd waypoint away from the initial direction.
    '''
    wp = []
    wp.append((x, y)) # First waypoint init pos
    for each in range(wp_number):
        n = random.randint(200, 1000)
        alpha = random.uniform(0, 0.7)
        if not each:
            alpha = 0
        wp_x = wp[each][0] + n * math.cos(math.radians(psi) + alpha)
        wp_y = wp[each][1] + n * math.sin(math.radians(psi) + alpha)
        # check if the waypoint path is intersecting with the shore polygon
        if path_crosses_land((wp[each][1], wp[each][0]), (wp_y, wp_x)):
            wp_x = wp[each][0]
            wp_y = wp[each][1]
        wp.append((wp_x, wp_y))
    return wp

def create_random_speed_plan(speed, wp_number):
    '''
    Creates random waypoints starting from the ship's position.
    wp_number: Number of waypoints to create.
    n: Random distance between waypoints.
    alpha: Angle in radians to make waypoints in zigzag shape.
            A bigger value makes the every odd waypoint away from the initial direction.
    '''
    speed_plan = []
    speed_plan.append(speed) # First speed init speed
    for each in range(wp_number):
        U = speed + random.uniform(-1, 1)
        speed_plan.append(U)
    return speed_plan

def head_on(os_max_speed, ts_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    distance_land = min_distance_to_land(y1, x1)
    x2 = x1 + distance_land * math.cos(math.radians(heading1))
    y2 = y1 + distance_land * math.sin(math.radians(heading1))
    speed2 = round(random.uniform(1, ts_max_speed), 1)
    heading2 = heading1 + 180 + random.uniform(-14, 14)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def overtaking(os_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    distance_land = min_distance_to_land(y1, x1)
    x2 = x1 + distance_land * math.cos(math.radians(heading1))
    y2 = y1 + distance_land * math.sin(math.radians(heading1))
    speed2 = round((speed1 - speed1 * random.uniform(0.5, 0.9)), 1)
    heading2 = heading1 + random.uniform(-13, 13)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def overtaken(os_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    distance_land = min_distance_to_land(y1, x1)
    x2 = x1 - distance_land * math.cos(math.radians(heading1))
    y2 = y1 - distance_land * math.sin(math.radians(heading1))
    speed2 = round((speed1 + speed1 * random.uniform(0.5, 0.9)), 1)
    heading2 = heading1 + random.uniform(-13, 13)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def crossing_give_way(os_max_speed, ts_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    n = random.uniform(0, 112.5)
    distance_land = min_distance_to_land(y1, x1)
    x2 = x1 + distance_land * math.cos(math.radians(heading1 + n))
    y2 = y1 + distance_land * math.sin(math.radians(heading1 + n))
    speed2 = round(random.uniform(1, ts_max_speed), 1)
    heading2 = heading1 - 90

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def crossing_stand_on(os_max_speed, ts_max_speed, ship_model_name):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(os_max_speed, ship_model_name)

    # random target ship considering own ship pose
    n = random.uniform(-112.5, 0)
    distance_land = min_distance_to_land(y1, x1)
    x2 = x1 + distance_land * math.cos(math.radians(heading1 + n))
    y2 = y1 + distance_land * math.sin(math.radians(heading1 + n))
    speed2 = round(random.randint(1, ts_max_speed), 1)
    heading2 = heading1 + 90

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def random_scenario_generator(scenario_num, os_max_speed, ts_max_speed, ship_model_name):
    '''
        scenario_num = 0 -> random selection
        scenario_num = 1 -> head on
        scenario_num = 2 -> overtaking
        scenario_num = 3 -> overtaken
        scenario_num = 4 -> crossing give way
        scenario_num = 5 -> crossing stand on
    '''
    if scenario_num == 0:
        n = random.randint(1, 5)
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(n, os_max_speed, ts_max_speed, ship_model_name)
    elif scenario_num == 1:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = head_on(os_max_speed, ts_max_speed, ship_model_name)
    elif scenario_num == 2:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = overtaking( os_max_speed, ship_model_name)
    elif scenario_num == 3:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = overtaken(os_max_speed, ship_model_name)
    elif scenario_num == 4:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = crossing_give_way( os_max_speed, ts_max_speed, ship_model_name)
    elif scenario_num == 5:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = crossing_stand_on(os_max_speed, ts_max_speed, ship_model_name)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2
