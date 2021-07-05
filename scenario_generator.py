import random
import math
import numpy as np
from functions import Ship
from map import start_position, min_distance_to_land
import pandas as pd


def random_pose(os_max_speed):
    # random ship length
    length = random.randint(10, 200)
    # ship draft
    draft = length/10
    # start position for ship
    x, y = start_position(draft)
    # random speed
    speed = round(random.uniform(1, os_max_speed), 1)
    # random heading angle in degrees
    heading = random.randint(0, 359)
    return x, y, speed, heading, length, draft


def head_on(os_max_speed, ts_max_speed):
    # random own ship
    x1, y1, speed1, heading1, length1, draft1 = random_pose(os_max_speed)

    # random target ship considering own ship pose
    distance_land = min_distance_to_land(x1, y1)
    x2 = x1 + distance_land * math.sin(math.radians(heading1))
    y2 = y1 + distance_land * math.cos(math.radians(heading1))
    speed2 = round(random.uniform(1, ts_max_speed), 1)
    heading2 = heading1 + 180 + random.uniform(-14, 14)
    length2 = random.randint(10, 200)
    draft2 = length2/10

    return x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2


def overtaking(os_max_speed):
    # random own ship
    x1, y1, speed1, heading1, length1, draft1 = random_pose(os_max_speed)

    # random target ship considering own ship pose
    distance_land = min_distance_to_land(x1, y1)
    x2 = x1 + distance_land * math.sin(math.radians(heading1))
    y2 = y1 + distance_land * math.cos(math.radians(heading1))
    speed2 = round((speed1 - speed1 * random.uniform(0.5, 0.9)), 1)
    heading2 = heading1 + random.uniform(-13, 13)
    length2 = random.randint(10, 200)
    draft2 = length2/10

    return x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2


def overtaken(os_max_speed):
    # random own ship
    x1, y1, speed1, heading1, length1, draft1 = random_pose(os_max_speed)

    # random target ship considering own ship pose
    distance_land = min_distance_to_land(x1, y1)
    x2 = x1 - distance_land * math.sin(math.radians(heading1))
    y2 = y1 - distance_land * math.cos(math.radians(heading1))
    speed2 = round((speed1 + speed1 * random.uniform(0.5, 0.9)), 1)
    heading2 = heading1 + random.uniform(-13, 13)
    length2 = random.randint(10, 200)
    draft2 = length2/2

    return x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2


def crossing_give_way(os_max_speed, ts_max_speed):
    # random own ship
    x1, y1, speed1, heading1, length1, draft1 = random_pose(os_max_speed)

    # random target ship considering own ship pose
    n = random.uniform(0, 112.5)
    distance_land = min_distance_to_land(x1, y1)
    x2 = x1 + distance_land * math.sin(math.radians(heading1 + n))
    y2 = y1 + distance_land * math.cos(math.radians(heading1 + n))
    speed2 = round(random.uniform(1, ts_max_speed), 1)
    heading2 = heading1 - 90
    length2 = random.randint(10, 200)
    draft2 = length2/10

    return x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2


def crossing_stand_on(os_max_speed, ts_max_speed):
    # random own ship
    x1, y1, speed1, heading1, length1, draft1 = random_pose(os_max_speed)

    # random target ship considering own ship pose
    n = random.uniform(-112.5, 0)
    distance_land = min_distance_to_land(x1, y1)
    x2 = x1 + distance_land * math.sin(math.radians(heading1 + n))
    y2 = y1 + distance_land * math.cos(math.radians(heading1 + n))
    speed2 = round(random.randint(1, ts_max_speed), 1)
    heading2 = heading1 + 90
    length2 = random.randint(10, 200)
    draft2 = length2/10

    return x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2


def random_scenario_generator(scenario_num, os_max_speed, ts_max_speed):
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
        x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2 = random_scenario_generator(n, os_max_speed, ts_max_speed)
    elif scenario_num == 1:
        x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2 = head_on(os_max_speed, ts_max_speed)
    elif scenario_num == 2:
        x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2 = overtaking( os_max_speed)
    elif scenario_num == 3:
        x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2 = overtaken(os_max_speed)
    elif scenario_num == 4:
        x1, y1, speed1, heading1, length1, draft1,  x2, y2, speed2, heading2, length2, draft2 = crossing_give_way( os_max_speed, ts_max_speed)
    elif scenario_num == 5:
        x1, y1, speed1, heading1, length1, draft1,  x2, y2, speed2, heading2, length2, draft2 = crossing_stand_on(os_max_speed, ts_max_speed)

    return x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2


def ship_generator(Ship, scenario_num, os_max_speed, ts_max_speed, ship_number):
    ship_list = []
    x1, y1, speed1, heading1, length1, draft1, x2, y2, speed2, heading2, length2, draft2 = random_scenario_generator(scenario_num, os_max_speed, ts_max_speed)
    ship1 = Ship(x1, y1, speed1, heading1, length1, draft1, mmsi='Ship1')
    ship2 = Ship(x2, y2, speed2, heading2, length2, draft2, mmsi='Ship2')
    ship_list.append(ship1)
    ship_list.append(ship2)

    ship_number = ship_number - 2
    for i in range(ship_number):
        x, y, speed, heading, length, draft = random_pose(ts_max_speed)
        ship_i = Ship(x, y, speed, heading, length, draft, mmsi=f'Ship{i+3}')
        ship_list.append(ship_i)

    return ship_list


def waypoint_generator(ships, waypoints_number):
    waypoint_list = []
    for i in range(len(ships)):
        waypoints = ships[i].waypoints(waypoints_number)
        waypoint_list.append(waypoints)
    return waypoint_list


"""
def ship_data(ships, waypoint_list, time, timestep):
    '''
    :param ships: List of ships which is created from ship_generator().
    :param waypoint_list: List of waypoints which is created from waypoint_generator().
    :param time: Array of time of the simulation.
    :param timestep: Defined time step (dt)
    :return: Returns data for visualization and ais_data for evaluation library (EvalTool()).
    '''
    ais_data = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'date_time_utc', 'sog', 'cog',
                                     'true_heading', 'nav_status', 'message_nr', 'source'])

    data = {}
    for i in range(1, len(ships)+1):
        x_i, y_i, x_i_t, y_i_t, w = [], [], [], [], []
        data[f'Ship{i}'] = [x_i, y_i, x_i_t, y_i_t, w]
        try: data[f'Ship{i}'][4] = waypoint_list[i-1]
        except: print(f'Ship{i} has no waypoints')

    for i in range(len(time)):
        for ix, ship in enumerate(ships):
            # Creates ships movement data
            ship.move(timestep)
            data[f'Ship{ix+1}'][0].append(int(ship.x))
            data[f'Ship{ix+1}'][1].append(int(ship.y))

            # Creates ships future positions for speed vector visualization
            ship.future_pos(10)
            data[f'Ship{ix + 1}'][2].append(int(ship.x_t))
            data[f'Ship{ix + 1}'][3].append(int(ship.y_t))

            # Ships follow waypoints
            if data[f'Ship{ix + 1}'][4]:
                for each in range(1, len(data[f'Ship{ix + 1}'][4]) - 1):
                    ship.follow_waypoints(timestep, data[f'Ship{ix + 1}'][4], each)

            # writing instantaneous ship data to the ais_data dataframe.
            row = {'mmsi': ship.mmsi, 'lon': ship.x, 'lat': ship.y, 'date_time_utc': i,
                   'sog': ship.v, 'cog': int(math.degrees(ship.c)), 'true_heading': int(math.degrees(ship.c)),
                   'nav_status': None, 'message_nr': ship.message_nr, 'source': ''}
            ais_data = ais_data.append(row, ignore_index=True)

    return data, ais_data
"""


def ship_data(ships, waypoint_list, time, timestep):
    '''
    :param ships: List of ships which is created from ship_generator().
    :param waypoint_list: List of waypoints which is created from waypoint_generator().
    :param time: Array of time of the simulation.
    :param timestep: Defined time step (dt)
    :return: Returns data for visualization and ais_data for evaluation library (EvalTool()).
    '''
    ais_data = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'date_time_utc', 'sog', 'cog',
                                     'true_heading', 'nav_status', 'message_nr', 'source'])

    data = {}
    for i in range(1, len(ships)+1):
        x_i = np.zeros(len(time))
        y_i = np.zeros(len(time))
        x_i_t = np.zeros(len(time))
        y_i_t = np.zeros(len(time))
        w = np.zeros(len(time))
        data[f'Ship{i}'] = [x_i, y_i, x_i_t, y_i_t, w]
        try: data[f'Ship{i}'][4] = waypoint_list[i-1]
        except: print(f'Ship{i} has no waypoints')

    for i in range(len(time)):
        for ix, ship in enumerate(ships):
            # Creates ships movement data
            ship.move(timestep)
            data[f'Ship{ix+1}'][0][i] = int(ship.x)
            data[f'Ship{ix+1}'][1][i] = int(ship.y)

            # Creates ships future positions for speed vector visualization
            ship.future_pos(10)
            data[f'Ship{ix + 1}'][2][i] = int(ship.x_t)
            data[f'Ship{ix + 1}'][3][i] = int(ship.y_t)

            # Ships follow waypoints
            if data[f'Ship{ix + 1}'][4]:
                for each in range(1, len(data[f'Ship{ix + 1}'][4]) - 1):
                    ship.follow_waypoints(timestep, data[f'Ship{ix + 1}'][4], each)

            # writing instantaneous ship data to the ais_data dataframe.
            row = {'mmsi': ship.mmsi, 'lon': ship.x, 'lat': ship.y, 'date_time_utc': i,
                   'sog': ship.v, 'cog': int(math.degrees(ship.c)), 'true_heading': int(math.degrees(ship.c)),
                   'nav_status': None, 'message_nr': ship.message_nr, 'source': ''}
            ais_data = ais_data.append(row, ignore_index=True)

    return data, ais_data
















