import random
import math
from functions import Ship
import pandas as pd


def random_pose(map_width, map_length, os_max_speed):
    # random x and y values from map dimension data
    x = random.randint(-map_width / 2, map_width / 2)
    y = random.randint(-map_length / 2, map_length / 2)

    # random speed
    speed = random.randint(1, os_max_speed)

    # random heading angle in degrees
    heading = random.randint(0, 359)

    return x, y, speed, heading


def head_on(map_width, map_length, os_max_speed, ts_max_speed):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(map_width, map_length, os_max_speed)

    # random target ship considering own ship pose
    x2 = x1 + random.randint(500, 1000) * math.sin(math.radians(heading1)) + random.randint(0, 200)
    y2 = y1 + random.randint(500, 1000) * math.cos(math.radians(heading1)) + random.randint(0, 200)
    speed2 = random.randint(1, ts_max_speed)
    heading2 = heading1 + 180 + random.uniform(-14, 14)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def overtaking(map_width, map_length, os_max_speed):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(map_width, map_length, os_max_speed)

    # random target ship considering own ship pose
    x2 = x1 + random.randint(500, 1000) * math.sin(math.radians(heading1)) + random.randint(0, 100)
    y2 = y1 + random.randint(500, 1000) * math.cos(math.radians(heading1)) + random.randint(0, 100)
    speed2 = speed1 - speed1 * random.uniform(0.5, 0.9)
    heading2 = heading1 + random.uniform(-13, 13)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def overtaken(map_width, map_length, os_max_speed):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(map_width, map_length, os_max_speed)

    # random target ship considering own ship pose
    x2 = x1 - random.randint(500, 1000) * math.sin(math.radians(heading1)) + random.randint(0, 100)
    y2 = y1 - random.randint(500, 1000) * math.cos(math.radians(heading1)) + random.randint(0, 100)
    speed2 = speed1 + speed1 * random.uniform(0.5, 0.9)
    heading2 = heading1 + random.uniform(-13, 13)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def crossing_give_way(map_width, map_length, os_max_speed, ts_max_speed):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(map_width, map_length, os_max_speed)

    # random target ship considering own ship pose
    n = random.uniform(0, 112.5)
    x2 = x1 + random.randint(500, 1000) * math.sin(math.radians(heading1 + n))
    y2 = y1 + random.randint(500, 1000) * math.cos(math.radians(heading1 + n))
    speed2 = random.randint(1, ts_max_speed)
    heading2 = heading1 - 90

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def crossing_stand_on(map_width, map_length, os_max_speed, ts_max_speed):
    # random own ship
    x1, y1, speed1, heading1 = random_pose(map_width, map_length, os_max_speed)

    # random target ship considering own ship pose
    n = random.uniform(-112.5, 0)
    x2 = x1 + random.randint(500, 1000) * math.sin(math.radians(heading1 + n))
    y2 = y1 + random.randint(500, 1000) * math.cos(math.radians(heading1 + n))
    speed2 = random.randint(1, ts_max_speed)
    heading2 = heading1 + 90

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def random_scenario_generator(scenario_num, map_width, map_length, os_max_speed, ts_max_speed):
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
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(n, map_width, map_length, os_max_speed, ts_max_speed)
    elif scenario_num == 1:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = head_on(map_width, map_length, os_max_speed, ts_max_speed)
    elif scenario_num == 2:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = overtaking(map_width, map_length, os_max_speed)
    elif scenario_num == 3:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = overtaken(map_width, map_length, os_max_speed)
    elif scenario_num == 4:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = crossing_give_way(map_width, map_length, os_max_speed, ts_max_speed)
    elif scenario_num == 5:
        x1, y1, speed1, heading1, x2, y2, speed2, heading2 = crossing_stand_on(map_width, map_length, os_max_speed, ts_max_speed)

    return x1, y1, speed1, heading1, x2, y2, speed2, heading2


def ship_generator(Ship, scenario_num, map_width, map_length, os_max_speed, ts_max_speed, ship_number):
    ship_list = []
    x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(scenario_num, map_width, map_length, os_max_speed, ts_max_speed)
    ship1 = Ship(x1, y1, speed1, heading1, name='Ship1')
    ship2 = Ship(x2, y2, speed2, heading2, name='Ship2')
    ship_list.append(ship1)
    ship_list.append(ship2)
    ship_number = ship_number - 2
    for i in range(ship_number):
        x, y, speed, heading = random_pose(map_width, map_length, ts_max_speed)
        ship_i = Ship(x, y, speed, heading, name=f'Ship{i+3}')
        ship_list.append(ship_i)
    return ship_list

def waypoint_generator(ships, waypoints_number):
    waypoint_list = []
    for i in range(len(ships)):
        waypoints = ships[i].waypoints(waypoints_number)
        waypoint_list.append(waypoints)
    return waypoint_list


def ship_data(ships, waypoints, time, timestep):
    data = {}
    for i in range(1, len(ships)+1):
        x_i, y_i, x_i_t, y_i_t, w = [], [], [], [], []
        data[f'Ship{i}'] = [x_i, y_i, x_i_t, y_i_t, w]
        try: data[f'Ship{i}'][4] = waypoints[i-1]
        except: print(f'Ship{i} has no waypoints')

    for i in range(len(time)):
        for ix, ship in enumerate(ships):
            ship.move(timestep)
            data[f'Ship{ix+1}'][0].append(int(ship.x))
            data[f'Ship{ix+1}'][1].append(int(ship.y))
            ship.future_pos(10)
            data[f'Ship{ix + 1}'][2].append(int(ship.x_t))
            data[f'Ship{ix + 1}'][3].append(int(ship.y_t))
            if data[f'Ship{ix + 1}'][4]:
                for each in range(1, len(data[f'Ship{ix + 1}'][4]) - 1):
                    ship.follow_waypoints(timestep, data[f'Ship{ix + 1}'][4], each)
    return data

















