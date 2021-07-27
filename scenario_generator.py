import random
import math
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import json
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})
from functions import Ship
from map import start_position, min_distance_to_land, enc
from sensors import *
from utils import create_ship_model
from read_config import read_ship_config
from shapely.geometry import Point, Polygon, LineString, GeometryCollection

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
        wp_line = LineString([(wp[each][1], wp[each][0]), (wp_y, wp_x)])
        if wp_line.intersects(enc.shore.geometry) == True:
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


""""def ship_generator(scenario_num, os_max_speed, ts_max_speed, ship_number, ship_model_name):
    ship_list = []
    x1, y1, speed1, heading1, x2, y2, speed2, heading2 = random_scenario_generator(scenario_num, os_max_speed, ts_max_speed, ship_model_name)
    ship1 = Ship(x1, y1, speed1, heading1, ship_model_name, mmsi='Ship1')
    ship2 = Ship(x2, y2, speed2, heading2, ship_model_name='random', mmsi='Ship2')
    ship_list.append(ship1)
    ship_list.append(ship2)
    ship_number = ship_number - 2
    for i in range(ship_number):
        x, y, speed, heading = random_pose(ts_max_speed, ship_model_name='random')
        ship_i = Ship(x, y, speed, heading, ship_model_name='random', mmsi=f'Ship{i+3}')
        ship_list.append(ship_i)

    return ship_list


def waypoint_generator(ships, waypoints_number):
    waypoint_list = []
    for i in range(len(ships)):
        waypoints = ships[i].waypoints(waypoints_number)
        waypoint_list.append(waypoints)
    return waypoint_list"""


def create_colav_input(ships, time):
    """
        Creates input data to use with PSB-MPC colav algorithm
    """
    colav_input = {}

    # time information
    colav_input['time'] = time

    # own ship states [x, y, psi, u, v, r]
    colav_input['os_states'] = np.array([round(ships[0].x, 2), round(ships[0].y, 2), int(ships[0].psi),
                                         round(ships[0].u, 2), round(ships[0].v, 2), round(ships[0].r, 0)])

    # own ship's reference surge and course to the next waypoint
    colav_input['ref_surge'] = round(ships[0].u, 2)
    colav_input['ref_course'] = int(ships[0].los_angle) # in radians

    # remaining waypoint coordinates
    colav_input['remaining_wp'] = ships[0].wp[ships[0].idx_next_wp:]

    # polygons coordinates
    #colav_input['polygons'] = enc.shore.mapping['coordinates']

    # target ships states [x, y, psi, u, v, A, B, C, D, ship_id]. [x, y, V_x, V_y, A, B, C, D, ID]
    other_ship_state_estimates = ships[0].get_converted_target_x_est()
    for ix, ship in enumerate(ships[1:]):
        colav_input[f'ts{ix}'] = np.append(other_ship_state_estimates[ix], [ship.length/2,
                                           ship.length/2, ship.length/4, ship.length/4, ship.mmsi])
    return colav_input


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

    for i, t in enumerate(time):
        for ix, ship in enumerate(ships):
            # Creates ships movement data
            ship.update_states(timestep)

            data[f'Ship{ix+1}'][0][i] = int(ship.x)
            data[f'Ship{ix+1}'][1][i] = int(ship.y)

            # Creates ships future positions for speed vector visualization
            data[f'Ship{ix + 1}'][2][i] = int(ship.x_t)
            data[f'Ship{ix + 1}'][3][i] = int(ship.y_t)

            # writing instantaneous ship data to the ais_data dataframe.
            row = {'mmsi': ship.mmsi, 'lon': ship.x, 'lat': ship.y, 'date_time_utc': i,
                   'sog': ship.u, 'cog': int(math.degrees(ship.psi)), 'true_heading': int(math.degrees(ship.psi)),
                   'nav_status': None, 'message_nr': ship.message_nr, 'source': ''}
            ais_data = ais_data.append(row, ignore_index=True)

        for ix, ship in enumerate(ships): # loop to update situational awareness
            x_true_list = [s.get_pose() for j, s in enumerate(ships) if j != ix]
            ship.update_target_x_est(x_true_list, t, timestep)

        # create_colav_input data every 10th second. This data will be used with PSB-MPC algorithm
        if t % 10 == 0:
            colav_input = create_colav_input(ships, i)
            # print(colav_input)    # Uncomment to see colav_input data
            # Here Trym's calculate_optimal_offsets function will be called with the colav_input.
    #print("Ship 2 state: ", ships[1].get_pose())
    #print("Ship 1 state_est of Ship 2: ", ships[0].target_ship_state_est[0])

    return data, ais_data, colav_input


def save_scenario_definition(pose_list, waypoint_list, speed_plan_list, savefile):
    """
        Saves the the scenario init to a json file as a dict at scenarios/savefile
        dict keys:
            poses[i]: pose [x,y,u,psi] for ship i 
            waypoints[i]: waypoints for ship i
            waypoints[i]: speed_plan for ship i
    """
    poses = []
    for i, pose in enumerate(pose_list):
        p = pose.copy()
        p[2] /= 0.51 #from m/s back to knots
        p[3] = int(np.rad2deg(pose[3])) #back to degrees
        poses.append(p)
    data = {
        "poses": poses,
        "waypoints": waypoint_list,
        "speed_plans": speed_plan_list  
        }
    json.dump( data, open(f'scenarios/{savefile}', 'w'), indent=2)

def load_scenario_definition(loadfile):
    """
        Loads the scenario init from a json file as a dict
        dict keys:
            poses[i]: pose [x,y,u,psi] for ship i
            waypoints[i]: waypoints for ship i
            speed_plans[i]: speed_plan for ship i        
    """
    data = json.load( open(f'scenarios/{loadfile}') )
    pose_list = data['poses']
    waypoint_list = [] #data['waypoints']
    speed_plan_list = data['speed_plans']
    
    for i in range(len(data["waypoints"])):
        wp = [tuple(row) for row in data["waypoints"][i]]
        waypoint_list.append(wp)
    return pose_list, waypoint_list, speed_plan_list


def get_ship_parameters(num_ships):
    """
        Reads the config for each ship,
        and initializes and coverts to right format
        If ship i doesn't have config, then default params are set
    """
    ship_model_list = []
    sensors_list = []
    LOS_params_list = []
    for i in range(num_ships):
        sensors = []
        # Config parameters for ship i
        ship_model, radar_active, radar_meas_rate, radar_sigma_z, \
        ais_active, ais_meas_rate, ais_sigma_z, ais_loss_prob, \
        delta, R_a = read_ship_config(f'SHIP{i+1}')
        
        if radar_active:
            radar = Radar(meas_rate=radar_meas_rate, sigma_z=radar_sigma_z)
            sensors.append(radar)
        if ais_active:
            ais = AIS(meas_rate=ais_meas_rate, sigma_z=ais_sigma_z, loss_prob=ais_loss_prob)
            sensors.append(ais)
        LOS_params = [delta, R_a]

        ship_model_list.append(ship_model)
        sensors_list.append(sensors)
        LOS_params_list.append(LOS_params)

    return ship_model_list, sensors_list, LOS_params_list

def init_scenario(new_scenario, scenario_file, num_ships, scenario_num=0, os_max_speed=20, ts_max_speed=20, wp_number=5):
    """
        Returns initialized and configured ships, and their route plans
        new_scenario=False: creates a new scenario based on scenario_num and saves
            the poses and plans to scenario_file
        new_scenario=True: Loads poses and plans from scenario_file

    """
    # Get config parameters for each ship
    ship_model_name_list, sensors_list, LOS_params_list = get_ship_parameters(num_ships)
    
    if not new_scenario:
        pose_list, waypoint_list, speed_plan_list = load_scenario_definition(scenario_file)
    else:
        pose_list, waypoint_list, speed_plan_list = create_scenario(num_ships, scenario_num, ship_model_name_list, os_max_speed, ts_max_speed, wp_number)
        save_scenario_definition(pose_list, waypoint_list, speed_plan_list, scenario_file)
    
    ship_list = []
    for i in range(num_ships):
        ship = Ship(pose=pose_list[i],
                    waypoints=waypoint_list[i],
                    speed_plan=speed_plan_list[i],
                    ship_model_name=ship_model_name_list[i],
                    mmsi=f'Ship{i+1}',
                    sensors=sensors_list[i],
                    LOS_params = LOS_params_list[i]
                    )
        ship_list.append(ship)
    return ship_list, waypoint_list, speed_plan_list

def create_scenario(num_ships, scenario_num, ship_model_name_list, os_max_speed, ts_max_speed, wp_number):
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
        speed_plan = create_random_speed_plan(pose_list[i][2]*0.51, wp_number)
        waypoint_list.append(wp)
        speed_plan_list.append(speed_plan)
    return pose_list, waypoint_list, speed_plan_list