import math
import numpy as np
import pandas as pd
import json
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})

from ship import Ship
from sensors import *
from read_config import read_ship_config
from scenario_generator import create_scenario


def ship_data(ships, waypoint_list, time, timestep):
    '''
        :param ships: List of initialized and configured ships.
            Includes initial pose, waypoints, speed plan
        :param waypoint_list: List of waypoints which is created from waypoint_generator().
        :param time: Array of time of the simulation.
        :param timestep: Defined time step (dt)
        :return: Returns data for visualization and ais_data for evaluation library (EvalTool()).
    '''
    ais_data = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'date_time_utc', 'sog', 'cog',
                                     'true_heading', 'nav_status', 'message_nr', 'source'])

    data = {}
    sim_data = {}
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
            if not t % 5: # Error when ais_dt < 5
                row = ship.get_ais_data(t)
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


def save_scenario_definition(pose_list, waypoint_list, speed_plan_list, savefile):
    """
        Saves the the scenario init to a json file as a dict at scenarios/savefile
        dict keys:
            pose_list[i]: pose [x,y,u,psi] for ship i 
            waypoint_list[i]: waypoints for ship i
            speed_plan_list[i]: speed_plan for ship i
    """
    
    data = {
        "poses": pose_list,
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
    for i in range(len(pose_list)):
        ship = Ship(pose=pose_list[i],
                    waypoints=waypoint_list[i],
                    speed_plan=speed_plan_list[i],
                    ship_model_name=ship_model_name_list[i],
                    mmsi= i+1, #f'Ship{i+1}',
                    sensors=sensors_list[i],
                    LOS_params = LOS_params_list[i]
                    )
        ship_list.append(ship)
    return ship_list, waypoint_list, speed_plan_list



