from scenario_generator import *
from animation import visualize
from map import *
from yaspin import yaspin


@yaspin(text="Running...")
def main():
    ###############################################
    # INITIALISATION
    ###############################################

    # time
    t = np.arange(time_start, time_end + time_step, time_step)

    # number of waypoints
    wp_number = waypoint_num

    # scenarios
    if new_scenario:
        ship_list = ship_generator(scenario_num=scenario_num, os_max_speed=os_max_speed,
                                ts_max_speed=ts_max_speed, ship_number=ship_num, ship_model_name=ship_model_name)

        waypoint_list = waypoint_generator(ships=ship_list, waypoints_number=wp_number)

        save_scenario_definition(ship_list, waypoint_list, scenario_file)
    else:
        ship_list, waypoint_list = load_scenario_definition(scenario_file)

    data, ais_data, colav_input = ship_data(ships=ship_list, waypoint_list=waypoint_list, time=t, timestep=time_step)

    # exporting ais_data.csv
    ais_data.to_csv('ais_data.csv')

    ###############################################
    # ANIMATION PART
    ###############################################
    visualize(data, wp_number, t)


if __name__ == '__main__':
    main()



