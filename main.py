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
    # Number of waypoints
    wp_number = waypoint_num

    # Initiates the scenario and configures the ships
    ship_list, waypoint_list, speed_plan_list = init_scenario(new_scenario=new_scenario,
                                                            scenario_file=scenario_file,
                                                            num_ships=ship_num,
                                                            scenario_num=scenario_num,
                                                            os_max_speed=os_max_speed,
                                                            ts_max_speed=ts_max_speed,
                                                            wp_number=wp_number
                                                            )

    data, ais_data, colav_input = ship_data(ships=ship_list, waypoint_list=waypoint_list, time=t, timestep=time_step)

    # exporting ais_data.csv
    ais_data.to_csv('ais_data.csv')

    ###############################################
    # ANIMATION PART
    ###############################################
    visualize(data, wp_number, t)


if __name__ == '__main__':
    main()



