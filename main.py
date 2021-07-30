from scenario_simulator import *
from animation import visualize
from map import *
from yaspin import yaspin
from os import walk
import sys
sys.path.insert(0, './UTC/colav/autoVerification/')
import matplotlib.pyplot as plt

from EvalTool import EvalTool


@yaspin(text="Running...")
def main():
    ###############################################
    # INITIALISATION
    ###############################################

    # time
    t = np.arange(time_start, time_end + time_step, time_step)

    if run_all_scenarios:
        ### Run all scenarios defined in the scenarios folder ###
        data_list, ais_data_list = [], [] 
        scenario_file_names = next(walk('scenarios'), (None, None, []))[2]
        for i in range(len(scenario_file_names)):
            scenario_file_name = scenario_file_names[i]
            # Initiates the scenario and configures the ships
            ship_list, waypoint_list, speed_plan_list = init_scenario(new_scenario=False,
                                                                    scenario_file=scenario_file_name,
                                                                    num_ships=ship_num, #don't care parameter
                                                                    scenario_num=scenario_num, #don't care parameter
                                                                    os_max_speed=os_max_speed, #don't care parameter
                                                                    ts_max_speed=ts_max_speed, #don't care parameter
                                                                    wp_number=waypoint_num
                                                                    )
            data, ais_data, colav_input = ship_data(ships=ship_list, waypoint_list=waypoint_list, time=t, timestep=time_step)
            data_list.append(data)
            ais_data_list.append(ais_data_list)
    else:
        ### Single scenario run ###
        # Initiates the scenario and configures the ships
        ship_list, waypoint_list, speed_plan_list = init_scenario(new_scenario=new_scenario,
                                                                scenario_file=scenario_file,
                                                                num_ships=ship_num,
                                                                scenario_num=scenario_num,
                                                                os_max_speed=os_max_speed,
                                                                ts_max_speed=ts_max_speed,
                                                                wp_number=waypoint_num
                                                                )

        data, ais_data, colav_input = ship_data(ships=ship_list, waypoint_list=waypoint_list, time=t, timestep=time_step)

        # exporting ais_data.csv
        ais_data.to_csv('ais_data.csv', sep=';')

        ###############################################
        # ANIMATION PART
        ###############################################
        if visulaize_scenario:
            visualize(data, t, show_waypoints=show_waypoints)
    if evaluate_results:
        verifier = EvalTool('ais_data.csv')
        verifier.evaluate_vessel_behavior()
        verifier.print_scores(verifier.vessels[0])
        verifier.plot_trajectories(ownship=verifier.vessels[0], show_cpa=True, vessels=None, show_maneuvers=True, legend=True,
                            savefig=True, filename="figure.png", ax=None)

if __name__ == '__main__':
    main()



