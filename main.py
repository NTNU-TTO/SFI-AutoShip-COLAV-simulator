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

    data_list, ais_data_list = [], []
    verifier_list = []
    if run_all_scenarios:
        ### Run all scenarios defined in the scenarios folder ###
        scenario_file_names = next(walk('scenarios'), (None, None, []))[2]
    else:
        ### Single scenario run ###
        scenario_file_names = [scenario_file]

    ###############################################
    # SIMULATION AND EVALUATION
    ###############################################
    for scenario_file_name in scenario_file_names:
        # Initiates the scenario and configures the ships
        ship_list, waypoint_list, speed_plan_list = init_scenario(new_scenario=new_scenario,
                                                                scenario_file=scenario_file_name
                                                                )

        data, ais_data, colav_input = run_scenario_simulation(ships=ship_list, time=t, timestep=time_step)
        data_list.append(data)
        # exporting ais_data.csv
        ais_data_filename = f'ais_data_{scenario_file_name[:-5]}.csv'
        ais_data.to_csv(ais_data_filename, sep=';')
        if evaluate_results:
            verifier = EvalTool(ais_data_filename)
            verifier.evaluate_vessel_behavior()
            verifier_list.append(verifier)
    
    ###############################################
    # ANIMATION PART
    ###############################################
    if visulaize_scenario:
        for data in data_list:
            visualize(data, t, show_waypoints=show_waypoints)
    if evaluate_results:
        for verifier in verifier_list:
            for vessel in verifier.vessels:
                verifier.print_scores(vessel)
            verifier.plot_trajectories(ownship=verifier.vessels[0], show_cpa=True, vessels=None, show_maneuvers=False, legend=True,
                                savefig=True, filename="figure.png", ax=None)

if __name__ == '__main__':
    main()



